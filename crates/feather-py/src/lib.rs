//! Python bindings.
//!
//! The Python-facing API lives in `python/feather/`, which is pure Python and
//! re-exports from this compiled module. Keeping the split means the ergonomics
//! (docstrings, overloads, Pydantic models) stay in Python where they belong,
//! and this module stays a thin conversion layer over `feather-core`.
//!
//! This layer converts; it does not validate. Everything that crosses is validated
//! by the core, so a binding that skips its own checks cannot hand over something
//! the engines cannot represent.
//!
//! [`FeatureStore`] is the one place the binding does real work: it resolves
//! `view:feature` references into one join per view, intersects the rows the views
//! agree on, and reattaches the entity frame's own columns by row index. Every
//! decision it makes is re-checked by the core.

use feather_core::Error as CoreError;
use pyo3::exceptions::{PyFileNotFoundError, PyOSError, PyValueError};
use pyo3::prelude::*;

#[cfg(feature = "offline")]
use std::collections::HashMap;

#[cfg(feature = "offline")]
use arrow::array::{Array, ArrayRef, Int64Array, UInt32Array};
#[cfg(feature = "offline")]
use arrow::compute::{concat_batches, take};
#[cfg(feature = "offline")]
use arrow::datatypes::{Field as ArrowField, Schema};
#[cfg(feature = "offline")]
use arrow::record_batch::RecordBatch;
#[cfg(feature = "offline")]
use pyo3_arrow::PyTable;

#[cfg(feature = "offline")]
use feather_core::offline::ROW_COLUMN;
#[cfg(feature = "offline")]
use feather_core::{Definitions, JoinOptions, Limits, OfflineEngine, OnMissing};

/// Read and validate a `feather.toml`, returned as JSON.
///
/// The core owns the file format, the schema, and `${VAR}` resolution, so this
/// binding contributes only the path.
#[pyfunction]
fn load_settings(path: &str) -> PyResult<String> {
    let settings = feather_core::load_settings(path).map_err(core_error)?;
    serde_json::to_string(&settings).map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Map a core failure onto the exception a Python caller expects.
///
/// I/O failures become the `OSError` family, since retrying or reporting them is
/// the caller's business. Everything else is a value the caller supplied, so it
/// becomes `ValueError` and the message names what is wrong.
fn core_error(error: CoreError) -> PyErr {
    // Render before matching, so the variant can be destructured by reference.
    let message = error.to_string();
    match &error {
        CoreError::SettingsIo { source, .. } if source.kind() == std::io::ErrorKind::NotFound => {
            PyFileNotFoundError::new_err(message)
        }
        CoreError::SettingsIo { .. } => PyOSError::new_err(message),
        #[cfg(feature = "offline")]
        CoreError::SpillDirectory { .. } => PyOSError::new_err(message),
        _ => PyValueError::new_err(message),
    }
}

/// One view's join, kept so several references into the same view share it.
#[cfg(feature = "offline")]
struct JoinedView {
    /// The view's result: `feather_row` followed by one column per declared feature.
    batch: RecordBatch,
    /// The entity frame's row index to its position in `batch`.
    positions: HashMap<i64, u32>,
    /// The entity frame's row indexes this view produced, ascending and unique.
    rows: Vec<i64>,
}

/// The rows present in both lists, which are each ascending and unique.
#[cfg(feature = "offline")]
fn intersect(a: &[i64], b: &[i64]) -> Vec<i64> {
    let mut out = Vec::with_capacity(a.len().min(b.len()));
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// A project's validated definitions, and the engine that joins them.
///
/// Constructed from a `feather.toml` path and the compiled definitions, both of
/// which the core validates. The engine is opened once and reused, because
/// opening it creates a private spill directory and may load a filesystem
/// extension.
///
/// `unsendable` because a DuckDB connection is `Send` but not `Sync`, which is
/// what a PyO3 class needs. So the store belongs to the thread that created it.
///
/// > `ponytail:` the ceiling is that a store made on one thread cannot be used on
/// > another, which a threaded caller would hit. Upgrade path: hold the engine in
/// > a `Mutex`, which is enough to make the class `Send + Sync` and would also
/// > let the join release the GIL, or make the engine `Sync` in the core.
#[cfg(feature = "offline")]
#[pyclass(unsendable)]
struct FeatureStore {
    engine: OfflineEngine,
    definitions: Definitions,
}

#[cfg(feature = "offline")]
#[pymethods]
impl FeatureStore {
    /// Open a store over a project.
    ///
    /// Args:
    ///     settings_path: The `feather.toml` to read. The core parses and
    ///         validates it, so this binding contributes only the path.
    ///     definitions_json: The project's compiled definitions, as
    ///         `Definitions::from_json` expects. The core validates them again.
    ///
    /// Raises:
    ///     FileNotFoundError: If no file is at `settings_path`.
    ///     OSError: If the settings cannot be read, or the spill directory cannot
    ///         be created.
    ///     ValueError: If the settings or the definitions are malformed.
    #[new]
    fn new(settings_path: &str, definitions_json: &str) -> PyResult<Self> {
        let settings = feather_core::load_settings(settings_path).map_err(core_error)?;
        let definitions = Definitions::from_json(definitions_json).map_err(core_error)?;
        // Definitions and settings first meet here, so a source naming a connection the
        // project does not have fails now rather than at the first read. The join checks
        // the same source again as a backstop.
        definitions
            .validate_sources(&settings.connections)
            .map_err(core_error)?;
        let engine =
            OfflineEngine::open(&Limits::default(), &settings.connections).map_err(core_error)?;
        Ok(Self {
            engine,
            definitions,
        })
    }

    /// Join the requested features onto an entity frame.
    ///
    /// Args:
    ///     entity_frame: The label rows, as any object exposing the Arrow
    ///         `__arrow_c_stream__` interface. A Polars DataFrame does; a pandas
    ///         DataFrame does once it has been converted to Arrow. The transfer
    ///         is zero-copy, but the join result is not: DuckDB materializes it.
    ///     features: `view:feature` references, in the order the caller wants the
    ///         columns back.
    ///     label_timestamp_column: The entity frame's label timestamp column.
    ///     strict: Exclude a feature row at exactly the label time.
    ///     on_missing: `null` keeps a row whose value is missing or expired and
    ///         leaves the value null; `drop` removes the row.
    ///
    /// Returns:
    ///     The entity frame's own columns, in the surviving row order, followed by
    ///     one column per requested feature in request order.
    ///
    /// Raises:
    ///     ValueError: If a reference is malformed, names an unknown view or
    ///         feature, or repeats one; if `on_missing` is neither `null` nor
    ///         `drop`; if a requested feature name collides with a column of the
    ///         entity frame; or if a join fails.
    #[pyo3(signature = (
        entity_frame,
        features,
        label_timestamp_column = "event_timestamp",
        strict = false,
        on_missing = "null",
    ))]
    fn get_historical_features(
        &self,
        entity_frame: PyTable,
        features: Vec<String>,
        label_timestamp_column: &str,
        strict: bool,
        on_missing: &str,
    ) -> PyResult<PyTable> {
        let on_missing = match on_missing {
            "null" => OnMissing::Null,
            "drop" => OnMissing::Drop,
            other => {
                return Err(PyValueError::new_err(format!(
                    "on_missing is `{other}`; expected `null` or `drop`"
                )));
            }
        };

        let references = parse_references(&features)?;

        // Checked before any query runs, so a name the project does not declare fails with
        // that name rather than as a column missing from the join's output.
        for reference in &references {
            let view = self.definitions.view(&reference.view).map_err(core_error)?;
            if view.field(&reference.feature).is_none() {
                return Err(PyValueError::new_err(format!(
                    "view `{}` declares no feature `{}`",
                    reference.view, reference.feature
                )));
            }
        }

        let (batches, entity_schema) = entity_frame.into_inner();
        let entity = concat_batches(&entity_schema, &batches).map_err(arrow_error)?;
        for reference in &references {
            if entity_schema.index_of(&reference.feature).is_ok() {
                return Err(PyValueError::new_err(format!(
                    "the entity frame already has a column named `{}`, which is the requested \
                     feature `{}`; rename it, or the two cannot be told apart in the result",
                    reference.feature, reference.view
                )));
            }
        }

        let options = JoinOptions {
            label_timestamp_column: label_timestamp_column.to_owned(),
            strict,
            on_missing,
        };

        // One join per distinct view, however many of its features were requested.
        let mut joined: Vec<JoinedView> = Vec::new();
        let mut index_of_view: HashMap<&str, usize> = HashMap::new();
        for reference in &references {
            if index_of_view.contains_key(reference.view.as_str()) {
                continue;
            }
            let view = self.definitions.view(&reference.view).map_err(core_error)?;
            let batch = self
                .engine
                .point_in_time_join(&entity, view, &options)
                .map_err(core_error)?;
            index_of_view.insert(&reference.view, joined.len());
            joined.push(index_of_view_of(batch)?);
        }

        // A row survives only if every requested view produced one, so `drop`
        // never leaves a null behind for a row it kept. Under `null` every view
        // covers every label row, and this intersection is the whole frame.
        let surviving = joined
            .iter()
            .map(|view| view.rows.as_slice())
            .fold(None::<Vec<i64>>, |acc, rows| {
                Some(match acc {
                    None => rows.to_vec(),
                    Some(acc) => intersect(&acc, rows),
                })
            })
            .unwrap_or_default();

        let mut columns: Vec<ArrayRef> =
            Vec::with_capacity(entity.num_columns() + references.len());
        let mut fields: Vec<ArrowField> =
            Vec::with_capacity(entity.num_columns() + references.len());

        let entity_rows = gather_indices(&surviving, |row| Some(row as u32))?;
        for (index, field) in entity_schema.fields().iter().enumerate() {
            columns.push(
                take(entity.column(index).as_ref(), &entity_rows, None).map_err(arrow_error)?,
            );
            fields.push(field.as_ref().clone());
        }

        // Per view, the surviving row to its position in that view's result.
        let view_rows: Vec<UInt32Array> = joined
            .iter()
            .map(|view| gather_indices(&surviving, |row| view.positions.get(&row).copied()))
            .collect::<PyResult<Vec<_>>>()?;

        for reference in &references {
            let view_index = index_of_view[reference.view.as_str()];
            let view = &joined[view_index];
            let column = view
                .batch
                .schema()
                .index_of(&reference.feature)
                .map_err(arrow_error)?;
            columns.push(
                take(
                    view.batch.column(column).as_ref(),
                    &view_rows[view_index],
                    None,
                )
                .map_err(arrow_error)?,
            );
            fields.push(ArrowField::new(
                &reference.feature,
                view.batch.column(column).data_type().clone(),
                true,
            ));
        }

        let schema = std::sync::Arc::new(Schema::new(fields));
        let output = RecordBatch::try_new(schema.clone(), columns).map_err(arrow_error)?;
        PyTable::try_new(vec![output], schema)
    }
}

/// One requested feature, split into the view it comes from and its name there.
#[cfg(feature = "offline")]
struct Reference {
    view: String,
    feature: String,
}

/// Split `view:feature` references, rejecting the shapes the engine cannot use.
///
/// The Python layer builds these, so this is a check of what the core accepts
/// rather than of what a user typed.
#[cfg(feature = "offline")]
fn parse_references(features: &[String]) -> PyResult<Vec<Reference>> {
    if features.is_empty() {
        return Err(PyValueError::new_err(
            "features must name at least one feature",
        ));
    }
    let mut references: Vec<Reference> = Vec::with_capacity(features.len());
    for reference in features {
        let (view, feature) = reference.split_once(':').ok_or_else(|| {
            PyValueError::new_err(format!(
                "malformed reference `{reference}`; expected `view:feature`"
            ))
        })?;
        if view.is_empty() || feature.is_empty() {
            return Err(PyValueError::new_err(format!(
                "malformed reference `{reference}`; expected `view:feature`"
            )));
        }
        if references
            .iter()
            .any(|seen| seen.view == view && seen.feature == feature)
        {
            return Err(PyValueError::new_err(format!(
                "the reference `{reference}` is requested twice"
            )));
        }
        references.push(Reference {
            view: view.to_owned(),
            feature: feature.to_owned(),
        });
    }
    Ok(references)
}

/// Index a view's result by the entity frame row it belongs to.
#[cfg(feature = "offline")]
fn index_of_view_of(batch: RecordBatch) -> PyResult<JoinedView> {
    let rows = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "the join returned a `{ROW_COLUMN}` column that is not an integer"
            ))
        })?;
    let mut positions = HashMap::with_capacity(rows.len());
    let mut indexes = Vec::with_capacity(rows.len());
    for (position, row) in rows.iter().enumerate() {
        let row = row.ok_or_else(|| {
            PyValueError::new_err(format!("the join returned a null `{ROW_COLUMN}`"))
        })?;
        positions.insert(row, position as u32);
        indexes.push(row);
    }
    Ok(JoinedView {
        batch,
        positions,
        rows: indexes,
    })
}

/// Build `take` indices for `rows`, failing rather than panicking on a gap.
///
/// The gap cannot happen: `surviving` is the intersection of every view's rows.
/// A panic here would abort the host process rather than return a bad result,
/// which is why it is a value.
#[cfg(feature = "offline")]
fn gather_indices(rows: &[i64], position: impl Fn(i64) -> Option<u32>) -> PyResult<UInt32Array> {
    let mut indices = Vec::with_capacity(rows.len());
    for row in rows {
        let found = position(*row).ok_or_else(|| {
            PyValueError::new_err(format!(
                "row {row} of the entity frame is missing from a joined view, which should not \
                 be reachable; please report it"
            ))
        })?;
        indices.push(found);
    }
    Ok(UInt32Array::from(indices))
}

/// Wrap an Arrow failure as the exception a Python caller expects.
#[cfg(feature = "offline")]
fn arrow_error(error: arrow::error::ArrowError) -> PyErr {
    PyValueError::new_err(format!("arrow: {error}"))
}

/// The compiled extension. Imported as `feather._core`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(load_settings, m)?)?;
    #[cfg(feature = "offline")]
    m.add_class::<FeatureStore>()?;
    Ok(())
}
