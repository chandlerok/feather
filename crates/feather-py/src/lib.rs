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
//! The online read is a case in point: the reference splitting, the
//! present-or-null reshaping, and the result table all live in the core, so this
//! module only opens the store, runs the read, and hands the table to the
//! capsule. The historical join is the one path that still does real work here,
//! resolving references into one join per view and reattaching the entity frame's
//! own columns by row index. Every decision it makes is re-checked by the core.

use feather_core::Error as CoreError;
use pyo3::exceptions::{PyFileNotFoundError, PyOSError, PyValueError};
use pyo3::prelude::*;

mod demo;

use std::collections::BTreeMap;
#[cfg(feature = "offline")]
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

#[cfg(feature = "offline")]
use arrow::array::{Array, ArrayRef, Int64Array, UInt32Array};
use arrow::compute::concat_batches;
#[cfg(feature = "offline")]
use arrow::compute::take;
#[cfg(feature = "offline")]
use arrow::datatypes::{Field as ArrowField, Schema};
#[cfg(feature = "offline")]
use arrow::record_batch::RecordBatch;
use pyo3_arrow::PyTable;

#[cfg(feature = "offline")]
use feather_core::offline::ROW_COLUMN;
// The online read path is not behind `offline`: reading online features never runs the
// point-in-time join, so a build that only reads online carries no DuckDB. `FjallStore`
// still needs `fjall`, because that feature is what provides it.
#[cfg(feature = "fjall")]
use feather_core::online::fjall::FjallStore;
#[cfg(not(feature = "fjall"))]
use feather_core::online::memory::MemoryStore;
use feather_core::online::{
    EntityRequest, OnlineStore, ProjectScan, ReadRequest, ViewRequest, WriteBatch,
};
use feather_core::settings::Store as StoreSettings;
use feather_core::{
    Definitions, FeatureView, assemble_online_result, entity_key_component, parse_references,
    read_entities,
};
#[cfg(feature = "offline")]
use feather_core::{
    JoinOptions, Limits, OfflineEngine, OnMissing, materialize as materialize_project,
};

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

/// A project's validated definitions, the engine that joins them, and the store they are
/// served from.
///
/// Constructed from a `feather.toml` path and the compiled definitions, both of which the core
/// validates. In a build that carries DuckDB, the engine is opened once and reused, because
/// opening it creates a private spill directory and may load a filesystem extension.
///
/// The state sits behind a `Mutex` so the class is `Send + Sync`, which a PyO3 class has to be
/// without `unsendable`. A DuckDB connection is `Send` but not `Sync`, and the mutex is what
/// turns that into a type PyO3 accepts. It also serializes calls that touch the engine, which
/// is what they need: one DuckDB connection is one session, and the online store is one
/// connection too.
///
/// Holding the lock across a long refresh is deliberate: what a second call cannot do is refresh
/// the same project concurrently, and a second refresh would only write the same values again.
///
/// The lock is taken *inside* the GIL-released region, in [`FeatureStore::with_inner`], and that
/// ordering is the point rather than a detail. See the note there for what goes wrong otherwise.
#[pyclass]
struct FeatureStore {
    inner: Mutex<Inner>,
}

/// Everything a store owns, behind its mutex.
struct Inner {
    /// The runtime the online store's I/O runs on.
    ///
    /// Created once with the store rather than per call, and multi-threaded because the
    /// serving transport runs one request per worker thread. Blocking the Python thread on it
    /// with `block_on` is sound here: that thread is not one of its workers, so there is no
    /// runtime to nest inside.
    runtime: tokio::runtime::Runtime,
    /// The offline engine, present only in a build that carries DuckDB.
    ///
    /// The online read path does not touch it, so a build without the `offline`
    /// feature has no field here and still serves: the historical join and the
    /// refresh are the two operations that need it, and a build that has neither
    /// does not pay for DuckDB.
    #[cfg(feature = "offline")]
    engine: OfflineEngine,
    definitions: Definitions,
    /// The project name, which namespaces every online key.
    project: String,
    /// Where the online store lives, resolved from the settings once so that the write path and
    /// `serve()` cannot each pick their own default and end up disagreeing.
    location: (PathBuf, u64, u64),
    /// The store features are served from, opened on the first online call.
    ///
    /// `None` until then, so that constructing a store over a project whose store directory is
    /// not yet created still works, and so that a project that never reads online never opens
    /// a database.
    online: Option<Online>,
}

/// The online store a project serves from, which one it is decided by the settings.
enum Online {
    /// Local mode: a build with no embedded store, so the values live in this process and die
    /// with it. Only reachable without `fjall`, because a build with it always opens the
    /// directory `serve()` opens.
    #[cfg(not(feature = "fjall"))]
    Memory(MemoryStore),
    /// The embedded store, at the directory the settings resolve to, so that a refresh writes
    /// the database `serve()` reads even when the settings declare no `[store]`.
    #[cfg(feature = "fjall")]
    Fjall(Box<FjallStore>),
}

impl OnlineStore for Online {
    async fn write(&mut self, batches: &[WriteBatch]) -> feather_core::Result<()> {
        match self {
            #[cfg(not(feature = "fjall"))]
            Online::Memory(store) => store.write(batches).await,
            #[cfg(feature = "fjall")]
            Online::Fjall(store) => store.write(batches).await,
        }
    }

    async fn read(
        &self,
        requests: &[ReadRequest],
    ) -> feather_core::Result<Vec<Vec<Option<Vec<u8>>>>> {
        match self {
            #[cfg(not(feature = "fjall"))]
            Online::Memory(store) => store.read(requests).await,
            #[cfg(feature = "fjall")]
            Online::Fjall(store) => store.read(requests).await,
        }
    }

    async fn delete_fields(
        &mut self,
        keys_and_fields: &[(Vec<u8>, Vec<String>)],
    ) -> feather_core::Result<()> {
        match self {
            #[cfg(not(feature = "fjall"))]
            Online::Memory(store) => store.delete_fields(keys_and_fields).await,
            #[cfg(feature = "fjall")]
            Online::Fjall(store) => store.delete_fields(keys_and_fields).await,
        }
    }
}

impl ProjectScan for Online {
    async fn hash_fields(&self, key: &[u8]) -> feather_core::Result<Vec<(String, Vec<u8>)>> {
        match self {
            #[cfg(not(feature = "fjall"))]
            Online::Memory(store) => store.hash_fields(key).await,
            #[cfg(feature = "fjall")]
            Online::Fjall(store) => store.hash_fields(key).await,
        }
    }

    async fn scan_entity_keys(
        &self,
        project: &str,
        exclude: &[u8],
    ) -> feather_core::Result<Vec<Vec<u8>>> {
        match self {
            #[cfg(not(feature = "fjall"))]
            Online::Memory(store) => store.scan_entity_keys(project, exclude).await,
            #[cfg(feature = "fjall")]
            Online::Fjall(store) => store.scan_entity_keys(project, exclude).await,
        }
    }
}

/// The error a poisoned lock becomes.
///
/// The one place the store's lock is taken.
///
/// The order of the two locks is the whole point, and it is the opposite of what looks natural.
/// The GIL is released *before* the mutex is taken, never after. Taking the mutex first looks
/// equivalent and is not: a second thread would then block on the mutex while still holding the
/// GIL, while this thread holds the mutex and waits to reacquire the GIL so that it can return.
/// Neither can proceed, so the interpreter hangs permanently rather than waiting. That is the
/// first thing a threaded server would hit.
///
/// So every method that needs the store's state goes through here, and none of them lock
/// directly. A call that arrives while the lock is held waits with the GIL released, which is a
/// wait rather than a deadlock.
///
/// Args:
///     py: The calling thread's token, used to release the GIL.
///     f: The work to run against the store's state, with the GIL released.
///
/// Returns:
///     Whatever `f` returned.
///
/// Raises:
///     OSError: If another thread panicked while holding the lock, so the state cannot be
///         trusted.
impl FeatureStore {
    fn with_inner<T, F>(&self, py: Python<'_>, f: F) -> PyResult<T>
    where
        F: FnOnce(&mut Inner) -> PyResult<T> + Send,
        T: Send,
    {
        py.detach(|| {
            let mut inner = self.inner.lock().map_err(poisoned)?;
            f(&mut inner)
        })
    }
}

/// A panic while another thread held the lock is what poisons it, which means the state inside it
/// is unknown; saying so beats unwrapping into a second panic.
fn poisoned<T>(_: std::sync::PoisonError<T>) -> PyErr {
    PyOSError::new_err(
        "another thread panicked while using this FeatureStore, so its state cannot be trusted",
    )
}

/// Open the project's online store if it is not open yet.
///
/// Opening is deferred to the first call that needs it, so a project can be constructed without
/// a store directory it cannot open and a project that only reads historical features never
/// opens one.
///
/// Args:
///     inner: The store's state.
///
/// Returns:
///     `Ok(())` once `inner.online` holds a store.
///
/// Raises:
///     OSError: If the store directory cannot be opened.
fn open_online(inner: &mut Inner) -> PyResult<()> {
    if inner.online.is_some() {
        return Ok(());
    }
    inner.online = Some(connect_online(inner.location.clone())?);
    Ok(())
}

/// Where the online store lives, and how large its cache and memtable are.
///
/// One resolver for both online paths, because they used to disagree. `serve()` fell back to
/// `.feather/online` when `feather.toml` declared no `[store]`, while materialization fell back
/// to an in-process map. A project with no `[store]` was therefore served out of a directory
/// nothing had ever written: the server started, resolved its service, answered every request,
/// and returned null for every column, with no error and no log line to say so. Resolving it in
/// one place is what makes that state unreachable rather than merely unlikely.
///
/// Args:
///     settings_path: The `feather.toml` the project was opened from. Only its parent directory
///         is read, and only when the settings declare no store of their own.
///     store: The settings' `[store]` table, or `None` for a project that declares none.
///
/// Returns:
///     The directory to open, with the cache and memtable sizes to open it at.
fn online_location(settings_path: &str, store: Option<&StoreSettings>) -> (PathBuf, u64, u64) {
    match store {
        Some(store) => (PathBuf::from(&store.path), store.cache(), store.memtable()),
        // A refresh creates this on its first write, beside the `feather.toml` that omits it.
        None => (
            PathBuf::from(settings_path)
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .join(".feather/online"),
            feather_core::settings::Store::DEFAULT_CACHE_BYTES,
            feather_core::settings::Store::DEFAULT_MEMTABLE_BYTES,
        ),
    }
}

/// Build the store at the location the settings resolve to.
///
/// Args:
///     location: The directory and sizes `online_location` resolved.
///
/// Returns:
///     The embedded store, in a build that has one. Without `fjall` there is nothing to embed,
///     and the values live in this process until it exits.
///
/// Raises:
///     OSError: If the store directory cannot be opened, naming the path.
fn connect_online(location: (PathBuf, u64, u64)) -> PyResult<Online> {
    #[cfg(feature = "fjall")]
    return Ok(Online::Fjall(Box::new(
        FjallStore::open(&location.0, location.1, location.2).map_err(core_error)?,
    )));
    #[cfg(not(feature = "fjall"))]
    let _ = location;
    #[cfg(not(feature = "fjall"))]
    return Ok(Online::Memory(MemoryStore::new()));
}

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
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|error| {
                PyOSError::new_err(format!(
                    "could not start the runtime the store needs: {error}"
                ))
            })?;
        // Definitions and settings first meet here, so a source naming a connection the
        // project does not have fails now rather than at the first read. The join checks
        // the same source again as a backstop.
        definitions
            .validate_sources(&settings.connections)
            .map_err(core_error)?;
        #[cfg(feature = "offline")]
        let engine =
            OfflineEngine::open(&Limits::default(), &settings.connections).map_err(core_error)?;
        Ok(Self {
            inner: Mutex::new(Inner {
                runtime,
                #[cfg(feature = "offline")]
                engine,
                project: definitions.project.clone(),
                definitions,
                location: online_location(settings_path, settings.store.as_ref()),
                online: None,
            }),
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
    #[cfg(feature = "offline")]
    #[pyo3(signature = (
        entity_frame,
        features,
        label_timestamp_column = "event_timestamp",
        strict = false,
        on_missing = "null",
    ))]
    fn get_historical_features(
        &self,
        py: Python<'_>,
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

        let references = parse_references(&features).map_err(core_error)?;

        // Read out of the capsule before the lock is taken: it is pure Arrow work with no Python
        // in it, and the frame's own columns are needed again after the joins.
        let (batches, entity_schema) = entity_frame.into_inner();
        let entity = concat_batches(&entity_schema, &batches).map_err(arrow_error)?;

        // The definition checks and the joins both need the engine, so the block runs with the
        // lock held and the GIL released. A DuckDB join over a large frame is seconds of work,
        // and holding the GIL for it would stop every other Python thread in the process.
        let (joined, index_of_view) = self.with_inner(py, |inner| {
            let Inner {
                engine,
                definitions,
                ..
            } = &mut *inner;

            // Checked before any query runs, so a name the project does not declare fails with
            // that name rather than as a column missing from the join's output.
            for reference in &references {
                let view = definitions.view(&reference.view).map_err(core_error)?;
                if view.field(&reference.feature).is_none() {
                    return Err(PyValueError::new_err(format!(
                        "view `{}` declares no feature `{}`",
                        reference.view, reference.feature
                    )));
                }
            }

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

            let mut joined: Vec<JoinedView> = Vec::new();
            let mut index_of_view: HashMap<String, usize> = HashMap::new();
            for reference in &references {
                if index_of_view.contains_key(reference.view.as_str()) {
                    continue;
                }
                let view = definitions.view(&reference.view).map_err(core_error)?;
                let batch = engine
                    .point_in_time_join(&entity, view, &options)
                    .map_err(core_error)?;
                index_of_view.insert(reference.view.clone(), joined.len());
                joined.push(index_of_view_of(batch)?);
            }
            Ok((joined, index_of_view))
        })?;

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

    /// Refresh feature values from their sources into the online store.
    ///
    /// A full refresh: every selected view is recomputed from its source and every value is
    /// overwritten. There is no watermark and nothing partial to reconcile, so a run that fails
    /// leaves the values it had not yet overwritten as they were and the call can simply be
    /// repeated. A run that is still going serves a mix of the values it has written and the ones
    /// it has not reached yet.
    ///
    /// Args:
    ///     views: The views to refresh, by name, or `None` for every view the project declares.
    ///
    /// Returns:
    ///     A report naming each refreshed view, the rows it wrote, the newest event timestamp
    ///     those rows carried, and how long it took, plus the views a previous refresh declared
    ///     and this one does not, whose fields were removed from the store.
    ///
    /// Raises:
    ///     ValueError: If a named view is not declared, or a source cannot be read as its view
    ///         declares it.
    ///     OSError: If the store directory cannot be opened.
    ///     OSError: If another thread panicked while holding this store.
    #[cfg(feature = "offline")]
    #[pyo3(signature = (views = None))]
    fn materialize(
        &self,
        py: Python<'_>,
        views: Option<Vec<String>>,
    ) -> PyResult<MaterializeReport> {
        let selected = views.unwrap_or_default();

        // Minutes of DuckDB work and network I/O, with the GIL released for all of it. The lock
        // is taken inside that region rather than before it; see `with_inner`.
        let report = self.with_inner(py, |inner| {
            open_online(inner)?;
            let Inner {
                runtime,
                engine,
                definitions,
                online,
                project,
                ..
            } = &mut *inner;
            let store = online
                .as_mut()
                .expect("open_online leaves a store in place");
            runtime
                .block_on(materialize_project(
                    store,
                    engine,
                    project,
                    &definitions.views,
                    &selected,
                ))
                .map_err(core_error)
        })?;

        Ok(MaterializeReport {
            views: report
                .views
                .into_iter()
                .map(|view| ViewRefresh {
                    name: view.name,
                    rows: view.rows,
                    max_event_timestamp_micros: view.max_event_timestamp_micros,
                    elapsed_seconds: view.elapsed.as_secs_f64(),
                })
                .collect(),
            retired: report.retired,
            total_rows: report.total_rows,
            elapsed_seconds: report.elapsed.as_secs_f64(),
        })
    }

    /// Read feature values for a frame of entities from the online store.
    ///
    /// Args:
    ///     entity_frame: The entities to read, as any object exposing the Arrow
    ///         `__arrow_c_stream__` interface. The join key column is named after the views'
    ///         entity's `join_key`, and one call reads one entity type.
    ///     features: `view:feature` references, in the order the caller wants the columns back.
    ///
    /// Returns:
    ///     An Arrow table: the entity frame's own columns in input order, then one column per
    ///     requested feature in request order, one row per input row. A value that is missing,
    ///     expired, or stored under a schema that no longer matches is a null; the three are not
    ///     distinguished here, which is the documented online contract.
    ///
    /// Raises:
    ///     TypeError: If `entity_frame` does not export Arrow buffers.
    ///     ValueError: If a reference is malformed, names an unknown view or feature, or is
    ///         requested twice; if the requested views do not declare the same entity, which is
    ///         both the same join key and the same entity name, since the hash key carries the
    ///         name and one request reads one entity type; if the frame has no such column or has
    ///         a null in it; or if a requested feature name collides with a column of the frame.
    ///     OSError: If the store directory cannot be opened.
    ///     OSError: If another thread panicked while holding this store.
    fn get_online_features(
        &self,
        py: Python<'_>,
        entity_frame: PyTable,
        features: Vec<String>,
    ) -> PyResult<PyTable> {
        let references = parse_references(&features).map_err(core_error)?;
        let (batches, entity_schema) = entity_frame.into_inner();
        let entity = concat_batches(&entity_schema, &batches).map_err(arrow_error)?;

        // Opening the store can block on a handshake, so it happens with the GIL released as
        // well. The lock is taken inside `with_inner` for the reason documented there.
        let (values, views, slots) = self.with_inner(py, |inner| {
            open_online(inner)?;
            let Inner {
                runtime,
                definitions,
                online,
                project,
                ..
            } = &mut *inner;
            let store = online
                .as_mut()
                .expect("open_online leaves a store in place");

            // One view slot per distinct view, in first-appearance order, holding that view's
            // requested features in request order. One `HMGET` per entity asks for one field per
            // view, so the whole request costs one store read per entity however many views it
            // names, which is the rule the storage layout exists to make possible.
            let mut views: BTreeMap<String, FeatureView> = BTreeMap::new();
            let mut slots: Vec<(usize, usize)> = Vec::with_capacity(references.len());
            let mut view_names: Vec<String> = Vec::new();
            let mut fields_per_view: Vec<Vec<String>> = Vec::new();
            let mut join_key: Option<String> = None;

            for reference in &references {
                let view = definitions.view(&reference.view).map_err(core_error)?;
                let field = view.field(&reference.feature).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "view `{}` declares no feature `{}`",
                        reference.view, reference.feature
                    ))
                })?;
                if entity_schema.index_of(&reference.feature).is_ok() {
                    return Err(PyValueError::new_err(format!(
                        "the entity frame already has a column named `{}`, which is the requested \
                     feature `{}`; rename it, or the two cannot be told apart in the result",
                        reference.feature, reference.view
                    )));
                }

                // One request reads one entity, because the entity name is part of the hash key.
                let key = view.entity().map_err(core_error)?.join_key.clone();
                match &join_key {
                    None => join_key = Some(key),
                    Some(existing) if existing == &key => {}
                    Some(existing) => {
                        return Err(PyValueError::new_err(format!(
                            "view `{}` is joined on `{key}` and another requested view on \
                         `{existing}`; one request reads one entity type",
                            reference.view
                        )));
                    }
                }

                let slot = match view_names.iter().position(|name| name == &reference.view) {
                    Some(slot) => slot,
                    None => {
                        view_names.push(reference.view.clone());
                        fields_per_view.push(Vec::new());
                        view_names.len() - 1
                    }
                };
                let feature_slot = fields_per_view[slot].len();
                fields_per_view[slot].push(field.name.clone());
                slots.push((slot, feature_slot));
                views.insert(reference.view.clone(), view.clone());
            }

            let join_key = join_key.expect("parse_references rejects an empty list");
            let key_index = entity_schema.index_of(&join_key).map_err(|_| {
                PyValueError::new_err(format!(
                    "the entity frame has no column `{join_key}`, which is the join key the \
                 requested views are keyed on"
                ))
            })?;

            // The entity keys are encoded here rather than in Python: the key format is the core's,
            // and a caller that encoded its own could disagree with what the write path stored.
            let key_column = entity.column(key_index).clone();
            let mut entities = Vec::with_capacity(entity.num_rows());
            for row in 0..entity.num_rows() {
                let component =
                    entity_key_component(key_column.as_ref(), row).map_err(core_error)?;
                entities.push(EntityRequest {
                    encoded_key: feather_core::encode_entity_key(&[&component])
                        .map_err(core_error)?,
                    views: view_names
                        .iter()
                        .zip(&fields_per_view)
                        .map(|(view, fields)| ViewRequest {
                            view: view.clone(),
                            fields: fields.clone(),
                        })
                        .collect(),
                });
            }

            let now = now_micros();
            let values = runtime
                .block_on(read_entities(store, project, &views, &entities, now))
                .map_err(core_error)?;
            Ok((values, views, slots))
        })?;

        // The reshaping is the core's, so a second binding produces the same
        // table for the same read rather than a near-copy of this loop.
        let output = assemble_online_result(&entity, &references, &views, &slots, &values)
            .map_err(core_error)?;
        let schema = output.schema();
        PyTable::try_new(vec![output], schema)
    }
}

/// What one view's refresh did.
#[cfg(feature = "offline")]
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct ViewRefresh {
    /// The view's name, as declared.
    #[pyo3(get)]
    name: String,
    /// Entity rows written, which is one store write per entity.
    #[pyo3(get)]
    rows: u64,
    /// The newest event timestamp any written row carried, in microseconds since the epoch, or
    /// `None` when the refresh wrote nothing.
    #[pyo3(get)]
    max_event_timestamp_micros: Option<i64>,
    /// How long this view's scan and writes took.
    #[pyo3(get)]
    elapsed_seconds: f64,
}

/// What one call to [`FeatureStore::materialize`] did.
#[cfg(feature = "offline")]
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct MaterializeReport {
    /// One entry per refreshed view, in declaration order.
    #[pyo3(get)]
    views: Vec<ViewRefresh>,
    /// The views a previous refresh declared and this one does not, whose fields were removed.
    #[pyo3(get)]
    retired: Vec<String>,
    /// Rows written across every refreshed view.
    #[pyo3(get)]
    total_rows: u64,
    /// How long the whole call took.
    #[pyo3(get)]
    elapsed_seconds: f64,
}

/// The current time, in microseconds since the epoch.
///
/// Read here and passed into the read path rather than read inside a query, so the TTL rule
/// stays testable without waiting for a clock.
fn now_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_micros() as i64)
        .unwrap_or(0)
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
fn arrow_error(error: arrow::error::ArrowError) -> PyErr {
    PyValueError::new_err(format!("arrow: {error}"))
}

/// Serve online features over Arrow Flight, blocking until stopped.
///
/// A module function rather than a method on the `FeatureStore` class, because the class is
/// `offline`-gated and serving must not be: a deployment that only reads online features never
/// runs a point-in-time join, so it should not link DuckDB. The Python wrapper exposes this as
/// `FeatureStore.serve()`, which is the API a user touches.
///
/// The store's directory is read from the project's `[store]` table, and defaults to
/// `.feather/online` beside the settings file. A database is opened by one process at a time,
/// so this is the only thing in a deployment that opens it.
///
/// Args:
///     settings_path: The `feather.toml` to read.
///     definitions_json: The project's compiled definitions.
///     service: The name of the feature service to serve.
///     addr: The socket to bind. Loopback by default, and the only thing protecting an
///         unauthenticated endpoint until issue #5 settles authentication.
///     workers: Tokio worker threads.
///
/// Returns:
///     Nothing, ever: it blocks. Ctrl-C is not wired yet, so the process is stopped by a
///     signal, and a database closed by a signal is recovered from its journal on next open.
#[cfg(feature = "serve")]
#[pyfunction]
#[pyo3(signature = (settings_path, definitions_json, service, addr = "127.0.0.1:8815", workers = 4))]
fn serve(
    py: Python<'_>,
    settings_path: &str,
    definitions_json: &str,
    service: &str,
    addr: &str,
    workers: usize,
) -> PyResult<()> {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    let settings = feather_core::load_settings(settings_path).map_err(core_error)?;
    let definitions = feather_core::Definitions::from_json(definitions_json).map_err(core_error)?;

    let declared = definitions
        .services
        .iter()
        .find(|s| s.name == service)
        .ok_or_else(|| PyValueError::new_err(format!("no feature service named `{service}`")))?;
    let references = declared.references().map_err(core_error)?;
    let features: Vec<(String, String)> = references
        .iter()
        .map(|(view, feature)| ((*view).to_owned(), (*feature).to_owned()))
        .collect();

    let views: BTreeMap<String, feather_core::FeatureView> = definitions
        .views
        .iter()
        .map(|v| (v.name.clone(), v.clone()))
        .collect();
    // The entity every view in the service shares. One hash key per entity is what makes a
    // single store read cover the whole request, so this is not per-view: a service whose
    // views are keyed on different entities would look up the wrong bytes, miss, and come back
    // as a null for every column, which reads as "feature absent" rather than "service
    // misconfigured". `get_online_features` already rejects that case, and so does this.
    let mut entity_name: Option<String> = None;
    for (view, _) in &references {
        let declared = views
            .get(*view)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "feature service `{service}` names view `{view}`, which the definitions do not declare"
                ))
            })?
            .entity()
            .map_err(core_error)?
            .name
            .clone();
        match &entity_name {
            None => entity_name = Some(declared),
            Some(first) if *first != declared => {
                return Err(PyValueError::new_err(format!(
                    "feature service `{service}` spans views keyed on different entities \
                     (`{first}` and `{declared}`); every view in a service must share one, \
                     because the entity is part of the store key"
                )));
            }
            Some(_) => {}
        }
    }
    let entity_name = entity_name.unwrap_or_default();

    let resolved = feather_serve::ResolvedService::resolve(
        &definitions.project,
        &feather_serve::ServiceSpec {
            name: service.to_owned(),
            entity_name,
            features,
        },
        &views,
    )
    .map_err(PyValueError::new_err)?;

    let (directory, cache, memtable) = online_location(settings_path, settings.store.as_ref());

    let socket: std::net::SocketAddr = addr
        .parse()
        .map_err(|e| PyValueError::new_err(format!("bad address `{addr}`: {e}")))?;

    let store = Arc::new(
        feather_core::online::fjall::FjallStore::open(&directory, cache, memtable)
            .map_err(core_error)?,
    );
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .enable_all()
        .build()
        .map_err(|e| PyOSError::new_err(e.to_string()))?;

    // The GIL released across the whole blocking serve, which is what lets the server's own
    // workers run on every core and the caller's other Python threads keep going. Measured in
    // `docs/serving-transport.md`: a Python host serves at 0.91 of a plain Rust one.
    py.detach(|| {
        runtime.block_on(feather_serve::serve(
            socket,
            store,
            resolved,
            std::future::pending(),
        ))
    })
    .map_err(|e| PyOSError::new_err(e.to_string()))
}

/// The compiled extension. Imported as `feather._core`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(load_settings, m)?)?;
    m.add_function(wrap_pyfunction!(demo::write_demo_data, m)?)?;
    m.add_class::<FeatureStore>()?;
    #[cfg(feature = "serve")]
    m.add_function(wrap_pyfunction!(serve, m)?)?;
    #[cfg(feature = "offline")]
    {
        m.add_class::<MaterializeReport>()?;
        m.add_class::<ViewRefresh>()?;
    }
    Ok(())
}
