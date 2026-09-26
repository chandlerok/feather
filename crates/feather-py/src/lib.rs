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

mod demo;

#[cfg(feature = "offline")]
use std::collections::{BTreeMap, HashMap};
#[cfg(feature = "offline")]
use std::sync::Mutex;

#[cfg(feature = "offline")]
use arrow::array::new_null_array;
#[cfg(feature = "offline")]
use arrow::array::{Array, ArrayRef, Int64Array, UInt32Array};
#[cfg(feature = "offline")]
use arrow::compute::{concat, concat_batches, take};
#[cfg(feature = "offline")]
use arrow::datatypes::{Field as ArrowField, Schema};
#[cfg(feature = "offline")]
use arrow::record_batch::RecordBatch;
#[cfg(feature = "offline")]
use pyo3_arrow::PyTable;

#[cfg(feature = "offline")]
use feather_core::offline::ROW_COLUMN;
#[cfg(feature = "offline")]
use feather_core::online::memory::MemoryStore;
#[cfg(all(feature = "offline", feature = "valkey"))]
use feather_core::online::valkey::ValkeyStore;
#[cfg(feature = "offline")]
use feather_core::online::{
    EntityRequest, OnlineStore, ProjectScan, ReadRequest, ViewRequest, ViewValues, WriteBatch,
};
#[cfg(feature = "offline")]
use feather_core::settings::Valkey as ValkeySettings;
#[cfg(feature = "offline")]
use feather_core::{
    Definitions, FeatureView, JoinOptions, Limits, OfflineEngine, OnMissing, entity_key_component,
    materialize as materialize_project, read_entities, value,
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
/// validates. The engine is opened once and reused, because opening it creates a private spill
/// directory and may load a filesystem extension.
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
#[cfg(feature = "offline")]
#[pyclass]
struct FeatureStore {
    inner: Mutex<Inner>,
}

/// Everything a store owns, behind its mutex.
#[cfg(feature = "offline")]
struct Inner {
    /// The runtime the online store's I/O runs on.
    ///
    /// Created once with the store rather than per call, and multi-threaded because the Valkey
    /// client's connection manager needs a reactor to drive it. Blocking the Python thread on
    /// it with `block_on` is sound here: that thread is not one of its workers, so there is no
    /// runtime to nest inside.
    runtime: tokio::runtime::Runtime,
    engine: OfflineEngine,
    definitions: Definitions,
    /// The project name, which namespaces every online key.
    project: String,
    /// The Valkey connection the settings declare, if any.
    valkey: Option<ValkeySettings>,
    /// The store features are served from, opened on the first online call.
    ///
    /// `None` until then, so that constructing a store over a project whose Valkey is down
    /// still works, and so that a project that never reads online never opens a connection.
    online: Option<Online>,
}

/// The online store a project serves from, which one it is decided by the settings.
#[cfg(feature = "offline")]
enum Online {
    /// Local mode: the settings declare no Valkey, so the values live in this process.
    Memory(MemoryStore),
    /// The settings declare a Valkey.
    #[cfg(feature = "valkey")]
    Valkey(Box<ValkeyStore>),
}

#[cfg(feature = "offline")]
impl OnlineStore for Online {
    async fn write(&mut self, batches: &[WriteBatch]) -> feather_core::Result<()> {
        match self {
            Online::Memory(store) => store.write(batches).await,
            #[cfg(feature = "valkey")]
            Online::Valkey(store) => store.write(batches).await,
        }
    }

    async fn read(
        &self,
        requests: &[ReadRequest],
    ) -> feather_core::Result<Vec<Vec<Option<Vec<u8>>>>> {
        match self {
            Online::Memory(store) => store.read(requests).await,
            #[cfg(feature = "valkey")]
            Online::Valkey(store) => store.read(requests).await,
        }
    }

    async fn delete_fields(
        &mut self,
        keys_and_fields: &[(Vec<u8>, Vec<String>)],
    ) -> feather_core::Result<()> {
        match self {
            Online::Memory(store) => store.delete_fields(keys_and_fields).await,
            #[cfg(feature = "valkey")]
            Online::Valkey(store) => store.delete_fields(keys_and_fields).await,
        }
    }
}

#[cfg(feature = "offline")]
impl ProjectScan for Online {
    async fn hash_fields(&self, key: &[u8]) -> feather_core::Result<Vec<(String, Vec<u8>)>> {
        match self {
            Online::Memory(store) => store.hash_fields(key).await,
            #[cfg(feature = "valkey")]
            Online::Valkey(store) => store.hash_fields(key).await,
        }
    }

    async fn scan_entity_keys(
        &self,
        project: &str,
        exclude: &[u8],
    ) -> feather_core::Result<Vec<Vec<u8>>> {
        match self {
            Online::Memory(store) => store.scan_entity_keys(project, exclude).await,
            #[cfg(feature = "valkey")]
            Online::Valkey(store) => store.scan_entity_keys(project, exclude).await,
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
#[cfg(feature = "offline")]
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

/// A panic while another thread held the lock is what poisons it, which means the engine's
/// state is unknown; saying so beats unwrapping into a second panic.
#[cfg(feature = "offline")]
fn poisoned<T>(_: std::sync::PoisonError<T>) -> PyErr {
    PyOSError::new_err(
        "another thread panicked while using this FeatureStore, so its engine cannot be trusted",
    )
}

/// Open the project's online store if it is not open yet.
///
/// Opening is deferred to the first call that needs it, so a project can be constructed without
/// a reachable Valkey and a project that only reads historical features never connects.
///
/// Args:
///     inner: The store's state.
///
/// Returns:
///     `Ok(())` once `inner.online` holds a store.
///
/// Raises:
///     ConnectionError: If the settings declare a Valkey that cannot be reached, naming it.
///     RuntimeError: If the settings declare a Valkey and this build has no Valkey support.
#[cfg(feature = "offline")]
fn open_online(inner: &mut Inner) -> PyResult<()> {
    if inner.online.is_some() {
        return Ok(());
    }
    let opened = {
        let Inner {
            runtime, valkey, ..
        } = &*inner;
        connect_online(runtime, valkey.as_ref())?
    };
    inner.online = Some(opened);
    Ok(())
}

/// Build the store the settings name.
///
/// Args:
///     runtime: The runtime the Valkey client's I/O runs on.
///     configured: The settings' `[valkey]` table, or `None` for local mode.
///
/// Returns:
///     A Valkey-backed store when the settings declare one, and the in-process store
///     otherwise, which is the document's local mode.
///
/// Raises:
///     ConnectionError: If the declared Valkey cannot be reached. The message names the
///         endpoint, because a connection failure that does not say what it could not reach is
///         the least useful kind.
///     RuntimeError: If the settings declare a Valkey and this build has no Valkey support.
#[cfg(feature = "offline")]
fn connect_online(
    runtime: &tokio::runtime::Runtime,
    configured: Option<&ValkeySettings>,
) -> PyResult<Online> {
    let Some(configured) = configured else {
        return Ok(Online::Memory(MemoryStore::new()));
    };
    #[cfg(feature = "valkey")]
    {
        // The settings keep the transport separate from the address, because a security flag
        // guessed wrong is worse than one an operator wrote down; the client wants one URL.
        let scheme = if configured.tls { "rediss" } else { "redis" };
        let url = format!("{scheme}://{}", configured.endpoint);
        let store = runtime
            .block_on(ValkeyStore::connect(&url))
            .map_err(|error| {
                // Named in full rather than imported: the only use is behind this feature, and an
                // import that is unused without it is one an unused-import cleanup will remove,
                // which breaks the `--all-features` build for the sake of the default one.
                pyo3::exceptions::PyConnectionError::new_err(format!(
                    "could not connect to the Valkey at {}: {error}",
                    configured.endpoint
                ))
            })?
            .with_field_expiration(configured.field_expiration);
        Ok(Online::Valkey(Box::new(store)))
    }
    #[cfg(not(feature = "valkey"))]
    {
        let _ = runtime;
        Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "the settings declare a Valkey at {}, and this build has no Valkey support; install a build with the `valkey` feature",
            configured.endpoint
        )))
    }
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
        let engine =
            OfflineEngine::open(&Limits::default(), &settings.connections).map_err(core_error)?;
        Ok(Self {
            inner: Mutex::new(Inner {
                runtime,
                engine,
                project: definitions.project.clone(),
                definitions,
                valkey: settings.valkey,
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

        let references = parse_references(&features)?;

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
    ///     ConnectionError: If the settings declare a Valkey that cannot be reached.
    ///     OSError: If another thread panicked while holding this store.
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
    ///     ConnectionError: If the settings declare a Valkey that cannot be reached.
    ///     OSError: If another thread panicked while holding this store.
    fn get_online_features(
        &self,
        py: Python<'_>,
        entity_frame: PyTable,
        features: Vec<String>,
    ) -> PyResult<PyTable> {
        let references = parse_references(&features)?;
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

        // The frame's own columns come back untouched: only the value writes are reordered, so
        // reattaching by position is exact rather than a join.
        let mut columns: Vec<ArrayRef> =
            Vec::with_capacity(entity.num_columns() + references.len());
        let mut fields: Vec<ArrowField> =
            Vec::with_capacity(entity.num_columns() + references.len());
        for (index, field) in entity_schema.fields().iter().enumerate() {
            columns.push(entity.column(index).clone());
            fields.push(field.as_ref().clone());
        }

        for (reference, (view_slot, feature_slot)) in references.iter().zip(&slots) {
            let declared = views[&reference.view]
                .field(&reference.feature)
                .expect("checked against the definitions above");
            let dtype = value::arrow_type(declared.dtype);

            // One element per entity that has a value, in entity order, and an index into it or
            // a null for each entity that does not. `take` turns a null index into a null of the
            // right type, which is how an absent value reaches the caller.
            let mut present: Vec<ArrayRef> = Vec::new();
            let mut indices: Vec<Option<u32>> = Vec::with_capacity(values.len());
            for per_view in &values {
                match &per_view[*view_slot] {
                    ViewValues::Present {
                        columns: decoded, ..
                    } => {
                        indices.push(Some(present.len() as u32));
                        present.push(decoded[*feature_slot].clone());
                    }
                    ViewValues::Missing(_) => indices.push(None),
                }
            }

            let column = if present.is_empty() {
                new_null_array(&dtype, values.len())
            } else {
                let borrowed: Vec<&dyn Array> =
                    present.iter().map(|array| array.as_ref()).collect();
                let decoded = concat(&borrowed).map_err(arrow_error)?;
                take(&decoded, &UInt32Array::from(indices), None).map_err(arrow_error)?
            };
            columns.push(column);
            fields.push(ArrowField::new(&reference.feature, dtype, true));
        }

        let schema = std::sync::Arc::new(Schema::new(fields));
        let output = RecordBatch::try_new(schema.clone(), columns).map_err(arrow_error)?;
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
#[cfg(feature = "offline")]
fn now_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_micros() as i64)
        .unwrap_or(0)
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
    m.add_function(wrap_pyfunction!(demo::write_demo_data, m)?)?;
    #[cfg(feature = "offline")]
    {
        m.add_class::<FeatureStore>()?;
        m.add_class::<MaterializeReport>()?;
        m.add_class::<ViewRefresh>()?;
    }
    Ok(())
}
