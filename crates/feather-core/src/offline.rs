//! Offline point-in-time joins, executed in the embedded DuckDB engine.
//!
//! One engine, DuckDB, for every source. A local Parquet file, an object-storage prefix, and
//! a warehouse table are read by different readers and joined the same way, so nothing here
//! selects a compute backend: [`Engine::relation`] is the one place a source becomes
//! a relation, and so the one place its kind and a file source's format decide which
//! reader its rows come from.
//!
//! The rules are the ones specified under "Point-in-time join semantics" in the
//! architecture document, and they live here rather than in the caller because getting them
//! subtly wrong produces a training set that looks fine and is not. The three that matter
//! most:
//!
//! - **TTL is a window, not a filter.** DuckDB's `ASOF JOIN` accepts exactly one inequality
//!   condition, on the ordering column, so the TTL predicate is evaluated after the join.
//!   A value exactly `ttl_days` old is still fresh; only an older one is stale, which is
//!   what "older than its TTL" means on the online path too.
//! - **Row order is restored.** `ASOF JOIN` does not preserve the left input's order, so the
//!   entity frame carries an explicit row index and the result is ordered by it. Without
//!   that step, labels and features misalign silently, which is the most damaging failure
//!   mode available here.
//! - **Nothing is coerced.** A key or timestamp type that does not match the source is an
//!   error rather than a cast, because an implicit cast is how an off-by-hours bug enters a
//!   training set.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use arrow::array::{ArrayRef, Int64Array};
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use duckdb::{Arrow, Connection};

use crate::definitions::{DType, FeatureView, FileFormat, Source};
use crate::error::{Error, Result};
use crate::settings::Connection as SettingsConnection;
use crate::value::arrow_type;

/// The column the join attaches to the entity frame, so output order is input order.
pub const ROW_COLUMN: &str = "feather_row";

/// Names used inside the generated SQL, prefixed so a collision with a caller's column is
/// not possible.
const LABELS_TABLE: &str = "feather_labels";
const RESULT_TABLE: &str = "feather_result";
const LABEL_KEY: &str = "feather_key";
const LABEL_TS: &str = "feather_ts";
const MATCHED_TS: &str = "feather_matched_ts";

/// The columns a materialization scan lays its batches out under.
///
/// Public because the write path names them when it reports a malformed batch, and one
/// spelling of a name is cheaper than two.
pub const SCAN_KEY_COLUMN: &str = "feather_entity_key";
pub const SCAN_TS_COLUMN: &str = "feather_event_ts";

/// The default ceiling on what the spill directory may hold.
///
/// DuckDB's own default is 90% of free disk, which an engine embedded in someone else's
/// process has no business taking.
pub const DEFAULT_SPILL_CAP: &str = "20GB";

/// How much memory DuckDB may use before it spills, and where it spills to.
///
/// Both belong to the deployment rather than to a query, which is why they are set once
/// when the engine opens.
#[derive(Debug, Clone)]
pub struct Limits {
    /// DuckDB's `memory_limit`, such as `2GB`.
    ///
    /// `None` uses the container's limit when there is one. DuckDB's own default is 80% of
    /// the machine's RAM, and inside a container that can be more than the container is
    /// allowed to use, so the kernel kills the process before DuckDB ever spills.
    pub memory_limit: Option<String>,
    /// The parent of this engine's spill directory, which is private to it.
    ///
    /// Local disk. DuckDB creates the directory, removes it on shutdown, and reads and
    /// writes it synchronously, so on a network mount every spill block becomes a round
    /// trip, which is the usual reason spilling is called catastrophic rather than slow.
    pub temp_directory: PathBuf,
    /// Cap on what the spill directory may hold, such as `20GB`.
    ///
    /// `None` leaves DuckDB's default of 90% of free disk.
    pub max_temp_directory_size: Option<String>,
    /// Where DuckDB looks for loadable extensions.
    ///
    /// `None` uses DuckDB's default, `~/.duckdb/extensions/<version>/<platform>`. Setting it
    /// is what lets an image bake the extension files in and read remote sources with no
    /// network at runtime.
    pub extension_directory: Option<PathBuf>,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            memory_limit: None,
            temp_directory: std::env::temp_dir().join("feather-spill"),
            max_temp_directory_size: Some(DEFAULT_SPILL_CAP.to_owned()),
            extension_directory: None,
        }
    }
}

/// What happens to a label row whose features are missing or expired.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OnMissing {
    /// Keep the row and leave its features null.
    ///
    /// The default, because dropping rows silently changes the size and class balance of a
    /// training set, and a job that shrinks without saying so is worse than one carrying
    /// explicit nulls.
    #[default]
    Null,
    /// Remove the row.
    Drop,
}

/// How one join is run.
#[derive(Debug, Clone)]
pub struct JoinOptions {
    /// The entity frame's label timestamp column.
    ///
    /// A name in the caller's frame, not in the source; the source's timestamp column is
    /// the view's `timestamp_field`.
    pub label_timestamp_column: String,
    /// Exclude a feature row at exactly the label time instead of including it.
    pub strict: bool,
    /// What to do with a row whose features are missing or expired.
    pub on_missing: OnMissing,
}

impl Default for JoinOptions {
    fn default() -> Self {
        Self {
            label_timestamp_column: "event_timestamp".to_owned(),
            strict: false,
            on_missing: OnMissing::Null,
        }
    }
}

/// An in-memory DuckDB engine.
pub struct Engine {
    /// The connection this engine reads through, or `None` after a failure that left its
    /// state unknown. Opening it lazily is what lets the next use reopen instead of reuse.
    ///
    /// A mutex rather than a `RefCell` so that `Engine` is `Sync`. The binding releases the GIL
    /// around a refresh, and PyO3 only allows that for a closure whose captures are all `Sync`,
    /// which a `RefCell` is not. The lock is uncontended: one DuckDB connection is one session,
    /// and the engine is used from the store's own mutex.
    connection: Mutex<Option<Connection>>,
    /// Retained so a discarded connection is rebuilt with the settings it had.
    limits: Limits,
    /// This engine's private spill directory. DuckDB removes the files inside it when the
    /// instance shuts down; the directory itself is removed when the engine is dropped.
    spill: PathBuf,
    /// The project's named connections, retained because a source resolves its
    /// `connection` by name when it is read. The credentials are already in DuckDB as
    /// secrets, so keeping them here adds no exposure.
    connections: BTreeMap<String, SettingsConnection>,
}

impl Engine {
    /// Open an engine that spills into a private directory under `limits.temp_directory`.
    ///
    /// Args:
    ///     limits: The memory ceiling, the spill parent directory, the spill cap, and where
    ///         loadable extensions live.
    ///     connections: The project's named connections, keyed by the name a source refers to.
    ///
    /// Returns:
    ///     The engine.
    ///
    /// Raises:
    ///     [`Error::SpillDirectory`] if the spill or extension directory cannot be created.
    ///         Checked here because DuckDB creates the directory it spills into but not its
    ///         parents, so a missing parent otherwise fails at the first spill, under load,
    ///         rather than at startup.
    ///     [`Error::DuckDb`] if the connection cannot be opened, a setting is rejected, or a
    ///         configured secret is malformed.
    pub fn open(
        limits: &Limits,
        connections: &BTreeMap<String, SettingsConnection>,
    ) -> Result<Self> {
        // Private to this engine: DuckDB removes its spill directory when the instance shuts
        // down, so a directory shared between engines means one engine's shutdown can delete
        // another's live spill files.
        let spill = limits.temp_directory.join(engine_id());
        std::fs::create_dir_all(&spill).map_err(|source| Error::SpillDirectory {
            path: spill.display().to_string(),
            source,
        })?;

        let engine = Self {
            connection: Mutex::new(None),
            limits: limits.clone(),
            spill,
            connections: connections.clone(),
        };
        engine.reopen()?;
        Ok(engine)
    }

    /// Open the connection this engine reads through, replacing a discarded one.
    ///
    /// Returns:
    ///     `Ok(())` once a connection with the engine's settings, secrets, and spill
    ///     directory is in place.
    ///
    /// Raises:
    ///     [`Error::DuckDb`] if the connection cannot be opened, a setting is rejected, or a
    ///         configured secret is malformed.
    fn reopen(&self) -> Result<()> {
        *lock(&self.connection) = Some(self.connect()?);
        Ok(())
    }

    /// The connection this engine reads through, opening one if the last was discarded.
    ///
    /// Returns:
    ///     A borrow of the live connection.
    ///
    /// Raises:
    ///     [`Error::DuckDb`] if a discarded connection cannot be reopened.
    fn connection(&self) -> Result<ConnectionRef<'_>> {
        if lock(&self.connection).is_none() {
            self.reopen()?;
        }
        Ok(ConnectionRef(lock(&self.connection)))
    }

    /// Discard the connection, so the next use opens a fresh one.
    ///
    /// Called where a failure left the connection's state unknown, which is a caught panic
    /// inside a streamed scan. A scan that is still holding the connection makes this wait for
    /// it rather than panicking, which is what the mutex buys: two scans over one engine on one
    /// thread, which nothing does today, would serialise instead of one of them aborting the
    /// process.
    fn discard_connection(&self) {
        *lock(&self.connection) = None;
    }

    /// Build a connection with this engine's settings, secrets, and spill directory.
    ///
    /// Split out of [`Engine::open`] so a connection discarded after a failure is rebuilt
    /// identically rather than approximately.
    ///
    /// Raises:
    ///     [`Error::SpillDirectory`] if the extension directory cannot be created.
    ///     [`Error::DuckDb`] if the connection cannot be opened, a setting is rejected, or a
    ///         configured secret is malformed.
    fn connect(&self) -> Result<Connection> {
        let connection = Connection::open_in_memory()?;

        let mut settings = vec![format!(
            "SET temp_directory = {}",
            quote_literal(&self.spill.display().to_string())
        )];
        // An explicit limit wins; otherwise take the container's, and otherwise leave
        // DuckDB's own default of 80% of RAM alone.
        if let Some(memory) = self
            .limits
            .memory_limit
            .clone()
            .or_else(cgroup_memory_limit)
        {
            settings.push(format!("SET memory_limit = {}", quote_literal(&memory)));
        }
        if let Some(cap) = &self.limits.max_temp_directory_size {
            settings.push(format!(
                "SET max_temp_directory_size = {}",
                quote_literal(cap)
            ));
        }
        if let Some(directory) = &self.limits.extension_directory {
            std::fs::create_dir_all(directory).map_err(|source| Error::SpillDirectory {
                path: directory.display().to_string(),
                source,
            })?;
            settings.push(format!(
                "SET extension_directory = {}",
                quote_literal(&directory.display().to_string())
            ));
        }
        connection.execute_batch(&settings.join("; "))?;

        // A configured object-store credential becomes a secret. Without one, DuckDB's
        // credential chain finds the instance role or the standard environment variables,
        // which is the normal case in the cloud and needs no configuration at all.
        //
        // The secret *type* comes from httpfs, so the extension has to be loaded before the
        // secret can be created. That is the only reason a configured connection loads it at
        // open; a remote path loads it on first read instead.
        let secrets: Vec<String> = self
            .connections
            .iter()
            .filter_map(|(name, configured)| s3_secret_sql(name, configured))
            .collect();
        if !secrets.is_empty() {
            connection.execute_batch("INSTALL httpfs; LOAD httpfs;")?;
            for sql in &secrets {
                connection.execute_batch(sql)?;
            }
        }

        Ok(connection)
    }

    /// The directory this engine spills into.
    pub fn spill_directory(&self) -> &Path {
        &self.spill
    }

    /// Load what a source needs to be read, the first time a view is joined.
    ///
    /// Deferred rather than done at open, so a project reading local files pays nothing, and
    /// done here rather than left to DuckDB's autoload so that a machine which cannot obtain
    /// the extension says so, instead of failing later with an opaque read error.
    ///
    /// Args:
    ///     view: The view whose source decides which extension, if any, is needed.
    ///
    /// Returns:
    ///     `Ok(())` once the filesystem or scanner the source needs is loaded.
    ///
    /// Raises:
    ///     [`Error::UnknownSourceFormat`] if a file source names a format Feather has no
    ///         reader for. Resolved before anything is installed, so a format mistake
    ///         fails as itself rather than after a fetch.
    ///     [`Error::UnknownConnection`] or [`Error::SourceConnectionKind`] if a Postgres
    ///         source names a connection the project cannot use. Checked before the
    ///         extension is installed, so a connection mistake fails as itself rather
    ///         than after a fetch.
    ///     [`Error::ExtensionUnavailable`] if a file source's format extension, or the
    ///         filesystem a remote path needs, cannot be installed or loaded. A Postgres
    ///         source's extension is not covered: it is still [`Error::DuckDb`].
    ///     [`Error::DuckDb`] if the connection cannot be used.
    fn ensure_source_loaded(&self, view: &FeatureView) -> Result<()> {
        let format = view.source_format()?;
        match &view.source {
            Source::File { path, .. } => {
                for (extension, reader) in extensions_for(path, format) {
                    self.connection()?
                        .execute_batch(&format!("INSTALL {extension}; LOAD {extension};"))
                        .map_err(|source| {
                            Error::extension_unavailable(
                                view.name.clone(),
                                extension,
                                reader,
                                source,
                            )
                        })?;
                }
            }
            Source::Postgres { connection, .. } => {
                self.postgres_conninfo(view, connection)?;
                self.connection()?
                    .execute_batch("INSTALL postgres; LOAD postgres;")?;
            }
        }
        Ok(())
    }

    /// The libpq connection string a view's Postgres source reads through.
    ///
    /// The name lookup, the kind check, and libpq's own string form meet here, so a
    /// reader has one place that turns `connection = "pg_prod"` into something connectable.
    fn postgres_conninfo(&self, view: &FeatureView, connection: &str) -> Result<String> {
        match self.connections.get(connection) {
            None => Err(Error::UnknownConnection {
                view: view.name.clone(),
                connection: connection.to_owned(),
            }),
            Some(configured) => configured
                .conninfo()
                .ok_or_else(|| Error::SourceConnectionKind {
                    view: view.name.clone(),
                    source_kind: view.source.kind().to_owned(),
                    connection: connection.to_owned(),
                    connection_kind: configured.kind().to_owned(),
                }),
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // This runs before the connection is dropped, so DuckDB has not cleared its own
        // spill files yet. Unlinking them anyway is safe on POSIX, and the point is that a
        // dropped engine leaves nothing behind for the next one to find.
        let _ = std::fs::remove_dir_all(&self.spill);
    }
}

/// A live connection, held for as long as the caller that borrows it.
///
/// The guard is what makes an engine `Sync` without an `unsafe impl`, and the `Deref` is what
/// keeps the `Option` out of every reader: `connection` opens one before handing this out, so
/// the only way to hold a `ConnectionRef` is to hold a connection.
struct ConnectionRef<'a>(MutexGuard<'a, Option<Connection>>);

impl std::ops::Deref for ConnectionRef<'_> {
    type Target = Connection;

    fn deref(&self) -> &Connection {
        self.0
            .as_ref()
            .expect("connection opens one before returning a reference to it")
    }
}

/// Take a mutex, ignoring poisoning.
///
/// A panic inside a scan poisons the connection's mutex while the connection is being thrown
/// away anyway, so treating it as fatal would replace one caught failure with a permanent one.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Distinguishes one engine's spill directory from another's.
fn engine_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    format!(
        "{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

/// The memory ceiling implied by the container this process is running in.
///
/// Returns `None` when there is no cgroup limit, which is the normal case on a laptop and
/// means DuckDB's own default applies.
fn cgroup_memory_limit() -> Option<String> {
    // cgroup v2, then v1. Neither path exists on macOS.
    for path in [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ] {
        if let Ok(contents) = std::fs::read_to_string(path) {
            return memory_limit_from_cgroup(&contents);
        }
    }
    None
}

/// The ceiling to set from a cgroup file's contents, at 80% of the limit.
///
/// Args:
///     contents: The file's contents, which may be the literal `max` when unlimited.
///
/// Returns:
///     A byte count as a string, or `None` when the file reports no real limit.
fn memory_limit_from_cgroup(contents: &str) -> Option<String> {
    let bytes: u64 = contents.trim().parse().ok()?;
    // Both cgroup versions report an enormous sentinel rather than a flag when unlimited.
    if bytes == 0 || bytes >= (1 << 62) {
        return None;
    }
    // Divided before multiplying, because 8x a limit near the sentinel would overflow.
    Some((bytes / 10 * 8).to_string())
}

/// Query methods, in their own block so the settings a deployment owns stay next to `open`.
impl Engine {
    /// Join one view onto an entity frame, in the frame's own row order.
    ///
    /// Args:
    ///     entity_frame: The label rows. Its entity key column is named after the view's
    ///         entity `join_key`, and its label timestamp column is named by
    ///         [`JoinOptions::label_timestamp_column`].
    ///     view: The view to join. Its source is read as Parquet or from a Postgres
    ///         table, depending on the kind it declares.
    ///     options: Ordering and missingness rules for this join.
    ///
    /// Returns:
    ///     A batch of [`ROW_COLUMN`] followed by one column per declared feature, ordered by
    ///     [`ROW_COLUMN`]. The row index is the position in `entity_frame` the features
    ///     belong to, which is what lets a caller reattach its own label columns and what
    ///     makes a dropped row visible.
    ///
    /// Raises:
    ///     [`Error::MissingColumn`] if either side lacks a column the join needs.
    ///     [`Error::UnsupportedOfflineType`] for a key or timestamp type the join cannot
    ///         compare.
    ///     [`Error::OfflineTypeMismatch`] if the two sides' key families differ, or a naive
    ///         timestamp is joined against a zoned one.
    ///     [`Error::ColumnTypeMismatch`] if a source column does not have the declared dtype.
    ///     [`Error::UnknownConnection`] or [`Error::SourceConnectionKind`] if a Postgres
    ///         source names a connection the project cannot use.
    ///     [`Error::AmbiguousTimestamp`] if the source holds more than one row for a key and
    ///         timestamp and declares no `created_timestamp_field` to break the tie.
    ///     [`Error::UnreadableSource`] if the source cannot be read at all, which names the
    ///         view it was being read for.
    ///     [`Error::DuckDb`] if the query itself fails.
    pub fn point_in_time_join(
        &self,
        entity_frame: &RecordBatch,
        view: &FeatureView,
        options: &JoinOptions,
    ) -> Result<RecordBatch> {
        // One entity per view in v1, which `entity` also enforces.
        let entity = view.entity()?;
        let key_column = entity.join_key.as_str();

        let frame_schema = entity_frame.schema();
        let key_index = column_index(&frame_schema, key_column)?;
        let label_index = column_index(&frame_schema, &options.label_timestamp_column)?;
        let key_type = frame_schema.field(key_index).data_type().clone();
        let label_type = frame_schema.field(label_index).data_type().clone();
        let key_family = KeyFamily::of_arrow(&key_type)?;
        let label_kind = TimestampKind::of_arrow(&label_type)?;

        self.ensure_source_loaded(view)?;

        let location = format!("source `{}`", view.source.description());
        let described = self.describe_source(view)?;

        let source_key = described
            .get(key_column)
            .ok_or_else(|| Error::MissingColumn {
                location: location.clone(),
                column: key_column.to_owned(),
            })?;
        if KeyFamily::of_duckdb(source_key)? != key_family {
            return Err(Error::OfflineTypeMismatch {
                label: key_column.to_owned(),
                label_type: key_type.to_string(),
                source_column: key_column.to_owned(),
                source_type: source_key.clone(),
            });
        }

        let source_ts_column = view.timestamp_field();
        let source_ts = described
            .get(source_ts_column)
            .ok_or_else(|| Error::MissingColumn {
                location: location.clone(),
                column: source_ts_column.to_owned(),
            })?;
        let source_ts_kind = TimestampKind::of_duckdb(source_ts)?;
        if !label_kind.comparable_with(source_ts_kind) {
            return Err(Error::OfflineTypeMismatch {
                label: options.label_timestamp_column.clone(),
                label_type: label_type.to_string(),
                source_column: source_ts_column.to_owned(),
                source_type: source_ts.clone(),
            });
        }

        // Checked here rather than left to the query, so a source schema change fails with a
        // named column rather than arriving as a column of nulls.
        for field in &view.features {
            let actual = described
                .get(&field.name)
                .ok_or_else(|| Error::MissingColumn {
                    location: location.clone(),
                    column: field.name.clone(),
                })?;
            let expected = duckdb_type_of(field.dtype);
            if normalize_type(actual) != expected {
                return Err(Error::ColumnTypeMismatch {
                    name: field.name.clone(),
                    declared: expected.to_owned(),
                    actual: actual.clone(),
                });
            }
        }

        self.stage_labels(entity_frame, &frame_schema, key_index, label_index)?;

        let relation = self.relation(view)?;
        let key_expr = quote_ident(key_column);
        let source_ts_value = source_ts_kind.to_micros(&quote_ident(source_ts_column));

        if view.created_timestamp_field.is_none() {
            self.reject_ambiguous_timestamps(
                view,
                &relation,
                &key_expr,
                &source_ts_value,
                AmbiguityScope::LabelKeys,
            )?;
        }

        let sql = join_sql(
            view,
            options,
            &relation,
            &key_expr,
            &source_ts_value,
            label_kind,
        );
        let batches = self.materialize_and_scan(&sql)?;

        let schema = Arc::new(Schema::new(output_fields(view)));
        if batches.is_empty() {
            return Ok(RecordBatch::new_empty(schema));
        }
        Ok(concat_batches(&schema, &batches)?)
    }

    /// The source's column names and DuckDB types, read without scanning any data.
    ///
    /// Raises:
    ///     [`Error::UnreadableSource`] if the source cannot be described, which is where a
    ///         wrong host, password, or table name surfaces. The failure is attributed to
    ///         the view here, because the scanner's own text holds neither the view nor the
    ///         connection name.
    fn describe_source(&self, view: &FeatureView) -> Result<TypeMap> {
        let sql = format!("DESCRIBE SELECT * FROM {}", self.relation(view)?);
        let connection = self.connection()?;
        let describe = || -> duckdb::Result<TypeMap> {
            let mut statement = connection.prepare(&sql)?;
            let mut rows = statement.query([])?;
            let mut types = TypeMap::default();
            while let Some(row) = rows.next()? {
                // DESCRIBE's first two columns are column_name and column_type.
                types.insert(row.get(0)?, row.get(1)?);
            }
            Ok(types)
        };
        // The scanner's text is kept, because it is the only description of what went wrong
        // and it may echo part of the conninfo. Wrapping rather than replacing it puts that
        // text behind a message that says which view was reading.
        describe().map_err(|error| Error::UnreadableSource {
            view: view.name.clone(),
            location: format!("source `{}`", view.source.description()),
            reason: error.to_string(),
        })
    }

    /// Put the entity frame into a temporary table with an explicit row index.
    fn stage_labels(
        &self,
        entity_frame: &RecordBatch,
        frame_schema: &Schema,
        key_index: usize,
        label_index: usize,
    ) -> Result<()> {
        let connection = self.connection()?;
        connection.execute_batch(&format!("DROP TABLE IF EXISTS {LABELS_TABLE}"))?;
        connection.execute_batch(&format!(
            "CREATE TEMPORARY TABLE {LABELS_TABLE} (feather_row BIGINT, {LABEL_KEY} {}, {LABEL_TS} {})",
            arrow_duckdb_type(frame_schema.field(key_index).data_type())?,
            arrow_duckdb_type(frame_schema.field(label_index).data_type())?,
        ))?;

        let batch = label_batch(entity_frame, frame_schema, key_index, label_index)?;
        let mut appender = connection.appender(LABELS_TABLE)?;
        appender.append_record_batch(batch)?;
        appender.flush()?;
        Ok(())
    }

    /// Fail if two source rows share a key and timestamp, so the winner would be undefined.
    fn reject_ambiguous_timestamps(
        &self,
        view: &FeatureView,
        relation: &str,
        key_expr: &str,
        source_ts_value: &str,
        scope: AmbiguityScope,
    ) -> Result<()> {
        // A join only cares about the keys its frame names, and restricting the check to them is
        // what keeps it bounded by the request rather than by the source. A refresh has no such
        // frame, so it looks at every key: which row wins has to be decided the same way for
        // every entity a source holds, not only the ones some label row happened to name.
        let restriction = match scope {
            AmbiguityScope::LabelKeys => {
                format!(" WHERE k IN (SELECT {LABEL_KEY} FROM {LABELS_TABLE})")
            }
            AmbiguityScope::WholeSource => String::new(),
        };
        // Rows the scan itself excludes are excluded here too. A null key or a null timestamp is
        // dropped before any value is written, so two such rows are not an ambiguity the refresh
        // has to resolve. Without this the check would group them and then read a null key back
        // as a string, which is an error rather than a value, so a source the scan handles fine
        // would fail the whole refresh with a message naming neither the source nor the rows.
        let sql = format!(
            "WITH source AS (
                 SELECT {key_expr} AS k, {source_ts_value} AS t FROM {relation}
                 WHERE {key_expr} IS NOT NULL AND {source_ts_value} IS NOT NULL
             )
             SELECT CAST(k AS VARCHAR), t, count(*) AS n
             FROM source{restriction}
             GROUP BY k, t
             HAVING count(*) > 1
             LIMIT 1"
        );
        let connection = self.connection()?;
        let mut statement = connection.prepare(&sql)?;
        let mut rows = statement.query([])?;
        if let Some(row) = rows.next()? {
            return Err(Error::AmbiguousTimestamp {
                view: view.name.clone(),
                key_column: view.entities[0].join_key.clone(),
                key: row.get(0)?,
                timestamp: row.get(1)?,
                rows: row.get(2)?,
            });
        }
        Ok(())
    }

    /// Execute the join into a temporary table, then read it back in row order.
    ///
    /// The detour through a table is deliberate. `query_arrow`'s iterator panics on a step
    /// error rather than yielding one, and this library is called across FFI, where a panic
    /// aborts the host process instead of returning a bad result. `CREATE TABLE ... AS`
    /// reports the same failure as a value, so the only thing the Arrow fetch reads is a
    /// plain scan. The `ORDER BY` is on the scan, not on the table, because a sort at
    /// materialization time would not order a later scan.
    fn materialize_and_scan(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        let connection = self.connection()?;
        connection.execute_batch(&format!("DROP TABLE IF EXISTS {RESULT_TABLE}"))?;
        connection.execute_batch(&format!("CREATE TEMPORARY TABLE {RESULT_TABLE} AS {sql}"))?;

        let scan = format!("SELECT * FROM {RESULT_TABLE} ORDER BY {ROW_COLUMN}");
        let mut statement = connection.prepare(&scan)?;
        let mut batches = Vec::new();
        for batch in statement.query_arrow([])? {
            batches.push(batch);
        }
        Ok(batches)
    }

    /// Stream the newest row per entity for a view, calling `sink` once per record batch.
    ///
    /// This is the materialization read. A full refresh of current values needs one row per
    /// entity and not the history, so the reduction happens inside DuckDB
    /// (`QUALIFY row_number() OVER (PARTITION BY ... ORDER BY ... DESC) = 1`) rather than in
    /// a caller that has already fetched everything. See "Latest-per-entity pushdown" in the
    /// architecture document.
    ///
    /// Each batch carries the view's feature columns in declaration order, then the entity
    /// join key, then the winning row's event timestamp as microseconds in an `Int64`. Every
    /// one of those columns is checked against the source's own types first, so a source
    /// schema change fails here with a named column rather than during the write.
    ///
    /// `sink` is awaited before the next batch is pulled, so a refresh holds one batch rather
    /// than a whole scan.
    ///
    /// Args:
    ///     view: The view whose source is scanned.
    ///     sink: Receives each batch, in the order DuckDB produces them.
    ///
    /// Returns:
    ///     `Ok(())` once the scan is exhausted and every batch has been accepted.
    ///
    /// Raises:
    ///     [`Error::MissingColumn`] if the source lacks a column the scan needs.
    ///     [`Error::UnsupportedOfflineType`] for a key or timestamp type the scan cannot
    ///         carry.
    ///     [`Error::ColumnTypeMismatch`] if a source column does not have the declared dtype.
    ///     [`Error::UnknownConnection`], [`Error::SourceConnectionKind`] or
    ///         [`Error::UnreadableSource`] exactly as in [`Engine::point_in_time_join`].
    ///     [`Error::StreamInterrupted`] if the scan failed after it had started, which is
    ///         where the connection is discarded and the next use reopens it.
    ///     [`Error::DuckDb`] if the query cannot be prepared.
    pub async fn scan_latest_per_entity<S: LatestBatchSink>(
        &self,
        view: &FeatureView,
        sink: &mut S,
    ) -> Result<()> {
        let sql = self.latest_per_entity_sql(view)?;
        let result = self.run_scan(&view.name, &sql, sink).await;
        // A caught panic leaves the connection's state unknown, so it is dropped and the
        // next use opens a fresh one. Every other failure arrived as a value from a call that
        // returned, which leaves the connection usable. This branch cannot be reached from a
        // test without a seam to force a mid-scan failure, which is #26's work rather than this
        // path's.
        if matches!(&result, Err(Error::StreamInterrupted { .. })) {
            self.discard_connection();
        }
        result
    }

    /// Run the reduced scan, handing each batch to `sink` before fetching the next.
    ///
    /// The connection is held for the whole scan, which is what the lint below notices. Holding
    /// it is safe rather than merely convenient: the lock is uncontended by construction, since
    /// every caller reaches the engine through the store's own mutex, and the one place that
    /// takes it exclusively, [`Engine::scan_latest_per_entity`], does so after this future has
    /// been dropped.
    #[allow(clippy::await_holding_lock)]
    async fn run_scan<S: LatestBatchSink>(
        &self,
        view: &str,
        sql: &str,
        sink: &mut S,
    ) -> Result<()> {
        let connection = self.connection()?;
        let mut statement = connection.prepare(sql)?;
        // `stream_arrow`, not `query_arrow`: the latter materializes the whole result on the
        // client before the first batch is yielded, which is what a refresh of a full
        // keyspace cannot do.
        let mut rows = statement.stream_arrow([])?;
        while let Some(batch) = next_batch(&mut rows, view)? {
            sink.accept(batch).await?;
        }
        Ok(())
    }

    /// The reduced scan for a view, after checking that its source can answer it.
    ///
    /// The type checks are the join's, for the same reason: a source schema change should
    /// fail with a named column rather than arrive as a column of nulls. The entity key is
    /// checked against the families [`crate::key::entity_key_component`] can encode, since
    /// that is what the write path converts it with.
    ///
    /// Raises:
    ///     [`Error::MissingColumn`], [`Error::UnsupportedOfflineType`] and
    ///         [`Error::ColumnTypeMismatch`] as described on
    ///         [`Engine::scan_latest_per_entity`].
    fn latest_per_entity_sql(&self, view: &FeatureView) -> Result<String> {
        let entity = view.entity()?;
        let key_column = entity.join_key.as_str();

        self.ensure_source_loaded(view)?;

        let location = format!("source `{}`", view.source.description());
        let described = self.describe_source(view)?;

        let source_key = described
            .get(key_column)
            .ok_or_else(|| Error::MissingColumn {
                location: location.clone(),
                column: key_column.to_owned(),
            })?;
        // The exact types whose Arrow counterparts the key encoder carries: DuckDB exports
        // INTEGER as Int32, BIGINT as Int64 and VARCHAR as Utf8, and those three are what
        // `entity_key_component` reads. `KeyFamily::of_duckdb` is wider on purpose, because the
        // join only ever compares keys, so a check in terms of the family would admit a
        // SMALLINT key and fail later in the sink instead of here.
        match normalize_type(source_key).as_str() {
            "INTEGER" | "BIGINT" | "VARCHAR" | "TEXT" | "STRING" => {}
            other => {
                return Err(Error::UnsupportedOfflineType {
                    role: "entity key".to_owned(),
                    dtype: other.to_owned(),
                });
            }
        }

        let source_ts_column = view.timestamp_field();
        let source_ts = described
            .get(source_ts_column)
            .ok_or_else(|| Error::MissingColumn {
                location: location.clone(),
                column: source_ts_column.to_owned(),
            })?;
        let source_ts_kind = TimestampKind::of_duckdb(source_ts)?;
        // The timestamp reaches the write path as an `Int64` of microseconds, so the types that
        // could not survive that are refused here. An integer column is cast rather than trusted:
        // DuckDB will export an `INTEGER` as Arrow `Int32`, which the sink cannot read, and the
        // catch-all for a family would let it through to fail there.
        let source_ts_expr = match normalize_type(source_ts).as_str() {
            "TIMESTAMP" | "TIMESTAMPTZ" | "TIMESTAMP WITH TIME ZONE" | "DATETIME" => {
                source_ts_kind.to_micros(&quote_ident(source_ts_column))
            }
            "TINYINT" | "SMALLINT" | "INTEGER" | "BIGINT" => {
                format!("CAST({} AS BIGINT)", quote_ident(source_ts_column))
            }
            other => {
                return Err(Error::UnsupportedOfflineType {
                    role: "timestamp".to_owned(),
                    dtype: other.to_owned(),
                });
            }
        };

        for field in &view.features {
            let actual = described
                .get(&field.name)
                .ok_or_else(|| Error::MissingColumn {
                    location: location.clone(),
                    column: field.name.clone(),
                })?;
            let expected = duckdb_type_of(field.dtype);
            if normalize_type(actual) != expected {
                return Err(Error::ColumnTypeMismatch {
                    name: field.name.clone(),
                    declared: expected.to_owned(),
                    actual: actual.clone(),
                });
            }
        }

        let relation = self.relation(view)?;

        // The same refusal the join makes, for the same reason: without a created timestamp to
        // break a tie, which of two rows sharing a key and an event timestamp wins would depend
        // on the query plan. A refresh has no label frame to restrict the check to, so it looks at
        // the whole source, and it looks before writing anything of this view's, so a source that
        // training refuses to read is not one serving quietly answers from. That is per view:
        // views are refreshed in declaration order, so the ones before this have already written
        // by the time the check runs.
        if view.created_timestamp_field.is_none() {
            self.reject_ambiguous_timestamps(
                view,
                &relation,
                &quote_ident(key_column),
                &source_ts_expr,
                AmbiguityScope::WholeSource,
            )?;
        }

        Ok(latest_per_entity_sql(
            view,
            &relation,
            &quote_ident(key_column),
            &source_ts_expr,
        ))
    }
}

/// A consumer of a materialization scan, one record batch at a time.
///
/// A trait rather than a closure because the sink is asynchronous: each batch is encoded and
/// written to the online store before the next one is pulled off the scan, and a closure that
/// returns a future borrowing its own captured state cannot be written as an `FnMut`.
#[allow(async_fn_in_trait)]
pub trait LatestBatchSink {
    /// Consume one batch of [`Engine::scan_latest_per_entity`]'s stream.
    ///
    /// Args:
    ///     batch: The view's feature columns, then the entity join key, then the winning
    ///         row's event timestamp in microseconds.
    ///
    /// Returns:
    ///     `Ok(())` once the batch is stored. An error stops the scan at this batch.
    async fn accept(&mut self, batch: RecordBatch) -> Result<()>;
}

/// The SQL that reduces a view's source to one row per entity.
///
/// The partition is the entity key alone, deliberately: materialization wants one row per
/// entity, so collapsing a key's whole history to its newest row is the result it is after.
/// The join must not copy this reduction, where the same collapse makes every label older than
/// the newest row match nothing. The tie-break is the join's: a `created_timestamp_field`, when
/// the view declares one, decides between rows sharing a key and an event timestamp.
fn latest_per_entity_sql(
    view: &FeatureView,
    relation: &str,
    key_expr: &str,
    source_ts_value: &str,
) -> String {
    let mut selected: Vec<String> = view
        .features
        .iter()
        .map(|field| quote_ident(&field.name))
        .collect();
    selected.push(format!("{key_expr} AS {SCAN_KEY_COLUMN}"));
    selected.push(format!("{source_ts_value} AS {SCAN_TS_COLUMN}"));

    let tie_break = match &view.created_timestamp_field {
        Some(created) => format!(", {} DESC", quote_ident(created)),
        None => String::new(),
    };

    // A row with a null entity key has no key to be written under, and one with a null event
    // timestamp has no freshness to record. Both are excluded here rather than streamed and
    // then dropped, so the decision stays where the scan already is.
    format!(
        "SELECT {}
         FROM {relation}
         WHERE {key_expr} IS NOT NULL AND {source_ts_value} IS NOT NULL
         QUALIFY row_number() OVER (PARTITION BY {key_expr} ORDER BY {source_ts_value} DESC{tie_break}) = 1",
        selected.join(", ")
    )
}

/// Pull the next batch of a streamed scan, turning the iterator's panic into a value.
///
/// `duckdb`'s `Arrow::next` panics when a chunk fetch or the Arrow conversion fails, rather
/// than yielding an error, and no lazy iterator in the version we depend on yields one. This
/// library is called across FFI, where a panic aborts the host process instead of returning a
/// bad result, so the panic is caught here.
///
/// The catch depends on unwinding being enabled. Nothing in this workspace sets
/// `panic = "abort"`; if a profile ever does, this degrades into the abort it exists to
/// prevent and has to be revisited. The default panic hook still prints the message to
/// stderr, because suppressing it would mean replacing a global hook.
fn next_batch(rows: &mut Arrow<'_>, view: &str) -> Result<Option<RecordBatch>> {
    catching_panics(|| rows.next()).map_err(|reason| Error::StreamInterrupted {
        view: view.to_owned(),
        reason,
    })
}

/// Call `fetch`, returning a panic's message instead of unwinding.
fn catching_panics<T>(fetch: impl FnOnce() -> T) -> std::result::Result<T, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(fetch)) {
        Ok(value) => Ok(value),
        Err(payload) => Err(panic_message(payload.as_ref())),
    }
}

/// A panic payload as text, which is a `String` for any `panic!` with a formatted message.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_owned()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "a panic with no message".to_owned()
    }
}

/// The join, with the TTL window applied after it.
fn join_sql(
    view: &FeatureView,
    options: &JoinOptions,
    relation: &str,
    key_expr: &str,
    source_ts_value: &str,
    label_kind: TimestampKind,
) -> String {
    let mut from_source = String::new();
    let mut from_matched = String::new();
    for field in &view.features {
        let name = quote_ident(&field.name);
        let _ = write!(from_source, ", {name} AS {name}");
        let _ = write!(from_matched, ", s.{name}");
    }

    // Deduplicated only when a created timestamp can break the tie, and only among the rows
    // that actually tie: the partition is the entity key and the timestamp, so the sort decides
    // which of several rows sharing one instant wins and nothing else. Partitioning by the key
    // alone, as the materialization reducer deliberately does, would collapse a key's whole
    // history to its newest row before the join, and every label older than that row would then
    // match nothing and come back null under the default policy. The document scopes this rule
    // to rows sharing "the same entity key and the same maximal timestamp": the partition
    // expresses one instant per group, and the maximality in that phrase is the match rule's.
    // With no created timestamp there is no column to break a tie with, and two rows sharing a
    // key and an instant were already refused by the ambiguity check, so the `None` arm has no
    // tie to break.
    let dedup = match &view.created_timestamp_field {
        Some(created) => format!(
            " QUALIFY row_number() OVER (PARTITION BY {key_expr}, {source_ts_value} ORDER BY {} DESC) = 1",
            quote_ident(created)
        ),
        None => String::new(),
    };

    // Microseconds on both sides, so the window is integer arithmetic and needs no date
    // functions, matching how the value codec stores timestamps. The label side is
    // normalized rather than taken raw: a label column that is a real timestamp would
    // otherwise subtract from the source's micros as `timestamp - bigint`, which DuckDB
    // refuses to bind.
    let ttl = view.ttl_days.map(|days| {
        format!(
            "(m.{LABEL_TS} - m.{MATCHED_TS}) <= {}",
            micros_of_days(days)
        )
    });

    // A row is fresh when the join found something and, if the view declares a TTL, when
    // that something is not older than it. A row exactly `ttl_days` old is still fresh.
    let fresh = match &ttl {
        Some(ttl) => format!("(m.{MATCHED_TS} IS NOT NULL AND {ttl})"),
        None => format!("m.{MATCHED_TS} IS NOT NULL"),
    };

    // Under `Null` every label row is kept, so an expired value has to come back null
    // rather than stale. Carrying an expired value into a training set is the failure this
    // rule exists to prevent, and it is silent.
    let mut output = String::new();
    for field in &view.features {
        let name = quote_ident(&field.name);
        match options.on_missing {
            OnMissing::Null => {
                let _ = write!(
                    output,
                    ", CASE WHEN {fresh} THEN m.{name} ELSE NULL END AS {name}"
                );
            }
            OnMissing::Drop => {
                let _ = write!(output, ", m.{name}");
            }
        }
    }

    let filter = match options.on_missing {
        OnMissing::Null => String::new(),
        OnMissing::Drop => format!(" WHERE {fresh}"),
    };

    let comparison = if options.strict { ">" } else { ">=" };
    let label_ts = label_kind.to_micros(&format!("l.{LABEL_TS}"));

    format!(
        "WITH source AS (
             SELECT {key_expr} AS {LABEL_KEY}, {source_ts_value} AS {LABEL_TS}{from_source}
             FROM {relation}{dedup}
         ),
         matched AS (
             SELECT l.feather_row AS feather_row,
                    {label_ts} AS {LABEL_TS},
                    s.{LABEL_TS} AS {MATCHED_TS}{from_matched}
             FROM {LABELS_TABLE} l
             ASOF LEFT JOIN source s
               ON l.{LABEL_KEY} = s.{LABEL_KEY} AND {label_ts} {comparison} s.{LABEL_TS}
         )
         SELECT m.feather_row{output} FROM matched m{filter}"
    )
}

/// How much of a source the ambiguity check looks at.
///
/// A view that declares no `created_timestamp_field` has no rule for two rows sharing a key and
/// an event timestamp, so which one wins would depend on the query plan. Both paths refuse that,
/// and this says how far each has to look to find it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AmbiguityScope {
    /// Only the keys an entity frame names, which bounds the check by the request.
    LabelKeys,
    /// Every key the source holds, which is what a refresh needs: it writes one row per entity
    /// for all of them.
    WholeSource,
}

/// A source's columns and their DuckDB type names.
#[derive(Debug, Default)]
struct TypeMap {
    types: Vec<(String, String)>,
}

impl TypeMap {
    fn insert(&mut self, name: String, dtype: String) {
        self.types.push((name, dtype));
    }

    fn get(&self, name: &str) -> Option<&String> {
        self.types
            .iter()
            .find(|(column, _)| column == name)
            .map(|(_, dtype)| dtype)
    }
}

/// The SQL family an entity key belongs to. Both sides must agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeyFamily {
    Integer,
    Text,
}

impl KeyFamily {
    fn of_arrow(dtype: &DataType) -> Result<Self> {
        match dtype {
            DataType::Int64 | DataType::Int32 => Ok(Self::Integer),
            DataType::Utf8 => Ok(Self::Text),
            other => Err(Error::UnsupportedOfflineType {
                role: "entity key".to_owned(),
                dtype: other.to_string(),
            }),
        }
    }

    fn of_duckdb(dtype: &str) -> Result<Self> {
        let normalized = normalize_type(dtype);
        if normalized.contains("INT") {
            return Ok(Self::Integer);
        }
        if matches!(normalized.as_str(), "VARCHAR" | "TEXT" | "STRING") {
            return Ok(Self::Text);
        }
        Err(Error::UnsupportedOfflineType {
            role: "entity key".to_owned(),
            dtype: dtype.to_owned(),
        })
    }
}

/// Where a timestamp's numbers come from, which decides the normalization and what may be
/// compared against what.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TimestampKind {
    /// An integer already counted in microseconds since the epoch, UTC by convention.
    IntegerMicros,
    /// A timestamp with no zone.
    Naive,
    /// A timestamp carrying its zone.
    Zoned,
}

impl TimestampKind {
    fn of_arrow(dtype: &DataType) -> Result<Self> {
        match dtype {
            DataType::Int64 => Ok(Self::IntegerMicros),
            DataType::Timestamp(TimeUnit::Microsecond, None) => Ok(Self::Naive),
            DataType::Timestamp(TimeUnit::Microsecond, Some(_)) => Ok(Self::Zoned),
            other => Err(Error::UnsupportedOfflineType {
                role: "timestamp".to_owned(),
                dtype: other.to_string(),
            }),
        }
    }

    fn of_duckdb(dtype: &str) -> Result<Self> {
        let normalized = normalize_type(dtype);
        match normalized.as_str() {
            "TIMESTAMP" | "DATETIME" => Ok(Self::Naive),
            "TIMESTAMP WITH TIME ZONE" | "TIMESTAMPTZ" => Ok(Self::Zoned),
            other if other.contains("INT") => Ok(Self::IntegerMicros),
            _ => Err(Error::UnsupportedOfflineType {
                role: "timestamp".to_owned(),
                dtype: dtype.to_owned(),
            }),
        }
    }

    /// Whether the two sides can be compared without casting one of them.
    ///
    /// A naive timestamp against a zoned one is refused: DuckDB would coerce, and an
    /// implicit coercion between those two is precisely how an off-by-hours bug enters a
    /// training set. Integer microseconds are unambiguous, so they compare with either.
    fn comparable_with(self, other: Self) -> bool {
        if self == Self::IntegerMicros || other == Self::IntegerMicros {
            return true;
        }
        self == other
    }

    /// A SQL expression yielding microseconds since the epoch.
    fn to_micros(self, column: &str) -> String {
        match self {
            // Already microseconds, so there is no conversion and no date arithmetic.
            Self::IntegerMicros => column.to_owned(),
            Self::Naive | Self::Zoned => format!("epoch_us({column})"),
        }
    }
}

impl Engine {
    /// The table expression a view's source becomes.
    ///
    /// The one place a source becomes a relation, so `DESCRIBE`, the join, and the
    /// ambiguity check all read the same expression rather than re-deriving it. For a
    /// file source the format, not the kind, selects the reader. A reader added later,
    /// a warehouse table or a table format, changes this and
    /// [`Engine::ensure_source_loaded`], which is where its extension is loaded.
    ///
    /// Args:
    ///     view: The view whose source is read.
    ///
    /// Returns:
    ///     A SQL expression yielding the source's rows, already quoted for DuckDB.
    ///
    /// Raises:
    ///     [`Error::UnknownSourceFormat`] if a file source names a format Feather has no
    ///         reader for. The same error the definitions loader raises, so a view read
    ///         without one still fails as itself.
    ///     [`Error::UnknownConnection`] or [`Error::SourceConnectionKind`] if a Postgres
    ///         source names a connection the project cannot use.
    fn relation(&self, view: &FeatureView) -> Result<String> {
        match &view.source {
            Source::File { path, .. } => {
                let (reader, _) = format_reader(view.source_format()?);
                Ok(format!("{reader}({})", quote_literal(path)))
            }
            Source::Postgres {
                connection,
                schema,
                table,
            } => {
                // The conninfo is itself a SQL literal, so it is escaped for libpq by
                // `postgres_conninfo` and then for DuckDB by `quote_literal`. Both
                // layers are required; see the comment on the libpq one.
                let conninfo = self.postgres_conninfo(view, connection)?;
                Ok(format!(
                    "postgres_scan({}, {}, {})",
                    quote_literal(&conninfo),
                    quote_literal(schema),
                    quote_literal(table),
                ))
            }
        }
    }
}

/// DuckDB's name for a declared dtype, which is what `DESCRIBE` reports.
fn duckdb_type_of(dtype: DType) -> &'static str {
    match dtype {
        DType::Int64 => "BIGINT",
        DType::Float64 => "DOUBLE",
        DType::Boolean => "BOOLEAN",
        DType::Utf8 => "VARCHAR",
        DType::TimestampMicros => "TIMESTAMP",
    }
}

/// DuckDB's name for a type an entity frame carries.
fn arrow_duckdb_type(dtype: &DataType) -> Result<&'static str> {
    match dtype {
        DataType::Int64 => Ok("BIGINT"),
        DataType::Int32 => Ok("INTEGER"),
        DataType::Utf8 => Ok("VARCHAR"),
        DataType::Timestamp(TimeUnit::Microsecond, None) => Ok("TIMESTAMP"),
        DataType::Timestamp(TimeUnit::Microsecond, Some(_)) => Ok("TIMESTAMPTZ"),
        other => Err(Error::UnsupportedOfflineType {
            role: "entity frame".to_owned(),
            dtype: other.to_string(),
        }),
    }
}

/// Upper-case a DuckDB type name and drop any width, so `VARCHAR(64)` compares as `VARCHAR`.
fn normalize_type(dtype: &str) -> String {
    dtype
        .split('(')
        .next()
        .unwrap_or(dtype)
        .trim()
        .to_ascii_uppercase()
}

/// Microseconds in a whole number of days.
fn micros_of_days(days: u32) -> i64 {
    i64::from(days) * 86_400_000_000
}

/// The index of a column, or an error naming what was looking for it.
fn column_index(schema: &Schema, name: &str) -> Result<usize> {
    schema.index_of(name).map_err(|_| Error::MissingColumn {
        location: "the entity frame".to_owned(),
        column: name.to_owned(),
    })
}

/// The entity frame reduced to what the join needs, plus its position.
fn label_batch(
    batch: &RecordBatch,
    schema: &Schema,
    key_index: usize,
    label_index: usize,
) -> Result<RecordBatch> {
    let rows = batch.num_rows();
    let fields = vec![
        Field::new(ROW_COLUMN, DataType::Int64, false),
        Field::new(LABEL_KEY, schema.field(key_index).data_type().clone(), true),
        Field::new(
            LABEL_TS,
            schema.field(label_index).data_type().clone(),
            true,
        ),
    ];
    let row_index: ArrayRef = Arc::new(Int64Array::from_iter_values(0..rows as i64));
    let columns = vec![
        row_index,
        batch.column(key_index).clone(),
        batch.column(label_index).clone(),
    ];
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

/// What the caller gets back: the row index, then one nullable column per feature.
fn output_fields(view: &FeatureView) -> Vec<Field> {
    let mut fields = vec![Field::new(ROW_COLUMN, DataType::Int64, false)];
    for field in &view.features {
        fields.push(Field::new(&field.name, arrow_type(field.dtype), true));
    }
    fields
}

/// The `CREATE SECRET` statement for a configured object-store connection, if it is one
/// DuckDB needs telling about.
///
/// The key id is not a secret and is written as is; the secret itself came from the
/// environment, since the config loader turns an unset variable into a load error.
fn s3_secret_sql(name: &str, configured: &SettingsConnection) -> Option<String> {
    match configured {
        SettingsConnection::S3 {
            region,
            key_id,
            secret,
            endpoint,
            use_ssl,
            url_style,
        } => {
            let mut options = vec![
                "TYPE s3".to_owned(),
                format!("KEY_ID {}", quote_literal(key_id)),
                format!("SECRET {}", quote_literal(secret.expose())),
                format!("REGION {}", quote_literal(region)),
            ];
            if let Some(endpoint) = endpoint {
                options.push(format!("ENDPOINT {}", quote_literal(endpoint)));
                options.push(format!("USE_SSL {use_ssl}"));
                // DuckDB's own default for S3 is vhost, and an S3-compatible server reached at
                // a host and port is addressed path style, so an endpoint implies path unless
                // the connection overrides it.
                options.push(format!(
                    "URL_STYLE {}",
                    quote_literal(url_style.as_deref().unwrap_or("path"))
                ));
            } else if let Some(style) = url_style {
                options.push(format!("URL_STYLE {}", quote_literal(style)));
            }
            Some(format!(
                "CREATE OR REPLACE SECRET {} ({})",
                quote_ident(name),
                options.join(", ")
            ))
        }
        // DuckDB's snowflake extension authenticates its own way, and no reader uses this kind
        // yet.
        SettingsConnection::Snowflake { .. } => None,
        // A Postgres source reads through the connection string the relation carries, so
        // there is no secret for the engine to hold: the scanner takes libpq's own form.
        SettingsConnection::Postgres { .. } => None,
    }
}

/// Whether a source path goes through a filesystem DuckDB loads rather than one it has
/// compiled in.
///
/// A URI scheme means a loadable filesystem, and `httpfs` covers S3, GCS, Azure Blob and
/// plain HTTP. A bare path is the local filesystem, which needs nothing loaded.
fn path_needs_filesystem(path: &str) -> bool {
    path.contains("://")
}

/// The extensions a file source needs installed, with the format each one reads.
///
/// An `Option<(&'static str, &'static str)>` alongside each extension is the `(name,
/// tier)` pair [`Error::extension_unavailable`] names, and it is `None` for anything
/// that is not a format's own reader: `httpfs` reads no format and is no tier, so a
/// remote Parquet source must not have one attributed to its filesystem.
///
/// A fixed-size array rather than a `Vec` grown by `push`, so the two cases are
/// visible side by side and there is no lint question about pushing conditionally
/// collected values. The order is the filesystem first, because a remote path cannot
/// be read at all without it, and the format's own extension second.
fn extensions_for(
    path: &str,
    format: FileFormat,
) -> [Option<(&'static str, Option<(&'static str, &'static str)>)>; 2] {
    let reader = format_reader(format).1;
    [
        (path_needs_filesystem(path).then_some("httpfs"), None),
        (
            reader,
            reader.map(|_| (format.as_str(), format.tier().as_str())),
        ),
    ]
}

/// How a file format is read: the table function that yields its rows, and the DuckDB
/// extension that has to be loaded first, or `None` when the bundled build already
/// contains it.
///
/// Both halves are here rather than on [`FileFormat`] because they are DuckDB's names,
/// and `definitions.rs` is the language-neutral contract that no engine appears in.
fn format_reader(format: FileFormat) -> (&'static str, Option<&'static str>) {
    match format {
        // The bundled build compiles the Parquet reader in, so installing it would be a
        // no-op that can still fail on a machine that cannot reach the repository.
        FileFormat::Parquet => ("read_parquet", None),
        // Loadable, third-party maintained, and pre-1.0, so a source naming it opts in
        // and nothing in Feather is compiled against it.
        FileFormat::Vortex => ("read_vortex", Some("vortex")),
    }
}

/// Quote an identifier, doubling any embedded quote.
fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// Quote a string literal, doubling any embedded quote.
fn quote_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use arrow::array::{Array, Float64Array, TimestampMicrosecondArray};
    use parquet::arrow::ArrowWriter;

    use super::*;
    use crate::definitions::{Entity, Field as FeatureField, SupportTier};

    /// A whole day, in the units the join compares in.
    const DAY: i64 = 86_400_000_000;

    static FIXTURE: AtomicU64 = AtomicU64::new(0);

    /// A Parquet file that removes itself when the test ends.
    struct Parquet {
        path: PathBuf,
    }

    impl Parquet {
        fn write(batch: &RecordBatch) -> Self {
            let path = std::env::temp_dir().join(format!(
                "feather-offline-{}-{}.parquet",
                std::process::id(),
                FIXTURE.fetch_add(1, Ordering::Relaxed)
            ));
            let file = std::fs::File::create(&path).expect("create fixture");
            let mut writer =
                ArrowWriter::try_new(file, batch.schema(), None).expect("parquet writer");
            writer.write(batch).expect("write fixture");
            writer.close().expect("close fixture");
            Self { path }
        }

        fn string(&self) -> String {
            self.path.display().to_string()
        }
    }

    impl Drop for Parquet {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    /// `user_id, event_timestamp, count` as integers.
    fn integer_source(rows: &[(i64, i64, i64)]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("user_id", DataType::Int64, false),
            Field::new("event_timestamp", DataType::Int64, false),
            Field::new("count", DataType::Int64, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.0))),
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.1))),
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.2))),
            ],
        )
        .expect("source batch")
    }

    /// `user_id, event_timestamp, created_at, count`, for tie breaking.
    fn tie_breakable_source(rows: &[(i64, i64, i64, i64)]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("user_id", DataType::Int64, false),
            Field::new("event_timestamp", DataType::Int64, false),
            Field::new("created_at", DataType::Int64, false),
            Field::new("count", DataType::Int64, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.0))),
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.1))),
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.2))),
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.3))),
            ],
        )
        .expect("source batch")
    }

    /// Label rows, with a nullable key so the null-key rule can be exercised.
    fn labels(rows: &[(Option<i64>, i64)]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("user_id", DataType::Int64, true),
            Field::new("event_timestamp", DataType::Int64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from_iter(rows.iter().map(|r| r.0))),
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.1))),
            ],
        )
        .expect("labels batch")
    }

    /// `user_id, event_timestamp, count`, with the key and the timestamp nullable, so a source
    /// that carries the nulls the scan drops can be built.
    fn nullable_source(rows: &[(Option<i64>, Option<i64>, i64)]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("user_id", DataType::Int64, true),
            Field::new("event_timestamp", DataType::Int64, true),
            Field::new("count", DataType::Int64, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from_iter(rows.iter().map(|r| r.0))),
                Arc::new(Int64Array::from_iter(rows.iter().map(|r| r.1))),
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.2))),
            ],
        )
        .expect("source batch")
    }

    fn view(path: &str, ttl_days: Option<u32>) -> FeatureView {
        FeatureView {
            name: "user_clicks".to_owned(),
            entities: vec![Entity::new("user_id", "user_id")],
            source: Source::file(path),
            features: vec![FeatureField::new("count", DType::Int64)],
            ttl_days,
            timestamp_field: None,
            created_timestamp_field: None,
        }
    }

    /// Open an engine with no configured connections, which is what a local project has.
    fn open(limits: &Limits) -> Result<Engine> {
        Engine::open(limits, &BTreeMap::new())
    }

    fn engine() -> Engine {
        open(&Limits::default()).expect("engine")
    }

    /// The `count` column of a result, as nullable values.
    fn counts(batch: &RecordBatch) -> Vec<Option<i64>> {
        let column = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("count column");
        (0..column.len())
            .map(|i| {
                if column.is_null(i) {
                    None
                } else {
                    Some(column.value(i))
                }
            })
            .collect()
    }

    /// The row index column, which is what output alignment is asserted on.
    fn rows(batch: &RecordBatch) -> Vec<i64> {
        let column = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("row column");
        (0..column.len()).map(|i| column.value(i)).collect()
    }

    #[test]
    fn matches_the_newest_row_at_or_before_the_label_time() {
        let source = Parquet::write(&integer_source(&[(1, 100, 10), (1, 200, 20), (1, 300, 30)]));
        let joined = engine()
            .point_in_time_join(
                &labels(&[(Some(1), 250)]),
                &view(&source.string(), None),
                &JoinOptions::default(),
            )
            .expect("join");

        assert_eq!(counts(&joined), [Some(20)]);
    }

    #[test]
    fn a_row_at_exactly_the_label_time_is_included() {
        let source = Parquet::write(&integer_source(&[(1, 100, 10), (1, 200, 20)]));
        let joined = engine()
            .point_in_time_join(
                &labels(&[(Some(1), 200)]),
                &view(&source.string(), None),
                &JoinOptions::default(),
            )
            .expect("join");

        assert_eq!(counts(&joined), [Some(20)]);
    }

    #[test]
    fn strict_excludes_a_row_at_exactly_the_label_time() {
        let source = Parquet::write(&integer_source(&[(1, 100, 10), (1, 200, 20)]));
        let options = JoinOptions {
            strict: true,
            ..JoinOptions::default()
        };
        let joined = engine()
            .point_in_time_join(
                &labels(&[(Some(1), 200)]),
                &view(&source.string(), None),
                &options,
            )
            .expect("join");

        assert_eq!(counts(&joined), [Some(10)]);
    }

    #[test]
    fn a_value_exactly_ttl_days_old_is_still_fresh() {
        let source = Parquet::write(&integer_source(&[(1, 0, 10)]));
        let joined = engine()
            .point_in_time_join(
                &labels(&[(Some(1), DAY)]),
                &view(&source.string(), Some(1)),
                &JoinOptions::default(),
            )
            .expect("join");

        assert_eq!(counts(&joined), [Some(10)]);
    }

    #[test]
    fn an_expired_value_is_null_by_default_and_dropped_on_request() {
        let source = Parquet::write(&integer_source(&[(1, 0, 10)]));
        let stale = labels(&[(Some(1), DAY + 1)]);

        let kept = engine()
            .point_in_time_join(
                &stale,
                &view(&source.string(), Some(1)),
                &JoinOptions::default(),
            )
            .expect("join");
        assert_eq!(counts(&kept), [None]);
        assert_eq!(rows(&kept), [0]);

        let dropped = engine()
            .point_in_time_join(
                &stale,
                &view(&source.string(), Some(1)),
                &JoinOptions {
                    on_missing: OnMissing::Drop,
                    ..JoinOptions::default()
                },
            )
            .expect("join");
        assert_eq!(dropped.num_rows(), 0);
    }

    #[test]
    fn a_label_with_no_match_keeps_its_place() {
        let source = Parquet::write(&integer_source(&[(1, 100, 10)]));
        let joined = engine()
            .point_in_time_join(
                &labels(&[(Some(2), 100)]),
                &view(&source.string(), None),
                &JoinOptions::default(),
            )
            .expect("join");

        assert_eq!(rows(&joined), [0]);
        assert_eq!(counts(&joined), [None]);
    }

    #[test]
    fn a_null_entity_key_never_matches() {
        let source = Parquet::write(&integer_source(&[(1, 100, 10)]));
        let joined = engine()
            .point_in_time_join(
                &labels(&[(None, 100)]),
                &view(&source.string(), None),
                &JoinOptions::default(),
            )
            .expect("join");

        assert_eq!(counts(&joined), [None]);
    }

    #[test]
    fn output_rows_follow_the_input_order_not_the_join_order() {
        let source = Parquet::write(&integer_source(&[(1, 100, 10), (1, 200, 20), (1, 300, 30)]));
        let joined = engine()
            .point_in_time_join(
                &labels(&[(Some(1), 300), (Some(1), 100), (Some(1), 200)]),
                &view(&source.string(), None),
                &JoinOptions::default(),
            )
            .expect("join");

        assert_eq!(rows(&joined), [0, 1, 2]);
        assert_eq!(counts(&joined), [Some(30), Some(10), Some(20)]);
    }

    #[test]
    fn duplicate_timestamps_without_a_created_column_are_rejected() {
        let source = Parquet::write(&integer_source(&[(1, 100, 10), (1, 100, 20)]));
        let error = engine()
            .point_in_time_join(
                &labels(&[(Some(1), 100)]),
                &view(&source.string(), None),
                &JoinOptions::default(),
            )
            .expect_err("must fail");

        assert!(matches!(error, Error::AmbiguousTimestamp { .. }), "{error}");
    }

    #[test]
    fn repeated_nulls_are_not_an_ambiguity_the_join_refuses() {
        // The check groups a key and a timestamp, and reading a null back out of either is a
        // column-type error rather than a value, so a source the join reads perfectly well used
        // to fail the whole join. The rows it has to ignore are the ones the scan drops: here two
        // rows with a null key at one timestamp, and two with a null event timestamp under entity
        // 4, which is the pair that reaches this path — the check here is restricted to the keys
        // the label frame names, so it is the labels that have to name the entity whose rows have
        // no timestamp.
        let source = Parquet::write(&nullable_source(&[
            (None, Some(10), 1),
            (None, Some(10), 2),
            (Some(4), None, 3),
            (Some(4), None, 4),
            (Some(3), Some(20), 5),
        ]));

        let joined = engine()
            .point_in_time_join(
                &labels(&[(Some(4), 20), (Some(3), 20)]),
                &view(&source.string(), None),
                &JoinOptions::default(),
            )
            .expect("a source with repeated nulls is not ambiguous");

        // Entity 4's only rows have no timestamp, so nothing matches it; entity 3's row is the
        // value the join is for.
        assert_eq!(rows(&joined), [0, 1]);
        assert_eq!(counts(&joined), [None, Some(5)]);
    }

    #[test]
    fn a_duplicate_timestamp_takes_the_latest_created_row() {
        let source = Parquet::write(&tie_breakable_source(&[(1, 100, 5, 10), (1, 100, 9, 20)]));
        let mut view = view(&source.string(), None);
        view.created_timestamp_field = Some("created_at".to_owned());

        let joined = engine()
            .point_in_time_join(&labels(&[(Some(1), 100)]), &view, &JoinOptions::default())
            .expect("join");

        assert_eq!(counts(&joined), [Some(20)]);
    }

    #[test]
    fn a_created_column_breaks_ties_without_collapsing_a_keys_history() {
        // Two rows share an instant under one key and a third sits later; a second key has a row
        // at that same instant. The tie-break must choose between the first two and leave the
        // earlier instant in place for the join to find, and the keys must stay apart: a
        // partition without the key reduces both keys' `ts = 100` rows to one and takes the
        // second key's label with it. The tied rows cross `created_at` with `count`, so ordering
        // on `count` instead picks the other row and changes the answer.
        let source = Parquet::write(&tie_breakable_source(&[
            (1, 100, 5, 20),
            (1, 100, 9, 10),
            (1, 200, 7, 30),
            (2, 100, 3, 40),
        ]));
        let mut view = view(&source.string(), None);
        view.created_timestamp_field = Some("created_at".to_owned());

        let joined = engine()
            .point_in_time_join(
                &labels(&[
                    (Some(1), 100),
                    (Some(1), 150),
                    (Some(1), 200),
                    (Some(2), 100),
                ]),
                &view,
                &JoinOptions::default(),
            )
            .expect("join");

        // The tied instant (1, 100) is won by `created_at` 9, whose `count` is 10, so both
        // labels at or after it take 10: the label at 150 has only that instant to match.
        assert_eq!(
            counts(&joined),
            [Some(10), Some(10), Some(30), Some(40)],
            "at 100 the greater created_at wins the tie, so the value is 10 and not 20; at \
             150 the older instant still matches, tie-broken the same way; at 200 the later \
             instant's 30 wins; and (2, 100) shows the second key keeping its own row"
        );
    }

    #[test]
    fn a_created_column_keeps_every_label_when_missing_rows_are_dropped() {
        // The same fixture under the other policy, because both read the same reduced source
        // and the pre-fix clause loses rows here as well: the labels at 100 and 150 would find
        // nothing to match and be removed, leaving two rows for four labels.
        let source = Parquet::write(&tie_breakable_source(&[
            (1, 100, 5, 20),
            (1, 100, 9, 10),
            (1, 200, 7, 30),
            (2, 100, 3, 40),
        ]));
        let mut view = view(&source.string(), None);
        view.created_timestamp_field = Some("created_at".to_owned());

        let joined = engine()
            .point_in_time_join(
                &labels(&[
                    (Some(1), 100),
                    (Some(1), 150),
                    (Some(1), 200),
                    (Some(2), 100),
                ]),
                &view,
                &JoinOptions {
                    on_missing: OnMissing::Drop,
                    ..JoinOptions::default()
                },
            )
            .expect("join");

        assert_eq!(
            rows(&joined),
            [0, 1, 2, 3],
            "one row per label, in label order"
        );
        assert_eq!(counts(&joined), [Some(10), Some(10), Some(30), Some(40)]);
    }

    #[test]
    fn integer_microseconds_compare_with_a_timestamp_source() {
        // The label frame carries micros and the source carries a naive timestamp. Both
        // normalize to micros, so this is allowed rather than an off-by-hours risk.
        let schema = Arc::new(Schema::new(vec![
            Field::new("user_id", DataType::Int64, false),
            Field::new(
                "event_timestamp",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("count", DataType::Int64, true),
        ]));
        // Already microseconds since the epoch, which is what both sides compare in.
        let stamp = 1_700_000_000_000_000i64;
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(
                    TimestampMicrosecondArray::from(vec![stamp])
                        .with_data_type(DataType::Timestamp(TimeUnit::Microsecond, None)),
                ),
                Arc::new(Int64Array::from(vec![42])),
            ],
        )
        .expect("source batch");
        let source = Parquet::write(&batch);

        let joined = engine()
            .point_in_time_join(
                &labels(&[(Some(1), stamp)]),
                &view(&source.string(), None),
                &JoinOptions::default(),
            )
            .expect("join");

        assert_eq!(counts(&joined), [Some(42)]);
    }

    #[test]
    fn a_timestamp_label_measures_its_ttl_window() {
        // The label frame carries a real timestamp rather than integer micros, which is
        // what a warehouse source produces. Both sides of the TTL subtraction have to be
        // normalized to micros, or DuckDB refuses to bind `timestamp - bigint`.
        let schema = Arc::new(Schema::new(vec![
            Field::new("user_id", DataType::Int64, false),
            Field::new(
                "event_timestamp",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("count", DataType::Int64, true),
        ]));
        let stamp = 1_700_000_000_000_000i64;
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(
                    TimestampMicrosecondArray::from(vec![stamp])
                        .with_data_type(DataType::Timestamp(TimeUnit::Microsecond, None)),
                ),
                Arc::new(Int64Array::from(vec![42])),
            ],
        )
        .expect("source batch");
        let source = Parquet::write(&batch);

        let label_schema = Arc::new(Schema::new(vec![
            Field::new("user_id", DataType::Int64, true),
            Field::new(
                "event_timestamp",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
        ]));
        let label_frame = RecordBatch::try_new(
            label_schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 1])),
                Arc::new(
                    TimestampMicrosecondArray::from(vec![stamp + DAY, stamp + 2 * DAY])
                        .with_data_type(DataType::Timestamp(TimeUnit::Microsecond, None)),
                ),
            ],
        )
        .expect("labels batch");

        let joined = engine()
            .point_in_time_join(
                &label_frame,
                &view(&source.string(), Some(1)),
                &JoinOptions::default(),
            )
            .expect("join");

        // Exactly one day old is still fresh; two days is not.
        assert_eq!(counts(&joined), [Some(42), None]);
        assert_eq!(rows(&joined), [0, 1]);
    }

    #[test]
    fn a_naive_timestamp_against_a_zoned_one_is_rejected() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("user_id", DataType::Int64, false),
            Field::new(
                "event_timestamp",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("count", DataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(
                    TimestampMicrosecondArray::from(vec![1_700_000_000_000_000])
                        .with_timezone("UTC"),
                ),
                Arc::new(Int64Array::from(vec![42])),
            ],
        )
        .expect("source batch");
        let source = Parquet::write(&batch);

        // The label frame's timestamp is naive, the source's is zoned.
        let label_schema = Arc::new(Schema::new(vec![
            Field::new("user_id", DataType::Int64, true),
            Field::new(
                "event_timestamp",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
        ]));
        let label_frame = RecordBatch::try_new(
            label_schema,
            vec![
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(TimestampMicrosecondArray::from(vec![1_700_000_000_000_000])),
            ],
        )
        .expect("labels batch");

        let error = engine()
            .point_in_time_join(
                &label_frame,
                &view(&source.string(), None),
                &JoinOptions::default(),
            )
            .expect_err("must fail");

        assert!(
            matches!(error, Error::OfflineTypeMismatch { .. }),
            "{error}"
        );
    }

    #[test]
    fn a_source_column_of_the_wrong_type_is_rejected() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("user_id", DataType::Int64, false),
            Field::new("event_timestamp", DataType::Int64, false),
            Field::new("count", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(Int64Array::from(vec![100])),
                Arc::new(Float64Array::from(vec![1.5])),
            ],
        )
        .expect("source batch");
        let source = Parquet::write(&batch);

        let error = engine()
            .point_in_time_join(
                &labels(&[(Some(1), 100)]),
                &view(&source.string(), None),
                &JoinOptions::default(),
            )
            .expect_err("must fail");

        assert!(matches!(error, Error::ColumnTypeMismatch { .. }), "{error}");
    }

    #[test]
    fn a_missing_source_column_is_named() {
        let source = Parquet::write(&integer_source(&[(1, 100, 10)]));
        let mut broken = view(&source.string(), None);
        broken.features = vec![FeatureField::new("nope", DType::Int64)];

        let error = engine()
            .point_in_time_join(&labels(&[(Some(1), 100)]), &broken, &JoinOptions::default())
            .expect_err("must fail");

        assert!(error.to_string().contains("nope"), "{error}");
    }

    #[test]
    fn a_read_failure_names_the_view_and_its_source() {
        // A source that does not exist, so the failure is the scanner's own. What is
        // asserted is the attribution wrapped around the scanner's text, which is why the
        // library's own wording is not asserted on.
        let error = engine()
            .point_in_time_join(
                &labels(&[(Some(1), 100)]),
                &view("no/such/source.parquet", None),
                &JoinOptions::default(),
            )
            .expect_err("must fail");

        assert!(matches!(error, Error::UnreadableSource { .. }), "{error}");
        assert!(error.to_string().contains("view `user_clicks`"), "{error}");
        assert!(
            error
                .to_string()
                .contains("source `no/such/source.parquet`"),
            "{error}"
        );
    }

    #[test]
    fn a_missing_label_column_is_named() {
        let source = Parquet::write(&integer_source(&[(1, 100, 10)]));
        let frame = labels(&[(Some(1), 100)]);
        let options = JoinOptions {
            label_timestamp_column: "absent".to_owned(),
            ..JoinOptions::default()
        };

        let error = engine()
            .point_in_time_join(&frame, &view(&source.string(), None), &options)
            .expect_err("must fail");

        assert!(error.to_string().contains("absent"), "{error}");
        assert!(error.to_string().contains("entity frame"), "{error}");
    }

    #[test]
    fn a_label_column_name_is_quoted_into_the_query() {
        // A name that would break the SQL if it were interpolated bare.
        let source = Parquet::write(&integer_source(&[(1, 100, 10)]));
        let named = Schema::new(vec![
            Field::new("user_id", DataType::Int64, true),
            Field::new("odd \" name", DataType::Int64, false),
        ]);
        let odd = RecordBatch::try_new(
            Arc::new(named),
            vec![
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(Int64Array::from(vec![100])),
            ],
        )
        .expect("labels batch");
        let options = JoinOptions {
            label_timestamp_column: "odd \" name".to_owned(),
            ..JoinOptions::default()
        };

        let joined = engine()
            .point_in_time_join(&odd, &view(&source.string(), None), &options)
            .expect("join");

        assert_eq!(counts(&joined), [Some(10)]);
    }

    #[test]
    fn only_a_uri_path_needs_a_loaded_filesystem() {
        assert!(path_needs_filesystem("s3://bucket/data.parquet"));
        assert!(path_needs_filesystem("https://example.com/data.parquet"));
        assert!(!path_needs_filesystem("data/stats.parquet"));
        assert!(!path_needs_filesystem("/abs/path/stats.parquet"));
    }

    /// A configured connection, built through the same wire shape the config loader produces.
    fn connection(json: &str) -> SettingsConnection {
        serde_json::from_str(json).expect("connection")
    }

    #[test]
    fn an_object_store_connection_becomes_a_secret() {
        let configured = connection(
            r#"{"type":"s3","region":"us-east-1","key_id":"AKIAEXAMPLE","secret":"shhh"}"#,
        );
        let sql = s3_secret_sql("s3_lake", &configured).expect("a secret");

        assert!(sql.contains("CREATE OR REPLACE SECRET"), "{sql}");
        assert!(sql.contains("KEY_ID 'AKIAEXAMPLE'"), "{sql}");
        assert!(sql.contains("REGION 'us-east-1'"), "{sql}");
        // Nothing about the endpoint, so this is AWS and DuckDB's own defaults apply.
        assert!(!sql.contains("ENDPOINT"), "{sql}");
        assert!(!sql.contains("URL_STYLE"), "{sql}");
    }

    #[test]
    fn a_compatible_endpoint_gets_path_style_and_no_tls() {
        // What a local S3-compatible server needs: DuckDB defaults to vhost addressing and
        // HTTPS, neither of which a host-and-port endpoint answers.
        let configured = connection(
            r#"{"type":"s3","region":"us-east-1","key_id":"feathertest","secret":"feathertest","endpoint":"localhost:9000","use_ssl":false}"#,
        );
        let sql = s3_secret_sql("s3_local", &configured).expect("a secret");

        assert!(sql.contains("ENDPOINT 'localhost:9000'"), "{sql}");
        assert!(sql.contains("USE_SSL false"), "{sql}");
        assert!(sql.contains("URL_STYLE 'path'"), "{sql}");
    }

    #[test]
    fn a_snowflake_connection_becomes_no_secret() {
        // Nothing reads that kind yet, and DuckDB's snowflake extension authenticates its own
        // way, so inventing a secret for it would be a guess.
        let configured = connection(
            r#"{"type":"snowflake","account":"a","warehouse":"w","username":"u","password":"p"}"#,
        );

        assert!(s3_secret_sql("snowflake_prod", &configured).is_none());
    }

    /// The same view shape as [`view`], over the given source.
    fn source_view(source: Source) -> FeatureView {
        let mut view = view("unused", None);
        view.source = source;
        view
    }

    #[test]
    fn a_file_source_with_no_format_declared_is_read_as_parquet() {
        // The default, so the pinned string is unchanged by the format knob: a project
        // that names no format must go on getting `read_parquet` and nothing else.
        let relation = engine()
            .relation(&view("data/user_stats.parquet", None))
            .expect("relation");

        assert_eq!(relation, "read_parquet('data/user_stats.parquet')");
    }

    #[test]
    fn a_file_source_naming_a_format_is_read_with_that_formats_reader() {
        let relation = engine()
            .relation(&source_view(Source::File {
                path: "data/clicks.vortex".to_owned(),
                format: Some("vortex".to_owned()),
            }))
            .expect("relation");

        assert_eq!(relation, "read_vortex('data/clicks.vortex')");
    }

    #[test]
    fn an_unknown_format_is_rejected_rather_than_read_as_parquet() {
        // The failure mode this rules out is a typo silently falling back to the default
        // and reading a Vortex file as Parquet, which is a decode error much later and
        // nowhere near the name that caused it.
        let view = source_view(Source::File {
            path: "data/clicks.vortex".to_owned(),
            format: Some("vortx".to_owned()),
        });

        let error = engine().relation(&view).expect_err("must fail");

        assert_eq!(
            error.to_string(),
            "view `user_clicks` reads source `data/clicks.vortex` in format `vortx`, which \
             Feather does not read; the formats are parquet, vortex"
        );
    }

    #[test]
    fn two_formats_resolve_from_one_project() {
        // The seam the format knob exists for: one engine, one set of connections, and
        // two sources whose formats differ. Nothing here reads a file, so the assertion
        // is about resolution rather than about either format's data.
        let connections = BTreeMap::from([(
            "s3_lake".to_owned(),
            connection(r#"{"type":"s3","region":"us-east-1","key_id":"k","secret":"s"}"#),
        )]);
        let settings = crate::settings::parse_settings(
            r#"
            project = "ads"
            definitions = ["user_clicks"]

            [connections.s3_lake]
            type = "s3"
            region = "us-east-1"
            key_id = "k"
            secret = "s"
            "#,
        )
        .expect("settings");
        assert_eq!(
            settings.connections, connections,
            "one settings, one connection"
        );
        let engine = Engine::open(&Limits::default(), &settings.connections).expect("engine");

        let parquet = engine
            .relation(&view("data/user_stats.parquet", None))
            .expect("parquet relation");
        let vortex = engine
            .relation(&source_view(Source::File {
                path: "s3://lake/clicks.vortex".to_owned(),
                format: Some("vortex".to_owned()),
            }))
            .expect("vortex relation");

        assert_eq!(parquet, "read_parquet('data/user_stats.parquet')");
        assert_eq!(vortex, "read_vortex('s3://lake/clicks.vortex')");
    }

    #[test]
    fn a_format_naming_no_compiled_reader_still_names_the_extension_it_needs_installed() {
        // The install loop is the only place a format turns into a fetch, so it is
        // asserted where it happens rather than through the reader mapping alone. A
        // vortex source must reach `INSTALL vortex`; a local Parquet source must reach
        // nothing, or a project that names no non-default format would touch the
        // extension repository.
        assert_eq!(
            extensions_for("data/clicks.vortex", FileFormat::Vortex),
            [
                (None, None),
                (Some("vortex"), Some(("vortex", "third-party maintained"))),
            ],
            "a vortex source installs the vortex extension, and names its tier"
        );
        assert_eq!(
            extensions_for("data/user_stats.parquet", FileFormat::Parquet),
            [(None, None), (None, None)],
            "a local parquet source installs nothing"
        );
    }

    #[test]
    fn a_filesystem_extension_is_never_credited_with_a_format_or_a_tier() {
        // `httpfs` reads no format and is no tier. A remote Parquet source needs it and
        // does not need the Parquet reader, so crediting it with a format and a Primary
        // tier would tell an operator the opposite of what is true: that the failure they
        // are reading about is a third-party path.
        assert_eq!(
            extensions_for("s3://lake/user_stats.parquet", FileFormat::Parquet),
            [(Some("httpfs"), None), (None, None)],
            "httpfs was attributed a format"
        );
        assert_eq!(
            extensions_for("s3://lake/clicks.vortex", FileFormat::Vortex),
            [
                (Some("httpfs"), None),
                (Some("vortex"), Some(("vortex", "third-party maintained"))),
            ],
            "a remote vortex source names the format for its reader and not for httpfs"
        );
    }

    #[test]
    fn each_format_names_the_extension_it_needs_loaded() {
        // Parquet is compiled into the bundled build, so naming no extension is what
        // keeps a local project from touching the extension repository at all. Vortex is
        // loadable and third-party maintained, which is why a source naming it opts in.
        assert_eq!(format_reader(FileFormat::Parquet), ("read_parquet", None));
        assert_eq!(
            format_reader(FileFormat::Vortex),
            ("read_vortex", Some("vortex"))
        );
        assert_eq!(FileFormat::Parquet.tier(), SupportTier::Primary);
        assert_eq!(FileFormat::Vortex.tier(), SupportTier::ThirdParty);
    }

    #[test]
    fn a_postgres_source_becomes_a_postgres_scan_relation() {
        let connections = BTreeMap::from([(
            "pg_prod".to_owned(),
            connection(
                r#"{"type":"postgres","host":"db.internal","port":5433,"database":"warehouse","user":"fs_runner","password":"fea'ther"}"#,
            ),
        )]);
        let engine = Engine::open(&Limits::default(), &connections).expect("engine");
        let view = source_view(Source::postgres("pg_prod", "public", "user_stats"));

        // The password is escaped for libpq (`\'`, not SQL's `''`) and the resulting
        // conninfo is then escaped for DuckDB by doubling every quote. Asserting the whole
        // string is what pins that both layers ran and in that order.
        let relation = engine.relation(&view).expect("relation");

        assert_eq!(
            relation,
            r"postgres_scan('host=''db.internal'' port=''5433'' dbname=''warehouse'' user=''fs_runner'' password=''fea\''ther''', 'public', 'user_stats')"
        );
        assert!(!relation.contains("password=''fea''ther''"), "{relation}");
    }

    #[test]
    fn a_postgres_source_naming_no_connection_is_rejected() {
        let view = {
            let mut view = source_view(Source::postgres("pg_prod", "public", "user_stats"));
            view.name = "user_stats".to_owned();
            view
        };

        let error = engine().relation(&view).expect_err("must fail");

        assert_eq!(
            error.to_string(),
            "view `user_stats` names connection `pg_prod`, which is not configured"
        );
    }

    #[test]
    fn a_postgres_source_through_another_kind_is_rejected() {
        let connections = BTreeMap::from([(
            "s3_lake".to_owned(),
            connection(r#"{"type":"s3","region":"us-east-1","key_id":"k","secret":"s"}"#),
        )]);
        let engine = Engine::open(&Limits::default(), &connections).expect("engine");
        let view = {
            let mut view = source_view(Source::postgres("s3_lake", "public", "user_stats"));
            view.name = "user_stats".to_owned();
            view
        };

        let error = engine.relation(&view).expect_err("must fail");

        assert_eq!(
            error.to_string(),
            "view `user_stats` reads a `postgres` source through connection `s3_lake`, \
             which is a `s3` connection"
        );
    }

    #[test]
    fn a_local_file_source_loads_no_extension() {
        // The deferred design: a project reading local files must not touch the extension
        // repository, so neither the httpfs nor the postgres scanner is loaded.
        let engine = engine();
        let source = Parquet::write(&integer_source(&[(1, 100, 10)]));

        engine
            .ensure_source_loaded(&view(&source.string(), None))
            .expect("a local file needs nothing loaded");

        let loaded: i64 = engine
            .connection()
            .expect("connection")
            .query_row(
                "SELECT count(*) FROM duckdb_extensions() \
                 WHERE loaded AND extension_name IN ('httpfs', 'postgres')",
                [],
                |row| row.get(0),
            )
            .expect("extensions");
        assert_eq!(loaded, 0, "a local file loaded a remote reader");
    }

    #[test]
    fn a_postgres_source_is_rejected_before_any_extension_loads() {
        // The connection is resolved before `INSTALL postgres`, so a name that is not
        // configured fails as that rather than after a fetch that may not even be possible.
        let engine = engine();
        let view = source_view(Source::postgres("absent", "public", "t"));

        let error = engine.ensure_source_loaded(&view).expect_err("must fail");

        assert!(matches!(error, Error::UnknownConnection { .. }), "{error}");
    }

    #[test]
    fn a_configured_connection_is_installed_as_a_secret() {
        let mut connections = BTreeMap::new();
        connections.insert(
            "s3_lake".to_owned(),
            connection(
                r#"{"type":"s3","region":"us-east-1","key_id":"AKIAEXAMPLE","secret":"shhh"}"#,
            ),
        );

        let engine = Engine::open(&Limits::default(), &connections).expect("engine");
        let names: String = engine
            .connection()
            .expect("connection")
            .query_row(
                "SELECT string_agg(name, ',') FROM duckdb_secrets()",
                [],
                |row| row.get(0),
            )
            .expect("secrets");

        assert!(names.contains("s3_lake"), "secrets were: {names}");
    }

    #[test]
    fn the_extension_directory_is_applied() {
        let directory = std::env::temp_dir().join(format!("feather-ext-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);

        let engine = open(&Limits {
            extension_directory: Some(directory.clone()),
            ..Limits::default()
        })
        .expect("engine");

        assert!(directory.is_dir());
        let resolved: String = engine
            .connection()
            .expect("connection")
            .query_row("SELECT current_setting('extension_directory')", [], |row| {
                row.get(0)
            })
            .expect("setting");
        assert!(resolved.contains("feather-ext-"), "{resolved}");

        drop(engine);
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn a_missing_spill_directory_is_created() {
        // DuckDB creates the directory it spills into but not its parents, so a nested path
        // that does not exist has to be created here, or the first spill fails instead.
        let parent = std::env::temp_dir().join(format!("feather-nested-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&parent);
        let nested = parent.join("a/b");

        let engine = open(&Limits {
            temp_directory: nested.clone(),
            ..Limits::default()
        })
        .expect("engine");

        assert!(engine.spill_directory().is_dir());
        assert!(engine.spill_directory().starts_with(&nested));

        drop(engine);
        let _ = std::fs::remove_dir_all(&parent);
    }

    #[test]
    fn each_engine_spills_into_its_own_directory() {
        // DuckDB removes its spill directory when the instance shuts down, so two engines
        // sharing one would let the first to finish delete the second's live spill files.
        let limits = Limits::default();
        let first = open(&limits).expect("first");
        let second = open(&limits).expect("second");

        assert_ne!(first.spill_directory(), second.spill_directory());
    }

    #[test]
    fn an_engines_spill_directory_is_removed_with_it() {
        let spill = {
            let engine = open(&Limits::default()).expect("engine");
            engine.spill_directory().to_path_buf()
        };

        assert!(!spill.exists(), "{} was left behind", spill.display());
    }

    #[test]
    fn a_cgroup_limit_becomes_eighty_percent_of_it() {
        assert_eq!(memory_limit_from_cgroup("1000\n"), Some("800".to_owned()));
        assert_eq!(
            memory_limit_from_cgroup("1000000000"),
            Some("800000000".to_owned())
        );
        // `max`, the v1 sentinel, and zero all mean no real limit, which leaves DuckDB's own
        // default of 80% of RAM in place.
        assert_eq!(memory_limit_from_cgroup("max"), None);
        assert_eq!(memory_limit_from_cgroup("9223372036854771712"), None);
        assert_eq!(memory_limit_from_cgroup("0"), None);
        assert_eq!(memory_limit_from_cgroup(""), None);
    }

    #[test]
    fn a_rejected_setting_proves_it_reaches_the_engine() {
        // A value DuckDB refuses is itself the assertion that the statement ran: if the
        // setting were never applied, both of these would open successfully.
        assert!(
            open(&Limits {
                memory_limit: Some("not-a-size".to_owned()),
                ..Limits::default()
            })
            .is_err()
        );
        assert!(
            open(&Limits {
                max_temp_directory_size: Some("not-a-size".to_owned()),
                ..Limits::default()
            })
            .is_err()
        );
    }

    #[test]
    fn a_constrained_join_agrees_with_an_unconstrained_one() {
        // The limit is below what this join needs, so DuckDB spills intermediates and reads
        // them back. Measured at 64MB against this data: it succeeds and writes about 1.7MB
        // to the spill directory. Threads are pinned because DuckDB budgets memory per
        // thread, so leaving them unpinned would move the floor with the core count and make
        // a fixed limit machine-dependent.
        let source = Parquet::write(&integer_source(
            &(0..1_000_000i64)
                .map(|i| (i % 5_000, i, i))
                .collect::<Vec<_>>(),
        ));
        let labels = labels(
            &(0..100_000i64)
                .map(|i| (Some(i % 5_000), i + 1))
                .collect::<Vec<_>>(),
        );
        let view = view(&source.string(), None);

        let unconstrained = engine()
            .point_in_time_join(&labels, &view, &JoinOptions::default())
            .expect("unconstrained join");

        let constrained = open(&Limits {
            memory_limit: Some("64MB".to_owned()),
            ..Limits::default()
        })
        .expect("engine");
        constrained
            .connection()
            .expect("connection")
            .execute_batch("SET threads = 2")
            .expect("threads");
        let result = constrained
            .point_in_time_join(&labels, &view, &JoinOptions::default())
            .expect("constrained join");

        assert_eq!(result.num_rows(), unconstrained.num_rows());
        assert_eq!(rows(&result), rows(&unconstrained));
        assert_eq!(counts(&result), counts(&unconstrained));
    }

    /// A sink that counts what it was handed, for tests that only need the scan to run.
    #[derive(Default)]
    struct Counted(usize);

    impl LatestBatchSink for Counted {
        async fn accept(&mut self, batch: RecordBatch) -> Result<()> {
            self.0 += batch.num_rows();
            Ok(())
        }
    }

    #[test]
    fn a_panic_inside_a_scan_becomes_an_error_rather_than_unwinding() {
        // `duckdb`'s Arrow iterator panics when a chunk fetch or the Arrow conversion fails, and
        // there is no lazy iterator that yields an error instead. This is the shim that puts the
        // failure back into the return type, and it is exercised directly because a genuine
        // mid-scan fetch failure cannot be forced from a test. The two payload shapes are the two
        // `panic!` produces, plus one with no message at all.
        let borrowed = catching_panics(|| -> Option<RecordBatch> { panic!("boom") })
            .expect_err("a panic is an error");
        assert_eq!(borrowed, "boom");

        let owned = catching_panics(|| -> Option<RecordBatch> {
            std::panic::panic_any(String::from("owned"))
        })
        .expect_err("a panic is an error");
        assert_eq!(owned, "owned");

        let bare = catching_panics(|| -> Option<RecordBatch> { std::panic::panic_any(7u8) })
            .expect_err("a panic is an error");
        assert_eq!(bare, "a panic with no message");

        // And a scan that does not panic is unaffected.
        let kept = catching_panics(|| 7u8).expect("no panic");
        assert_eq!(kept, 7);
    }

    #[tokio::test]
    async fn a_scan_over_a_source_that_disappeared_returns_an_error_and_the_engine_answers() {
        let source = Parquet::write(&integer_source(&[(1, 100, 10)]));
        let declared = view(&source.string(), None);
        let engine = engine();

        // The fixture removes its file when it is dropped, so the source a view names is gone by
        // the time anything reads it. The failure comes back as a value, which is the property
        // that matters for a library called across FFI: a panic here would abort the host.
        drop(source);

        let mut sink = Counted::default();
        let failed = engine.scan_latest_per_entity(&declared, &mut sink).await;
        assert!(
            failed.is_err(),
            "a source that cannot be read has to be an error, not a panic"
        );
        assert_eq!(sink.0, 0);

        // The engine is usable afterwards, which is what a caller does next.
        let other = Parquet::write(&integer_source(&[(1, 300, 30)]));
        let mut again = Counted::default();
        engine
            .scan_latest_per_entity(&view(&other.string(), None), &mut again)
            .await
            .expect("the engine still answers");
        assert_eq!(again.0, 1);
    }

    #[test]
    fn a_discarded_connection_is_opened_again_by_the_next_query() {
        let source = Parquet::write(&integer_source(&[(1, 100, 10)]));
        let engine = engine();
        let options = JoinOptions::default();
        let joined = engine
            .point_in_time_join(
                &labels(&[(Some(1), 150)]),
                &view(&source.string(), None),
                &options,
            )
            .expect("join");
        assert_eq!(counts(&joined), [Some(10)]);

        // What a caught panic inside a scan does to the connection: it is dropped, because a
        // panic during a fetch leaves DuckDB's session state unknown and reusing it would be
        // worse than the failure that was just caught.
        engine.discard_connection();

        let reopened = engine
            .point_in_time_join(
                &labels(&[(Some(1), 150)]),
                &view(&source.string(), None),
                &options,
            )
            .expect("join through a fresh connection");
        assert_eq!(counts(&reopened), [Some(10)]);
    }
}
