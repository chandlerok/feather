//! Offline point-in-time joins, executed in the embedded DuckDB engine.
//!
//! One engine, DuckDB, for every source. A local Parquet file, an object-storage prefix,
//! and a warehouse table differ only in their path scheme, so nothing here selects a
//! compute backend.
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

use arrow::array::{ArrayRef, Int64Array};
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use duckdb::Connection;

use crate::definitions::{DType, FeatureView, Source};
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
    connection: Connection,
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

        let connection = Connection::open_in_memory()?;

        let mut settings = vec![format!(
            "SET temp_directory = {}",
            quote_literal(&spill.display().to_string())
        )];
        // An explicit limit wins; otherwise take the container's, and otherwise leave
        // DuckDB's own default of 80% of RAM alone.
        if let Some(memory) = limits.memory_limit.clone().or_else(cgroup_memory_limit) {
            settings.push(format!("SET memory_limit = {}", quote_literal(&memory)));
        }
        if let Some(cap) = &limits.max_temp_directory_size {
            settings.push(format!(
                "SET max_temp_directory_size = {}",
                quote_literal(cap)
            ));
        }
        if let Some(directory) = &limits.extension_directory {
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
        let secrets: Vec<String> = connections
            .iter()
            .filter_map(|(name, configured)| s3_secret_sql(name, configured))
            .collect();
        if !secrets.is_empty() {
            connection.execute_batch("INSTALL httpfs; LOAD httpfs;")?;
            for sql in &secrets {
                connection.execute_batch(sql)?;
            }
        }

        Ok(Self {
            connection,
            spill,
            connections: connections.clone(),
        })
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
    ///     [`Error::UnknownConnection`] or [`Error::SourceConnectionKind`] if a Postgres
    ///         source names a connection the project cannot use. Checked before the
    ///         extension is installed, so a connection mistake fails as itself rather
    ///         than after a fetch.
    ///     [`Error::DuckDb`] if the extension cannot be installed or loaded.
    fn ensure_source_loaded(&self, view: &FeatureView) -> Result<()> {
        match &view.source {
            Source::File { path } if path_needs_filesystem(path) => {
                self.connection
                    .execute_batch("INSTALL httpfs; LOAD httpfs;")?;
            }
            Source::File { .. } => {}
            Source::Postgres { connection, .. } => {
                self.postgres_conninfo(view, connection)?;
                self.connection
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
            self.reject_ambiguous_timestamps(view, &relation, &key_expr, &source_ts_value)?;
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
    fn describe_source(&self, view: &FeatureView) -> Result<TypeMap> {
        let sql = format!("DESCRIBE SELECT * FROM {}", self.relation(view)?);
        let mut statement = self.connection.prepare(&sql)?;
        let mut rows = statement.query([])?;
        let mut types = TypeMap::default();
        while let Some(row) = rows.next()? {
            // DESCRIBE's first two columns are column_name and column_type.
            types.insert(row.get(0)?, row.get(1)?);
        }
        Ok(types)
    }

    /// Put the entity frame into a temporary table with an explicit row index.
    fn stage_labels(
        &self,
        entity_frame: &RecordBatch,
        frame_schema: &Schema,
        key_index: usize,
        label_index: usize,
    ) -> Result<()> {
        self.connection
            .execute_batch(&format!("DROP TABLE IF EXISTS {LABELS_TABLE}"))?;
        self.connection.execute_batch(&format!(
            "CREATE TEMPORARY TABLE {LABELS_TABLE} (feather_row BIGINT, {LABEL_KEY} {}, {LABEL_TS} {})",
            arrow_duckdb_type(frame_schema.field(key_index).data_type())?,
            arrow_duckdb_type(frame_schema.field(label_index).data_type())?,
        ))?;

        let batch = label_batch(entity_frame, frame_schema, key_index, label_index)?;
        let mut appender = self.connection.appender(LABELS_TABLE)?;
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
    ) -> Result<()> {
        // Restricted to the keys in the label frame: a duplicate among keys nobody asked
        // about cannot affect this join, and the restriction keeps the scan bounded.
        let sql = format!(
            "WITH source AS (SELECT {key_expr} AS k, {source_ts_value} AS t FROM {relation})
             SELECT CAST(k AS VARCHAR), t, count(*) AS n
             FROM source
             WHERE k IN (SELECT {LABEL_KEY} FROM {LABELS_TABLE})
             GROUP BY k, t
             HAVING count(*) > 1
             LIMIT 1"
        );
        let mut statement = self.connection.prepare(&sql)?;
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
        self.connection
            .execute_batch(&format!("DROP TABLE IF EXISTS {RESULT_TABLE}"))?;
        self.connection
            .execute_batch(&format!("CREATE TEMPORARY TABLE {RESULT_TABLE} AS {sql}"))?;

        let scan = format!("SELECT * FROM {RESULT_TABLE} ORDER BY {ROW_COLUMN}");
        let mut statement = self.connection.prepare(&scan)?;
        let mut batches = Vec::new();
        for batch in statement.query_arrow([])? {
            batches.push(batch);
        }
        Ok(batches)
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

    // Deduplicated only when a created timestamp can break the tie. Without one, the
    // ambiguity was already rejected, so the sort here cannot change which row wins.
    let dedup = match &view.created_timestamp_field {
        Some(created) => format!(
            " QUALIFY row_number() OVER (PARTITION BY {key_expr} ORDER BY {source_ts_value} DESC, {} DESC) = 1",
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
    /// ambiguity check all read the same expression rather than re-deriving it. A
    /// reader added later, a warehouse table or a table format, changes only this.
    ///
    /// Args:
    ///     view: The view whose source is read.
    ///
    /// Returns:
    ///     A SQL expression yielding the source's rows, already quoted for DuckDB.
    ///
    /// Raises:
    ///     [`Error::UnknownConnection`] or [`Error::SourceConnectionKind`] if a Postgres
    ///         source names a connection the project cannot use.
    fn relation(&self, view: &FeatureView) -> Result<String> {
        match &view.source {
            Source::File { path } => Ok(format!("read_parquet({})", quote_literal(path))),
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
    use crate::definitions::{Entity, Field as FeatureField};

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
    fn a_file_source_becomes_a_parquet_relation() {
        let relation = engine()
            .relation(&view("data/user_stats.parquet", None))
            .expect("relation");

        assert_eq!(relation, "read_parquet('data/user_stats.parquet')");
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
            .connection
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
            .connection
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
            .connection
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
            .connection
            .execute_batch("SET threads = 2")
            .expect("threads");
        let result = constrained
            .point_in_time_join(&labels, &view, &JoinOptions::default())
            .expect("constrained join");

        assert_eq!(result.num_rows(), unconstrained.num_rows());
        assert_eq!(rows(&result), rows(&unconstrained));
        assert_eq!(counts(&result), counts(&unconstrained));
    }
}
