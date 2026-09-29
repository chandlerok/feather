//! Integration tests against a real Postgres server.
//!
//! Run with `mise run postgres:up`, which starts Postgres (see docker-compose.yml). The
//! server is deliberately ordinary: one node, the default `public` schema, and a
//! superuser whose name and password come from the compose file.
//!
//! What the test proves is that reading a source from a database changes nothing about
//! the join: the same rows read from a local Parquet file and from a Postgres table join
//! to the same batch, row for row and null for null. Everything the read needs on the way
//! there, the connection string assembled from the configured connection and the scanner
//! the `postgres` extension provides, is the engine's own, because the join goes through
//! `Engine::open` rather than re-implementing that wiring.
//!
//! The connection the engine resolves comes from a `feather.toml` parsed by the core, with
//! its password interpolated, so the route a project actually takes is the one under test:
//! `feather.toml` to `Settings`, to `Connection::conninfo`, to the `postgres_scan` relation.
//!
//! Seeding goes through DuckDB too, with `ATTACH ... (TYPE postgres)` and `CREATE TABLE`,
//! so no `psql` step exists anywhere. One thing Postgres itself forces: the escaping test
//! needs a login role whose password holds a space, and `postgres_execute` runs that DDL
//! on the server, still through DuckDB.
//!
//! There is no skip guard. A server that never came up fails the test, for the same
//! reason the Valkey and RustFS integration tests do not skip: a container that is broken
//! in CI should be visible there rather than quietly passing.
//!
//! The endpoint and credentials come from the environment so the test can run against
//! another server: `FEATHER_POSTGRES_HOST`, `FEATHER_POSTGRES_PORT`,
//! `FEATHER_POSTGRES_DATABASE`, `FEATHER_POSTGRES_USER`, and `FEATHER_POSTGRES_PASSWORD`.
//! The defaults are the ones docker-compose.yml starts.

#![cfg(feature = "offline")]

use std::net::{TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use arrow::array::{Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use duckdb::Connection as DuckDb;
use feather_core::definitions::{DType, Entity, FeatureView, Field as FeatureField, Source};
use feather_core::offline::{Engine, JoinOptions, Limits};
use feather_core::settings::{Connection, Settings, parse_settings_with};
use parquet::arrow::ArrowWriter;

/// The `[connections]` name the engine resolves for the ordinary read.
const CONNECTION: &str = "pg_prod";

/// How long a wait for the server may last before the test names the endpoint and gives
/// up. Comfortably longer than a cold container start, short enough that a container that
/// never came up fails the run instead of hanging it.
const READY_TIMEOUT: Duration = Duration::from_secs(30);

/// How long one connection attempt may take, so a black-holed address cannot stretch the
/// wait past `READY_TIMEOUT`.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

/// How often the readiness wait retries while the server is not listening yet.
const POLL_INTERVAL: Duration = Duration::from_millis(250);

/// The password the escaping test gives its role: a space makes a bare value parse as two
/// settings, and a quote is the case where libpq's escaping differs from SQL's.
const AWKWARD_PASSWORD: &str = "fea 'ther";

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn host() -> String {
    std::env::var("FEATHER_POSTGRES_HOST").unwrap_or_else(|_| "127.0.0.1".to_owned())
}

fn port() -> u16 {
    std::env::var("FEATHER_POSTGRES_PORT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5433)
}

fn database() -> String {
    std::env::var("FEATHER_POSTGRES_DATABASE").unwrap_or_else(|_| "feathertest".to_owned())
}

fn user() -> String {
    std::env::var("FEATHER_POSTGRES_USER").unwrap_or_else(|_| "feathertest".to_owned())
}

fn password() -> String {
    std::env::var("FEATHER_POSTGRES_PASSWORD").unwrap_or_else(|_| "feathertest".to_owned())
}

fn endpoint() -> String {
    format!("{}:{}", host(), port())
}

/// A name unique to this run and this call, so parallel runs and leftovers cannot collide.
fn unique(tag: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!(
        "feather_{tag}_{}_{}_{nanos}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    )
}

/// A configured connection, built through the core's own settings loader.
///
/// The `feather.toml` text is the real route into a `Connection`, and the password is
/// interpolated through the loader's environment closure rather than written into the file,
/// so resolving `${POSTGRES_PASSWORD}` is part of what the read exercises. Everything else
/// about the entry is what a project's file would say.
fn settings(user: &str, password: &str) -> Settings {
    let text = format!(
        "project = \"feather_test\"\n\
         definitions = [\"definitions/test.py\"]\n\
         \n\
         [connections.{CONNECTION}]\n\
         type = \"postgres\"\n\
         host = \"{host}\"\n\
         port = {port}\n\
         database = \"{database}\"\n\
         user = \"{user}\"\n\
         password = \"${{POSTGRES_PASSWORD}}\"\n",
        host = host(),
        port = port(),
        database = database(),
    );

    parse_settings_with(&text, &|name| match name {
        "POSTGRES_PASSWORD" => Ok(password.to_owned()),
        _ => Err(std::env::VarError::NotPresent),
    })
    .expect("the settings parse")
}

fn connection(user: &str, password: &str) -> Connection {
    settings(user, password).connections[CONNECTION].clone()
}

fn conninfo(user: &str, password: &str) -> String {
    connection(user, password).conninfo().expect("a conninfo")
}

/// Wait until something is listening at `endpoint`, which is a host and port.
fn wait_until_listening(endpoint: &str) {
    let deadline = Instant::now() + READY_TIMEOUT;
    loop {
        if accepts_connection(endpoint) {
            return;
        }
        if Instant::now() >= deadline {
            panic!(
                "nothing accepted a connection at {endpoint} within {READY_TIMEOUT:?}. \
                 `mise run postgres:up` starts the server."
            );
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

/// Whether anything is listening at a host and port.
fn accepts_connection(endpoint: &str) -> bool {
    let Ok(addresses) = endpoint.to_socket_addrs() else {
        return false;
    };
    addresses
        .into_iter()
        .any(|address| TcpStream::connect_timeout(&address, CONNECT_TIMEOUT).is_ok())
}

/// A SQL string literal. The values interpolated into the setup are this test's own, but a
/// credential is the wrong place to find out that a quote was not doubled.
fn literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

/// The test's own connection to the server, used to set its objects up and remove them.
///
/// The engine only reads, so the writes are a DuckDB connection of the test's own,
/// attached through libpq with the same credentials the configured connection carries.
/// Attaching is inserting a row into a table of the test's own, so no server-side seeding
/// tool is involved.
struct Admin {
    duckdb: DuckDb,
}

impl Admin {
    fn attach() -> Self {
        wait_until_listening(&endpoint());
        let duckdb = DuckDb::open_in_memory().expect("setup connection");
        duckdb
            .execute_batch("INSTALL postgres; LOAD postgres;")
            .expect("load the postgres extension");
        duckdb
            .execute_batch(&format!(
                "ATTACH {} AS pg (TYPE postgres);",
                literal(&conninfo(&user(), &password()))
            ))
            .unwrap_or_else(|error| panic!("attaching to Postgres at {}: {error}", endpoint()));
        Self { duckdb }
    }

    fn create_table(&self, table: &str, rows: &[(i64, i64, i64)]) {
        self.duckdb
            .execute_batch(&format!(
                "CREATE TABLE pg.public.{table} \
                 (user_id BIGINT, event_timestamp BIGINT, count BIGINT)"
            ))
            .unwrap_or_else(|error| panic!("creating {table}: {error}"));
        let values = rows
            .iter()
            .map(|(user, timestamp, count)| format!("({user}, {timestamp}, {count})"))
            .collect::<Vec<_>>()
            .join(", ");
        self.duckdb
            .execute_batch(&format!("INSERT INTO pg.public.{table} VALUES {values}"))
            .unwrap_or_else(|error| panic!("filling {table}: {error}"));
    }

    fn drop_table(&self, table: &str) {
        let _ = self
            .duckdb
            .execute_batch(&format!("DROP TABLE IF EXISTS pg.public.{table}"));
    }

    /// Create a login role with the given password and let it read one table.
    ///
    /// The password is a Postgres string literal here, so its quote is doubled. That is
    /// the SQL layer; the libpq layer that escapes it again is the engine's, which is what
    /// the test is checking.
    fn create_reader(&self, role: &str, password: &str, table: &str) {
        let sql_password = password.replace('\'', "''");
        self.duckdb
            .execute_batch(&format!(
                "CALL postgres_execute('pg', \
                 $$CREATE ROLE {role} LOGIN PASSWORD '{sql_password}'$$)"
            ))
            .unwrap_or_else(|error| panic!("creating role {role}: {error}"));
        self.duckdb
            .execute_batch(&format!(
                "CALL postgres_execute('pg', $$GRANT SELECT ON public.{table} TO {role}$$)"
            ))
            .unwrap_or_else(|error| panic!("granting {role} select on {table}: {error}"));
    }

    fn drop_role(&self, role: &str) {
        let _ = self.duckdb.execute_batch(&format!(
            "CALL postgres_execute('pg', $$DROP ROLE IF EXISTS {role}$$)"
        ));
    }
}

/// A table, and optionally a role, removed when the test ends.
///
/// The table is dropped first: Postgres refuses to drop a role that still has privileges
/// granted to it, and dropping the table removes that dependency. A table left behind by a
/// run that was killed outright is harmless, because the name is unique to the run.
struct Fixture {
    admin: Admin,
    table: String,
    role: Option<String>,
}

impl Fixture {
    fn start(tag: &str) -> Self {
        let admin = Admin::attach();
        let table = unique(tag);
        Self {
            admin,
            table,
            role: None,
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.admin.drop_table(&self.table);
        if let Some(role) = self.role.take() {
            self.admin.drop_role(&role);
        }
    }
}

/// A Parquet file that removes itself when the test ends.
struct Parquet {
    path: PathBuf,
}

impl Parquet {
    fn write(batch: &RecordBatch) -> Self {
        let path = std::env::temp_dir().join(format!(
            "feather-postgres-{}-{}.parquet",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let file = std::fs::File::create(&path).expect("create fixture");
        let mut writer = ArrowWriter::try_new(file, batch.schema(), None).expect("parquet writer");
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

/// Label rows, with a nullable key so the null-key rule is exercised.
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

/// The label frame both tests join, chosen so the result holds a matched and an unmatched
/// row and is not in the source's own order.
fn entity_frame() -> RecordBatch {
    labels(&[
        (Some(2), 150),
        (Some(1), 150),
        (None, 150),
        (Some(1), 50),
        (Some(1), 500),
    ])
}

/// Two users 300 microseconds apart, so the newest row at or before a label time is
/// unambiguous and one label falls between two rows of the same user.
const ROWS: [(i64, i64, i64); 3] = [(1, 100, 10), (1, 400, 40), (2, 150, 7)];

fn file_view(path: &str) -> FeatureView {
    FeatureView {
        name: "user_clicks".to_owned(),
        entities: vec![Entity::new("user_id", "user_id")],
        source: Source::file(path),
        features: vec![FeatureField::new("count", DType::Int64)],
        ttl_days: None,
        timestamp_field: None,
        created_timestamp_field: None,
        pushable: false,
    }
}

fn postgres_view(table: &str) -> FeatureView {
    FeatureView {
        name: "user_clicks".to_owned(),
        entities: vec![Entity::new("user_id", "user_id")],
        source: Source::postgres(CONNECTION, "public", table),
        features: vec![FeatureField::new("count", DType::Int64)],
        ttl_days: None,
        timestamp_field: None,
        created_timestamp_field: None,
        pushable: false,
    }
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

/// Both reads agree, row for row and null for null, in the frame's own order.
fn assert_reads_agree(from_file: &RecordBatch, from_postgres: &RecordBatch, location: &str) {
    assert_eq!(
        from_file, from_postgres,
        "the same rows joined differently over {location}"
    );

    // Guards, so the equality above cannot pass vacuously: the batch they agree on is one
    // row per label in label order, and holds both a matched and an unmatched feature.
    let frame = entity_frame();
    assert_eq!(
        rows(from_file),
        (0..frame.num_rows() as i64).collect::<Vec<_>>()
    );
    let seen = counts(from_file);
    assert!(
        seen.iter().any(Option::is_some) && seen.iter().any(Option::is_none),
        "expected matched and unmatched rows, joined to {seen:?}"
    );
}

#[test]
fn a_join_over_postgres_matches_the_same_join_over_a_local_file() {
    let fixture = Fixture::start("join");
    fixture.admin.create_table(&fixture.table, &ROWS);

    let local = Parquet::write(&integer_source(&ROWS));
    let connections = settings(&user(), &password()).connections;
    let engine = Engine::open(&Limits::default(), &connections).expect("engine");

    let options = JoinOptions::default();
    let from_file = engine
        .point_in_time_join(&entity_frame(), &file_view(&local.string()), &options)
        .expect("join over the local file");
    let from_postgres = engine
        .point_in_time_join(&entity_frame(), &postgres_view(&fixture.table), &options)
        .expect("join over the postgres table");

    assert_reads_agree(&from_file, &from_postgres, &fixture.table);
}

#[test]
fn a_postgres_source_authenticates_with_a_password_that_needs_escaping() {
    // The case that proves the connection string is escaped at all: the default password
    // has no space or quote in it, so a bare value would parse and connect. This one makes
    // a bare value parse as two settings and a SQL-style doubled quote be refused, and a
    // malformed conninfo and a wrong password raise the same error, so the only usable
    // assertion is that the read succeeds with real rows in it.
    let mut fixture = Fixture::start("escape");
    fixture.admin.create_table(&fixture.table, &ROWS);
    let reader = unique("reader");
    // Recorded before the role exists, so a failure below still tries to drop it. The DDL
    // is idempotent, so that ordering costs nothing.
    fixture.role = Some(reader.clone());
    fixture
        .admin
        .create_reader(&reader, AWKWARD_PASSWORD, &fixture.table);

    let local = Parquet::write(&integer_source(&ROWS));
    let connections = settings(&reader, AWKWARD_PASSWORD).connections;
    let engine = Engine::open(&Limits::default(), &connections).expect("engine");

    let options = JoinOptions::default();
    let from_file = engine
        .point_in_time_join(&entity_frame(), &file_view(&local.string()), &options)
        .expect("join over the local file");
    let from_postgres = engine
        .point_in_time_join(&entity_frame(), &postgres_view(&fixture.table), &options)
        .unwrap_or_else(|error| {
            panic!("joining as a role whose password is {AWKWARD_PASSWORD:?}: {error}")
        });

    assert_reads_agree(&from_file, &from_postgres, &fixture.table);
}
