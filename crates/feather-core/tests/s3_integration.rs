//! Integration tests against a real S3-compatible object store.
//!
//! Run with `mise run s3:up`, which starts RustFS (see docker-compose.yml) and creates
//! the bucket these tests read from. The server is deliberately ordinary: one node, one
//! disk, no TLS, addressed path style, which is what an S3-compatible store reached at a
//! host and port looks like.
//!
//! What the test proves is that reading the source from an object store changes nothing
//! about the join: the same rows read from a local file and from `s3://` join to the same
//! batch, row for row and null for null. Everything the remote read needs on the way
//! there, the secret built from the configured connection and the filesystem `httpfs`
//! provides, is the engine's own, because the join goes through `Engine::open` rather than
//! re-implementing that wiring.
//!
//! There is no skip guard. An endpoint that never came up fails the test, for the same
//! reason the other integration tests do not skip: a container that is broken in CI
//! should be visible there rather than quietly passing.
//!
//! The endpoint, credentials, and bucket come from the environment so the test can run
//! against another S3-compatible server: `FEATHER_S3_ENDPOINT` (a host and port, since
//! `use_ssl` carries the scheme), `FEATHER_S3_ACCESS_KEY_ID`, `FEATHER_S3_SECRET_ACCESS_KEY`,
//! and `FEATHER_S3_BUCKET`. The defaults are the ones docker-compose.yml starts.

#![cfg(feature = "offline")]

use std::collections::BTreeMap;
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use arrow::array::{Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use feather_core::definitions::{DType, Entity, FeatureView, Field as FeatureField, Source};
use feather_core::offline::{Engine, JoinOptions, Limits};
use feather_core::settings::Connection;
use parquet::arrow::ArrowWriter;

/// The region the connection declares. The settings model requires one, and an
/// S3-compatible server accepts `us-east-1`, so this is fixed rather than read from the
/// environment.
const REGION: &str = "us-east-1";

/// How long a wait for the server may last before the test names the endpoint and gives
/// up. Comfortably longer than a cold container start, short enough that a container that
/// never came up fails the run instead of hanging it.
const READY_TIMEOUT: Duration = Duration::from_secs(30);

/// How long one connection attempt may take, so a black-holed address cannot stretch the
/// wait past `READY_TIMEOUT`.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

/// How often the readiness wait retries while the server is not listening yet.
const POLL_INTERVAL: Duration = Duration::from_millis(250);

static FIXTURE: AtomicU64 = AtomicU64::new(0);

/// The configured connection the engine is opened with, built through the wire shape the
/// settings loader produces for a `[connections]` table with `type = "s3"`.
fn connection() -> Connection {
    serde_json::from_value(serde_json::json!({
        "type": "s3",
        "region": REGION,
        "key_id": key_id(),
        "secret": secret(),
        "endpoint": endpoint(),
        // What an S3-compatible server needs: DuckDB's defaults, virtual hosted style and
        // TLS, are AWS's and this server answers neither.
        "use_ssl": false,
        "url_style": "path",
    }))
    .expect("connection")
}

/// The `s3://` location this run reads from, unique per run so parallel runs and leftovers
/// cannot collide.
fn location(tag: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!(
        "s3://{}/{tag}-{}-{nanos}.parquet",
        bucket(),
        std::process::id()
    )
}

fn endpoint() -> String {
    std::env::var("FEATHER_S3_ENDPOINT").unwrap_or_else(|_| "127.0.0.1:9000".to_owned())
}

/// The key id is not a secret, so it is read and used as a plain string, as the settings
/// model does.
fn key_id() -> String {
    std::env::var("FEATHER_S3_ACCESS_KEY_ID").unwrap_or_else(|_| "feathertest".to_owned())
}

fn secret() -> String {
    std::env::var("FEATHER_S3_SECRET_ACCESS_KEY").unwrap_or_else(|_| "feathertest".to_owned())
}

fn bucket() -> String {
    std::env::var("FEATHER_S3_BUCKET").unwrap_or_else(|_| "feather-test".to_owned())
}

/// Wait until something is listening at `endpoint`, which is a host and port.
///
/// Docker Compose cannot wait for this itself: the RustFS image has no healthcheck, and
/// `--wait` returns when the container is running rather than when the server accepts
/// connections.
fn wait_until_listening(endpoint: &str) {
    let deadline = Instant::now() + READY_TIMEOUT;
    loop {
        if accepts_connection(endpoint) {
            return;
        }
        if Instant::now() >= deadline {
            panic!(
                "nothing accepted a connection at {endpoint} within {READY_TIMEOUT:?}. \
                 `mise run s3:up` starts the server and creates the bucket the test reads."
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

/// Put a local Parquet file in the bucket, so both sides of the comparison read the same
/// rows.
///
/// The engine only reads, so the write is a DuckDB connection of the test's own, and it is
/// configured with the `s3_*` settings `httpfs` registers rather than with a secret. The
/// secret the engine builds from the connection is part of what this test covers, and
/// rebuilding it here would cover the copy instead.
///
/// The object is left behind: nothing in DuckDB's SQL removes a remote file, the key is
/// unique per run, and the bucket holds nothing but these.
fn upload(local: &Path, remote: &str) {
    let connection = duckdb::Connection::open_in_memory().expect("upload connection");
    connection
        .execute_batch(&format!(
            "INSTALL httpfs; LOAD httpfs;
             SET s3_region={};
             SET s3_endpoint={};
             SET s3_use_ssl=false;
             SET s3_url_style='path';
             SET s3_access_key_id={};
             SET s3_secret_access_key={};",
            literal(REGION),
            literal(&endpoint()),
            literal(&key_id()),
            literal(&secret()),
        ))
        .unwrap_or_else(|error| panic!("configuring the upload to {}: {error}", endpoint()));

    connection
        .execute_batch(&format!(
            "COPY (SELECT * FROM read_parquet({})) TO {} (FORMAT PARQUET)",
            literal(&local.display().to_string()),
            literal(remote),
        ))
        .unwrap_or_else(|error| {
            panic!(
                "writing {remote} failed: {error}. A missing bucket is the usual cause, \
                 and `mise run s3:up` creates it."
            )
        });
}

/// A SQL string literal. The values interpolated into the upload are this test's own, but
/// a credential is the wrong place to find out that a quote was not doubled.
fn literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

/// A Parquet file that removes itself when the test ends.
struct Parquet {
    path: PathBuf,
}

impl Parquet {
    fn write(batch: &RecordBatch) -> Self {
        let path = std::env::temp_dir().join(format!(
            "feather-s3-{}-{}.parquet",
            std::process::id(),
            FIXTURE.fetch_add(1, Ordering::Relaxed)
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
fn a_join_over_s3_matches_the_same_join_over_a_local_file() {
    let endpoint = endpoint();
    wait_until_listening(&endpoint);

    // Two users, timestamps far enough apart that the newest row at or before a label time
    // is unambiguous, and a row that the earlier labels must not pick up.
    let source = Parquet::write(&integer_source(&[(1, 100, 10), (1, 400, 40), (2, 150, 7)]));
    let remote = location("join");
    upload(&source.path, &remote);

    let connections = BTreeMap::from([("s3_lake".to_owned(), connection())]);
    let engine = Engine::open(&Limits::default(), &connections).expect("engine");

    // The same label frame for both joins: a label on a row's own timestamp, a label
    // between two rows of one user, a null key, a user with no row at or before its label
    // time, and a label after every row. The order is not the source's, so a result that
    // happened to come back sorted cannot hide a misalignment, and a null key and a
    // missing row keep the comparison honest about nulls.
    let entity_frame = labels(&[
        (Some(2), 150),
        (Some(1), 150),
        (None, 150),
        (Some(1), 50),
        (Some(1), 500),
    ]);
    let options = JoinOptions::default();

    let local = engine
        .point_in_time_join(&entity_frame, &view(&source.string(), None), &options)
        .expect("join over the local file");
    let over_s3 = engine
        .point_in_time_join(&entity_frame, &view(&remote, None), &options)
        .expect("join over s3");

    // The claim the design rests on. Not a hand-written expectation: the two reads have to
    // agree, row for row, null for null, in the frame's own order.
    assert_eq!(
        local, over_s3,
        "the same rows joined differently over {remote}"
    );

    // Guards, so the equality above cannot pass vacuously: the batch they agree on is one
    // row per label in label order, and holds both a matched and an unmatched feature.
    assert_eq!(
        rows(&local),
        (0..entity_frame.num_rows() as i64).collect::<Vec<_>>()
    );
    let seen = counts(&local);
    assert!(
        seen.iter().any(Option::is_some) && seen.iter().any(Option::is_none),
        "expected matched and unmatched rows, joined to {seen:?}"
    );
}
