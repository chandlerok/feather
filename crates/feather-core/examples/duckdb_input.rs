//! Compares the three ways to get an Arrow entity frame into DuckDB for a
//! point-in-time join.
//!
//! Run with `cargo run --release -p feather-core --features offline \
//!   --example duckdb_input`.
//!
//! `ENTITIES` overrides the frame size (default 500000).
//!
//! The three routes:
//!
//! 1. **Arrow appender** (`appender-arrow`). Temporary table plus
//!    `append_record_batch`. Columnar, no per-row work, no serialization. Costs
//!    extra C++ template compilation on a clean build.
//! 2. **Row appender**. Same table, filled one row at a time. No extra build
//!    cost, but per-row work, which the design forbids.
//! 3. **Temp Parquet**. Write the batch to a file and `read_parquet` it. No extra
//!    build cost and it reuses existing code, but adds the serialization round
//!    trip the design forbids, plus a file to clean up.
//!
//! Two things this measures that are easy to miss:
//!
//! - The join is timed **before and after `ANALYZE`**. A table created by
//!   `CREATE TABLE AS` carries statistics; a table filled through the appender may
//!   not, and an ASOF join without statistics can plan worse. If the two columns
//!   differ, the appender path needs an explicit `ANALYZE` and that is a real
//!   requirement, not a micro-optimisation.
//! - Each mechanism runs several times and the minimum is reported, because a
//!   single join timing varied by more than 30% between mechanisms doing
//!   identical work.

use std::fs;
use std::sync::Arc;
use std::time::{Duration, Instant};

use duckdb::Connection;
use duckdb::arrow::array::{ArrayRef, Int64Array};
use duckdb::arrow::datatypes::{DataType, Field as ArrowField, Schema};
use duckdb::arrow::record_batch::RecordBatch;

/// Rows on the feature side, so the join has something to do.
const FEATURE_ROWS: usize = 2_000_000;
/// Repetitions per mechanism; the minimum is reported.
const REPEATS: usize = 3;
const PARQUET_PATH: &str = "/tmp/feather-labels.parquet";

fn entities() -> usize {
    std::env::var("ENTITIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v| *v > 0)
        .unwrap_or(500_000)
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn labels(n: usize) -> RecordBatch {
    let ids: ArrayRef = Arc::new(Int64Array::from(
        (0..n).map(|i| i as i64).collect::<Vec<_>>(),
    ));
    let ts: ArrayRef = Arc::new(Int64Array::from(
        (0..n)
            .map(|i| 1_700_000_000_000_000i64 + i as i64)
            .collect::<Vec<_>>(),
    ));
    let schema = Arc::new(Schema::new(vec![
        ArrowField::new("user_id", DataType::Int64, false),
        ArrowField::new("event_timestamp", DataType::Int64, false),
    ]));
    RecordBatch::try_new(schema, vec![ids, ts]).expect("labels batch")
}

fn connect() -> Connection {
    let conn = Connection::open_in_memory().expect("open duckdb");
    conn.execute_batch(&format!(
        "CREATE TABLE features AS
         SELECT (i % 1000000) AS user_id,
                1700000000000000 + (i * 7) AS event_timestamp,
                (i % 1000) AS count
         FROM range({FEATURE_ROWS}) t(i);"
    ))
    .expect("build features");
    conn
}

/// `ASOF LEFT JOIN` so unmatched labels survive, which is the documented
/// `on_missing: null` default.
fn join_sql(from: &str) -> String {
    format!(
        "SELECT count(*) FROM {from} l ASOF LEFT JOIN features f
         ON l.user_id = f.user_id AND l.event_timestamp >= f.event_timestamp"
    )
}

fn run_join(conn: &Connection, from: &str) -> Duration {
    let sql = join_sql(from);
    let started = Instant::now();
    let mut stmt = conn.prepare(&sql).expect("prepare join");
    let rows: i64 = stmt.query_row([], |r| r.get(0)).expect("run join");
    debug_assert!(rows > 0);
    started.elapsed()
}

#[derive(Default, Clone, Copy)]
struct Timings {
    load: Duration,
    join_plain: Duration,
    join_analyzed: Duration,
}

impl Timings {
    fn min_with(self, other: Self) -> Self {
        Self {
            load: self.load.min(other.load),
            join_plain: self.join_plain.min(other.join_plain),
            join_analyzed: self.join_analyzed.min(other.join_analyzed),
        }
    }
}

struct Mechanism {
    name: &'static str,
    note: &'static str,
    run: fn(usize) -> Timings,
}

fn arrow_appender(n: usize) -> Timings {
    let conn = connect();
    conn.execute_batch("CREATE TEMPORARY TABLE labels (user_id BIGINT, event_timestamp BIGINT)")
        .expect("create labels");
    let batch = labels(n);

    let started = Instant::now();
    let mut appender = conn.appender("labels").expect("appender");
    appender.append_record_batch(batch).expect("append batch");
    appender.flush().expect("flush");
    let load = started.elapsed();
    drop(appender);

    let join_plain = run_join(&conn, "labels");
    conn.execute_batch("ANALYZE labels").expect("analyze");
    let join_analyzed = run_join(&conn, "labels");

    Timings {
        load,
        join_plain,
        join_analyzed,
    }
}

fn row_appender(n: usize) -> Timings {
    let conn = connect();
    conn.execute_batch("CREATE TEMPORARY TABLE labels (user_id BIGINT, event_timestamp BIGINT)")
        .expect("create labels");

    let started = Instant::now();
    let mut appender = conn.appender("labels").expect("appender");
    appender
        .append_rows((0..n).map(|i| (i as i64, 1_700_000_000_000_000i64 + i as i64)))
        .expect("append rows");
    appender.flush().expect("flush");
    let load = started.elapsed();
    drop(appender);

    let join_plain = run_join(&conn, "labels");
    conn.execute_batch("ANALYZE labels").expect("analyze");
    let join_analyzed = run_join(&conn, "labels");

    Timings {
        load,
        join_plain,
        join_analyzed,
    }
}

fn temp_parquet(n: usize) -> Timings {
    let conn = connect();
    let batch = labels(n);

    let started = Instant::now();
    let file = fs::File::create(PARQUET_PATH).expect("create parquet");
    let mut writer =
        parquet::arrow::ArrowWriter::try_new(file, batch.schema(), None).expect("parquet writer");
    writer.write(&batch).expect("write parquet");
    writer.close().expect("close parquet");
    let load = started.elapsed();

    // The join reads the file, so its time includes the decode. There is no table
    // to analyze, so this route has no second measurement. Leaving it zero keeps
    // it out of the ANALYZE comparison below, where repeating the plain figure
    // would make the appender's statistics look like a non-issue.
    let from = format!("read_parquet('{PARQUET_PATH}')");
    let join_plain = run_join(&conn, &from);
    let join_analyzed = Duration::ZERO;

    let _ = fs::remove_file(PARQUET_PATH);
    Timings {
        load,
        join_plain,
        join_analyzed,
    }
}

fn main() {
    let n = entities();
    println!("entity frame input mechanisms");
    println!("  entities      {n}");
    println!("  feature rows  {FEATURE_ROWS}");
    println!("  repeats       {REPEATS} (minimum reported)\n");

    let mechanisms = [
        Mechanism {
            name: "arrow appender",
            note: "columnar, no serialization; extra C++ on clean build",
            run: arrow_appender,
        },
        Mechanism {
            name: "row appender",
            note: "per-row work; no extra build cost",
            run: row_appender,
        },
        Mechanism {
            name: "temp parquet",
            note: "serialization round trip; leaves a file",
            run: temp_parquet,
        },
    ];

    let mut measured = Vec::new();
    for m in &mechanisms {
        let mut best = Timings::default();
        for _ in 0..REPEATS {
            best = if best.load.is_zero() {
                (m.run)(n)
            } else {
                best.min_with((m.run)(n))
            };
        }
        measured.push((m, best));
    }

    println!(
        "{:<16} {:>9} {:>12} {:>12}  note",
        "mechanism", "load ms", "join ms", "join+ANALYZE"
    );
    println!("{}", "-".repeat(100));
    for (m, t) in &measured {
        let analyzed = if t.join_analyzed.is_zero() {
            "n/a".to_string()
        } else {
            format!("{:.1}", ms(t.join_analyzed))
        };
        println!(
            "{:<16} {:>9.1} {:>12.1} {:>12}  {}",
            m.name,
            ms(t.load),
            ms(t.join_plain),
            analyzed,
            m.note
        );
    }

    let fastest_load = measured
        .iter()
        .map(|(_, t)| t.load)
        .min()
        .unwrap_or_default();
    let slowest_load = measured
        .iter()
        .map(|(_, t)| t.load)
        .max()
        .unwrap_or_default();
    if fastest_load > Duration::ZERO {
        println!(
            "\nload: best is {:.1}x faster than worst ({:.1}ms vs {:.1}ms)",
            slowest_load.as_secs_f64() / fastest_load.as_secs_f64(),
            ms(fastest_load),
            ms(slowest_load)
        );
    }

    let with_analyze = measured
        .iter()
        .filter(|(_, t)| !t.join_analyzed.is_zero())
        .map(|(_, t)| t.join_analyzed)
        .min()
        .unwrap_or_default();
    let without = measured
        .iter()
        .map(|(_, t)| t.join_plain)
        .min()
        .unwrap_or_default();
    println!(
        "join: best without ANALYZE {:.1}ms, best with {:.1}ms",
        ms(without),
        ms(with_analyze)
    );
}
