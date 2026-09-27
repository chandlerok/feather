//! Measures the layout claim: whether sorting a feature source by
//! `(entity key, event_timestamp)` is worth more than the format it is stored in.
//!
//! Run with `cargo run --release -p feather-core --features offline --example layout`.
//!
//! The join is the one `examples/duckdb_input.rs` times, over the same generated
//! feature table, so the two examples agree on what a feature source looks like. That
//! table is written twice, and only the row order differs: the same writer, the same
//! row group size, the same rows. `sorted` is ordered by
//! `(user_id, event_timestamp)`, so each row group covers one run of keys. `unsorted`
//! is a deterministic permutation of the same rows, so no row group has any local key
//! structure at all. See [`shuffle`] for why the permutation is the control and the
//! table's natural order is not.
//!
//! Two label ranges, because "lets DuckDB skip row groups for the key range" is
//! conditional on there being a key range to skip:
//!
//! - `half` walks the lower half of the key space, so the sorted file's later row
//!   groups are wholly outside it and the unsorted file's are not.
//! - `full` walks every key, so no row group is out of range in either file.
//!
//! The sorted case should win the first by more than the second. Winning both by the
//! same wide margin would be a property of this generated data rather than of the
//! layout, which is the reading to check for before quoting any of it.
//!
//! Three figures per case, and they answer different questions:
//!
//! - **bytes read** is what the parquet reader actually pulled through, from the
//!   kernel's `rchar` counter across the join. DuckDB's own query profile was the
//!   first choice and cannot answer it: on 1.105.05 the `READ_PARQUET` operator
//!   reports `Total Files Read` and nothing about rows or bytes, so the number would
//!   have to be invented.
//! - **row groups out of range** is read from the file's footer, so it is the most the
//!   layout could let the reader skip, not a record of what it skipped. Comparing it
//!   with the bytes actually read is what tells the two apart.
//! - **join ms** is the minimum over `REPEATS` runs, for the reason
//!   `examples/duckdb_input.rs` gives.
//!
//! A filtered scan at the end asks whether the skipping happens for *any* query shape or
//! only fails for the join. That distinction is the difference between "sorting is still
//! worth doing" and "sorting enables pruning".

use std::fs;
use std::sync::Arc;
use std::time::{Duration, Instant};

use duckdb::Connection;
use duckdb::arrow::array::{ArrayRef, Int64Array};
use duckdb::arrow::datatypes::{DataType, Field as ArrowField, Schema};
use duckdb::arrow::record_batch::RecordBatch;

/// Rows on the feature side, generated exactly as `examples/duckdb_input.rs` does.
const FEATURE_ROWS: usize = 2_000_000;
/// Distinct entity keys, so every key has two feature rows. `duckdb_input.rs` spells
/// this `1000000` in its SQL; naming it here is what keeps the two in step.
const DISTINCT_KEYS: usize = 1_000_000;
/// Label rows, in both cases. The ranges differ, not the count, so a comparison between
/// layouts within a case is over identical work.
const LABELS: usize = 500_000;
/// Rows per Parquet row group, stated so the row-group figures mean something.
/// `FEATURE_ROWS` is 16 groups of these, and each sorted group covers 65536 keys.
const ROW_GROUP_ROWS: usize = 131_072;
/// Repetitions per figure; the minimum join time and the median byte count are reported.
/// Seven, because the layout's effect on join time is small enough that three runs
/// cannot separate it from the run-to-run spread `duckdb_input.rs` already reports.
const REPEATS: usize = 7;

const UNSORTED_PATH: &str = "/tmp/feather-layout-unsorted.parquet";
const SORTED_PATH: &str = "/tmp/feather-layout-sorted.parquet";

/// The columns the feature file carries, so the shuffle can order by `i` without
/// writing it out.
const FEATURE_COLUMNS: &str = "user_id, event_timestamp, count";

fn main() {
    let conn = connect();

    let sizes = [
        write_layout(&conn, UNSORTED_PATH, &shuffle()),
        write_layout(&conn, SORTED_PATH, "ORDER BY user_id, event_timestamp"),
    ];

    println!("feature source layout");
    println!("  feature rows     {FEATURE_ROWS}");
    println!("  distinct keys    {DISTINCT_KEYS}");
    println!("  labels           {LABELS}");
    println!("  row group rows   {ROW_GROUP_ROWS}");
    println!("  unsorted on disk {0} bytes", sizes[0]);
    println!("  sorted on disk   {0} bytes", sizes[1]);

    // A label frame's key is `i * stride` and its timestamp rises with `i`, so `stride`
    // is the only thing that decides how much of the key space the join touches.
    for (name, stride) in [("half", 1), ("full", 2)] {
        let last_key = (LABELS * stride - 1) as i64;
        println!();
        println!("label range {name}: {LABELS} labels over keys 0..{last_key}");
        println!(
            "{:<10} {:>14} {:>12} {:>14} {:>10} {:>10}",
            "layout", "bytes read", "groups", "out of range", "join ms", "rows out"
        );
        println!("{}", "-".repeat(76));

        let mut measured = Vec::new();
        for (label, path) in [("unsorted", UNSORTED_PATH), ("sorted", SORTED_PATH)] {
            let result = measure(&conn, path, stride, last_key);
            println!(
                "{:<10} {:>14} {:>12} {:>14} {:>10} {:>10}",
                label,
                shown(result.bytes_read),
                result.groups,
                result.groups_out_of_range,
                format!("{:.1}", ms(result.join)),
                result.rows,
            );
            measured.push(result);
        }

        let sorted = measured.pop().expect("both layouts measured");
        let unsorted = measured.pop().expect("both layouts measured");
        if unsorted.join > Duration::ZERO {
            println!(
                "  join: sorted is {:.2}x the unsorted time ({:.1}ms vs {:.1}ms)",
                unsorted.join.as_secs_f64() / sorted.join.as_secs_f64(),
                ms(sorted.join),
                ms(unsorted.join)
            );
        }
        if let (Some(before), Some(after)) = (unsorted.bytes_read, sorted.bytes_read) {
            if after > 0 {
                println!(
                    "  bytes read: sorted reads {:.2}x the unsorted amount ({after} vs {before})",
                    before as f64 / after as f64
                );
            }
        }
    }

    // Whether the row-group skipping in the claim happens at all, or only fails for the
    // join above. A static filter is pushed into the reader, so this is the case where
    // pruning is available if it is available anywhere: the same 16 row groups, the same
    // half of the key space, and a predicate the reader can use before reading anything.
    let last_key = (LABELS - 1) as i64;
    println!();
    println!("filtered scan: keys 0..{last_key}, a static filter the reader can use");
    println!(
        "{:<10} {:>14} {:>12} {:>14}",
        "layout", "bytes read", "groups", "out of range"
    );
    println!("{}", "-".repeat(56));
    for (label, path) in [("unsorted", UNSORTED_PATH), ("sorted", SORTED_PATH)] {
        let groups = key_ranges(&conn, path);
        let measured = measure_scan(&conn, path, last_key);
        println!(
            "{:<10} {:>14} {:>12} {:>14}",
            label,
            shown(measured),
            groups.len(),
            groups
                .iter()
                .filter(|(low, high, _)| *high < 0 || *low > last_key)
                .count(),
        );
    }

    println!();
    println!("wrote {UNSORTED_PATH} and {SORTED_PATH}");
    let _ = fs::remove_file(UNSORTED_PATH);
    let _ = fs::remove_file(SORTED_PATH);
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// A figure that could not be measured, as a `?` rather than a zero.
///
/// A zero would read as "nothing was read", which is a different claim from "not
/// measured", and only the second is true.
fn shown(figure: Option<u64>) -> String {
    match figure {
        Some(number) => number.to_string(),
        None => "?".to_owned(),
    }
}

/// The unsorted layout: a deterministic permutation of the rows.
///
/// The first version of this used the table's natural order, which is a poor baseline
/// for a layout comparison and quietly made the claim look better than it is. Keys were
/// generated as `i % 1000000`, so consecutive rows had consecutive keys, every row group
/// was already sorted within itself, and 7 of the 16 groups fell outside the `half`
/// label range without being sorted at all. A permutation is the honest control: the
/// same rows, no local key structure anywhere.
///
/// `2654435761` is Knuth's multiplicative constant, coprime with `FEATURE_ROWS` (which
/// is `2^7 * 5^6`), so `i * k mod FEATURE_ROWS` is a bijection and the shuffled file
/// holds exactly the rows the sorted one does.
fn shuffle() -> String {
    format!("ORDER BY (i * 2654435761) % {FEATURE_ROWS}")
}

/// An in-memory connection holding the generated feature table.
fn connect() -> Connection {
    let conn = Connection::open_in_memory().expect("open duckdb");
    conn.execute_batch(&format!(
        "CREATE TABLE features AS
         SELECT i,
                (i % {DISTINCT_KEYS}) AS user_id,
                1700000000000000 + (i * 7) AS event_timestamp,
                (i % 1000) AS count
         FROM range({FEATURE_ROWS}) t(i);"
    ))
    .expect("build features");
    conn
}

/// Write one layout, and return the file's size on disk.
fn write_layout(conn: &Connection, path: &str, order: &str) -> u64 {
    let _ = fs::remove_file(path);
    conn.execute_batch(&format!(
        "COPY (SELECT {FEATURE_COLUMNS} FROM features {order}) TO '{path}' \
         (FORMAT PARQUET, ROW_GROUP_SIZE {ROW_GROUP_ROWS})"
    ))
    .expect("write parquet");
    fs::metadata(path).expect("stat parquet").len()
}

fn labels(n: usize, stride: usize) -> RecordBatch {
    let ids: ArrayRef = Arc::new(Int64Array::from(
        (0..n).map(|i| (i * stride) as i64).collect::<Vec<_>>(),
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

/// `ASOF LEFT JOIN` against the file, as `examples/duckdb_input.rs` writes it: the
/// feature source stays an inline `read_parquet`, so there is no table to `ANALYZE` and
/// the pruning under test is the reader's own use of the footer statistics, which is the
/// mechanism the claim names.
fn join_sql(path: &str) -> String {
    format!(
        "SELECT count(*) FROM labels l ASOF LEFT JOIN read_parquet('{path}') f
         ON l.user_id = f.user_id AND l.event_timestamp >= f.event_timestamp"
    )
}

struct Measured {
    join: Duration,
    rows: i64,
    bytes_read: Option<u64>,
    groups: usize,
    groups_out_of_range: usize,
}

/// Time the join, count the bytes it read, and read the file's footer for the pruning
/// the layout made available.
fn measure(conn: &Connection, path: &str, stride: usize, last_key: i64) -> Measured {
    conn.execute_batch("DROP TABLE IF EXISTS labels")
        .expect("drop labels");
    conn.execute_batch("CREATE TEMPORARY TABLE labels (user_id BIGINT, event_timestamp BIGINT)")
        .expect("create labels");
    let batch = labels(LABELS, stride);
    {
        let mut appender = conn.appender("labels").expect("appender");
        appender.append_record_batch(batch).expect("append batch");
        appender.flush().expect("flush");
    }

    let mut join = Duration::MAX;
    let mut bytes = Vec::new();
    let mut rows = 0;
    for _ in 0..REPEATS {
        let before = rchar();
        let started = Instant::now();
        let matched = run_join(conn, path);
        let elapsed = started.elapsed();
        if let (Some(before), Some(after)) = (before, rchar()) {
            if after > before {
                bytes.push(after - before);
            }
        }
        join = join.min(elapsed);
        rows = matched;
    }
    // Every case is an `ASOF LEFT JOIN`, so the row count is the label count whatever
    // the layout did. A different number means the two cases are not comparable.
    assert_eq!(rows, LABELS as i64, "every label row survives the join");
    bytes.sort_unstable();

    let groups = key_ranges(conn, path);
    Measured {
        join,
        rows,
        bytes_read: bytes.get(bytes.len() / 2).copied(),
        groups: groups.len(),
        groups_out_of_range: groups
            .iter()
            .filter(|(low, high, _)| *high < 0 || *low > last_key)
            .count(),
    }
}

/// A scan with a static key-range filter, which the reader can push down.
///
/// The ASOF join reads its build side before it knows any label keys, so a dynamic
/// filter arrives too late to skip a row group. This query shape puts the predicate in
/// the plan instead, which is what tells the two apart.
fn scan_sql(path: &str, last_key: i64) -> String {
    format!("SELECT count(*) FROM read_parquet('{path}') WHERE user_id <= {last_key}")
}

/// The median bytes one filtered scan read, or `None` if the counter was unavailable.
fn measure_scan(conn: &Connection, path: &str, last_key: i64) -> Option<u64> {
    let sql = scan_sql(path, last_key);
    let mut bytes = Vec::new();
    for _ in 0..REPEATS {
        let before = rchar();
        let rows: i64 = conn
            .query_row(&sql, [], |row| row.get(0))
            .expect("run filtered scan");
        assert!(rows > 0, "the filter's range must match rows");
        if let (Some(before), Some(after)) = (before, rchar()) {
            if after > before {
                bytes.push(after - before);
            }
        }
    }
    bytes.sort_unstable();
    bytes.get(bytes.len() / 2).copied()
}

fn run_join(conn: &Connection, path: &str) -> i64 {
    let sql = join_sql(path);
    let mut statement = conn.prepare(&sql).expect("prepare join");
    statement.query_row([], |row| row.get(0)).expect("run join")
}

/// Bytes this process has read from files so far, from the kernel's own counter.
///
/// `rchar` is the total returned by `read`, not only what reached the disk, so a file
/// already in the page cache still counts. That is the right meaning here: what the
/// parquet reader had to pull through, which is the thing row-group skipping reduces.
/// It is process-wide, so it is only meaningful across a span where nothing else reads,
/// which is why it is sampled immediately either side of one join.
fn rchar() -> Option<u64> {
    let counters = fs::read_to_string("/proc/self/io").ok()?;
    for line in counters.lines() {
        if let Some(value) = line.strip_prefix("rchar:") {
            return value.trim().parse().ok();
        }
    }
    None
}

/// Each row group's key range and compressed size, read from the file's own footer.
///
/// The pruning opportunity rather than a record of what the reader did: this is what
/// could have been skipped given the label keys, from the same statistics the reader
/// uses to decide. The measured bytes are what it actually read, and the two agreeing
/// is what confirms the mechanism.
fn key_ranges(conn: &Connection, path: &str) -> Vec<(i64, i64, u64)> {
    let sql = format!(
        "SELECT max(CASE WHEN path_in_schema = 'user_id'
                         THEN try_cast(stats_min_value AS BIGINT) END),
                max(CASE WHEN path_in_schema = 'user_id'
                         THEN try_cast(stats_max_value AS BIGINT) END),
                sum(total_compressed_size)::BIGINT
         FROM parquet_metadata('{path}')
         GROUP BY row_group_id
         ORDER BY row_group_id"
    );
    let mut statement = match conn.prepare(&sql) {
        Ok(statement) => statement,
        Err(error) => panic!("prepare parquet_metadata: {error}"),
    };
    let mut rows = match statement.query([]) {
        Ok(rows) => rows,
        Err(error) => panic!("query parquet_metadata: {error}"),
    };
    let mut ranges = Vec::new();
    while let Some(row) = rows.next().expect("read row groups") {
        let low: Option<i64> = row.get(0).expect("min");
        let high: Option<i64> = row.get(1).expect("max");
        let bytes: Option<u64> = row.get(2).expect("bytes");
        if let (Some(low), Some(high)) = (low, high) {
            ranges.push((low, high, bytes.unwrap_or(0)));
        }
    }
    ranges
}
