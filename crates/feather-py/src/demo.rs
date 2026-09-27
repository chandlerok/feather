//! The sample project `feather demo` writes.
//!
//! Two Parquet files, because a feature store has nothing to show without data:
//! a feature table the generated view reads, and a label set a historical join
//! is taken against. The writer lives here rather than in `feather-core` because
//! it is scaffolding for the CLI, not part of the engine.
//!
//! Timestamps are int64 microseconds since the epoch rather than a Parquet
//! timestamp column. The core accepts either and normalizes both, and integers
//! are the form the bundled DuckDB build compares with, since it is built
//! without ICU.
//!
//! The rows are relative to the current day, not fixed dates, because the
//! generated view expires values after 30 days. A fixed table would read as
//! every feature missing a year after the project was created.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{ArrayRef, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;

/// Users in the generated project. Enough for a training set to be worth having
/// and few enough that the files stay small.
const USERS: i64 = 250;

/// Days of history the feature table holds, one row per user per day.
const FEATURE_DAYS: usize = 14;

/// Label rows per user. Each sits half a day after a feature row, so the join
/// takes that day's features rather than the next row, and its target is the
/// following day's purchases.
const LABEL_DAYS: usize = 5;

const MICROS_PER_DAY: i64 = 86_400_000_000;
const MICROS_PER_HOUR: i64 = 3_600_000_000;

/// Write the demo project into `directory`, which is created if it is missing.
///
/// The paths written are returned, so the caller reports what was produced
/// rather than repeating the names. Each file is renamed into place once it is
/// whole, so a failure part-way through leaves the earlier file written and the
/// later one missing, rather than a truncated Parquet under the name the
/// project reads.
#[pyfunction]
pub fn write_demo_data(directory: PathBuf) -> PyResult<Vec<String>> {
    let data = directory.join("data");
    std::fs::create_dir_all(&data).map_err(|error| message(&data, error))?;

    let (features, labels) = tables(midnight_micros(now_micros()?));
    let stats = data.join("user_stats.parquet");
    let training = data.join("training_labels.parquet");
    write_parquet(&stats, &features)?;
    write_parquet(&training, &labels)?;
    Ok(vec![
        stats.display().to_string(),
        training.display().to_string(),
    ])
}

/// The current time in microseconds since the epoch.
fn now_micros() -> PyResult<i64> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| PyOSError::new_err("the system clock is set before 1970"))?;
    i64::try_from(elapsed.as_micros())
        .map_err(|_| PyOSError::new_err("the current time does not fit in int64 microseconds"))
}

/// The most recent midnight, so a day boundary does not drift with the run.
fn midnight_micros(micros: i64) -> i64 {
    micros.div_euclid(MICROS_PER_DAY) * MICROS_PER_DAY
}

/// The feature table and the label set, built from the same pass so a label's
/// target really is the next day's data rather than a value invented twice.
fn tables(anchor: i64) -> (RecordBatch, RecordBatch) {
    let mut random = Random(0x5eed_1234_9abc_def0);
    let mut purchases = vec![vec![0_i64; USERS as usize]; FEATURE_DAYS];

    let mut user_ids = Vec::with_capacity(USERS as usize * FEATURE_DAYS);
    let mut timestamps = Vec::with_capacity(user_ids.capacity());
    let mut clicks = Vec::with_capacity(user_ids.capacity());
    let mut ordered = Vec::with_capacity(user_ids.capacity());
    for (day, row) in purchases.iter_mut().enumerate() {
        for (user, slot) in row.iter_mut().enumerate() {
            let clicked = (random.next() % 40) as i64;
            let bought = if random.next() % 4 == 0 {
                1 + (random.next() % 3) as i64
            } else {
                0
            };
            *slot = bought;
            user_ids.push(user as i64 + 1);
            timestamps.push(timestamp_of(anchor, day));
            clicks.push(clicked);
            ordered.push(bought);
        }
    }
    let features = batch(vec![
        ("user_id", DataType::Int64, user_ids),
        ("event_timestamp", DataType::Int64, timestamps),
        ("click_count", DataType::Int64, clicks),
        ("purchase_count", DataType::Int64, ordered),
    ]);

    // The newest feature day has no following day to be the target of, so the
    // labels stop one short of it, and each is paired with the day it predicts.
    let first = FEATURE_DAYS - 1 - LABEL_DAYS;
    let pairs = purchases[first..FEATURE_DAYS - 1]
        .iter()
        .zip(&purchases[first + 1..]);
    let mut user_ids = Vec::with_capacity(USERS as usize * LABEL_DAYS);
    let mut timestamps = Vec::with_capacity(user_ids.capacity());
    let mut labels = Vec::with_capacity(user_ids.capacity());
    for (offset, (_, tomorrow)) in pairs.enumerate() {
        for (user, bought) in tomorrow.iter().enumerate() {
            user_ids.push(user as i64 + 1);
            timestamps.push(timestamp_of(anchor, first + offset) + 12 * MICROS_PER_HOUR);
            labels.push(i64::from(*bought > 0));
        }
    }
    let labels = batch(vec![
        ("user_id", DataType::Int64, user_ids),
        ("event_timestamp", DataType::Int64, timestamps),
        ("label", DataType::Int64, labels),
    ]);
    (features, labels)
}

/// The event timestamp of feature day `day`, counted back from the anchor.
fn timestamp_of(anchor: i64, day: usize) -> i64 {
    anchor - (FEATURE_DAYS as i64 - 1 - day as i64) * MICROS_PER_DAY
}

/// One non-null int64 column per entry, in the order given.
///
/// Single pass over the columns, because a `DataType` is not `Copy` and a second
/// borrow of the input would have to clone one per column to move it out.
fn batch(columns: Vec<(&str, DataType, Vec<i64>)>) -> RecordBatch {
    let mut fields = Vec::with_capacity(columns.len());
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(columns.len());
    for (name, dtype, values) in columns {
        fields.push(Field::new(name, dtype, false));
        arrays.push(Arc::new(Int64Array::from(values)) as ArrayRef);
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
        .expect("the demo columns are non-null and the same length")
}

/// A linear congruential generator with a fixed seed, so two runs on the same
/// day differ only in their timestamps. A dependency would be more than this
/// needs.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
}

/// Write one Parquet file, so that a reader never sees a partial one.
///
/// The bytes go to a temporary name in the same directory, are flushed to the
/// device, and only then renamed over the destination. `rename(2)` within a
/// directory is atomic, so a reader sees either the file that was there before
/// or the whole new one, and a write that fails part-way through leaves the
/// previous file alone instead of a truncated Parquet in its place.
///
/// The `?` operator cannot express "and delete the temporary file", so the two
/// fallible steps run into one `Result` and the cleanup is the `if` under it.
/// That is the only place either step can leave something behind.
///
/// `ponytail:` the atomicity is per file, so a `demo` that fails on its second
/// file leaves the first one written and the second missing. The pair cannot be
/// renamed as a unit without staging the whole `data/` directory and swapping
/// it, and that swap cannot be done in place: `rename` will not replace a
/// non-empty directory, so the old one has to be moved aside first, which opens
/// a window where `data/` is not there at all. That is a worse failure than the
/// one this fixes, and the recovery for this one is the `--force` the CLI
/// already names in the message it prints. Revisit if a half-written demo
/// project reaches a user.
fn write_parquet(path: &Path, table: &RecordBatch) -> PyResult<()> {
    let temporary = temporary_path(path);
    let written = write_and_sync(&temporary, table)
        .and_then(|()| std::fs::rename(&temporary, path).map_err(|error| message(path, error)));
    if written.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    written
}

/// Write the Parquet bytes to `path` and flush them to the device.
///
/// The flush is what makes the rename that follows mean anything: a rename
/// publishes a name, and without the fsync the bytes behind that name can still
/// be only in the page cache if the process dies.
///
/// `close` consumes the writer and drops the `File` with it, so the handle to
/// sync is taken as a second one first. `try_clone` duplicates the descriptor
/// rather than opening the file again, so both handles name the same bytes.
///
/// Two closures rather than one, because the Parquet layer and the file layer
/// raise different error types and a closure has a single one: `io_failure`
/// covers the create, the clone and the flush, `failure` the three steps the
/// writer takes.
fn write_and_sync(path: &Path, table: &RecordBatch) -> PyResult<()> {
    let io_failure = |error| message(path, error);
    let failure = |error| message(path, error);
    let file = File::create(path).map_err(io_failure)?;
    let handle = file.try_clone().map_err(io_failure)?;
    let mut writer = ArrowWriter::try_new(file, table.schema(), None).map_err(failure)?;
    writer.write(table).map_err(failure)?;
    writer.close().map_err(failure)?;
    handle.sync_all().map_err(io_failure)
}

/// A sibling of `path` to write through before the rename.
///
/// The same directory, because a rename that crosses directories is a copy, and a
/// copy is not atomic. The name is hidden and carries this process's id, so a
/// second `feather demo` running elsewhere writes a temporary file of its own
/// rather than truncating this one. Two threads of one process racing on the
/// same directory are not a case this supports.
fn temporary_path(path: &Path) -> PathBuf {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    path.with_file_name(format!(".{name}.{}.tmp", std::process::id()))
}

/// One message for every failure under one path, whatever type the layer raised.
///
/// The two crates this touches disagree on their error types, so the helper takes
/// anything that displays rather than naming either.
fn message(path: &Path, error: impl std::fmt::Display) -> PyErr {
    PyOSError::new_err(format!("{}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::*;

    /// A fixed afternoon, so the tests do not depend on the day they run.
    const ANCHOR: i64 = 1_700_000_000_000_000;

    fn column<'a>(table: &'a RecordBatch, name: &str) -> &'a Int64Array {
        table
            .column_by_name(name)
            .expect("the column is declared")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("the column is int64")
    }

    #[test]
    fn a_label_reads_the_features_of_its_own_day() {
        let (features, labels) = tables(ANCHOR);
        let users = column(&features, "user_id");
        let timestamps = column(&features, "event_timestamp");
        let rows: HashSet<(i64, i64)> = (0..users.len())
            .map(|row| (users.value(row), timestamps.value(row)))
            .collect();
        let label_users = column(&labels, "user_id");
        let label_times = column(&labels, "event_timestamp");
        for row in 0..label_times.len() {
            let user = label_users.value(row);
            // Half a day after a feature row, so that day's row is the newest one
            // at or before the label. The next one for this user being a day later
            // is what makes that the newest: nothing sits in between.
            assert!(rows.contains(&(user, label_times.value(row) - 12 * MICROS_PER_HOUR)));
            assert!(rows.contains(&(
                user,
                label_times.value(row) - 12 * MICROS_PER_HOUR + MICROS_PER_DAY
            )));
        }
    }

    #[test]
    fn the_target_is_the_following_day() {
        let (features, labels) = tables(ANCHOR);
        let users = column(&features, "user_id");
        let timestamps = column(&features, "event_timestamp");
        let counts = column(&features, "purchase_count");
        let bought: HashMap<(i64, i64), i64> = (0..users.len())
            .map(|row| ((users.value(row), timestamps.value(row)), counts.value(row)))
            .collect();
        let label_users = column(&labels, "user_id");
        let label_times = column(&labels, "event_timestamp");
        let targets = column(&labels, "label");
        let mut classes = HashSet::new();
        for row in 0..targets.len() {
            let read_at = label_times.value(row) - 12 * MICROS_PER_HOUR;
            let predicted = bought[&(label_users.value(row), read_at + MICROS_PER_DAY)];
            assert_eq!(targets.value(row), i64::from(predicted > 0));
            classes.insert(targets.value(row));
        }
        // Both classes, or the target is a constant and nothing was demonstrated.
        assert_eq!(classes, HashSet::from([0, 1]));
    }

    #[test]
    fn every_row_lands_inside_the_thirty_day_ttl() {
        // What this pins: `tables` lays its rows out relative to the anchor it is
        // given, and every one of them lands inside the view's 30-day TTL. The
        // anchor is the expression `write_demo_data` itself builds from, rather
        // than a constant this test declares, so the arithmetic is checked
        // against the same midnight the writer would use.
        //
        // What this does not pin: that `write_demo_data` supplies that anchor. It
        // never calls the writer, so reverting the generator to a fixed date
        // leaves this green. What fails then is
        // `test_the_refreshed_values_are_served_from_the_same_store`, which reads
        // the values back and counts nulls, and a demo dated outside the TTL
        // produces nulls.
        let anchor = midnight_micros(now_micros().expect("the clock is set after 1970"));
        let (features, labels) = tables(anchor);
        for table in [&features, &labels] {
            for value in column(table, "event_timestamp").iter().flatten() {
                assert!(value <= anchor);
                assert!(value > anchor - 30 * MICROS_PER_DAY);
            }
        }
    }
}
