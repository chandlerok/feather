//! Full-refresh materialization.
//!
//! A refresh recomputes every value a view serves from the view's source and overwrites it
//! in place. There is no watermark and no progress state, which is a correctness decision
//! before a simplicity one: a crashed run leaves the previous run's values in place, still
//! inside their TTL, and the next run overwrites all of them, so there is no partial state to
//! reconcile. See "Materialization" in the architecture document.
//!
//! The pipeline never leaves Arrow and holds one record batch at a time. DuckDB reduces the
//! source to the newest row per entity and streams batches out, each batch is encoded and
//! written before the next one is pulled, and nothing accumulates per entity-view pair. What
//! Feast pays for, three full copies of the whole result before the first write, is not paid
//! here.
//!
//! Two rules about freshness and expiry decide whether the read path agrees with the write
//! path, and both are easy to get wrong in a way nothing notices until a value is stale:
//!
//! - The freshness field holds the **winning row's event timestamp**, not the time of the
//!   write. `read_entities` compares it against `ttl_days`, so a wall-clock value would make
//!   every value look fresh forever.
//! - A value's expiry is the event timestamp plus the TTL, as an absolute instant. A TTL
//!   measured from the write would give a 40-day-old value another 30 days of life.

use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use arrow::array::{Array, Int64Array};
use arrow::record_batch::RecordBatch;

use crate::definitions::FeatureView;
use crate::error::{Error, Result};
use crate::key::{
    encode_entity_key_for, entity_hash_key, entity_key_component, freshness_field, value_field,
};
use crate::offline::{Engine, LatestBatchSink, SCAN_TS_COLUMN};
use crate::online::{OnlineStore, WriteBatch, WrittenField, encode_freshness};
use crate::value::encode_batch;

/// What one view's refresh did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewRefresh {
    /// The view's name, as declared.
    pub name: String,
    /// Entity rows written, which is one `HSET` per entity.
    pub rows: u64,
    /// The newest event timestamp any written row carried, in microseconds since the epoch.
    ///
    /// `None` when the refresh wrote nothing, which is a view whose source had no row with
    /// both an entity key and an event timestamp.
    pub max_event_timestamp_micros: Option<i64>,
    /// How long this view's scan and writes took.
    pub elapsed: Duration,
}

/// What one call to [`materialize`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializeReport {
    /// One entry per refreshed view, in declaration order.
    pub views: Vec<ViewRefresh>,
    /// Rows written across every refreshed view.
    pub total_rows: u64,
    /// How long the whole call took.
    pub elapsed: Duration,
}

/// Refresh feature values in the online store.
///
/// Every selected view is recomputed from its source and every value is overwritten, so the
/// call is idempotent and safe to re-run after a failure. Each view's source is read once, and
/// each entity's fields are written in one store batch.
///
/// Args:
///     store: The online store to write.
///     engine: The DuckDB engine the sources are read through.
///     project: The project name, which namespaces every key that is written.
///     declared: Every view the project declares. A name in `selected` is resolved here, so a
///         view the project does not declare is an error rather than a silent no-op.
///     selected: The names to refresh, or an empty slice to refresh every declared view. An
///         empty slice is the `views = None` case at the Python surface.
///
/// Returns:
///     A [`MaterializeReport`] naming each refreshed view, the rows it wrote, the newest event
///     timestamp those rows carried, and how long it took.
///
/// Raises:
///     [`Error::UnknownView`] if `selected` names a view the project does not declare.
///     [`Error::UnsupportedEntityCount`] if a view does not declare exactly one entity.
///     [`Error::MissingColumn`], [`Error::UnsupportedOfflineType`],
///     [`Error::ColumnTypeMismatch`], [`Error::UnknownConnection`],
///     [`Error::SourceConnectionKind`] or [`Error::UnreadableSource`] if a view's source
///     cannot be read as the view declares it.
///     [`Error::StreamInterrupted`] if a scan failed after it had started.
///     [`Error::NullEntityKey`] if a source row's entity key is null, which the scan's own
///         filter already excludes.
///     [`Error::UnsupportedKeyType`] if a source's entity key is not an integer or a string.
///     Whatever the store reports for a failed write.
pub async fn materialize<S: OnlineStore>(
    store: &mut S,
    engine: &Engine,
    project: &str,
    declared: &[FeatureView],
    selected: &[String],
) -> Result<MaterializeReport> {
    let started = Instant::now();
    let chosen = select_views(declared, selected)?;

    let mut views = Vec::with_capacity(chosen.len());
    for view in chosen {
        let view_started = Instant::now();
        let mut sink = StoreSink::new(project, view, store);
        engine.scan_latest_per_entity(view, &mut sink).await?;
        views.push(ViewRefresh {
            name: view.name.clone(),
            rows: sink.rows,
            max_event_timestamp_micros: sink.max_event_timestamp_micros,
            elapsed: view_started.elapsed(),
        });
    }

    Ok(MaterializeReport {
        total_rows: views.iter().map(|view| view.rows).sum(),
        elapsed: started.elapsed(),
        views,
    })
}

/// The views a refresh was asked for, in declaration order.
///
/// A name that is repeated selects its view once, so the work is done once.
///
/// Args:
///     declared: Every view the project declares.
///     selected: The names to refresh, or an empty slice for every declared view.
///
/// Returns:
///     The views to refresh, in the order `declared` lists them, so a report reads the way the
///     project does.
///
/// Raises:
///     [`Error::UnknownView`] if a name is not declared.
fn select_views<'a>(
    declared: &'a [FeatureView],
    selected: &[String],
) -> Result<Vec<&'a FeatureView>> {
    let wanted: BTreeSet<&str> = selected.iter().map(String::as_str).collect();
    for name in &wanted {
        if !declared.iter().any(|view| view.name == *name) {
            return Err(Error::UnknownView((*name).to_owned()));
        }
    }
    Ok(declared
        .iter()
        .filter(|view| wanted.is_empty() || wanted.contains(view.name.as_str()))
        .collect())
}

/// The instant a value stops being readable, in Unix seconds.
///
/// Measured from the winning row's event timestamp rather than from the time of the write. A
/// 40-day-old value under a 30-day TTL has already expired, and an expiry measured from the
/// write would give it another 30 days of life; the read-time check would then disagree with
/// what the server holds. `None` for a view that declares no TTL.
fn value_expiry_unix_secs(view: &FeatureView, event_micros: i64) -> Option<i64> {
    let ttl_days = view.ttl_days?;
    Some(event_micros.div_euclid(1_000_000) + i64::from(ttl_days) * 86_400)
}

/// The sink that turns one view's scan into one store write per entity.
struct StoreSink<'a, S> {
    project: &'a str,
    view: &'a FeatureView,
    store: &'a mut S,
    rows: u64,
    max_event_timestamp_micros: Option<i64>,
}

impl<'a, S> StoreSink<'a, S> {
    /// A sink that writes `view`'s fields into `project`'s keys in `store`.
    fn new(project: &'a str, view: &'a FeatureView, store: &'a mut S) -> Self {
        Self {
            project,
            view,
            store,
            rows: 0,
            max_event_timestamp_micros: None,
        }
    }
}

impl<S: OnlineStore> LatestBatchSink for StoreSink<'_, S> {
    /// Encode one streamed batch and write it, then return so the next batch can be pulled.
    ///
    /// The batch arrives as the view's feature columns in declaration order, then the entity
    /// join key, then the winning row's event timestamp; see
    /// [`Engine::scan_latest_per_entity`]. One write call per batch is what bounds what a
    /// refresh holds: nothing here spans two batches.
    async fn accept(&mut self, batch: RecordBatch) -> Result<()> {
        let entity = self.view.entity()?;
        let features = self.view.features.len();
        if batch.num_columns() != features + 2 {
            return Err(Error::ColumnCountMismatch {
                view: self.view.name.clone(),
                fields: features + 2,
                columns: batch.num_columns(),
            });
        }

        let key_column = batch.column(features);
        let timestamp_column = batch.column(features + 1);
        let Some(event_timestamps) = timestamp_column.as_any().downcast_ref::<Int64Array>() else {
            return Err(Error::ColumnTypeMismatch {
                name: SCAN_TS_COLUMN.to_owned(),
                declared: "int64".to_owned(),
                actual: timestamp_column.data_type().to_string(),
            });
        };

        let encoded = encode_batch(
            &self.view.features,
            &batch.columns()[..features],
            0..batch.num_rows(),
        )?;

        let mut writes = Vec::with_capacity(encoded.len());
        for (row, range) in encoded.ranges.iter().enumerate() {
            let component = entity_key_component(key_column.as_ref(), row)?;
            let key = entity_hash_key(
                self.project,
                &entity.name,
                &encode_entity_key_for(&entity.name, &[&component])?,
            );
            let event_micros = event_timestamps.value(row);
            self.max_event_timestamp_micros = Some(
                self.max_event_timestamp_micros
                    .map_or(event_micros, |highest| highest.max(event_micros)),
            );
            writes.push(WriteBatch {
                key,
                fields: vec![
                    WrittenField::new(
                        value_field(&self.view.name),
                        encoded.buf[range.clone()].to_vec(),
                        value_expiry_unix_secs(self.view, event_micros),
                    ),
                    // The freshness field is deliberately left unexpired. The read path tells
                    // `Expired` from `NeverWritten` by comparing this field against the view's
                    // TTL, and reclaiming it would collapse two states the contract keeps
                    // apart.
                    WrittenField::new(
                        freshness_field(&self.view.name),
                        encode_freshness(event_micros),
                        None,
                    ),
                ],
            });
        }

        self.store.write(&writes).await?;
        self.rows += encoded.len() as u64;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use arrow::array::{Array, ArrayRef, Int64Array};
    use arrow::datatypes::{DataType, Field as ArrowField, Schema};
    use parquet::arrow::ArrowWriter;

    use super::*;
    use crate::definitions::{DType, Entity, Field as FeatureField, Source};
    use crate::offline::{Engine, Limits};
    use crate::online::memory::MemoryStore;
    use crate::value::{SchemaTag, decode_batch};

    static FIXTURE: AtomicU64 = AtomicU64::new(0);

    /// A Parquet file that removes itself when the test ends.
    struct Parquet {
        path: PathBuf,
    }

    impl Parquet {
        fn write(batch: &RecordBatch) -> Self {
            let path = std::env::temp_dir().join(format!(
                "feather-materialize-{}-{}.parquet",
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

    /// `user_id, event_timestamp, count` as non-null integers.
    fn source(rows: &[(i64, i64, i64)]) -> RecordBatch {
        let fields = vec![
            ArrowField::new("user_id", DataType::Int64, false),
            ArrowField::new("event_timestamp", DataType::Int64, false),
            ArrowField::new("count", DataType::Int64, false),
        ];
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.0))),
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.1))),
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.2))),
        ];
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).expect("source batch")
    }

    /// The same, with a null entity key and a null event timestamp both representable.
    fn source_with_nulls(rows: &[(Option<i64>, Option<i64>, i64)]) -> RecordBatch {
        let fields = vec![
            ArrowField::new("user_id", DataType::Int64, true),
            ArrowField::new("event_timestamp", DataType::Int64, true),
            ArrowField::new("count", DataType::Int64, false),
        ];
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from_iter(rows.iter().map(|r| r.0))),
            Arc::new(Int64Array::from_iter(rows.iter().map(|r| r.1))),
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.2))),
        ];
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).expect("source batch")
    }

    /// One view over a fixture, with the entity the fixtures are keyed on.
    fn a_view(path: &str, ttl_days: Option<u32>) -> FeatureView {
        FeatureView {
            name: "clicks".to_owned(),
            entities: vec![Entity::new("user_id", "user_id")],
            source: Source::file(path),
            features: vec![FeatureField::new("count", DType::Int64)],
            ttl_days,
            timestamp_field: None,
            created_timestamp_field: None,
        }
    }

    fn an_engine() -> Engine {
        Engine::open(&Limits::default(), &BTreeMap::new()).expect("engine")
    }

    /// The hash key one entity's values are written under.
    fn key_of(entity: &[u8]) -> Vec<u8> {
        entity_hash_key(
            "ads",
            "user_id",
            &encode_entity_key_for("user_id", &[entity]).unwrap(),
        )
    }

    /// The `count` stored for one entity, decoded the way a read would decode it.
    fn stored_count(store: &MemoryStore, view: &FeatureView, entity: &[u8]) -> Option<i64> {
        let blob = store
            .fields(&key_of(entity))?
            .get(&value_field(&view.name))?
            .clone();
        let columns = decode_batch(
            &view.features,
            SchemaTag::of(&view.features),
            &[blob.as_slice()],
        )
        .expect("decode");
        let counts = columns[0].as_any().downcast_ref::<Int64Array>()?;
        Some(counts.value(0))
    }

    /// The freshness stored for one entity, as it was written.
    fn stored_freshness(store: &MemoryStore, entity: &[u8]) -> Option<i64> {
        let bytes = store
            .fields(&key_of(entity))?
            .get(&freshness_field("clicks"))?
            .clone();
        let array: [u8; 8] = bytes.try_into().ok()?;
        Some(i64::from_le_bytes(array))
    }

    /// Refreshes one view against a fixture and returns the store and the report.
    async fn refresh(rows: &[(i64, i64, i64)]) -> (MemoryStore, MaterializeReport) {
        refresh_ttl(rows, None).await
    }

    /// The same, for a view that declares a TTL.
    async fn refresh_ttl(
        rows: &[(i64, i64, i64)],
        ttl_days: Option<u32>,
    ) -> (MemoryStore, MaterializeReport) {
        let source = Parquet::write(&source(rows));
        let view = a_view(&source.string(), ttl_days);
        let mut store = MemoryStore::new();
        let report = materialize(
            &mut store,
            &an_engine(),
            "ads",
            std::slice::from_ref(&view),
            &[],
        )
        .await
        .expect("materialize");
        (store, report)
    }

    /// The current time, in microseconds since the epoch.
    fn now_micros() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("a clock after 1970")
            .as_micros() as i64
    }

    #[tokio::test]
    async fn the_newest_row_per_entity_wins_whatever_order_it_is_scanned_in() {
        // Entity 1's newest row is scanned first and its oldest last, so a write that took the
        // last row it saw would produce 20 rather than 30.
        let (store, report) =
            refresh(&[(1, 300, 30), (1, 100, 10), (1, 200, 20), (2, 150, 99)]).await;

        assert_eq!(stored_count(&store, &a_view("x", None), b"1"), Some(30));
        assert_eq!(stored_count(&store, &a_view("x", None), b"2"), Some(99));
        assert_eq!(
            store.hash_count(),
            2,
            "one hash per entity, and only per entity"
        );
        assert_eq!(report.total_rows, 2);
        assert_eq!(report.views[0].rows, 2);
    }

    #[tokio::test]
    async fn freshness_is_the_winning_rows_event_timestamp() {
        let (store, report) = refresh(&[(1, 300, 30), (1, 500, 50), (1, 400, 40)]).await;

        // The winning row is the one at 500, not the newest timestamp in the batch and not the
        // time of the write.
        assert_eq!(stored_freshness(&store, b"1"), Some(500));
        assert_eq!(report.views[0].max_event_timestamp_micros, Some(500));
        assert!(
            report.views[0].max_event_timestamp_micros.unwrap() < 1_600_000_000_000_000,
            "a wall-clock write time would be far larger than the event timestamps here"
        );
    }

    #[tokio::test]
    async fn a_values_expiry_is_its_event_timestamp_plus_the_ttl() {
        // 100 microseconds is just past the epoch. The expiry is counted from there and not
        // from now, so it lands 30 days after the event rather than 30 days after this run.
        let (store, _) = refresh_ttl(&[(1, 100, 10)], Some(30)).await;

        assert_eq!(
            store.field_expiry(&key_of(b"1"), &value_field("clicks")),
            Some(30 * 86_400)
        );
        assert_eq!(
            store.field_expiry(&key_of(b"1"), &freshness_field("clicks")),
            None,
            "the freshness field must stay readable, or Expired and NeverWritten collapse"
        );
    }

    #[tokio::test]
    async fn a_value_already_older_than_its_ttl_expires_at_once() {
        // 2020-09-13, which is years older than a 30-day TTL.
        let long_ago = 1_600_000_000_000_000;
        let (store, _) = refresh_ttl(&[(1, long_ago, 10)], Some(30)).await;

        let expiry = store
            .field_expiry(&key_of(b"1"), &value_field("clicks"))
            .expect("a TTL sets an expiry");
        assert!(
            expiry < now_micros() / 1_000_000,
            "an expiry measured from the write would have handed this value another 30 days"
        );
    }

    #[tokio::test]
    async fn a_view_without_a_ttl_sets_no_expiry() {
        let (store, _) = refresh(&[(1, 100, 10)]).await;

        assert_eq!(
            store.field_expiry(&key_of(b"1"), &value_field("clicks")),
            None
        );
    }

    #[tokio::test]
    async fn a_row_with_no_key_or_no_timestamp_is_not_written() {
        let source = Parquet::write(&source_with_nulls(&[
            (None, Some(10), 1),
            (Some(2), None, 2),
            (Some(3), Some(20), 3),
        ]));
        let view = a_view(&source.string(), None);
        let mut store = MemoryStore::new();
        let report = materialize(
            &mut store,
            &an_engine(),
            "ads",
            std::slice::from_ref(&view),
            &[],
        )
        .await
        .expect("materialize");

        assert_eq!(
            store.hash_count(),
            1,
            "only the row with both a key and a timestamp"
        );
        assert_eq!(stored_count(&store, &view, b"3"), Some(3));
        assert_eq!(report.total_rows, 1);
    }

    #[tokio::test]
    async fn a_named_subset_refreshes_only_that_view() {
        let source = Parquet::write(&source(&[(1, 100, 10)]));
        let clicks = a_view(&source.string(), None);
        let mut stats = a_view(&source.string(), None);
        stats.name = "stats".to_owned();
        let declared = vec![clicks.clone(), stats.clone()];

        let mut store = MemoryStore::new();
        let report = materialize(
            &mut store,
            &an_engine(),
            "ads",
            &declared,
            &["stats".to_owned()],
        )
        .await
        .expect("materialize");

        assert_eq!(report.views.len(), 1);
        assert_eq!(report.views[0].name, "stats");
        assert_eq!(
            stored_freshness(&store, b"1"),
            None,
            "clicks was not refreshed"
        );
        assert!(
            store
                .fields(&key_of(b"1"))
                .expect("stats was written")
                .contains_key(&value_field("stats"))
        );
    }

    #[tokio::test]
    async fn a_view_the_project_does_not_declare_is_refused() {
        let source = Parquet::write(&source(&[(1, 100, 10)]));
        let view = a_view(&source.string(), None);
        let mut store = MemoryStore::new();
        let refused = materialize(
            &mut store,
            &an_engine(),
            "ads",
            std::slice::from_ref(&view),
            &["absent".to_owned()],
        )
        .await;

        assert!(matches!(refused, Err(Error::UnknownView(name)) if name == "absent"));
        assert_eq!(store.hash_count(), 0, "a rejected refresh writes nothing");
    }

    /// The batches one scan produced, so the streaming contract is asserted rather than
    /// assumed.
    #[derive(Default)]
    struct Batches {
        count: usize,
        rows: usize,
        columns: usize,
        max_event_timestamp_micros: Option<i64>,
    }

    impl LatestBatchSink for Batches {
        async fn accept(&mut self, batch: RecordBatch) -> Result<()> {
            self.count += 1;
            self.rows += batch.num_rows();
            self.columns = batch.num_columns();
            let timestamps = batch
                .column(batch.num_columns() - 1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("the last column of a scan is the event timestamp")
                .clone();
            for row in 0..timestamps.len() {
                self.max_event_timestamp_micros = Some(
                    self.max_event_timestamp_micros
                        .map_or(timestamps.value(row), |highest| {
                            highest.max(timestamps.value(row))
                        }),
                );
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn a_scan_streams_more_than_one_batch_and_lays_them_out_as_documented() {
        // More entities than one DuckDB chunk, so a scan that collected everything would show
        // up here as a single batch.
        let rows: Vec<(i64, i64, i64)> = (0..5_000).map(|i| (i, 100 + i, i)).collect();
        let source = Parquet::write(&source(&rows));
        let view = a_view(&source.string(), None);

        let mut sink = Batches::default();
        an_engine()
            .scan_latest_per_entity(&view, &mut sink)
            .await
            .expect("scan");

        assert!(
            sink.count > 1,
            "expected a stream of batches, got {}",
            sink.count
        );
        assert_eq!(sink.rows, 5_000);
        assert_eq!(
            sink.columns, 3,
            "one feature, the entity key, and the timestamp"
        );
        assert_eq!(sink.max_event_timestamp_micros, Some(5_099));
    }
}
