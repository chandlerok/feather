//! Full-refresh materialization.
//!
//! A refresh recomputes every value a view serves from the view's source and overwrites it
//! in place. There is no watermark and no progress state, which is a correctness decision
//! before a simplicity one: a run that dies partway leaves some values from the previous run
//! in place, still inside their TTL, and the entities it had already overwritten holding what
//! it wrote; the next run overwrites all of them, so there is nothing partial to reconcile.
//! What a live run costs is that serving sees a mix of old and new values for as long as it
//! takes. See "Materialization" in the architecture document.
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

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use arrow::array::{Array, Int64Array};
use arrow::record_batch::RecordBatch;

use crate::definitions::FeatureView;
use crate::error::{Error, Result};
use crate::key::{
    encode_entity_key_for, entity_hash_key, entity_key_component, freshness_field,
    is_entity_hash_key, value_field, views_registry_key,
};
use crate::offline::{Engine, LatestBatchSink, SCAN_TS_COLUMN};
use crate::online::{OnlineStore, ProjectScan, WriteBatch, WrittenField, encode_freshness};
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
    /// The views a previous refresh declared and this one does not, whose fields were removed
    /// from the store. Empty on every run but the one that follows a rename or a removal.
    pub retired: Vec<String>,
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
///     [`Error::AmbiguousTimestamp`] if the source holds more than one row for one entity at
///         one event timestamp that nothing else separates: no `created_timestamp_field`, or
///         rows that share the one it declares. That is the same refusal the join makes, and it
///         is a per-view guarantee rather than a per-refresh one: a refresh scans and writes
///         its views in declaration order, so every view declared before this one has already
///         been refreshed by the time the check runs.
///     [`Error::NullEntityKey`] if a source row's entity key is null, which the scan's own
///         filter already excludes.
///     [`Error::UnsupportedKeyType`] if a source's entity key is not an integer or a string.
///     Whatever the store reports for a failed write.
pub async fn materialize<S: OnlineStore + ProjectScan>(
    store: &mut S,
    engine: &Engine,
    project: &str,
    declared: &[FeatureView],
    selected: &[String],
) -> Result<MaterializeReport> {
    let started = Instant::now();

    // The registry is read before anything is written, because the previously declared view set
    // is what identifies a view that has been renamed or removed. A project that has never been
    // refreshed has no registry yet, which reads as an empty set.
    let registry_key = views_registry_key(project);
    let mut registry: BTreeMap<String, Vec<u8>> = store
        .hash_fields(&registry_key)
        .await?
        .into_iter()
        .collect();

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

    // A retired view is one the previous refresh declared and this one does not. Its fields
    // cannot be found from the declarations alone: the entities it used to cover do not have to
    // appear in any source any more, so the keyspace is the only complete list of where they
    // are. This is therefore the one part of a refresh that is not a pure function of its
    // inputs, and it is why the walk happens here and nowhere else.
    let declared_names: BTreeSet<&str> = declared.iter().map(|view| view.name.as_str()).collect();
    let retired: Vec<String> = registry
        .keys()
        .filter(|name| !declared_names.contains(name.as_str()))
        .cloned()
        .collect();
    collect_orphans(store, project, &retired).await?;

    // A retired view leaves the registry too. Retaining the map below is not enough on its own:
    // `OnlineStore::write` sets the fields it is given and leaves every other field alone, which
    // is what makes a full refresh safe to re-run, so a name that stays in the store is retired
    // by this refresh and by every refresh after it, each of which walks the whole keyspace to
    // delete fields that are already gone. Removing the field is one `HDEL` at one key, and
    // removing a field that is already absent is a no-op, so this is safe to repeat.
    if !retired.is_empty() {
        store
            .delete_fields(&[(registry_key.clone(), retired.clone())])
            .await?;
    }

    // The registry is written last, after every value write. The order is the whole argument
    // for what a crashed run costs: a run that dies before this point leaves the retired views
    // in the registry, so the next run computes the same `retired` set and walks the keyspace
    // again, which changes nothing because removing a field that is already gone is a no-op.
    // Writing the registry first would tell the next run those views were never declared, and
    // their fields would stay in the store for good.
    //
    // What is written is the old registry restricted to the declared set, then updated with the
    // timestamps this refresh saw: `(old ∩ declared) ∪ refreshed`. Not the declared set itself.
    // A declared view whose every refresh has yet to write a row has no field at all, and a view
    // that was not selected this time keeps the timestamp an earlier refresh gave it: what the
    // registry records is the newest event timestamp *any* refresh has seen for a view, and a
    // refresh that did not touch a view did not see an older one, so leaving the timestamp alone
    // is what keeps the recorded freshness from going backwards.
    registry.retain(|name, _| declared_names.contains(name.as_str()));
    for view in &views {
        if let Some(highest) = view.max_event_timestamp_micros {
            registry.insert(view.name.clone(), encode_freshness(highest));
        }
    }
    // An empty registry is not written. `MemoryStore` creates a hash for every key it is handed
    // and an LSM queues no record at all for a batch with no fields, so writing one would
    // leave the double holding a key the server never creates, and nothing was recorded either
    // way: the next refresh reads the same empty registry and diffs the same declared set.
    if !registry.is_empty() {
        store
            .write(&[WriteBatch {
                key: registry_key,
                fields: registry
                    .into_iter()
                    .map(|(name, value)| WrittenField::new(name, value, None))
                    .collect(),
            }])
            .await?;
    }

    Ok(MaterializeReport {
        total_rows: views.iter().map(|view| view.rows).sum(),
        elapsed: started.elapsed(),
        views,
        retired,
    })
}

/// Remove a retired view's fields from every entity hash of a project.
///
/// The refresh already knows which views retired, from the registry it read. What it does not
/// know is where their fields are, so the project's keyspace is walked once. That walk is the
/// only reason a refresh reads keys it was not asked to write.
///
/// Args:
///     store: The online store to delete from.
///     project: The project whose keyspace is walked.
///     retired: The view names to remove the fields of. An empty list walks nothing.
///
/// Returns:
///     How many entity hashes were visited, whether or not they held anything.
///
/// Raises:
///     Whatever the store reports for a failed scan or deletion.
///
/// The deletion is idempotent: removing a field that is already gone is a no-op, so a run that
/// died after deleting can be repeated. That is what lets the registry be written last, and the
/// comment on that write says why the order matters.
pub async fn collect_orphans<S: OnlineStore + ProjectScan>(
    store: &mut S,
    project: &str,
    retired: &[String],
) -> Result<usize> {
    if retired.is_empty() {
        return Ok(0);
    }

    let registry_key = views_registry_key(project);
    let scanned = store.scan_entity_keys(project, &registry_key).await?;

    let mut fields = Vec::with_capacity(retired.len() * 2);
    for view in retired {
        fields.push(value_field(view));
        fields.push(freshness_field(view));
    }

    let mut deletions = Vec::with_capacity(scanned.len());
    for key in scanned {
        // Only a key shaped like an entity hash is touched. `HDEL` against a key that is not a
        // hash fails with `WRONGTYPE`, so a walk that deleted blindly would fail a refresh over
        // a key that has nothing to do with the view being retired.
        if is_entity_hash_key(&key, project) {
            deletions.push((key, fields.clone()));
        }
    }

    let visited = deletions.len();
    store.delete_fields(&deletions).await?;
    Ok(visited)
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
///
/// Rounded up, so rounding is not what makes the server reclaim a field while the read-time
/// check would still call it fresh. Rounding down opens a window just under a second wide where
/// a value the read path would have served is already gone, which reports it as never written
/// instead. The two sides read different clocks — the server's, against this absolute instant,
/// and the caller's `now`, against the recorded freshness — so a skew between them is the one
/// way left for the server to reclaim early, and no rounding direction can cover it.
fn value_expiry_unix_secs(view: &FeatureView, event_micros: i64) -> Option<i64> {
    let ttl_days = view.ttl_days?;
    // Rounded up rather than down. `i64::div_ceil` is still unstable in this toolchain, so the
    // half second of arithmetic is spelled out rather than imported behind a feature gate.
    let remainder = event_micros.rem_euclid(1_000_000) != 0;
    let seconds = event_micros.div_euclid(1_000_000) + i64::from(remainder);
    Some(seconds + i64::from(ttl_days) * 86_400)
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
    use crate::online::ReadRequest;
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

    /// `user_id, event_timestamp, created_at, count`, for the tie-break a view can declare.
    fn source_with_created(rows: &[(i64, i64, i64, i64)]) -> RecordBatch {
        let fields = vec![
            ArrowField::new("user_id", DataType::Int64, false),
            ArrowField::new("event_timestamp", DataType::Int64, false),
            ArrowField::new("created_at", DataType::Int64, false),
            ArrowField::new("count", DataType::Int64, false),
        ];
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.0))),
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.1))),
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.2))),
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.3))),
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
            .get(&value_field(&view.name))?;
        let columns =
            decode_batch(&view.features, SchemaTag::of(&view.features), &[blob]).expect("decode");
        let counts = columns[0].as_any().downcast_ref::<Int64Array>()?;
        Some(counts.value(0))
    }

    /// The freshness stored for one entity, as it was written.
    fn stored_freshness(store: &MemoryStore, entity: &[u8]) -> Option<i64> {
        let bytes = store
            .fields(&key_of(entity))?
            .get(&freshness_field("clicks"))?;
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
        assert!(
            store.fields(&key_of(b"3")).is_none(),
            "an entity the source never mentions gets no hash"
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

        // Rounded up to a whole second, because the server's absolute expiry is in seconds: 100
        // microseconds past the epoch is second 1. Rounding down would let the server reclaim
        // the field while the read-time check still calls it fresh, which turns a served value
        // into a missing one.
        assert_eq!(
            store.field_expiry(&key_of(b"1"), &value_field("clicks")),
            Some(1 + 30 * 86_400)
        );
        assert_eq!(
            store.field_expiry(&key_of(b"1"), &freshness_field("clicks")),
            None,
            "the freshness field must stay readable, or Expired and NeverWritten collapse"
        );
    }

    #[tokio::test]
    async fn a_whole_second_event_timestamp_expires_exactly() {
        // The rounding is up and never further: an event timestamp already on a second boundary
        // must not gain an extra second, or the expiry drifts away from the declared TTL.
        let (store, _) = refresh_ttl(&[(1, 2_000_000, 10)], Some(30)).await;

        assert_eq!(
            store.field_expiry(&key_of(b"1"), &value_field("clicks")),
            Some(2 + 30 * 86_400)
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

        assert!(
            store.fields(&key_of(b"1")).is_none(),
            "a row with a null key has no key to be written under"
        );
        assert!(
            store.fields(&key_of(b"2")).is_none(),
            "a row with a null timestamp has no freshness to record"
        );
        assert_eq!(stored_count(&store, &view, b"3"), Some(3));
        assert_eq!(report.total_rows, 1);
    }

    #[tokio::test]
    async fn a_source_with_repeated_nulls_is_not_ambiguous() {
        // Two rows with a null key at one timestamp. That is the shape the ambiguity check used
        // to group: it reads the key back as a string, a null is a column-type error rather than
        // a value, so a source the scan handles fine failed the whole refresh with a message
        // naming neither the source nor the rows. The two rows whose timestamp is null under one
        // entity are the same shape on the join path, where the check only looks at the keys a
        // frame names. Neither pair is an ambiguity a refresh has to resolve.
        let source = Parquet::write(&source_with_nulls(&[
            (None, Some(10), 1),
            (None, Some(10), 2),
            (Some(4), None, 3),
            (Some(4), None, 4),
            (Some(3), Some(20), 5),
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
        .expect("a source with repeated nulls is not ambiguous");

        assert_eq!(stored_count(&store, &view, b"3"), Some(5));
        assert_eq!(report.total_rows, 1);
        assert_eq!(report.views[0].max_event_timestamp_micros, Some(20));
        assert!(
            store.fields(&key_of(b"4")).is_none(),
            "a row with no timestamp has no freshness to record"
        );
    }

    #[tokio::test]
    async fn a_refresh_whose_source_yields_nothing_writes_no_registry() {
        // The registry is written only when it has something to record. A refresh over an empty
        // source has nothing, and the two stores disagreed about that: `MemoryStore` created an
        // empty record for the key while the LSM queued nothing at all and left the key
        // absent, so a test could see a registry the server never had.
        let source = Parquet::write(&source(&[]));
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
        .expect("an empty source is not an error");

        assert_eq!(report.views[0].rows, 0);
        assert_eq!(report.views[0].max_event_timestamp_micros, None);
        assert!(report.retired.is_empty());
        assert_eq!(report.total_rows, 0);
        assert!(
            store.fields(&views_registry_key("ads")).is_none(),
            "nothing was written, so no key exists, in either store"
        );
        assert_eq!(store.hash_count(), 0);
    }

    #[tokio::test]
    async fn every_refresh_writes_the_registry_with_each_views_newest_event_timestamp() {
        let (store, _) = refresh(&[(1, 300, 30), (1, 500, 50), (2, 900, 90)]).await;

        let registry = store
            .fields(&views_registry_key("ads"))
            .expect("the refresh writes the registry");
        assert_eq!(
            registry.get("clicks"),
            Some(encode_freshness(900).as_slice())
        );
    }

    #[tokio::test]
    async fn refreshing_a_subset_leaves_the_other_views_in_the_registry() {
        let source = Parquet::write(&source(&[(1, 100, 10)]));
        let clicks = a_view(&source.string(), None);
        let mut stats = a_view(&source.string(), None);
        stats.name = "stats".to_owned();
        let declared = vec![clicks, stats];

        let mut store = MemoryStore::new();
        let engine = an_engine();
        materialize(&mut store, &engine, "ads", &declared, &[])
            .await
            .expect("first refresh");
        materialize(&mut store, &engine, "ads", &declared, &["stats".to_owned()])
            .await
            .expect("subset refresh");

        // Both views stay declared, so neither is an orphan and neither leaves the registry.
        let registry = store.fields(&views_registry_key("ads")).expect("registry");
        assert_eq!(
            registry.get("clicks"),
            Some(encode_freshness(100).as_slice())
        );
        assert_eq!(
            registry.get("stats"),
            Some(encode_freshness(100).as_slice())
        );
    }

    /// A store that counts the keyspace walks a refresh performs, to pin that a refresh with
    /// nothing retired does not walk at all.
    struct CountingWalk {
        inner: MemoryStore,
        walks: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl OnlineStore for CountingWalk {
        async fn write(&mut self, batches: &[WriteBatch]) -> Result<()> {
            self.inner.write(batches).await
        }

        async fn read(&self, requests: &[ReadRequest]) -> Result<Vec<Vec<Option<Vec<u8>>>>> {
            self.inner.read(requests).await
        }

        async fn delete_fields(&mut self, keys: &[(Vec<u8>, Vec<String>)]) -> Result<()> {
            self.inner.delete_fields(keys).await
        }
    }

    impl ProjectScan for CountingWalk {
        async fn hash_fields(&self, key: &[u8]) -> Result<Vec<(String, Vec<u8>)>> {
            self.inner.hash_fields(key).await
        }

        async fn scan_entity_keys(&self, project: &str, exclude: &[u8]) -> Result<Vec<Vec<u8>>> {
            self.walks.fetch_add(1, Ordering::SeqCst);
            self.inner.scan_entity_keys(project, exclude).await
        }
    }

    /// Two views over one fixture, so one can be retired and the other left alone.
    fn two_views(path: &str) -> Vec<FeatureView> {
        let mut stats = a_view(path, None);
        stats.name = "stats".to_owned();
        vec![a_view(path, None), stats]
    }

    #[tokio::test]
    async fn a_retired_views_fields_go_and_a_live_views_fields_stay() {
        let source = Parquet::write(&source(&[(1, 100, 10), (2, 200, 20)]));
        let declared = two_views(&source.string());
        let mut store = MemoryStore::new();
        let engine = an_engine();
        materialize(&mut store, &engine, "ads", &declared, &[])
            .await
            .expect("refresh both views");

        // Only `clicks` is declared now, so `stats` is retired and its fields have to go.
        let report = materialize(&mut store, &engine, "ads", &declared[..1], &[])
            .await
            .expect("retire stats");

        assert_eq!(report.retired, vec!["stats".to_owned()]);
        for entity in [b"1".as_slice(), b"2".as_slice()] {
            let fields = store
                .fields(&key_of(entity))
                .expect("a live entity keeps its hash");
            assert!(fields.contains_key(&value_field("clicks")));
            assert!(fields.contains_key(&freshness_field("clicks")));
            assert!(!fields.contains_key(&value_field("stats")));
            assert!(!fields.contains_key(&freshness_field("stats")));
        }
    }

    #[tokio::test]
    async fn collecting_the_same_orphans_twice_is_the_same_as_collecting_them_once() {
        let source = Parquet::write(&source(&[(1, 100, 10)]));
        let declared = two_views(&source.string());
        let mut store = MemoryStore::new();
        materialize(&mut store, &an_engine(), "ads", &declared, &[])
            .await
            .expect("refresh both views");

        let once = collect_orphans(&mut store, "ads", &["stats".to_owned()])
            .await
            .expect("collect");
        let twice = collect_orphans(&mut store, "ads", &["stats".to_owned()])
            .await
            .expect("collect again");

        assert_eq!(once, 1);
        assert_eq!(
            twice, 1,
            "the walk happens either way; the deletion is a no-op"
        );
        let fields = store.fields(&key_of(b"1")).expect("hash");
        assert!(fields.contains_key(&value_field("clicks")));
        assert!(!fields.contains_key(&value_field("stats")));
    }

    #[tokio::test]
    async fn a_key_that_is_not_an_entity_hash_is_left_alone() {
        let source = Parquet::write(&source(&[(1, 100, 10)]));
        let declared = two_views(&source.string());
        let mut store = MemoryStore::new();
        materialize(&mut store, &an_engine(), "ads", &declared, &[])
            .await
            .expect("refresh both views");

        // Something else under the same project prefix, holding a field named like a retired
        // view's. A walk that deleted blindly would send `HDEL` at it.
        store
            .write(&[WriteBatch {
                key: b"ads:meta".to_vec(),
                fields: vec![WrittenField::new(value_field("stats"), b"x".to_vec(), None)],
            }])
            .await
            .expect("stray key");

        collect_orphans(&mut store, "ads", &["stats".to_owned()])
            .await
            .expect("collect");

        assert!(
            store
                .fields(b"ads:meta")
                .is_some_and(|fields| fields.contains_key(&value_field("stats"))),
            "a key that is not shaped like an entity hash is not touched"
        );
    }

    #[tokio::test]
    async fn a_retired_name_leaves_the_registry_so_later_refreshes_do_not_walk_again() {
        let source = Parquet::write(&source(&[(1, 100, 10)]));
        let declared = two_views(&source.string());
        let walks = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut store = CountingWalk {
            inner: MemoryStore::new(),
            walks: Arc::clone(&walks),
        };
        let engine = an_engine();

        // Both views declared, so both are in the registry.
        materialize(&mut store, &engine, "ads", &declared, &[])
            .await
            .expect("refresh both views");
        assert_eq!(walks.load(Ordering::SeqCst), 0);

        // `stats` is dropped, which retires it and costs exactly one walk.
        let renamed = materialize(&mut store, &engine, "ads", &declared[..1], &[])
            .await
            .expect("retire stats");
        assert_eq!(renamed.retired, vec!["stats".to_owned()]);
        assert_eq!(walks.load(Ordering::SeqCst), 1);
        let registry = store
            .inner
            .fields(&views_registry_key("ads"))
            .expect("registry");
        assert!(
            !registry.contains_key("stats"),
            "a retired name has to leave the registry, or every later refresh retires it again"
        );

        // A third refresh over the same declared set has nothing to retire and must not walk:
        // the name is gone from the registry, so it is not retired a second time.
        let again = materialize(&mut store, &engine, "ads", &declared[..1], &[])
            .await
            .expect("refresh again");
        assert!(again.retired.is_empty());
        assert_eq!(walks.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn an_unchanged_declared_set_never_walks_the_keyspace() {
        let source = Parquet::write(&source(&[(1, 100, 10)]));
        let declared = two_views(&source.string());
        let walks = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut store = CountingWalk {
            inner: MemoryStore::new(),
            walks: Arc::clone(&walks),
        };
        let engine = an_engine();
        materialize(&mut store, &engine, "ads", &declared, &[])
            .await
            .expect("first refresh");
        materialize(&mut store, &engine, "ads", &declared, &[])
            .await
            .expect("second refresh");

        assert_eq!(
            walks.load(Ordering::SeqCst),
            0,
            "the registry exists and nothing retired, so there is nothing to collect"
        );
        // Two refreshes, and the registry still names both views.
        let registry = store
            .inner
            .fields(&views_registry_key("ads"))
            .expect("registry");
        assert_eq!(registry.len(), 2);
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
    async fn a_source_a_join_would_refuse_is_refused_before_anything_is_written() {
        // Two rows for one entity at one event timestamp, and no `created_timestamp_field` to
        // decide between them, so which one wins would depend on the query plan. The join refuses
        // that source; a refresh has to refuse it too, or serving would quietly answer from a row
        // training would not read. The refusal is this view's rather than the refresh's: views
        // are scanned and written in declaration order, so every view declared before this one
        // has already been refreshed by the time the check runs, and what holds is that this view
        // wrote nothing.
        let source = Parquet::write(&source(&[(1, 100, 10), (1, 100, 11)]));
        let view = a_view(&source.string(), None);
        let mut store = MemoryStore::new();

        let refused = materialize(
            &mut store,
            &an_engine(),
            "ads",
            std::slice::from_ref(&view),
            &[],
        )
        .await;

        assert!(
            matches!(refused, Err(Error::AmbiguousTimestamp { .. })),
            "expected an ambiguous source to be refused, got {refused:?}"
        );
        assert_eq!(
            store.hash_count(),
            0,
            "the refusal comes before this view's first write, so nothing is stored"
        );
    }

    #[tokio::test]
    async fn a_created_timestamp_breaks_the_tie_and_the_newer_creation_wins() {
        // The same two rows, with a created timestamp, which is the rule the join uses for the
        // same ambiguity.
        let source = Parquet::write(&source_with_created(&[
            (1, 100, 5, 10),
            (1, 100, 9, 11),
            (1, 50, 1, 7),
        ]));
        let mut view = a_view(&source.string(), None);
        view.created_timestamp_field = Some("created_at".to_owned());

        let mut store = MemoryStore::new();
        materialize(
            &mut store,
            &an_engine(),
            "ads",
            std::slice::from_ref(&view),
            &[],
        )
        .await
        .expect("materialize");

        assert_eq!(
            stored_count(&store, &view, b"1"),
            Some(11),
            "the later creation wins"
        );
        assert_eq!(stored_freshness(&store, b"1"), Some(100));
    }

    #[tokio::test]
    async fn rows_the_created_timestamp_cannot_separate_are_refused_too() {
        // Three rows for one entity at one event timestamp that also share the created
        // timestamp. The reducer's tie-break has run out of columns here, so its `row_number()`
        // would pick whichever row the plan produced, and a refresh would store that as the
        // entity's current value. The join refuses this source, so the refresh does too.
        let source = Parquet::write(&source_with_created(&[
            (1, 100, 5, 10),
            (1, 100, 5, 11),
            (1, 100, 5, 12),
        ]));
        let mut view = a_view(&source.string(), None);
        view.created_timestamp_field = Some("created_at".to_owned());

        let mut store = MemoryStore::new();
        let refused = materialize(
            &mut store,
            &an_engine(),
            "ads",
            std::slice::from_ref(&view),
            &[],
        )
        .await;

        assert!(
            matches!(
                refused,
                Err(Error::AmbiguousTimestamp {
                    rows: 3,
                    ref tie_break,
                    ..
                }) if tie_break == "tied on `created_at`"
            ),
            "expected a tie no created timestamp breaks to be refused, got {refused:?}"
        );
        assert_eq!(store.hash_count(), 0, "nothing is stored");
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
        assert_eq!(
            store.hash_count(),
            0,
            "a rejected refresh writes nothing, not even the registry"
        );
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
