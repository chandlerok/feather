//! Online storage and the read path.
//!
//! The store trait is deliberately narrow: it moves opaque field values in and
//! out of hashes. Encoding, TTL, and missingness live above it, so the same
//! logic runs against the in-process map and against the embedded LSM.
//!
//! There is no cache in the store.

pub mod memory;

#[cfg(feature = "fjall")]
pub mod fjall;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::new_null_array;
use arrow::array::{Array, ArrayRef, UInt32Array};
use arrow::compute::{concat, take};
use arrow::datatypes::{Field as ArrowField, Schema};
use arrow::record_batch::RecordBatch;

use crate::definitions::{FeatureView, Reference};
use crate::error::{Error, Result};
use crate::key::{freshness_field, value_field};
use crate::value::{SchemaTag, arrow_type, decode_batch};

/// One field of a write: its value, and when it stops being readable.
///
/// The expiry rides with the value rather than in a call of its own because the two belong
/// together: a field written without the expiry that is meant to reclaim it is a field the
/// server keeps until its hash is rewritten or deleted. The unit is named in the field rather
/// than left to the reader, because a bare `i64` expiry inside a codec that otherwise counts
/// microseconds is a bug waiting to happen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenField {
    pub name: String,
    pub value: Vec<u8>,
    /// Absolute expiry as seconds since the Unix epoch, or `None` for a field that never
    /// expires.
    ///
    /// Absolute rather than a TTL: a value that is already older than its TTL has to expire at
    /// once, and a relative TTL measured from the write would give it a second lease on life.
    pub expires_at_unix_secs: Option<i64>,
}

impl WrittenField {
    /// A field, with an absolute expiry when it has one.
    ///
    /// Args:
    ///     name: The field name.
    ///     value: The encoded value.
    ///     expires_at_unix_secs: When the field stops being readable, or `None` to leave it to
    ///         the read-time check alone.
    ///
    /// Returns:
    ///     The field.
    pub fn new(name: impl Into<String>, value: Vec<u8>, expires_at_unix_secs: Option<i64>) -> Self {
        Self {
            name: name.into(),
            value,
            expires_at_unix_secs,
        }
    }
}

/// One write: a hash key and the fields to set in it.
///
/// "Set" rather than "insert", because a field already in the hash is overwritten. That is
/// what makes a full refresh safe to re-run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteBatch {
    pub key: Vec<u8>,
    pub fields: Vec<WrittenField>,
}

/// One `HMGET`: a hash key and the fields to read from it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadRequest {
    pub key: Vec<u8>,
    pub fields: Vec<String>,
}

/// A minimal hash store. Async because the real implementation is, and because
/// pretending otherwise would make the serving path look cheaper than it is.
#[allow(async_fn_in_trait)]
pub trait OnlineStore {
    /// Write fields into hashes. Fields not mentioned are left alone, which is
    /// what makes a full refresh safe to re-run.
    ///
    /// A field may carry an absolute expiry. A store that cannot honour one writes the value
    /// anyway: the read-time TTL check in [`read_entities`] is what decides whether a value is
    /// served, so an unexpired leftover costs reclamation and never correctness.
    async fn write(&mut self, batches: &[WriteBatch]) -> Result<()>;

    /// Read fields from hashes, one `HMGET` per request, in request order.
    ///
    /// The outer `Vec` is per request and the inner is per requested field, in
    /// the order requested. A field that is absent yields `None`, which is how a
    /// TTL-expired or never-written value arrives here.
    async fn read(&self, requests: &[ReadRequest]) -> Result<Vec<Vec<Option<Vec<u8>>>>>;

    /// Remove fields, used when a view is retired.
    async fn delete_fields(&mut self, keys_and_fields: &[(Vec<u8>, Vec<String>)]) -> Result<()>;
}

/// The reads that look at a project's whole keyspace rather than at one hash in it.
///
/// Deliberately separate from [`OnlineStore`] rather than a method on it. `read_entities` bounds
/// its store by that trait, and the serving path has no business reaching for a keyspace walk:
/// the narrowness of that trait is what keeps the cost of a read explicable. A refresh is the
/// only caller.
#[allow(async_fn_in_trait)]
pub trait ProjectScan {
    /// Every field of one hash, as `(name, value)` pairs, and empty when there is no such key.
    ///
    /// A whole hash rather than named fields, which is what [`OnlineStore::read`] takes, because
    /// the project registry is read to learn which fields it holds. Asking for them one at a
    /// time would mean knowing the answer in advance.
    ///
    /// Args:
    ///     key: The hash key.
    ///
    /// Returns:
    ///     Each field with its value. An absent key yields an empty list rather than an error,
    ///     which is the first refresh of a project.
    async fn hash_fields(&self, key: &[u8]) -> Result<Vec<(String, Vec<u8>)>>;

    /// Every key in a project's namespace except `exclude`, which is compared exactly.
    ///
    /// Used to find the entity hashes a retired view's fields may still sit in. An entity that
    /// the retired view used to cover does not have to appear in the source any more, so there
    /// is no way to enumerate them from the declarations: the keyspace itself is the only
    /// complete list.
    ///
    /// Args:
    ///     project: The project whose keys are wanted.
    ///     exclude: A key to leave out. The caller names the project registry, which lives
    ///         under the same prefix and is not an entity hash.
    ///
    /// Returns:
    ///     The keys, in no particular order. The caller filters them further; this is a key
    ///     listing, not a promise that every key in it is an entity hash.
    async fn scan_entity_keys(&self, project: &str, exclude: &[u8]) -> Result<Vec<Vec<u8>>>;
}

/// The key prefix every key of a project shares.
///
/// One spelling of the prefix, so a walk and a filter cannot disagree about where a project's
/// keys start.
pub fn project_key_prefix(project: &str) -> Vec<u8> {
    let mut prefix = Vec::with_capacity(project.len() + 1);
    prefix.extend_from_slice(project.as_bytes());
    prefix.push(b':');
    prefix
}

/// A project name escaped for a `SCAN` pattern.
///
/// `MATCH` reads `*`, `?`, `[` and `\` as pattern syntax, so a project named `ads*` would
/// otherwise walk another project's keys. Backslash is the escape for all four.
pub fn glob_escape(project: &str) -> String {
    let mut escaped = String::with_capacity(project.len());
    for character in project.chars() {
        if matches!(character, '*' | '?' | '[' | ']' | '\\') {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

/// Why a view's values are not usable for an entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Missing {
    /// The freshness field is absent, so the view was never written for this
    /// entity.
    NeverWritten,
    /// Freshness is present but older than the view's TTL.
    Expired,
    /// A value carried a schema tag that does not match the current definition.
    SchemaMismatch,
}

impl Missing {
    /// The reason, as it appears in the response state mask.
    pub const fn as_str(self) -> &'static str {
        match self {
            Missing::NeverWritten => "missing",
            Missing::Expired => "expired",
            Missing::SchemaMismatch => "schema_mismatch",
        }
    }
}

/// What a read produced for one view of one entity.
#[derive(Debug)]
pub enum ViewValues {
    Present {
        columns: Vec<ArrayRef>,
        freshness_micros: i64,
    },
    Missing(Missing),
}

/// One view's worth of a read request.
#[derive(Debug, Clone)]
pub struct ViewRequest {
    pub view: String,
    /// Feature names, in the order the caller wants them back.
    pub fields: Vec<String>,
}

/// One entity's worth of a read request.
#[derive(Debug, Clone)]
pub struct EntityRequest {
    /// The encoded entity key, as produced by [`crate::key::encode_entity_key`].
    pub encoded_key: Vec<u8>,
    pub views: Vec<ViewRequest>,
}

/// Read a batch of entities.
///
/// Issues exactly one store read per entity regardless of how many views are
/// requested. That is the rule the whole layout exists to make possible: Feast
/// issues one call per feature view, and a user with 11 views reported "abysmal"
/// retrieval times (feast-dev/feast#3596).
///
/// One field per view is requested, holding the view's whole encoded vector, and
/// the caller's requested features are projected out after decoding. Reading a
/// subset therefore costs decode time for the columns it did not ask for, which
/// is the trade recorded on [`crate::key::value_field`].
///
/// `now_micros` is a parameter rather than a clock read so TTL behaviour is
/// testable without sleeping.
pub async fn read_entities<S: OnlineStore>(
    store: &S,
    project: &str,
    views: &BTreeMap<String, FeatureView>,
    entities: &[EntityRequest],
    now_micros: i64,
) -> Result<Vec<Vec<ViewValues>>> {
    /// Where one view's value and freshness sit in the flat field list, and which
    /// of the view's columns the caller actually asked for.
    struct Plan<'a> {
        view: &'a FeatureView,
        /// Output column order: index into the view's `features`.
        requested: Vec<usize>,
        value_at: usize,
        freshness_at: usize,
    }

    let mut requests = Vec::with_capacity(entities.len());
    let mut plans: Vec<Vec<Plan<'_>>> = Vec::with_capacity(entities.len());

    for entity in entities {
        let mut fields = Vec::new();
        let mut per_view = Vec::with_capacity(entity.views.len());

        for request in &entity.views {
            let view = views
                .get(&request.view)
                .ok_or_else(|| Error::UnknownView(request.view.clone()))?;

            let mut requested = Vec::with_capacity(request.fields.len());
            for name in &request.fields {
                let index = view
                    .features
                    .iter()
                    .position(|f| &f.name == name)
                    .ok_or_else(|| Error::UnknownFeature {
                        view: view.name.clone(),
                        name: name.clone(),
                    })?;
                requested.push(index);
            }

            let value_at = fields.len();
            fields.push(value_field(&view.name));
            let freshness_at = fields.len();
            fields.push(freshness_field(&view.name));

            per_view.push(Plan {
                view,
                requested,
                value_at,
                freshness_at,
            });
        }

        requests.push(ReadRequest {
            key: crate::key::entity_hash_key(
                project,
                shared_entity_name(&entity.views, views)?,
                &entity.encoded_key,
            ),
            fields,
        });
        plans.push(per_view);
    }

    let raw = store.read(&requests).await?;

    let mut out = Vec::with_capacity(entities.len());
    for (entity_index, per_view) in plans.iter().enumerate() {
        let values = &raw[entity_index];
        let mut results = Vec::with_capacity(per_view.len());

        for plan in per_view {
            let view = plan.view;
            let tag = SchemaTag::of(&view.features);

            let Some(freshness_bytes) = values.get(plan.freshness_at).and_then(|v| v.as_deref())
            else {
                results.push(ViewValues::Missing(Missing::NeverWritten));
                continue;
            };
            let freshness_micros = read_micros(freshness_bytes)?;

            if let Some(ttl) = view.ttl_days {
                let ttl_micros = i64::from(ttl) * 86_400 * 1_000_000;
                if now_micros.saturating_sub(freshness_micros) > ttl_micros {
                    results.push(ViewValues::Missing(Missing::Expired));
                    continue;
                }
            }

            let Some(blob) = values.get(plan.value_at).and_then(|v| v.as_deref()) else {
                results.push(ViewValues::Missing(Missing::NeverWritten));
                continue;
            };

            match decode_batch(&view.features, tag, &[blob]) {
                Ok(all_columns) => {
                    let columns = plan
                        .requested
                        .iter()
                        .map(|i| all_columns[*i].clone())
                        .collect();
                    results.push(ViewValues::Present {
                        columns,
                        freshness_micros,
                    });
                }
                // A tag mismatch is the documented null window during a dtype
                // change, not an error. Anything else is real.
                Err(Error::SchemaTagMismatch { .. }) => {
                    results.push(ViewValues::Missing(Missing::SchemaMismatch));
                }
                Err(other) => return Err(other),
            }
        }

        out.push(results);
    }

    Ok(out)
}

/// Assemble the online read's result table.
///
/// The entity frame's own columns come back untouched, one per input row, followed
/// by one column per requested feature in request order. A value that is missing,
/// expired, or stored under a schema the definition no longer matches reaches the
/// caller as a null, and the three are not distinguished here, which is the
/// documented online contract.
///
/// This lives in the core so the reshaping is one implementation rather than one
/// per binding: the index-into-present-or-null step and the column naming are the
/// parts a second binding would otherwise write twice and let drift.
///
/// Args:
///     entity: The entity frame, already read out of whatever the caller passed.
///     references: The parsed request, in the order the columns should come back.
///     views: The views the references resolved to, keyed by view name. Each is
///         looked up for the declared dtype, which is what a null column is typed by.
///     slots: One `(view, feature)` index pair per reference, into `values`,
///         which is the position `read_entities` wrote it to. A length that
///         does not match `references` is an error, not a shorter table.
///     values: What `read_entities` produced, one inner vector per entity, each
///         holding one [`ViewValues`] per distinct view in first-appearance order.
///
/// Returns:
///     The entity frame's columns followed by one column per requested feature.
pub fn assemble_online_result(
    entity: &RecordBatch,
    references: &[Reference],
    views: &BTreeMap<String, FeatureView>,
    slots: &[(usize, usize)],
    values: &[Vec<ViewValues>],
) -> Result<RecordBatch> {
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(entity.num_columns() + references.len());
    let mut fields: Vec<ArrowField> = Vec::with_capacity(entity.num_columns() + references.len());
    for (index, field) in entity.schema().fields().iter().enumerate() {
        columns.push(entity.column(index).clone());
        fields.push(field.as_ref().clone());
    }

    // `zip` stops at the shorter of the two lists, so a caller that supplied one
    // slot per view rather than per reference would get a table missing its
    // trailing columns, with nothing to report. The position that has no
    // counterpart is named, in whichever list ran out.
    if references.len() != slots.len() {
        return Err(Error::RowOutOfRange {
            row: references.len().max(slots.len()),
            len: references.len().min(slots.len()),
        });
    }

    for (reference, (view_slot, feature_slot)) in references.iter().zip(slots) {
        let declared = views
            .get(&reference.view)
            .ok_or_else(|| Error::UnknownView(reference.view.clone()))?
            .field(&reference.feature)
            .ok_or_else(|| Error::UnknownFeature {
                view: reference.view.clone(),
                name: reference.feature.clone(),
            })?;
        let dtype = arrow_type(declared.dtype);

        // One element per entity, in entity order, pointing at that entity's value
        // among the ones that have one or at a null for one that does not. `take`
        // turns a null index into a null of the right type, which is how an absent
        // value reaches the caller.
        let mut present: Vec<ArrayRef> = Vec::new();
        let mut indices: Vec<Option<u32>> = Vec::with_capacity(values.len());
        for per_view in values {
            let value = per_view.get(*view_slot).ok_or(Error::RowOutOfRange {
                row: *view_slot,
                len: per_view.len(),
            })?;
            match value {
                ViewValues::Present {
                    columns: decoded, ..
                } => {
                    let column = decoded.get(*feature_slot).ok_or(Error::RowOutOfRange {
                        row: *feature_slot,
                        len: decoded.len(),
                    })?;
                    indices.push(Some(present.len() as u32));
                    present.push(column.clone());
                }
                ViewValues::Missing(_) => indices.push(None),
            }
        }

        let column = if present.is_empty() {
            new_null_array(&dtype, values.len())
        } else {
            let borrowed: Vec<&dyn Array> = present.iter().map(|array| array.as_ref()).collect();
            let decoded = concat(&borrowed)?;
            take(&decoded, &UInt32Array::from(indices), None)?
        };
        columns.push(column);
        fields.push(ArrowField::new(&reference.feature, dtype, true));
    }

    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

/// v1 supports exactly one entity per view, and a single request must resolve to a
/// single entity name because that name is part of the hash key.
fn shared_entity_name<'a>(
    requests: &[ViewRequest],
    views: &'a BTreeMap<String, FeatureView>,
) -> Result<&'a str> {
    let mut resolved: Option<&'a str> = None;
    for request in requests {
        let view = views
            .get(&request.view)
            .ok_or_else(|| Error::UnknownView(request.view.clone()))?;
        let name = view.entity()?.name.as_str();
        match resolved {
            None => resolved = Some(name),
            Some(existing) if existing == name => {}
            Some(existing) => {
                return Err(Error::MixedEntities {
                    first: existing.to_owned(),
                    second: name.to_owned(),
                });
            }
        }
    }
    resolved.ok_or_else(|| Error::MalformedFeatureReference {
        reference: "(no views requested)".to_owned(),
    })
}

fn read_micros(bytes: &[u8]) -> Result<i64> {
    let array: [u8; 8] = bytes.try_into().map_err(|_| Error::MalformedValue {
        field: "freshness".to_owned(),
        reason: format!("expected 8 bytes, got {}", bytes.len()),
    })?;
    Ok(i64::from_le_bytes(array))
}

/// Encode a freshness timestamp the way the read path expects it.
pub fn encode_freshness(micros: i64) -> Vec<u8> {
    micros.to_le_bytes().to_vec()
}

/// Convert a TTL in days to a duration.
pub fn ttl_duration(ttl_days: u32) -> Duration {
    Duration::from_secs(u64::from(ttl_days) * 86_400)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definitions::{DType, Entity, Field, Source, parse_references};
    use crate::online::memory::MemoryStore;
    use crate::value::encode_batch;
    use arrow::array::{ArrayRef, Int64Array};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const NOW: i64 = 1_700_000_000_000_000;

    fn view(name: &str, ttl_days: Option<u32>) -> FeatureView {
        FeatureView {
            name: name.to_owned(),
            entities: vec![Entity::new("user_id", "user_id")],
            source: Source::file("data/x.parquet"),
            features: vec![Field::new("count", DType::Int64)],
            ttl_days,
            timestamp_field: None,
            created_timestamp_field: None,
        }
    }

    fn views(names: &[(&str, Option<u32>)]) -> BTreeMap<String, FeatureView> {
        names
            .iter()
            .map(|(n, ttl)| ((*n).to_owned(), view(n, *ttl)))
            .collect()
    }

    fn encoded_vector(tag_source: &FeatureView, value: i64) -> Vec<u8> {
        let columns: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![value]))];
        encode_batch(&tag_source.features, &columns, 0..1)
            .unwrap()
            .row(0)
            .unwrap()
            .to_vec()
    }

    /// An entity key as the read path is given it: the encoded form the binding builds, not the
    /// raw component. A raw component is a key the encoder would never produce, so it is also a
    /// key garbage collection skips, and a fixture built from one cannot show what a real key
    /// does.
    fn encoded_key(value: &str) -> Vec<u8> {
        crate::key::encode_entity_key(&[value.as_bytes()]).expect("encode the key")
    }

    /// `seed` and `request` take the join key value and encode it, so the hash a fixture writes
    /// and the hash a read looks up cannot be spelled differently.
    async fn seed(
        store: &mut MemoryStore,
        project: &str,
        view: &FeatureView,
        entity: &str,
        value: i64,
        freshness: i64,
    ) {
        let key = crate::key::entity_hash_key(project, "user_id", &encoded_key(entity));
        store
            .write(&[WriteBatch {
                key,
                fields: vec![
                    WrittenField::new(value_field(&view.name), encoded_vector(view, value), None),
                    WrittenField::new(
                        freshness_field(&view.name),
                        encode_freshness(freshness),
                        None,
                    ),
                ],
            }])
            .await
            .unwrap();
    }

    fn request(view_name: &str, entity: &str) -> EntityRequest {
        EntityRequest {
            encoded_key: encoded_key(entity),
            views: vec![ViewRequest {
                view: view_name.to_owned(),
                fields: vec!["count".to_owned()],
            }],
        }
    }

    #[tokio::test]
    async fn reads_back_a_written_vector() {
        let views = views(&[("clicks", Some(30))]);
        let view = views.get("clicks").unwrap();
        let mut store = MemoryStore::new();
        seed(&mut store, "ads", view, "u1", 42, NOW).await;

        let out = read_entities(&store, "ads", &views, &[request("clicks", "u1")], NOW)
            .await
            .unwrap();
        match &out[0][0] {
            ViewValues::Present {
                columns,
                freshness_micros,
            } => {
                assert_eq!(*freshness_micros, NOW);
                let counts = columns[0].as_any().downcast_ref::<Int64Array>().unwrap();
                assert_eq!(counts.value(0), 42);
            }
            other => panic!("expected Present, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn an_expired_value_is_missing_not_stale() {
        let views = views(&[("clicks", Some(1))]);
        let view = views.get("clicks").unwrap();
        let mut store = MemoryStore::new();
        // Written two days ago against a one-day TTL.
        seed(
            &mut store,
            "ads",
            view,
            "u1",
            1,
            NOW - 2 * 86_400 * 1_000_000,
        )
        .await;

        let out = read_entities(&store, "ads", &views, &[request("clicks", "u1")], NOW)
            .await
            .unwrap();
        assert!(matches!(out[0][0], ViewValues::Missing(Missing::Expired)));
    }

    #[tokio::test]
    async fn a_never_written_entity_is_missing() {
        let views = views(&[("clicks", Some(30))]);
        let store = MemoryStore::new();
        let out = read_entities(&store, "ads", &views, &[request("clicks", "nobody")], NOW)
            .await
            .unwrap();
        assert!(matches!(
            out[0][0],
            ViewValues::Missing(Missing::NeverWritten)
        ));
    }

    #[tokio::test]
    async fn a_schema_change_reads_as_missing_rather_than_garbage() {
        let views = views(&[("clicks", Some(30))]);
        let old = views.get("clicks").unwrap();
        let mut store = MemoryStore::new();
        seed(&mut store, "ads", old, "u1", 7, NOW).await;

        // The definition changes dtype; the stored tag no longer matches.
        let mut changed = views.clone();
        changed.get_mut("clicks").unwrap().features = vec![Field::new("count", DType::Float64)];

        let out = read_entities(&store, "ads", &changed, &[request("clicks", "u1")], NOW)
            .await
            .unwrap();
        assert!(matches!(
            out[0][0],
            ViewValues::Missing(Missing::SchemaMismatch)
        ));
    }

    /// Counts store reads, to prove the one-read-per-entity rule holds.
    struct CountingStore {
        inner: MemoryStore,
        reads: Arc<AtomicUsize>,
    }

    impl OnlineStore for CountingStore {
        async fn write(&mut self, batches: &[WriteBatch]) -> Result<()> {
            self.inner.write(batches).await
        }

        async fn read(&self, requests: &[ReadRequest]) -> Result<Vec<Vec<Option<Vec<u8>>>>> {
            self.reads.fetch_add(requests.len(), Ordering::SeqCst);
            self.inner.read(requests).await
        }

        async fn delete_fields(&mut self, keys: &[(Vec<u8>, Vec<String>)]) -> Result<()> {
            self.inner.delete_fields(keys).await
        }
    }

    #[tokio::test]
    async fn reads_once_per_entity_however_many_views() {
        let many = views(&[
            ("a", Some(30)),
            ("b", Some(30)),
            ("c", Some(30)),
            ("d", Some(30)),
        ]);
        let mut inner = MemoryStore::new();
        for name in ["a", "b", "c", "d"] {
            seed(&mut inner, "ads", many.get(name).unwrap(), "u1", 1, NOW).await;
        }

        let reads = Arc::new(AtomicUsize::new(0));
        let store = CountingStore {
            inner,
            reads: Arc::clone(&reads),
        };

        let mut entity = request("a", "u1");
        for name in ["b", "c", "d"] {
            entity.views.push(ViewRequest {
                view: name.to_owned(),
                fields: vec!["count".to_owned()],
            });
        }

        let out = read_entities(&store, "ads", &many, &[entity], NOW)
            .await
            .unwrap();
        assert_eq!(out[0].len(), 4);
        assert_eq!(
            reads.load(Ordering::SeqCst),
            1,
            "four views on one entity must still be one store read"
        );
    }

    #[tokio::test]
    async fn views_disagreeing_on_their_entity_are_rejected() {
        let mut many = views(&[("a", None), ("b", None)]);
        many.get_mut("b").unwrap().entities = vec![Entity::new("item_id", "item_id")];
        let store = MemoryStore::new();

        let entity = EntityRequest {
            encoded_key: encoded_key("u1"),
            views: vec![
                ViewRequest {
                    view: "a".to_owned(),
                    fields: vec!["count".to_owned()],
                },
                ViewRequest {
                    view: "b".to_owned(),
                    fields: vec!["count".to_owned()],
                },
            ],
        };

        assert!(matches!(
            read_entities(&store, "ads", &many, &[entity], NOW).await,
            Err(Error::MixedEntities { .. })
        ));
    }

    #[tokio::test]
    async fn a_keyspace_scan_returns_the_entity_hashes_and_not_the_registry() {
        use crate::key::views_registry_key;
        let mut store = MemoryStore::new();
        let registry = views_registry_key("ads");
        let mut keys = vec![registry.clone()];
        for entity in ["u1", "u2"] {
            let key = crate::key::entity_hash_key("ads", "user_id", &encoded_key(entity));
            store
                .write(&[WriteBatch {
                    key: key.clone(),
                    fields: vec![WrittenField::new(
                        crate::key::value_field("clicks"),
                        vec![1],
                        None,
                    )],
                }])
                .await
                .unwrap();
            keys.push(key);
        }
        store
            .write(&[WriteBatch {
                key: registry.clone(),
                fields: vec![WrittenField::new("clicks", vec![2], None)],
            }])
            .await
            .unwrap();

        let mut scanned = store.scan_entity_keys("ads", &registry).await.unwrap();
        scanned.sort();
        keys.remove(0);
        keys.sort();
        assert_eq!(scanned, keys);

        // Another project shares no prefix with this one.
        assert!(
            store
                .scan_entity_keys("other", b"other:views")
                .await
                .unwrap()
                .is_empty()
        );

        let fields = store.hash_fields(&registry).await.unwrap();
        assert_eq!(fields, vec![("clicks".to_owned(), vec![2])]);
    }

    #[tokio::test]
    async fn a_retired_view_can_be_deleted() {
        let views = views(&[("clicks", None)]);
        let view = views.get("clicks").unwrap();
        let mut store = MemoryStore::new();
        seed(&mut store, "ads", view, "u1", 1, NOW).await;
        assert_eq!(store.hash_count(), 1);

        let key = crate::key::entity_hash_key("ads", "user_id", &encoded_key("u1"));
        store
            .delete_fields(&[(key, vec![value_field("clicks")])])
            .await
            .unwrap();
        assert!(
            !store
                .fields(&crate::key::entity_hash_key(
                    "ads",
                    "user_id",
                    &encoded_key("u1"),
                ))
                .unwrap()
                .contains_key(&value_field("clicks"))
        );
    }

    /// One entity frame row, so a fixture can vary which entity is present.
    fn entity_frame(keys: &[i64]) -> RecordBatch {
        let columns: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(keys.to_vec()))];
        let schema = Arc::new(Schema::new(vec![ArrowField::new(
            "user_id",
            arrow::datatypes::DataType::Int64,
            false,
        )]));
        RecordBatch::try_new(schema, columns).expect("entity frame")
    }

    /// A present value of one Int64 column, which is what the assembly reads out of.
    fn present(value: i64) -> ViewValues {
        ViewValues::Present {
            columns: vec![Arc::new(Int64Array::from(vec![value]))],
            freshness_micros: NOW,
        }
    }

    #[test]
    fn a_present_value_reaches_its_column() {
        let entity = entity_frame(&[1, 2]);
        let views = views(&[("clicks", None)]);
        let references = parse_references(&["clicks:count"]).expect("valid");
        // One view slot, one feature slot, and a value present for both entities.
        let slots = vec![(0, 0)];
        let values = vec![vec![present(10)], vec![present(20)]];

        let out = assemble_online_result(&entity, &references, &views, &slots, &values)
            .expect("assembled");

        assert_eq!(out.num_rows(), 2);
        let count = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(count.values(), &[10, 20]);
        assert_eq!(out.schema().field(1).name(), "count");
    }

    #[test]
    fn a_missing_value_becomes_a_null_of_the_declared_type() {
        let entity = entity_frame(&[1, 2]);
        let views = views(&[("clicks", None)]);
        let references = parse_references(&["clicks:count"]).expect("valid");
        let slots = vec![(0, 0)];
        // The second entity never wrote the view, so its value is absent.
        let values = vec![
            vec![present(10)],
            vec![ViewValues::Missing(Missing::NeverWritten)],
        ];

        let out = assemble_online_result(&entity, &references, &views, &slots, &values)
            .expect("assembled");

        let count = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        // The first entity's value is untouched; the second is a null rather than
        // a zero, so a caller can tell an absent value from a real one.
        assert!(!count.is_null(0));
        assert_eq!(count.value(0), 10);
        assert!(count.is_null(1));
    }

    #[test]
    fn an_all_missing_view_produces_an_all_null_column() {
        let entity = entity_frame(&[1, 2, 3]);
        let views = views(&[("clicks", None)]);
        let references = parse_references(&["clicks:count"]).expect("valid");
        let slots = vec![(0, 0)];
        let values = vec![
            vec![ViewValues::Missing(Missing::NeverWritten)],
            vec![ViewValues::Missing(Missing::Expired)],
            vec![ViewValues::Missing(Missing::SchemaMismatch)],
        ];

        let out = assemble_online_result(&entity, &references, &views, &slots, &values)
            .expect("assembled");

        let count = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(count.null_count(), 3);
    }

    #[test]
    fn the_entity_columns_come_back_first_and_untouched() {
        let entity = entity_frame(&[7, 8]);
        let views = views(&[("clicks", None)]);
        let references = parse_references(&["clicks:count"]).expect("valid");
        let slots = vec![(0, 0)];
        let values = vec![vec![present(1)], vec![present(2)]];

        let out = assemble_online_result(&entity, &references, &views, &slots, &values)
            .expect("assembled");

        assert_eq!(out.schema().field(0).name(), "user_id");
        let key = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(key.values(), &[7, 8]);
    }

    #[test]
    fn a_slot_per_view_rather_than_per_reference_is_refused() {
        let entity = entity_frame(&[1, 2]);
        let views = views(&[("clicks", None), ("buys", None)]);
        let references = parse_references(&["clicks:count", "buys:count"]).expect("valid");
        // One slot where the request asked for two columns. `zip` would have
        // stopped there and returned a table with the second column missing.
        let slots = vec![(0, 0)];
        let values = vec![vec![present(10)], vec![present(20)]];

        assert!(matches!(
            assemble_online_result(&entity, &references, &views, &slots, &values),
            Err(Error::RowOutOfRange { row: 2, len: 1 })
        ));
    }

    #[test]
    fn a_view_slot_outside_the_read_is_refused() {
        let entity = entity_frame(&[1, 2]);
        let views = views(&[("clicks", None)]);
        let references = parse_references(&["clicks:count"]).expect("valid");
        // The right number of slots, pointing at a view this read never made.
        let slots = vec![(1, 0)];
        let values = vec![vec![present(10)], vec![present(20)]];

        assert!(matches!(
            assemble_online_result(&entity, &references, &views, &slots, &values),
            Err(Error::RowOutOfRange { row: 1, len: 1 })
        ));
    }

    #[test]
    fn a_feature_slot_outside_the_columns_is_refused() {
        let entity = entity_frame(&[1, 2]);
        let views = views(&[("clicks", None)]);
        let references = parse_references(&["clicks:count"]).expect("valid");
        // `clicks` declares one feature, so the second column is not there.
        let slots = vec![(0, 1)];
        let values = vec![vec![present(10)], vec![present(20)]];

        assert!(matches!(
            assemble_online_result(&entity, &references, &views, &slots, &values),
            Err(Error::RowOutOfRange { row: 1, len: 1 })
        ));
    }
}
