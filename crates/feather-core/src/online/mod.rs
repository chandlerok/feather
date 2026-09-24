//! Online storage and the read path.
//!
//! The store trait is deliberately narrow: it moves opaque field values in and
//! out of hashes. Encoding, TTL, and missingness live above it, so the same
//! logic runs against the in-memory store and against Valkey.
//!
//! There is no cache in the store. L1 belongs in the serving layer where the
//! invalidation stream can reach it.

pub mod memory;

#[cfg(feature = "valkey")]
pub mod valkey;

use std::collections::BTreeMap;
use std::time::Duration;

use arrow::array::ArrayRef;

use crate::definitions::FeatureView;
use crate::error::{Error, Result};
use crate::key::{freshness_field, value_field};
use crate::value::{SchemaTag, decode_batch};

/// One `HSET`: a hash key and the fields to write into it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteBatch {
    pub key: Vec<u8>,
    pub fields: Vec<(String, Vec<u8>)>,
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
    use crate::definitions::{DType, Entity, Field, FileSource};
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
            source: FileSource::new("data/x.parquet"),
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

    async fn seed(
        store: &mut MemoryStore,
        project: &str,
        view: &FeatureView,
        entity: &[u8],
        value: i64,
        freshness: i64,
    ) {
        let key = crate::key::entity_hash_key(project, "user_id", entity);
        store
            .write(&[WriteBatch {
                key,
                fields: vec![
                    (value_field(&view.name), encoded_vector(view, value)),
                    (freshness_field(&view.name), encode_freshness(freshness)),
                ],
            }])
            .await
            .unwrap();
    }

    fn request(view_name: &str, entity: &[u8]) -> EntityRequest {
        EntityRequest {
            encoded_key: entity.to_vec(),
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
        seed(&mut store, "ads", view, b"u1", 42, NOW).await;

        let out = read_entities(&store, "ads", &views, &[request("clicks", b"u1")], NOW)
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
            b"u1",
            1,
            NOW - 2 * 86_400 * 1_000_000,
        )
        .await;

        let out = read_entities(&store, "ads", &views, &[request("clicks", b"u1")], NOW)
            .await
            .unwrap();
        assert!(matches!(out[0][0], ViewValues::Missing(Missing::Expired)));
    }

    #[tokio::test]
    async fn a_never_written_entity_is_missing() {
        let views = views(&[("clicks", Some(30))]);
        let store = MemoryStore::new();
        let out = read_entities(&store, "ads", &views, &[request("clicks", b"nobody")], NOW)
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
        seed(&mut store, "ads", old, b"u1", 7, NOW).await;

        // The definition changes dtype; the stored tag no longer matches.
        let mut changed = views.clone();
        changed.get_mut("clicks").unwrap().features = vec![Field::new("count", DType::Float64)];

        let out = read_entities(&store, "ads", &changed, &[request("clicks", b"u1")], NOW)
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
            seed(&mut inner, "ads", many.get(name).unwrap(), b"u1", 1, NOW).await;
        }

        let reads = Arc::new(AtomicUsize::new(0));
        let store = CountingStore {
            inner,
            reads: Arc::clone(&reads),
        };

        let mut entity = request("a", b"u1");
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
            encoded_key: b"u1".to_vec(),
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
    async fn a_retired_view_can_be_deleted() {
        let views = views(&[("clicks", None)]);
        let view = views.get("clicks").unwrap();
        let mut store = MemoryStore::new();
        seed(&mut store, "ads", view, b"u1", 1, NOW).await;
        assert_eq!(store.hash_count(), 1);

        let key = crate::key::entity_hash_key("ads", "user_id", b"u1");
        store
            .delete_fields(&[(key, vec![value_field("clicks")])])
            .await
            .unwrap();
        assert!(
            !store
                .fields(&crate::key::entity_hash_key("ads", "user_id", b"u1"))
                .unwrap()
                .contains_key(&value_field("clicks"))
        );
    }
}
