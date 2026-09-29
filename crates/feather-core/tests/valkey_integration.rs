//! Integration tests against a real Valkey.
//!
//! Run with `mise run test:integration`, which starts the container first. These
//! are excluded from the default test run so unit tests need no container.
//!
//! The container is deliberately constrained (see docker-compose.yml): 2 CPUs,
//! 512MiB, `maxmemory` 384MiB with `allkeys-lru`. That matters here because the
//! eviction path is part of what this exercises.

#![cfg(feature = "valkey")]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use arrow::array::{ArrayRef, Int64Array};
use feather_core::definitions::{DType, Entity, FeatureView, Field, Source};
use feather_core::key::{encode_entity_key, entity_hash_key, freshness_field, value_field};
use feather_core::online::valkey::FieldExpiry;
use feather_core::online::{
    EntityRequest, FreshnessGuard, Missing, OnlineStore, ReadRequest, ViewRequest, ViewValues,
    WriteBatch, WrittenField, push_record, read_entities,
};
use feather_core::value::{SchemaTag, encode_batch};
use feather_core::{Error, Result, ValkeyStore};

fn url() -> String {
    std::env::var("VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_owned())
}

/// A unique project per test, so parallel runs and leftovers cannot collide.
fn project(tag: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("feathertest_{tag}_{nanos}")
}

fn view(name: &str, ttl_days: Option<u32>) -> FeatureView {
    FeatureView {
        name: name.to_owned(),
        entities: vec![Entity::new("user_id", "user_id")],
        source: Source::file("data/x.parquet"),
        features: vec![
            Field::new("count", DType::Int64),
            Field::new("label", DType::Utf8),
        ],
        ttl_days,
        timestamp_field: None,
        created_timestamp_field: None,
    }
}

fn views(entries: &[(&str, Option<u32>)]) -> BTreeMap<String, FeatureView> {
    entries
        .iter()
        .map(|(n, ttl)| ((*n).to_owned(), view(n, *ttl)))
        .collect()
}

fn now_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

/// Encode one entity's vector for a view.
fn encode_vector(view: &FeatureView, count: i64, label: &str) -> Vec<u8> {
    use arrow::array::StringArray;
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![count])),
        Arc::new(StringArray::from(vec![label])),
    ];
    encode_batch(&view.features, &columns, 0..1)
        .expect("encode")
        .row(0)
        .expect("row")
        .to_vec()
}

/// A write from `(name, value)` pairs, none of which expire.
///
/// These tests exercise the read path, so what they write carries no expiry; the write path's
/// own expiry is asserted against the in-memory store and in `materialize.rs`.
fn write_batch(project: &str, entity: &[u8], fields: Vec<(String, Vec<u8>)>) -> WriteBatch {
    let encoded = encode_entity_key(&[entity]).expect("encode key");
    WriteBatch::new(
        entity_hash_key(project, "user_id", &encoded),
        fields
            .into_iter()
            .map(|(name, value)| WrittenField::new(name, value, None))
            .collect(),
    )
}

/// A guarded write of one view's value and freshness, as `push_record` and a refresh both issue.
fn guarded_batch(
    project: &str,
    view: &FeatureView,
    entity: &[u8],
    count: i64,
    label: &str,
    event_micros: i64,
    expires_at_unix_secs: Option<i64>,
) -> WriteBatch {
    let encoded = encode_entity_key(&[entity]).expect("encode key");
    WriteBatch::guarded(
        entity_hash_key(project, "user_id", &encoded),
        vec![
            WrittenField::new(
                value_field(&view.name),
                encode_vector(view, count, label),
                expires_at_unix_secs,
            ),
            WrittenField::new(
                freshness_field(&view.name),
                event_micros.to_le_bytes().to_vec(),
                None,
            ),
        ],
        FreshnessGuard::for_view(&view.name, event_micros),
    )
}

/// The `count` the store holds for one entity, read back the way the serving read decodes it.
async fn stored_count(
    store: &ValkeyStore,
    project: &str,
    views: &BTreeMap<String, FeatureView>,
    entity: &[u8],
) -> i64 {
    let encoded = encode_entity_key(&[entity]).expect("encode key");
    let out = read_entities(
        store,
        project,
        views,
        &[EntityRequest {
            encoded_key: encoded,
            views: vec![ViewRequest {
                view: "clicks".to_owned(),
                fields: vec!["count".to_owned()],
            }],
        }],
        now_micros(),
    )
    .await
    .expect("read");
    match &out[0][0] {
        ViewValues::Present { columns, .. } => columns[0]
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("an int64 column")
            .value(0),
        other => panic!("expected Present, got {other:?}"),
    }
}

#[tokio::test]
async fn a_guarded_write_lands_when_nothing_is_recorded() -> Result<()> {
    let project = project("guard-first");
    let views = views(&[("clicks", Some(30))]);
    let clicks = views.get("clicks").unwrap();
    let mut store = ValkeyStore::connect(&url()).await?;

    let applied = store
        .write(&[guarded_batch(
            &project, clicks, b"u1", 42, "gold", 1000, None,
        )])
        .await?;

    assert_eq!(
        applied,
        vec![true],
        "the first guarded write has nothing to beat"
    );
    assert_eq!(stored_count(&store, &project, &views, b"u1").await, 42);
    Ok(())
}

#[tokio::test]
async fn a_guarded_write_is_refused_at_the_same_timestamp() -> Result<()> {
    // A push retried, or a refresh re-run over unchanged source rows. Refusing is what makes
    // both safe to repeat.
    let project = project("guard-equal");
    let views = views(&[("clicks", Some(30))]);
    let clicks = views.get("clicks").unwrap();
    let mut store = ValkeyStore::connect(&url()).await?;

    let first = store
        .write(&[guarded_batch(
            &project, clicks, b"u1", 42, "gold", 1000, None,
        )])
        .await?;
    let second = store
        .write(&[guarded_batch(
            &project, clicks, b"u1", 99, "silver", 1000, None,
        )])
        .await?;

    assert_eq!(first, vec![true]);
    assert_eq!(second, vec![false], "an equal timestamp is not newer");
    assert_eq!(
        stored_count(&store, &project, &views, b"u1").await,
        42,
        "a refused write leaves the stored value alone"
    );
    Ok(())
}

#[tokio::test]
async fn a_guarded_write_is_refused_behind_a_newer_one() -> Result<()> {
    let project = project("guard-older");
    let views = views(&[("clicks", Some(30))]);
    let clicks = views.get("clicks").unwrap();
    let mut store = ValkeyStore::connect(&url()).await?;

    store
        .write(&[guarded_batch(
            &project, clicks, b"u1", 42, "gold", 2000, None,
        )])
        .await?;
    let applied = store
        .write(&[guarded_batch(
            &project, clicks, b"u1", 99, "silver", 1000, None,
        )])
        .await?;

    assert_eq!(applied, vec![false]);
    assert_eq!(stored_count(&store, &project, &views, b"u1").await, 42);
    Ok(())
}

#[tokio::test]
async fn a_guarded_write_lands_ahead_of_an_older_one() -> Result<()> {
    let project = project("guard-newer");
    let views = views(&[("clicks", Some(30))]);
    let clicks = views.get("clicks").unwrap();
    let mut store = ValkeyStore::connect(&url()).await?;

    store
        .write(&[guarded_batch(
            &project, clicks, b"u1", 42, "gold", 1000, None,
        )])
        .await?;
    let applied = store
        .write(&[guarded_batch(
            &project, clicks, b"u1", 99, "silver", 2000, None,
        )])
        .await?;

    assert_eq!(applied, vec![true]);
    assert_eq!(stored_count(&store, &project, &views, b"u1").await, 99);
    Ok(())
}

#[tokio::test]
async fn one_batch_is_refused_without_holding_up_the_others() -> Result<()> {
    // A refresh writes one batch per entity, so a single refused row must not cost the whole
    // write. The replies are positional, which is what the chunked pipeline depends on.
    let project = project("guard-mixed");
    let views = views(&[("clicks", Some(30))]);
    let clicks = views.get("clicks").unwrap();
    let mut store = ValkeyStore::connect(&url()).await?;

    store
        .write(&[guarded_batch(&project, clicks, b"u1", 1, "a", 5000, None)])
        .await?;
    let applied = store
        .write(&[
            guarded_batch(&project, clicks, b"u1", 99, "stale", 1000, None),
            guarded_batch(&project, clicks, b"u2", 99, "fresh", 6000, None),
        ])
        .await?;

    assert_eq!(applied, vec![false, true]);
    assert_eq!(stored_count(&store, &project, &views, b"u1").await, 1);
    assert_eq!(stored_count(&store, &project, &views, b"u2").await, 99);
    Ok(())
}

#[tokio::test]
async fn a_guarded_write_sets_and_clears_the_servers_field_expiry() -> Result<()> {
    // The script writes a value and its expiry in one step, so it has to agree with
    // `queue_write` about both halves: set the expiry the write asked for, and let the plain
    // `HSET` clear it when a later write asks for none.
    // `rewriting_a_field_without_an_expiry_clears_the_servers_expiry` pins the same rule for the
    // unguarded path, and the two together are what keep the script from being a second,
    // divergent writer.
    const TTL_SECS: i64 = 3;

    let project = project("guard-expiry");
    let views = views(&[("clicks", None)]);
    let clicks = views.get("clicks").unwrap();
    let mut store = ValkeyStore::connect(&url()).await?;
    assert_ne!(
        store.field_expiry(),
        FieldExpiry::None,
        "this test needs a server with hash field expiration"
    );

    let event = now_micros();
    let deadline = event / 1_000_000 + TTL_SECS;
    let rewritten_key = entity_hash_key(&project, "user_id", &encode_entity_key(&[b"u1"])?);
    let control_key = entity_hash_key(&project, "user_id", &encode_entity_key(&[b"u2"])?);
    store
        .write(&[
            guarded_batch(&project, clicks, b"u1", 1, "x", event, Some(deadline)),
            // The same field with the same expiry on another entity, never rewritten, as the
            // control the reclaimed field is compared against.
            guarded_batch(&project, clicks, b"u2", 3, "z", event, Some(deadline)),
        ])
        .await?;

    // A guarded rewrite of the same timestamp, this time with no expiry, which is the shape a
    // view that drops its `ttl_days` produces. The guard lets it through because the timestamp
    // is not older than what is recorded.
    let rewritten = encode_vector(clicks, 2, "y");
    let applied = store
        .write(&[WriteBatch::guarded(
            rewritten_key.clone(),
            vec![WrittenField::new(
                value_field("clicks"),
                rewritten.clone(),
                None,
            )],
            FreshnessGuard::for_view("clicks", event),
        )])
        .await?;
    assert_eq!(
        applied,
        vec![true],
        "an equal timestamp still lands a guarded rewrite"
    );

    let live = store
        .read(&[
            ReadRequest {
                key: rewritten_key.clone(),
                fields: vec![value_field("clicks")],
            },
            ReadRequest {
                key: control_key.clone(),
                fields: vec![value_field("clicks")],
            },
        ])
        .await?;
    assert_eq!(
        live[0][0].as_deref(),
        Some(rewritten.as_slice()),
        "the guarded rewrite lands while the field is still inside its expiry"
    );
    assert!(
        live[1][0].is_some(),
        "the control is inside its expiry too, so it is still readable here"
    );

    let remaining = deadline - now_micros() / 1_000_000;
    std::thread::sleep(Duration::from_secs(remaining.max(0) as u64 + 1));

    let after = store
        .read(&[
            ReadRequest {
                key: rewritten_key,
                fields: vec![value_field("clicks")],
            },
            ReadRequest {
                key: control_key,
                fields: vec![value_field("clicks")],
            },
        ])
        .await?;
    assert!(
        after[0][0].as_deref() == Some(rewritten.as_slice()),
        "the rewrite cleared the field's TTL, so it outlives the deadline, got {:?}",
        after[0][0]
    );
    assert!(
        after[1][0].is_none(),
        "the control still carries its expiry, so the script asked for one and the server took \
         it, got {:?}",
        after[1][0]
    );

    cleanup(&mut store, &project, clicks, &[b"u1", b"u2"]).await?;
    Ok(())
}

#[tokio::test]
async fn a_push_through_the_core_lands_and_is_refused_when_stale() -> Result<()> {
    // The shape the Python surface issues, end to end through the real store.
    let project = project("push-core");
    let views = views(&[("clicks", Some(30))]);
    let clicks = views.get("clicks").unwrap();
    let mut store = ValkeyStore::connect(&url()).await?;
    use arrow::array::StringArray;
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![7])),
        Arc::new(StringArray::from(vec!["gold"])),
    ];

    let first = push_record(&mut store, &project, clicks, b"u1", &columns, 1000).await?;
    assert!(first);
    assert_eq!(stored_count(&store, &project, &views, b"u1").await, 7);

    assert!(!push_record(&mut store, &project, clicks, b"u1", &columns, 1000).await?);
    assert!(
        push_record(&mut store, &project, clicks, b"u1", &columns, 1001).await?,
        "a newer push lands"
    );
    Ok(())
}

#[tokio::test]
async fn round_trips_through_real_valkey() -> Result<()> {
    let project = project("roundtrip");
    let views = views(&[("clicks", Some(30))]);
    let clicks = views.get("clicks").unwrap();
    let mut store = ValkeyStore::connect(&url()).await?;

    let now = now_micros();
    store
        .write(&[write_batch(
            &project,
            b"u1",
            vec![
                (value_field("clicks"), encode_vector(clicks, 42, "gold")),
                (freshness_field("clicks"), now.to_le_bytes().to_vec()),
            ],
        )])
        .await?;

    let out = read_entities(
        &store,
        &project,
        &views,
        &[EntityRequest {
            encoded_key: encode_entity_key(&[b"u1"])?,
            views: vec![ViewRequest {
                view: "clicks".to_owned(),
                fields: vec!["count".to_owned(), "label".to_owned()],
            }],
        }],
        now,
    )
    .await?;

    match &out[0][0] {
        ViewValues::Present {
            columns,
            freshness_micros,
        } => {
            assert_eq!(*freshness_micros, now);
            let counts = columns[0].as_any().downcast_ref::<Int64Array>().unwrap();
            assert_eq!(counts.value(0), 42);
            let labels = columns[1]
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap();
            assert_eq!(labels.value(0), "gold");
        }
        other => panic!("expected Present, got {other:?}"),
    }

    cleanup(&mut store, &project, clicks, &[b"u1"]).await?;
    Ok(())
}

#[tokio::test]
async fn an_expired_value_reads_as_missing() -> Result<()> {
    let project = project("expiry");
    let views = views(&[("clicks", Some(1))]);
    let clicks = views.get("clicks").unwrap();
    let mut store = ValkeyStore::connect(&url()).await?;

    // Written two days ago against a one-day TTL.
    let stale = now_micros() - 2 * 86_400 * 1_000_000;
    store
        .write(&[write_batch(
            &project,
            b"u1",
            vec![
                (value_field("clicks"), encode_vector(clicks, 1, "x")),
                (freshness_field("clicks"), stale.to_le_bytes().to_vec()),
            ],
        )])
        .await?;

    let out = read_entities(
        &store,
        &project,
        &views,
        &[EntityRequest {
            encoded_key: encode_entity_key(&[b"u1"])?,
            views: vec![ViewRequest {
                view: "clicks".to_owned(),
                fields: vec!["count".to_owned()],
            }],
        }],
        now_micros(),
    )
    .await?;

    assert!(matches!(out[0][0], ViewValues::Missing(Missing::Expired)));
    cleanup(&mut store, &project, clicks, &[b"u1"]).await?;
    Ok(())
}

#[tokio::test]
async fn a_retired_view_can_be_deleted() -> Result<()> {
    let project = project("gc");
    let views = views(&[("clicks", None)]);
    let clicks = views.get("clicks").unwrap();
    let mut store = ValkeyStore::connect(&url()).await?;

    let now = now_micros();
    store
        .write(&[write_batch(
            &project,
            b"u1",
            vec![
                (value_field("clicks"), encode_vector(clicks, 5, "y")),
                (freshness_field("clicks"), now.to_le_bytes().to_vec()),
            ],
        )])
        .await?;

    cleanup(&mut store, &project, clicks, &[b"u1"]).await?;

    let key = entity_hash_key(&project, "user_id", &encode_entity_key(&[b"u1"])?);
    let after = store
        .read(&[ReadRequest {
            key,
            fields: vec![value_field("clicks")],
        }])
        .await?;
    assert!(after[0][0].is_none(), "field should be gone after HDEL");
    Ok(())
}

/// Counts reads to prove the one-read-per-entity rule survives the real client.
struct Counting {
    inner: ValkeyStore,
    reads: Arc<AtomicUsize>,
}

impl OnlineStore for Counting {
    async fn write(&mut self, batches: &[WriteBatch]) -> Result<Vec<bool>> {
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
async fn one_round_trip_serves_many_views() -> Result<()> {
    let project = project("manyviews");
    let names = ["a", "b", "c", "d", "e"];
    let views = views(&names.map(|n| (n, None)));
    let mut inner = ValkeyStore::connect(&url()).await?;

    let now = now_micros();
    let mut batches = Vec::new();
    for name in names {
        let view = views.get(name).unwrap();
        batches.push(write_batch(
            &project,
            b"u1",
            vec![
                (value_field(name), encode_vector(view, 1, name)),
                (freshness_field(name), now.to_le_bytes().to_vec()),
            ],
        ));
    }
    inner.write(&batches).await?;

    let reads = Arc::new(AtomicUsize::new(0));
    let store = Counting {
        inner,
        reads: Arc::clone(&reads),
    };

    let entity = EntityRequest {
        encoded_key: encode_entity_key(&[b"u1"])?,
        views: names
            .iter()
            .map(|n| ViewRequest {
                view: (*n).to_owned(),
                fields: vec!["count".to_owned()],
            })
            .collect(),
    };

    let out = read_entities(&store, &project, &views, &[entity], now).await?;
    assert_eq!(out[0].len(), names.len());
    assert_eq!(
        reads.load(Ordering::SeqCst),
        1,
        "five views on one entity must be one store read"
    );

    let mut store = store;
    for name in names {
        cleanup(&mut store, &project, views.get(name).unwrap(), &[b"u1"]).await?;
    }
    Ok(())
}

#[tokio::test]
async fn an_unknown_entity_is_missing_not_an_error() -> Result<()> {
    let project = project("absent");
    let views = views(&[("clicks", None)]);
    let store = ValkeyStore::connect(&url()).await?;

    let out = read_entities(
        &store,
        &project,
        &views,
        &[EntityRequest {
            encoded_key: encode_entity_key(&[b"nobody"])?,
            views: vec![ViewRequest {
                view: "clicks".to_owned(),
                fields: vec!["count".to_owned()],
            }],
        }],
        now_micros(),
    )
    .await?;

    assert!(matches!(
        out[0][0],
        ViewValues::Missing(Missing::NeverWritten)
    ));
    Ok(())
}

#[tokio::test]
async fn a_tampered_value_is_treated_as_missing() -> Result<()> {
    let project = project("tamper");
    let views = views(&[("clicks", None)]);
    let clicks = views.get("clicks").unwrap();
    let mut store = ValkeyStore::connect(&url()).await?;

    // Written under an older schema: same field names and order, different dtype
    // for `count`. The stored bytes therefore carry a tag that no longer matches.
    let now = now_micros();
    store
        .write(&[write_batch(
            &project,
            b"u1",
            vec![
                (value_field("clicks"), encode_vector(clicks, 1, "z")),
                (freshness_field("clicks"), now.to_le_bytes().to_vec()),
            ],
        )])
        .await?;

    let changed_view = FeatureView {
        features: vec![
            Field::new("count", DType::Float64),
            Field::new("label", DType::Utf8),
        ],
        ..clicks.clone()
    };
    let changed: BTreeMap<String, FeatureView> =
        std::iter::once(("clicks".to_owned(), changed_view)).collect();

    let out = read_entities(
        &store,
        &project,
        &changed,
        &[EntityRequest {
            encoded_key: encode_entity_key(&[b"u1"])?,
            views: vec![ViewRequest {
                view: "clicks".to_owned(),
                fields: vec!["count".to_owned()],
            }],
        }],
        now,
    )
    .await?;

    assert!(matches!(
        out[0][0],
        ViewValues::Missing(Missing::SchemaMismatch)
    ));
    cleanup(&mut store, &project, clicks, &[b"u1"]).await?;
    Ok(())
}

/// The schema tag is the one value that must survive a round trip through a real
/// server byte-for-byte, so it is asserted directly rather than inferred.
#[tokio::test]
async fn the_stored_bytes_match_what_was_written() -> Result<()> {
    let project = project("bytes");
    let views = views(&[("clicks", None)]);
    let clicks = views.get("clicks").unwrap();
    let mut store = ValkeyStore::connect(&url()).await?;

    let encoded = encode_vector(clicks, 7, "q");
    let now = now_micros();
    store
        .write(&[write_batch(
            &project,
            b"u1",
            vec![
                (value_field("clicks"), encoded.clone()),
                (freshness_field("clicks"), now.to_le_bytes().to_vec()),
            ],
        )])
        .await?;

    let key = entity_hash_key(&project, "user_id", &encode_entity_key(&[b"u1"])?);
    let read = store
        .read(&[ReadRequest {
            key,
            fields: vec![value_field("clicks")],
        }])
        .await?;
    assert_eq!(read[0][0].as_deref(), Some(encoded.as_slice()));

    // And the tag decodes back to the definition's tag.
    let tag = SchemaTag::of(&clicks.features);
    assert_eq!(u32::from_le_bytes(encoded[0..4].try_into().unwrap()), tag.0);

    cleanup(&mut store, &project, clicks, &[b"u1"]).await?;
    Ok(())
}

/// A field written with an absolute expiry is reclaimed by the server, and its freshness
/// field is not.
///
/// The observable is deliberate: the expiry is set in the past, so the server deletes the
/// value field at once. The freshness field carries no expiry, so it survives, and the read
/// path therefore reports `NeverWritten` (a present freshness with an absent value) rather
/// than `Expired` (an old freshness with a value that may still be there). That is exactly the
/// pair the read path keeps distinct, checked against a real server rather than a fake.
#[tokio::test]
async fn the_server_reclaims_an_expired_value_and_keeps_its_freshness() -> Result<()> {
    let project = project("fieldexpiry");
    let views = views(&[("clicks", Some(30))]);
    let clicks = views.get("clicks").unwrap();
    let mut store = ValkeyStore::connect(&url()).await?;
    assert_ne!(
        store.field_expiry(),
        FieldExpiry::None,
        "this integration suite needs a server with hash field expiration"
    );

    let now = now_micros();
    let yesterday = now / 1_000_000 - 86_400;
    store
        .write(&[WriteBatch::new(
            entity_hash_key(&project, "user_id", &encode_entity_key(&[b"u1"])?),
            vec![
                WrittenField::new(
                    value_field("clicks"),
                    encode_vector(clicks, 42, "gold"),
                    Some(yesterday),
                ),
                WrittenField::new(freshness_field("clicks"), now.to_le_bytes().to_vec(), None),
            ],
        )])
        .await?;

    let out = read_entities(
        &store,
        &project,
        &views,
        &[EntityRequest {
            encoded_key: encode_entity_key(&[b"u1"])?,
            views: vec![ViewRequest {
                view: "clicks".to_owned(),
                fields: vec!["count".to_owned()],
            }],
        }],
        now,
    )
    .await?;
    assert!(
        matches!(out[0][0], ViewValues::Missing(Missing::NeverWritten)),
        "expected the value field to be gone and the freshness field to remain, got {:?}",
        out[0][0]
    );

    cleanup(&mut store, &project, clicks, &[b"u1"]).await?;
    Ok(())
}

/// A field rewritten with no expiry loses the TTL the server already holds for it.
///
/// A plain `HSET` clears a field's TTL the way `SET` clears a key's without `KEEPTTL`, which is
/// the opposite of what an earlier round of this branch assumed. The consequence is user-visible
/// and intended: a view that drops its `ttl_days` stops asking for an expiry, the rewrite clears
/// the old one, and its values stop expiring server-side, leaving the read-time TTL check as the
/// only thing that stops serving them.
///
/// The observable is the server's own reclamation. Two entity hashes are written with an expiry a
/// few seconds out, and one of them is rewritten with none immediately afterwards; the rewritten
/// field is asserted readable before that deadline and still readable after it, while the one left
/// alone is gone. The field left alone is the control, so the test discriminates in both
/// directions: a server that kept the TTL through a plain `HSET` would reclaim the rewritten
/// field after its deadline, and one that never reclaimed anything would leave the control
/// readable too.
#[tokio::test]
async fn rewriting_a_field_without_an_expiry_clears_the_servers_expiry() -> Result<()> {
    // Slack between the writes, and the point past which the control must be gone. Both client
    // and server read the same clock: the expiry is absolute, and this host is the one running
    // the container.
    const TTL_SECS: i64 = 3;

    let project = project("hsetexpiry");
    let views = views(&[("clicks", None)]);
    let clicks = views.get("clicks").unwrap();
    let mut store = ValkeyStore::connect(&url()).await?;
    assert_ne!(
        store.field_expiry(),
        FieldExpiry::None,
        "this test needs a server with hash field expiration"
    );

    let rewritten_key = entity_hash_key(&project, "user_id", &encode_entity_key(&[b"u1"])?);
    let control_key = entity_hash_key(&project, "user_id", &encode_entity_key(&[b"u2"])?);
    let deadline = now_micros() / 1_000_000 + TTL_SECS;
    store
        .write(&[
            WriteBatch::new(
                rewritten_key.clone(),
                vec![WrittenField::new(
                    value_field("clicks"),
                    encode_vector(clicks, 1, "x"),
                    Some(deadline),
                )],
            ),
            // The same field with the same expiry on another entity, never rewritten.
            WriteBatch::new(
                control_key.clone(),
                vec![WrittenField::new(
                    value_field("clicks"),
                    encode_vector(clicks, 3, "z"),
                    Some(deadline),
                )],
            ),
        ])
        .await?;

    // The first field again, this time with no expiry, which the write path queues as a plain
    // `HSET`.
    let rewritten = encode_vector(clicks, 2, "y");
    store
        .write(&[WriteBatch::new(
            rewritten_key.clone(),
            vec![WrittenField::new(
                value_field("clicks"),
                rewritten.clone(),
                None,
            )],
        )])
        .await?;

    let live = store
        .read(&[
            ReadRequest {
                key: rewritten_key.clone(),
                fields: vec![value_field("clicks")],
            },
            ReadRequest {
                key: control_key.clone(),
                fields: vec![value_field("clicks")],
            },
        ])
        .await?;
    assert_eq!(
        live[0][0].as_deref(),
        Some(rewritten.as_slice()),
        "the rewrite lands while the field is still inside its expiry"
    );
    assert!(
        live[1][0].is_some(),
        "the control is inside its expiry too, so it is still readable here"
    );

    let remaining = deadline - now_micros() / 1_000_000;
    std::thread::sleep(Duration::from_secs(remaining.max(0) as u64 + 1));

    let after = store
        .read(&[
            ReadRequest {
                key: rewritten_key,
                fields: vec![value_field("clicks")],
            },
            ReadRequest {
                key: control_key,
                fields: vec![value_field("clicks")],
            },
        ])
        .await?;
    assert_eq!(
        after[0][0].as_deref(),
        Some(rewritten.as_slice()),
        "the rewrite cleared the field's TTL, so it outlives the deadline it was written with"
    );
    assert!(
        after[1][0].is_none(),
        "the control still carries its expiry, so the server reclaimed it, got {:?}",
        after[1][0]
    );

    cleanup(&mut store, &project, clicks, &[b"u1", b"u2"]).await?;
    Ok(())
}

/// The real client must reject a malformed key rather than silently writing it.
#[tokio::test]
async fn a_rejected_key_never_reaches_the_server() -> Result<()> {
    let oversize = vec![b'x'; feather_core::MAX_COMPONENT_LEN + 1];
    assert!(matches!(
        encode_entity_key(&[&oversize]),
        Err(Error::EntityKeyComponentTooLong { .. })
    ));
    Ok(())
}

async fn cleanup<S: OnlineStore>(
    store: &mut S,
    project: &str,
    view: &FeatureView,
    entities: &[&[u8]],
) -> Result<()> {
    let mut deletions = Vec::new();
    for entity in entities {
        let key = entity_hash_key(project, "user_id", &encode_entity_key(&[entity])?);
        // One value field per view plus its freshness field.
        let fields: Vec<String> = std::iter::once(value_field(&view.name))
            .chain(std::iter::once(freshness_field(&view.name)))
            .collect();
        deletions.push((key, fields));
    }
    store.delete_fields(&deletions).await
}
