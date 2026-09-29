//! Load test against a real Valkey.
//!
//! Run with `mise run bench:load`, which starts the constrained container first.
//! Environment:
//!
//! - `VALKEY_URL` (default `redis://127.0.0.1:6379`)
//! - `ENTITIES` (default 100000)
//! - `VIEWS` (default 4)
//! - `REQUESTS` (default 20000)
//! - `BATCH` (default 100)
//!
//! The architecture document says no performance figure may be quoted before
//! publishing a benchmark with hardware, key cardinality, and payload size. This
//! prints all three so a number can never be quoted without them.
//!
//! This measures the library, not Valkey. For the server's own ceiling, compare
//! against `mise run bench:valkey`, which runs `valkey-benchmark` on the same
//! container.
//!
//! The `valkey` feature is required, declared via `required-features` in
//! Cargo.toml rather than a `#![cfg]` here: an inner cfg would cfg out the whole
//! file including `main`, which breaks a plain `cargo test`.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
use feather_core::definitions::{DType, Entity, FeatureView, Field, Source};
use feather_core::key::{encode_entity_key, entity_hash_key, freshness_field, value_field};
use feather_core::online::{
    EntityRequest, OnlineStore, ViewRequest, ViewValues, WriteBatch, WrittenField, read_entities,
};
use feather_core::value::encode_batch;
use feather_core::{Result, ValkeyStore};

/// Features per view. Deliberately mixed types so the vector size is realistic
/// rather than a row of int64s.
const FEATURES_PER_VIEW: usize = 8;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

fn url() -> String {
    std::env::var("VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_owned())
}

/// A view with a realistic mix: counters, a score, a flag, and a label.
fn view(name: &str) -> FeatureView {
    let mut features = vec![
        Field::new("count", DType::Int64),
        Field::new("score", DType::Float64),
        Field::new("active", DType::Boolean),
        Field::new("label", DType::Utf8),
    ];
    // Pad with counters to reach the target width.
    for i in features.len()..FEATURES_PER_VIEW {
        features.push(Field::new(format!("counter_{i}"), DType::Int64));
    }
    FeatureView {
        name: name.to_owned(),
        entities: vec![Entity::new("user_id", "user_id")],
        source: Source::file("data/user_stats.parquet"),
        features,
        ttl_days: Some(30),
        timestamp_field: None,
        created_timestamp_field: None,
        pushable: false,
    }
}

/// Encode one entity's vector for a view, with deterministic values.
fn encode_vector(view: &FeatureView, seed: usize) -> Vec<u8> {
    let n = 1;
    let columns: Vec<ArrayRef> = view
        .features
        .iter()
        .map(|f| -> ArrayRef {
            match f.dtype {
                DType::Int64 => Arc::new(Int64Array::from(vec![(seed % 10_000) as i64])),
                DType::Float64 => Arc::new(Float64Array::from(vec![(seed % 1_000) as f64 / 7.0])),
                DType::Boolean => Arc::new(BooleanArray::from(vec![seed % 2 == 0])),
                DType::Utf8 => Arc::new(StringArray::from(vec![format!("label-{}", seed % 100)])),
                DType::TimestampMicros => {
                    Arc::new(arrow::array::TimestampMicrosecondArray::from(vec![
                        seed as i64,
                    ]))
                }
            }
        })
        .collect();
    encode_batch(&view.features, &columns, 0..n)
        .expect("encode")
        .row(0)
        .expect("row")
        .to_vec()
}

fn now_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

/// Nearest-rank percentile. Good enough for a load test and honest about it.
fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

async fn run() -> Result<()> {
    let entities = env_usize("ENTITIES", 100_000);
    let view_count = env_usize("VIEWS", 4);
    let requests = env_usize("REQUESTS", 20_000);
    let batch = env_usize("BATCH", 100);
    let project = "featherbench";

    let views: BTreeMap<String, FeatureView> = (0..view_count)
        .map(|i| {
            let name = format!("v{i}");
            (name.clone(), view(&name))
        })
        .collect();
    let names: Vec<String> = views.keys().cloned().collect();

    println!("feather online load test");
    println!("  url            {}", url());
    println!("  entities       {entities}");
    println!("  views/entity   {view_count}");
    println!("  features/view  {FEATURES_PER_VIEW}");
    println!("  requests       {requests}");
    println!("  batch          {batch}");

    let vector_bytes = views
        .values()
        .next()
        .map(|v| encode_vector(v, 0).len())
        .unwrap_or(0);
    println!("  vector bytes   {vector_bytes} (one view, {FEATURES_PER_VIEW} features)");
    println!("  fields/entity  {view_count} value + {view_count} freshness");

    let mut store = ValkeyStore::connect(&url()).await?;
    let now = now_micros();

    // ---------------------------------------------------------------- write
    let started = Instant::now();
    let mut written = 0usize;
    let chunk = 5_000usize;
    for start in (0..entities).step_by(chunk) {
        let end = (start + chunk).min(entities);
        let mut batches = Vec::with_capacity((end - start) * view_count);
        for e in start..end {
            let encoded = encode_entity_key(&[format!("u{e}").as_bytes()])?;
            let key = entity_hash_key(project, "user_id", &encoded);
            for name in &names {
                let v = views.get(name).expect("view exists");
                batches.push(WriteBatch::new(
                    key.clone(),
                    vec![
                        WrittenField::new(value_field(name), encode_vector(v, e), None),
                        WrittenField::new(freshness_field(name), now.to_le_bytes().to_vec(), None),
                    ],
                ));
            }
        }
        written += batches.len();
        store.write(&batches).await?;
    }
    let write_elapsed = started.elapsed();
    let field_writes = written * 2; // value + freshness per write
    println!(
        "\nwrite (materialization shape)\n  {} hashes, {written} vectors, {field_writes} fields",
        entities
    );
    println!(
        "  {:.1}s total, {:.0} vectors/s, {:.0} fields/s",
        write_elapsed.as_secs_f64(),
        written as f64 / write_elapsed.as_secs_f64(),
        field_writes as f64 / write_elapsed.as_secs_f64()
    );

    // ----------------------------------------------------------------- read
    let all_views: Vec<ViewRequest> = names
        .iter()
        .map(|n| ViewRequest {
            view: n.clone(),
            fields: vec!["count".to_owned(), "score".to_owned(), "label".to_owned()],
        })
        .collect();

    let mut single = Vec::with_capacity(requests);
    for r in 0..requests {
        let e = r % entities;
        let encoded = encode_entity_key(&[format!("u{e}").as_bytes()])?;
        let entity = EntityRequest {
            encoded_key: encoded,
            views: all_views.clone(),
        };
        let t = Instant::now();
        let out = read_entities(&store, project, &views, &[entity], now_micros()).await?;
        single.push(t.elapsed());
        debug_assert!(matches!(out[0][0], ViewValues::Present { .. }));
    }
    single.sort();

    println!("\nread: 1 entity x {view_count} views ({requests} requests)");
    println!(
        "  p50 {:.3}ms  p95 {:.3}ms  p99 {:.3}ms  max {:.3}ms",
        ms(percentile(&single, 50.0)),
        ms(percentile(&single, 95.0)),
        ms(percentile(&single, 99.0)),
        ms(*single.last().expect("non-empty"))
    );

    let rounds = (requests / batch).max(1);
    let mut batched = Vec::with_capacity(rounds);
    let mut entities_read = 0usize;
    for r in 0..rounds {
        let mut group = Vec::with_capacity(batch);
        for i in 0..batch {
            let e = (r * batch + i) % entities;
            group.push(EntityRequest {
                encoded_key: encode_entity_key(&[format!("u{e}").as_bytes()])?,
                views: all_views.clone(),
            });
        }
        let t = Instant::now();
        let out = read_entities(&store, project, &views, &group, now_micros()).await?;
        batched.push(t.elapsed());
        entities_read += out.len();
    }
    batched.sort();

    println!("\nread: {batch} entities x {view_count} views ({rounds} requests)");
    println!(
        "  p50 {:.3}ms  p95 {:.3}ms  p99 {:.3}ms  max {:.3}ms",
        ms(percentile(&batched, 50.0)),
        ms(percentile(&batched, 95.0)),
        ms(percentile(&batched, 99.0)),
        ms(*batched.last().expect("non-empty"))
    );
    let total: Duration = batched.iter().sum();
    println!(
        "  {:.0} entities/s, {:.2}us per entity",
        entities_read as f64 / total.as_secs_f64(),
        total.as_micros() as f64 / entities_read as f64
    );

    println!(
        "\ncontext: container limited to 2 CPUs / 512MiB, maxmemory 384MiB allkeys-lru.\n\
         Compare against `mise run bench:valkey` for the server's own ceiling on the same container."
    );
    Ok(())
}

fn main() -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| feather_core::Error::MalformedValue {
            field: "runtime".to_owned(),
            reason: e.to_string(),
        })?;
    runtime.block_on(run())
}
