//! Fill a store directory with entities for the serving-transport measurement.
//!
//! Writes what the materialization writes, not filler: real encoded view rows through
//! `encode_batch` and a real freshness timestamp through `encode_freshness`. That matters
//! because the serving path decodes and TTL-checks what it reads, so a store of arbitrary bytes
//! would fail the read rather than measure it.
//!
//! Run once, then hand the directory to each host in turn. A store takes an exclusive lock on
//! open, so the hosts cannot share one at the same time.

use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array};

use feather_core::definitions::FeatureView;
use feather_core::key::{encode_entity_key, entity_hash_key, freshness_field, value_field};
use feather_core::online::fjall::FjallStore;
use feather_core::online::{OnlineStore, WriteBatch, WrittenField, encode_freshness};
use feather_core::value::encode_batch;

#[path = "bench_views.rs"]
mod bench;
use bench::{ENTITY_NAME, PROJECT, VIEWS, bench_views};

fn now_unix_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or_default()
}

/// One view's encoded row for one entity, plus its freshness.
///
/// One array per declared field, which is what `encode_batch` takes; a single array for all of
/// them is a `ColumnCountMismatch`.
fn encode_row(view: &FeatureView, entity: u64) -> Vec<u8> {
    let columns: Vec<ArrayRef> = (0..view.features.len())
        .map(|f| {
            let array: ArrayRef = Arc::new(Float64Array::from(vec![entity as f64 + f as f64]));
            array
        })
        .collect();
    encode_batch(&view.features, &columns, 0..1)
        .expect("encode")
        .row(0)
        .expect("row")
        .to_vec()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let entities: u64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(200_000);
    let views_count: usize = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(VIEWS);
    // The fill writes the views the fixture declares. A caller asking for a different count is
    // refused rather than silently ignored, because a printed figure that does not match the
    // store it describes is exactly what makes a measurement unreproducible.
    if views_count != VIEWS {
        return Err(format!(
            "the bench fixture declares {VIEWS} views; `fill_store`'s second argument is \
             {views_count}, which would print a figure the store does not match"
        )
        .into());
    }
    let dir = PathBuf::from(
        std::env::args()
            .nth(3)
            .unwrap_or_else(|| "/tmp/feather-serve-bench".into()),
    );

    let _ = std::fs::remove_dir_all(&dir);
    let mut store = FjallStore::open(&dir, 64 << 20, 64 << 20)?;
    let views = bench_views();
    let freshness = encode_freshness(now_unix_micros());

    let started = std::time::Instant::now();
    let mut written = 0u64;
    for chunk_start in (0..entities).step_by(4096) {
        let mut batches = Vec::new();
        for e in chunk_start..(chunk_start + 4096).min(entities) {
            let encoded = encode_entity_key(&[b"u1", format!("{e:012}").as_bytes()])?;
            let key = entity_hash_key(PROJECT, ENTITY_NAME, &encoded);
            let mut fields = Vec::with_capacity(views_count * 2);
            for view in views.values() {
                fields.push(WrittenField::new(
                    value_field(&view.name),
                    encode_row(view, e),
                    None,
                ));
                fields.push(WrittenField::new(
                    freshness_field(&view.name),
                    freshness.clone(),
                    None,
                ));
            }
            batches.push(WriteBatch { key, fields });
        }
        written += batches.len() as u64;
        store.write(&batches).await?;
    }
    let elapsed = started.elapsed();

    println!(
        "filled {written} entities x {views_count} views in {:.2}s",
        elapsed.as_secs_f64()
    );
    println!(
        "{:.0} records/s",
        (written * views_count as u64) as f64 / elapsed.as_secs_f64()
    );
    println!("dir: {}", dir.display());
    println!("project: {PROJECT}  entity: {ENTITY_NAME}  views: {views_count}");
    Ok(())
}
