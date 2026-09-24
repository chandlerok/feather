//! Measures value decode cost in isolation, to decide whether lazy per-column
//! decoding is worth building.
//!
//! Run with `cargo run --release -p feather-core --example decodebench`.
//!
//! The question this answers: `decode_batch` decodes every column of a view and
//! the caller projects the ones it asked for. With a fixed-stride layout, a
//! decoder could read only the requested columns by computing offsets from the
//! schema. Is that worth building?
//!
//! It is only worth it if decode is a visible share of a read. The reference
//! point is the measured single-entity read across four views: p50 0.327ms, so
//! roughly 327,000ns of wall clock, most of which is the round trip. Any decode
//! figure well under that is noise in the profile.
//!
//! No features required: this is pure encoding, no Valkey and no DuckDB.

use std::sync::Arc;
use std::time::Instant;

use arrow::array::{
    ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray, TimestampMicrosecondArray,
};
use feather_core::definitions::{DType, Entity, FeatureView, Field, FileSource};
use feather_core::value::{SchemaTag, decode_batch, encode_batch};

/// Measured p50 of a single-entity read across four views, in nanoseconds.
/// Recorded here so the comparison is explicit rather than implied.
const REFERENCE_READ_NS: f64 = 327_000.0;

const ENTITIES: usize = 1_000;
const ITERATIONS: usize = 100;

/// A view whose first four features are the realistic mix and whose remainder are
/// counters, so width grows without changing the shape of the interesting part.
fn build_view(width: usize) -> FeatureView {
    let mut features = vec![
        Field::new("count", DType::Int64),
        Field::new("score", DType::Float64),
        Field::new("active", DType::Boolean),
        Field::new("label", DType::Utf8),
    ];
    for i in features.len()..width {
        features.push(Field::new(format!("counter_{i}"), DType::Int64));
    }
    FeatureView {
        name: "wide".to_owned(),
        entities: vec![Entity::new("user_id", "user_id")],
        source: FileSource::new("data/x.parquet"),
        features,
        ttl_days: None,
        timestamp_field: None,
        created_timestamp_field: None,
    }
}

fn build_columns(view: &FeatureView, rows: usize) -> Vec<ArrayRef> {
    view.features
        .iter()
        .map(|f| -> ArrayRef {
            match f.dtype {
                DType::Int64 => Arc::new(Int64Array::from(
                    (0..rows).map(|i| Some(i as i64)).collect::<Vec<_>>(),
                )),
                DType::Float64 => Arc::new(Float64Array::from(
                    (0..rows).map(|i| Some(i as f64 / 3.0)).collect::<Vec<_>>(),
                )),
                DType::Boolean => Arc::new(BooleanArray::from(
                    (0..rows).map(|i| Some(i % 2 == 0)).collect::<Vec<_>>(),
                )),
                DType::Utf8 => Arc::new(StringArray::from(
                    (0..rows)
                        .map(|i| Some(format!("label-{}", i % 100)))
                        .collect::<Vec<_>>(),
                )),
                DType::TimestampMicros => Arc::new(TimestampMicrosecondArray::from(
                    (0..rows).map(|i| Some(i as i64)).collect::<Vec<_>>(),
                )),
            }
        })
        .collect()
}

fn measure(width: usize) {
    let view = build_view(width);
    let columns = build_columns(&view, ENTITIES);
    let encoded = encode_batch(&view.features, &columns, 0..ENTITIES).expect("encode");
    let tag = SchemaTag::of(&view.features);
    let bufs: Vec<&[u8]> = (0..ENTITIES)
        .map(|i| encoded.row(i).expect("row"))
        .collect();

    let vector_bytes = bufs[0].len();

    // Warm up so the first iteration's allocation does not dominate.
    let warm = decode_batch(&view.features, tag, &bufs).expect("warm decode");
    std::hint::black_box(&warm);

    let started = Instant::now();
    for _ in 0..ITERATIONS {
        let decoded = decode_batch(&view.features, tag, &bufs).expect("decode");
        std::hint::black_box(&decoded);
    }
    let elapsed = started.elapsed();

    let per_entity_ns = elapsed.as_nanos() as f64 / (ITERATIONS * ENTITIES) as f64;
    let share = per_entity_ns / REFERENCE_READ_NS * 100.0;
    // If only three columns were wanted, a lazy decoder would skip the rest. For
    // a fixed-width-only view that is close to proportional.
    let lazy_share = share * (3.0 / width as f64);

    println!(
        "  width {:>4}  {:>5} bytes/vector  {:>9.0} ns/entity  {:>6.2}% of a read  \
         (lazy would be ~{:.2}%)",
        width, vector_bytes, per_entity_ns, share, lazy_share
    );
}

fn main() {
    println!("value decode cost, isolated from the network");
    println!("  {ENTITIES} entities per batch, {ITERATIONS} iterations, release build\n");
    println!(
        "  reference: a measured single-entity read across four views is {:.0}ns p50\n",
        REFERENCE_READ_NS
    );

    for width in [8usize, 32, 128, 512] {
        measure(width);
    }

    println!(
        "\nReading: a lazy decoder would help in proportion to decode's share of a read.\n\
         If every width above is a rounding error against the round trip, the optimisation\n\
         is not worth building yet, and the trigger is a view wide enough to change that."
    );
}
