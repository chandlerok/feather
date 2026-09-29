//! Shared fixture for the serving-transport measurement.
//!
//! Included by the examples with `#[path]`, not a module of the crate, so the measurement's
//! fixture never reaches the library a user links against. Cargo is set to
//! `autoexamples = false` so this file is not built as an example of its own.
//!
//! Each including example uses part of this, so the unused half of it is expected rather than a
//! mistake, which is what the allow is for.

// Included whole by two different binaries, each of which needs a different half of it.
#![allow(dead_code)]

use std::collections::BTreeMap;

use feather_core::definitions::{DType, Entity, FeatureView, Field, Source};

pub const PROJECT: &str = "ads";
pub const ENTITY_NAME: &str = "user";
pub const VIEWS: usize = 4;
pub const FEATURES_PER_VIEW: usize = 8;
/// The service every host resolves, so a client sees the same schema whichever is serving.
pub const SERVICE: &str = "serving";
/// Features requested per view. More than one, so the projection out of a view's whole encoded
/// vector is exercised rather than a single column read.
pub const REQUESTED: &[&str] = &["f0", "f3"];

/// The views the measurement serves. Declared here rather than parsed from definitions so the
/// bench needs no source file, and shaped like a project's own views.
pub fn bench_views() -> BTreeMap<String, FeatureView> {
    (0..VIEWS)
        .map(|v| {
            let name = format!("view{v}");
            let view = FeatureView {
                name: name.clone(),
                entities: vec![Entity::new("user_id", "user_id")],
                source: Source::file("data/x.parquet"),
                features: (0..FEATURES_PER_VIEW)
                    .map(|f| Field::new(format!("f{f}"), DType::Float64))
                    .collect(),
                // No expiry, so the read-time check is not what the measurement is looking at.
                ttl_days: None,
                timestamp_field: None,
                created_timestamp_field: None,
            };
            (name, view)
        })
        .collect()
}

/// The `(view, feature)` pairs the service declares, in response order.
pub fn requested_features() -> Vec<(String, String)> {
    bench_views()
        .keys()
        .flat_map(|v| REQUESTED.iter().map(move |f| (v.clone(), (*f).to_owned())))
        .collect()
}
