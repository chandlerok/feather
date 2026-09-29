//! The Python-to-Rust definition contract, asserted from the Rust side.
//!
//! `tests/test_wire.py` pins what the Python wire models emit. Nothing pinned that
//! the core could read it, so a key or an enum variant added on one side alone
//! would have surfaced at first use rather than in CI. Both sides now read
//! `tests/fixtures/definitions.json`: the Python test asserts its own output
//! equals that file, and this one asserts the core deserializes it.

use feather_core::{DType, Definitions, Source};

/// The payload the Python wire layer emits.
fn fixture() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/definitions.json"
    );
    std::fs::read_to_string(path).expect("the shared fixture is readable")
}

#[test]
fn the_python_payload_deserializes_field_for_field() {
    let definitions = Definitions::from_json(&fixture()).expect("the fixture is valid");

    assert_eq!(definitions.project, "ads");

    assert_eq!(definitions.views.len(), 3);
    let view = &definitions.views[0];
    assert_eq!(view.name, "user_clicks");
    assert_eq!(view.entities.len(), 1);
    assert_eq!(view.entities[0].name, "user_id");
    assert_eq!(view.entities[0].join_key, "user_id");
    // The source is tagged, so the kind Python wrote is the kind the core reads back.
    assert_eq!(
        view.source,
        Source::file("data/user_stats.parquet"),
        "source is {}",
        view.source.description()
    );
    assert_eq!(view.features.len(), 1);
    assert_eq!(view.features[0].name, "click_count");
    assert_eq!(view.features[0].dtype, DType::Int64);

    assert_eq!(view.ttl_days, Some(30));
    // Python writes null for an unset optional, which is distinct from the key
    // being absent. Both have to mean "no override".
    assert_eq!(view.timestamp_field, None);
    assert_eq!(view.created_timestamp_field, None);
    // And a null timestamp_field means the core's default applies, not one the
    // Python layer invented.
    assert_eq!(view.timestamp_field(), "event_timestamp");

    // The other kind, so a source's tag is compared across the boundary rather than
    // asserted twice. A rename applied on one side alone would otherwise stay green.
    let postgres = &definitions.views[1];
    assert_eq!(postgres.name, "user_stats");
    assert_eq!(
        postgres.source,
        Source::postgres("pg_prod", "public", "user_stats"),
        "source is {}",
        postgres.source.description()
    );
    assert_eq!(postgres.source.kind(), "postgres");
    assert_eq!(postgres.features.len(), 1);
    assert_eq!(postgres.features[0].name, "lifetime_value");
    assert_eq!(postgres.features[0].dtype, DType::Float64);

    // The one value that is a new contract rather than an absence of one: the
    // non-default format's wire name, spelled once in the fixture and read here. The
    // `null` above is pinned by the null case; `"vortex"` is the string a rename on one
    // side alone would move, so it is compared across the boundary rather than
    // asserted twice, once per language.
    let vortex = &definitions.views[2];
    assert_eq!(
        vortex.source,
        Source::File {
            path: "s3://lake/clicks.vortex".to_owned(),
            format: Some("vortex".to_owned()),
        },
        "source is {}",
        vortex.source.description()
    );
    assert_eq!(
        vortex.source_format().expect("format"),
        feather_core::FileFormat::Vortex
    );

    assert_eq!(definitions.services.len(), 1);
    assert_eq!(definitions.services[0].name, "ranking");
    assert_eq!(
        definitions.services[0].features,
        vec!["user_clicks:click_count".to_owned()]
    );
}

#[test]
fn the_optional_keys_may_be_omitted() {
    // Python always writes nulls, so this covers what `#[serde(default)]` is for:
    // a binding that leaves an unset optional out instead of nulling it.
    let json = serde_json::json!({
        "project": "ads",
        "views": [{
            "name": "user_clicks",
            "entities": [{"name": "user_id", "join_key": "user_id"}],
            "source": {"type": "file", "path": "data/user_stats.parquet"},
            "features": [{"name": "click_count", "dtype": "int64"}],
        }],
    })
    .to_string();

    let definitions = Definitions::from_json(&json).expect("valid");

    let view = &definitions.views[0];
    assert_eq!(view.ttl_days, None);
    assert_eq!(view.timestamp_field, None);
    assert_eq!(view.created_timestamp_field, None);
    assert!(definitions.services.is_empty());
}

#[test]
fn every_dtype_wire_name_is_read() {
    // The half of the dtype contract the Python test cannot see: it compares
    // Python spellings against Python spellings, so only this side pins the
    // strings that actually cross.
    let cases = [
        ("int64", DType::Int64),
        ("float64", DType::Float64),
        ("boolean", DType::Boolean),
        ("utf8", DType::Utf8),
        ("timestamp_micros", DType::TimestampMicros),
    ];

    for (wire, expected) in cases {
        let json = serde_json::json!({
            "project": "ads",
            "views": [{
                "name": "v",
                "entities": [{"name": "e", "join_key": "e"}],
                "source": {"type": "file", "path": "p"},
                "features": [{"name": "f", "dtype": wire}],
            }],
        })
        .to_string();

        let definitions = Definitions::from_json(&json).expect("valid");

        assert_eq!(
            definitions.views[0].features[0].dtype, expected,
            "wire name `{wire}`"
        );
    }
}

#[test]
fn an_unknown_dtype_wire_name_is_rejected() {
    // Proven rather than assumed, since the point of pinning the names above is
    // that an unlisted one cannot get through.
    let json = serde_json::json!({
        "project": "ads",
        "views": [{
            "name": "v",
            "entities": [{"name": "e", "join_key": "e"}],
            "source": {"type": "file", "path": "p"},
            "features": [{"name": "f", "dtype": "int32"}],
        }],
    })
    .to_string();

    let error = Definitions::from_json(&json).expect_err("must fail");

    assert!(
        error.to_string().contains("malformed definitions"),
        "{error}"
    );
}

#[test]
fn json_that_is_not_definitions_is_rejected() {
    let error = Definitions::from_json(r#"{"project": 1}"#).expect_err("must fail");

    assert!(
        error.to_string().contains("malformed definitions"),
        "{error}"
    );
}

#[test]
fn a_postgres_source_is_read_from_its_tag() {
    // The second kind, so the tag itself is what selects the variant rather than the
    // presence of a key.
    let json = serde_json::json!({
        "project": "ads",
        "views": [{
            "name": "user_stats",
            "entities": [{"name": "user_id", "join_key": "user_id"}],
            "source": {
                "type": "postgres",
                "connection": "pg_prod",
                "schema": "public",
                "table": "user_stats",
            },
            "features": [{"name": "ltv", "dtype": "float64"}],
        }],
    })
    .to_string();

    let definitions = Definitions::from_json(&json).expect("valid");

    assert_eq!(
        definitions.views[0].source,
        Source::postgres("pg_prod", "public", "user_stats")
    );
    // Resolving a connection is a separate step, so a source with an unconfigured name
    // still deserializes and is rejected by `validate_sources` instead.
    assert_eq!(
        definitions.views[0].source.connection_name(),
        Some("pg_prod")
    );
}

#[test]
fn an_untagged_source_is_rejected() {
    // The old shape. A path without a tag has no kind, and guessing one is how a
    // Postgres source would be silently read as a file.
    let json = serde_json::json!({
        "project": "ads",
        "views": [{
            "name": "v",
            "entities": [{"name": "e", "join_key": "e"}],
            "source": {"path": "data/user_stats.parquet"},
            "features": [{"name": "f", "dtype": "int64"}],
        }],
    })
    .to_string();

    let error = Definitions::from_json(&json).expect_err("must fail");

    assert!(
        error.to_string().contains("malformed definitions"),
        "{error}"
    );
}

#[test]
fn a_source_field_that_is_empty_is_rejected() {
    // The Python mirror makes every source field non-empty, so this is the direction the
    // stated authority has to hold: the core rejects what the binding would.
    let json = serde_json::json!({
        "project": "ads",
        "views": [{
            "name": "user_stats",
            "entities": [{"name": "user_id", "join_key": "user_id"}],
            "source": {
                "type": "postgres",
                "connection": "pg_prod",
                "schema": "public",
                "table": "",
            },
            "features": [{"name": "ltv", "dtype": "float64"}],
        }],
    })
    .to_string();

    let error = Definitions::from_json(&json).expect_err("must fail");

    assert_eq!(
        error.to_string(),
        "view `user_stats` declares an empty `table` in its source"
    );
}

#[test]
fn a_source_key_from_another_kind_is_rejected() {
    // `deny_unknown_fields` on the enum, so a key that belongs to the other variant is
    // a mistake rather than something ignored.
    let json = serde_json::json!({
        "project": "ads",
        "views": [{
            "name": "v",
            "entities": [{"name": "e", "join_key": "e"}],
            "source": {"type": "file", "path": "p", "table": "t"},
            "features": [{"name": "f", "dtype": "int64"}],
        }],
    })
    .to_string();

    let error = Definitions::from_json(&json).expect_err("must fail");

    assert!(
        error.to_string().contains("malformed definitions"),
        "{error}"
    );
}
