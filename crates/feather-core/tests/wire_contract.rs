//! The Python-to-Rust definition contract, asserted from the Rust side.
//!
//! `tests/test_wire.py` pins what the Python wire models emit. Nothing pinned that
//! the core could read it, so a key or an enum variant added on one side alone
//! would have surfaced at first use rather than in CI. Both sides now read
//! `tests/fixtures/definitions.json`: the Python test asserts its own output
//! equals that file, and this one asserts the core deserializes it.

use feather_core::{DType, Definitions};

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

    assert_eq!(definitions.views.len(), 1);
    let view = &definitions.views[0];
    assert_eq!(view.name, "user_clicks");
    assert_eq!(view.entities.len(), 1);
    assert_eq!(view.entities[0].name, "user_id");
    assert_eq!(view.entities[0].join_key, "user_id");
    assert_eq!(view.source.path, "data/user_stats.parquet");
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
            "source": {"path": "data/user_stats.parquet"},
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
                "source": {"path": "p"},
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
            "source": {"path": "p"},
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
