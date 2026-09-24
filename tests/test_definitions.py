"""Contract tests for the Python-to-Rust definition wire format.

Rust deserializes these models with serde and does not re-validate them, so a
rename here would fail at runtime rather than at import. These tests pin the
exact keys and values Rust expects.
"""

import json

import pydantic
import pytest

from feather import (
    DType,
    Entity,
    FeatureService,
    FeatureStoreConfig,
    FeatureView,
    Field,
    FileSource,
)


def a_view(**overrides: object) -> FeatureView:
    """Build a minimal valid view, with overrides applied.

    Returns:
        The view.
    """
    base: dict[str, object] = {
        "name": "user_clicks",
        "entities": [Entity(name="user_id", join_key="user_id")],
        "source": FileSource(path="data/user_stats.parquet"),
        "features": [Field(name="click_count", dtype=DType.INT64)],
    }
    base.update(overrides)
    # model_validate rather than the constructor so that deliberately invalid
    # overrides in tests do not need a type suppression.
    return FeatureView.model_validate(base)


def test_dtype_values_are_the_rust_wire_names() -> None:
    """The enum values are the serde rename_all = snake_case names."""
    assert [d.value for d in DType] == [
        "int64",
        "float64",
        "boolean",
        "utf8",
        "timestamp_micros",
    ]


def test_definition_json_uses_the_keys_rust_expects() -> None:
    """The serialized shape matches the Rust structs field for field."""
    config = FeatureStoreConfig(
        project="ads",
        views=[a_view(ttl_days=30)],
        services=[FeatureService(name="ranking", features=["user_clicks:click_count"])],
    )
    payload = json.loads(config.model_dump_json())

    assert payload["project"] == "ads"
    view = payload["views"][0]
    assert view["name"] == "user_clicks"
    assert view["entities"] == [{"name": "user_id", "join_key": "user_id"}]
    assert view["source"] == {"path": "data/user_stats.parquet"}
    assert view["features"] == [{"name": "click_count", "dtype": "int64"}]
    assert view["ttl_days"] == 30
    # Absent optional fields serialize as null, which serde's default handles.
    assert view["timestamp_field"] is None
    assert view["created_timestamp_field"] is None
    assert payload["services"][0] == {
        "name": "ranking",
        "features": ["user_clicks:click_count"],
    }


def test_a_view_without_a_ttl_serializes_as_null() -> None:
    """No TTL means no expiry, and serde's Option handles the null."""
    payload = json.loads(FeatureStoreConfig(project="ads", views=[a_view()]).model_dump_json())
    assert payload["views"][0]["ttl_days"] is None


def test_definitions_are_immutable() -> None:
    """Definitions are data; mutating one after validation is a bug."""
    view = a_view()
    attribute = "name"
    with pytest.raises(pydantic.ValidationError):
        setattr(view, attribute, "other")


def test_unknown_keys_are_rejected() -> None:
    """A typo in a definition is an error, not something to ignore."""
    # Built as a dict so the deliberate typo does not need a type suppression.
    kwargs: dict[str, object] = {
        "name": "v",
        "entities": [Entity(name="e", join_key="e")],
        "source": FileSource(path="p"),
        "features": [Field(name="f", dtype=DType.INT64)],
        "ttl_day": 30,
    }
    with pytest.raises(pydantic.ValidationError):
        FeatureView.model_validate(kwargs)


def test_more_than_one_entity_is_rejected() -> None:
    """v1 keys one hash per entity, so a multi-entity view cannot be stored."""
    with pytest.raises(pydantic.ValidationError, match="exactly one"):
        a_view(
            entities=[
                Entity(name="user_id", join_key="user_id"),
                Entity(name="item_id", join_key="item_id"),
            ]
        )


def test_duplicate_feature_names_are_rejected() -> None:
    """Duplicate names would collide on one field name in the hash."""
    with pytest.raises(pydantic.ValidationError, match="repeats a feature name"):
        a_view(
            features=[
                Field(name="count", dtype=DType.INT64),
                Field(name="count", dtype=DType.FLOAT64),
            ]
        )


def test_a_shared_timestamp_field_is_rejected() -> None:
    """The event timestamp and created timestamp cannot be the same column."""
    with pytest.raises(pydantic.ValidationError, match="both the event"):
        a_view(timestamp_field="ts", created_timestamp_field="ts")


def test_a_malformed_service_reference_is_rejected() -> None:
    """References are 'view:feature', so a bare name is an error."""
    with pytest.raises(pydantic.ValidationError, match="malformed reference"):
        FeatureService(name="s", features=["click_count"])


def test_service_references_split_on_the_first_colon() -> None:
    """A feature name containing a colon still splits correctly."""
    service = FeatureService(name="s", features=["view:a:b"])
    assert service.references() == [("view", "a:b")]


def test_cross_object_references_are_validated() -> None:
    """Pydantic cannot see across objects, so this is checked explicitly."""
    config = FeatureStoreConfig(
        project="ads",
        views=[a_view()],
        services=[FeatureService(name="s", features=["user_clicks:nope"])],
    )
    with pytest.raises(ValueError, match="has no such feature"):
        config.validate_references()


def test_a_valid_config_passes_reference_validation() -> None:
    """The happy path does not raise."""
    config = FeatureStoreConfig(
        project="ads",
        views=[a_view()],
        services=[FeatureService(name="s", features=["user_clicks:click_count"])],
    )
    config.validate_references()


def test_duplicate_view_names_are_rejected() -> None:
    """Two views with one name would collide in the hash field namespace."""
    config = FeatureStoreConfig(project="ads", views=[a_view(), a_view()])
    with pytest.raises(ValueError, match="share a name"):
        config.validate_references()


def test_an_empty_project_serializes_cleanly() -> None:
    """Empty definition lists are valid, which is what `feather init` produces."""
    payload = json.loads(FeatureStoreConfig(project="ads").model_dump_json())
    assert payload == {"project": "ads", "views": [], "services": []}
