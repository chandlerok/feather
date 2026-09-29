"""The declared-view layer: schema in the class body, configuration on the decorator.

Views are declared at module level here, the way a real definitions module does,
so the checker sees a concrete class and its fields.
"""

import json

import pydantic
import pytest

from feather import (
    Entity,
    FeatureService,
    FeatureStoreConfig,
    FeatureView,
    Field,
    FileSource,
    Float64,
    Int64,
    PostgresSource,
    Utf8,
    config_to_wire,
    feature_view,
    view_fields,
    view_to_wire,
)

USER = Entity(name="user_id", join_key="user_id")
SOURCE = FileSource(path="data/user_stats.parquet")
POSTGRES_SOURCE = PostgresSource(connection="pg_prod", schema="public", table="user_stats")


@feature_view(name="user_clicks", entity=USER, source=SOURCE, ttl_days=30)
class UserClicks(FeatureView):
    click_count = Field(Int64)
    purchase_count = Field(Int64)


@feature_view(name="user_stats", entity=USER, source=SOURCE)
class UserStats(FeatureView):
    ltv = Field(Float64)


@feature_view(name="shadow", entity=USER, source=SOURCE)
class Shadow(FeatureView):
    """Every one of these names is a config key or a base-class concern."""

    name = Field(Utf8)
    source = Field(Utf8)
    ttl_days = Field(Int64)
    config = Field(Utf8)
    fields = Field(Utf8)


@feature_view(name="user_clicks", entity=USER, source=SOURCE)
class OtherUserClicks(FeatureView):
    """Shares a wire name with UserClicks, for the duplicate check."""

    other = Field(Int64)


@feature_view(name="user_stats_pg", entity=USER, source=POSTGRES_SOURCE)
class UserStatsPg(FeatureView):
    ltv = Field(Float64)


def test_a_declared_view_compiles_to_the_wire_model() -> None:
    """The declarative layer produces what Rust deserializes."""
    wire = view_to_wire(UserClicks)
    assert wire.name == "user_clicks"
    assert wire.entities == [USER]
    assert wire.source == SOURCE
    assert wire.ttl_days == 30
    assert [f.name for f in wire.features] == ["click_count", "purchase_count"]
    assert [f.dtype for f in wire.features] == ["int64", "int64"]


def test_the_compiled_json_is_the_rust_contract() -> None:
    """End to end: a declared view serializes with the keys Rust expects.

    The same shape is asserted in test_wire against the wire models directly, so
    a change to either layer breaks one of the two tests.
    """
    config = FeatureStoreConfig(project="ads", views=[UserClicks])
    payload = json.loads(config_to_wire(config).model_dump_json())
    assert payload["views"][0] == {
        "name": "user_clicks",
        "entities": [{"name": "user_id", "join_key": "user_id"}],
        "source": {"type": "file", "path": "data/user_stats.parquet", "format": None},
        "features": [
            {"name": "click_count", "dtype": "int64"},
            {"name": "purchase_count", "dtype": "int64"},
        ],
        "ttl_days": 30,
        "timestamp_field": None,
        "created_timestamp_field": None,
        "pushable": False,
    }


def test_a_view_over_a_postgres_table_compiles_to_the_tagged_source() -> None:
    """Either source kind reaches Rust the same way, tagged with its kind."""
    config = FeatureStoreConfig(project="ads", views=[UserStatsPg])
    payload = json.loads(config_to_wire(config).model_dump_json())

    assert payload["views"][0]["source"] == {
        "type": "postgres",
        "connection": "pg_prod",
        "schema": "public",
        "table": "user_stats",
    }


def test_a_feature_may_be_called_name_or_source() -> None:
    """This is why configuration is on the decorator rather than the body.

    With config in the body, the view name and every config key would be taken
    out of the feature namespace, so none of these could be declared.
    """
    wire = view_to_wire(Shadow)
    assert wire.name == "shadow"
    assert [f.name for f in wire.features] == [
        "name",
        "source",
        "ttl_days",
        "config",
        "fields",
    ]


def test_the_base_class_declares_nothing_but_dunders() -> None:
    """Pins that the base cannot take a name away from a schema."""
    non_dunder = [
        name for name in vars(FeatureView) if not (name.startswith("__") and name.endswith("__"))
    ]
    assert non_dunder == []


def test_fields_carry_their_view_and_keep_declaration_order() -> None:
    """A field is self-describing, which is what lets a reference be an object."""
    fields = view_fields(UserClicks)
    assert [f.name for f in fields] == ["click_count", "purchase_count"]
    assert {f.view for f in fields} == {"user_clicks"}
    assert all(f.dtype is Int64 for f in fields)


def test_a_reference_renders_as_the_wire_form() -> None:
    """The string form still exists; it is just derived rather than written."""
    assert repr(UserClicks.click_count) == "user_clicks:click_count (int64)"


def test_a_view_declaring_no_fields_is_rejected_at_decoration() -> None:
    """The usual cause is the decorator on the wrong class."""
    with pytest.raises(ValueError, match="declares no fields"):

        @feature_view(name="empty", entity=USER, source=SOURCE)
        class Empty(FeatureView):
            pass


def test_a_malformed_view_fails_at_decoration() -> None:
    """Failing at import is the point: an engine never sees a bad definition."""
    with pytest.raises(pydantic.ValidationError):

        @feature_view(name="bad", entity=USER, source=SOURCE, ttl_days=0)
        class Bad(FeatureView):
            f = Field(Int64)


def test_a_service_expands_a_whole_view() -> None:
    """Passing the view means every field on it, in declaration order."""
    service = FeatureService(name="ranking", features=[UserClicks])
    assert [(f.view, f.name) for f in service.features] == [
        ("user_clicks", "click_count"),
        ("user_clicks", "purchase_count"),
    ]


def test_a_service_takes_a_subset_in_order() -> None:
    """A subset is chosen by referring to the fields, so it cannot misspell one."""
    service = FeatureService(
        name="ranking",
        features=[UserClicks.purchase_count, UserStats.ltv],
    )
    config = FeatureStoreConfig(project="ads", views=[UserClicks, UserStats], services=[service])
    payload = json.loads(config_to_wire(config).model_dump_json())
    assert payload["services"] == [
        {
            "name": "ranking",
            "features": ["user_clicks:purchase_count", "user_stats:ltv"],
        }
    ]


def test_two_views_sharing_a_wire_name_are_rejected() -> None:
    """Two views with one name collide in the hash field namespace."""
    config = FeatureStoreConfig(project="ads", views=[UserClicks, OtherUserClicks])
    with pytest.raises(ValueError, match="share a name"):
        config_to_wire(config)


def test_a_service_naming_an_undeclared_view_is_rejected() -> None:
    """The residual cross-reference: the field exists, but not in this project."""
    service = FeatureService(name="ranking", features=[UserClicks.click_count])
    config = FeatureStoreConfig(project="ads", views=[UserStats], services=[service])
    with pytest.raises(KeyError, match="no feature view named"):
        config_to_wire(config)


def test_an_empty_project_compiles_cleanly() -> None:
    """Empty definitions are valid, which is what `feather init` produces."""
    config = FeatureStoreConfig(project="ads")
    assert json.loads(config_to_wire(config).model_dump_json()) == {
        "project": "ads",
        "views": [],
        "services": [],
    }
