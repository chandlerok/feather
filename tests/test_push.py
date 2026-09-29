"""Pushing one entity's values for one view, and what refuses to let a push land.

The write path's own rules are pinned by the Rust suite beside
`crates/feather-core/src/online/mod.rs` and `materialize.rs`: that a push covers a view in full,
that the guard compares event timestamps, and that a refresh and a push each defer to the newer
of the two. What is pinned here is the binding and the surface: that a declared view is
accepted as the object the project wrote, that the values convert into the same vector a refresh
would have written, that a serving read returns what a push landed, and that the refusals a
caller can provoke arrive as the errors the docstring names.

Both directions of the guard are here on purpose. A push that only ever landed would say nothing
about whether a refresh undoes it, and a refresh that only ever won would say nothing about a
producer that already holds a newer value.
"""

from __future__ import annotations

import importlib.util
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import polars as pl
import pytest

from feather import FeatureStore

if TYPE_CHECKING:
    from pathlib import Path

DAY = 86_400_000_000
"""Microseconds in a day, which is the unit every timestamp here is counted in."""

FEATURES = ("click_count", "purchase_count")
"""The view's two features, in the order they are declared."""

CLICKS = """
from feather import Entity, FeatureView, Field, FileSource, feature_view
from feather.types import Int64

user_entity = Entity(name="user", join_key="user_id")
clicks_source = FileSource(path="{clicks}")


@feature_view(name="clicks", entity=user_entity, source=clicks_source, ttl_days={ttl_days})
class Clicks(FeatureView):
    click_count = Field(Int64)
    purchase_count = Field(Int64)
"""

SETTINGS = """
project = "ads"
definitions = ["definitions/clicks.py"]
"""


def now_micros() -> int:
    """The current time, in microseconds since the epoch.

    Returns:
        The current time.
    """
    return time.time_ns() // 1_000


def entities(user_ids: list[int]) -> pl.DataFrame:
    """The entity frame every read here starts from.

    Args:
        user_ids: The join key values to read.

    Returns:
        The frame, with one column carrying a label so a test can tell the frame's own columns
        from the requested ones.
    """
    return pl.DataFrame({"user_id": user_ids, "label": [f"u{i}" for i in user_ids]})


class Project:
    """A one-view project over a single-row source, and a store open on it.

    The source holds one entity at ``base``, so a read for any other entity is returning only
    what a push put there, and a read for this one is returning what a refresh or a push left.
    """

    def __init__(self, tmp_path: Path, *, ttl_days: int = 30) -> None:
        """Write the project and open a store on it.

        Args:
            tmp_path: The pytest temp directory.
            ttl_days: The view's `ttl_days`.
        """
        (tmp_path / "definitions").mkdir(exist_ok=True)
        clicks = tmp_path / "clicks.parquet"
        self.base = now_micros() - DAY
        pl.DataFrame(
            {
                "user_id": [1],
                "event_timestamp": pl.Series([self.base], dtype=pl.Int64).cast(pl.Datetime("us")),
                "click_count": [10],
                "purchase_count": [100],
            }
        ).write_parquet(clicks)

        (tmp_path / "definitions/clicks.py").write_text(
            CLICKS.format(clicks=clicks, ttl_days=ttl_days), encoding="utf-8"
        )
        config = tmp_path / "feather.toml"
        config.write_text(SETTINGS, encoding="utf-8")

        spec = importlib.util.spec_from_file_location("clicks", tmp_path / "definitions/clicks.py")
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.store = FeatureStore(config)
        self.Clicks: Any = module.Clicks

    def read(self, user_id: int) -> dict[str, list[Any]]:
        """Read both of the view's features for one entity.

        Args:
            user_id: The join key value to read.

        Returns:
            Each declared feature's column, as the read returned it. A value that is missing,
            expired, or absent is a null.
        """
        frame = pl.DataFrame(
            self.store.get_online_features(
                entities([user_id]),
                [self.Clicks.click_count, self.Clicks.purchase_count],
            )
        )
        return {name: frame[name].to_list() for name in FEATURES}

    def push(self, user_id: int, values: dict[str, Any], event_timestamp: int) -> bool:
        """Push a whole view's vector for one entity.

        Args:
            user_id: The join key value to write.
            values: One value per declared feature.
            event_timestamp: The event timestamp the values describe.

        Returns:
            Whether the values were written.
        """
        return self.store.push(
            self.Clicks,
            {"user_id": user_id},
            values,
            event_timestamp=event_timestamp,
        )


@pytest.fixture
def project(tmp_path: Path) -> Project:
    """A one-view project over a single-row source.

    Args:
        tmp_path: The pytest temp directory.

    Returns:
        The project.
    """
    return Project(tmp_path)


def test_a_push_is_read_back_by_a_serving_read(project: Project) -> None:
    assert project.push(7, {"click_count": 12, "purchase_count": 120}, project.base + DAY)
    assert project.read(7) == {"click_count": [12], "purchase_count": [120]}


def test_a_push_lands_for_an_entity_the_source_never_had(project: Project) -> None:
    # The reason a push exists: a value computed outside the project reaches the online store
    # before, or without, the source ever producing a row for that entity.
    assert project.push(99, {"click_count": 1, "purchase_count": 2}, project.base)
    assert project.read(99) == {"click_count": [1], "purchase_count": [2]}


def test_a_push_asks_for_every_feature_of_the_view(project: Project) -> None:
    # A view's features share one stored field, so a subset would be a read-modify-write that
    # races a refresh. The refusal is the point, and the message has to name what is missing.
    with pytest.raises(ValueError, match="purchase_count"):
        project.push(7, {"click_count": 12}, project.base + DAY)


def test_a_push_refuses_a_feature_the_view_does_not_declare(project: Project) -> None:
    with pytest.raises(ValueError, match="not a feature of this view"):
        project.push(
            7,
            {"click_count": 12, "purchase_count": 120, "ltv": 4.5},
            project.base + DAY,
        )


def test_a_push_may_carry_a_null(project: Project) -> None:
    # A null is a value the push states, not a gap in what it was given, and it has to survive
    # the encode and decode as a null rather than arriving as a missing value.
    assert project.push(7, {"click_count": None, "purchase_count": 120}, project.base + DAY)
    assert project.read(7) == {"click_count": [None], "purchase_count": [120]}


def test_a_push_wants_the_declared_view_not_a_string(project: Project) -> None:
    # The project declares a view as a class, and a reference is an object, so a push takes the
    # same object a read does rather than a name that is looked up at runtime.
    with pytest.raises(TypeError, match="declared feature view"):
        project.store.push(
            "clicks",  # type: ignore[arg-type]
            {"user_id": 7},
            {"click_count": 1, "purchase_count": 2},
            event_timestamp=project.base,
        )


def test_a_push_wants_a_join_key_that_is_an_integer_or_a_string(project: Project) -> None:
    with pytest.raises(TypeError, match="an integer or a string"):
        project.store.push(
            project.Clicks,
            {"user_id": 1.5},  # type: ignore[dict-item]
            {"click_count": 1, "purchase_count": 2},
            event_timestamp=project.base,
        )


def test_a_push_wants_exactly_the_views_join_key(project: Project) -> None:
    with pytest.raises(ValueError, match="nothing else"):
        project.store.push(
            project.Clicks,
            {"user_id": 7, "tenant": "acme"},
            {"click_count": 1, "purchase_count": 2},
            event_timestamp=project.base,
        )


def test_a_push_of_the_same_timestamp_is_refused(project: Project) -> None:
    # What makes a push safe to retry: a repeat claims a timestamp the store already holds, and
    # rewriting the same bytes is not an improvement.
    assert project.push(7, {"click_count": 12, "purchase_count": 120}, project.base + DAY)
    assert not project.push(7, {"click_count": 99, "purchase_count": 990}, project.base + DAY)
    assert project.read(7) == {"click_count": [12], "purchase_count": [120]}


def test_a_refresh_leaves_a_fresher_push_alone(project: Project) -> None:
    # The case the guard exists for. A producer pushed a value the source has not caught up
    # with, and a scheduled refresh runs over the older source row afterwards.
    assert project.push(1, {"click_count": 12, "purchase_count": 120}, project.base + DAY)

    report = project.store.materialize()

    assert project.read(1) == {
        "click_count": [12],
        "purchase_count": [120],
    }, "the refresh must not undo the newer push"
    assert report.total_rows == 0, "a refused row is not a written row"
    assert report.total_rows_refused == 1
    assert report.views[0].rows_refused == 1


def test_a_refresh_overwrites_a_push_the_source_has_caught_up_with(project: Project) -> None:
    # The other direction, and the one that says the source is still the authority: once it
    # holds something newer, the refresh wins.
    assert project.push(1, {"click_count": 12, "purchase_count": 120}, project.base - DAY)

    report = project.store.materialize()

    assert project.read(1) == {
        "click_count": [10],
        "purchase_count": [100],
    }, "the source is the authority over what it has seen"
    assert report.total_rows == 1
    assert report.total_rows_refused == 0


def test_a_push_older_than_what_is_stored_is_refused(project: Project) -> None:
    # A refresh has landed, and a producer replays an event the refresh has already passed.
    project.store.materialize()
    assert not project.push(1, {"click_count": 12, "purchase_count": 120}, project.base - DAY)
    assert project.read(1) == {"click_count": [10], "purchase_count": [100]}


def test_a_push_leaves_the_other_entities_values_alone(project: Project) -> None:
    # Every entity has its own hash, so a push writes one key. An entity the refresh filled
    # keeps what the refresh gave it.
    project.store.materialize()
    assert project.push(7, {"click_count": 12, "purchase_count": 120}, project.base + DAY)
    assert project.read(1) == {"click_count": [10], "purchase_count": [100]}
    assert project.read(7) == {"click_count": [12], "purchase_count": [120]}


def test_a_push_is_expired_by_the_views_ttl(tmp_path: Path) -> None:
    # A pushed value is a value like any other, so a view's TTL applies to it. The read-time
    # check is strictly older than the TTL, so the push is two days old against a one-day TTL:
    # a value exactly at the deadline is still fresh, and a test that put it there would pass
    # for the wrong reason.
    project = Project(tmp_path, ttl_days=1)
    assert project.push(7, {"click_count": 12, "purchase_count": 120}, project.base - DAY)
    assert project.read(7) == {"click_count": [None], "purchase_count": [None]}


STAMPS = """
from feather import Entity, FeatureView, Field, FileSource, feature_view
from feather.types import Int64, TimestampMicros

user_entity = Entity(name="user", join_key="user_id")
stamps_source = FileSource(path="{stamps}")


@feature_view(name="stamps", entity=user_entity, source=stamps_source, ttl_days=30)
class Stamps(FeatureView):
    last_seen = Field(TimestampMicros)
    visits = Field(Int64)
"""


def test_a_push_encodes_every_dtype_the_core_declares(tmp_path: Path) -> None:
    # A timestamp feature is stored as a timestamp column, not as the integer one an `Int64`
    # feature needs, and the codec compares the column it is given against the dtype it
    # declared. A push that built the wrong array for one dtype would encode a vector the
    # encoder refuses, so every dtype a view can declare is pushed here at once.
    (tmp_path / "definitions").mkdir(exist_ok=True)
    stamps = tmp_path / "stamps.parquet"
    base = now_micros() - DAY
    pl.DataFrame(
        {
            "user_id": [1],
            "event_timestamp": pl.Series([base], dtype=pl.Int64).cast(pl.Datetime("us")),
            "last_seen": pl.Series([base], dtype=pl.Int64).cast(pl.Datetime("us")),
            "visits": [3],
        }
    ).write_parquet(stamps)
    (tmp_path / "definitions/stamps.py").write_text(STAMPS.format(stamps=stamps), encoding="utf-8")
    config = tmp_path / "feather.toml"
    config.write_text(
        'project = "ads"\ndefinitions = ["definitions/stamps.py"]\n', encoding="utf-8"
    )
    spec = importlib.util.spec_from_file_location("stamps", tmp_path / "definitions/stamps.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    store = FeatureStore(config)
    seen = now_micros() - 3_600_000_000
    assert store.push(
        module.Stamps,
        {"user_id": 7},
        {"last_seen": seen, "visits": 4},
        event_timestamp=base + DAY,
    )

    frame = pl.DataFrame(
        store.get_online_features(
            entities([7]),
            [module.Stamps.last_seen, module.Stamps.visits],
        )
    )
    assert frame["visits"].to_list() == [4]
    # The read is declared to return the feature as a microsecond timestamp, so the pushed
    # value comes back as a datetime rather than as the integer it was given as. The declared
    # type carries no timezone, so the datetime is naive, and it is built from a timedelta
    # rather than a float so a large epoch keeps its microseconds.
    expected = datetime(1970, 1, 1) + timedelta(microseconds=seen)
    assert frame["last_seen"].to_list() == [expected]
