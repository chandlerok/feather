"""The online path: a refresh writes values, and a serving read returns them.

The write path's own rules are pinned by the Rust suite beside
`crates/feather-core/src/materialize.rs`: which row wins per entity, what the freshness field
holds, when a value is expired, and when a keyspace walk happens. What is pinned here is the
binding and the surface: that a project can be refreshed and read through `FeatureStore`, that
the result is the entity frame's own columns followed by one column per requested feature in
request order, and that a value the store cannot serve arrives as a null.

The round trip is also the read-side test the review asked for: the entity keys in the frame are
integers, and the read has to find exactly what the refresh wrote for those integers.
"""

import dataclasses
import importlib.util
import time
import uuid
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from feather import FeatureStore, MaterializeReport, ViewRefresh  # noqa: F401

DAY = 86_400_000_000
"""Microseconds in a day, which is the unit every timestamp here is counted in."""

CLICKS = """
from feather import Entity, FeatureView, Field, FileSource, feature_view
from feather.types import Int64

user_entity = Entity(name="user_id", join_key="user_id")
clicks_source = FileSource(path="{clicks}")


@feature_view(name="clicks", entity=user_entity, source=clicks_source, ttl_days={ttl_days})
class Clicks(FeatureView):
    click_count = Field(Int64)
    purchase_count = Field(Int64)
"""

SETTINGS = """
project = "{project}"
definitions = ["definitions/clicks.py"]
{valkey}
"""

VALKEY = """
[valkey]
endpoint = "127.0.0.1:6379"
tls = false
"""


def now_micros() -> int:
    """The current time, in microseconds since the epoch.

    Returns:
        The current time.
    """
    return time.time_ns() // 1_000


def micros(values: list[int]) -> pl.Series:
    """Build a microsecond timestamp column from epoch counts.

    Args:
        values: Microseconds since the epoch.

    Returns:
        The column, as a naive microsecond timestamp.
    """
    return pl.Series(values, dtype=pl.Int64).cast(pl.Datetime("us"))


def entities(user_ids: list[int]) -> pl.DataFrame:
    """The entity frame every read here starts from.

    Args:
        user_ids: The join key values to read.

    Returns:
        The frame, with one column carrying a label so the test can tell the frame's own columns
        from the requested ones.
    """
    return pl.DataFrame({"user_id": user_ids, "label": [f"u{i}" for i in user_ids]})


@dataclasses.dataclass(frozen=True)
class Project:
    """A written project, with the pieces a test needs to talk about it."""

    store: FeatureStore
    Clicks: Any


def make_project(
    tmp_path: Path,
    *,
    ttl_days: int = 30,
    age_days: int = 1,
    project: str = "ads",
    valkey: bool = False,
) -> Project:
    """Write a one-view project over Parquet, and open a store on it.

    Args:
        tmp_path: The pytest temp directory.
        ttl_days: The view's `ttl_days`.
        age_days: How old the newest source row is, in days.
        project: The project name, which namespaces every online key.
        valkey: Whether `feather.toml` declares a Valkey.

    Returns:
        The project.
    """
    (tmp_path / "definitions").mkdir(exist_ok=True)
    clicks = tmp_path / "clicks.parquet"
    base = now_micros() - age_days * DAY

    # Three entities, entity 1 with two rows so the newest one has to win, and the rows in an
    # order that is not the timestamp order.
    pl.DataFrame(
        {
            "user_id": [2, 1, 1, 3],
            "event_timestamp": micros([base, base - DAY, base, base]),
            "click_count": [20, 9, 10, 5],
            "purchase_count": [200, 90, 100, 50],
        }
    ).write_parquet(clicks)

    (tmp_path / "definitions/clicks.py").write_text(
        CLICKS.format(clicks=clicks, ttl_days=ttl_days), encoding="utf-8"
    )
    config = tmp_path / "feather.toml"
    config.write_text(
        SETTINGS.format(project=project, valkey=VALKEY if valkey else ""), encoding="utf-8"
    )

    spec = importlib.util.spec_from_file_location("clicks", tmp_path / "definitions/clicks.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return Project(store=FeatureStore(config), Clicks=module.Clicks)


@pytest.fixture
def project(tmp_path: Path) -> Project:
    """A local-mode project, which reads and writes in this process.

    Args:
        tmp_path: The pytest temp directory.

    Returns:
        The project.
    """
    return make_project(tmp_path)


def test_a_refresh_then_a_serving_read_round_trips(project: Project) -> None:
    report = project.store.materialize()

    frame = pl.DataFrame(
        project.store.get_online_features(
            entity_df=entities([1, 2, 3, 99]),
            features=[project.Clicks.click_count, project.Clicks.purchase_count],
        )
    )

    assert report.total_rows == 3, "one write per entity, and the newest row per entity"
    assert frame.columns == ["user_id", "label", "click_count", "purchase_count"]
    # Entity 99 was never materialized, so it reads as a null rather than as an error.
    assert frame["click_count"].to_list() == [10, 20, 5, None]
    assert frame["purchase_count"].to_list() == [100, 200, 50, None]


def test_the_requested_order_is_the_returned_order(project: Project) -> None:
    project.store.materialize()

    frame = pl.DataFrame(
        project.store.get_online_features(
            entity_df=entities([1]),
            features=[project.Clicks.purchase_count, project.Clicks.click_count],
        )
    )

    assert frame.columns == ["user_id", "label", "purchase_count", "click_count"]


def test_a_value_older_than_its_ttl_reads_as_null(tmp_path: Path) -> None:
    # The newest source row is 40 days old and the view declares a 30 day TTL, so the value is
    # written and then refuses to be served.
    project = make_project(tmp_path, ttl_days=30, age_days=40)
    project.store.materialize()

    frame = pl.DataFrame(
        project.store.get_online_features(
            entity_df=entities([1]), features=[project.Clicks.click_count]
        )
    )

    assert frame["click_count"].to_list() == [None]


def test_the_report_names_the_view_and_its_newest_timestamp(project: Project) -> None:
    report = project.store.materialize()

    assert isinstance(report, MaterializeReport)
    assert [view.name for view in report.views] == ["clicks"]
    assert report.views[0].rows == 3
    assert report.retired == []
    assert report.views[0].max_event_timestamp_micros is not None
    assert report.views[0].elapsed_seconds >= 0.0
    assert report.elapsed_seconds >= 0.0


def test_a_subset_refresh_names_only_the_views_it_refreshed(project: Project) -> None:
    report = project.store.materialize(views=["clicks"])

    assert [view.name for view in report.views] == ["clicks"]


def test_a_view_the_project_does_not_declare_is_refused(project: Project) -> None:
    with pytest.raises(ValueError, match="absent"):
        project.store.materialize(views=["absent"])


def test_a_null_entity_key_is_refused(project: Project) -> None:
    project.store.materialize()

    with pytest.raises(ValueError, match="null"):
        project.store.get_online_features(
            entity_df=pl.DataFrame({"user_id": [1, None]}),
            features=[project.Clicks.click_count],
        )


def test_a_missing_join_key_column_is_refused(project: Project) -> None:
    project.store.materialize()

    with pytest.raises(ValueError, match="user_id"):
        project.store.get_online_features(
            entity_df=pl.DataFrame({"other": [1]}),
            features=[project.Clicks.click_count],
        )


@pytest.mark.integration
def test_a_renamed_view_is_retired_and_leaves_the_registry(tmp_path: Path) -> None:
    """A rename, over a store that outlives one `FeatureStore`, which local mode cannot be.

    The registry is in Valkey, so a second store over the same project sees what the first one
    wrote. In local mode the store is per instance, so this is the only place the retire path can
    be exercised end to end; the Rust suite pins the rule itself.
    """
    project = make_project(tmp_path, project=f"feathertest_retire_{uuid.uuid4().hex}", valkey=True)
    project.store.materialize()

    module = tmp_path / "definitions/clicks.py"
    module.write_text(
        module.read_text(encoding="utf-8").replace('name="clicks"', 'name="renamed"'),
        encoding="utf-8",
    )
    renamed = FeatureStore(tmp_path / "feather.toml")

    first = renamed.materialize()
    assert first.retired == ["clicks"], "the view the project no longer declares is retired"

    second = renamed.materialize()
    assert second.retired == [], "a retired name leaves the registry, so it retires once"


@pytest.mark.integration
def test_a_refresh_and_a_read_round_trip_through_valkey(tmp_path: Path) -> None:
    """The same round trip against a real Valkey, which is what `mise run test:integration` runs.

    The project name carries a nonce so a leftover key from another run cannot be read back as
    this run's value.
    """
    project = make_project(tmp_path, project=f"feathertest_online_{uuid.uuid4().hex}", valkey=True)
    report = project.store.materialize()

    frame = pl.DataFrame(
        project.store.get_online_features(
            entity_df=entities([1, 2, 3, 99]),
            features=[project.Clicks.click_count],
        )
    )

    assert report.views[0].rows == 3
    assert frame["click_count"].to_list() == [10, 20, 5, None], "read back through Valkey"

    # A second store over the same project sees the same values, which is only possible if the
    # first one wrote them into the server rather than into its own process.
    reopened = FeatureStore(tmp_path / "feather.toml")
    assert pl.DataFrame(
        reopened.get_online_features(entity_df=entities([1]), features=[project.Clicks.click_count])
    )["click_count"].to_list() == [10]

    # Everything is already in the registry, so nothing is retired and no keyspace walk runs.
    assert project.store.materialize().retired == []
