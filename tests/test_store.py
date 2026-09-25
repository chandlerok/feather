"""The Python read path: declared definitions in, a training set out.

The join rules themselves are pinned by the Rust conformance suite beside
`crates/feather-core/src/offline.rs`. What is pinned here is the binding: that the
modules `feather.toml` names are imported and resolved, that the entity frame's own
columns come back attached to the features in the frame's row order, and that a
failure arrives as the exception a Python caller would catch.
"""

import dataclasses
import importlib.util
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from feather import FeatureStore, Field
from feather.types import Int64

DAY = 86_400_000_000
"""Microseconds in a day, which is the unit every timestamp here is counted in."""

BASE = 1_700_000_000_000_000
"""2023-11-14, in microseconds since the epoch."""

SETTINGS = """
project = "ads"
definitions = ["definitions/clicks.py", "definitions/stats.py"]
"""

CLICKS = """
from feather import Entity, FeatureView, Field, FileSource, feature_view
from feather.types import Int64

user_entity = Entity(name="user_id", join_key="user_id")
clicks_source = FileSource(path="{clicks}")


@feature_view(name="clicks", entity=user_entity, source=clicks_source, ttl_days=30)
class Clicks(FeatureView):
    click_count = Field(Int64)
    purchase_count = Field(Int64)
"""

STATS = """
from feather import Entity, FeatureView, Field, FileSource, feature_view
from feather.types import Int64

user_entity = Entity(name="user_id", join_key="user_id")
stats_source = FileSource(path="{stats}")


@feature_view(name="stats", entity=user_entity, source=stats_source, ttl_days=30)
class Stats(FeatureView):
    lifetime_value = Field(Int64)
"""

UNDECLARED = """
from feather import Entity, FeatureView, Field, FileSource, feature_view
from feather.types import Int64

user_entity = Entity(name="user_id", join_key="user_id")
other_source = FileSource(path="{other}")


@feature_view(name="other", entity=user_entity, source=other_source)
class Other(FeatureView):
    unseen = Field(Int64)
"""


def micros(values: list[int]) -> pl.Series:
    """Build a microsecond timestamp column from epoch counts.

    Args:
        values: Microseconds since the epoch.

    Returns:
        The column, as a naive microsecond timestamp.
    """
    return pl.Series(values, dtype=pl.Int64).cast(pl.Datetime("us"))


def labels() -> pl.DataFrame:
    """The label set every test joins onto.

    Returns:
        One row per interesting case: a row whose value is the newest at its
        timestamp, a row with nothing before it, a row whose value is older than
        the view's TTL, and a row for an entity the source never mentions.
    """
    return pl.DataFrame(
        {
            "user_id": [1, 2, 1, 1, 3],
            "event_timestamp": micros(
                [BASE + DAY, BASE, BASE + 2 * DAY, BASE + 40 * DAY, BASE + DAY]
            ),
            "label": ["at-source", "before-source", "after", "expired", "unknown-user"],
        }
    )


def load(path: Path) -> Any:
    """Load a definition module by path, the way the store does.

    Loading rather than importing keeps each test's module separate: a definition
    module is cached in ``sys.modules`` by name, and two tests naming the same
    module would otherwise share the first one's classes.

    Args:
        path: The module to load.

    Returns:
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclasses.dataclass(frozen=True)
class Project:
    """A written project, with the pieces a test needs to talk about it."""

    config: Path
    store: FeatureStore
    Clicks: Any
    Stats: Any


@pytest.fixture
def project(tmp_path: Path) -> Project:
    """Write a two-view project over Parquet, and open a store on it.

    Args:
        tmp_path: The pytest temp directory.

    Returns:
        The project.
    """
    (tmp_path / "definitions").mkdir()
    clicks = tmp_path / "clicks.parquet"
    stats = tmp_path / "stats.parquet"

    pl.DataFrame(
        {
            "user_id": [1, 1, 2],
            "event_timestamp": micros([BASE, BASE + DAY, BASE + DAY]),
            "click_count": [10, 20, 5],
            "purchase_count": [100, 200, 50],
        }
    ).write_parquet(clicks)
    pl.DataFrame(
        {
            "user_id": [1, 2],
            "event_timestamp": micros([BASE, BASE]),
            "lifetime_value": [1000, 2000],
        }
    ).write_parquet(stats)

    (tmp_path / "definitions/clicks.py").write_text(CLICKS.format(clicks=clicks), encoding="utf-8")
    (tmp_path / "definitions/stats.py").write_text(STATS.format(stats=stats), encoding="utf-8")
    config = tmp_path / "feather.toml"
    config.write_text(SETTINGS, encoding="utf-8")

    return Project(
        config=config,
        store=FeatureStore(config),
        Clicks=load(tmp_path / "definitions/clicks.py").Clicks,
        Stats=load(tmp_path / "definitions/stats.py").Stats,
    )


def test_features_arrive_beside_the_frames_own_columns(project: Project) -> None:
    result = project.store.get_historical_features(
        entity_df=labels(), features=[project.Clicks.click_count]
    )
    frame = pl.DataFrame(result)

    assert frame.columns == ["user_id", "event_timestamp", "label", "click_count"]
    assert frame["label"].to_list() == [
        "at-source",
        "before-source",
        "after",
        "expired",
        "unknown-user",
    ]
    assert frame["click_count"].to_list() == [20, None, 20, None, None]


def test_a_label_takes_the_newest_row_at_or_before_it(project: Project) -> None:
    frame = pl.DataFrame(
        project.store.get_historical_features(
            entity_df=labels(), features=[project.Clicks.click_count]
        )
    )

    # `at-source` is exactly at the second source row's timestamp, and `after` is a
    # day past it. Both take the newer row rather than the first one.
    assert frame["click_count"].to_list()[0] == 20
    assert frame["click_count"].to_list()[2] == 20


def test_a_value_older_than_its_ttl_is_null(project: Project) -> None:
    frame = pl.DataFrame(
        project.store.get_historical_features(
            entity_df=labels(), features=[project.Clicks.click_count]
        )
    )

    # 40 days past a source row that is a day old is 39 days, past the 30-day TTL.
    assert frame.filter(pl.col("label") == "expired")["click_count"].to_list() == [None]


def test_a_label_with_nothing_before_it_keeps_its_place(project: Project) -> None:
    frame = pl.DataFrame(
        project.store.get_historical_features(
            entity_df=labels(), features=[project.Clicks.click_count]
        )
    )

    assert frame.height == 5
    assert frame.filter(pl.col("label") == "before-source")["click_count"].to_list() == [None]


def test_on_missing_drop_removes_the_rows_with_nothing(project: Project) -> None:
    frame = pl.DataFrame(
        project.store.get_historical_features(
            entity_df=labels(), features=[project.Clicks.click_count], on_missing="drop"
        )
    )

    assert frame["label"].to_list() == ["at-source", "after"]
    assert frame["click_count"].to_list() == [20, 20]


def test_drop_keeps_a_row_only_when_every_view_has_a_value(project: Project) -> None:
    frame = pl.DataFrame(
        project.store.get_historical_features(
            entity_df=labels(),
            features=[project.Clicks.click_count, project.Stats.lifetime_value],
            on_missing="drop",
        )
    )

    # `before-source` has a lifetime value but no click count, and the row is gone:
    # a row is dropped if any requested view has nothing, so no kept row holds a
    # null that `drop` was asked to remove.
    assert frame["label"].to_list() == ["at-source", "after"]
    assert frame["lifetime_value"].to_list() == [1000, 1000]


def test_strict_excludes_a_row_at_exactly_the_label_time(project: Project) -> None:
    frame = pl.DataFrame(
        project.store.get_historical_features(
            entity_df=labels(), features=[project.Clicks.click_count], strict=True
        )
    )

    assert frame["click_count"].to_list() == [10, None, 20, None, None]


def test_columns_follow_the_request_order(project: Project) -> None:
    frame = pl.DataFrame(
        project.store.get_historical_features(
            entity_df=labels(),
            features=[project.Clicks.purchase_count, project.Clicks.click_count],
        )
    )

    assert frame.columns == [
        "user_id",
        "event_timestamp",
        "label",
        "purchase_count",
        "click_count",
    ]


def test_two_views_are_joined_in_one_call(project: Project) -> None:
    frame = pl.DataFrame(
        project.store.get_historical_features(
            entity_df=labels(),
            features=[project.Clicks.click_count, project.Stats.lifetime_value],
        )
    )

    assert frame.columns[-2:] == ["click_count", "lifetime_value"]
    assert frame["lifetime_value"].to_list() == [1000, 2000, 1000, None, None]


def test_a_label_timestamp_column_can_be_named(project: Project) -> None:
    frame = pl.DataFrame(
        project.store.get_historical_features(
            entity_df=labels().rename({"event_timestamp": "as_of"}),
            features=[project.Clicks.click_count],
            timestamp_column="as_of",
        )
    )

    assert frame["click_count"].to_list() == [20, None, 20, None, None]


def test_a_feature_colliding_with_a_label_column_is_refused(project: Project) -> None:
    frame = labels().with_columns(pl.lit(0).alias("click_count"))

    with pytest.raises(ValueError, match="already has a column named `click_count`"):
        project.store.get_historical_features(
            entity_df=frame, features=[project.Clicks.click_count]
        )


def test_an_undeclared_view_is_refused(project: Project) -> None:
    other = project.config.parent / "definitions/other.py"
    other.write_text(UNDECLARED.format(other=project.config.parent / "stats.parquet"))

    with pytest.raises(ValueError, match="other"):
        project.store.get_historical_features(
            entity_df=labels(), features=[load(other).Other.unseen]
        )


def test_a_feature_the_view_does_not_declare_is_refused(project: Project) -> None:
    unseen = Field(Int64)
    unseen.name = "unseen"
    unseen.view = "clicks"

    with pytest.raises(ValueError, match="unseen"):
        project.store.get_historical_features(entity_df=labels(), features=[unseen])


def test_a_field_never_bound_to_a_view_is_refused(project: Project) -> None:
    with pytest.raises(ValueError, match="not bound to a view"):
        project.store.get_historical_features(entity_df=labels(), features=[Field(Int64)])


def test_an_empty_feature_list_is_refused(project: Project) -> None:
    with pytest.raises(ValueError, match="at least one feature"):
        project.store.get_historical_features(entity_df=labels(), features=[])


def test_a_repeated_feature_is_refused(project: Project) -> None:
    with pytest.raises(ValueError, match="twice"):
        project.store.get_historical_features(
            entity_df=labels(),
            features=[project.Clicks.click_count, project.Clicks.click_count],
        )


def test_an_unknown_missing_policy_is_refused(project: Project) -> None:
    # `Any` because the value is deliberately outside the literal set. Typing it
    # precisely would make this a static error, and what is under test is the
    # runtime refusal.
    unknown: Any = "keep"

    with pytest.raises(ValueError, match="on_missing"):
        project.store.get_historical_features(
            entity_df=labels(),
            features=[project.Clicks.click_count],
            on_missing=unknown,
        )


def test_a_pandas_frame_is_refused_by_name(project: Project) -> None:
    class Pandas:
        """Stands in for a pandas DataFrame, which offers ``__dataframe__`` only."""

        def __dataframe__(self, *args: Any, **kwargs: Any) -> None:
            """Present the interchange protocol that Feather cannot read."""

    not_a_frame: Any = Pandas()

    with pytest.raises(TypeError, match="pandas DataFrame"):
        project.store.get_historical_features(
            entity_df=not_a_frame,
            features=[project.Clicks.click_count],
        )


def test_a_frame_without_arrow_buffers_is_refused(project: Project) -> None:
    not_a_frame: Any = {"user_id": [1]}

    with pytest.raises(TypeError, match="__arrow_c_stream__"):
        project.store.get_historical_features(
            entity_df=not_a_frame,
            features=[project.Clicks.click_count],
        )


def test_a_missing_definition_module_is_reported(tmp_path: Path) -> None:
    config = tmp_path / "feather.toml"
    config.write_text(SETTINGS, encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        FeatureStore(config)


def test_a_definition_the_core_rejects_is_reported(tmp_path: Path) -> None:
    (tmp_path / "definitions").mkdir()
    (tmp_path / "definitions/clicks.py").write_text(
        "from feather import Entity, FeatureView, Field, FileSource, feature_view\n"
        "from feather.types import Int64\n"
        "\n"
        "user_entity = Entity(name='user_id', join_key='user_id')\n"
        "clicks_source = FileSource(path='clicks.parquet')\n"
        "\n"
        "\n"
        "@feature_view(name='clicks', entity=user_entity, source=clicks_source)\n"
        "class Clicks(FeatureView):\n"
        "    pass\n",
        encoding="utf-8",
    )
    config = tmp_path / "feather.toml"
    config.write_text('project = "ads"\ndefinitions = ["definitions/clicks.py"]\n')

    with pytest.raises(ValueError, match="declares no fields"):
        FeatureStore(config)


def test_malformed_settings_are_reported(tmp_path: Path) -> None:
    config = tmp_path / "feather.toml"
    config.write_text('project = "ads"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="definitions"):
        FeatureStore(config)
