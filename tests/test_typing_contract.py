"""The static guarantees, pinned by the type checker rather than by running code.

Two things are asserted here, and neither is an ordinary test:

- The positive cases use `assert_type`, so a reference that stops being a
  `Field[Int64]` fails the type check.
- The negative cases carry a `# pyrefly: ignore[kind]`. `unused-ignore` is enabled
  in `[tool.pyrefly.errors]`, so an ignore that is no longer needed is itself an
  error: a check that quietly stops firing fails here instead of passing.

The dialect is load-bearing. Pyrefly also suppresses a diagnostic for mypy's
`# type: ignore[code]`, but it does not audit that form for staleness, so one of
those would pass whether or not the error still exists. Only its own
`# pyrefly: ignore[kind]` form gets both halves, and the kind has to be pyrefly's
kind name rather than mypy's.

Nothing in here is called. The module exists to be type checked.
"""

from typing import assert_type

from feather import (
    DType,
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    Int64,
    PostgresSource,
    Utf8,
    feature_view,
)

USER = Entity(name="user_id", join_key="user_id")
SOURCE = FileSource(path="data/user_stats.parquet")
POSTGRES = PostgresSource(connection="pg_prod", schema="public", table="user_stats")


@feature_view(name="user_clicks", entity=USER, source=SOURCE)
class UserClicks(FeatureView):
    click_count = Field(Int64)
    label = Field(Utf8)


@feature_view(name="user_stats", entity=USER, source=POSTGRES)
class UserStats(FeatureView):
    order_count = Field(Int64)


class NotAView:
    """Not a declared view, for the misuse case below."""


def _positive() -> None:
    """References are typed, and configuration is an ordinary checked call."""
    assert_type(UserClicks.click_count, Field[Int64])
    assert_type(UserClicks.label, Field[Utf8])

    # Either source kind is a source, and neither narrows what the decorator accepts.
    assert_type(UserStats.order_count, Field[Int64])

    # A selection may mix dtypes, and a whole view is allowed alongside fields.
    FeatureService(name="mixed", features=[UserClicks.click_count, UserClicks.label])
    FeatureService(name="whole", features=[UserClicks])


def _negative() -> None:
    """Every line here must fail the type check, or its ignore is unused."""
    # A misspelled feature is an attribute error, not a runtime lookup failure.
    _ = UserClicks.click_cout  # pyrefly: ignore[missing-attribute]

    # A string reference cannot be checked at all, so the checked path refuses it.
    raw_reference = ["user_clicks:click_count"]
    FeatureService(name="raw", features=raw_reference)  # pyrefly: ignore[bad-argument-type]

    # A dtype has to be a marker rather than an arbitrary Python type.
    Field(int)  # pyrefly: ignore[bad-specialization]

    # Configuration lives on the decorator, so its typos are caught here.
    @feature_view(
        name="x",
        entity=USER,
        source=SOURCE,
        ttl_dayz=1,  # pyrefly: ignore[unexpected-keyword]
    )
    class BadConfig(FeatureView):
        f = Field(Int64)

    # The decorator only applies to a declared view.
    apply_view = feature_view(name="x", entity=USER, source=SOURCE)
    apply_view(NotAView)  # pyrefly: ignore[bad-specialization]

    # A source is a model rather than a path, so a bare string is rejected here rather
    # than at the first read.
    feature_view(
        name="x",
        entity=USER,
        source="data/user_stats.parquet",  # pyrefly: ignore[bad-argument-type]
    )


def _bad_marker() -> None:
    """A marker cannot name a wire string Rust does not know.

    Declared inside a function that is never called, so it does not join
    `DType.__subclasses__()` and `test_types` still sees the real set.
    """

    class Int32(DType):
        wire = "int32"  # pyrefly: ignore[bad-assignment]
