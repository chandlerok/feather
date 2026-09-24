"""The static guarantees, pinned by the type checker rather than by running code.

Two things are asserted here, and neither is an ordinary test:

- The positive cases use `assert_type`, so if a reference stops being a
  `Field[Int64]` this file fails the type check.
- The negative cases carry a `# type: ignore[...]`. Because mypy runs with
  `warn_unused_ignores`, an ignore that is no longer needed is itself an error,
  so a check that quietly stops firing fails this file instead of passing.

Two shapes are deliberately avoided, because the two checkers we run disagree on
them and a single ignore cannot satisfy both:

- A decorator applied to a bad class: mypy attributes the error to the class
  statement, pyright to the decorator line. The application is therefore written
  out as a call, where both agree.
- `assert_type` on a marker's `wire`: mypy infers the specific literal and pyright
  the declared union. The closed set is pinned by the marker case below and by
  `test_types`, which needs no agreement about inference.

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
    Utf8,
    feature_view,
)

USER = Entity(name="user_id", join_key="user_id")
SOURCE = FileSource(path="data/user_stats.parquet")


@feature_view(name="user_clicks", entity=USER, source=SOURCE)
class UserClicks(FeatureView):
    click_count = Field(Int64)
    label = Field(Utf8)


class NotAView:
    """Not a declared view, for the misuse case below."""


def _positive() -> None:
    """References are typed, and configuration is an ordinary checked call."""
    assert_type(UserClicks.click_count, Field[Int64])
    assert_type(UserClicks.label, Field[Utf8])

    # A selection may mix dtypes, and a whole view is allowed alongside fields.
    FeatureService(name="mixed", features=[UserClicks.click_count, UserClicks.label])
    FeatureService(name="whole", features=[UserClicks])


def _negative() -> None:
    """Every line here must fail the type check, or its ignore is unused."""
    # A misspelled feature is an attribute error, not a runtime lookup failure.
    _ = UserClicks.click_cout  # type: ignore[attr-defined]

    # A string reference cannot be checked at all, so the checked path refuses it.
    FeatureService(name="raw", features=["user_clicks:click_count"])  # type: ignore[list-item]

    # A dtype has to be a marker rather than an arbitrary Python type.
    Field(int)  # type: ignore[type-var]

    # Configuration lives on the decorator, so its typos are caught here.
    @feature_view(name="x", entity=USER, source=SOURCE, ttl_dayz=1)  # type: ignore[call-arg]
    class BadConfig(FeatureView):
        f = Field(Int64)

    # The decorator only applies to a declared view.
    feature_view(name="x", entity=USER, source=SOURCE)(NotAView)  # type: ignore[type-var]


def _bad_marker() -> None:
    """A marker cannot name a wire string Rust does not know.

    Declared inside a function that is never called, so it does not join
    `DType.__subclasses__()` and `test_types` still sees the real set.
    """

    class Int32(DType):
        wire = "int32"  # type: ignore[assignment]
