"""Feather: an opinionated feature store with a Rust core.

Definitions are declared in Python and validated by Pydantic at authoring time, so
a mistake is reported in the language it was written in. That is an early warning,
not the contract: the Rust core validates everything it accepts, so an engine never
works from a value it cannot represent.

Importing this package does not require the compiled extension. Only the calls that
reach into Rust do, so the definition layer stays usable and testable without a
build.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from feather._wire import Entity, FileSource, PostgresSource
from feather.definitions import (
    FeatureService,
    FeatureStoreConfig,
    FeatureView,
    Field,
    config_to_wire,
    feature_view,
    view_fields,
    view_to_wire,
)
from feather.settings import FeatherSettings, load_settings
from feather.store import FeatureStore, MissingPolicy
from feather.types import (
    Boolean,
    DType,
    Float64,
    Int64,
    TimestampMicros,
    Utf8,
    WireDType,
)

try:
    __version__ = version("feather-py")
except PackageNotFoundError:  # pragma: no cover - running from a bare source tree
    __version__ = "0.0.0+uninstalled"

_COMPILED_EXPORTS = ("MaterializeReport", "ViewRefresh")
"""Names that live in the compiled extension, resolved on first use.

Exported here so a caller can name the type it gets back from
:meth:`feather.FeatureStore.materialize` without reaching into ``feather._core``, and resolved
lazily because importing this package must not require a built extension. The type checker reads
the import below rather than the runtime lookup, which is why both exist.
"""

if TYPE_CHECKING:
    from feather._core import MaterializeReport, ViewRefresh


def __getattr__(name: str) -> object:
    """Resolve a name that needs the compiled extension.

    Args:
        name: The attribute Python could not find.

    Returns:
        The attribute, for one of the names in :data:`_COMPILED_EXPORTS`.

    Raises:
        AttributeError: If ``name`` is not one of them.
    """
    if name in _COMPILED_EXPORTS:
        from feather import _core

        return getattr(_core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """The names this module offers, including the lazily resolved ones.

    Returns:
        The module's attributes.
    """
    return sorted([*globals(), *_COMPILED_EXPORTS])


__all__ = [
    "Boolean",
    "DType",
    "Entity",
    "FeatherSettings",
    "FeatureService",
    "FeatureStore",
    "FeatureStoreConfig",
    "FeatureView",
    "Field",
    "FileSource",
    "Float64",
    "Int64",
    "MaterializeReport",
    "MissingPolicy",
    "PostgresSource",
    "TimestampMicros",
    "Utf8",
    "ViewRefresh",
    "WireDType",
    "__version__",
    "config_to_wire",
    "feature_view",
    "load_settings",
    "view_fields",
    "view_to_wire",
]
