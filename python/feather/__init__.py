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
        AttributeError: If ``name`` is not one of them, or if it is and the compiled
            extension is not built. An ``AttributeError`` rather than the ``ImportError`` the
            import itself raises, because this runs under ``hasattr`` and under ``from feather
            import *``, where an unbuilt extension means the name is absent rather than that
            the import failed. The message says which extension to build.
    """
    if name in _COMPILED_EXPORTS:
        try:
            from feather import _core
        except ImportError as unbuilt:
            raise AttributeError(
                f"{__name__} has no attribute {name!r}: it lives in the compiled extension "
                f"`feather._core`, which is not built ({unbuilt})"
            ) from unbuilt
        return getattr(_core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """The names this module offers, including the lazily resolved ones.

    The compiled names are listed only when the extension imports, so a completion list does
    not advertise a name that would raise.

    Returns:
        The module's attributes.
    """
    names = [*globals()]
    try:
        from feather import _core  # noqa: F401
    except ImportError:
        return sorted(names)
    names.extend(_COMPILED_EXPORTS)
    return sorted(names)


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
