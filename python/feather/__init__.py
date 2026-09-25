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

from feather._wire import Entity, FileSource
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
    "MissingPolicy",
    "TimestampMicros",
    "Utf8",
    "WireDType",
    "__version__",
    "config_to_wire",
    "feature_view",
    "load_settings",
    "view_fields",
    "view_to_wire",
]
