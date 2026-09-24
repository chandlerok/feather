"""Feather: an opinionated feature store with a Rust core.

Definitions are declared in Python and validated by Pydantic before anything
crosses into Rust. The engines are in Rust; they receive the validated
definitions as JSON and do not re-validate them.

Importing this package does not require the compiled extension. Only the parts
that actually call into Rust do, so the definition layer stays usable and
testable without a build.
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
    __version__ = version("feather-store")
except PackageNotFoundError:  # pragma: no cover - running from a bare source tree
    __version__ = "0.0.0+uninstalled"

__all__ = [
    "Boolean",
    "DType",
    "Entity",
    "FeatureService",
    "FeatureStoreConfig",
    "FeatureView",
    "Field",
    "FileSource",
    "Float64",
    "Int64",
    "TimestampMicros",
    "Utf8",
    "WireDType",
    "__version__",
    "config_to_wire",
    "feature_view",
    "view_fields",
    "view_to_wire",
]
