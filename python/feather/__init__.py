"""Feather: an opinionated feature store with a Rust core.

Definitions are declared here in Python and validated by Pydantic. The engines
are in Rust; they receive the validated definitions as JSON and do not
re-validate them.

Importing this package does not require the compiled extension. Only the parts
that actually call into Rust do, so the definition layer stays usable and
testable without a build.
"""

from importlib.metadata import PackageNotFoundError, version

from feather.definitions import (
    Entity,
    FeatureService,
    FeatureStoreConfig,
    FeatureView,
    Field,
    FileSource,
)
from feather.types import DType

try:
    __version__ = version("feather-store")
except PackageNotFoundError:  # pragma: no cover - running from a bare source tree
    __version__ = "0.0.0+uninstalled"

__all__ = [
    "DType",
    "Entity",
    "FeatureService",
    "FeatureStoreConfig",
    "FeatureView",
    "Field",
    "FileSource",
    "__version__",
]
