"""Type stub for the compiled extension.

Hand-written rather than generated. PyO3's `generate-stubs` is self-described as
in development and cannot introspect a function-style `#[pymodule]`, which is the
shape maturin generates. Hand-writing the stub for the small surface we expose is
cheaper than depending on that, and it keeps type checking independent of whether
the extension has been built.

Keep this in sync with `crates/feather-py/src/lib.rs`.
"""

from typing import Protocol

__version__: str

class ArrowStreamExportable(Protocol):
    """An object that exports Arrow data through the PyCapsule interface.

    What :meth:`FeatureStore.get_historical_features` accepts and returns. A Polars
    ``DataFrame`` satisfies it, and so does a pyarrow ``Table``, an ``arro3`` table,
    or anything else implementing the interface.
    """

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object:
        """Export the data as an Arrow C stream capsule."""
        ...

class FeatureStore:
    """A project's definitions, and the DuckDB engine that joins them."""

    def __init__(self, settings_path: str, definitions_json: str) -> None: ...
    def get_historical_features(
        self,
        entity_frame: ArrowStreamExportable,
        features: list[str],
        label_timestamp_column: str = ...,
        strict: bool = ...,
        on_missing: str = ...,
    ) -> ArrowStreamExportable: ...

def load_settings(path: str) -> str:
    """Read and validate a ``feather.toml``, as JSON.

    Args:
        path: The file to read.

    Returns:
        The validated settings, serialized. `feather.settings` wraps this.

    Raises:
        FileNotFoundError: If no file is at ``path``.
        OSError: If the file exists but cannot be read.
        ValueError: If it is not valid TOML, does not match the schema, or holds a
            ``${VAR}`` reference that cannot be resolved.
    """
    ...
