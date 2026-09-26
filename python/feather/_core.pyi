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

def write_demo_data(directory: str) -> list[str]:
    """Write the demo project's two Parquet files into ``directory/data``.

    The feature table is one row per user per day for the last two weeks, and the
    label set is five rows per user, each half a day after a feature row, with a
    target taken from the following day. Both carry timestamps as int64
    microseconds since the epoch, relative to the current day rather than fixed
    dates, so the generated view's 30-day TTL still holds whenever it is run.

    Args:
        directory: The project directory, created if it does not exist. ``data``
            is created inside it.

    Returns:
        The paths written, the feature table first.

    Raises:
        OSError: If a directory or file cannot be created.
    """
    ...

class ArrowStreamExportable(Protocol):
    """An object that exports Arrow data through the PyCapsule interface.

    What :meth:`FeatureStore.get_historical_features` accepts and returns. A Polars
    ``DataFrame`` satisfies it, and so does a pyarrow ``Table``, an ``arro3`` table,
    or anything else implementing the interface.
    """

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object:
        """Export the data as an Arrow C stream capsule."""
        ...

class ViewRefresh:
    """One view's refresh, as :meth:`FeatureStore.materialize` reports it."""

    name: str
    rows: int
    max_event_timestamp_micros: int | None
    """The newest event timestamp any written row carried, in microseconds since the epoch, or
    ``None`` when the refresh wrote no rows."""
    elapsed_seconds: float

class MaterializeReport:
    """What one call to :meth:`FeatureStore.materialize` did."""

    views: list[ViewRefresh]
    retired: list[str]
    """The views a previous refresh declared and this one does not, whose fields were removed.
    Empty on every run but the one that follows a rename or a removal."""
    total_rows: int
    elapsed_seconds: float

class FeatureStore:
    """A project's definitions, the engine that joins them, and the store they are served from."""

    def __init__(self, settings_path: str, definitions_json: str) -> None: ...
    def materialize(self, views: list[str] | None = None) -> MaterializeReport:
        """Refresh feature values from their sources into the online store.

        Args:
            views: The views to refresh, by name, or ``None`` for every view the project
                declares.

        Returns:
            What each refreshed view wrote, and the views this run retired.

        Raises:
            ValueError: If a named view is not declared, or a source cannot be read as its view
                declares it.
            ConnectionError: If the settings declare a Valkey that cannot be reached.
        """
        ...
    def get_online_features(
        self,
        entity_frame: ArrowStreamExportable,
        features: list[str],
    ) -> ArrowStreamExportable:
        """Read feature values for a frame of entities from the online store.

        Args:
            entity_frame: The entities to read, exporting Arrow buffers through
                ``__arrow_c_stream__``. Its join key column is named after the views' entity's
                ``join_key``.
            features: ``view:feature`` references, in the order the caller wants the columns
                back.

        Returns:
            The entity frame's own columns in input order, then one column per requested feature
            in request order. A value that is missing, expired, or stored under a schema the
            definition no longer matches is a null.

        Raises:
            ValueError: If a reference is malformed, names an unknown view or feature, or is
                requested twice; if the requested views do not declare the same entity, which is
                both the same join key and the same entity name; if ``entity_frame`` lacks the
                join key column or holds a null in it; or if a requested feature name collides
                with a column of the frame.
            ConnectionError: If the settings declare a Valkey that cannot be reached.
        """
        ...
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
