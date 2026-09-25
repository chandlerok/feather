"""The project paths: declared definitions in, a training set or an online read out.

Definitions are imported rather than compiled, so there is no registry and no
generated artifact. What this module adds is the resolution the core cannot do
itself: a reference to a feature is a Python object, and turning it into a
``view:feature`` pair needs the view name the decorator bound to it.

Nothing here imports a dataframe library. The frame crosses into Rust through the
Arrow PyCapsule interface, so the caller's own Arrow buffers are read directly and
``feather`` never needs Polars to be installed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from feather.definitions import (
    FeatureStoreConfig,
    FeatureView,
    Field,
    config_to_wire,
)
from feather.settings import DEFAULT_PATH, load_settings

if TYPE_CHECKING:
    import os
    from collections.abc import Sequence

    from feather._core import ArrowStreamExportable, MaterializeReport

MissingPolicy = Literal["null", "drop"]
"""What to do with a label row whose features are missing or expired."""


class FeatureStore:
    """A project: its imported definitions, and the engine that joins them.

    The definitions ``feather.toml`` names are imported, and importing them runs no
    logic beyond the declarations themselves, so a project can be loaded without
    pulling in a dataframe library or a warehouse client.
    """

    def __init__(self, path: str | os.PathLike[str] = DEFAULT_PATH) -> None:
        """Read a project's settings and definitions, then open the engine.

        Args:
            path: The ``feather.toml`` to read. A path in its ``definitions`` list
                is resolved against the file's own directory, not the working
                directory.

        Raises:
            FileNotFoundError: If the settings file, or a definition module it
                names, does not exist.
            OSError: If a file exists but cannot be read, or the engine's spill
                directory cannot be created.
            ValueError: If the settings or a definition are malformed. The core is
                what decides this, so the message comes from the same place an
                engine failure would.
            ImportError: If a definition module fails to import.
        """
        settings = load_settings(path)
        definitions = config_to_wire(
            FeatureStoreConfig(
                project=settings.project,
                views=_import_views(settings.definitions, Path(path).parent),
            )
        )
        # Imported here rather than at module scope, so that importing `feather`
        # still works without the compiled extension. Only this call needs it.
        from feather import _core

        self._store = _core.FeatureStore(str(path), definitions.model_dump_json())

    def get_historical_features(
        self,
        entity_df: ArrowStreamExportable,
        features: Sequence[Field[Any]],
        *,
        timestamp_column: str = "event_timestamp",
        strict: bool = False,
        on_missing: MissingPolicy = "null",
    ) -> ArrowStreamExportable:
        """Join the requested features onto an entity frame.

        The features are read at or before each label row's own timestamp, which is
        what keeps a training set free of the future. Each label row keeps its
        position: the result is row-for-row aligned with ``entity_df`` unless a row
        was dropped.

        Args:
            entity_df: The label rows, as an object exporting Arrow buffers through
                ``__arrow_c_stream__``. A Polars DataFrame does. The transfer is by
                reference, so Rust reads the frame's existing buffers rather than a
                serialized copy.
            features: The declared fields to join, as ``View.field``. The order
                given here is the order of the returned columns, and each field's
                own view decides which source is read.
            timestamp_column: The entity frame's label timestamp column. The
                source's timestamp column is the view's own ``timestamp_field``,
                which defaults to ``event_timestamp`` in the core.
            strict: Exclude a feature row at exactly the label time instead of
                including it. The default is inclusive.
            on_missing: ``"null"`` keeps a label row whose value is missing or
                expired and leaves that value null, which is the default because
                dropping rows silently changes the size and class balance of a
                training set. ``"drop"`` removes the row. A row is removed only if
                every requested view has nothing for it, so a kept row is never
                left holding a null.

        Returns:
            An Arrow table: the entity frame's own columns in the surviving row
            order, then one column per requested feature. Only the input transfer
            is zero-copy; the join result is a new relation that DuckDB
            materializes. The table exports ``__arrow_c_stream__``, so
            ``polars.DataFrame(result)`` accepts it, as do pyarrow, arro3, and
            nanoarrow.

        Raises:
            TypeError: If ``entity_df`` does not export Arrow buffers. A pandas
                DataFrame is the common case, and the message names the conversion.
            ValueError: If a feature is not bound to a view, is requested twice,
                names a view or feature the project does not declare, or collides
                with a column of ``entity_df``; or if a join fails.
        """
        if not hasattr(entity_df, "__arrow_c_stream__"):
            raise TypeError(_why_not_arrow(entity_df))
        return self._store.get_historical_features(
            entity_df,
            _references(features),
            timestamp_column,
            strict,
            on_missing,
        )

    def materialize(self, views: Sequence[str] | None = None) -> MaterializeReport:
        """Refresh feature values from their sources into the online store.

        A full refresh: every selected view is recomputed from its source and every value is
        overwritten in place. There is no watermark and nothing partial to reconcile, so a run
        that fails leaves the values it had not yet overwritten as they were and the call can
        simply be repeated. A run that is still going serves a mix of the values it has written
        and the ones it has not reached yet. A view that a previous refresh declared and this one
        does not is retired: its fields are removed from every entity that still carries them,
        and it leaves the project's view registry.

        Args:
            views: The views to refresh, by name, or ``None`` for every view the project
                declares. Refreshing a subset leaves the other views' values and their registry
                entries alone.

        Returns:
            A report naming each refreshed view, the rows it wrote, the newest event timestamp
            those rows carried, and how long each took, plus the views this run retired and the
            totals across the run. The timestamp is in microseconds since the epoch, matching
            every other timestamp in this library.

        Raises:
            ValueError: If a named view is not declared, or a source cannot be read as its view
                declares it.
            ConnectionError: If ``feather.toml`` declares a Valkey that cannot be reached.
        """
        return self._store.materialize(None if views is None else list(views))

    def get_online_features(
        self,
        entity_df: ArrowStreamExportable,
        features: Sequence[Field[Any]],
    ) -> ArrowStreamExportable:
        """Read feature values for a frame of entities from the online store.

        This is the serving read, so a value's freshness is what decides whether it is returned:
        a value older than its view's ``ttl_days`` reads as null, exactly as a value that was
        never written does. One store read is issued per entity however many views are
        requested, which is the rule the storage layout exists to make possible.

        Args:
            entity_df: The entities to read, as an object exporting Arrow buffers through
                ``__arrow_c_stream__``. A Polars DataFrame does. Its join key column is named
                after the views' entity's ``join_key``, and one call reads one entity type.
            features: The declared fields to read, as ``View.field``. The order given here is
                the order of the returned columns, and each field's own view decides which
                stored vector it comes from.

        Returns:
            An Arrow table: the entity frame's own columns in input order, then one column per
            requested feature in request order, one row per input row. A feature whose value is
            missing, expired, or stored under a schema the current definition no longer matches
            is a null. The three causes are deliberately not distinguished here; the online
            contract is nulls.

        Raises:
            TypeError: If ``entity_df`` does not export Arrow buffers. A pandas DataFrame is the
                common case, and the message names the conversion.
            ValueError: If a feature is not bound to a view, is requested twice, names a view or
                feature the project does not declare, or collides with a column of
                ``entity_df``; if the requested views do not declare the same entity, which is
                both the same join key and the same entity name, since one call reads one entity
                type; if ``entity_df`` lacks the join key column or holds a null in it; or if the
                read fails.
            ConnectionError: If ``feather.toml`` declares a Valkey that cannot be reached.
        """
        if not hasattr(entity_df, "__arrow_c_stream__"):
            raise TypeError(_why_not_arrow(entity_df))
        return self._store.get_online_features(entity_df, _references(features))


def _references(features: Sequence[Field[Any]]) -> list[str]:
    """Compile field references into the ``view:feature`` form the core takes.

    Args:
        features: The declared fields to join.

    Returns:
        One reference per field, in the order given.

    Raises:
        TypeError: If an entry is not a declared field.
        ValueError: If the list is empty, a field was never bound to a view, or the
            same feature is named twice.
    """
    if not features:
        raise ValueError("features must name at least one feature")
    references: list[str] = []
    for feature in features:
        if not isinstance(feature, Field):
            raise TypeError(f"expected a declared field such as View.name, got {feature!r}")
        if not feature.view:
            raise ValueError(
                f"{feature!r} is not bound to a view; declare it in a feature view class body"
            )
        references.append(f"{feature.view}:{feature.name}")
    if len(set(references)) != len(references):
        raise ValueError("the same feature is requested twice")
    return references


def _why_not_arrow(entity_df: object) -> str:
    """Explain why a frame cannot be read, naming pandas because it is the common case.

    Args:
        entity_df: Whatever the caller passed.

    Returns:
        The message for the ``TypeError``.
    """
    if hasattr(entity_df, "__dataframe__"):
        return (
            "a pandas DataFrame does not export Arrow buffers, so reading it would copy "
            "its data; convert it first, with polars.DataFrame(entity_df) or "
            "pyarrow.Table.from_pandas(entity_df)"
        )
    return (
        f"{type(entity_df).__name__} does not export the Arrow __arrow_c_stream__ "
        "interface, so its buffers cannot be read by reference"
    )


def _import_views(modules: Sequence[str], root: Path) -> list[type[FeatureView]]:
    """Import the modules a project declares, and collect the views they define.

    Loaded by path rather than by module name. A definition module depends only on
    ``feather`` by design, so it needs no package context, and loading it this way
    puts nothing on ``sys.path``.

    Args:
        modules: The paths ``feather.toml`` lists.
        root: The directory holding ``feather.toml``, which a relative path is
            resolved against.

    Returns:
        The declared views, in declaration order, module by module.

    Raises:
        FileNotFoundError: If a listed module does not exist.
        ImportError: If a module fails to import.
    """
    views: list[type[FeatureView]] = []
    for module in modules:
        path = Path(module)
        if not path.is_absolute():
            path = root / path
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load a definition module from {path}")
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        # Filtered by `__module__` rather than taking every value, so a view
        # imported into a second module is collected once, where it was declared.
        views.extend(
            value
            for value in vars(loaded).values()
            if isinstance(value, type)
            and issubclass(value, FeatureView)
            and value is not FeatureView
            and value.__module__ == spec.name
        )
    return views


__all__ = ["FeatureStore", "MissingPolicy"]
