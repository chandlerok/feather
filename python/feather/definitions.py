"""Feature definitions.

Definitions are Python classes. A field's dtype is a type parameter and its name
is the attribute it is assigned to, so a reference to a feature is a checked
attribute access rather than a string lookup:

    @feature_view(name="user_clicks", entity=user, source=FileSource(path="p"))
    class UserClicks(FeatureView):
        click_count = Field(Int64)

    UserClicks.click_count        # Field[Int64]
    UserClicks.click_cout         # a type error, not a runtime lookup failure

Configuration is decorator arguments rather than class attributes, so the class
body is only the schema. That is what lets a feature be called ``name``,
``source``, or ``ttl_days``: there is nothing in the body for it to collide with.
The base class is empty for the same reason, and compiling is a function rather
than a method, so not even a method name is reserved.

There is deliberately no string form on this path. A string cannot be checked at
all, so ``features=`` takes objects; it is a type error to pass
``"user_clicks:click_count"``. Callers that need a string, such as a debug or
ad-hoc path, use the wire models in :mod:`feather._wire`, which are documented as
unchecked.

Declared views compile to those wire models, which is what Rust deserializes, so
Rust sees exactly the same JSON as it did before this layer existed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

from feather import _wire
from feather.types import DType

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from feather._wire import Entity, Source

T = TypeVar("T", bound=DType, covariant=True)
"""A field's dtype. Covariant so a selection can mix dtypes in one list."""


@dataclass(frozen=True)
class _ViewConfig:
    """What the decorator carries, kept out of the schema namespace.

    Attributes:
        name: The view's wire name.
        entity: The entity the view is keyed by.
        source: The offline source the features are read from.
        ttl_days: Whole days before a value is stale, or None for no expiry.
        timestamp_field: Source column holding the event timestamp.
        created_timestamp_field: Source column holding the created timestamp.
    """

    name: str
    entity: Entity
    source: Source
    ttl_days: int | None
    timestamp_field: str | None
    created_timestamp_field: str | None


class Field(Generic[T]):
    """A declared feature.

    Assigned in a :class:`FeatureView` class body, where the attribute name
    becomes the feature name.
    """

    name: str
    view: str
    dtype: type[T]

    def __init__(self, dtype: type[T]) -> None:
        """Record the storage type.

        Args:
            dtype: One of the markers in :mod:`feather.types`.
        """
        self.dtype = dtype
        self.name = ""
        self.view = ""

    def __set_name__(self, owner: type[Any], name: str) -> None:
        """Take the attribute name as the feature name.

        Args:
            owner: The class the field was assigned in. Unused: the view name is
                bound by the decorator, which runs once the class exists.
            name: The attribute name.
        """
        self.name = name

    def __repr__(self) -> str:
        """Render as a qualified name and wire type.

        Returns:
            The reference, such as ``user_clicks:click_count (int64)``.
        """
        qualified = f"{self.view}:{self.name}" if self.view else self.name
        return f"{qualified} ({self.dtype.wire})"


class FeatureView:
    """Marker base for a declared view.

    Deliberately empty, and pinned by a test. Any attribute here would take that
    name away from the schema, so a view could no longer declare a feature
    called ``config`` or ``fields``. The configuration lives on the decorator
    instead, under a dunder name.
    """

    __feather_view__: ClassVar[_ViewConfig]


S = TypeVar("S", bound="type[FeatureView]")
"""A declared view class. Bounded so a decorator cannot be applied to anything else."""


def feature_view(
    *,
    name: str,
    entity: Entity,
    source: Source,
    ttl_days: int | None = None,
    timestamp_field: str | None = None,
    created_timestamp_field: str | None = None,
) -> Callable[[S], S]:
    """Declare a feature view around a schema class.

    Args:
        name: The view's wire name. This is what the key layout and the
            ``view:feature`` convention use.
        entity: The entity the view is keyed by. Not a list: v1 keys one hash per
            entity, so the wire model rejects more than one.
        source: The offline source the features are read from: ``FileSource`` for a
            file, which names the format it is read as, or ``PostgresSource`` for a
            table in a configured database.
        ttl_days: Whole days before a value is stale, or None for no expiry.
            Deliberately not a duration: Rust stores whole days, so a
            ``timedelta`` would silently truncate.
        timestamp_field: Source column holding the event timestamp. Defaults to
            ``event_timestamp`` in Rust, not here.
        created_timestamp_field: Source column holding the created timestamp.

    Returns:
        A decorator that returns the class unchanged, with its configuration
        attached and its fields bound to the view name.

    Raises:
        ValueError: If the class declares no fields.
        pydantic.ValidationError: If the resulting view is malformed. Raised at
            decoration, so a bad definition fails at import rather than when an
            engine starts.
    """

    def decorate(cls: S) -> S:
        config = _ViewConfig(
            name=name,
            entity=entity,
            source=source,
            ttl_days=ttl_days,
            timestamp_field=timestamp_field,
            created_timestamp_field=created_timestamp_field,
        )
        fields = view_fields(cls)
        for field in fields:
            field.view = name
        # Cast rather than assigning on `cls` directly: S is bounded by
        # type[FeatureView], and a checker may narrow an assignment to that bound
        # type variable instead of resolving the ClassVar through it.
        cast("type[FeatureView]", cls).__feather_view__ = config
        # Compiling is validating, so doing it here makes a malformed view fail
        # at import rather than at first use.
        view_to_wire(cls)
        return cls

    return decorate


def view_fields(view: type[FeatureView]) -> tuple[Field[Any], ...]:
    """Return a view's declared fields, in declaration order.

    Args:
        view: The declared view class.

    Raises:
        ValueError: If the class declares no fields, which is almost always the
            decorator applied to the wrong class.
    """
    fields = tuple(value for value in vars(view).values() if isinstance(value, Field))
    if not fields:
        raise ValueError(
            f"feature view {view.__name__} declares no fields; "
            "is the decorator applied to the schema class?"
        )
    return fields


def view_to_wire(view: type[FeatureView]) -> _wire.FeatureView:
    """Compile a declared view to the wire model Rust deserializes.

    Args:
        view: The declared view class.

    Returns:
        The compiled view.

    Raises:
        pydantic.ValidationError: If the view is one the engines cannot
            represent.
    """
    config = view.__feather_view__
    return _wire.FeatureView(
        name=config.name,
        entities=[config.entity],
        source=config.source,
        features=[_wire.Field(name=f.name, dtype=f.dtype.wire) for f in view_fields(view)],
        ttl_days=config.ttl_days,
        timestamp_field=config.timestamp_field,
        created_timestamp_field=config.created_timestamp_field,
    )


class FeatureService:
    """A named projection: a name and the features it exposes.

    Accepts declared fields, or a whole view to mean every field on it. The
    name is the wire handle: ``FeatureStore.serve()`` takes it, and the server
    resolves this field set against the project's definitions once at startup,
    so a service naming a view or a feature the project does not declare fails
    at boot rather than on the first request. In-process reads do not use it;
    they name fields directly.

    Deliberately not a registry object: it carries no entities, no version, and
    no infrastructure, because entities come from the referenced views.
    """

    name: str
    features: tuple[Field[Any], ...]

    def __init__(self, name: str, features: Sequence[Field[Any] | type[FeatureView]]) -> None:
        """Resolve the references to fields.

        Args:
            name: The service name.
            features: Declared fields, or views meaning all of their fields.

        Raises:
            ValueError: If a view in ``features`` declares no fields.
        """
        self.name = name
        resolved: list[Field[Any]] = []
        for item in features:
            if isinstance(item, Field):
                resolved.append(item)
            else:
                resolved.extend(view_fields(item))
        self.features = tuple(resolved)


class FeatureStoreConfig:
    """A project: the views and services it declares.

    Thin on purpose. It holds declarations and compiles them; the wire model owns
    the invariants.
    """

    project: str
    views: tuple[type[FeatureView], ...]
    services: tuple[FeatureService, ...]

    def __init__(
        self,
        project: str,
        views: Sequence[type[FeatureView]] = (),
        services: Sequence[FeatureService] = (),
    ) -> None:
        """Record the project.

        Args:
            project: The project name.
            views: The declared views.
            services: The declared services.
        """
        self.project = project
        self.views = tuple(views)
        self.services = tuple(services)


def config_to_wire(config: FeatureStoreConfig) -> _wire.FeatureStoreConfig:
    """Compile a project to the wire model Rust deserializes.

    Args:
        config: The project.

    Returns:
        The compiled project.

    Raises:
        KeyError: If a service references a view the project does not declare.
        pydantic.ValidationError: If a definition is malformed.
    """
    wire = _wire.FeatureStoreConfig(
        project=config.project,
        views=[view_to_wire(view) for view in config.views],
        services=[
            _wire.FeatureService(
                name=service.name,
                features=[f"{field.view}:{field.name}" for field in service.features],
            )
            for service in config.services
        ],
    )
    wire.validate_references()
    return wire


__all__ = [
    "FeatureService",
    "FeatureStoreConfig",
    "FeatureView",
    "Field",
    "config_to_wire",
    "feature_view",
    "view_fields",
    "view_to_wire",
]
