"""Definition models: the wire format the Rust core deserializes.

These are the models that cross into Rust. They bound the shape on the Python side,
so a malformed definition fails while it is being written rather than at engine
startup, but the core is the authority: it validates what it accepts, and these
constraints are a duplicate of the core's rules rather than the rule itself.

They are deliberately not the authoring surface: features are declared with
:mod:`feather.definitions`, which compiles to these. The split exists because a
reference to a feature has to be a checked object rather than a string, and only a
class-body declaration can be checked.

``Entity``, ``FileSource``, and ``PostgresSource`` are used directly when authoring as
well, because each is a leaf value with no schema of its own.
"""

import warnings
from typing import Annotated, Literal

import pydantic

from feather.types import WireDType

NonEmptyStr = Annotated[str, pydantic.StringConstraints(min_length=1)]


class DefinitionModel(pydantic.BaseModel):
    """Base for definition models.

    Definitions are data, not code: they are immutable, and an unknown key is a
    typo rather than something to ignore.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")


class Entity(DefinitionModel):
    """A join key that features are grouped by.

    `join_key` names the column in the source. The entity *values* are what the
    key encoder caps at 512 bytes per component, and those only exist at runtime,
    so that limit is enforced in Rust rather than here.
    """

    name: NonEmptyStr
    join_key: NonEmptyStr


class Field(DefinitionModel):
    """One feature and its storage type, as it crosses the boundary.

    `dtype` is a wire name, not a Python type. Rust maps it to an Arrow type
    (`int64` to `Int64`, `timestamp_micros` to a microsecond timestamp, and so
    on) and the value codec writes fixed-width columns at a fixed stride, which
    is why the set is closed.
    """

    name: NonEmptyStr
    dtype: WireDType


class FileSource(DefinitionModel):
    """A file-backed offline source.

    A file, local or reached through a URI scheme such as ``s3://``, read in the
    format ``format`` names. ``type`` is what the union below discriminates on, and it
    is defaulted so that ``FileSource(path="...")`` stays the way a file source is
    written; an omitted ``format`` is the default one, Parquet.

    ``format`` is a plain string rather than a ``Literal`` of the known names on
    purpose. The Rust core is the authority on which formats have readers, and it
    reports one it does not know against the view and the path that declared it, which
    is a better error than a second list here rejecting the value with no idea which
    source it came from.
    """

    type: Literal["file"] = "file"
    path: NonEmptyStr
    format: NonEmptyStr | None = None


# `schema` is the key the core expects for a Postgres table, and it is also the name of a
# deprecated method on `pydantic.BaseModel`, so pydantic warns that the field shadows it.
# The method is not used anywhere here, and renaming the field would put the wire contract
# behind an alias, which is a worse problem than the shadowing. Scoped to this class
# rather than the whole process, so a field that shadows something for a real reason
# still warns.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message='Field name "schema" in "PostgresSource" shadows an attribute',
        category=UserWarning,
    )

    class PostgresSource(DefinitionModel):
        """A table in a Postgres database.

        ``connection`` names a ``[connections]`` entry rather than carrying credentials,
        because a definition module is committed.
        """

        type: Literal["postgres"] = "postgres"
        connection: NonEmptyStr
        # The name is the wire key the core reads, and it also collides with the
        # deprecated `BaseModel.schema()` method. Pyrefly sees the override as
        # inconsistent because that method is not a field; nothing here calls it, and
        # the ignore is audited for staleness by `unused-ignore` in pyproject.toml.
        schema: NonEmptyStr  # pyrefly: ignore [bad-override]
        table: NonEmptyStr


Source = Annotated[FileSource | PostgresSource, pydantic.Field(discriminator="type")]
"""A view's source, discriminated on ``type``."""


class FeatureView(DefinitionModel):
    """Features derived from one source, keyed by one entity."""

    name: NonEmptyStr
    entities: Annotated[list[Entity], pydantic.Field(min_length=1)]
    source: Source
    features: Annotated[list[Field], pydantic.Field(min_length=1)]
    ttl_days: Annotated[int, pydantic.Field(gt=0)] | None = None
    timestamp_field: NonEmptyStr | None = None
    created_timestamp_field: NonEmptyStr | None = None
    pushable: bool = False
    """Whether this view accepts a push, and so whether its writes are guarded.

    ``False`` by default: a view nobody pushes to has one writer and no ordering to
    enforce, so it keeps the plain write path. A push to a view that did not declare
    this is an error rather than an unguarded write, because the next refresh would
    revert it.
    """

    @pydantic.model_validator(mode="after")
    def _check(self) -> "FeatureView":
        """Reject shapes the Rust engines cannot represent.

        Returns:
            The validated view.

        Raises:
            ValueError: If the view declares more than one entity, repeats a
                field name, or declares the same field as both timestamp and
                created timestamp.
        """
        if len(self.entities) > 1:
            raise ValueError(
                f"view {self.name!r} declares {len(self.entities)} entities; "
                "v1 supports exactly one per view"
            )
        names = [f.name for f in self.features]
        if len(set(names)) != len(names):
            raise ValueError(f"view {self.name!r} repeats a feature name")
        if (
            self.timestamp_field is not None
            and self.timestamp_field == self.created_timestamp_field
        ):
            raise ValueError(
                f"view {self.name!r} uses {self.timestamp_field!r} as both the event "
                "timestamp and the created timestamp"
            )
        return self


class FeatureService(DefinitionModel):
    """A named projection: a name and a list of ``view:feature`` references.

    Deliberately not a registry object. It carries no entities, no version, and
    no infrastructure, because entities come from the referenced views.
    """

    name: NonEmptyStr
    features: Annotated[list[NonEmptyStr], pydantic.Field(min_length=1)]

    @pydantic.field_validator("features")
    @classmethod
    def _check_references(cls, value: list[str]) -> list[str]:
        """Reject references that are not ``view:feature``.

        Returns:
            The validated references.

        Raises:
            ValueError: If a reference is malformed or repeated.
        """
        for reference in value:
            view, separator, feature = reference.partition(":")
            if not separator or not view or not feature:
                raise ValueError(f"malformed reference {reference!r}; expected 'view:feature'")
        if len(set(value)) != len(value):
            raise ValueError("a feature service repeats a reference")
        return value

    def references(self) -> list[tuple[str, str]]:
        """Split the references into ``(view, feature)`` pairs.

        Returns:
            One pair per reference, in declaration order.
        """
        pairs: list[tuple[str, str]] = []
        for reference in self.features:
            view, _, feature = reference.partition(":")
            pairs.append((view, feature))
        return pairs


class FeatureStoreConfig(DefinitionModel):
    """A project's definitions, as they cross the boundary.

    This is the object serialized and handed to Rust.
    """

    project: NonEmptyStr
    views: list[FeatureView] = pydantic.Field(default_factory=list)
    services: list[FeatureService] = pydantic.Field(default_factory=list)

    def view(self, name: str) -> FeatureView:
        """Look up a view by name.

        Returns:
            The view.

        Raises:
            KeyError: If no view has that name.
        """
        for candidate in self.views:
            if candidate.name == name:
                return candidate
        raise KeyError(f"no feature view named {name!r}")

    def validate_references(self) -> None:
        """Check references that span objects, which Pydantic cannot see.

        Raises:
            ValueError: If a view name is repeated, a service name is repeated,
                or a service references an unknown view or feature.
            KeyError: If a service references a view the project does not define.
        """
        view_names = [v.name for v in self.views]
        if len(set(view_names)) != len(view_names):
            raise ValueError("two feature views share a name")

        service_names = [s.name for s in self.services]
        if len(set(service_names)) != len(service_names):
            raise ValueError("two feature services share a name")

        for service in self.services:
            for view_name, feature_name in service.references():
                view = self.view(view_name)
                if all(f.name != feature_name for f in view.features):
                    raise ValueError(
                        f"feature service {service.name!r} references "
                        f"{view_name}:{feature_name}, but {view_name!r} has no such feature"
                    )
