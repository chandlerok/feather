"""Feature definitions.

Python is the validating authority. Rust mirrors these models and deserializes
them, so this module owns the field names and the wire format; the Rust side
does not re-validate what Pydantic has already checked.

The one thing Pydantic cannot check is a cross-reference, such as a feature
service naming a view that does not exist. That is validated by
:meth:`FeatureStoreConfig.validate_references`.
"""

from typing import Annotated

import pydantic

from feather.types import DType

NonEmptyStr = Annotated[str, pydantic.StringConstraints(min_length=1)]

# Mirrors MAX_COMPONENT_LEN in feather-core's key module.
MAX_ENTITY_KEY_COMPONENT_LEN = 512


class DefinitionModel(pydantic.BaseModel):
    """Base for definition models.

    Definitions are data, not code: they are immutable, and an unknown key is a
    typo rather than something to ignore.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")


class Entity(DefinitionModel):
    """A join key that features are grouped by."""

    name: NonEmptyStr
    join_key: NonEmptyStr


class Field(DefinitionModel):
    """One feature and its storage type."""

    name: NonEmptyStr
    dtype: DType


class FileSource(DefinitionModel):
    """A file-backed offline source.

    A single kind in v1. Object storage and warehouse tiers use the same model
    with a different path scheme, so no discriminator is needed yet.
    """

    path: NonEmptyStr


class FeatureView(DefinitionModel):
    """Features derived from one source, keyed by one entity."""

    name: NonEmptyStr
    entities: Annotated[list[Entity], pydantic.Field(min_length=1)]
    source: FileSource
    features: Annotated[list[Field], pydantic.Field(min_length=1)]
    ttl_days: Annotated[int, pydantic.Field(gt=0)] | None = None
    timestamp_field: NonEmptyStr | None = None
    created_timestamp_field: NonEmptyStr | None = None

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
    """A project's definitions.

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
