"""Project settings, read from ``feather.toml``.

The core owns this file: it parses the TOML, resolves ``${VAR}`` from the
environment, and decides whether the values are valid. This module is the typed
view of the result, so Python code gets autocomplete and a checked attribute set
without a second implementation of the schema.

The models below therefore mirror the core's, and their constraints are a
deliberate duplicate rather than the rule. A mismatch between the two is caught
here, at the binding, instead of inside an engine.

Offline sources are not here. A source is declared on the view it feeds, as
``FileSource(path=...)`` or ``PostgresSource(connection=...)`` in a definition module.
What this file holds is the credentials a source must not carry, since a definition
module is committed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import pydantic

from feather._wire import NonEmptyStr

if TYPE_CHECKING:
    import os

DEFAULT_PATH = Path("feather.toml")
"""The file :func:`load_settings` reads when no path is given."""


class SnowflakeConnection(pydantic.BaseModel):
    """A Snowflake account a source can read from."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    type: Literal["snowflake"]
    account: NonEmptyStr
    warehouse: NonEmptyStr
    username: NonEmptyStr
    password: pydantic.SecretStr


class S3Connection(pydantic.BaseModel):
    """An S3 or S3-compatible object store a source can read from.

    ``key_id`` is not a secret and is left readable, so a misconfigured key is
    diagnosable from a log. ``secret`` is not.

    ``endpoint`` is for a server that is not AWS. DuckDB addresses S3 virtually by
    default and over HTTPS, which a host and port endpoint does not answer, so an
    endpoint implies path style unless ``url_style`` says otherwise.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    type: Literal["s3"]
    region: NonEmptyStr
    key_id: NonEmptyStr
    secret: pydantic.SecretStr
    endpoint: NonEmptyStr | None = None
    use_ssl: bool = True
    url_style: Literal["path", "vhost"] | None = None


class PostgresConnection(pydantic.BaseModel):
    """A Postgres database a source can read from.

    ``host`` and ``port`` are libpq's rather than a URL's, which is why they are two
    fields and not one. ``port`` is bounded to what the core can carry, which is a
    ``u16``, so 0 is allowed: the bound is for catching a value the core cannot
    represent, not for being a second and stricter contract. Whether libpq then reads 0
    as "use the default port" is unverified here, and is not what the bound rests on.
    ``password`` is a secret and stays out of ``repr`` the way the other credentials do.

    ``ssl_mode`` is any of libpq's ``sslmode`` values, lower case, and has no default.
    libpq already prefers TLS when the server offers it, and a security flag guessed
    wrong is worse than one an operator had to write down. The values are lower case
    because libpq compares them that way: `sslmode=REQUIRE` is refused with "invalid
    sslmode value", so accepting upper case here would only move the failure later.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    type: Literal["postgres"]
    host: NonEmptyStr
    port: Annotated[int, pydantic.Field(ge=0, le=65535)] = 5432
    database: NonEmptyStr
    user: NonEmptyStr
    password: pydantic.SecretStr
    ssl_mode: (
        Literal["disable", "allow", "prefer", "require", "verify-ca", "verify-full"] | None
    ) = None


Connection = Annotated[
    SnowflakeConnection | S3Connection | PostgresConnection,
    pydantic.Field(discriminator="type"),
]
"""A named credential set, discriminated on ``type``."""


class Store(pydantic.BaseModel):
    """The embedded online store, which a serving process owns.

    An LSM rather than an in-memory map, so a dataset larger than memory is a disk read
    rather than an eviction, and a restart is a remap rather than a full
    re-materialization. A database is opened by one process at a time, which is why the
    serving process is the only thing that opens it.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    path: NonEmptyStr
    cache_bytes: pydantic.PositiveInt | None = None
    memtable_bytes: pydantic.PositiveInt | None = None


class FeatherSettings(pydantic.BaseModel):
    """A validated ``feather.toml``.

    An absent ``store`` is local mode: an in-process DuckDB over local files, served from
    an in-process online store.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    project: NonEmptyStr
    definitions: Annotated[list[NonEmptyStr], pydantic.Field(min_length=1)]
    connections: dict[NonEmptyStr, Connection] = pydantic.Field(default_factory=dict)
    store: Store | None = None


def load_settings(path: str | os.PathLike[str] = DEFAULT_PATH) -> FeatherSettings:
    """Read and validate a ``feather.toml``.

    Args:
        path: The file to read.

    Returns:
        The validated settings, as the core resolved them.

    Raises:
        FileNotFoundError: If no file is at ``path``.
        OSError: If the file exists but cannot be read.
        ValueError: If it is not valid TOML, holds an unknown key, is missing a
            required one, or has a ``${VAR}`` reference that cannot be resolved.
            The message carries the dotted key path.
    """
    # Imported here rather than at module scope, so that importing `feather` still
    # works without the compiled extension. Only this call needs it.
    from feather import _core

    return FeatherSettings.model_validate_json(_core.load_settings(str(path)))


__all__ = [
    "DEFAULT_PATH",
    "Connection",
    "FeatherSettings",
    "PostgresConnection",
    "S3Connection",
    "SnowflakeConnection",
    "load_settings",
]
