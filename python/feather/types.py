"""Feature value types.

A dtype is a class rather than an enum member because it is used as a type
argument: ``Field[Int64]``. A value cannot be a type argument, and the enum-alias
form fails in the worst way: the type checker reports "Variable Int64 is not valid
as a type" while runtime accepts it silently, so the annotation looks checked and
checks nothing.

``WireDType`` is the authoritative list of wire names, and it is the
Python-to-Rust contract. Every marker declares its ``wire`` against that
``Literal``, so the type checker rejects a marker naming a string Rust does not
know.
"""

from typing import ClassVar, Literal, get_args

WireDType = Literal[
    "int64",
    "float64",
    "boolean",
    "utf8",
    "timestamp_micros",
]


class DType:
    """Base for a storage type. Markers are used as types, never instantiated."""

    wire: ClassVar[WireDType]


class Int64(DType):
    """A signed 64-bit integer."""

    wire = "int64"


class Float64(DType):
    """An IEEE 754 double."""

    wire = "float64"


class Boolean(DType):
    """A boolean."""

    wire = "boolean"


class Utf8(DType):
    """A UTF-8 string, the only variable-width type the value codec supports."""

    wire = "utf8"


class TimestampMicros(DType):
    """Microseconds since the Unix epoch, in UTC."""

    wire = "timestamp_micros"


def markers() -> tuple[type[DType], ...]:
    """Return every declared marker.

    Returns:
        The markers, in declaration order.
    """
    return tuple(DType.__subclasses__())


def wire_names() -> frozenset[str]:
    """Return the wire names as a plain set.

    Returns:
        Every name in ``WireDType``.
    """
    return frozenset(get_args(WireDType))


__all__ = [
    "Boolean",
    "DType",
    "Float64",
    "Int64",
    "TimestampMicros",
    "Utf8",
    "WireDType",
    "markers",
    "wire_names",
]
