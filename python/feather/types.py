"""Feature value types.

The names are the wire format. Rust deserializes them, so changing a value here
is a breaking change to the Python-to-Rust contract.
"""

from enum import StrEnum


class DType(StrEnum):
    """The storage type of a feature.

    A closed set. Every variant is either fixed-width, which the value encoding
    writes with no per-value prefix, or UTF-8, which is the only variable-width
    case supported.
    """

    INT64 = "int64"
    FLOAT64 = "float64"
    BOOLEAN = "boolean"
    UTF8 = "utf8"
    TIMESTAMP_MICROS = "timestamp_micros"


Int64 = DType.INT64
Float64 = DType.FLOAT64
Boolean = DType.BOOLEAN
Utf8 = DType.UTF8
TimestampMicros = DType.TIMESTAMP_MICROS

__all__ = [
    "Boolean",
    "DType",
    "Float64",
    "Int64",
    "TimestampMicros",
    "Utf8",
]
