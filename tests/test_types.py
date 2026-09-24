"""The dtype markers and their contract with Rust.

Rust deserializes `dtype` as a snake_case string, so the wire names are the
contract. The set is closed on purpose: the value codec writes fixed-width columns
at a fixed stride, so a type it does not know is a decode failure rather than a
new capability.
"""

from feather.types import markers, wire_names


def test_the_wire_names_are_the_rust_variants() -> None:
    """A marker without the Rust variant fails at deserialization, which is a
    runtime failure in a place nothing else tests."""
    assert wire_names() == {
        "int64",
        "float64",
        "boolean",
        "utf8",
        "timestamp_micros",
    }


def test_every_marker_declares_a_known_wire_name() -> None:
    """The markers and the literal have to describe the same set."""
    assert {marker.wire for marker in markers()} == wire_names()


def test_markers_are_exactly_the_declared_types() -> None:
    """A sixth marker is a second source of truth rather than a feature."""
    assert [marker.__name__ for marker in markers()] == [
        "Int64",
        "Float64",
        "Boolean",
        "Utf8",
        "TimestampMicros",
    ]
