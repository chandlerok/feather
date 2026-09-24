"""Type stub for the compiled extension.

Hand-written rather than generated. PyO3's `generate-stubs` is self-described as
in development and cannot introspect a function-style `#[pymodule]`, which is the
shape maturin generates. Hand-writing the stub for the small surface we expose is
cheaper than depending on that, and it keeps type checking independent of whether
the extension has been built.

Keep this in sync with `crates/feather-py/src/lib.rs`.
"""

__version__: str

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
