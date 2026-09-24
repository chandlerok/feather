//! Python bindings.
//!
//! The Python-facing API lives in `python/feather/`, which is pure Python and
//! re-exports from this compiled module. Keeping the split means the ergonomics
//! (docstrings, overloads, Pydantic models) stay in Python where they belong,
//! and this module stays a thin conversion layer.

use pyo3::prelude::*;

/// The compiled extension. Imported as `feather._core`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
