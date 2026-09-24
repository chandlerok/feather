//! Python bindings.
//!
//! The Python-facing API lives in `python/feather/`, which is pure Python and
//! re-exports from this compiled module. Keeping the split means the ergonomics
//! (docstrings, overloads, Pydantic models) stay in Python where they belong,
//! and this module stays a thin conversion layer over `feather-core`.
//!
//! This layer converts; it does not validate. Everything that crosses is validated
//! by the core, so a binding that skips its own checks cannot hand over something
//! the engines cannot represent.

use feather_core::Error as CoreError;
use pyo3::exceptions::{PyFileNotFoundError, PyOSError, PyValueError};
use pyo3::prelude::*;

/// Read and validate a `feather.toml`, returned as JSON.
///
/// The core owns the file format, the schema, and `${VAR}` resolution, so this
/// binding contributes only the path.
#[pyfunction]
fn load_settings(path: &str) -> PyResult<String> {
    let settings = feather_core::load_settings(path).map_err(settings_error)?;
    serde_json::to_string(&settings).map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Map a settings failure onto the exception a Python caller expects.
fn settings_error(error: CoreError) -> PyErr {
    // Render before matching, so the variant can be destructured by reference.
    let message = error.to_string();
    match &error {
        CoreError::SettingsIo { source, .. } if source.kind() == std::io::ErrorKind::NotFound => {
            PyFileNotFoundError::new_err(message)
        }
        CoreError::SettingsIo { .. } => PyOSError::new_err(message),
        _ => PyValueError::new_err(message),
    }
}

/// The compiled extension. Imported as `feather._core`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(load_settings, m)?)?;
    Ok(())
}
