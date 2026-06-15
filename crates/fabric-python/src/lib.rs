//! Python native bindings for NeMo Fabric.

use std::path::PathBuf;

use fabric_core::{RunRequest, doctor_plan, load_fabric_document, resolve_run_plan, run_plan};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Return the Fabric core version.
#[pyfunction]
fn version() -> PyResult<String> {
    Ok(fabric_core::version().to_string())
}

/// Validate an agent directory, manifest, or profile config.
#[pyfunction]
fn validate(path: String) -> PyResult<String> {
    load_fabric_document(&path).map_err(to_py_error)?;
    Ok(format!("validated {path}"))
}

/// Load and return the normalized Fabric document as JSON.
#[pyfunction]
fn inspect(path: String) -> PyResult<String> {
    let document = load_fabric_document(path).map_err(to_py_error)?;
    to_json(&document)
}

/// Resolve an agent/profile into a runnable plan and return JSON.
#[pyfunction]
#[pyo3(signature = (path, profile=None))]
fn plan(path: String, profile: Option<String>) -> PyResult<String> {
    let plan = resolve_run_plan(path, profile.as_deref()).map_err(to_py_error)?;
    to_json(&plan)
}

/// Diagnose a resolved agent/profile without installing or running it.
#[pyfunction]
#[pyo3(signature = (path, profile=None))]
fn doctor(path: String, profile: Option<String>) -> PyResult<String> {
    let plan = resolve_run_plan(path, profile.as_deref()).map_err(to_py_error)?;
    to_json(&doctor_plan(&plan))
}

/// Run an agent/profile through its Fabric adapter and return JSON.
#[pyfunction]
#[pyo3(signature = (path, profile=None, input_text=None, input_file=None, request_json=None, request_file=None))]
fn run(
    path: String,
    profile: Option<String>,
    input_text: Option<String>,
    input_file: Option<String>,
    request_json: Option<String>,
    request_file: Option<String>,
) -> PyResult<String> {
    let plan = resolve_run_plan(path, profile.as_deref()).map_err(to_py_error)?;
    let request = match (request_file, request_json, input_file) {
        (Some(path), None, None) => std::fs::read_to_string(PathBuf::from(&path))
            .map_err(|error| PyRuntimeError::new_err(format!("failed to read {path}: {error}")))
            .and_then(parse_run_request)?,
        (None, Some(json), None) => parse_run_request(json)?,
        (None, None, Some(path)) => {
            let input = std::fs::read_to_string(PathBuf::from(&path)).map_err(|error| {
                PyRuntimeError::new_err(format!("failed to read {path}: {error}"))
            })?;
            RunRequest::text(input)
        }
        (None, None, None) => RunRequest::text(input_text.unwrap_or_default()),
        _ => {
            return Err(PyRuntimeError::new_err(
                "request_file, request_json, and input_file are mutually exclusive",
            ));
        }
    };
    let result = run_plan(&plan, request).map_err(to_py_error)?;
    to_json(&result)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(validate, m)?)?;
    m.add_function(wrap_pyfunction!(inspect, m)?)?;
    m.add_function(wrap_pyfunction!(plan, m)?)?;
    m.add_function(wrap_pyfunction!(doctor, m)?)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}

fn to_json<T>(value: &T) -> PyResult<String>
where
    T: serde::Serialize,
{
    serde_json::to_string_pretty(value).map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

fn to_py_error(error: fabric_core::FabricError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn parse_run_request(contents: String) -> PyResult<RunRequest> {
    serde_json::from_str(&contents).map_err(|error| PyRuntimeError::new_err(error.to_string()))
}
