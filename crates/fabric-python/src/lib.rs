// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python native bindings for NeMo Fabric.

use std::path::PathBuf;

use nemo_fabric_core::{
    FabricConfig, ResolveContext, RunPlan, RunRequest, RuntimeHandle, doctor_plan,
    resolve_run_plan_from_config, run_plan,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Return the Fabric core version.
#[pyfunction]
fn version() -> PyResult<String> {
    Ok(nemo_fabric_core::version().to_string())
}

/// Execute the Rust experimentation CLI using arguments supplied by the console bridge.
#[pyfunction]
fn cli_main(args: Vec<String>) -> PyResult<()> {
    nemo_fabric_cli::app::execute_from(std::iter::once("nemo-fabric".to_string()).chain(args))
        .map_err(PyRuntimeError::new_err)
}

/// Resolve typed config JSON into a runnable plan and return JSON.
#[pyfunction]
#[pyo3(signature = (config_json, base_dir=None))]
fn plan_config(py: Python<'_>, config_json: String, base_dir: Option<String>) -> PyResult<String> {
    let config = parse_config(config_json)?;
    let plan = py
        .detach(|| resolve_run_plan_from_config(config, resolve_context(base_dir)))
        .map_err(to_py_error)?;
    to_json(&plan)
}

/// Diagnose typed config JSON without installing or running it.
#[pyfunction]
#[pyo3(signature = (config_json, base_dir=None))]
fn doctor_config(
    py: Python<'_>,
    config_json: String,
    base_dir: Option<String>,
) -> PyResult<String> {
    let config = parse_config(config_json)?;
    let plan = py
        .detach(|| resolve_run_plan_from_config(config, resolve_context(base_dir)))
        .map_err(to_py_error)?;
    to_json(&doctor_plan(&plan))
}

/// Run typed config JSON through its Fabric adapter and return JSON.
#[pyfunction]
#[pyo3(signature = (config_json, base_dir=None, input_text=None, input_file=None, request_json=None, request_file=None))]
fn run_config(
    py: Python<'_>,
    config_json: String,
    base_dir: Option<String>,
    input_text: Option<String>,
    input_file: Option<String>,
    request_json: Option<String>,
    request_file: Option<String>,
) -> PyResult<String> {
    let config = parse_config(config_json)?;
    let plan = py
        .detach(|| resolve_run_plan_from_config(config, resolve_context(base_dir)))
        .map_err(to_py_error)?;
    let request = match (request_file, request_json, input_file, input_text) {
        (Some(path), None, None, None) => std::fs::read_to_string(PathBuf::from(&path))
            .map_err(|error| PyRuntimeError::new_err(format!("failed to read {path}: {error}")))
            .and_then(parse_run_request)?,
        (None, Some(json), None, None) => parse_run_request(json)?,
        (None, None, Some(path), None) => {
            let input = std::fs::read_to_string(PathBuf::from(&path)).map_err(|error| {
                PyRuntimeError::new_err(format!("failed to read {path}: {error}"))
            })?;
            RunRequest::text(input)
        }
        (None, None, None, Some(text)) => RunRequest::text(text),
        (None, None, None, None) => RunRequest::text(""),
        _ => {
            return Err(PyRuntimeError::new_err(
                "input_text, input_file, request_json, and request_file are mutually exclusive",
            ));
        }
    };
    let result = py
        .detach(|| run_plan(&plan, request))
        .map_err(to_py_error)?;
    to_json(&result)
}

/// Start a runtime for a resolved run plan and return its RuntimeHandle JSON.
#[pyfunction]
fn start_runtime(py: Python<'_>, plan_json: String) -> PyResult<String> {
    let plan = parse_run_plan(plan_json)?;
    let runtime = py
        .detach(|| nemo_fabric_core::start_runtime(&plan))
        .map_err(to_py_error)?;
    to_json(&runtime)
}

/// Invoke a previously started runtime and return RunResult JSON.
#[pyfunction]
fn invoke_runtime(
    py: Python<'_>,
    plan_json: String,
    runtime_json: String,
    request_json: String,
) -> PyResult<String> {
    let plan = parse_run_plan(plan_json)?;
    let runtime = parse_runtime_handle(runtime_json)?;
    let request = parse_run_request(request_json)?;
    let result = py
        .detach(|| nemo_fabric_core::invoke_runtime(&plan, &runtime, request))
        .map_err(to_py_error)?;
    to_json(&result)
}

/// Stop a previously started runtime and return FabricEvent list JSON.
#[pyfunction]
fn stop_runtime(py: Python<'_>, plan_json: String, runtime_json: String) -> PyResult<String> {
    let plan = parse_run_plan(plan_json)?;
    let runtime = parse_runtime_handle(runtime_json)?;
    let events = py
        .detach(|| nemo_fabric_core::stop_runtime(&plan, &runtime))
        .map_err(to_py_error)?;
    to_json(&events)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(cli_main, m)?)?;
    m.add_function(wrap_pyfunction!(plan_config, m)?)?;
    m.add_function(wrap_pyfunction!(doctor_config, m)?)?;
    m.add_function(wrap_pyfunction!(run_config, m)?)?;
    m.add_function(wrap_pyfunction!(start_runtime, m)?)?;
    m.add_function(wrap_pyfunction!(invoke_runtime, m)?)?;
    m.add_function(wrap_pyfunction!(stop_runtime, m)?)?;
    Ok(())
}

fn to_json<T>(value: &T) -> PyResult<String>
where
    T: serde::Serialize,
{
    serde_json::to_string_pretty(value).map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

fn to_py_error(error: nemo_fabric_core::FabricError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn resolve_context(base_dir: Option<String>) -> ResolveContext {
    ResolveContext::new(base_dir.unwrap_or_else(|| ".".to_string()))
}

fn parse_config(contents: String) -> PyResult<FabricConfig> {
    serde_json::from_str(&contents).map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

fn parse_run_request(contents: String) -> PyResult<RunRequest> {
    serde_json::from_str(&contents).map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

fn parse_run_plan(contents: String) -> PyResult<RunPlan> {
    serde_json::from_str(&contents).map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

fn parse_runtime_handle(contents: String) -> PyResult<RuntimeHandle> {
    serde_json::from_str(&contents).map_err(|error| PyRuntimeError::new_err(error.to_string()))
}
