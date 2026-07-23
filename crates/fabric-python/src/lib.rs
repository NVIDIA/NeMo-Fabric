// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python native bindings for NeMo Fabric.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use nemo_fabric_core::{
    FabricConfig, ResolveContext, RunPlan, RunRequest, RuntimeHandle, doctor_plan,
    resolve_run_plan_from_config_with_adapter_directories, run_plan,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

const ADAPTER_PYTHON_ENV: &str = "ADAPTER_PYTHON";
const ADAPTER_PYTHON_QUERY_TIMEOUT: Duration = Duration::from_secs(5);
const ADAPTER_PYTHON_QUERY_POLL_INTERVAL: Duration = Duration::from_millis(10);
const PYTHON_DATA_PATH_SCRIPT: &str =
    "import json, sysconfig; print(json.dumps(sysconfig.get_path('data')))";

/// Return the Fabric core version.
#[pyfunction]
fn version() -> PyResult<String> {
    Ok(nemo_fabric_core::version().to_string())
}

/// Resolve typed config JSON into a runnable plan and return JSON.
#[pyfunction]
#[pyo3(signature = (config_json, base_dir=None))]
fn plan_config(py: Python<'_>, config_json: String, base_dir: Option<String>) -> PyResult<String> {
    let config = parse_config(config_json)?;
    let (context, adapter_directories) = resolve_context(py, base_dir)?;
    let plan = py
        .detach(|| {
            resolve_run_plan_from_config_with_adapter_directories(
                config,
                context,
                &adapter_directories,
            )
        })
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
    let (context, adapter_directories) = resolve_context(py, base_dir)?;
    let plan = py
        .detach(|| {
            resolve_run_plan_from_config_with_adapter_directories(
                config,
                context,
                &adapter_directories,
            )
        })
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
    let (context, adapter_directories) = resolve_context(py, base_dir)?;
    let plan = py
        .detach(|| {
            resolve_run_plan_from_config_with_adapter_directories(
                config,
                context,
                &adapter_directories,
            )
        })
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

fn resolve_context(
    py: Python<'_>,
    base_dir: Option<String>,
) -> PyResult<(ResolveContext, Vec<PathBuf>)> {
    let base_dir = PathBuf::from(base_dir.unwrap_or_else(|| ".".to_string()));
    let data_path = match std::env::var_os(ADAPTER_PYTHON_ENV) {
        Some(adapter_python) if !adapter_python.is_empty() => {
            let adapter_python = resolve_adapter_python(&base_dir, adapter_python);
            py.detach(|| query_python_data_path(&adapter_python))
                .map_err(PyRuntimeError::new_err)?
        }
        _ => py
            .import("sysconfig")?
            .call_method1("get_path", ("data",))?
            .extract()?,
    };
    // Stopgap: Python adapter wheels install descriptors under the interpreter's
    // data root. ADAPTER_PYTHON is authoritative when set so descriptor metadata
    // matches the adapter code that will execute. A provider-backed adapter
    // registry should replace this implicit environment scan.
    let installed_adapters = PathBuf::from(data_path)
        .join("share")
        .join("nemo-fabric")
        .join("adapters");
    Ok((ResolveContext::new(base_dir), vec![installed_adapters]))
}

fn resolve_adapter_python(base_dir: &Path, adapter_python: OsString) -> PathBuf {
    let adapter_python = PathBuf::from(adapter_python);
    if adapter_python.is_absolute() || adapter_python.components().count() == 1 {
        adapter_python
    } else {
        base_dir.join(adapter_python)
    }
}

fn query_python_data_path(adapter_python: &Path) -> Result<String, String> {
    let mut child = Command::new(adapter_python)
        .arg("-c")
        .arg(PYTHON_DATA_PATH_SCRIPT)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            format!(
                "failed to query {ADAPTER_PYTHON_ENV} `{}` for its data path: {error}",
                adapter_python.display()
            )
        })?;
    let deadline = Instant::now() + ADAPTER_PYTHON_QUERY_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() < deadline => {
                thread::sleep(ADAPTER_PYTHON_QUERY_POLL_INTERVAL);
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!(
                    "{ADAPTER_PYTHON_ENV} `{}` timed out after {} seconds while reporting its data path",
                    adapter_python.display(),
                    ADAPTER_PYTHON_QUERY_TIMEOUT.as_secs()
                ));
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!(
                    "failed to wait for {ADAPTER_PYTHON_ENV} `{}` while querying its data path: {error}",
                    adapter_python.display()
                ));
            }
        }
    }
    let output = child.wait_with_output().map_err(|error| {
        format!(
            "failed to collect {ADAPTER_PYTHON_ENV} `{}` data path output: {error}",
            adapter_python.display()
        )
    })?;
    if !output.status.success() {
        return Err(format!(
            "{ADAPTER_PYTHON_ENV} `{}` could not report its data path: {}",
            adapter_python.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let data_path: String = serde_json::from_slice(&output.stdout).map_err(|error| {
        format!(
            "{ADAPTER_PYTHON_ENV} `{}` returned an invalid data path: {error}",
            adapter_python.display()
        )
    })?;
    if data_path.is_empty() {
        return Err(format!(
            "{ADAPTER_PYTHON_ENV} `{}` returned an empty data path",
            adapter_python.display()
        ));
    }
    Ok(data_path)
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
