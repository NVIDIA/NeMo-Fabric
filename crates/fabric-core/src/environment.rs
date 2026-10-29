// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-side execution-environment lifecycle and provider process boundary.
//!
//! Prepares, attaches, and releases local or OpenShell environments. For
//! OpenShell, it starts the optional provider child over JSON/stdio, forwards
//! environment operations, and returns verified environment handles. Agent
//! lifecycle and task SDK execution live outside this module.

use std::collections::{BTreeMap, HashMap};
use std::ffi::OsString;
#[cfg(unix)]
use std::io::Write;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
#[cfg(unix)]
use std::process::ChildStdin;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, LazyLock, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::{ControlLocation, EnvironmentOwnership, EnvironmentPlan, RunPlan};
use crate::error::{FabricError, Result};
use crate::runtime::{
    EnvironmentHandle, EnvironmentReference, absolute_path, new_id, resolve_path,
};
use crate::task_command::{TaskCommand, TaskCommandResponse};

const OPEN_SHELL_PROVIDER_ID: &str = "openshell";
const OPEN_SHELL_PROVIDER_COMMAND: &str = "fabric-environment-openshell";
const OPEN_SHELL_PROVIDER_COMMAND_ENV: &str = "NEMO_FABRIC_OPEN_SHELL_PROVIDER";
const PROVIDER_PROTOCOL_VERSION: &str = "fabric.environment-provider.v1alpha1";
const MAX_PROVIDER_REQUEST_BYTES: usize = 1024 * 1024;
// Artifact collection can return 2 MiB of bytes as decimal JSON arrays, which
// expand to at most about 8 MiB before the provider protocol envelope.
const MAX_PROVIDER_RESPONSE_BYTES: usize = 10 * 1024 * 1024;
const DEFAULT_PROVIDER_OPERATION_TIMEOUT_SECONDS: u64 = 30;
const PROVIDER_RESPONSE_GRACE_SECONDS: u64 = 15;
const PROVIDER_GATEWAY_CALL_TIMEOUT_SECONDS: u64 = 30;

/// Internal environment preparation contract.
///
/// This boundary is intentionally crate-private while the first non-local
/// provider proves the lifecycle. It does not yet define a public provider ABI.
pub(crate) trait EnvironmentProvider: Sync {
    /// Resolve or prepare the environment represented by a run plan.
    fn prepare(&self, plan: &RunPlan) -> Result<EnvironmentHandle>;

    /// Verify and attach to a caller-owned provider resource.
    fn attach(&self, plan: &RunPlan, reference: &EnvironmentReference)
    -> Result<EnvironmentHandle>;
}

struct LocalEnvironmentProvider;
struct OpenShellEnvironmentProvider;

static LOCAL_ENVIRONMENT_PROVIDER: LocalEnvironmentProvider = LocalEnvironmentProvider;
static OPEN_SHELL_ENVIRONMENT_PROVIDER: OpenShellEnvironmentProvider = OpenShellEnvironmentProvider;
static OPEN_SHELL_PROVIDER_PROCESS: LazyLock<Mutex<Option<ProviderProcess>>> =
    LazyLock::new(|| Mutex::new(None));
static OPEN_SHELL_RUNTIME_PROCESSES: LazyLock<Mutex<HashMap<String, Arc<Mutex<ProviderProcess>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Resolve a built-in environment provider by its stable configuration id.
pub(crate) fn resolve_environment_provider(
    provider_id: &str,
) -> Option<&'static dyn EnvironmentProvider> {
    match provider_id {
        "local" => Some(&LOCAL_ENVIRONMENT_PROVIDER),
        OPEN_SHELL_PROVIDER_ID if cfg!(unix) => Some(&OPEN_SHELL_ENVIRONMENT_PROVIDER),
        _ => None,
    }
}

impl EnvironmentProvider for LocalEnvironmentProvider {
    fn prepare(&self, plan: &RunPlan) -> Result<EnvironmentHandle> {
        let mut metadata = BTreeMap::new();
        let mut connection = BTreeMap::new();
        let (
            control_location,
            ownership,
            workspace,
            artifacts,
            environment_env,
            connection_settings,
            environment_metadata,
            settings,
        ) = if let Some(environment) = &plan.environment_plan {
            (
                environment.control_location,
                environment.ownership,
                environment.workspace.clone(),
                environment.artifacts.clone(),
                environment.env.clone(),
                environment.connection.clone(),
                environment.metadata.clone(),
                environment.settings.clone(),
            )
        } else {
            (
                ControlLocation::ExternalControl,
                EnvironmentOwnership::CallerOwned,
                Some(plan.base_dir.clone()),
                plan.config
                    .runtime
                    .artifacts
                    .as_ref()
                    .map(|artifacts| resolve_path(&plan.base_dir, artifacts)),
                BTreeMap::new(),
                serde_json::Map::new(),
                serde_json::Map::new(),
                serde_json::Map::new(),
            )
        };
        let workspace = match workspace {
            Some(path) => Some(absolute_path(path)?),
            None => None,
        };
        connection.extend(connection_settings);
        metadata.extend(settings);
        metadata.extend(environment_metadata);
        Ok(EnvironmentHandle {
            environment_id: new_id("environment"),
            provider: "local".to_string(),
            control_location,
            workspace,
            artifacts,
            env: environment_env,
            ownership,
            connection,
            metadata,
        })
    }

    fn attach(
        &self,
        _plan: &RunPlan,
        _reference: &EnvironmentReference,
    ) -> Result<EnvironmentHandle> {
        Err(FabricError::InvalidConfig {
            field: "environment_reference.provider".to_string(),
            reason: "the local provider does not attach external resources".to_string(),
        })
    }
}

impl EnvironmentProvider for OpenShellEnvironmentProvider {
    fn prepare(&self, plan: &RunPlan) -> Result<EnvironmentHandle> {
        let environment =
            plan.environment_plan
                .as_ref()
                .ok_or_else(|| FabricError::InvalidConfig {
                    field: "environment".to_string(),
                    reason: "the openshell provider requires an explicit environment configuration"
                        .to_string(),
                })?;
        validate_open_shell_profile(environment, EnvironmentOwnership::FabricOwned)?;

        let environment_id = new_id("environment");
        let mut dispatched = environment.clone();
        dispatched.metadata.insert(
            "fabric.provenance.enabled".into(),
            serde_json::json!(plan.config.provenance.enabled),
        );
        let prepared: PreparedEnvironment = self.request(ProviderOperation::Prepare {
            environment_id: &environment_id,
            environment: &dispatched,
        })?;
        let workspace =
            prepared
                .workspace
                .ok_or_else(|| FabricError::EnvironmentProviderOperation {
                    provider: OPEN_SHELL_PROVIDER_ID.to_string(),
                    operation: "prepare".to_string(),
                    code: "workspace_unavailable".to_string(),
                    message: "OpenShell provider response omitted the sandbox workspace"
                        .to_string(),
                })?;
        let mut metadata = prepared.metadata.into_iter().collect::<BTreeMap<_, _>>();
        metadata.extend(environment.metadata.clone());

        Ok(EnvironmentHandle {
            environment_id,
            provider: OPEN_SHELL_PROVIDER_ID.to_string(),
            control_location: environment.control_location,
            workspace: Some(workspace),
            artifacts: prepared.artifacts,
            env: environment.env.clone(),
            ownership: environment.ownership,
            connection: prepared.connection.into_iter().collect(),
            metadata,
        })
    }

    fn attach(
        &self,
        plan: &RunPlan,
        reference: &EnvironmentReference,
    ) -> Result<EnvironmentHandle> {
        let environment =
            plan.environment_plan
                .as_ref()
                .ok_or_else(|| FabricError::InvalidConfig {
                    field: "environment".to_string(),
                    reason: "the openshell provider requires an explicit environment configuration"
                        .to_string(),
                })?;
        validate_open_shell_profile(environment, EnvironmentOwnership::CallerOwned)?;

        let environment_id = new_id("environment");
        let mut dispatched = environment.clone();
        dispatched.metadata.insert(
            "fabric.provenance.enabled".into(),
            serde_json::json!(plan.config.provenance.enabled),
        );
        let prepared: PreparedEnvironment = self.request(ProviderOperation::Attach {
            environment_id: &environment_id,
            environment: &dispatched,
            reference,
        })?;
        let workspace =
            prepared
                .workspace
                .ok_or_else(|| FabricError::EnvironmentProviderOperation {
                    provider: OPEN_SHELL_PROVIDER_ID.to_string(),
                    operation: "attach".to_string(),
                    code: "workspace_unavailable".to_string(),
                    message: "OpenShell provider response omitted the sandbox workspace"
                        .to_string(),
                })?;
        let mut metadata = prepared.metadata.into_iter().collect::<BTreeMap<_, _>>();
        metadata.extend(environment.metadata.clone());

        Ok(EnvironmentHandle {
            environment_id,
            provider: OPEN_SHELL_PROVIDER_ID.to_string(),
            control_location: environment.control_location,
            workspace: Some(workspace),
            artifacts: prepared.artifacts,
            env: environment.env.clone(),
            ownership: environment.ownership,
            connection: prepared.connection.into_iter().collect(),
            metadata,
        })
    }
}

impl OpenShellEnvironmentProvider {
    fn request<T>(&self, operation: ProviderOperation<'_>) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        request_provider_process(provider_command(), operation)
    }
}

/// Release or detach an environment through its registered provider.
pub(crate) fn release_environment(
    environment: &EnvironmentHandle,
) -> Result<ProviderReleaseOutput> {
    match environment.provider.as_str() {
        "local" => Ok(ProviderReleaseOutput {
            released: false,
            detached: true,
        }),
        OPEN_SHELL_PROVIDER_ID if cfg!(unix) => {
            OPEN_SHELL_ENVIRONMENT_PROVIDER.request(ProviderOperation::Release { environment })
        }
        provider => Err(FabricError::UnsupportedEnvironmentRelease {
            provider: provider.to_string(),
        }),
    }
}

/// Execute a buffered SDK command through the selected task provider.
pub(crate) fn command_runtime(
    environment: &EnvironmentHandle,
    command: &TaskCommand,
    timeout_seconds: u64,
) -> Result<TaskCommandResponse> {
    if environment.provider != OPEN_SHELL_PROVIDER_ID {
        return Err(FabricError::UnsupportedEnvironmentProvider {
            provider: environment.provider.clone(),
            adapter_kind: crate::config::AdapterKind::Process,
        });
    }
    OPEN_SHELL_ENVIRONMENT_PROVIDER.request(ProviderOperation::RuntimeCommand {
        environment,
        command,
        timeout_seconds,
    })
}

/// Read only declared files beneath the task artifact root.
pub(crate) fn collect_artifacts(
    environment: &EnvironmentHandle,
    paths: &[String],
) -> Result<Vec<CollectedArtifact>> {
    OPEN_SHELL_ENVIRONMENT_PROVIDER
        .request(ProviderOperation::CollectArtifacts { environment, paths })
}

fn validate_open_shell_profile(
    environment: &EnvironmentPlan,
    expected_ownership: EnvironmentOwnership,
) -> Result<()> {
    if environment.control_location != ControlLocation::InEnvControl {
        return Err(FabricError::InvalidConfig {
            field: "environment.control_location".to_string(),
            reason: "the experimental openshell provider supports only `in_env_control`"
                .to_string(),
        });
    }
    if environment.ownership != expected_ownership {
        return Err(FabricError::InvalidConfig {
            field: "environment.ownership".to_string(),
            reason: format!(
                "this operation requires `{}`",
                match expected_ownership {
                    EnvironmentOwnership::CallerOwned => "caller_owned",
                    EnvironmentOwnership::FabricOwned => "fabric_owned",
                }
            ),
        });
    }
    if environment.workspace.is_none() {
        return Err(FabricError::InvalidConfig {
            field: "environment.workspace".to_string(),
            reason: "the experimental openshell provider requires an absolute sandbox workspace"
                .to_string(),
        });
    }
    Ok(())
}

fn provider_command() -> OsString {
    std::env::var_os(OPEN_SHELL_PROVIDER_COMMAND_ENV)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| OsString::from(OPEN_SHELL_PROVIDER_COMMAND))
}

fn request_provider_process<T>(command: OsString, operation: ProviderOperation<'_>) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    if matches!(&operation, ProviderOperation::RuntimeCommand { .. }) {
        return request_runtime_provider_process(command, operation);
    }
    let operation_name = operation.name();
    let response_timeout = operation.response_timeout();
    let request_id = new_id("environment-provider-request");
    let request = ProviderRequest {
        protocol_version: PROVIDER_PROTOCOL_VERSION,
        request_id: &request_id,
        operation,
    };
    let mut encoded = serde_json::to_vec(&request).map_err(FabricError::SerializeJson)?;
    if encoded.len() > MAX_PROVIDER_REQUEST_BYTES {
        return provider_error(
            operation_name,
            "request_too_large",
            format!(
                "request is {} bytes; the limit is {MAX_PROVIDER_REQUEST_BYTES}",
                encoded.len()
            ),
        );
    }
    encoded.push(b'\n');

    let mut slot = OPEN_SHELL_PROVIDER_PROCESS
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let replace = match slot.as_mut() {
        Some(process) if process.command != command => true,
        Some(process) => process.has_exited(operation_name)?,
        None => false,
    };
    if replace && let Some(mut process) = slot.take() {
        process.shutdown();
    }
    if slot.is_none() {
        *slot = Some(ProviderProcess::spawn(command, operation_name)?);
    }
    let process = slot.as_mut().expect("provider process initialized");
    if process
        .write_request(&encoded, Duration::from_secs(10))
        .is_err()
    {
        process.shutdown();
        *slot = None;
        return provider_error(
            operation_name,
            "provider_protocol_error",
            "could not write the bounded provider request".to_string(),
        );
    }
    let output = match process.responses.recv_timeout(response_timeout) {
        Ok(Ok(output)) => output,
        Ok(Err(_)) => {
            process.shutdown();
            *slot = None;
            return provider_error(
                operation_name,
                "provider_protocol_error",
                "could not read provider response".to_string(),
            );
        }
        Err(RecvTimeoutError::Timeout) => {
            process.shutdown();
            *slot = None;
            return provider_error(
                operation_name,
                "provider_protocol_error",
                "provider response deadline expired".to_string(),
            );
        }
        Err(RecvTimeoutError::Disconnected) => {
            process.shutdown();
            *slot = None;
            return provider_error(
                operation_name,
                "provider_exited",
                "provider closed its response stream".to_string(),
            );
        }
    };
    if output.len() > MAX_PROVIDER_RESPONSE_BYTES {
        process.shutdown();
        *slot = None;
        return provider_error(
            operation_name,
            "response_too_large",
            format!(
                "response is {} bytes; the limit is {MAX_PROVIDER_RESPONSE_BYTES}",
                output.len()
            ),
        );
    }
    if output.is_empty() || output.last() != Some(&b'\n') {
        process.shutdown();
        *slot = None;
        return provider_error(
            operation_name,
            "provider_exited",
            "provider response was incomplete".to_string(),
        );
    }

    let response: ProviderResponse = match serde_json::from_slice(&output) {
        Ok(response) => response,
        Err(_) => {
            process.shutdown();
            *slot = None;
            return provider_error(
                operation_name,
                "provider_protocol_error",
                "provider returned invalid JSON".to_string(),
            );
        }
    };
    if response.protocol_version != PROVIDER_PROTOCOL_VERSION {
        process.shutdown();
        *slot = None;
        return provider_error(
            operation_name,
            "provider_protocol_mismatch",
            "provider protocol version did not match".to_string(),
        );
    }
    if response.request_id != request_id {
        process.shutdown();
        *slot = None;
        return provider_error(
            operation_name,
            "provider_correlation_mismatch",
            "provider response request id did not match the request".to_string(),
        );
    }
    match response.outcome {
        ProviderOutcome::Succeeded { output } => {
            serde_json::from_value(output).map_err(|_| FabricError::EnvironmentProviderOperation {
                provider: OPEN_SHELL_PROVIDER_ID.to_string(),
                operation: operation_name.to_string(),
                code: "provider_protocol_error".to_string(),
                message: "provider output did not match the operation contract".to_string(),
            })
        }
        ProviderOutcome::Failed { error } => {
            provider_error(operation_name, &error.code, error.message)
        }
    }
}

// A dedicated provider process handles each health check and exits after the response.
fn request_check_provider_process<T>(
    command: OsString,
    operation: ProviderOperation<'_>,
    timeout_seconds: f64,
) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let started = Instant::now();
    let budget = Duration::try_from_secs_f64(timeout_seconds).unwrap_or(Duration::MAX);
    let remaining = || {
        let remaining = budget.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            Err(provider_failure(
                "runtime_command",
                "probe_timeout",
                "health deadline expired".into(),
            ))
        } else {
            Ok(remaining)
        }
    };
    let request_id = new_id("health-provider-request");
    let mut process =
        ProviderProcess::spawn_with_deadline(command, "runtime_command", Some((started, budget)))?;
    let result = (|| {
        let mut request = serde_json::to_value(ProviderRequest {
            protocol_version: PROVIDER_PROTOCOL_VERSION,
            request_id: &request_id,
            operation,
        })
        .map_err(FabricError::SerializeJson)?;
        let seconds = remaining()?.as_secs_f64();
        request["command"]["timeout_seconds"] = Value::from(seconds);
        request["timeout_seconds"] = Value::from(seconds.ceil().clamp(1.0, 320.0) as u64);
        let mut encoded = serde_json::to_vec(&request).map_err(FabricError::SerializeJson)?;
        if encoded.len() > MAX_PROVIDER_REQUEST_BYTES {
            return provider_error(
                "runtime_command",
                "request_too_large",
                "health request exceeds its limit".into(),
            );
        }
        encoded.push(b'\n');
        process.write_request(&encoded, remaining()?).map_err(|_| {
            provider_failure(
                "runtime_command",
                "provider_protocol_error",
                "health request could not be delivered".into(),
            )
        })?;
        let output = match process.responses.recv_timeout(remaining()?) {
            Ok(Ok(output)) => output,
            Err(RecvTimeoutError::Timeout) => {
                return provider_error(
                    "runtime_command",
                    "probe_timeout",
                    "health deadline expired".into(),
                );
            }
            _ => {
                return provider_error(
                    "runtime_command",
                    "provider_protocol_error",
                    "health response is unavailable".into(),
                );
            }
        };
        remaining()?;
        if output.is_empty() || output.len() > 16 * 1024 || output.last() != Some(&b'\n') {
            return provider_error(
                "runtime_command",
                "provider_protocol_error",
                "invalid health response framing".into(),
            );
        }
        let response: ProviderResponse = serde_json::from_slice(&output).map_err(|_| {
            provider_failure(
                "runtime_command",
                "provider_protocol_error",
                "invalid health response".into(),
            )
        })?;
        if response.protocol_version != PROVIDER_PROTOCOL_VERSION
            || response.request_id != request_id
        {
            return provider_error(
                "runtime_command",
                "provider_correlation_mismatch",
                "health response correlation failed".into(),
            );
        }
        match response.outcome {
            ProviderOutcome::Succeeded { output } => serde_json::from_value(output).map_err(|_| {
                provider_failure(
                    "runtime_command",
                    "provider_protocol_error",
                    "invalid health command result".into(),
                )
            }),
            ProviderOutcome::Failed { error } => provider_error(
                "runtime_command",
                if matches!(error.code.as_str(), "probe_timeout" | "gateway_timeout") {
                    "probe_timeout"
                } else {
                    "probe_error"
                },
                "health command failed".into(),
            ),
        }
    })();
    process.shutdown_check(started, budget);
    result
}

fn request_runtime_provider_process<T>(
    command: OsString,
    operation: ProviderOperation<'_>,
) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let ProviderOperation::RuntimeCommand {
        environment,
        command: task_command,
        ..
    } = &operation
    else {
        unreachable!("runtime provider path requires a runtime command")
    };
    if let TaskCommand::Check {
        timeout_seconds, ..
    } = task_command
    {
        let budget = *timeout_seconds;
        return request_check_provider_process(command, operation, budget);
    }
    let operation_name = operation.name();
    let request_id = new_id("environment-provider-request");
    let environment_key = environment_resource_key(environment)?;
    let (identity, is_start, is_stop) = match task_command {
        TaskCommand::Check { .. } => unreachable!("health uses a separate provider process"),
        TaskCommand::Start { .. } => (request_id.clone(), true, false),
        TaskCommand::Invoke { runtime_id, .. } => (runtime_id.clone(), false, false),
        TaskCommand::Stop { runtime_id, .. } => (runtime_id.clone(), false, true),
    };
    let process_key = serde_json::json!([environment_key, identity]).to_string();
    let response_timeout = operation.response_timeout();
    let request = ProviderRequest {
        protocol_version: PROVIDER_PROTOCOL_VERSION,
        request_id: &request_id,
        operation,
    };
    let mut encoded = serde_json::to_vec(&request).map_err(FabricError::SerializeJson)?;
    if encoded.len() > MAX_PROVIDER_REQUEST_BYTES {
        return provider_error(
            operation_name,
            "request_too_large",
            format!(
                "request is {} bytes; the limit is {MAX_PROVIDER_REQUEST_BYTES}",
                encoded.len()
            ),
        );
    }
    encoded.push(b'\n');

    let process = runtime_provider_process(&process_key, &command, operation_name)?;
    let output = {
        let mut process_guard = process.lock().unwrap_or_else(|error| error.into_inner());
        let output = (|| -> Result<Vec<u8>> {
            if process_guard.has_exited(operation_name)? {
                return Err(provider_failure(
                    operation_name,
                    "provider_exited",
                    "runtime provider process exited before the command".to_string(),
                ));
            }
            process_guard
                .write_request(&encoded, Duration::from_secs(10))
                .map_err(|_| {
                    provider_failure(
                        operation_name,
                        "provider_protocol_error",
                        "could not write the bounded provider request".to_string(),
                    )
                })?;
            match process_guard.responses.recv_timeout(response_timeout) {
                Ok(Ok(output)) => Ok(output),
                Ok(Err(_)) => Err(provider_failure(
                    operation_name,
                    "provider_protocol_error",
                    "could not read provider response".to_string(),
                )),
                Err(RecvTimeoutError::Timeout) => Err(provider_failure(
                    operation_name,
                    "provider_protocol_error",
                    "provider response deadline expired".to_string(),
                )),
                Err(RecvTimeoutError::Disconnected) => Err(provider_failure(
                    operation_name,
                    "provider_exited",
                    "provider closed its response stream".to_string(),
                )),
            }
        })();
        if output.is_err() {
            process_guard.shutdown();
        }
        output
    };
    let output = match output {
        Ok(output) => output,
        Err(error) => {
            remove_runtime_provider_process(&process_key, &process);
            return Err(error);
        }
    };
    if output.len() > MAX_PROVIDER_RESPONSE_BYTES
        || output.is_empty()
        || output.last() != Some(&b'\n')
    {
        retire_runtime_provider_process(&process_key, &process);
        return provider_error(
            operation_name,
            "response_too_large",
            "provider response was incomplete or exceeded its size limit".to_string(),
        );
    }
    let response: ProviderResponse = match serde_json::from_slice(&output) {
        Ok(response) => response,
        Err(_) => {
            retire_runtime_provider_process(&process_key, &process);
            return provider_error(
                operation_name,
                "provider_protocol_error",
                "provider returned invalid JSON".to_string(),
            );
        }
    };
    if response.protocol_version != PROVIDER_PROTOCOL_VERSION || response.request_id != request_id {
        retire_runtime_provider_process(&process_key, &process);
        return provider_error(
            operation_name,
            "provider_correlation_mismatch",
            "provider response did not match the request".to_string(),
        );
    }
    let output = match response.outcome {
        ProviderOutcome::Succeeded { output } => output,
        ProviderOutcome::Failed { error } => {
            return provider_error(operation_name, &error.code, error.message);
        }
    };
    let runtime_id = if is_start && output["status"] == "returned" {
        output["value"]["runtime_id"].as_str().map(str::to_owned)
    } else {
        None
    };
    let stop_succeeded = is_stop && output["status"] == "returned" && output["value"].is_null();
    let result = serde_json::from_value(output.clone()).map_err(|_| {
        FabricError::EnvironmentProviderOperation {
            provider: OPEN_SHELL_PROVIDER_ID.to_string(),
            operation: operation_name.to_string(),
            code: "provider_protocol_error".to_string(),
            message: "provider output did not match the operation contract".to_string(),
        }
    })?;
    if let Some(runtime_id) = runtime_id {
        let next_key = serde_json::json!([environment_key, runtime_id]).to_string();
        let mut processes = OPEN_SHELL_RUNTIME_PROCESSES
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if next_key != process_key {
            if processes.contains_key(&next_key) {
                drop(processes);
                retire_runtime_provider_process(&process_key, &process);
                return provider_error(
                    operation_name,
                    "runtime_identity_conflict",
                    "task SDK returned a runtime ID that is already active".to_string(),
                );
            }
            if processes
                .get(&process_key)
                .is_some_and(|current| Arc::ptr_eq(current, &process))
            {
                processes.remove(&process_key);
                processes.insert(next_key, Arc::clone(&process));
            }
        }
    } else if stop_succeeded {
        retire_runtime_provider_process(&process_key, &process);
    }
    Ok(result)
}

fn runtime_provider_process(
    process_key: &str,
    command: &OsString,
    operation: &str,
) -> Result<Arc<Mutex<ProviderProcess>>> {
    loop {
        let existing = OPEN_SHELL_RUNTIME_PROCESSES
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(process_key)
            .cloned();
        if let Some(process) = existing {
            let same_command = process
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .command
                == *command;
            if same_command {
                return Ok(process);
            }
            retire_runtime_provider_process(process_key, &process);
            continue;
        }

        let candidate = Arc::new(Mutex::new(ProviderProcess::spawn(
            command.clone(),
            operation,
        )?));
        let selected = {
            let mut processes = OPEN_SHELL_RUNTIME_PROCESSES
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            Arc::clone(
                processes
                    .entry(process_key.to_string())
                    .or_insert_with(|| Arc::clone(&candidate)),
            )
        };
        if Arc::ptr_eq(&candidate, &selected) {
            return Ok(candidate);
        }
        candidate
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .shutdown();
    }
}

fn remove_runtime_provider_process(process_key: &str, process: &Arc<Mutex<ProviderProcess>>) {
    let mut processes = OPEN_SHELL_RUNTIME_PROCESSES
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if processes
        .get(process_key)
        .is_some_and(|current| Arc::ptr_eq(current, process))
    {
        processes.remove(process_key);
    }
}

fn retire_runtime_provider_process(process_key: &str, process: &Arc<Mutex<ProviderProcess>>) {
    remove_runtime_provider_process(process_key, process);
    process
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .shutdown();
}

fn environment_resource_key(environment: &EnvironmentHandle) -> Result<String> {
    let text = |name: &str| {
        environment
            .connection
            .get(name)
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
    };
    let gateway = text("gateway").ok_or_else(|| FabricError::InvalidConfig {
        field: "environment.connection.gateway".to_string(),
        reason: "a verified OpenShell binding is required".to_string(),
    })?;
    let sandbox_id = text("sandbox_id").ok_or_else(|| FabricError::InvalidConfig {
        field: "environment.connection.sandbox_id".to_string(),
        reason: "a verified OpenShell binding is required".to_string(),
    })?;
    Ok(serde_json::json!([
        gateway.trim_end_matches('/'),
        text("workspace").unwrap_or("default"),
        sandbox_id
    ])
    .to_string())
}

struct ProviderPipe<R> {
    inner: R,
    deadline: Option<(Instant, Duration)>,
    stopping: Arc<std::sync::atomic::AtomicBool>,
}

impl<R: Read> Read for ProviderPipe<R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let Some((started, budget)) = self.deadline else {
            return self.inner.read(buffer);
        };
        loop {
            if self.stopping.load(std::sync::atomic::Ordering::Relaxed)
                || started.elapsed() >= budget
            {
                return Ok(0);
            }
            match self.inner.read(buffer) {
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(
                        Duration::from_millis(2).min(budget.saturating_sub(started.elapsed())),
                    );
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                result => return result,
            }
        }
    }
}

#[cfg(unix)]
fn set_provider_pipe_nonblocking(pipe: &impl std::os::fd::AsRawFd) -> std::io::Result<()> {
    let descriptor = pipe.as_raw_fd();
    let flags = unsafe { libc::fcntl(descriptor, libc::F_GETFL) };
    if flags < 0 || unsafe { libc::fcntl(descriptor, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0
    {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

fn reap_health_provider(mut child: Child, started: Instant, budget: Duration) {
    let _ = child.kill();
    loop {
        if matches!(child.try_wait(), Ok(Some(_))) {
            return;
        }
        if started.elapsed() >= budget {
            let _ = thread::Builder::new()
                .name("fabric-health-provider-reaper".into())
                .spawn(move || {
                    let _ = child.wait();
                });
            return;
        }
        thread::sleep(Duration::from_millis(2).min(budget.saturating_sub(started.elapsed())));
    }
}

struct ProviderProcess {
    command: OsString,
    child: Child,
    #[cfg(unix)]
    stdin: ChildStdin,
    responses: Receiver<std::io::Result<Vec<u8>>>,
    responses_thread: Option<thread::JoinHandle<()>>,
    stderr_thread: Option<thread::JoinHandle<()>>,
    stopping: Arc<std::sync::atomic::AtomicBool>,
}

impl ProviderProcess {
    fn spawn(command: OsString, operation: &str) -> Result<Self> {
        Self::spawn_with_deadline(command, operation, None)
    }

    fn spawn_with_deadline(
        command: OsString,
        operation: &str,
        deadline: Option<(Instant, Duration)>,
    ) -> Result<Self> {
        let mut child = Command::new(&command)
            .args(["serve", "--stdio"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| FabricError::EnvironmentProviderOperation {
                provider: OPEN_SHELL_PROVIDER_ID.to_string(),
                operation: operation.to_string(),
                code: "provider_unavailable".to_string(),
                message: format!("could not start `{}`: {error}", command.to_string_lossy()),
            })?;
        #[cfg(unix)]
        let stdin =
            child
                .stdin
                .take()
                .ok_or_else(|| FabricError::EnvironmentProviderOperation {
                    provider: OPEN_SHELL_PROVIDER_ID.to_string(),
                    operation: operation.to_string(),
                    code: "provider_protocol_error".to_string(),
                    message: "provider stdin was unavailable".to_string(),
                })?;
        let stdout =
            child
                .stdout
                .take()
                .ok_or_else(|| FabricError::EnvironmentProviderOperation {
                    provider: OPEN_SHELL_PROVIDER_ID.to_string(),
                    operation: operation.to_string(),
                    code: "provider_protocol_error".to_string(),
                    message: "provider stdout was unavailable".to_string(),
                })?;
        let stderr =
            child
                .stderr
                .take()
                .ok_or_else(|| FabricError::EnvironmentProviderOperation {
                    provider: OPEN_SHELL_PROVIDER_ID.to_string(),
                    operation: operation.to_string(),
                    code: "provider_protocol_error".to_string(),
                    message: "provider stderr was unavailable".to_string(),
                })?;

        let stopping = Arc::new(std::sync::atomic::AtomicBool::new(false));
        #[cfg(unix)]
        if let Some((started, budget)) = deadline
            && (set_provider_pipe_nonblocking(&stdout).is_err()
                || set_provider_pipe_nonblocking(&stderr).is_err())
        {
            reap_health_provider(child, started, budget);
            return provider_error(
                operation,
                "provider_io",
                "health provider pipes could not be configured".into(),
            );
        }
        let stdout = ProviderPipe {
            inner: stdout,
            deadline,
            stopping: stopping.clone(),
        };
        let mut stderr = ProviderPipe {
            inner: stderr,
            deadline,
            stopping: stopping.clone(),
        };
        let (response_sender, responses) = mpsc::sync_channel(1);
        let responses_thread = thread::spawn(move || {
            let mut stdout = BufReader::new(stdout);
            loop {
                let mut output = Vec::new();
                let read = {
                    let mut bounded = stdout
                        .by_ref()
                        .take((MAX_PROVIDER_RESPONSE_BYTES + 1) as u64);
                    bounded.read_until(b'\n', &mut output)
                };
                match read {
                    Ok(0) => break,
                    Ok(_) => {
                        if response_sender.try_send(Ok(output)).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = response_sender.try_send(Err(error));
                        break;
                    }
                }
            }
        });
        // Drain stderr to preserve transport liveness; routine errors never expose it.
        let stderr_thread = thread::spawn(move || {
            let mut chunk = [0_u8; 4096];
            while let Ok(read) = stderr.read(&mut chunk) {
                if read == 0 {
                    break;
                }
            }
        });
        Ok(Self {
            command,
            child,
            #[cfg(unix)]
            stdin,
            responses,
            responses_thread: Some(responses_thread),
            stderr_thread: Some(stderr_thread),
            stopping,
        })
    }

    #[cfg(unix)]
    fn write_request(&mut self, encoded: &[u8], timeout: Duration) -> std::io::Result<()> {
        use std::os::fd::AsRawFd;
        let descriptor = self.stdin.as_raw_fd();
        // The owned pipe remains nonblocking for subsequent bounded writes.
        let flags = unsafe { libc::fcntl(descriptor, libc::F_GETFL) };
        if flags < 0
            || unsafe { libc::fcntl(descriptor, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0
        {
            return Err(std::io::Error::last_os_error());
        }
        let deadline = Instant::now() + timeout;
        let mut remaining = encoded;
        while !remaining.is_empty() {
            if Instant::now() >= deadline {
                return Err(std::io::ErrorKind::TimedOut.into());
            }
            match self.stdin.write(remaining) {
                Ok(0) => return Err(std::io::ErrorKind::WriteZero.into()),
                Ok(written) => remaining = &remaining[written..],
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(5))
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    #[cfg(not(unix))]
    fn write_request(&mut self, _encoded: &[u8], _timeout: Duration) -> std::io::Result<()> {
        Err(std::io::ErrorKind::Unsupported.into())
    }

    fn has_exited(&mut self, operation: &str) -> Result<bool> {
        self.child
            .try_wait()
            .map(|status| status.is_some())
            .map_err(|error| FabricError::EnvironmentProviderOperation {
                provider: OPEN_SHELL_PROVIDER_ID.to_string(),
                operation: operation.to_string(),
                code: "provider_wait_failed".to_string(),
                message: error.to_string(),
            })
    }

    fn shutdown_check(mut self, started: Instant, budget: Duration) {
        self.stopping
            .store(true, std::sync::atomic::Ordering::Relaxed);
        reap_health_provider(self.child, started, budget);
        for reader in [self.responses_thread.take(), self.stderr_thread.take()]
            .into_iter()
            .flatten()
        {
            while !reader.is_finished() && started.elapsed() < budget {
                thread::sleep(
                    Duration::from_millis(2).min(budget.saturating_sub(started.elapsed())),
                );
            }
            if reader.is_finished() {
                let _ = reader.join();
            }
        }
    }

    fn shutdown(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(thread) = self.responses_thread.take() {
            let _ = thread.join();
        }
        if let Some(thread) = self.stderr_thread.take() {
            let _ = thread.join();
        }
    }
}

fn provider_error<T>(operation: &str, code: &str, message: String) -> Result<T> {
    Err(provider_failure(operation, code, message))
}

fn provider_failure(operation: &str, code: &str, message: String) -> FabricError {
    FabricError::EnvironmentProviderOperation {
        provider: OPEN_SHELL_PROVIDER_ID.to_string(),
        operation: operation.to_string(),
        code: code.to_string(),
        message,
    }
}

#[derive(Serialize)]
struct ProviderRequest<'a> {
    protocol_version: &'static str,
    request_id: &'a str,
    #[serde(flatten)]
    operation: ProviderOperation<'a>,
}

#[derive(Serialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
enum ProviderOperation<'a> {
    Prepare {
        environment_id: &'a str,
        environment: &'a EnvironmentPlan,
    },
    Attach {
        environment_id: &'a str,
        environment: &'a EnvironmentPlan,
        reference: &'a EnvironmentReference,
    },
    RuntimeCommand {
        environment: &'a EnvironmentHandle,
        command: &'a TaskCommand,
        timeout_seconds: u64,
    },
    CollectArtifacts {
        environment: &'a EnvironmentHandle,
        paths: &'a [String],
    },
    Release {
        environment: &'a EnvironmentHandle,
    },
}

impl ProviderOperation<'_> {
    fn name(&self) -> &'static str {
        match self {
            Self::Prepare { .. } => "prepare",
            Self::Attach { .. } => "attach",
            Self::RuntimeCommand { .. } => "runtime_command",
            Self::CollectArtifacts { .. } => "collect_artifacts",
            Self::Release { .. } => "release",
        }
    }

    fn response_timeout(&self) -> Duration {
        let operation_seconds = match self {
            Self::Prepare { environment, .. } => readiness_seconds(environment, true)
                .saturating_add(setting_seconds(
                    &environment.settings,
                    "delete_timeout_seconds",
                    30,
                ))
                .saturating_add(6 * PROVIDER_GATEWAY_CALL_TIMEOUT_SECONDS),
            Self::Attach { environment, .. } => readiness_seconds(environment, false)
                .saturating_add(3 * PROVIDER_GATEWAY_CALL_TIMEOUT_SECONDS),
            Self::RuntimeCommand {
                timeout_seconds, ..
            } => timeout_seconds.saturating_add(2 * PROVIDER_GATEWAY_CALL_TIMEOUT_SECONDS),
            Self::CollectArtifacts { .. } => 30 + 2 * PROVIDER_GATEWAY_CALL_TIMEOUT_SECONDS,
            Self::Release { environment } => connection_seconds(
                &environment.connection,
                "delete_timeout_seconds",
                DEFAULT_PROVIDER_OPERATION_TIMEOUT_SECONDS,
            )
            .saturating_add(3 * PROVIDER_GATEWAY_CALL_TIMEOUT_SECONDS),
        };
        Duration::from_secs(operation_seconds.saturating_add(PROVIDER_RESPONSE_GRACE_SECONDS))
    }
}

fn readiness_seconds(environment: &EnvironmentPlan, includes_sandbox_wait: bool) -> u64 {
    let provider_count = environment
        .settings
        .get("providers")
        .and_then(Value::as_array)
        .map_or(0, |providers| providers.len() as u64);
    let waits = provider_count.saturating_add(u64::from(includes_sandbox_wait));
    setting_seconds(&environment.settings, "ready_timeout_seconds", 60).saturating_mul(waits)
}

fn setting_seconds(settings: &serde_json::Map<String, Value>, key: &str, default: u64) -> u64 {
    settings
        .get(key)
        .and_then(Value::as_u64)
        .unwrap_or(default)
        .min(600)
}

fn connection_seconds(connection: &BTreeMap<String, Value>, key: &str, default: u64) -> u64 {
    connection
        .get(key)
        .and_then(Value::as_u64)
        .unwrap_or(default)
        .min(600)
}

#[derive(Deserialize)]
struct ProviderResponse {
    protocol_version: String,
    request_id: String,
    #[serde(flatten)]
    outcome: ProviderOutcome,
}

#[derive(Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum ProviderOutcome {
    Succeeded { output: Value },
    Failed { error: ProviderFailure },
}

#[derive(Deserialize)]
struct ProviderFailure {
    code: String,
    message: String,
}

#[derive(Deserialize)]
struct PreparedEnvironment {
    workspace: Option<PathBuf>,
    artifacts: Option<PathBuf>,
    #[serde(default)]
    connection: serde_json::Map<String, Value>,
    #[serde(default)]
    metadata: serde_json::Map<String, Value>,
}

#[derive(Debug, Deserialize, PartialEq)]
pub(crate) struct ProviderReleaseOutput {
    pub(crate) released: bool,
    pub(crate) detached: bool,
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn prepare_response_timeout_covers_cleanup_after_readiness() {
        let environment: EnvironmentPlan = serde_json::from_value(serde_json::json!({
            "provider": OPEN_SHELL_PROVIDER_ID,
            "control_location": ControlLocation::InEnvControl,
            "ownership": EnvironmentOwnership::FabricOwned,
            "workspace": null,
            "artifacts": null,
            "env": {},
            "env_names": [],
            "connection": {},
            "metadata": {},
            "settings": {
                "ready_timeout_seconds": 60,
                "delete_timeout_seconds": 30
            }
        }))
        .expect("environment fixture");
        let operation = ProviderOperation::Prepare {
            environment_id: "environment-test",
            environment: &environment,
        };

        assert_eq!(operation.response_timeout(), Duration::from_secs(285));
    }

    #[test]
    fn attach_response_timeout_covers_each_provider_readiness_wait() {
        let environment: EnvironmentPlan = serde_json::from_value(serde_json::json!({
            "provider": OPEN_SHELL_PROVIDER_ID,
            "control_location": ControlLocation::InEnvControl,
            "ownership": EnvironmentOwnership::CallerOwned,
            "workspace": null,
            "artifacts": null,
            "env": {},
            "env_names": [],
            "connection": {},
            "metadata": {},
            "settings": {
                "ready_timeout_seconds": 180,
                "providers": ["model", "tools"]
            }
        }))
        .expect("environment fixture");
        let reference = EnvironmentReference {
            provider: OPEN_SHELL_PROVIDER_ID.to_string(),
            resource: BTreeMap::new(),
        };
        let attach = ProviderOperation::Attach {
            environment_id: "environment-test",
            environment: &environment,
            reference: &reference,
        };
        assert_eq!(attach.response_timeout(), Duration::from_secs(465));

        let prepare = ProviderOperation::Prepare {
            environment_id: "environment-test",
            environment: &environment,
        };
        assert_eq!(prepare.response_timeout(), Duration::from_secs(765));
    }

    #[test]
    fn runtime_operation_deadlines_cover_gateway_setup_and_execution() {
        let environment = EnvironmentHandle {
            environment_id: "environment-test".to_string(),
            provider: "openshell".to_string(),
            control_location: ControlLocation::InEnvControl,
            ownership: EnvironmentOwnership::CallerOwned,
            workspace: Some(PathBuf::from("/sandbox/workspace")),
            artifacts: Some(PathBuf::from("/sandbox/artifacts")),
            env: BTreeMap::new(),
            metadata: BTreeMap::new(),
            connection: BTreeMap::new(),
        };
        let command = TaskCommand::Stop {
            agent: "echo".to_string(),
            runtime_id: "runtime-task".to_string(),
        };
        let runtime = ProviderOperation::RuntimeCommand {
            environment: &environment,
            command: &command,
            timeout_seconds: 315,
        };
        assert_eq!(runtime.response_timeout(), Duration::from_secs(390));

        let paths = Vec::new();
        let artifacts = ProviderOperation::CollectArtifacts {
            environment: &environment,
            paths: &paths,
        };
        assert_eq!(artifacts.response_timeout(), Duration::from_secs(105));
    }

    fn provider_script(body: &str) -> (PathBuf, OsString) {
        let directory = std::env::temp_dir().join(new_id("fabric-provider-test"));
        fs::create_dir_all(&directory).expect("create test directory");
        let path = directory.join("provider");
        fs::write(&path, format!("#!/usr/bin/env python3\n{body}\n")).expect("write test provider");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).expect("executable provider");
        (directory, path.into_os_string())
    }

    #[test]
    fn request_write_is_bounded_when_the_provider_does_not_read() {
        let (directory, command) = provider_script("import time\ntime.sleep(30)");
        let mut process = ProviderProcess::spawn(command, "prepare").expect("start provider");
        let started = Instant::now();
        let error = process
            .write_request(&vec![b'x'; 1024 * 1024], Duration::from_millis(30))
            .expect_err("blocked pipe write must expire");
        assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
        assert!(started.elapsed() < Duration::from_secs(1));
        process.shutdown();
        fs::remove_dir_all(directory).expect("remove fixture");
    }

    #[cfg(unix)]
    #[test]
    fn runtime_provider_workers_are_scoped_to_runtime_ids() {
        let (directory, command) = provider_script("import time\ntime.sleep(30)");
        let prefix = new_id("runtime-worker-test");
        let first_key = format!("{prefix}:sandbox:runtime-one");
        let second_key = format!("{prefix}:sandbox:runtime-two");
        let first = runtime_provider_process(&first_key, &command, "runtime_command")
            .expect("start first runtime worker");
        let first_again = runtime_provider_process(&first_key, &command, "runtime_command")
            .expect("reuse first runtime worker");
        let second = runtime_provider_process(&second_key, &command, "runtime_command")
            .expect("start second runtime worker");

        assert!(Arc::ptr_eq(&first, &first_again));
        assert!(!Arc::ptr_eq(&first, &second));

        retire_runtime_provider_process(&first_key, &first);
        retire_runtime_provider_process(&second_key, &second);
        fs::remove_dir_all(directory).expect("remove fixture");
    }

    #[test]
    fn health_provider_cleanup_is_bounded_with_inherited_pipes() {
        for pipe in ["stdout", "stderr", "both"] {
            for response in ["timeout", "valid", "malformed"] {
                let (directory, command) = provider_script(&format!(
                    r#"
import json, subprocess, sys, time
from pathlib import Path
request = json.loads(sys.stdin.readline())
directory = Path(__file__).parent
child_code = '''
import os, sys, time
from pathlib import Path
directory = Path(sys.argv[1])
(directory / 'holding').write_text(str(os.getpid()))
deadline = time.monotonic() + 2
while time.monotonic() < deadline and not (directory / 'release').exists():
    time.sleep(0.005)
(directory / 'released').touch()
'''
pipe = {pipe:?}
subprocess.Popen(
    [sys.executable, '-c', child_code, str(directory)],
    stdin=subprocess.DEVNULL,
    stdout=None if pipe in ('stdout', 'both') else subprocess.DEVNULL,
    stderr=None if pipe in ('stderr', 'both') else subprocess.DEVNULL,
    start_new_session=True,
)
while not (directory / 'holding').exists():
    time.sleep(0.005)
response = {response:?}
if response == 'timeout':
    time.sleep(5)
elif response == 'malformed':
    print('{{', flush=True)
else:
    print(json.dumps({{'protocol_version': request['protocol_version'],
                      'request_id': request['request_id'], 'status': 'succeeded',
                      'output': {{'returned': True}}}}), flush=True)
"#
                ));
                let environment: EnvironmentHandle = serde_json::from_value(serde_json::json!({
                    "environment_id": "health-cleanup",
                    "provider": "openshell",
                    "control_location": "in_env_control",
                    "ownership": "caller_owned"
                }))
                .expect("environment fixture");
                let timeout_seconds = if response == "timeout" { 0.3 } else { 1.0 };
                let task = TaskCommand::Check {
                    timeout_seconds,
                    agent: None,
                    runtime_id: None,
                    level: None,
                };
                let started = Instant::now();
                let result = request_check_provider_process::<Value>(
                    command,
                    ProviderOperation::RuntimeCommand {
                        environment: &environment,
                        command: &task,
                        timeout_seconds: 1,
                    },
                    timeout_seconds,
                );
                let elapsed = started.elapsed();
                fs::write(directory.join("release"), "").expect("release pipe holder");
                for _ in 0..100 {
                    if directory.join("released").exists() {
                        break;
                    }
                    thread::sleep(Duration::from_millis(5));
                }
                fs::remove_dir_all(directory).expect("remove fixture");
                assert!(
                    elapsed < Duration::from_millis(1500),
                    "{pipe}/{response} cleanup took {elapsed:?}"
                );
                if response == "valid" {
                    assert_eq!(
                        result.expect("valid health response"),
                        serde_json::json!({"returned": true})
                    );
                } else {
                    assert!(result.is_err(), "{pipe}/{response} must fail");
                }
            }
        }
    }

    #[test]
    fn malformed_provider_frames_never_expose_stderr_or_response_data() {
        let (directory, command) = provider_script(
            r#"
import json, sys
request = json.loads(sys.stdin.readline())
print('PRIVATE stderr', file=sys.stderr, flush=True)
print(json.dumps({'protocol_version': 'PRIVATE response', 'request_id': request['request_id'], 'status': 'succeeded', 'output': {}}), flush=True)
"#,
        );
        let environment = EnvironmentHandle {
            environment_id: "environment-test".to_string(),
            provider: "openshell".to_string(),
            control_location: ControlLocation::InEnvControl,
            ownership: EnvironmentOwnership::CallerOwned,
            workspace: Some(PathBuf::from("/sandbox/workspace")),
            artifacts: None,
            env: BTreeMap::new(),
            connection: BTreeMap::new(),
            metadata: BTreeMap::new(),
        };
        let error = request_provider_process::<ProviderReleaseOutput>(
            command,
            ProviderOperation::Release {
                environment: &environment,
            },
        )
        .expect_err("reject an incompatible provider");
        assert!(
            matches!(&error, FabricError::EnvironmentProviderOperation {code, ..} if code == "provider_protocol_mismatch")
        );
        assert!(!error.to_string().contains("PRIVATE"));
        fs::remove_dir_all(directory).expect("remove fixture");
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CollectedArtifact {
    pub(crate) path: String,
    pub(crate) content: Vec<u8>,
}
