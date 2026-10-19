// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenShell environment-provider process used by host Fabric core.
//!
//! Serves private JSON/stdio requests to create, attach, inspect, execute in,
//! and release sandboxes through the OpenShell gateway. It verifies sandbox
//! identity and policy before operations and exposes bounded sandbox exec;
//! agent runtime state is outside this provider.

use std::collections::HashMap;
use std::future::Future;
use std::io::{BufRead, Write};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use openshell_sdk::raw::proto;
use openshell_sdk::{
    AuthConfig, ClientConfig, DeleteOptions, DeletionOutcome, DeletionResult, ExecOptions,
    OpenShellClient, SandboxPhase, SandboxRef, SdkError, ServiceStatus,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use thiserror::Error;

const PROTOCOL_VERSION: &str = "fabric.environment-provider.v1alpha1";
const MAX_REQUEST_BYTES: usize = 1024 * 1024;
const MAX_EXEC_OUTPUT_BYTES: usize = 3 * 1024 * 1024;
const DEFAULT_READY_TIMEOUT_SECONDS: u64 = 60;
const DEFAULT_DELETE_TIMEOUT_SECONDS: u64 = 30;
const DEFAULT_EXEC_TIMEOUT_SECONDS: u64 = 30;
const GATEWAY_CALL_TIMEOUT_SECONDS: u64 = 30;
const MAX_OPERATION_TIMEOUT_SECONDS: u64 = 600;

// OpenShell 0.1.2 limits DNS-routable sandbox names to 19 bytes. Keep the
// environment ID in the label for exact identity checks during reconciliation.
fn generated_sandbox_name(environment_id: &str) -> String {
    let digest = Sha256::digest(environment_id.as_bytes());
    let mut prefix = [0_u8; 8];
    prefix.copy_from_slice(&digest[..8]);
    format!("nf-{:016x}", u64::from_be_bytes(prefix))
}

/// Provider process failure.
#[derive(Debug, Error)]
pub enum ProviderError {
    /// Standard input or output failed.
    #[error("provider I/O failed: {0}")]
    Io(#[from] std::io::Error),
    /// The request violated the provider contract.
    #[error("{message}")]
    Contract {
        /// Stable error code.
        code: String,
        /// Sanitized detail.
        message: String,
    },
}

impl ProviderError {
    fn contract(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Contract {
            code: code.into(),
            message: message.into(),
        }
    }

    fn code(&self) -> &str {
        match self {
            Self::Io(_) => "provider_io",
            Self::Contract { code, .. } => code,
        }
    }
}

/// Serve newline-delimited provider requests on standard input and output.
pub async fn serve_stdio() -> Result<(), ProviderError> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    serve(stdin.lock(), stdout.lock(), &SdkGatewayFactory).await
}

async fn serve<R, W, F>(mut reader: R, mut writer: W, factory: &F) -> Result<(), ProviderError>
where
    R: BufRead,
    W: Write,
    F: GatewayFactory,
{
    loop {
        let line = match read_request_frame(&mut reader)? {
            RequestFrame::EndOfInput => return Ok(()),
            RequestFrame::Oversized => {
                let response = ProviderResponse::failed(
                    "unknown",
                    "request_too_large",
                    format!("request exceeds the {MAX_REQUEST_BYTES}-byte limit"),
                );
                serde_json::to_writer(&mut writer, &response).map_err(|_error| {
                    ProviderError::contract(
                        "response_serialization",
                        "provider response could not be encoded",
                    )
                })?;
                writer.write_all(b"\n")?;
                writer.flush()?;
                continue;
            }
            RequestFrame::Data(line) => line,
        };
        let response = match serde_json::from_slice::<ProviderRequest>(&line) {
            Ok(request) => handle_request(request, factory).await,
            Err(_error) => ProviderResponse::failed(
                "unknown",
                "invalid_request",
                "request was not valid provider JSON",
            ),
        };
        serde_json::to_writer(&mut writer, &response).map_err(|_error| {
            ProviderError::contract(
                "response_serialization",
                "provider response could not be encoded",
            )
        })?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }
}

enum RequestFrame {
    EndOfInput,
    Data(Vec<u8>),
    Oversized,
}

fn read_request_frame(reader: &mut impl BufRead) -> std::io::Result<RequestFrame> {
    let mut line = Vec::new();
    let mut oversized = false;
    loop {
        let (consumed, terminated) = {
            let available = reader.fill_buf()?;
            if available.is_empty() {
                return if line.is_empty() {
                    Ok(RequestFrame::EndOfInput)
                } else if oversized {
                    Ok(RequestFrame::Oversized)
                } else {
                    Ok(RequestFrame::Data(line))
                };
            }
            let newline = available.iter().position(|byte| *byte == b'\n');
            let consumed = newline.map_or(available.len(), |position| position + 1);
            if !oversized {
                let remaining = (MAX_REQUEST_BYTES + 1).saturating_sub(line.len());
                line.extend_from_slice(&available[..consumed.min(remaining)]);
                oversized = line.len() > MAX_REQUEST_BYTES;
            }
            (consumed, newline.is_some())
        };
        reader.consume(consumed);
        if terminated {
            return if oversized {
                Ok(RequestFrame::Oversized)
            } else {
                Ok(RequestFrame::Data(line))
            };
        }
    }
}

async fn handle_request<F>(request: ProviderRequest, factory: &F) -> ProviderResponse
where
    F: GatewayFactory,
{
    let request_id = request.request_id.clone();
    if request.protocol_version != PROTOCOL_VERSION {
        return ProviderResponse::failed(
            &request_id,
            "protocol_mismatch",
            format!(
                "expected `{PROTOCOL_VERSION}` but received `{}`",
                request.protocol_version
            ),
        );
    }
    match handle_operation(request.operation, factory).await {
        Ok(output) => ProviderResponse::succeeded(&request_id, output),
        Err(error) => ProviderResponse::failed(&request_id, error.code(), error.to_string()),
    }
}

async fn handle_operation<F>(
    operation: ProviderOperation,
    factory: &F,
) -> Result<Value, ProviderError>
where
    F: GatewayFactory,
{
    match operation {
        ProviderOperation::RuntimeCommand {
            environment,
            command,
            timeout_seconds,
        } => runtime_command(environment, command, timeout_seconds, factory).await,
        ProviderOperation::CollectArtifacts { environment, paths } => {
            collect_artifacts(environment, paths, factory).await
        }
        ProviderOperation::Prepare {
            environment_id,
            environment,
        } => prepare_environment(&environment_id, environment, factory).await,
        ProviderOperation::Attach {
            environment_id,
            environment,
            reference,
        } => attach_environment(&environment_id, environment, reference, factory).await,
        ProviderOperation::Inspect { environment } => {
            inspect_environment(environment, factory).await
        }
        ProviderOperation::Exec {
            environment,
            command,
            workdir,
            env,
            timeout_seconds,
            stdin,
        } => {
            exec_environment(
                environment,
                command,
                workdir,
                env,
                timeout_seconds,
                stdin,
                factory,
            )
            .await
        }
        ProviderOperation::Release { environment } => {
            release_environment(environment, factory).await
        }
    }
}

async fn runtime_command<F: GatewayFactory>(
    environment: EnvironmentHandle,
    command: TaskCommand,
    timeout_seconds: u64,
    factory: &F,
) -> Result<Value, ProviderError> {
    if !(1..=320).contains(&timeout_seconds) {
        return Err(ProviderError::contract(
            "invalid_runtime_command",
            "runtime command timeout is unsupported",
        ));
    }
    let started = Instant::now();
    let health_budget = command
        .check_timeout()
        .map(|seconds| Duration::try_from_secs_f64(seconds).unwrap_or(Duration::MAX));
    let remaining = |maximum: Duration| {
        let duration = health_budget
            .map(|budget| budget.saturating_sub(started.elapsed()).min(maximum))
            .unwrap_or(maximum);
        if duration.is_zero() {
            Err(ProviderError::contract(
                "probe_timeout",
                "health deadline expired",
            ))
        } else {
            Ok(duration)
        }
    };
    let (operation, mut arguments, payload) = command.arguments()?;
    let binding = SandboxBinding::from_environment(&environment)?;
    let call_timeout = Duration::from_secs(GATEWAY_CALL_TIMEOUT_SECONDS);
    let client = bounded_gateway_call(
        "connect",
        remaining(call_timeout)?,
        factory.connect(&binding.connection),
    )
    .await?;
    let sandbox = bounded_gateway_call(
        "get",
        remaining(call_timeout)?,
        client.get(binding.connection.workspace.as_deref(), &binding.name),
    )
    .await?;
    binding.verify(&sandbox)?;
    let stdin = payload
        .map(|value| serde_json::to_vec(&value))
        .transpose()
        .map_err(|_| {
            ProviderError::contract(
                "invalid_runtime_command",
                "SDK command payload could not be encoded",
            )
        })?;
    if stdin.as_ref().is_some_and(|value| value.len() > 512 * 1024) {
        return Err(ProviderError::contract(
            "request_too_large",
            "SDK command payload exceeds its limit",
        ));
    }
    let exec_timeout = remaining(Duration::from_secs(timeout_seconds))?;
    if health_budget.is_some() {
        let index = arguments
            .iter()
            .position(|value| value == "--timeout-seconds")
            .expect("health timeout argument");
        arguments[index + 1] = exec_timeout.as_secs_f64().to_string();
    }
    let executed = bounded_gateway_call(
        "exec",
        exec_timeout,
        client.exec(
            binding.connection.workspace.as_deref(),
            &binding.name,
            arguments,
            ExecRequest {
                workdir: None,
                environment: HashMap::new(),
                timeout: exec_timeout,
                stdin,
            },
        ),
    )
    .await?;
    if executed.stdout.len().saturating_add(executed.stderr.len()) > 4 * 1024 * 1024 {
        return Err(ProviderError::contract(
            "response_too_large",
            "task command output exceeded its limit; effects are unknown",
        ));
    }
    if executed.stdout.is_empty() {
        let diagnostic: TaskDiagnosticEnvelope =
            serde_json::from_slice(&executed.stderr).map_err(|_| {
                ProviderError::contract("invalid_response", "task command outcome is unknown")
            })?;
        let error = diagnostic.error;
        if executed.exit_code == 0
            || diagnostic.operation != operation
            || !matches!(diagnostic.status.as_str(), "failed" | "unsupported")
            || !matches!(error.effects.as_str(), "none" | "applied" | "unknown")
            || !matches!(
                error.stage.as_str(),
                "request" | "validation" | "start" | "invoke" | "stop" | "check" | "transport"
            )
            || !valid_diagnostic_code(&error.code)
        {
            return Err(ProviderError::contract(
                "invalid_response",
                "task command diagnostic was invalid; effects are unknown",
            ));
        }
        return Ok(
            json!({"status": "rejected", "error": {"code": error.code, "stage": error.stage, "effects": error.effects}}),
        );
    }
    decode_sdk_command_response(operation, executed.exit_code, &executed.stdout)
}

fn decode_sdk_command_response(
    operation: &str,
    exit_code: i32,
    stdout: &[u8],
) -> Result<Value, ProviderError> {
    if stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .count()
        != 1
    {
        return Err(ProviderError::contract(
            "invalid_response",
            "task command output was not one SDK response envelope",
        ));
    }
    let response: Value = serde_json::from_slice(stdout).map_err(|_| {
        ProviderError::contract("invalid_response", "task command did not return SDK JSON")
    })?;
    let Some(response) = response.as_object() else {
        return Err(ProviderError::contract(
            "invalid_response",
            "task command response was not an SDK envelope",
        ));
    };
    if response.len() != 5
        || response.get("operation").and_then(Value::as_str) != Some(operation)
        || !matches!(
            response.get("status").and_then(Value::as_str),
            Some("succeeded" | "failed" | "unsupported")
        )
        || !response
            .get("changed")
            .is_some_and(|value| value.is_null() || value.is_boolean())
        || !response
            .get("result")
            .is_some_and(|value| value.is_null() || value.is_object())
        || !response.contains_key("error")
        || !response.contains_key("operation")
        || !response.contains_key("status")
        || !response.contains_key("changed")
        || !response.contains_key("result")
    {
        return Err(ProviderError::contract(
            "invalid_response",
            "task command returned an invalid SDK response envelope",
        ));
    }

    let status = response["status"].as_str().unwrap_or_default();
    let result = &response["result"];
    let error = &response["error"];
    if operation == "check" {
        if response["changed"] != false || (!error.is_null() && error["effects"] != "none") {
            return Err(ProviderError::contract(
                "invalid_response",
                "health command reported lifecycle effects",
            ));
        }
        if result.is_object() {
            let expected = match result["outcome"].as_str() {
                Some("passed") if result["reason_code"].is_null() => "succeeded",
                Some("failed" | "unknown") => "failed",
                Some("unsupported") => "unsupported",
                _ => {
                    return Err(ProviderError::contract(
                        "invalid_response",
                        "invalid health outcome",
                    ));
                }
            };
            let passed = expected == "succeeded";
            if status != expected
                || (exit_code == 0) != passed
                || passed != error.is_null()
                || (!passed
                    && (!error.is_object()
                        || error.as_object().is_none_or(|value| value.len() != 4)
                        || error["code"] != result["reason_code"]
                        || error["stage"] != "check"
                        || !error["message"].is_string()
                        || !valid_diagnostic_code(error["code"].as_str().unwrap_or_default())))
            {
                return Err(ProviderError::contract(
                    "invalid_response",
                    "health outcome and command envelope disagree",
                ));
            }
            return Ok(json!({"status": "returned", "value": result}));
        }
    }
    if status == "succeeded" {
        if exit_code != 0
            || !error.is_null()
            || (operation == "stop" && !result.is_null())
            || (operation != "stop" && !result.is_object())
        {
            return Err(ProviderError::contract(
                "invalid_response",
                "task command exit code and successful SDK result disagree",
            ));
        }
        return Ok(json!({"status": "returned", "value": result}));
    }

    let Some(error) = error.as_object() else {
        return Err(ProviderError::contract(
            "invalid_response",
            "failed SDK command omitted its diagnostic",
        ));
    };
    let code = error
        .get("code")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let stage = error
        .get("stage")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let effects = error
        .get("effects")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if exit_code == 0
        || error.len() != 4
        || !error.contains_key("message")
        || error.get("message").and_then(Value::as_str).is_none()
        || !valid_diagnostic_code(code)
        || !matches!(
            stage,
            "request" | "validation" | "start" | "invoke" | "stop" | "check" | "transport"
        )
        || !matches!(effects, "none" | "applied" | "unknown")
    {
        return Err(ProviderError::contract(
            "invalid_response",
            "failed SDK command diagnostic was invalid",
        ));
    }
    Ok(json!({
        "status": "rejected",
        "error": {"code": code, "stage": stage, "effects": effects}
    }))
}

fn valid_diagnostic_code(code: &str) -> bool {
    matches!(
        code,
        "platform_unsupported"
            | "cleanup_unconfirmed"
            | "invalid_request"
            | "interface_unsupported"
            | "wrong_agent"
            | "validation_failed"
            | "validation_unavailable"
            | "start_failed"
            | "start_uncertain"
            | "runtime_id_conflict"
            | "stop_failed"
            | "invoke_failed"
            | "host_stopping"
            | "runtime_unavailable"
            | "runtime_exists"
            | "streaming_unsupported"
            | "host_unavailable"
            | "invalid_response"
            | "request_too_large"
            | "response_too_large"
            | "invalid_input"
            | "server_stopping"
            | "probe_timeout"
            | "probe_error"
    )
}

async fn collect_artifacts<F: GatewayFactory>(
    environment: EnvironmentHandle,
    paths: Vec<String>,
    factory: &F,
) -> Result<Value, ProviderError> {
    if paths.len() > 16
        || paths.iter().any(|path| {
            path.is_empty()
                || path.contains('\\')
                || path.contains('\0')
                || path
                    .split('/')
                    .any(|part| part.is_empty() || part == "." || part == "..")
        })
    {
        return Err(ProviderError::contract(
            "invalid_artifact_path",
            "declared artifact paths are invalid",
        ));
    }
    let root = environment
        .artifacts
        .clone()
        .unwrap_or_else(|| "/sandbox/artifacts".to_string());
    let interpreter = environment
        .env
        .get("ADAPTER_PYTHON")
        .cloned()
        .unwrap_or_else(|| "/opt/fabric/bin/python".to_string());
    let binding = SandboxBinding::from_environment(&environment)?;
    let call_timeout = Duration::from_secs(GATEWAY_CALL_TIMEOUT_SECONDS);
    let client = bounded_gateway_call(
        "connect",
        call_timeout,
        factory.connect(&binding.connection),
    )
    .await?;
    let sandbox = bounded_gateway_call(
        "get",
        call_timeout,
        client.get(binding.connection.workspace.as_deref(), &binding.name),
    )
    .await?;
    binding.verify(&sandbox)?;
    let stdin = serde_json::to_vec(&json!({"root": root, "paths": paths})).map_err(|_| {
        ProviderError::contract(
            "invalid_artifact_path",
            "artifact request could not be encoded",
        )
    })?;
    let executed = bounded_gateway_call(
        "exec",
        Duration::from_secs(30),
        client.exec(
            binding.connection.workspace.as_deref(),
            &binding.name,
            vec![
                interpreter,
                "-c".to_string(),
                include_str!("read_artifacts.py").to_string(),
            ],
            ExecRequest {
                workdir: None,
                environment: HashMap::new(),
                timeout: Duration::from_secs(30),
                stdin: Some(stdin),
            },
        ),
    )
    .await?;
    if executed.exit_code != 0
        || executed.stdout.len().saturating_add(executed.stderr.len()) > 3 * 1024 * 1024
    {
        return Err(ProviderError::contract(
            "artifact_read_failed",
            "declared artifacts could not be read as bounded regular files",
        ));
    }
    let response: ArtifactResponse = serde_json::from_slice(&executed.stdout).map_err(|_| {
        ProviderError::contract(
            "artifact_read_failed",
            "artifact reader returned an invalid response",
        )
    })?;
    if response.files.len() != paths.len()
        || response
            .files
            .iter()
            .zip(paths)
            .any(|(file, path)| file.path != path || file.content.len() > 256 * 1024)
        || response
            .files
            .iter()
            .map(|file| file.content.len())
            .sum::<usize>()
            > 512 * 1024
    {
        return Err(ProviderError::contract(
            "artifact_read_failed",
            "artifact reader failed correlation or size checks",
        ));
    }
    serde_json::to_value(response.files).map_err(|_| {
        ProviderError::contract(
            "artifact_read_failed",
            "artifact output could not be encoded",
        )
    })
}

async fn prepare_environment<F>(
    environment_id: &str,
    environment: EnvironmentPlan,
    factory: &F,
) -> Result<Value, ProviderError>
where
    F: GatewayFactory,
{
    validate_profile(&environment, "fabric_owned")?;
    let connection = OpenShellConnection::from_map(&environment.connection)?;
    let settings = OpenShellSettings::from_map(environment.settings)?;
    let policy_attached = settings.policy.is_some();
    let client = bounded_gateway_call(
        "connect",
        Duration::from_secs(GATEWAY_CALL_TIMEOUT_SECONDS),
        factory.connect(&connection),
    )
    .await?;
    let health = bounded_gateway_call(
        "health",
        Duration::from_secs(GATEWAY_CALL_TIMEOUT_SECONDS),
        client.health(),
    )
    .await?;
    if health.status != GatewayStatus::Healthy {
        return Err(ProviderError::contract(
            "gateway_unhealthy",
            format!("OpenShell gateway reported {:?}", health.status),
        ));
    }

    let sandbox_name = settings
        .sandbox_name
        .clone()
        .unwrap_or_else(|| generated_sandbox_name(environment_id));
    let sandbox = match bounded_gateway_call(
        "create",
        Duration::from_secs(GATEWAY_CALL_TIMEOUT_SECONDS),
        client.create(
            connection.workspace.as_deref(),
            SandboxCreate {
                name: Some(sandbox_name.clone()),
                image: settings.image.clone(),
                labels: HashMap::from([(
                    "nemo-fabric.environment-id".to_string(),
                    environment_id.to_string(),
                )]),
                environment: environment.env,
                providers: settings.providers,
                command: settings.command,
                policy: settings.policy,
            },
        ),
    )
    .await
    {
        Ok(sandbox) => sandbox,
        Err(error) => {
            let current = bounded_gateway_call(
                "get",
                Duration::from_secs(GATEWAY_CALL_TIMEOUT_SECONDS),
                client.get(connection.workspace.as_deref(), &sandbox_name),
            )
            .await;
            match current {
                Ok(current)
                    if current
                        .labels
                        .get("nemo-fabric.environment-id")
                        .is_some_and(|value| value == environment_id) =>
                {
                    cleanup_created_sandbox(
                        client.as_ref(),
                        connection.workspace.as_deref(),
                        &current.name,
                        &current.id,
                        settings.delete_timeout_seconds,
                    )
                    .await?;
                    return Err(error);
                }
                Ok(_) => return Err(error),
                Err(lookup_error) if lookup_error.code() == "not_found" => return Err(error),
                Err(_) => return Err(unconfirmed_cleanup_error(&sandbox_name, "unknown")),
            }
        }
    };
    let ready = match bounded_gateway_call(
        "wait_ready",
        Duration::from_secs(settings.ready_timeout_seconds),
        client.wait_ready(
            connection.workspace.as_deref(),
            &sandbox.name,
            Duration::from_secs(settings.ready_timeout_seconds),
        ),
    )
    .await
    {
        Ok(ready) => ready,
        Err(error) => {
            cleanup_created_sandbox(
                client.as_ref(),
                connection.workspace.as_deref(),
                &sandbox.name,
                &sandbox.id,
                settings.delete_timeout_seconds,
            )
            .await?;
            return Err(error);
        }
    };
    if ready.id != sandbox.id
        || ready.workspace != connection.workspace.as_deref().unwrap_or("default")
        || sandbox.workspace != connection.workspace.as_deref().unwrap_or("default")
    {
        cleanup_created_sandbox(
            client.as_ref(),
            connection.workspace.as_deref(),
            &sandbox.name,
            &sandbox.id,
            settings.delete_timeout_seconds,
        )
        .await?;
        return Err(ProviderError::contract(
            "sandbox_identity_changed",
            "OpenShell returned a different sandbox id while waiting for readiness",
        ));
    }

    let mut provider_connection = connection.to_safe_map();
    provider_connection.insert("sandbox_id".to_string(), json!(ready.id));
    provider_connection.insert("sandbox_name".to_string(), json!(ready.name));
    provider_connection.insert(
        "delete_timeout_seconds".to_string(),
        json!(settings.delete_timeout_seconds),
    );
    provider_connection.insert(
        "exec_timeout_seconds".to_string(),
        json!(settings.exec_timeout_seconds),
    );
    Ok(json!({
        "workspace": environment.workspace,
        "artifacts": environment.artifacts,
        "connection": provider_connection,
        "metadata": {
            "openshell.gateway_version": health.version,
            "openshell.sandbox_id": ready.id,
            "openshell.sandbox_name": ready.name,
            "openshell.sandbox_phase": ready.phase.as_str(),
            "openshell.sandbox_resource_version": ready.resource_version,
            "openshell.runtime_image": settings.image,
            "openshell.policy_attached": policy_attached,
        },
    }))
}

async fn attach_environment<F>(
    environment_id: &str,
    environment: EnvironmentPlan,
    reference: EnvironmentReference,
    factory: &F,
) -> Result<Value, ProviderError>
where
    F: GatewayFactory,
{
    validate_profile(&environment, "caller_owned")?;
    if reference.provider != "openshell" {
        return Err(ProviderError::contract(
            "invalid_reference",
            "environment reference provider must be `openshell`",
        ));
    }
    let resource = OpenShellResourceReference::from_map(reference.resource)?;
    let connection = OpenShellConnection::from_map(&environment.connection)?;
    let settings = OpenShellSettings::from_map(environment.settings)?;
    let call_timeout = Duration::from_secs(GATEWAY_CALL_TIMEOUT_SECONDS);
    let client =
        bounded_gateway_call("connect", call_timeout, factory.connect(&connection)).await?;
    let health = bounded_gateway_call("health", call_timeout, client.health()).await?;
    if health.status != GatewayStatus::Healthy {
        return Err(ProviderError::contract(
            "gateway_unhealthy",
            format!("OpenShell gateway reported {:?}", health.status),
        ));
    }
    let sandbox = bounded_gateway_call(
        "get",
        call_timeout,
        client.get(connection.workspace.as_deref(), &resource.sandbox_name),
    )
    .await?;
    if sandbox.id != resource.sandbox_id
        || sandbox.workspace != connection.workspace.as_deref().unwrap_or("default")
    {
        return Err(ProviderError::contract(
            "sandbox_identity_changed",
            "OpenShell sandbox name resolved to a different id",
        ));
    }
    if !matches!(sandbox.phase, ProviderSandboxPhase::Ready) {
        return Err(ProviderError::contract(
            "sandbox_not_ready",
            format!("OpenShell sandbox is `{}`", sandbox.phase.as_str()),
        ));
    }
    if sandbox.image.as_deref() != Some(settings.image.as_str()) {
        return Err(ProviderError::contract(
            "sandbox_image_mismatch",
            "OpenShell sandbox image does not match settings.image",
        ));
    }
    if sandbox.command.as_deref() != Some(settings.command.as_slice()) {
        return Err(ProviderError::contract(
            "sandbox_command_mismatch",
            "OpenShell sandbox command does not match settings.command",
        ));
    }
    if let Some(expected_policy) = settings.policy.as_ref()
        && sandbox.policy.as_ref() != Some(expected_policy)
    {
        return Err(ProviderError::contract(
            "sandbox_policy_mismatch",
            "OpenShell sandbox policy does not match settings.policy_yaml",
        ));
    }

    let mut provider_connection = connection.to_safe_map();
    provider_connection.insert("sandbox_id".to_string(), json!(sandbox.id));
    provider_connection.insert("sandbox_name".to_string(), json!(sandbox.name));
    provider_connection.insert(
        "delete_timeout_seconds".to_string(),
        json!(settings.delete_timeout_seconds),
    );
    provider_connection.insert(
        "exec_timeout_seconds".to_string(),
        json!(settings.exec_timeout_seconds),
    );
    Ok(json!({
        "workspace": environment.workspace,
        "artifacts": environment.artifacts,
        "connection": provider_connection,
        "metadata": {
            "openshell.gateway_version": health.version,
            "openshell.sandbox_id": sandbox.id,
            "openshell.sandbox_name": sandbox.name,
            "openshell.sandbox_phase": sandbox.phase.as_str(),
            "openshell.sandbox_resource_version": sandbox.resource_version,
            "openshell.runtime_image": settings.image,
            "openshell.policy_attached": sandbox.policy.is_some(),
            "fabric.environment_binding": environment_id,
        },
    }))
}

async fn cleanup_created_sandbox(
    client: &dyn Gateway,
    workspace: Option<&str>,
    name: &str,
    sandbox_id: &str,
    timeout_seconds: u64,
) -> Result<(), ProviderError> {
    let timeout = Duration::from_secs(GATEWAY_CALL_TIMEOUT_SECONDS);
    let current = bounded_gateway_call("get", timeout, client.get(workspace, name))
        .await
        .map_err(|_| unconfirmed_cleanup_error(name, sandbox_id))?;
    if current.id != sandbox_id {
        return Err(unconfirmed_cleanup_error(name, sandbox_id));
    }
    let deletion = bounded_gateway_call("delete", timeout, client.delete(workspace, name))
        .await
        .map_err(|_| unconfirmed_cleanup_error(name, sandbox_id))?;
    if matches!(
        deletion.outcome,
        DeletionOutcome::Accepted | DeletionOutcome::Completed
    ) && deletion.sandbox_id.as_deref() == Some(sandbox_id)
    {
        bounded_gateway_call(
            "wait_deleted",
            Duration::from_secs(timeout_seconds),
            client.wait_deleted(
                workspace,
                name,
                Duration::from_secs(timeout_seconds),
                Some(sandbox_id),
            ),
        )
        .await
        .map_err(|_| unconfirmed_cleanup_error(name, sandbox_id))?;
    } else {
        return Err(unconfirmed_cleanup_error(name, sandbox_id));
    }
    Ok(())
}

fn unconfirmed_cleanup_error(name: &str, sandbox_id: &str) -> ProviderError {
    ProviderError::contract(
        "cleanup_unconfirmed",
        format!("cleanup was not confirmed for sandbox `{name}` (id `{sandbox_id}`)"),
    )
}

async fn bounded_gateway_call<T>(
    operation: &str,
    timeout: Duration,
    call: impl Future<Output = Result<T, ProviderError>>,
) -> Result<T, ProviderError> {
    tokio::time::timeout(timeout, call).await.map_err(|_| {
        ProviderError::contract(
            "gateway_timeout",
            format!("OpenShell {operation} call exceeded its time limit"),
        )
    })?
}

async fn inspect_environment<F>(
    environment: EnvironmentHandle,
    factory: &F,
) -> Result<Value, ProviderError>
where
    F: GatewayFactory,
{
    let binding = SandboxBinding::from_environment(&environment)?;
    let call_timeout = Duration::from_secs(GATEWAY_CALL_TIMEOUT_SECONDS);
    let client = bounded_gateway_call(
        "connect",
        call_timeout,
        factory.connect(&binding.connection),
    )
    .await?;
    let sandbox = bounded_gateway_call(
        "get",
        call_timeout,
        client.get(binding.connection.workspace.as_deref(), &binding.name),
    )
    .await?;
    binding.verify(&sandbox)?;
    Ok(json!({
        "sandbox_id": sandbox.id,
        "sandbox_name": sandbox.name,
        "workspace": sandbox.workspace,
        "phase": sandbox.phase.as_str(),
        "resource_version": sandbox.resource_version,
        "exit_code": sandbox.exit_code,
    }))
}

#[allow(clippy::too_many_arguments)]
async fn exec_environment<F>(
    environment: EnvironmentHandle,
    command: Vec<String>,
    workdir: Option<String>,
    env: HashMap<String, String>,
    timeout_seconds: Option<u64>,
    stdin: Option<Vec<u8>>,
    factory: &F,
) -> Result<Value, ProviderError>
where
    F: GatewayFactory,
{
    if command.is_empty() || command.iter().any(String::is_empty) {
        return Err(ProviderError::contract(
            "invalid_exec",
            "exec command must contain only non-empty arguments",
        ));
    }
    let binding = SandboxBinding::from_environment(&environment)?;
    let timeout = timeout_seconds
        .or(binding.connection.exec_timeout_seconds)
        .unwrap_or(DEFAULT_EXEC_TIMEOUT_SECONDS);
    if !(1..=MAX_OPERATION_TIMEOUT_SECONDS).contains(&timeout) {
        return Err(ProviderError::contract(
            "invalid_exec",
            "exec timeout must be between 1 and 600 seconds",
        ));
    }
    let call_timeout = Duration::from_secs(GATEWAY_CALL_TIMEOUT_SECONDS);
    let client = bounded_gateway_call(
        "connect",
        call_timeout,
        factory.connect(&binding.connection),
    )
    .await?;
    let sandbox = bounded_gateway_call(
        "get",
        call_timeout,
        client.get(binding.connection.workspace.as_deref(), &binding.name),
    )
    .await?;
    binding.verify(&sandbox)?;
    let result = bounded_gateway_call(
        "exec",
        Duration::from_secs(timeout),
        client.exec(
            binding.connection.workspace.as_deref(),
            &binding.name,
            command,
            ExecRequest {
                workdir,
                environment: env,
                timeout: Duration::from_secs(timeout),
                stdin,
            },
        ),
    )
    .await?;
    if result.stdout.len().saturating_add(result.stderr.len()) > MAX_EXEC_OUTPUT_BYTES {
        return Err(ProviderError::contract(
            "exec_output_too_large",
            format!("buffered exec output exceeds the {MAX_EXEC_OUTPUT_BYTES}-byte limit"),
        ));
    }
    Ok(json!({
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }))
}

async fn release_environment<F>(
    environment: EnvironmentHandle,
    factory: &F,
) -> Result<Value, ProviderError>
where
    F: GatewayFactory,
{
    if environment.ownership != "fabric_owned" {
        return Ok(json!({"released": false, "detached": true}));
    }
    let binding = SandboxBinding::from_environment(&environment)?;
    let call_timeout = Duration::from_secs(GATEWAY_CALL_TIMEOUT_SECONDS);
    let client = bounded_gateway_call(
        "connect",
        call_timeout,
        factory.connect(&binding.connection),
    )
    .await?;
    match bounded_gateway_call(
        "get",
        call_timeout,
        client.get(binding.connection.workspace.as_deref(), &binding.name),
    )
    .await
    {
        Ok(sandbox) => binding.verify(&sandbox)?,
        Err(error) if error.code() == "not_found" => {
            return Ok(json!({"released": false, "detached": false}));
        }
        Err(error) => return Err(error),
    }
    let deletion = bounded_gateway_call(
        "delete",
        call_timeout,
        client.delete(binding.connection.workspace.as_deref(), &binding.name),
    )
    .await?;
    match deletion.outcome {
        DeletionOutcome::AlreadyAbsent => return Ok(json!({"released": false, "detached": false})),
        DeletionOutcome::Accepted | DeletionOutcome::Completed => {
            if deletion.sandbox_id.as_deref() != Some(binding.id.as_str()) {
                return Err(ProviderError::contract(
                    "sandbox_identity_changed",
                    "deletion did not identify the bound sandbox",
                ));
            }
        }
        _ => {
            return Err(ProviderError::contract(
                "deletion_unconfirmed",
                "OpenShell did not confirm a recognized deletion outcome",
            ));
        }
    }
    let delete_timeout = binding
        .connection
        .delete_timeout_seconds
        .unwrap_or(DEFAULT_DELETE_TIMEOUT_SECONDS);
    bounded_gateway_call(
        "wait_deleted",
        Duration::from_secs(delete_timeout),
        client.wait_deleted(
            binding.connection.workspace.as_deref(),
            &binding.name,
            Duration::from_secs(delete_timeout),
            Some(&binding.id),
        ),
    )
    .await?;
    Ok(json!({"released": true, "detached": false}))
}

fn validate_profile(
    environment: &EnvironmentPlan,
    expected_ownership: &str,
) -> Result<(), ProviderError> {
    if environment.provider != "openshell" {
        return Err(ProviderError::contract(
            "invalid_profile",
            "environment.provider must be `openshell`",
        ));
    }
    if environment.control_location != "in_env_control" {
        return Err(ProviderError::contract(
            "invalid_profile",
            "environment.control_location must be `in_env_control`",
        ));
    }
    if environment.ownership != expected_ownership {
        return Err(ProviderError::contract(
            "invalid_profile",
            format!("environment.ownership must be `{expected_ownership}`"),
        ));
    }
    if environment.workspace.is_none() {
        return Err(ProviderError::contract(
            "invalid_profile",
            "environment.workspace must be an absolute path inside the sandbox",
        ));
    }
    for (field, path) in [
        ("environment.workspace", environment.workspace.as_deref()),
        ("environment.artifacts", environment.artifacts.as_deref()),
    ] {
        if let Some(path) = path
            && (!path.starts_with('/')
                || path.contains('\\')
                || path.contains('\0')
                || path.split('/').any(|part| part == ".."))
        {
            return Err(ProviderError::contract(
                "invalid_profile",
                format!("{field} must be an absolute path inside the sandbox"),
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct ProviderRequest {
    protocol_version: String,
    request_id: String,
    #[serde(flatten)]
    operation: ProviderOperation,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
enum ProviderOperation {
    RuntimeCommand {
        environment: EnvironmentHandle,
        command: TaskCommand,
        timeout_seconds: u64,
    },
    CollectArtifacts {
        environment: EnvironmentHandle,
        paths: Vec<String>,
    },
    Prepare {
        environment_id: String,
        environment: EnvironmentPlan,
    },
    Attach {
        environment_id: String,
        environment: EnvironmentPlan,
        reference: EnvironmentReference,
    },
    Inspect {
        environment: EnvironmentHandle,
    },
    Exec {
        environment: EnvironmentHandle,
        command: Vec<String>,
        #[serde(default)]
        workdir: Option<String>,
        #[serde(default)]
        env: HashMap<String, String>,
        #[serde(default)]
        timeout_seconds: Option<u64>,
        #[serde(default)]
        stdin: Option<Vec<u8>>,
    },
    Release {
        environment: EnvironmentHandle,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
enum TaskCommand {
    Check {
        timeout_seconds: f64,
    },
    Start {
        agent: String,
        config: Value,
    },
    Invoke {
        agent: String,
        runtime_id: String,
        request: Value,
    },
    Stop {
        agent: String,
        runtime_id: String,
    },
}
impl TaskCommand {
    fn check_timeout(&self) -> Option<f64> {
        match self {
            Self::Check { timeout_seconds } => Some(*timeout_seconds),
            _ => None,
        }
    }

    fn arguments(self) -> Result<(&'static str, Vec<String>, Option<Value>), ProviderError> {
        let mut arguments = vec!["fabric-agent-ctl".to_string()];
        let (operation, agent, runtime_id, payload) = match self {
            Self::Check { timeout_seconds } => {
                if !timeout_seconds.is_finite() || timeout_seconds <= 0.0 {
                    return Err(ProviderError::contract(
                        "invalid_runtime_command",
                        "health timeout must be positive and finite",
                    ));
                }
                arguments.extend([
                    "check".into(),
                    "--timeout-seconds".into(),
                    timeout_seconds.to_string(),
                ]);
                return Ok(("check", arguments, None));
            }
            Self::Start { agent, config } => ("start", agent, None, Some(config)),
            Self::Invoke {
                agent,
                runtime_id,
                request,
            } => ("invoke", agent, Some(runtime_id), Some(request)),
            Self::Stop { agent, runtime_id } => ("stop", agent, Some(runtime_id), None),
        };
        if agent.trim().is_empty()
            || agent.contains('\0')
            || runtime_id
                .as_ref()
                .is_some_and(|value| value.trim().is_empty() || value.contains('\0'))
            || payload.as_ref().is_some_and(|value| !value.is_object())
        {
            return Err(ProviderError::contract(
                "invalid_runtime_command",
                "command routing or SDK payload is invalid",
            ));
        }
        arguments.extend([operation.to_string(), "--agent".to_string(), agent]);
        if let Some(runtime_id) = runtime_id {
            arguments.extend(["--runtime-id".to_string(), runtime_id]);
        }
        if payload.is_some() {
            arguments.extend([
                if operation == "start" {
                    "--config"
                } else {
                    "--request"
                }
                .to_string(),
                "-".to_string(),
            ]);
        }
        Ok((operation, arguments, payload))
    }
}
#[derive(Deserialize)]
struct TaskDiagnosticEnvelope {
    operation: String,
    status: String,
    error: TaskDiagnostic,
}
#[derive(Deserialize)]
struct TaskDiagnostic {
    code: String,
    stage: String,
    effects: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactResponse {
    files: Vec<CollectedArtifact>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CollectedArtifact {
    path: String,
    content: Vec<u8>,
}

#[derive(Debug, Deserialize)]
struct EnvironmentPlan {
    provider: String,
    control_location: String,
    ownership: String,
    workspace: Option<String>,
    artifacts: Option<String>,
    #[serde(default)]
    env: HashMap<String, String>,
    #[serde(default)]
    connection: Map<String, Value>,
    #[serde(default)]
    settings: Map<String, Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EnvironmentReference {
    provider: String,
    #[serde(default)]
    resource: Map<String, Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenShellResourceReference {
    sandbox_name: String,
    sandbox_id: String,
}

impl OpenShellResourceReference {
    fn from_map(resource: Map<String, Value>) -> Result<Self, ProviderError> {
        let reference: Self =
            serde_json::from_value(Value::Object(resource)).map_err(|_error| {
                ProviderError::contract("invalid_reference", "invalid OpenShell resource reference")
            })?;
        if reference.sandbox_name.trim().is_empty() || reference.sandbox_id.trim().is_empty() {
            return Err(ProviderError::contract(
                "invalid_reference",
                "sandbox_name and sandbox_id must not be blank",
            ));
        }
        Ok(reference)
    }
}

#[derive(Debug, Deserialize)]
struct EnvironmentHandle {
    environment_id: String,
    provider: String,
    ownership: String,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    artifacts: Option<String>,
    #[serde(default)]
    connection: Map<String, Value>,
    #[serde(default)]
    env: HashMap<String, String>,
}

#[derive(Debug, Serialize)]
struct ProviderResponse {
    protocol_version: &'static str,
    request_id: String,
    #[serde(flatten)]
    outcome: ProviderOutcome,
}

impl ProviderResponse {
    fn succeeded(request_id: &str, output: Value) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            request_id: request_id.to_string(),
            outcome: ProviderOutcome::Succeeded { output },
        }
    }

    fn failed(request_id: &str, code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            request_id: request_id.to_string(),
            outcome: ProviderOutcome::Failed {
                error: ProviderFailure {
                    code: code.into(),
                    message: message.into(),
                },
            },
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum ProviderOutcome {
    Succeeded { output: Value },
    Failed { error: ProviderFailure },
}

#[derive(Debug, Serialize)]
struct ProviderFailure {
    code: String,
    message: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenShellConnection {
    gateway: String,
    #[serde(default)]
    token_env: Option<String>,
    #[serde(default)]
    ca_cert_env: Option<String>,
    #[serde(default)]
    insecure_skip_verify: bool,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    sandbox_id: Option<String>,
    #[serde(default)]
    sandbox_name: Option<String>,
    #[serde(default)]
    delete_timeout_seconds: Option<u64>,
    #[serde(default)]
    exec_timeout_seconds: Option<u64>,
}

impl OpenShellConnection {
    fn from_map(map: &Map<String, Value>) -> Result<Self, ProviderError> {
        let connection: Self =
            serde_json::from_value(Value::Object(map.clone())).map_err(|_error| {
                ProviderError::contract(
                    "invalid_connection",
                    "invalid OpenShell connection settings",
                )
            })?;
        if connection.gateway.trim().is_empty() {
            return Err(ProviderError::contract(
                "invalid_connection",
                "connection.gateway must not be empty",
            ));
        }
        for (field, value) in [
            ("connection.token_env", connection.token_env.as_deref()),
            ("connection.ca_cert_env", connection.ca_cert_env.as_deref()),
            ("connection.workspace", connection.workspace.as_deref()),
        ] {
            if value.is_some_and(|value| value.trim().is_empty()) {
                return Err(ProviderError::contract(
                    "invalid_connection",
                    format!("{field} must not be blank"),
                ));
            }
        }
        for timeout in [
            connection.delete_timeout_seconds,
            connection.exec_timeout_seconds,
        ]
        .into_iter()
        .flatten()
        {
            if !(1..=MAX_OPERATION_TIMEOUT_SECONDS).contains(&timeout) {
                return Err(ProviderError::contract(
                    "invalid_connection",
                    "connection timeouts must be between 1 and 600 seconds",
                ));
            }
        }
        Ok(connection)
    }

    fn to_safe_map(&self) -> Map<String, Value> {
        let mut output = Map::from_iter([("gateway".to_string(), json!(self.gateway))]);
        if let Some(value) = &self.token_env {
            output.insert("token_env".to_string(), json!(value));
        }
        if let Some(value) = &self.ca_cert_env {
            output.insert("ca_cert_env".to_string(), json!(value));
        }
        if self.insecure_skip_verify {
            output.insert("insecure_skip_verify".to_string(), json!(true));
        }
        if let Some(value) = &self.workspace {
            output.insert("workspace".to_string(), json!(value));
        }
        output
    }

    fn client_config(&self) -> Result<ClientConfig, ProviderError> {
        if self.token_env.is_some()
            && (self.insecure_skip_verify || !self.gateway.starts_with("https://"))
        {
            return Err(ProviderError::contract(
                "invalid_connection",
                "connection.token_env requires an HTTPS gateway with certificate verification enabled",
            ));
        }
        let mut config = ClientConfig::new(&self.gateway);
        config.insecure_skip_verify = self.insecure_skip_verify;
        if let Some(name) = &self.ca_cert_env {
            config.ca_cert = Some(std::env::var(name).map_err(|_| {
                ProviderError::contract(
                    "credential_unavailable",
                    format!("connection.ca_cert_env references unset environment variable `{name}`"),
                )
            })?.into_bytes());
        }
        if let Some(name) = &self.token_env {
            let token = std::env::var(name).map_err(|_| {
                ProviderError::contract(
                    "credential_unavailable",
                    format!("connection.token_env references unset environment variable `{name}`"),
                )
            })?;
            config.auth = Some(AuthConfig::oidc(token));
        }
        Ok(config)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenShellSettings {
    image: String,
    sandbox_name: Option<String>,
    command: Vec<String>,
    #[serde(skip)]
    policy: Option<proto::SandboxPolicy>,
    #[serde(default)]
    providers: Vec<String>,
    #[serde(default = "default_ready_timeout_seconds")]
    ready_timeout_seconds: u64,
    #[serde(default = "default_delete_timeout_seconds")]
    delete_timeout_seconds: u64,
    #[serde(default = "default_exec_timeout_seconds")]
    exec_timeout_seconds: u64,
}

impl OpenShellSettings {
    fn from_map(map: Map<String, Value>) -> Result<Self, ProviderError> {
        let mut map = map;
        if let Some(paths) = map.remove("task_discovery_paths") {
            serde_json::from_value::<Vec<String>>(paths).map_err(|_error| {
                ProviderError::contract(
                    "invalid_settings",
                    "settings.task_discovery_paths must be a list of strings",
                )
            })?;
        }
        let policy = map
            .remove("policy_yaml")
            .map(|value| {
                let yaml = value.as_str().ok_or_else(|| {
                    ProviderError::contract(
                        "invalid_settings",
                        "settings.policy_yaml must be a YAML string",
                    )
                })?;
                openshell_policy::parse_sandbox_policy(yaml).map_err(|_error| {
                    ProviderError::contract("invalid_settings", "settings.policy_yaml is invalid")
                })
            })
            .transpose()?;
        let settings: Self = serde_json::from_value(Value::Object(map)).map_err(|_error| {
            ProviderError::contract("invalid_settings", "invalid OpenShell provider settings")
        })?;
        validate_pinned_image(&settings.image)?;
        if settings.command.is_empty() || settings.command.iter().any(String::is_empty) {
            return Err(ProviderError::contract(
                "invalid_settings",
                "settings.command must contain only non-empty arguments",
            ));
        }
        if settings.sandbox_name.as_deref().is_some_and(str::is_empty) {
            return Err(ProviderError::contract(
                "invalid_settings",
                "settings.sandbox_name must not be empty",
            ));
        }
        for (field, value) in [
            (
                "settings.ready_timeout_seconds",
                settings.ready_timeout_seconds,
            ),
            (
                "settings.delete_timeout_seconds",
                settings.delete_timeout_seconds,
            ),
            (
                "settings.exec_timeout_seconds",
                settings.exec_timeout_seconds,
            ),
        ] {
            if !(1..=MAX_OPERATION_TIMEOUT_SECONDS).contains(&value) {
                return Err(ProviderError::contract(
                    "invalid_settings",
                    format!("{field} must be between 1 and {MAX_OPERATION_TIMEOUT_SECONDS}"),
                ));
            }
        }
        Ok(Self { policy, ..settings })
    }
}

fn validate_pinned_image(image: &str) -> Result<(), ProviderError> {
    if let Some(digest) = image.strip_prefix("sha256:")
        && digest.len() == 64
        && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Ok(());
    }
    let Some((repository, digest)) = image.rsplit_once("@sha256:") else {
        return Err(ProviderError::contract(
            "invalid_settings",
            "settings.image must be pinned by an OCI `@sha256:` digest or local Docker `sha256:` image id",
        ));
    };
    if repository.is_empty()
        || digest.len() != 64
        || !digest.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(ProviderError::contract(
            "invalid_settings",
            "settings.image contains an invalid SHA-256 digest",
        ));
    }
    Ok(())
}

const fn default_ready_timeout_seconds() -> u64 {
    DEFAULT_READY_TIMEOUT_SECONDS
}

const fn default_delete_timeout_seconds() -> u64 {
    DEFAULT_DELETE_TIMEOUT_SECONDS
}

const fn default_exec_timeout_seconds() -> u64 {
    DEFAULT_EXEC_TIMEOUT_SECONDS
}

struct SandboxBinding {
    connection: OpenShellConnection,
    id: String,
    name: String,
}

impl SandboxBinding {
    fn from_environment(environment: &EnvironmentHandle) -> Result<Self, ProviderError> {
        if environment.provider != "openshell" {
            return Err(ProviderError::contract(
                "invalid_environment",
                "environment handle provider must be `openshell`",
            ));
        }
        if environment.environment_id.trim().is_empty()
            || environment.workspace.as_deref().is_none_or(|path| {
                !path.starts_with('/')
                    || path.contains('\\')
                    || path.split('/').any(|part| part == "..")
            })
            || environment.artifacts.as_deref().is_some_and(|path| {
                !path.starts_with('/')
                    || path.contains('\\')
                    || path.split('/').any(|part| part == "..")
            })
        {
            return Err(ProviderError::contract(
                "invalid_environment",
                "environment identity and task roots are invalid",
            ));
        }
        let connection = OpenShellConnection::from_map(&environment.connection)?;
        let id = connection.sandbox_id.clone().ok_or_else(|| {
            ProviderError::contract(
                "invalid_environment",
                "environment connection is missing sandbox_id",
            )
        })?;
        let name = connection.sandbox_name.clone().ok_or_else(|| {
            ProviderError::contract(
                "invalid_environment",
                "environment connection is missing sandbox_name",
            )
        })?;
        Ok(Self {
            connection,
            id,
            name,
        })
    }

    fn verify(&self, sandbox: &SandboxSnapshot) -> Result<(), ProviderError> {
        if sandbox.id != self.id
            || sandbox.workspace != self.connection.workspace.as_deref().unwrap_or("default")
        {
            return Err(ProviderError::contract(
                "sandbox_identity_changed",
                "sandbox name now resolves to a different OpenShell id",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GatewayStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unspecified,
}

struct GatewayHealth {
    status: GatewayStatus,
    version: String,
}

struct SandboxCreate {
    name: Option<String>,
    image: String,
    labels: HashMap<String, String>,
    environment: HashMap<String, String>,
    providers: Vec<String>,
    command: Vec<String>,
    policy: Option<proto::SandboxPolicy>,
}

struct SandboxSnapshot {
    id: String,
    name: String,
    workspace: String,
    labels: HashMap<String, String>,
    phase: ProviderSandboxPhase,
    resource_version: u64,
    exit_code: Option<i32>,
    image: Option<String>,
    command: Option<Vec<String>>,
    policy: Option<proto::SandboxPolicy>,
}

#[derive(Clone, Copy)]
enum ProviderSandboxPhase {
    Provisioning,
    Ready,
    Error,
    Deleting,
    Stopping,
    Stopped,
    Starting,
    Completed,
    Unknown,
}

impl ProviderSandboxPhase {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Provisioning => "provisioning",
            Self::Ready => "ready",
            Self::Error => "error",
            Self::Deleting => "deleting",
            Self::Stopping => "stopping",
            Self::Stopped => "stopped",
            Self::Starting => "starting",
            Self::Completed => "completed",
            Self::Unknown => "unknown",
        }
    }
}

struct ExecRequest {
    workdir: Option<String>,
    environment: HashMap<String, String>,
    timeout: Duration,
    stdin: Option<Vec<u8>>,
}

struct ExecResponse {
    exit_code: i32,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

#[async_trait]
trait GatewayFactory: Sync {
    async fn connect(
        &self,
        connection: &OpenShellConnection,
    ) -> Result<Box<dyn Gateway>, ProviderError>;
}

#[async_trait]
trait Gateway: Send + Sync {
    async fn health(&self) -> Result<GatewayHealth, ProviderError>;
    async fn create(
        &self,
        workspace: Option<&str>,
        request: SandboxCreate,
    ) -> Result<SandboxSnapshot, ProviderError>;
    async fn get(
        &self,
        workspace: Option<&str>,
        name: &str,
    ) -> Result<SandboxSnapshot, ProviderError>;
    async fn wait_ready(
        &self,
        workspace: Option<&str>,
        name: &str,
        timeout: Duration,
    ) -> Result<SandboxSnapshot, ProviderError>;
    async fn exec(
        &self,
        workspace: Option<&str>,
        name: &str,
        command: Vec<String>,
        request: ExecRequest,
    ) -> Result<ExecResponse, ProviderError>;
    async fn delete(
        &self,
        workspace: Option<&str>,
        name: &str,
    ) -> Result<DeletionResult, ProviderError>;
    async fn wait_deleted(
        &self,
        workspace: Option<&str>,
        name: &str,
        timeout: Duration,
        expected_sandbox_id: Option<&str>,
    ) -> Result<(), ProviderError>;
}

struct SdkGatewayFactory;

#[async_trait]
impl GatewayFactory for SdkGatewayFactory {
    async fn connect(
        &self,
        connection: &OpenShellConnection,
    ) -> Result<Box<dyn Gateway>, ProviderError> {
        let client = OpenShellClient::connect(connection.client_config()?)
            .await
            .map_err(map_sdk_error)?;
        Ok(Box::new(SdkGateway { client }))
    }
}

struct SdkGateway {
    client: OpenShellClient,
}

#[async_trait]
impl Gateway for SdkGateway {
    async fn health(&self) -> Result<GatewayHealth, ProviderError> {
        let health = self.client.health().await.map_err(map_sdk_error)?;
        let status = match health.status {
            ServiceStatus::Healthy => GatewayStatus::Healthy,
            ServiceStatus::Degraded => GatewayStatus::Degraded,
            ServiceStatus::Unhealthy => GatewayStatus::Unhealthy,
            _ => GatewayStatus::Unspecified,
        };
        Ok(GatewayHealth {
            status,
            version: health.version,
        })
    }

    async fn create(
        &self,
        workspace: Option<&str>,
        request: SandboxCreate,
    ) -> Result<SandboxSnapshot, ProviderError> {
        let request = proto::CreateSandboxRequest {
            spec: Some(proto::SandboxSpec {
                environment: request.environment,
                template: Some(proto::SandboxTemplate {
                    image: request.image,
                    ..proto::SandboxTemplate::default()
                }),
                policy: request.policy,
                providers: request.providers,
                command: request.command,
                tty: false,
                ..proto::SandboxSpec::default()
            }),
            name: request.name.unwrap_or_default(),
            labels: request.labels,
            workspace_scope: Some(proto::workspace_selector(workspace.unwrap_or("default"))),
            ..proto::CreateSandboxRequest::default()
        };
        let response = self
            .client
            .raw_grpc()
            .create_sandbox(request)
            .await
            .map_err(|_error| {
                ProviderError::contract(grpc_error_code(_error.code()), "OpenShell create failed")
            })?
            .into_inner();
        let sandbox = response.sandbox.ok_or_else(|| {
            ProviderError::contract(
                "sdk_protocol",
                "OpenShell create response did not contain a sandbox",
            )
        })?;
        Ok(raw_sandbox_snapshot(sandbox))
    }

    async fn get(
        &self,
        workspace: Option<&str>,
        name: &str,
    ) -> Result<SandboxSnapshot, ProviderError> {
        let response = self
            .client
            .raw_grpc()
            .get_sandbox(proto::GetSandboxRequest {
                name: name.to_string(),
                workspace_scope: Some(proto::workspace_selector(workspace.unwrap_or("default"))),
            })
            .await
            .map_err(|_error| {
                ProviderError::contract(grpc_error_code(_error.code()), "OpenShell get failed")
            })?
            .into_inner();
        let sandbox = response.sandbox.ok_or_else(|| {
            ProviderError::contract(
                "sdk_protocol",
                "OpenShell get response did not contain a sandbox",
            )
        })?;
        Ok(raw_sandbox_snapshot(sandbox))
    }

    async fn wait_ready(
        &self,
        workspace: Option<&str>,
        name: &str,
        timeout: Duration,
    ) -> Result<SandboxSnapshot, ProviderError> {
        let sandbox = match workspace {
            Some(workspace) => {
                self.client
                    .workspace(workspace)
                    .wait_ready(name, timeout)
                    .await
            }
            None => self.client.wait_ready(name, timeout).await,
        }
        .map_err(map_sdk_error)?;
        Ok(sandbox_snapshot(sandbox))
    }

    async fn exec(
        &self,
        workspace: Option<&str>,
        name: &str,
        command: Vec<String>,
        request: ExecRequest,
    ) -> Result<ExecResponse, ProviderError> {
        let options = ExecOptions {
            workdir: request.workdir,
            environment: request.environment,
            timeout: Some(request.timeout),
            stdin: request.stdin,
            no_login_shell: true,
        };
        let result = match workspace {
            Some(workspace) => {
                self.client
                    .workspace(workspace)
                    .exec(name, &command, options)
                    .await
            }
            None => self.client.exec(name, &command, options).await,
        }
        .map_err(map_sdk_error)?;
        Ok(ExecResponse {
            exit_code: result.exit_code,
            stdout: result.stdout,
            stderr: result.stderr,
        })
    }

    async fn delete(
        &self,
        workspace: Option<&str>,
        name: &str,
    ) -> Result<DeletionResult, ProviderError> {
        match workspace {
            Some(workspace) => {
                self.client
                    .workspace(workspace)
                    .delete_sandbox(
                        name,
                        DeleteOptions {
                            allow_missing: true,
                        },
                    )
                    .await
            }
            None => {
                self.client
                    .delete_sandbox(
                        name,
                        DeleteOptions {
                            allow_missing: true,
                        },
                    )
                    .await
            }
        }
        .map_err(map_sdk_error)
    }

    async fn wait_deleted(
        &self,
        workspace: Option<&str>,
        name: &str,
        timeout: Duration,
        expected_sandbox_id: Option<&str>,
    ) -> Result<(), ProviderError> {
        match workspace {
            Some(workspace) => {
                self.client
                    .workspace(workspace)
                    .wait_deleted(name, timeout, expected_sandbox_id)
                    .await
            }
            None => {
                self.client
                    .wait_deleted(name, timeout, expected_sandbox_id)
                    .await
            }
        }
        .map_err(map_sdk_error)
    }
}

fn sandbox_snapshot(sandbox: SandboxRef) -> SandboxSnapshot {
    SandboxSnapshot {
        id: sandbox.id,
        name: sandbox.name,
        workspace: sandbox.workspace,
        labels: HashMap::new(),
        phase: match sandbox.phase {
            SandboxPhase::Provisioning => ProviderSandboxPhase::Provisioning,
            SandboxPhase::Ready => ProviderSandboxPhase::Ready,
            SandboxPhase::Error => ProviderSandboxPhase::Error,
            SandboxPhase::Deleting => ProviderSandboxPhase::Deleting,
            SandboxPhase::Stopping => ProviderSandboxPhase::Stopping,
            SandboxPhase::Stopped => ProviderSandboxPhase::Stopped,
            SandboxPhase::Starting => ProviderSandboxPhase::Starting,
            SandboxPhase::Completed => ProviderSandboxPhase::Completed,
            _ => ProviderSandboxPhase::Unknown,
        },
        resource_version: sandbox.resource_version,
        exit_code: sandbox.exit_code,
        image: None,
        command: None,
        policy: None,
    }
}

fn raw_sandbox_snapshot(sandbox: proto::Sandbox) -> SandboxSnapshot {
    let phase =
        proto::SandboxPhase::try_from(sandbox.phase()).unwrap_or(proto::SandboxPhase::Unspecified);
    let exit_code = sandbox.status.as_ref().and_then(|status| status.exit_code);
    let image = sandbox
        .spec
        .as_ref()
        .and_then(|spec| spec.template.as_ref())
        .map(|template| template.image.clone());
    let command = sandbox.spec.as_ref().map(|spec| spec.command.clone());
    let policy = sandbox.spec.as_ref().and_then(|spec| spec.policy.clone());
    let metadata = sandbox.metadata.unwrap_or_default();
    SandboxSnapshot {
        id: metadata.id,
        name: metadata.name,
        workspace: metadata.workspace,
        labels: metadata.labels,
        phase: match phase {
            proto::SandboxPhase::Provisioning => ProviderSandboxPhase::Provisioning,
            proto::SandboxPhase::Ready => ProviderSandboxPhase::Ready,
            proto::SandboxPhase::Error => ProviderSandboxPhase::Error,
            proto::SandboxPhase::Deleting => ProviderSandboxPhase::Deleting,
            proto::SandboxPhase::Stopping => ProviderSandboxPhase::Stopping,
            proto::SandboxPhase::Stopped => ProviderSandboxPhase::Stopped,
            proto::SandboxPhase::Starting => ProviderSandboxPhase::Starting,
            proto::SandboxPhase::Completed => ProviderSandboxPhase::Completed,
            _ => ProviderSandboxPhase::Unknown,
        },
        resource_version: metadata.resource_version,
        exit_code,
        image,
        command,
        policy,
    }
}

fn map_sdk_error(error: SdkError) -> ProviderError {
    ProviderError::contract(error.code(), "OpenShell SDK operation failed")
}

fn grpc_error_code(code: tonic::Code) -> &'static str {
    match code {
        tonic::Code::NotFound => "not_found",
        tonic::Code::AlreadyExists => "already_exists",
        tonic::Code::Unauthenticated | tonic::Code::PermissionDenied => "auth",
        _ => "rpc",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_check_maps_to_the_named_health_command() {
        let command: TaskCommand = serde_json::from_value(json!({
            "operation": "check", "timeout_seconds": 0.25
        }))
        .unwrap();
        let (operation, arguments, input) = command.arguments().unwrap();
        assert_eq!(operation, "check");
        assert_eq!(
            arguments,
            ["fabric-agent-ctl", "check", "--timeout-seconds", "0.25"]
        );
        assert!(input.is_none());
    }

    #[test]
    fn failed_health_evidence_is_returned_to_the_sdk() {
        let report = json!({
            "schema_version": "fabric.check/v1alpha1", "scope": "server",
            "runtime_id": null, "requested_level": null,
            "outcome": "failed", "reason_code": "server_stopping",
            "completed_at": "2026-10-19T12:00:00Z", "elapsed_ms": 1,
            "server": {"state": "stopping"}, "runtime": null,
            "checks": [{"name": "server", "outcome": "failed",
                "reason_code": "server_stopping", "source": "fabric.runtime_server",
                "observed_at": "2026-10-19T12:00:00Z"}]
        });
        let response = json!({"operation": "check", "status": "failed", "changed": false,
            "result": report, "error": {"code": "server_stopping", "stage": "check",
                "effects": "none", "message": "PRIVATE"}});
        let decoded =
            decode_sdk_command_response("check", 1, response.to_string().as_bytes()).unwrap();
        assert_eq!(decoded, json!({"status": "returned", "value": report}));
        assert!(!decoded.to_string().contains("PRIVATE"));
    }
    use std::sync::{Arc, Mutex};

    #[test]
    fn generated_sandbox_names_fit_the_openshell_gateway_contract() {
        let id = "nemo-fabric-environment-1760559084-12345-678";
        let name = generated_sandbox_name(id);
        assert_eq!(name.len(), 19);
        assert!(name.starts_with("nf-"));
        assert!(
            name.bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        );
        assert_eq!(name, generated_sandbox_name(id));
        assert_ne!(name, generated_sandbox_name("another-environment-id"));
    }

    #[tokio::test]
    async fn bounded_gateway_call_reports_timeouts_without_sdk_details() {
        let error = bounded_gateway_call(
            "create",
            Duration::ZERO,
            std::future::pending::<Result<(), ProviderError>>(),
        )
        .await
        .expect_err("zero deadline must time out");

        assert_eq!(error.code(), "gateway_timeout");
        assert!(error.to_string().contains("create"));
    }

    #[test]
    fn settings_require_a_digest_pinned_image() {
        let error = OpenShellSettings::from_map(Map::from_iter([
            ("image".to_string(), json!("example/agent-runtime:latest")),
            ("command".to_string(), json!(["fabric-agent-ctl", "serve"])),
        ]))
        .expect_err("mutable image tag must fail");

        assert_eq!(error.code(), "invalid_settings");
        assert!(error.to_string().contains("sha256:"));
    }

    #[test]
    fn settings_accept_an_immutable_local_docker_image_id() {
        OpenShellSettings::from_map(Map::from_iter([
            (
                "image".to_string(),
                json!(format!("sha256:{}", "a".repeat(64))),
            ),
            ("command".to_string(), json!(["fabric-agent-ctl", "serve"])),
        ]))
        .expect("immutable local image id");
    }

    #[test]
    fn runtime_command_unwraps_successful_sdk_envelopes() {
        let handle = json!({"runtime_id": "runtime-1", "agent_name": "echo"});
        let start = json!({
            "operation": "start",
            "status": "succeeded",
            "changed": true,
            "result": handle,
            "error": null
        });
        let decoded = decode_sdk_command_response("start", 0, start.to_string().as_bytes())
            .expect("start envelope");
        assert_eq!(decoded, json!({"status": "returned", "value": handle}));

        let invoke_result = json!({"status": "failed", "output": null});
        let invoke = json!({
            "operation": "invoke",
            "status": "succeeded",
            "changed": null,
            "result": invoke_result,
            "error": null
        });
        let decoded = decode_sdk_command_response("invoke", 0, invoke.to_string().as_bytes())
            .expect("invoke envelope");
        assert_eq!(
            decoded,
            json!({"status": "returned", "value": invoke_result})
        );

        let stop = json!({
            "operation": "stop",
            "status": "succeeded",
            "changed": true,
            "result": null,
            "error": null
        });
        let decoded = decode_sdk_command_response("stop", 0, stop.to_string().as_bytes())
            .expect("stop envelope");
        assert_eq!(decoded, json!({"status": "returned", "value": null}));
    }

    #[test]
    fn runtime_command_preserves_known_sdk_rejections() {
        let rejected = json!({
            "operation": "start",
            "status": "failed",
            "changed": false,
            "result": null,
            "error": {
                "code": "validation_failed",
                "stage": "validation",
                "message": "configuration rejected",
                "effects": "none"
            }
        });
        let decoded = decode_sdk_command_response("start", 1, rejected.to_string().as_bytes())
            .expect("rejection envelope");
        assert_eq!(
            decoded,
            json!({
                "status": "rejected",
                "error": {
                    "code": "validation_failed",
                    "stage": "validation",
                    "effects": "none"
                }
            })
        );
    }

    #[test]
    fn runtime_command_rejects_malformed_sdk_envelopes() {
        let raw_handle = json!({"runtime_id": "runtime-1"});
        assert_eq!(
            decode_sdk_command_response("start", 0, raw_handle.to_string().as_bytes())
                .expect_err("raw handle is not a command response")
                .code(),
            "invalid_response"
        );
        let bad_exit = json!({
            "operation": "start",
            "status": "succeeded",
            "changed": true,
            "result": {"runtime_id": "runtime-1"},
            "error": null
        });
        assert_eq!(
            decode_sdk_command_response("start", 1, bad_exit.to_string().as_bytes())
                .expect_err("successful envelope with failed exit")
                .code(),
            "invalid_response"
        );
    }

    #[test]
    fn settings_accept_task_discovery_paths_for_host_projection() {
        OpenShellSettings::from_map(Map::from_iter([
            (
                "image".to_string(),
                json!(format!("sha256:{}", "a".repeat(64))),
            ),
            ("command".to_string(), json!(["fabric-agent-ctl", "serve"])),
            (
                "task_discovery_paths".to_string(),
                json!(["/opt/fabric/agents/echo"]),
            ),
        ]))
        .expect("host-only task discovery paths are not OpenShell SDK settings");

        let error = OpenShellSettings::from_map(Map::from_iter([
            (
                "image".to_string(),
                json!(format!("sha256:{}", "a".repeat(64))),
            ),
            ("command".to_string(), json!(["fabric-agent-ctl", "serve"])),
            ("task_discovery_paths".to_string(), json!("/opt/echo")),
        ]))
        .expect_err("task discovery paths must be a list");
        assert_eq!(error.code(), "invalid_settings");
    }

    #[test]
    fn settings_parse_a_creation_time_policy() {
        let settings = OpenShellSettings::from_map(Map::from_iter([
            (
                "image".to_string(),
                json!(format!("example/agent-runtime@sha256:{}", "a".repeat(64))),
            ),
            ("command".to_string(), json!(["fabric-agent-ctl", "serve"])),
            (
                "policy_yaml".to_string(),
                json!("version: 1\nfilesystem_policy:\n  include_workdir: true\n"),
            ),
        ]))
        .expect("valid settings");

        assert_eq!(settings.policy.expect("policy").version, 1);
    }

    #[test]
    fn openshell_profile_requires_a_sandbox_workspace() {
        let mut environment = environment_plan_fixture();
        environment.workspace = None;

        let error =
            validate_profile(&environment, "fabric_owned").expect_err("workspace must be explicit");

        assert_eq!(error.code(), "invalid_profile");
        assert!(error.to_string().contains("environment.workspace"));
    }

    #[test]
    fn connection_rejects_literal_unknown_credential_fields() {
        let error = OpenShellConnection::from_map(&Map::from_iter([
            ("gateway".to_string(), json!("https://gateway.example")),
            ("token".to_string(), json!("literal-secret")),
        ]))
        .expect_err("literal token field must fail closed");

        assert_eq!(error.code(), "invalid_connection");
        assert!(!error.to_string().contains("literal-secret"));
    }

    #[test]
    fn safe_connection_projection_keeps_references_not_credentials() {
        let connection = OpenShellConnection::from_map(&Map::from_iter([
            ("gateway".to_string(), json!("https://gateway.example")),
            ("token_env".to_string(), json!("OPEN_SHELL_TOKEN")),
            ("workspace".to_string(), json!("fabric-demo")),
        ]))
        .expect("valid connection");

        let safe = connection.to_safe_map();

        assert_eq!(safe["token_env"], json!("OPEN_SHELL_TOKEN"));
        assert!(!safe.contains_key("token"));
    }

    #[test]
    fn credentialed_connections_require_verified_https() {
        for (gateway, insecure_skip_verify) in [
            ("http://gateway.example", false),
            ("https://gateway.example", true),
        ] {
            let connection = OpenShellConnection::from_map(&Map::from_iter([
                ("gateway".to_string(), json!(gateway)),
                ("token_env".to_string(), json!("OPEN_SHELL_TOKEN")),
                (
                    "insecure_skip_verify".to_string(),
                    json!(insecure_skip_verify),
                ),
            ]))
            .expect("connection shape");

            let Err(error) = connection.client_config() else {
                panic!("credential transport must fail closed");
            };
            assert_eq!(error.code(), "invalid_connection");
            assert!(error.to_string().contains("HTTPS"));
        }
    }

    #[test]
    fn grpc_statuses_map_to_stable_provider_codes() {
        assert_eq!(grpc_error_code(tonic::Code::NotFound), "not_found");
        assert_eq!(
            grpc_error_code(tonic::Code::AlreadyExists),
            "already_exists"
        );
        assert_eq!(grpc_error_code(tonic::Code::Unauthenticated), "auth");
        assert_eq!(grpc_error_code(tonic::Code::PermissionDenied), "auth");
        assert_eq!(grpc_error_code(tonic::Code::Unavailable), "rpc");
    }

    #[tokio::test]
    async fn malformed_request_returns_a_correlated_protocol_failure() {
        let input = b"not-json\n".as_slice();
        let mut output = Vec::new();

        serve(input, &mut output, &NeverConnect)
            .await
            .expect("serve request");

        let response: Value = serde_json::from_slice(&output).expect("response JSON");
        assert_eq!(response["protocol_version"], PROTOCOL_VERSION);
        assert_eq!(response["request_id"], "unknown");
        assert_eq!(response["status"], "failed");
        assert_eq!(response["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn stdio_protocol_handles_multiple_requests_on_one_channel() {
        let input = b"not-json\nstill-not-json\n".as_slice();
        let mut output = Vec::new();

        serve(input, &mut output, &NeverConnect)
            .await
            .expect("serve requests");

        let responses = output
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice::<Value>(line).expect("response JSON"))
            .collect::<Vec<_>>();
        assert_eq!(responses.len(), 2);
        assert!(
            responses
                .iter()
                .all(|response| response["error"]["code"] == "invalid_request")
        );
    }

    #[tokio::test]
    async fn oversized_request_is_drained_before_the_next_frame() {
        let mut input = vec![b'x'; MAX_REQUEST_BYTES + 1];
        input.extend_from_slice(b"\nnot-json\n");
        let mut output = Vec::new();

        serve(input.as_slice(), &mut output, &NeverConnect)
            .await
            .expect("serve requests");

        let responses = output
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice::<Value>(line).expect("response JSON"))
            .collect::<Vec<_>>();
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0]["error"]["code"], "request_too_large");
        assert_eq!(responses[1]["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn prepare_runs_health_create_and_readiness_through_the_sdk_boundary() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let factory = MockFactory {
            calls: Arc::clone(&calls),
            ready_id: "sandbox-id-1",
            deletion: None,
            execution: None,
        };
        let environment = EnvironmentPlan {
            provider: "openshell".to_string(),
            control_location: "in_env_control".to_string(),
            ownership: "fabric_owned".to_string(),
            workspace: Some("/sandbox".to_string()),
            artifacts: Some("/sandbox/artifacts".to_string()),
            env: HashMap::from([("FABRIC_VISIBLE".to_string(), "yes".to_string())]),
            connection: Map::from_iter([
                ("gateway".to_string(), json!("https://gateway.example")),
                ("workspace".to_string(), json!("fabric-demo")),
            ]),
            settings: Map::from_iter([
                (
                    "image".to_string(),
                    json!(format!("example/agent-runtime@sha256:{}", "a".repeat(64))),
                ),
                ("command".to_string(), json!(["fabric-agent-ctl", "serve"])),
                (
                    "policy_yaml".to_string(),
                    json!("version: 1\nfilesystem_policy:\n  include_workdir: true\n"),
                ),
            ]),
        };

        let output = prepare_environment("environment-1", environment, &factory)
            .await
            .expect("prepare environment");

        assert_eq!(output["workspace"], "/sandbox");
        assert_eq!(output["artifacts"], "/sandbox/artifacts");
        assert_eq!(output["connection"]["sandbox_id"], "sandbox-id-1");
        assert_eq!(output["connection"]["sandbox_name"], "fabric-sandbox-1");
        assert_eq!(output["metadata"]["openshell.sandbox_phase"], "ready");
        assert_eq!(output["metadata"]["openshell.sandbox_resource_version"], 1);
        assert_eq!(output["metadata"]["openshell.policy_attached"], true);
        assert_eq!(
            *calls.lock().expect("calls"),
            [
                "connect",
                "health",
                "create:fabric-demo:environment-1",
                "wait_ready:fabric-demo:fabric-sandbox-1",
            ]
        );
    }

    #[tokio::test]
    async fn attach_verifies_a_caller_owned_sandbox_without_creating_it() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let factory = MockFactory {
            calls: Arc::clone(&calls),
            ready_id: "sandbox-id-1",
            deletion: None,
            execution: None,
        };
        let mut environment = environment_plan_fixture();
        environment.ownership = "caller_owned".to_string();
        let reference = EnvironmentReference {
            provider: "openshell".to_string(),
            resource: Map::from_iter([
                ("sandbox_name".to_string(), json!("fabric-sandbox-1")),
                ("sandbox_id".to_string(), json!("sandbox-id-1")),
            ]),
        };

        let output = attach_environment("environment-attach", environment, reference, &factory)
            .await
            .expect("attach environment");

        assert_eq!(output["connection"]["sandbox_id"], "sandbox-id-1");
        assert_eq!(output["connection"]["sandbox_name"], "fabric-sandbox-1");
        assert_eq!(output["metadata"]["openshell.policy_attached"], true);
        assert_eq!(
            output["metadata"]["fabric.environment_binding"],
            "environment-attach"
        );
        assert_eq!(
            *calls.lock().expect("calls"),
            ["connect", "health", "get:fabric-demo:fabric-sandbox-1",]
        );
    }

    #[tokio::test]
    async fn attach_rejects_a_sandbox_name_that_resolves_to_another_identity() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let factory = MockFactory {
            calls: Arc::clone(&calls),
            ready_id: "different-sandbox-id",
            deletion: None,
            execution: None,
        };
        let mut environment = environment_plan_fixture();
        environment.ownership = "caller_owned".to_string();
        let reference = EnvironmentReference {
            provider: "openshell".to_string(),
            resource: Map::from_iter([
                ("sandbox_name".to_string(), json!("fabric-sandbox-1")),
                ("sandbox_id".to_string(), json!("sandbox-id-1")),
            ]),
        };

        let error = attach_environment("environment-attach", environment, reference, &factory)
            .await
            .expect_err("changed identity must fail closed");

        assert_eq!(error.code(), "sandbox_identity_changed");
        assert_eq!(
            *calls.lock().expect("calls"),
            ["connect", "health", "get:fabric-demo:fabric-sandbox-1",]
        );
    }

    #[tokio::test]
    async fn inspect_exec_and_release_preserve_the_bound_sandbox_identity() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let factory = MockFactory {
            calls: Arc::clone(&calls),
            ready_id: "sandbox-id-1",
            deletion: None,
            execution: None,
        };
        let environment = environment_handle_fixture();

        let inspection = inspect_environment(environment_handle_fixture(), &factory)
            .await
            .expect("inspect environment");
        let execution = exec_environment(
            environment_handle_fixture(),
            vec!["python".to_string(), "--version".to_string()],
            Some("/sandbox".to_string()),
            HashMap::new(),
            Some(12),
            None,
            &factory,
        )
        .await
        .expect("exec environment control");
        let release = release_environment(environment, &factory)
            .await
            .expect("release environment");

        assert_eq!(inspection["sandbox_id"], "sandbox-id-1");
        assert_eq!(inspection["phase"], "ready");
        assert_eq!(execution["exit_code"], 0);
        assert_eq!(execution["stdout"], json!([111, 107]));
        assert_eq!(release, json!({"released": true, "detached": false}));
        assert_eq!(
            *calls.lock().expect("calls"),
            [
                "connect",
                "get:fabric-demo:fabric-sandbox-1",
                "connect",
                "get:fabric-demo:fabric-sandbox-1",
                "exec:fabric-demo:fabric-sandbox-1:python --version:12",
                "connect",
                "get:fabric-demo:fabric-sandbox-1",
                "delete:fabric-demo:fabric-sandbox-1",
                "wait_deleted:fabric-demo:fabric-sandbox-1:30",
            ]
        );
    }

    #[tokio::test]
    async fn release_detaches_a_caller_owned_sandbox_without_connecting() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let factory = MockFactory {
            calls: Arc::clone(&calls),
            ready_id: "sandbox-id-1",
            deletion: None,
            execution: None,
        };
        let mut environment = environment_handle_fixture();
        environment.ownership = "caller_owned".to_string();

        let release = release_environment(environment, &factory)
            .await
            .expect("detach environment");

        assert_eq!(release, json!({"released": false, "detached": true}));
        assert!(calls.lock().expect("calls").is_empty());
    }

    #[tokio::test]
    async fn prepare_preserves_a_replacement_when_readiness_returns_a_different_identity() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let factory = MockFactory {
            calls: Arc::clone(&calls),
            ready_id: "replaced-sandbox-id",
            deletion: None,
            execution: None,
        };
        let environment = EnvironmentPlan {
            provider: "openshell".to_string(),
            control_location: "in_env_control".to_string(),
            ownership: "fabric_owned".to_string(),
            workspace: Some("/sandbox".to_string()),
            artifacts: Some("/sandbox/artifacts".to_string()),
            env: HashMap::from([("FABRIC_VISIBLE".to_string(), "yes".to_string())]),
            connection: Map::from_iter([
                ("gateway".to_string(), json!("https://gateway.example")),
                ("workspace".to_string(), json!("fabric-demo")),
            ]),
            settings: Map::from_iter([
                (
                    "image".to_string(),
                    json!(format!("example/agent-runtime@sha256:{}", "a".repeat(64))),
                ),
                ("command".to_string(), json!(["fabric-agent-ctl", "serve"])),
            ]),
        };

        let error = prepare_environment("environment-1", environment, &factory)
            .await
            .expect_err("identity replacement must fail");

        assert_eq!(error.code(), "cleanup_unconfirmed");
        assert!(error.to_string().contains("sandbox-id-1"));
        assert!(error.to_string().contains("fabric-sandbox-1"));
        assert!(
            !calls
                .lock()
                .expect("calls")
                .iter()
                .any(|call| call.starts_with("delete:") || call.starts_with("wait_deleted:"))
        );
    }

    #[tokio::test]
    async fn failed_prepare_cleanup_reports_the_sandbox_identity() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let client = MockGateway {
            calls,
            ready_id: "sandbox-id-1",
            deletion: Some(DeletionResult {
                outcome: DeletionOutcome::Unspecified,
                sandbox_id: Some("sandbox-id-1".to_string()),
            }),
            execution: None,
        };

        let error = cleanup_created_sandbox(
            &client,
            Some("fabric-demo"),
            "fabric-sandbox-1",
            "sandbox-id-1",
            30,
        )
        .await
        .expect_err("unrecognized deletion must be reported");

        assert_eq!(error.code(), "cleanup_unconfirmed");
        assert!(error.to_string().contains("fabric-sandbox-1"));
        assert!(error.to_string().contains("sandbox-id-1"));
    }

    #[tokio::test]
    async fn caller_owned_attachment_uses_the_explicit_default_workspace() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let factory = MockFactory {
            calls: Arc::clone(&calls),
            ready_id: "sandbox-id-1",
            deletion: None,
            execution: None,
        };
        let mut environment = environment_plan_fixture();
        environment.ownership = "caller_owned".to_string();
        environment.connection.remove("workspace");
        let reference = EnvironmentReference {
            provider: "openshell".to_string(),
            resource: Map::from_iter([
                ("sandbox_name".to_string(), json!("fabric-sandbox-1")),
                ("sandbox_id".to_string(), json!("sandbox-id-1")),
            ]),
        };
        let output = attach_environment("environment-default", environment, reference, &factory)
            .await
            .expect("attach default workspace");
        assert_eq!(output["connection"]["sandbox_id"], "sandbox-id-1");
        assert!(
            calls
                .lock()
                .expect("calls")
                .contains(&"get:default:fabric-sandbox-1".to_string())
        );
    }

    #[tokio::test]
    async fn deletion_outcomes_wait_for_the_original_identity() {
        for outcome in [
            DeletionOutcome::Accepted,
            DeletionOutcome::Completed,
            DeletionOutcome::AlreadyAbsent,
            DeletionOutcome::Unspecified,
            DeletionOutcome::Unknown(99),
        ] {
            let calls = Arc::new(Mutex::new(Vec::new()));
            let factory = MockFactory {
                calls: Arc::clone(&calls),
                ready_id: "sandbox-id-1",
                deletion: Some(DeletionResult {
                    outcome,
                    sandbox_id: if outcome == DeletionOutcome::AlreadyAbsent {
                        None
                    } else {
                        Some("sandbox-id-1".to_string())
                    },
                }),
                execution: None,
            };
            let result = release_environment(environment_handle_fixture(), &factory).await;
            match outcome {
                DeletionOutcome::Accepted | DeletionOutcome::Completed => {
                    assert_eq!(
                        result.expect("confirmed deletion"),
                        json!({"released": true, "detached": false})
                    );
                    assert!(
                        calls
                            .lock()
                            .expect("calls")
                            .iter()
                            .any(|call| call.starts_with("wait_deleted:"))
                    );
                }
                DeletionOutcome::AlreadyAbsent => assert_eq!(
                    result.expect("already absent"),
                    json!({"released": false, "detached": false})
                ),
                _ => assert_eq!(
                    result.expect_err("unrecognized acknowledgement").code(),
                    "deletion_unconfirmed"
                ),
            }
        }
    }

    #[tokio::test]
    async fn deletion_cannot_establish_success_for_another_identity() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let factory = MockFactory {
            calls: Arc::clone(&calls),
            ready_id: "sandbox-id-1",
            deletion: Some(DeletionResult {
                outcome: DeletionOutcome::Accepted,
                sandbox_id: Some("replacement".to_string()),
            }),
            execution: None,
        };
        let error = release_environment(environment_handle_fixture(), &factory)
            .await
            .expect_err("wrong deletion identity");
        assert_eq!(error.code(), "sandbox_identity_changed");
        assert!(
            !calls
                .lock()
                .expect("calls")
                .iter()
                .any(|call| call.starts_with("wait_deleted:"))
        );
    }

    #[tokio::test]
    async fn release_preserves_a_same_name_replacement() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let factory = MockFactory {
            calls: Arc::clone(&calls),
            ready_id: "replacement",
            deletion: None,
            execution: None,
        };
        let error = release_environment(environment_handle_fixture(), &factory)
            .await
            .expect_err("replacement must survive");
        assert_eq!(error.code(), "sandbox_identity_changed");
        assert!(
            !calls
                .lock()
                .expect("calls")
                .iter()
                .any(|call| call.starts_with("delete:"))
        );
    }

    #[tokio::test]
    async fn invalid_timeout_ranges_are_rejected_before_gateway_connection() {
        for field in [
            "ready_timeout_seconds",
            "delete_timeout_seconds",
            "exec_timeout_seconds",
        ] {
            for value in [0, 601, u64::MAX] {
                let mut environment = environment_plan_fixture();
                environment.settings.insert(field.to_string(), json!(value));
                let error =
                    prepare_environment("environment-invalid-timeout", environment, &NeverConnect)
                        .await
                        .expect_err("invalid timeout");
                assert_eq!(error.code(), "invalid_settings");
            }
        }
    }

    #[tokio::test]
    async fn sdk_failed_result_is_returned_inside_a_successful_command_envelope() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let result = json!({"runtime_id": "runtime-task", "request_id": "request-host", "status": "failed", "output": null, "artifacts": {"artifacts": []}});
        let envelope = json!({
            "operation": "invoke",
            "status": "succeeded",
            "changed": null,
            "result": result,
            "error": null
        });
        let factory = MockFactory {
            calls: Arc::clone(&calls),
            ready_id: "sandbox-id-1",
            deletion: None,
            execution: Some((
                0,
                serde_json::to_vec(&envelope).expect("SDK JSON"),
                b"PRIVATE ignored stderr".to_vec(),
            )),
        };
        let command = TaskCommand::Invoke {
            agent: "echo".to_string(),
            runtime_id: "runtime-task".to_string(),
            request: json!({"request_id": "request-host", "input": [true, null], "context": {"turn": 2}}),
        };
        let output = runtime_command(environment_handle_fixture(), command, 30, &factory)
            .await
            .expect("failed SDK return");
        assert_eq!(output, json!({"status": "returned", "value": result}));
        assert!(
            calls
                .lock()
                .expect("calls")
                .iter()
                .any(|call| call.contains(
                    "fabric-agent-ctl invoke --agent echo --runtime-id runtime-task --request -"
                ))
        );
    }

    #[tokio::test]
    async fn task_rejection_keeps_unknown_effects_and_excludes_diagnostic_values() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let diagnostic = json!({"operation": "start", "status": "failed", "error": {"code": "start_failed", "stage": "start", "effects": "unknown", "message": "PRIVATE"}});
        let factory = MockFactory {
            calls,
            ready_id: "sandbox-id-1",
            deletion: None,
            execution: Some((
                1,
                Vec::new(),
                serde_json::to_vec(&diagnostic).expect("diagnostic JSON"),
            )),
        };
        let output = runtime_command(
            environment_handle_fixture(),
            TaskCommand::Start {
                agent: "echo".to_string(),
                config: json!({}),
            },
            30,
            &factory,
        )
        .await
        .expect("observed task rejection");
        assert_eq!(
            output,
            json!({"status": "rejected", "error": {"code": "start_failed", "stage": "start", "effects": "unknown"}})
        );
        assert!(!output.to_string().contains("PRIVATE"));
    }

    #[tokio::test]
    async fn unknown_start_diagnostic_is_preserved_by_the_provider() {
        let diagnostic = json!({"operation": "start", "status": "failed", "error": {"code": "start_uncertain", "stage": "start", "effects": "unknown", "message": "PRIVATE"}});
        let factory = MockFactory {
            calls: Arc::new(Mutex::new(Vec::new())),
            ready_id: "sandbox-id-1",
            deletion: None,
            execution: Some((
                1,
                Vec::new(),
                serde_json::to_vec(&diagnostic).expect("diagnostic JSON"),
            )),
        };

        let output = runtime_command(
            environment_handle_fixture(),
            TaskCommand::Start {
                agent: "echo".to_string(),
                config: json!({}),
            },
            30,
            &factory,
        )
        .await
        .expect("observed task rejection");

        assert_eq!(
            output,
            json!({"status": "rejected", "error": {"code": "start_uncertain", "stage": "start", "effects": "unknown"}})
        );
        assert!(!output.to_string().contains("PRIVATE"));
    }

    #[test]
    fn invocation_arguments_preserve_the_full_sdk_request() {
        let request = json!({"request_id": "caller-1", "input": ["review", null], "relay_session_root": "8da8e329-a5c3-49d3-8004-6f31e6833736", "context": {"turn": 2}, "overrides": {"max_iterations": 3}});
        let (_, arguments, payload) = TaskCommand::Invoke {
            agent: "echo".to_string(),
            runtime_id: "runtime-task".to_string(),
            request: request.clone(),
        }
        .arguments()
        .expect("SDK invocation");
        assert_eq!(payload, Some(request));
        assert_eq!(
            arguments,
            [
                "fabric-agent-ctl",
                "invoke",
                "--agent",
                "echo",
                "--runtime-id",
                "runtime-task",
                "--request",
                "-"
            ]
        );
    }

    #[test]
    fn core_invocation_golden_contract_decodes_without_payload_changes() {
        let expected: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/openshell_echo/invoke-command.json"
        ))
        .expect("golden contract");
        let command: TaskCommand =
            serde_json::from_value(expected.clone()).expect("core command contract");
        let (operation, arguments, payload) = command.arguments().expect("SDK CLI routing");
        assert_eq!(operation, "invoke");
        assert_eq!(payload, Some(expected["request"].clone()));
        assert_eq!(
            arguments,
            [
                "fabric-agent-ctl",
                "invoke",
                "--agent",
                "echo",
                "--runtime-id",
                "runtime-task",
                "--request",
                "-"
            ]
        );
    }

    struct NeverConnect;

    #[async_trait]
    impl GatewayFactory for NeverConnect {
        async fn connect(
            &self,
            _connection: &OpenShellConnection,
        ) -> Result<Box<dyn Gateway>, ProviderError> {
            panic!("malformed requests must not connect")
        }
    }

    struct MockFactory {
        calls: Arc<Mutex<Vec<String>>>,
        ready_id: &'static str,
        deletion: Option<DeletionResult>,
        execution: Option<(i32, Vec<u8>, Vec<u8>)>,
    }

    #[async_trait]
    impl GatewayFactory for MockFactory {
        async fn connect(
            &self,
            _connection: &OpenShellConnection,
        ) -> Result<Box<dyn Gateway>, ProviderError> {
            self.calls
                .lock()
                .expect("calls")
                .push("connect".to_string());
            Ok(Box::new(MockGateway {
                calls: Arc::clone(&self.calls),
                ready_id: self.ready_id,
                deletion: self.deletion.clone(),
                execution: self.execution.clone(),
            }))
        }
    }

    struct MockGateway {
        calls: Arc<Mutex<Vec<String>>>,
        ready_id: &'static str,
        deletion: Option<DeletionResult>,
        execution: Option<(i32, Vec<u8>, Vec<u8>)>,
    }

    #[async_trait]
    impl Gateway for MockGateway {
        async fn health(&self) -> Result<GatewayHealth, ProviderError> {
            self.calls.lock().expect("calls").push("health".to_string());
            Ok(GatewayHealth {
                status: GatewayStatus::Healthy,
                version: "test-1.0.0".to_string(),
            })
        }

        async fn create(
            &self,
            workspace: Option<&str>,
            request: SandboxCreate,
        ) -> Result<SandboxSnapshot, ProviderError> {
            let name = request.name.as_deref().expect("sandbox name");
            assert!(
                name.len() <= 19,
                "OpenShell rejects sandbox names longer than 19 bytes: {name}"
            );
            assert!(
                name.bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
            );
            assert_eq!(request.environment["FABRIC_VISIBLE"], "yes");
            assert_eq!(request.command, ["fabric-agent-ctl", "serve"]);
            self.calls.lock().expect("calls").push(format!(
                "create:{}:{}",
                workspace.unwrap_or("default"),
                request.labels["nemo-fabric.environment-id"]
            ));
            let mut sandbox = sandbox_snapshot_fixture(ProviderSandboxPhase::Provisioning);
            sandbox.workspace = workspace.unwrap_or("default").to_string();
            Ok(sandbox)
        }

        async fn get(
            &self,
            workspace: Option<&str>,
            name: &str,
        ) -> Result<SandboxSnapshot, ProviderError> {
            self.calls
                .lock()
                .expect("calls")
                .push(format!("get:{}:{name}", workspace.unwrap_or("default")));
            let mut sandbox = sandbox_snapshot_fixture(ProviderSandboxPhase::Ready);
            sandbox.id = self.ready_id.to_string();
            sandbox.workspace = workspace.unwrap_or("default").to_string();
            Ok(sandbox)
        }

        async fn wait_ready(
            &self,
            workspace: Option<&str>,
            name: &str,
            _timeout: Duration,
        ) -> Result<SandboxSnapshot, ProviderError> {
            self.calls.lock().expect("calls").push(format!(
                "wait_ready:{}:{name}",
                workspace.unwrap_or("default")
            ));
            let mut sandbox = sandbox_snapshot_fixture(ProviderSandboxPhase::Ready);
            sandbox.id = self.ready_id.to_string();
            sandbox.workspace = workspace.unwrap_or("default").to_string();
            Ok(sandbox)
        }

        async fn exec(
            &self,
            workspace: Option<&str>,
            name: &str,
            command: Vec<String>,
            request: ExecRequest,
        ) -> Result<ExecResponse, ProviderError> {
            self.calls.lock().expect("calls").push(format!(
                "exec:{}:{name}:{}:{}",
                workspace.unwrap_or("default"),
                command.join(" "),
                request.timeout.as_secs()
            ));
            if let Some((exit_code, stdout, stderr)) = &self.execution {
                return Ok(ExecResponse {
                    exit_code: *exit_code,
                    stdout: stdout.clone(),
                    stderr: stderr.clone(),
                });
            }
            let stdout = b"ok".to_vec();
            Ok(ExecResponse {
                exit_code: 0,
                stdout,
                stderr: Vec::new(),
            })
        }

        async fn delete(
            &self,
            workspace: Option<&str>,
            name: &str,
        ) -> Result<DeletionResult, ProviderError> {
            self.calls
                .lock()
                .expect("calls")
                .push(format!("delete:{}:{name}", workspace.unwrap_or("default")));
            Ok(self.deletion.clone().unwrap_or(DeletionResult {
                outcome: DeletionOutcome::Accepted,
                sandbox_id: Some(self.ready_id.to_string()),
            }))
        }

        async fn wait_deleted(
            &self,
            workspace: Option<&str>,
            name: &str,
            timeout: Duration,
            expected_sandbox_id: Option<&str>,
        ) -> Result<(), ProviderError> {
            self.calls.lock().expect("calls").push(format!(
                "wait_deleted:{}:{name}:{}",
                workspace.unwrap_or("default"),
                timeout.as_secs()
            ));
            assert_eq!(expected_sandbox_id, Some(self.ready_id));
            Ok(())
        }
    }

    fn environment_plan_fixture() -> EnvironmentPlan {
        EnvironmentPlan {
            provider: "openshell".to_string(),
            control_location: "in_env_control".to_string(),
            ownership: "fabric_owned".to_string(),
            workspace: Some("/sandbox".to_string()),
            artifacts: Some("/sandbox/artifacts".to_string()),
            env: HashMap::from([("FABRIC_VISIBLE".to_string(), "yes".to_string())]),
            connection: Map::from_iter([
                ("gateway".to_string(), json!("https://gateway.example")),
                ("workspace".to_string(), json!("fabric-demo")),
            ]),
            settings: Map::from_iter([
                (
                    "image".to_string(),
                    json!(format!(
                        "registry.example/agent-runtime@sha256:{}",
                        "a".repeat(64)
                    )),
                ),
                ("command".to_string(), json!(["fabric-agent-ctl", "serve"])),
            ]),
        }
    }

    fn environment_handle_fixture() -> EnvironmentHandle {
        EnvironmentHandle {
            environment_id: "environment-1".to_string(),
            provider: "openshell".to_string(),
            ownership: "fabric_owned".to_string(),
            env: HashMap::new(),
            workspace: Some("/sandbox".to_string()),
            artifacts: Some("/sandbox/artifacts".to_string()),
            connection: Map::from_iter([
                ("gateway".to_string(), json!("https://gateway.example")),
                ("workspace".to_string(), json!("fabric-demo")),
                ("sandbox_id".to_string(), json!("sandbox-id-1")),
                ("sandbox_name".to_string(), json!("fabric-sandbox-1")),
                ("delete_timeout_seconds".to_string(), json!(30)),
                ("exec_timeout_seconds".to_string(), json!(20)),
            ]),
        }
    }

    fn sandbox_snapshot_fixture(phase: ProviderSandboxPhase) -> SandboxSnapshot {
        SandboxSnapshot {
            id: "sandbox-id-1".to_string(),
            name: "fabric-sandbox-1".to_string(),
            workspace: "fabric-demo".to_string(),
            labels: HashMap::new(),
            phase,
            resource_version: 1,
            exit_code: None,
            image: Some(
                "registry.example/agent-runtime@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .to_string(),
            ),
            command: Some(vec![
                "fabric-agent-ctl".to_string(),
                "serve".to_string(),
            ]),
            policy: Some(
                openshell_policy::parse_sandbox_policy(
                    "version: 1\nfilesystem_policy:\n  include_workdir: true\n",
                )
                .expect("fixture policy"),
            ),
        }
    }
}
