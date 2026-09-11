// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental `OpenShell` environment provider for NVIDIA `NeMo` Fabric.

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::time::Duration;

use async_trait::async_trait;
use openshell_sdk::raw::proto;
use openshell_sdk::{
    AuthConfig, ClientConfig, ExecOptions, OpenShellClient, SandboxPhase, SandboxRef, SdkError,
    ServiceStatus,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use thiserror::Error;

const PROTOCOL_VERSION: &str = "fabric.environment-provider.v1alpha1";
const RUNTIME_CONTROL_PROTOCOL_VERSION: &str = "fabric.runtime-control.v1alpha1";
const MAX_REQUEST_BYTES: usize = 256 * 1024;
const MAX_EXEC_OUTPUT_BYTES: usize = 3 * 1024 * 1024;
const MAX_ARTIFACT_BYTES: usize = 256 * 1024;
const MAX_ARTIFACT_TOTAL_BYTES: usize = 512 * 1024;
const MAX_ARTIFACT_FILES: usize = 16;
const DEFAULT_READY_TIMEOUT_SECONDS: u64 = 60;
const DEFAULT_DELETE_TIMEOUT_SECONDS: u64 = 30;
const DEFAULT_EXEC_TIMEOUT_SECONDS: u64 = 30;

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
    let mut line = Vec::new();
    loop {
        line.clear();
        let read = reader.read_until(b'\n', &mut line)?;
        if read == 0 {
            return Ok(());
        }
        if line.len() > MAX_REQUEST_BYTES {
            return Err(ProviderError::contract(
                "request_too_large",
                format!("request exceeds the {MAX_REQUEST_BYTES}-byte limit"),
            ));
        }
        let response = match serde_json::from_slice::<ProviderRequest>(&line) {
            Ok(request) => handle_request(request, factory).await,
            Err(error) => ProviderResponse::failed(
                "unknown",
                "invalid_request",
                format!("request was not valid provider JSON: {error}"),
            ),
        };
        serde_json::to_writer(&mut writer, &response).map_err(|error| {
            ProviderError::contract("response_serialization", error.to_string())
        })?;
        writer.write_all(b"\n")?;
        writer.flush()?;
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
        ProviderOperation::RuntimeControl {
            environment,
            request,
        } => runtime_control(environment, request, factory).await,
        ProviderOperation::CollectArtifacts {
            environment,
            artifacts,
        } => collect_artifacts(environment, artifacts, factory).await,
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
    let client = factory.connect(&connection).await?;
    let health = client.health().await?;
    if health.status != GatewayStatus::Healthy {
        return Err(ProviderError::contract(
            "gateway_unhealthy",
            format!("OpenShell gateway reported {:?}", health.status),
        ));
    }

    let sandbox = client
        .create(
            connection.workspace.as_deref(),
            SandboxCreate {
                name: settings.sandbox_name.clone(),
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
        )
        .await?;
    let ready = match client
        .wait_ready(
            connection.workspace.as_deref(),
            &sandbox.name,
            Duration::from_secs(settings.ready_timeout_seconds),
        )
        .await
    {
        Ok(ready) => ready,
        Err(error) => {
            cleanup_created_sandbox(
                client.as_ref(),
                connection.workspace.as_deref(),
                &sandbox.name,
                settings.delete_timeout_seconds,
            )
            .await;
            return Err(error);
        }
    };
    if ready.id != sandbox.id {
        cleanup_created_sandbox(
            client.as_ref(),
            connection.workspace.as_deref(),
            &sandbox.name,
            settings.delete_timeout_seconds,
        )
        .await;
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
    let client = factory.connect(&connection).await?;
    let health = client.health().await?;
    if health.status != GatewayStatus::Healthy {
        return Err(ProviderError::contract(
            "gateway_unhealthy",
            format!("OpenShell gateway reported {:?}", health.status),
        ));
    }
    let sandbox = client
        .get(connection.workspace.as_deref(), &resource.sandbox_name)
        .await?;
    if sandbox.id != resource.sandbox_id {
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
    timeout_seconds: u64,
) {
    let _ = client.delete(workspace, name).await;
    let _ = client
        .wait_deleted(workspace, name, Duration::from_secs(timeout_seconds))
        .await;
}

async fn inspect_environment<F>(
    environment: EnvironmentHandle,
    factory: &F,
) -> Result<Value, ProviderError>
where
    F: GatewayFactory,
{
    let binding = SandboxBinding::from_environment(&environment)?;
    let client = factory.connect(&binding.connection).await?;
    let sandbox = client
        .get(binding.connection.workspace.as_deref(), &binding.name)
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
    let client = factory.connect(&binding.connection).await?;
    let sandbox = client
        .get(binding.connection.workspace.as_deref(), &binding.name)
        .await?;
    binding.verify(&sandbox)?;
    let timeout = timeout_seconds
        .or(binding.connection.exec_timeout_seconds)
        .unwrap_or(DEFAULT_EXEC_TIMEOUT_SECONDS);
    let result = client
        .exec(
            binding.connection.workspace.as_deref(),
            &binding.name,
            command,
            ExecRequest {
                workdir,
                environment: env,
                timeout: Duration::from_secs(timeout),
                stdin,
            },
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

async fn runtime_control<F>(
    environment: EnvironmentHandle,
    request: RuntimeControlRequest,
    factory: &F,
) -> Result<Value, ProviderError>
where
    F: GatewayFactory,
{
    request.validate(&environment)?;
    let binding = SandboxBinding::from_environment(&environment)?;
    let client = factory.connect(&binding.connection).await?;
    let sandbox = client
        .get(binding.connection.workspace.as_deref(), &binding.name)
        .await?;
    binding.verify(&sandbox)?;
    let stdin = serde_json::to_vec(&request).map_err(|error| {
        ProviderError::contract(
            "runtime_control_protocol_error",
            format!("could not encode runtime-control request: {error}"),
        )
    })?;
    let timeout = request.timeout_seconds.checked_add(5).ok_or_else(|| {
        ProviderError::contract(
            "invalid_runtime_control_request",
            "runtime-control timeout is too large",
        )
    })?;
    let result = client
        .exec(
            binding.connection.workspace.as_deref(),
            &binding.name,
            vec!["fabric-runtime-ctl".to_string(), request.operation.clone()],
            ExecRequest {
                workdir: environment.workspace,
                environment: HashMap::new(),
                timeout: Duration::from_secs(timeout),
                stdin: Some(stdin),
            },
        )
        .await?;
    if result.stdout.len().saturating_add(result.stderr.len()) > MAX_EXEC_OUTPUT_BYTES {
        return Err(ProviderError::contract(
            "exec_output_too_large",
            format!("buffered exec output exceeds the {MAX_EXEC_OUTPUT_BYTES}-byte limit"),
        ));
    }
    if result.exit_code != 0 {
        let diagnostics = String::from_utf8_lossy(&result.stderr);
        return Err(ProviderError::contract(
            "runtime_control_failed",
            if diagnostics.trim().is_empty() {
                format!("fabric-runtime-ctl exited with {}", result.exit_code)
            } else {
                format!(
                    "fabric-runtime-ctl exited with {}: {}",
                    result.exit_code,
                    diagnostics.trim()
                )
            },
        ));
    }
    let output: Value = serde_json::from_slice(&result.stdout).map_err(|error| {
        ProviderError::contract(
            "runtime_control_protocol_error",
            format!("fabric-runtime-ctl returned invalid JSON: {error}"),
        )
    })?;
    let identity: RuntimeControlResponseIdentity =
        serde_json::from_value(output.clone()).map_err(|error| {
            ProviderError::contract(
                "runtime_control_protocol_error",
                format!("runtime-control response did not match the contract: {error}"),
            )
        })?;
    identity.validate(&request)?;
    Ok(output)
}

async fn collect_artifacts<F>(
    environment: EnvironmentHandle,
    artifacts: Vec<ArtifactRequest>,
    factory: &F,
) -> Result<Value, ProviderError>
where
    F: GatewayFactory,
{
    if artifacts.is_empty() || artifacts.len() > MAX_ARTIFACT_FILES {
        return Err(ProviderError::contract(
            "invalid_artifact_request",
            format!("artifact count must be between 1 and {MAX_ARTIFACT_FILES}"),
        ));
    }
    let root = environment.artifacts.as_deref().ok_or_else(|| {
        ProviderError::contract(
            "artifact_root_unavailable",
            "environment handle is missing its agent artifact root",
        )
    })?;
    if !root.starts_with('/') {
        return Err(ProviderError::contract(
            "invalid_artifact_request",
            "agent artifact root must be absolute",
        ));
    }
    for artifact in &artifacts {
        validate_artifact_path(&artifact.path)?;
    }

    let binding = SandboxBinding::from_environment(&environment)?;
    let client = factory.connect(&binding.connection).await?;
    let sandbox = client
        .get(binding.connection.workspace.as_deref(), &binding.name)
        .await?;
    binding.verify(&sandbox)?;

    let mut total = 0usize;
    let mut collected = Vec::with_capacity(artifacts.len());
    for artifact in artifacts {
        let stdin = serde_json::to_vec(&json!({
            "root": root,
            "path": artifact.path,
            "max_bytes": MAX_ARTIFACT_BYTES,
        }))
        .map_err(|error| {
            ProviderError::contract(
                "artifact_protocol_error",
                format!("could not encode artifact request: {error}"),
            )
        })?;
        let result = client
            .exec(
                binding.connection.workspace.as_deref(),
                &binding.name,
                vec![
                    "fabric-runtime-ctl".to_string(),
                    "collect-artifact".to_string(),
                ],
                ExecRequest {
                    workdir: environment.workspace.clone(),
                    environment: HashMap::new(),
                    timeout: Duration::from_secs(
                        binding
                            .connection
                            .exec_timeout_seconds
                            .unwrap_or(DEFAULT_EXEC_TIMEOUT_SECONDS),
                    ),
                    stdin: Some(stdin),
                },
            )
            .await?;
        if result.exit_code != 0 {
            return Err(ProviderError::contract(
                "artifact_export_failed",
                format!(
                    "fabric-runtime-ctl could not export declared artifact `{}`",
                    artifact.path
                ),
            ));
        }
        if result.stdout.len() > MAX_ARTIFACT_BYTES {
            return Err(ProviderError::contract(
                "artifact_too_large",
                format!(
                    "declared artifact `{}` exceeded its size limit",
                    artifact.path
                ),
            ));
        }
        total = total.checked_add(result.stdout.len()).ok_or_else(|| {
            ProviderError::contract("artifact_too_large", "artifact byte count overflowed")
        })?;
        if total > MAX_ARTIFACT_TOTAL_BYTES {
            return Err(ProviderError::contract(
                "artifact_set_too_large",
                format!(
                    "declared artifacts exceeded the {MAX_ARTIFACT_TOTAL_BYTES}-byte total limit"
                ),
            ));
        }
        collected.push(CollectedArtifact {
            path: artifact.path,
            content: result.stdout,
        });
    }
    serde_json::to_value(collected).map_err(|error| {
        ProviderError::contract(
            "artifact_protocol_error",
            format!("could not encode collected artifacts: {error}"),
        )
    })
}

fn validate_artifact_path(path: &str) -> Result<(), ProviderError> {
    let path = std::path::Path::new(path);
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(ProviderError::contract(
            "invalid_artifact_path",
            "artifact path must be a non-empty relative path without traversal",
        ));
    }
    Ok(())
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
    let client = factory.connect(&binding.connection).await?;
    match client
        .get(binding.connection.workspace.as_deref(), &binding.name)
        .await
    {
        Ok(sandbox) => binding.verify(&sandbox)?,
        Err(error) if error.code() == "not_found" => {
            return Ok(json!({"released": false, "detached": false}));
        }
        Err(error) => return Err(error),
    }
    let released = client
        .delete(binding.connection.workspace.as_deref(), &binding.name)
        .await?;
    client
        .wait_deleted(
            binding.connection.workspace.as_deref(),
            &binding.name,
            Duration::from_secs(
                binding
                    .connection
                    .delete_timeout_seconds
                    .unwrap_or(DEFAULT_DELETE_TIMEOUT_SECONDS),
            ),
        )
        .await?;
    Ok(json!({"released": released, "detached": false}))
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
    for (field, path) in [
        ("environment.workspace", environment.workspace.as_deref()),
        ("environment.artifacts", environment.artifacts.as_deref()),
    ] {
        if let Some(path) = path
            && !path.starts_with('/')
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
    RuntimeControl {
        environment: EnvironmentHandle,
        request: RuntimeControlRequest,
    },
    CollectArtifacts {
        environment: EnvironmentHandle,
        artifacts: Vec<ArtifactRequest>,
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
        let reference: Self = serde_json::from_value(Value::Object(resource)).map_err(|error| {
            ProviderError::contract(
                "invalid_reference",
                format!("invalid OpenShell resource reference: {error}"),
            )
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
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactRequest {
    path: String,
}

#[derive(Debug, Serialize)]
struct CollectedArtifact {
    path: String,
    content: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RuntimeControlRequest {
    protocol_version: String,
    operation_id: String,
    environment_id: String,
    runtime_id: String,
    timeout_seconds: u64,
    operation: String,
    #[serde(flatten)]
    payload: Map<String, Value>,
}

impl RuntimeControlRequest {
    fn validate(&self, environment: &EnvironmentHandle) -> Result<(), ProviderError> {
        if self.protocol_version != RUNTIME_CONTROL_PROTOCOL_VERSION {
            return Err(ProviderError::contract(
                "runtime_control_protocol_mismatch",
                format!(
                    "expected `{RUNTIME_CONTROL_PROTOCOL_VERSION}` but received `{}`",
                    self.protocol_version
                ),
            ));
        }
        if self.operation_id.trim().is_empty()
            || self.environment_id.trim().is_empty()
            || self.runtime_id.trim().is_empty()
            || self.timeout_seconds == 0
        {
            return Err(ProviderError::contract(
                "invalid_runtime_control_request",
                "runtime-control identity and timeout fields must be non-empty",
            ));
        }
        if !matches!(self.operation.as_str(), "start" | "invoke" | "stop") {
            return Err(ProviderError::contract(
                "invalid_runtime_control_request",
                format!("unsupported runtime-control operation `{}`", self.operation),
            ));
        }
        if self.environment_id != environment.environment_id {
            return Err(ProviderError::contract(
                "runtime_control_environment_mismatch",
                "runtime-control request environment id does not match the environment handle",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct RuntimeControlResponseIdentity {
    protocol_version: String,
    operation_id: String,
    environment_id: String,
    runtime_id: String,
    operation: String,
    status: String,
}

impl RuntimeControlResponseIdentity {
    fn validate(&self, request: &RuntimeControlRequest) -> Result<(), ProviderError> {
        for (field, expected, actual) in [
            (
                "protocol_version",
                RUNTIME_CONTROL_PROTOCOL_VERSION,
                self.protocol_version.as_str(),
            ),
            ("operation_id", &request.operation_id, &self.operation_id),
            (
                "environment_id",
                &request.environment_id,
                &self.environment_id,
            ),
            ("runtime_id", &request.runtime_id, &self.runtime_id),
            ("operation", &request.operation, &self.operation),
        ] {
            if expected != actual {
                return Err(ProviderError::contract(
                    "runtime_control_correlation_mismatch",
                    format!("runtime-control response `{field}` did not match the request"),
                ));
            }
        }
        if !matches!(self.status.as_str(), "succeeded" | "failed") {
            return Err(ProviderError::contract(
                "runtime_control_protocol_error",
                "runtime-control response status must be `succeeded` or `failed`",
            ));
        }
        Ok(())
    }
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
            serde_json::from_value(Value::Object(map.clone())).map_err(|error| {
                ProviderError::contract(
                    "invalid_connection",
                    format!("invalid OpenShell connection settings: {error}"),
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
        let policy = map
            .remove("policy_yaml")
            .map(|value| {
                let yaml = value.as_str().ok_or_else(|| {
                    ProviderError::contract(
                        "invalid_settings",
                        "settings.policy_yaml must be a YAML string",
                    )
                })?;
                openshell_policy::parse_sandbox_policy(yaml).map_err(|error| {
                    ProviderError::contract(
                        "invalid_settings",
                        format!("settings.policy_yaml is invalid: {error}"),
                    )
                })
            })
            .transpose()?;
        let settings: Self = serde_json::from_value(Value::Object(map)).map_err(|error| {
            ProviderError::contract(
                "invalid_settings",
                format!("invalid OpenShell provider settings: {error}"),
            )
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
            if value == 0 {
                return Err(ProviderError::contract(
                    "invalid_settings",
                    format!("{field} must be greater than zero"),
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
        if sandbox.id != self.id {
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
    async fn delete(&self, workspace: Option<&str>, name: &str) -> Result<bool, ProviderError>;
    async fn wait_deleted(
        &self,
        workspace: Option<&str>,
        name: &str,
        timeout: Duration,
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
            workspace: workspace.unwrap_or_default().to_string(),
            ..proto::CreateSandboxRequest::default()
        };
        let response = self
            .client
            .raw_grpc()
            .create_sandbox(request)
            .await
            .map_err(|error| {
                ProviderError::contract("sdk_rpc", format!("OpenShell create failed: {error}"))
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
                workspace: workspace.unwrap_or_default().to_string(),
            })
            .await
            .map_err(|error| {
                ProviderError::contract("sdk_rpc", format!("OpenShell get failed: {error}"))
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

    async fn delete(&self, workspace: Option<&str>, name: &str) -> Result<bool, ProviderError> {
        match workspace {
            Some(workspace) => self.client.workspace(workspace).delete_sandbox(name).await,
            None => self.client.delete_sandbox(name).await,
        }
        .map_err(map_sdk_error)
    }

    async fn wait_deleted(
        &self,
        workspace: Option<&str>,
        name: &str,
        timeout: Duration,
    ) -> Result<(), ProviderError> {
        match workspace {
            Some(workspace) => {
                self.client
                    .workspace(workspace)
                    .wait_deleted(name, timeout)
                    .await
            }
            None => self.client.wait_deleted(name, timeout).await,
        }
        .map_err(map_sdk_error)
    }
}

fn sandbox_snapshot(sandbox: SandboxRef) -> SandboxSnapshot {
    SandboxSnapshot {
        id: sandbox.id,
        name: sandbox.name,
        workspace: sandbox.workspace,
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
    ProviderError::contract(error.code(), error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn settings_require_a_digest_pinned_image() {
        let error = OpenShellSettings::from_map(Map::from_iter([
            ("image".to_string(), json!("example/agent-runtime:latest")),
            (
                "command".to_string(),
                json!(["fabric-runtime-server", "serve"]),
            ),
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
            (
                "command".to_string(),
                json!(["fabric-runtime-server", "serve"]),
            ),
        ]))
        .expect("immutable local image id");
    }

    #[test]
    fn settings_parse_a_creation_time_policy() {
        let settings = OpenShellSettings::from_map(Map::from_iter([
            (
                "image".to_string(),
                json!(format!("example/agent-runtime@sha256:{}", "a".repeat(64))),
            ),
            (
                "command".to_string(),
                json!(["fabric-runtime-server", "serve"]),
            ),
            (
                "policy_yaml".to_string(),
                json!("version: 1\nfilesystem_policy:\n  include_workdir: true\n"),
            ),
        ]))
        .expect("valid settings");

        assert_eq!(settings.policy.expect("policy").version, 1);
    }

    #[test]
    fn connection_rejects_literal_unknown_credential_fields() {
        let error = OpenShellConnection::from_map(&Map::from_iter([
            ("gateway".to_string(), json!("https://gateway.example")),
            ("token".to_string(), json!("literal-secret")),
        ]))
        .expect_err("literal token field must fail closed");

        assert_eq!(error.code(), "invalid_connection");
        assert!(error.to_string().contains("unknown field `token`"));
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
    async fn prepare_runs_health_create_and_readiness_through_the_sdk_boundary() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let factory = MockFactory {
            calls: Arc::clone(&calls),
            ready_id: "sandbox-id-1",
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
                (
                    "command".to_string(),
                    json!(["fabric-runtime-server", "serve"]),
                ),
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
        };
        let environment = environment_handle_fixture();

        let inspection = inspect_environment(environment_handle_fixture(), &factory)
            .await
            .expect("inspect environment");
        let execution = exec_environment(
            environment_handle_fixture(),
            vec!["fabric-runtime-ctl".to_string(), "inspect".to_string()],
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
                "exec:fabric-demo:fabric-sandbox-1:fabric-runtime-ctl inspect:12",
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
    async fn runtime_control_executes_only_the_typed_correlated_operation() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let factory = MockFactory {
            calls: Arc::clone(&calls),
            ready_id: "sandbox-id-1",
        };
        let request = RuntimeControlRequest {
            protocol_version: RUNTIME_CONTROL_PROTOCOL_VERSION.to_string(),
            operation_id: "operation-1".to_string(),
            environment_id: "environment-1".to_string(),
            runtime_id: "runtime-1".to_string(),
            timeout_seconds: 12,
            operation: "invoke".to_string(),
            payload: Map::from_iter([("lifecycle".to_string(), json!({"operation": "invoke"}))]),
        };

        let output = runtime_control(environment_handle_fixture(), request, &factory)
            .await
            .expect("runtime control");

        assert_eq!(output["operation_id"], "operation-1");
        assert_eq!(output["environment_id"], "environment-1");
        assert_eq!(output["runtime_id"], "runtime-1");
        assert_eq!(output["operation"], "invoke");
        assert_eq!(output["status"], "succeeded");
        assert_eq!(
            *calls.lock().expect("calls"),
            [
                "connect",
                "get:fabric-demo:fabric-sandbox-1",
                "exec:fabric-demo:fabric-sandbox-1:fabric-runtime-ctl invoke:17",
            ]
        );
    }

    #[tokio::test]
    async fn artifact_collection_is_typed_bounded_and_identity_checked() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let factory = MockFactory {
            calls: Arc::clone(&calls),
            ready_id: "sandbox-id-1",
        };

        let output = collect_artifacts(
            environment_handle_fixture(),
            vec![ArtifactRequest {
                path: "delivery-receipt.json".to_string(),
            }],
            &factory,
        )
        .await
        .expect("collect artifact");

        assert_eq!(output[0]["path"], "delivery-receipt.json");
        assert_eq!(output[0]["content"], json!([111, 107]));
        assert_eq!(
            *calls.lock().expect("calls"),
            [
                "connect",
                "get:fabric-demo:fabric-sandbox-1",
                "exec:fabric-demo:fabric-sandbox-1:fabric-runtime-ctl collect-artifact:20",
            ]
        );
    }

    #[tokio::test]
    async fn artifact_collection_rejects_traversal_before_connecting() {
        let error = collect_artifacts(
            environment_handle_fixture(),
            vec![ArtifactRequest {
                path: "../secret".to_string(),
            }],
            &NeverConnect,
        )
        .await
        .expect_err("traversal must fail");

        assert_eq!(error.code(), "invalid_artifact_path");
    }

    #[test]
    fn runtime_control_rejects_an_uncorrelated_response() {
        let request = RuntimeControlRequest {
            protocol_version: RUNTIME_CONTROL_PROTOCOL_VERSION.to_string(),
            operation_id: "operation-1".to_string(),
            environment_id: "environment-1".to_string(),
            runtime_id: "runtime-1".to_string(),
            timeout_seconds: 12,
            operation: "invoke".to_string(),
            payload: Map::new(),
        };
        let response = RuntimeControlResponseIdentity {
            protocol_version: RUNTIME_CONTROL_PROTOCOL_VERSION.to_string(),
            operation_id: "operation-1".to_string(),
            environment_id: "environment-1".to_string(),
            runtime_id: "another-runtime".to_string(),
            operation: "invoke".to_string(),
            status: "succeeded".to_string(),
        };

        let error = response
            .validate(&request)
            .expect_err("runtime mismatch must fail closed");

        assert_eq!(error.code(), "runtime_control_correlation_mismatch");
    }

    #[tokio::test]
    async fn prepare_cleans_up_when_readiness_returns_a_different_identity() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let factory = MockFactory {
            calls: Arc::clone(&calls),
            ready_id: "replaced-sandbox-id",
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
                (
                    "command".to_string(),
                    json!(["fabric-runtime-server", "serve"]),
                ),
            ]),
        };

        let error = prepare_environment("environment-1", environment, &factory)
            .await
            .expect_err("identity replacement must fail");

        assert_eq!(error.code(), "sandbox_identity_changed");
        assert!(calls.lock().expect("calls").ends_with(&[
            "delete:fabric-demo:fabric-sandbox-1".to_string(),
            "wait_deleted:fabric-demo:fabric-sandbox-1:30".to_string(),
        ]));
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
            }))
        }
    }

    struct MockGateway {
        calls: Arc<Mutex<Vec<String>>>,
        ready_id: &'static str,
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
            assert_eq!(request.environment["FABRIC_VISIBLE"], "yes");
            assert_eq!(request.command, ["fabric-runtime-server", "serve"]);
            self.calls.lock().expect("calls").push(format!(
                "create:{}:{}",
                workspace.expect("workspace"),
                request.labels["nemo-fabric.environment-id"]
            ));
            Ok(sandbox_snapshot_fixture(ProviderSandboxPhase::Provisioning))
        }

        async fn get(
            &self,
            workspace: Option<&str>,
            name: &str,
        ) -> Result<SandboxSnapshot, ProviderError> {
            self.calls
                .lock()
                .expect("calls")
                .push(format!("get:{}:{name}", workspace.expect("workspace")));
            let mut sandbox = sandbox_snapshot_fixture(ProviderSandboxPhase::Ready);
            sandbox.id = self.ready_id.to_string();
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
                workspace.expect("workspace")
            ));
            let mut sandbox = sandbox_snapshot_fixture(ProviderSandboxPhase::Ready);
            sandbox.id = self.ready_id.to_string();
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
                workspace.expect("workspace"),
                command.join(" "),
                request.timeout.as_secs()
            ));
            let stdout = if matches!(
                command.get(1).map(String::as_str),
                Some("start" | "invoke" | "stop")
            ) {
                let request: Value = serde_json::from_slice(
                    request.stdin.as_deref().expect("runtime request stdin"),
                )
                .expect("runtime request JSON");
                serde_json::to_vec(&json!({
                    "protocol_version": RUNTIME_CONTROL_PROTOCOL_VERSION,
                    "operation_id": request["operation_id"],
                    "environment_id": request["environment_id"],
                    "runtime_id": request["runtime_id"],
                    "operation": request["operation"],
                    "status": "succeeded",
                    "output": null,
                }))
                .expect("runtime response JSON")
            } else {
                b"ok".to_vec()
            };
            Ok(ExecResponse {
                exit_code: 0,
                stdout,
                stderr: Vec::new(),
            })
        }

        async fn delete(&self, workspace: Option<&str>, name: &str) -> Result<bool, ProviderError> {
            self.calls
                .lock()
                .expect("calls")
                .push(format!("delete:{}:{name}", workspace.expect("workspace")));
            Ok(true)
        }

        async fn wait_deleted(
            &self,
            workspace: Option<&str>,
            name: &str,
            timeout: Duration,
        ) -> Result<(), ProviderError> {
            self.calls.lock().expect("calls").push(format!(
                "wait_deleted:{}:{name}:{}",
                workspace.expect("workspace"),
                timeout.as_secs()
            ));
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
                (
                    "command".to_string(),
                    json!(["fabric-runtime-server", "serve"]),
                ),
            ]),
        }
    }

    fn environment_handle_fixture() -> EnvironmentHandle {
        EnvironmentHandle {
            environment_id: "environment-1".to_string(),
            provider: "openshell".to_string(),
            ownership: "fabric_owned".to_string(),
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
            phase,
            resource_version: 1,
            exit_code: None,
            image: Some(
                "registry.example/agent-runtime@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .to_string(),
            ),
            command: Some(vec![
                "fabric-runtime-server".to_string(),
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
