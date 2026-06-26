// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime invocation helpers.

use std::collections::BTreeMap;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::config::{
    AdapterKind, CapabilityPlan, ControlLocation, EffectiveConfig, EnvironmentOwnership, RunPlan,
    RuntimeMode, TelemetryPlan,
};
use crate::error::{FabricError, Result};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// A request passed to a Fabric-managed harness runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Default)]
pub struct RunRequest {
    /// Request id.
    pub request_id: String,
    /// Request payload for the harness.
    #[serde(default)]
    pub input: Value,
    /// Runtime context such as task, rollout, session, or caller metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub context: BTreeMap<String, Value>,
    /// Per-invocation overrides allowed by the resolved profile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overrides: Option<Value>,
}

impl RunRequest {
    /// Build a text request.
    pub fn text(input: impl Into<String>) -> Self {
        Self {
            request_id: new_id("request"),
            input: Value::String(input.into()),
            context: BTreeMap::new(),
            overrides: None,
        }
    }
}

/// Result from a Fabric-managed harness invocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RunResult {
    /// Stable agent name.
    pub agent_name: String,
    /// Selected profile name when loaded through an agent manifest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    /// Harness type used for this run.
    pub harness_type: String,
    /// Adapter used for this run.
    pub adapter_kind: AdapterKind,
    /// Adapter implementation id when an adapter descriptor was resolved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_id: Option<String>,
    /// Runtime handle id.
    pub runtime_id: String,
    /// Invocation handle id.
    pub invocation_id: String,
    /// Request id.
    pub request_id: String,
    /// Runtime status.
    pub status: RunStatus,
    /// Primary output.
    #[serde(default)]
    pub output: Value,
    /// Error metadata when applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorInfo>,
    /// Artifacts produced or collected by this run.
    #[serde(default)]
    pub artifacts: ArtifactManifest,
    /// Telemetry reference when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry: Option<TelemetryRef>,
    /// Fabric lifecycle/progress events emitted during the run.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<FabricEvent>,
    /// Adapter-specific metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

/// Runtime completion status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    /// The harness completed successfully.
    Succeeded,
    /// The harness completed with a non-zero status.
    Failed,
    /// The invocation or runtime was cancelled.
    Cancelled,
}

/// Normalized error metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ErrorInfo {
    /// Fabric lifecycle stage where the failure surfaced.
    pub stage: ErrorStage,
    /// Stable error code.
    pub code: String,
    /// Human-readable error message.
    pub message: String,
    /// Whether Fabric considers this failure safe for a consumer-level retry.
    #[serde(default)]
    pub retryable: bool,
    /// Adapter or runtime metadata useful for diagnostics.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

/// Fabric lifecycle stage associated with an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ErrorStage {
    /// Configuration or profile loading failed.
    Config,
    /// Effective config planning failed.
    Plan,
    /// Environment preparation failed.
    Prepare,
    /// Runtime start/connect failed.
    Start,
    /// Runtime invocation failed.
    Invoke,
    /// Runtime stop/detach failed.
    Stop,
    /// Environment release failed.
    Release,
    /// Artifact collection or writing failed.
    Artifact,
}

/// Manifest of run artifacts.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ArtifactManifest {
    /// Artifact root directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root: Option<PathBuf>,
    /// Artifact entries.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<ArtifactRef>,
}

/// Reference to one artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ArtifactRef {
    /// Logical artifact name.
    pub name: String,
    /// Artifact kind.
    pub kind: String,
    /// Artifact path.
    pub path: PathBuf,
    /// Optional media type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
}

/// Reference to telemetry emitted by Relay or another configured telemetry path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TelemetryRef {
    /// Whether Relay was enabled for this run.
    pub relay_enabled: bool,
    /// Telemetry metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

/// Fabric lifecycle or progress event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FabricEvent {
    /// Event id.
    pub event_id: String,
    /// Unix timestamp in milliseconds.
    pub timestamp_millis: u128,
    /// Event kind.
    pub kind: String,
    /// Event message.
    pub message: String,
    /// Event metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

/// Resolved execution environment context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EnvironmentHandle {
    /// Environment handle id.
    pub environment_id: String,
    /// Environment provider.
    pub provider: String,
    /// Where Fabric control code runs.
    pub control_location: ControlLocation,
    /// Workspace visible to the harness runtime.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace: Option<PathBuf>,
    /// Artifact root visible to the harness runtime.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifacts: Option<PathBuf>,
    /// Whether Fabric owns the environment resource.
    pub ownership: EnvironmentOwnership,
    /// Provider connection metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub connection: BTreeMap<String, Value>,
    /// Provider-specific metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

/// Active or resumable harness runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RuntimeHandle {
    /// Runtime handle id.
    pub runtime_id: String,
    /// Agent name.
    pub agent_name: String,
    /// Harness type.
    pub harness_type: String,
    /// Runtime mode.
    pub mode: RuntimeMode,
    /// Adapter kind.
    pub adapter_kind: AdapterKind,
    /// Adapter implementation id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_id: Option<String>,
    /// Prepared environment.
    pub environment: EnvironmentHandle,
}

/// One request sent to a runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct InvocationHandle {
    /// Invocation handle id.
    pub invocation_id: String,
    /// Request id.
    pub request_id: String,
    /// Runtime id.
    pub runtime_id: String,
}

/// Per-run/per-invocation context passed to harness adapters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RuntimeContext {
    /// Runtime handle id.
    pub runtime_id: String,
    /// Invocation handle id.
    pub invocation_id: String,
    /// Request id.
    pub request_id: String,
    /// Prepared execution environment.
    pub environment: EnvironmentHandle,
    /// Artifact manifest visible to the adapter at invocation start.
    pub artifacts: ArtifactManifest,
    /// Runtime telemetry context generated for this invocation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry: Option<RuntimeTelemetryContext>,
}

/// Runtime telemetry config passed to adapters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RuntimeTelemetryContext {
    /// Whether Relay is enabled for this invocation.
    pub relay_enabled: bool,
    /// Generated Relay config path for this invocation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_path: Option<PathBuf>,
    /// Environment variables Fabric applies while invoking the adapter.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    /// Additional telemetry metadata surfaced to consumers and adapters.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

/// Adapter-facing invocation payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AdapterInvocation {
    /// Merged agent config and provenance.
    pub effective_config: EffectiveConfig,
    /// Per-runtime/per-invocation execution context.
    pub runtime_context: RuntimeContext,
    /// Per-invocation request.
    pub request: RunRequest,
    /// Derived capability routing plan for the selected adapter.
    #[serde(default)]
    pub capability_plan: CapabilityPlan,
    /// Derived telemetry plan for the selected adapter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry_plan: Option<TelemetryPlan>,
}

trait RuntimeAdapter {
    fn start(&self, plan: &RunPlan, environment: EnvironmentHandle) -> Result<RuntimeHandle>;
    fn invoke(
        &self,
        plan: &RunPlan,
        runtime: &RuntimeHandle,
        request: RunRequest,
    ) -> Result<RunResult>;
    fn stop(&self, runtime: &RuntimeHandle) -> Result<Vec<FabricEvent>>;
}

struct ProcessAdapter;
struct PythonAdapter;

#[derive(Debug, Clone)]
struct RelayRuntimeConfig {
    path: PathBuf,
    env: BTreeMap<String, String>,
}

/// Invoke a Fabric run plan.
pub fn run_plan(plan: &RunPlan, request: RunRequest) -> Result<RunResult> {
    let runtime = start_runtime(plan)?;
    let mut result = invoke_runtime(plan, &runtime, request)?;
    result.events.extend(stop_runtime(plan, &runtime)?);
    Ok(result)
}

/// Resolve or attach to the execution environment context for a run plan.
pub fn prepare_environment(plan: &RunPlan) -> Result<EnvironmentHandle> {
    let mut metadata = BTreeMap::new();
    let mut connection = BTreeMap::new();
    let (
        provider,
        control_location,
        ownership,
        workspace,
        artifacts,
        connection_settings,
        environment_metadata,
        settings,
    ) = if let Some(environment) = &plan.environment_plan {
        (
            environment.provider.clone(),
            environment.control_location,
            environment.ownership,
            environment.workspace.clone(),
            environment.artifacts.clone(),
            environment.connection.clone(),
            environment.metadata.clone(),
            environment.settings.clone(),
        )
    } else {
        (
            "local".to_string(),
            ControlLocation::ExternalControl,
            EnvironmentOwnership::CallerOwned,
            Some(plan.agent_root.clone()),
            plan.config
                .runtime
                .artifacts
                .as_ref()
                .map(|artifacts| resolve_path(&plan.config_root, artifacts)),
            serde_json::Map::new(),
            serde_json::Map::new(),
            serde_json::Map::new(),
        )
    };
    let workspace = match workspace {
        Some(path) => Some(absolute_path(path)?),
        None => None,
    };
    for (key, value) in connection_settings {
        connection.insert(key, value);
    }
    for (key, value) in settings {
        metadata.insert(key, value);
    }
    for (key, value) in environment_metadata {
        metadata.insert(key, value);
    }
    Ok(EnvironmentHandle {
        environment_id: new_id("environment"),
        provider,
        control_location,
        workspace,
        artifacts,
        ownership,
        connection,
        metadata,
    })
}

/// Start or connect to a harness runtime.
pub fn start_runtime(plan: &RunPlan) -> Result<RuntimeHandle> {
    let environment = prepare_environment(plan)?;
    match adapter_kind(plan) {
        AdapterKind::Process => ProcessAdapter.start(plan, environment),
        AdapterKind::Python => PythonAdapter.start(plan, environment),
        adapter_kind => Err(FabricError::UnsupportedRuntimeAdapter {
            harness: harness_type(plan),
            adapter_kind,
        }),
    }
}

/// Invoke a started harness runtime.
pub fn invoke_runtime(
    plan: &RunPlan,
    runtime: &RuntimeHandle,
    request: RunRequest,
) -> Result<RunResult> {
    match adapter_kind(plan) {
        AdapterKind::Process => ProcessAdapter.invoke(plan, runtime, request),
        AdapterKind::Python => PythonAdapter.invoke(plan, runtime, request),
        adapter_kind => Err(FabricError::UnsupportedRuntimeAdapter {
            harness: harness_type(plan),
            adapter_kind,
        }),
    }
}

/// Stop or detach from a harness runtime.
pub fn stop_runtime(_plan: &RunPlan, runtime: &RuntimeHandle) -> Result<Vec<FabricEvent>> {
    match runtime.adapter_kind {
        AdapterKind::Process => ProcessAdapter.stop(runtime),
        AdapterKind::Python => PythonAdapter.stop(runtime),
        adapter_kind => Err(FabricError::UnsupportedRuntimeAdapter {
            harness: runtime.harness_type.clone(),
            adapter_kind,
        }),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ProcessAdapterSettings {
    command: String,
    #[serde(default)]
    script: Option<PathBuf>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    cwd: Option<PathBuf>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    #[serde(default)]
    stdin_payload: ProcessStdinPayload,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ProcessStdinPayload {
    #[default]
    Input,
    FabricRequest,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct PythonAdapterSettings {
    script: PathBuf,
    #[serde(default)]
    python: Option<PathBuf>,
    #[serde(default)]
    python_env: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    cwd: Option<PathBuf>,
    #[serde(default)]
    env: BTreeMap<String, String>,
}

impl RuntimeAdapter for ProcessAdapter {
    fn start(&self, plan: &RunPlan, environment: EnvironmentHandle) -> Result<RuntimeHandle> {
        if environment.provider != "local" {
            return Err(FabricError::UnsupportedEnvironmentProvider {
                provider: environment.provider,
                adapter_kind: AdapterKind::Process,
            });
        }
        Ok(RuntimeHandle {
            runtime_id: new_id("runtime"),
            agent_name: plan.agent_name.clone(),
            harness_type: harness_type(plan),
            mode: plan.config.runtime.mode,
            adapter_kind: adapter_kind(plan),
            adapter_id: adapter_id(plan),
            environment,
        })
    }

    fn invoke(
        &self,
        plan: &RunPlan,
        runtime: &RuntimeHandle,
        request: RunRequest,
    ) -> Result<RunResult> {
        run_process_adapter(plan, runtime, request)
    }

    fn stop(&self, runtime: &RuntimeHandle) -> Result<Vec<FabricEvent>> {
        Ok(vec![event_with_metadata(
            "runtime_stop",
            format!("stopped runtime {}", runtime.runtime_id),
            BTreeMap::from([(
                "runtime_id".to_string(),
                Value::String(runtime.runtime_id.clone()),
            )]),
        )])
    }
}

impl RuntimeAdapter for PythonAdapter {
    fn start(&self, plan: &RunPlan, environment: EnvironmentHandle) -> Result<RuntimeHandle> {
        if environment.provider != "local" {
            return Err(FabricError::UnsupportedEnvironmentProvider {
                provider: environment.provider,
                adapter_kind: AdapterKind::Python,
            });
        }
        Ok(RuntimeHandle {
            runtime_id: new_id("runtime"),
            agent_name: plan.agent_name.clone(),
            harness_type: harness_type(plan),
            mode: plan.config.runtime.mode,
            adapter_kind: adapter_kind(plan),
            adapter_id: adapter_id(plan),
            environment,
        })
    }

    fn invoke(
        &self,
        plan: &RunPlan,
        runtime: &RuntimeHandle,
        request: RunRequest,
    ) -> Result<RunResult> {
        run_python_adapter(plan, runtime, request)
    }

    fn stop(&self, runtime: &RuntimeHandle) -> Result<Vec<FabricEvent>> {
        Ok(vec![event_with_metadata(
            "runtime_stop",
            format!("stopped runtime {}", runtime.runtime_id),
            BTreeMap::from([(
                "runtime_id".to_string(),
                Value::String(runtime.runtime_id.clone()),
            )]),
        )])
    }
}

fn run_process_adapter(
    plan: &RunPlan,
    runtime: &RuntimeHandle,
    mut request: RunRequest,
) -> Result<RunResult> {
    if request.request_id.is_empty() {
        request.request_id = new_id("request");
    }
    let invocation = InvocationHandle {
        invocation_id: new_id("invocation"),
        request_id: request.request_id.clone(),
        runtime_id: runtime.runtime_id.clone(),
    };
    let settings = parse_process_settings(plan)?;
    let command_path = resolve_command_path(
        adapter_setting_root(plan, "command"),
        Path::new(&settings.command),
    );
    let command_display = command_path.to_string_lossy().into_owned();
    let command_args = process_command_args(plan, &settings);
    let cwd = settings
        .cwd
        .as_ref()
        .map(|path| resolve_path(&plan.config_root, path))
        .or_else(|| runtime.environment.workspace.clone())
        .unwrap_or_else(|| plan.agent_root.clone());
    let mut artifacts = artifact_manifest(plan)?;
    let relay_config =
        prepare_relay_runtime_config(plan, runtime, &invocation, &request, &mut artifacts)?;
    let adapter_payload = fabric_adapter_payload(
        plan,
        runtime,
        &invocation,
        &request,
        &artifacts,
        relay_config.as_ref(),
    )?;
    let fabric_home = prepare_fabric_home(&artifacts, runtime, &invocation)?;
    let fabric_invocation = write_fabric_invocation(&fabric_home, &adapter_payload)?;

    let mut command = Command::new(&command_path);
    command
        .args(&command_args)
        .current_dir(&cwd)
        .envs(&settings.env)
        .envs(relay_env(&relay_config))
        .env("FABRIC_HOME", &fabric_home)
        .env("FABRIC_INVOCATION", &fabric_invocation)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(root) = artifacts.root.as_ref() {
        command.env("FABRIC_ARTIFACTS", root);
    }

    let mut events = vec![event_with_metadata(
        "runtime_start",
        format!("started runtime {}", runtime.runtime_id),
        BTreeMap::from([
            (
                "runtime_id".to_string(),
                Value::String(runtime.runtime_id.clone()),
            ),
            (
                "environment_id".to_string(),
                Value::String(runtime.environment.environment_id.clone()),
            ),
            (
                "environment_provider".to_string(),
                Value::String(runtime.environment.provider.clone()),
            ),
        ]),
    )];
    events.push(event_with_metadata(
        "invocation_start",
        format!("starting process adapter for {}", harness_type(plan)),
        BTreeMap::from([
            (
                "runtime_id".to_string(),
                Value::String(runtime.runtime_id.clone()),
            ),
            (
                "invocation_id".to_string(),
                Value::String(invocation.invocation_id.clone()),
            ),
        ]),
    ));
    let mut child = command
        .spawn()
        .map_err(|source| FabricError::ProcessRunner {
            command: command_display.clone(),
            source,
        })?;

    let stdin_payload = match settings.stdin_payload {
        ProcessStdinPayload::Input => value_to_stdin(&request.input)?,
        ProcessStdinPayload::FabricRequest => adapter_payload,
    };
    if !stdin_payload.is_empty() {
        if let Some(mut stdin) = child.stdin.take() {
            write_child_stdin(&mut stdin, &stdin_payload, &command_display)?;
        }
    }

    let output = child
        .wait_with_output()
        .map_err(|source| FabricError::ProcessRunner {
            command: command_display.clone(),
            source,
        })?;

    let exit_code = output.status.code();
    let status = if output.status.success() {
        RunStatus::Succeeded
    } else {
        RunStatus::Failed
    };
    events.push(event_with_metadata(
        "invocation_end",
        format!("process adapter completed with status {:?}", status),
        BTreeMap::from([
            (
                "runtime_id".to_string(),
                Value::String(runtime.runtime_id.clone()),
            ),
            (
                "invocation_id".to_string(),
                Value::String(invocation.invocation_id.clone()),
            ),
        ]),
    ));

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    if !stdout.is_empty() {
        write_artifact(
            &mut artifacts,
            "stdout",
            "log",
            "stdout.txt",
            &stdout,
            "text/plain",
        )?;
    }
    if !stderr.is_empty() {
        write_artifact(
            &mut artifacts,
            "stderr",
            "log",
            "stderr.txt",
            &stderr,
            "text/plain",
        )?;
    }
    collect_workspace_artifacts(&mut artifacts, runtime, &mut events)?;

    let mut metadata = BTreeMap::new();
    metadata.insert(
        "adapter_runner".to_string(),
        Value::String("process".to_string()),
    );
    metadata.insert("command".to_string(), Value::String(command_display));
    metadata.insert(
        "args".to_string(),
        Value::Array(command_args.into_iter().map(Value::String).collect()),
    );
    metadata.insert(
        "fabric_home".to_string(),
        Value::String(fabric_home.to_string_lossy().into_owned()),
    );
    metadata.insert(
        "fabric_invocation".to_string(),
        Value::String(fabric_invocation.to_string_lossy().into_owned()),
    );
    if let Some(exit_code) = exit_code {
        metadata.insert("exit_code".to_string(), Value::from(exit_code));
    }
    metadata.insert(
        "cwd".to_string(),
        Value::String(cwd.to_string_lossy().into_owned()),
    );
    metadata.insert(
        "environment_provider".to_string(),
        Value::String(runtime.environment.provider.clone()),
    );

    let error = if status == RunStatus::Failed {
        Some(adapter_exit_error(
            "process_exit_nonzero",
            "process exited with non-zero status",
            &stderr,
            &metadata,
        ))
    } else {
        None
    };

    let parsed_output = parse_stdout_output(&stdout);
    promote_relay_artifacts_to_manifest(&parsed_output, &mut artifacts);

    Ok(RunResult {
        agent_name: plan.agent_name.clone(),
        profile: plan.profile.clone(),
        harness_type: harness_type(plan),
        adapter_kind: adapter_kind(plan),
        adapter_id: adapter_id(plan),
        runtime_id: invocation.runtime_id,
        invocation_id: invocation.invocation_id,
        request_id: request.request_id,
        status,
        output: parsed_output,
        error,
        artifacts,
        telemetry: telemetry_ref(plan, relay_config.as_ref()),
        events,
        metadata,
    })
}

fn run_python_adapter(
    plan: &RunPlan,
    runtime: &RuntimeHandle,
    mut request: RunRequest,
) -> Result<RunResult> {
    if request.request_id.is_empty() {
        request.request_id = new_id("request");
    }
    let invocation = InvocationHandle {
        invocation_id: new_id("invocation"),
        request_id: request.request_id.clone(),
        runtime_id: runtime.runtime_id.clone(),
    };
    let settings = parse_python_settings(plan)?;
    let script = absolute_path(resolve_path(
        adapter_setting_root(plan, "script"),
        &settings.script,
    ))?;
    let cwd = settings
        .cwd
        .as_ref()
        .map(|path| resolve_path(&plan.config_root, path))
        .or_else(|| runtime.environment.workspace.clone())
        .unwrap_or_else(|| plan.agent_root.clone());

    let python = resolve_python_command(&plan.config_root, &settings);
    let mut artifacts = artifact_manifest(plan)?;
    let relay_config =
        prepare_relay_runtime_config(plan, runtime, &invocation, &request, &mut artifacts)?;

    let mut command = Command::new(&python);
    command
        .arg(&script)
        .args(&settings.args)
        .current_dir(&cwd)
        .envs(&settings.env)
        .envs(relay_env(&relay_config))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut events = vec![event_with_metadata(
        "runtime_start",
        format!("started runtime {}", runtime.runtime_id),
        BTreeMap::from([
            (
                "runtime_id".to_string(),
                Value::String(runtime.runtime_id.clone()),
            ),
            (
                "environment_id".to_string(),
                Value::String(runtime.environment.environment_id.clone()),
            ),
            (
                "environment_provider".to_string(),
                Value::String(runtime.environment.provider.clone()),
            ),
        ]),
    )];
    events.push(event_with_metadata(
        "invocation_start",
        format!("starting python adapter for {}", harness_type(plan)),
        BTreeMap::from([
            (
                "runtime_id".to_string(),
                Value::String(runtime.runtime_id.clone()),
            ),
            (
                "invocation_id".to_string(),
                Value::String(invocation.invocation_id.clone()),
            ),
            (
                "script".to_string(),
                Value::String(script.to_string_lossy().into_owned()),
            ),
        ]),
    ));

    let mut child = command
        .spawn()
        .map_err(|source| FabricError::ProcessRunner {
            command: python.to_string_lossy().into_owned(),
            source,
        })?;

    let stdin_payload = fabric_adapter_payload(
        plan,
        runtime,
        &invocation,
        &request,
        &artifacts,
        relay_config.as_ref(),
    )?;
    if let Some(mut stdin) = child.stdin.take() {
        write_child_stdin(&mut stdin, &stdin_payload, &python.to_string_lossy())?;
    }

    let output = child
        .wait_with_output()
        .map_err(|source| FabricError::ProcessRunner {
            command: python.to_string_lossy().into_owned(),
            source,
        })?;

    let exit_code = output.status.code();
    let status = if output.status.success() {
        RunStatus::Succeeded
    } else {
        RunStatus::Failed
    };
    events.push(event_with_metadata(
        "invocation_end",
        format!("python adapter completed with status {:?}", status),
        BTreeMap::from([
            (
                "runtime_id".to_string(),
                Value::String(runtime.runtime_id.clone()),
            ),
            (
                "invocation_id".to_string(),
                Value::String(invocation.invocation_id.clone()),
            ),
        ]),
    ));

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    if !stdout.is_empty() {
        write_artifact(
            &mut artifacts,
            "stdout",
            "log",
            "stdout.txt",
            &stdout,
            "text/plain",
        )?;
    }
    if !stderr.is_empty() {
        write_artifact(
            &mut artifacts,
            "stderr",
            "log",
            "stderr.txt",
            &stderr,
            "text/plain",
        )?;
    }
    collect_workspace_artifacts(&mut artifacts, runtime, &mut events)?;

    let mut metadata = BTreeMap::new();
    metadata.insert(
        "adapter_runner".to_string(),
        Value::String("python".to_string()),
    );
    metadata.insert(
        "python".to_string(),
        Value::String(python.to_string_lossy().into_owned()),
    );
    metadata.insert(
        "script".to_string(),
        Value::String(script.to_string_lossy().into_owned()),
    );
    metadata.insert(
        "args".to_string(),
        Value::Array(settings.args.into_iter().map(Value::String).collect()),
    );
    if let Some(exit_code) = exit_code {
        metadata.insert("exit_code".to_string(), Value::from(exit_code));
    }
    metadata.insert(
        "cwd".to_string(),
        Value::String(cwd.to_string_lossy().into_owned()),
    );
    metadata.insert(
        "environment_provider".to_string(),
        Value::String(runtime.environment.provider.clone()),
    );

    let error = if status == RunStatus::Failed {
        Some(adapter_exit_error(
            "python_adapter_exit_nonzero",
            "python adapter exited with non-zero status",
            &stderr,
            &metadata,
        ))
    } else {
        None
    };

    let parsed_output = parse_stdout_output(&stdout);
    promote_relay_artifacts_to_manifest(&parsed_output, &mut artifacts);

    Ok(RunResult {
        agent_name: plan.agent_name.clone(),
        profile: plan.profile.clone(),
        harness_type: harness_type(plan),
        adapter_kind: adapter_kind(plan),
        adapter_id: adapter_id(plan),
        runtime_id: invocation.runtime_id,
        invocation_id: invocation.invocation_id,
        request_id: request.request_id,
        status,
        output: parsed_output,
        error,
        artifacts,
        telemetry: telemetry_ref(plan, relay_config.as_ref()),
        events,
        metadata,
    })
}

fn adapter_id(plan: &RunPlan) -> Option<String> {
    plan.adapter_descriptor
        .as_ref()
        .map(|adapter| adapter.descriptor.adapter_id.clone())
        .or_else(|| Some(plan.config.harness.adapter_id.clone()))
}

fn write_child_stdin(stdin: &mut impl Write, payload: &str, command: &str) -> Result<()> {
    match stdin.write_all(payload.as_bytes()) {
        Ok(()) => Ok(()),
        Err(source) if source.kind() == ErrorKind::BrokenPipe => Ok(()),
        Err(source) => Err(FabricError::ProcessRunner {
            command: command.to_string(),
            source,
        }),
    }
}

fn harness_type(plan: &RunPlan) -> String {
    adapter_id(plan).unwrap_or_else(|| "unknown".to_string())
}

fn adapter_kind(plan: &RunPlan) -> AdapterKind {
    plan.adapter_descriptor
        .as_ref()
        .map(|adapter| adapter.descriptor.adapter_kind)
        .unwrap_or(AdapterKind::Process)
}

fn adapter_exit_error(
    code: &str,
    default_message: &str,
    stderr: &str,
    metadata: &BTreeMap<String, Value>,
) -> ErrorInfo {
    ErrorInfo {
        stage: ErrorStage::Invoke,
        code: code.to_string(),
        message: if stderr.is_empty() {
            default_message.to_string()
        } else {
            stderr.to_string()
        },
        retryable: false,
        metadata: metadata.clone(),
    }
}

fn parse_python_settings(plan: &RunPlan) -> Result<PythonAdapterSettings> {
    let value = Value::Object(merged_adapter_settings(plan));
    serde_json::from_value(value).map_err(|source| FabricError::InvalidPythonSettings {
        path: plan.config_path.clone(),
        source,
    })
}

fn parse_process_settings(plan: &RunPlan) -> Result<ProcessAdapterSettings> {
    let value = Value::Object(merged_adapter_settings(plan));
    serde_json::from_value(value).map_err(|source| FabricError::InvalidProcessSettings {
        path: plan.config_path.clone(),
        source,
    })
}

fn merged_adapter_settings(plan: &RunPlan) -> Map<String, Value> {
    let mut settings = plan
        .adapter_descriptor
        .as_ref()
        .map(|adapter| adapter.descriptor.runner.clone())
        .unwrap_or_default();
    settings.extend(plan.config.harness.settings.clone());
    settings
}

fn adapter_setting_root<'a>(plan: &'a RunPlan, key: &str) -> &'a Path {
    if plan.config.harness.settings.contains_key(key) {
        return &plan.config_root;
    }
    plan.adapter_descriptor
        .as_ref()
        .map(|adapter| adapter.root.as_path())
        .unwrap_or(&plan.config_root)
}

fn fabric_adapter_payload(
    plan: &RunPlan,
    runtime: &RuntimeHandle,
    invocation: &InvocationHandle,
    request: &RunRequest,
    artifacts: &ArtifactManifest,
    relay_config: Option<&RelayRuntimeConfig>,
) -> Result<String> {
    serde_json::to_string_pretty(&adapter_invocation(
        plan,
        runtime,
        invocation,
        request,
        artifacts,
        relay_config,
    )?)
    .map_err(FabricError::SerializeJson)
}

fn adapter_invocation(
    plan: &RunPlan,
    runtime: &RuntimeHandle,
    invocation: &InvocationHandle,
    request: &RunRequest,
    artifacts: &ArtifactManifest,
    relay_config: Option<&RelayRuntimeConfig>,
) -> Result<AdapterInvocation> {
    let mut effective_config = plan.effective_config.clone();
    effective_config.agent_root = absolute_path(effective_config.agent_root)?;
    effective_config.config_path = absolute_path(effective_config.config_path)?;
    effective_config.config_root = absolute_path(effective_config.config_root)?;
    Ok(AdapterInvocation {
        effective_config,
        runtime_context: RuntimeContext {
            runtime_id: runtime.runtime_id.clone(),
            invocation_id: invocation.invocation_id.clone(),
            request_id: request.request_id.clone(),
            environment: runtime.environment.clone(),
            artifacts: artifacts.clone(),
            telemetry: runtime_telemetry_context(plan, relay_config),
        },
        request: request.clone(),
        capability_plan: plan.capability_plan.clone(),
        telemetry_plan: plan.telemetry_plan.clone(),
    })
}

fn runtime_telemetry_context(
    plan: &RunPlan,
    relay_config: Option<&RelayRuntimeConfig>,
) -> Option<RuntimeTelemetryContext> {
    let telemetry = plan.telemetry_plan.as_ref()?;
    let mut metadata = BTreeMap::new();
    if let Some(mode) = &telemetry.relay_mode {
        metadata.insert("relay_mode".to_string(), Value::String(mode.clone()));
    }
    if let Some(project) = &telemetry.relay_project {
        metadata.insert("relay_project".to_string(), Value::String(project.clone()));
    }
    if let Some(output_dir) = &telemetry.relay_output_dir {
        metadata.insert(
            "relay_output_dir".to_string(),
            Value::String(output_dir.to_string_lossy().into_owned()),
        );
    }
    if !telemetry.adapter_outputs.is_empty() {
        metadata.insert(
            "adapter_outputs".to_string(),
            Value::Array(
                telemetry
                    .adapter_outputs
                    .iter()
                    .map(|output| Value::String(output.clone()))
                    .collect(),
            ),
        );
    }
    Some(RuntimeTelemetryContext {
        relay_enabled: telemetry.relay_enabled,
        config_path: relay_config.map(|relay| relay.path.clone()),
        env: relay_config
            .map(|relay| relay.env.clone())
            .unwrap_or_default(),
        metadata,
    })
}

fn resolve_path(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    root.join(path)
}

fn resolve_command_path(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() || path.components().count() > 1 {
        return resolve_path(root, path);
    }
    path.to_path_buf()
}

fn resolve_python_command(root: &Path, settings: &PythonAdapterSettings) -> PathBuf {
    if let Some(path) = settings.python.as_ref() {
        return resolve_command_path(root, path);
    }
    if let Some(env_name) = settings.python_env.as_ref() {
        if let Some(path) = std::env::var_os(env_name) {
            return resolve_command_path(root, Path::new(&path));
        }
    }
    PathBuf::from("python3")
}

fn absolute_path(path: PathBuf) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path);
    }
    let cwd = std::env::current_dir().map_err(|source| FabricError::ProcessRunner {
        command: "current_dir".to_string(),
        source,
    })?;
    Ok(cwd.join(path))
}

fn parse_stdout_output(stdout: &str) -> Value {
    serde_json::from_str(stdout).unwrap_or_else(|_| Value::String(stdout.to_string()))
}

#[derive(Debug, Default, Deserialize)]
struct RelayArtifactOutput {
    #[serde(default)]
    relay_artifacts: Vec<Value>,
}

#[derive(Debug, Deserialize)]
struct RelayArtifactCandidate {
    kind: String,
    path: PathBuf,
}

fn promote_relay_artifacts_to_manifest(output: &Value, manifest: &mut ArtifactManifest) {
    let relay_output: RelayArtifactOutput =
        serde_json::from_value(output.clone()).unwrap_or_default();

    for artifact in relay_output.relay_artifacts {
        let Ok(artifact) = serde_json::from_value::<RelayArtifactCandidate>(artifact) else {
            continue;
        };
        let kind = artifact.kind.as_str();
        if !matches!(kind, "atof" | "atif") {
            continue;
        }
        if artifact.path.as_os_str().is_empty() {
            continue;
        }

        let path = resolve_relay_artifact_path(manifest, &artifact.path);
        if !path.exists()
            || manifest
                .artifacts
                .iter()
                .any(|artifact| artifact.path == path)
        {
            continue;
        }

        let name = unique_relay_artifact_name(manifest, kind);
        manifest.artifacts.push(ArtifactRef {
            name,
            kind: kind.to_string(),
            path,
            media_type: relay_artifact_media_type(kind).map(str::to_string),
        });
    }
}

fn resolve_relay_artifact_path(manifest: &ArtifactManifest, path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    manifest
        .root
        .as_ref()
        .map(|root| root.join(path))
        .unwrap_or_else(|| path.to_path_buf())
}

fn relay_artifact_media_type(kind: &str) -> Option<&'static str> {
    match kind {
        "atof" => Some("application/x-ndjson"),
        "atif" => Some("application/json"),
        _ => None,
    }
}

fn unique_relay_artifact_name(manifest: &ArtifactManifest, kind: &str) -> String {
    let base = format!("relay_{kind}");
    if !manifest
        .artifacts
        .iter()
        .any(|artifact| artifact.name == base)
    {
        return base;
    }

    let mut index = 2;
    loop {
        let candidate = format!("{base}_{index}");
        if !manifest
            .artifacts
            .iter()
            .any(|artifact| artifact.name == candidate)
        {
            return candidate;
        }
        index += 1;
    }
}

fn process_command_args(plan: &RunPlan, settings: &ProcessAdapterSettings) -> Vec<String> {
    let mut args = Vec::new();
    if let Some(script) = settings.script.as_ref() {
        let root = adapter_setting_root(plan, "script");
        args.push(resolve_path(root, script).to_string_lossy().into_owned());
    }
    args.extend(settings.args.clone());
    args
}

fn value_to_stdin(value: &Value) -> Result<String> {
    match value {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        value => serde_json::to_string_pretty(value).map_err(FabricError::SerializeJson),
    }
}

fn artifact_manifest(plan: &RunPlan) -> Result<ArtifactManifest> {
    let root = plan
        .config
        .runtime
        .artifacts
        .as_ref()
        .map(|path| resolve_path(&plan.config_root, path))
        .or_else(|| {
            plan.environment_plan
                .as_ref()
                .and_then(|environment| environment.artifacts.clone())
        });
    if let Some(root) = &root {
        std::fs::create_dir_all(root).map_err(|source| FabricError::Write {
            path: root.clone(),
            source,
        })?;
    }
    Ok(ArtifactManifest {
        root,
        artifacts: Vec::new(),
    })
}

fn prepare_fabric_home(
    manifest: &ArtifactManifest,
    runtime: &RuntimeHandle,
    invocation: &InvocationHandle,
) -> Result<PathBuf> {
    let root = manifest
        .root
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join("nemo-fabric"));
    let fabric_home = root
        .join(".fabric")
        .join(&runtime.runtime_id)
        .join(&invocation.invocation_id);
    std::fs::create_dir_all(&fabric_home).map_err(|source| FabricError::Write {
        path: fabric_home.clone(),
        source,
    })?;
    Ok(fabric_home)
}

fn write_fabric_invocation(fabric_home: &Path, payload: &str) -> Result<PathBuf> {
    let path = fabric_home.join("adapter-invocation.json");
    std::fs::write(&path, payload).map_err(|source| FabricError::Write {
        path: path.clone(),
        source,
    })?;
    Ok(path)
}

fn write_artifact(
    manifest: &mut ArtifactManifest,
    name: &str,
    kind: &str,
    filename: &str,
    contents: &str,
    media_type: &str,
) -> Result<()> {
    let Some(root) = &manifest.root else {
        return Ok(());
    };
    let path = root.join(filename);
    std::fs::write(&path, contents).map_err(|source| FabricError::Write {
        path: path.clone(),
        source,
    })?;
    manifest.artifacts.push(ArtifactRef {
        name: name.to_string(),
        kind: kind.to_string(),
        path,
        media_type: Some(media_type.to_string()),
    });
    Ok(())
}

fn collect_workspace_artifacts(
    manifest: &mut ArtifactManifest,
    runtime: &RuntimeHandle,
    events: &mut Vec<FabricEvent>,
) -> Result<()> {
    let Some(workspace) = runtime.environment.workspace.as_ref() else {
        return Ok(());
    };
    if !workspace.join(".git").exists() {
        return Ok(());
    }
    let Ok(status) = Command::new("git")
        .arg("-C")
        .arg(workspace)
        .arg("status")
        .arg("--short")
        .output()
    else {
        return Ok(());
    };
    if !status.status.success() {
        return Ok(());
    }
    let status_text = String::from_utf8_lossy(&status.stdout).into_owned();
    if status_text.trim().is_empty() {
        return Ok(());
    }
    let mut patch = String::new();
    let Ok(diff) = Command::new("git")
        .arg("-C")
        .arg(workspace)
        .arg("diff")
        .arg("--binary")
        .arg("--")
        .arg(".")
        .output()
    else {
        return Ok(());
    };
    if !diff.status.success() {
        return Ok(());
    }
    patch.push_str(&String::from_utf8_lossy(&diff.stdout));
    patch.push_str(&untracked_workspace_patch(workspace)?);
    if patch.trim().is_empty() {
        return Ok(());
    }
    write_artifact(
        manifest,
        "workspace_patch",
        "patch",
        "workspace.patch",
        &patch,
        "text/x-diff",
    )?;
    write_artifact(
        manifest,
        "workspace_status",
        "log",
        "workspace-status.txt",
        &status_text,
        "text/plain",
    )?;
    events.push(event_with_metadata(
        "artifact_collect",
        "collected workspace patch artifact",
        BTreeMap::from([
            (
                "runtime_id".to_string(),
                Value::String(runtime.runtime_id.clone()),
            ),
            (
                "workspace".to_string(),
                Value::String(workspace.to_string_lossy().into_owned()),
            ),
        ]),
    ));
    Ok(())
}

fn untracked_workspace_patch(workspace: &Path) -> Result<String> {
    let Ok(untracked) = Command::new("git")
        .arg("-C")
        .arg(workspace)
        .arg("ls-files")
        .arg("--others")
        .arg("--exclude-standard")
        .arg("-z")
        .output()
    else {
        return Ok(String::new());
    };
    if !untracked.status.success() || untracked.stdout.is_empty() {
        return Ok(String::new());
    }
    let mut patch = String::new();
    for raw_path in untracked.stdout.split(|byte| *byte == 0) {
        if raw_path.is_empty() {
            continue;
        }
        let relative_path = String::from_utf8_lossy(raw_path);
        let Ok(diff) = Command::new("git")
            .arg("-C")
            .arg(workspace)
            .arg("diff")
            .arg("--binary")
            .arg("--no-index")
            .arg("--")
            .arg("/dev/null")
            .arg(relative_path.as_ref())
            .output()
        else {
            continue;
        };
        if !diff.stdout.is_empty() {
            if !patch.is_empty() && !patch.ends_with('\n') {
                patch.push('\n');
            }
            patch.push_str(&String::from_utf8_lossy(&diff.stdout));
        }
    }
    Ok(patch)
}

fn prepare_relay_runtime_config(
    plan: &RunPlan,
    runtime: &RuntimeHandle,
    invocation: &InvocationHandle,
    request: &RunRequest,
    artifacts: &mut ArtifactManifest,
) -> Result<Option<RelayRuntimeConfig>> {
    let Some(telemetry) = plan.telemetry_plan.as_ref() else {
        return Ok(None);
    };
    if !telemetry.relay_enabled {
        return Ok(None);
    }
    let Some(root) = artifacts.root.clone() else {
        return Ok(None);
    };
    let relay_config = serde_json::json!({
        "schema_version": "fabric.relay/v1alpha1",
        "relay": {
            "enabled": true,
            "mode": telemetry.relay_mode.as_deref().unwrap_or("sdk"),
            "project": telemetry.relay_project.clone(),
            "output_dir": telemetry
                .relay_output_dir
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned()),
            "config": telemetry
                .relay_config
                .clone()
                .unwrap_or_else(|| Value::Object(Default::default())),
        },
        "fabric": {
            "agent_name": plan.agent_name.clone(),
            "profile": plan.profile.clone(),
            "harness_type": harness_type(plan),
            "adapter_id": adapter_id(plan),
            "runtime_id": runtime.runtime_id.clone(),
            "invocation_id": invocation.invocation_id.clone(),
            "request_id": request.request_id.clone(),
            "adapter_outputs": telemetry.adapter_outputs.clone(),
        }
    });
    let contents =
        serde_json::to_string_pretty(&relay_config).map_err(FabricError::SerializeJson)?;
    write_artifact(
        artifacts,
        "relay_config",
        "telemetry_config",
        "relay-config.json",
        &contents,
        "application/json",
    )?;
    let path = absolute_path(root.join("relay-config.json"))?;
    let mode = telemetry
        .relay_mode
        .clone()
        .unwrap_or_else(|| "sdk".to_string());
    Ok(Some(RelayRuntimeConfig {
        path: path.clone(),
        env: BTreeMap::from([
            ("FABRIC_RELAY_ENABLED".to_string(), "true".to_string()),
            ("FABRIC_RELAY_MODE".to_string(), mode),
            (
                "FABRIC_RELAY_CONFIG_PATH".to_string(),
                path.to_string_lossy().into_owned(),
            ),
        ]),
    }))
}

fn relay_env(relay_config: &Option<RelayRuntimeConfig>) -> BTreeMap<String, String> {
    relay_config
        .as_ref()
        .map(|relay| relay.env.clone())
        .unwrap_or_default()
}

fn telemetry_ref(
    plan: &RunPlan,
    relay_runtime: Option<&RelayRuntimeConfig>,
) -> Option<TelemetryRef> {
    let telemetry = plan.telemetry_plan.as_ref()?;
    let mut metadata = BTreeMap::new();
    if let Some(mode) = &telemetry.relay_mode {
        metadata.insert("relay_mode".to_string(), Value::String(mode.clone()));
    }
    if let Some(project) = &telemetry.relay_project {
        metadata.insert("relay_project".to_string(), Value::String(project.clone()));
    }
    if let Some(output_dir) = &telemetry.relay_output_dir {
        metadata.insert(
            "relay_output_dir".to_string(),
            Value::String(output_dir.to_string_lossy().into_owned()),
        );
    }
    if let Some(config) = &telemetry.relay_config {
        metadata.insert("relay_config".to_string(), config.clone());
    }
    if !telemetry.adapter_outputs.is_empty() {
        metadata.insert(
            "adapter_outputs".to_string(),
            Value::Array(
                telemetry
                    .adapter_outputs
                    .iter()
                    .map(|output| Value::String(output.clone()))
                    .collect(),
            ),
        );
    }
    if let Some(relay_runtime) = relay_runtime {
        metadata.insert(
            "relay_config_path".to_string(),
            Value::String(relay_runtime.path.to_string_lossy().into_owned()),
        );
    }
    Some(TelemetryRef {
        relay_enabled: telemetry.relay_enabled,
        metadata,
    })
}

fn event_with_metadata(
    kind: impl Into<String>,
    message: impl Into<String>,
    metadata: BTreeMap<String, Value>,
) -> FabricEvent {
    FabricEvent {
        event_id: new_id("event"),
        timestamp_millis: now_millis(),
        kind: kind.into(),
        message: message.into(),
        metadata,
    }
}

fn new_id(prefix: &str) -> String {
    // The atomic counter only differentiates ids within a single process; a
    // process-backed runner spawns a fresh `fabric-cli` per call, resetting it
    // to 1. Include the process id (distinct across concurrently running
    // processes) so ids stay unique when two runs land in the same millisecond.
    let counter = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{}-{}-{counter}", now_millis(), std::process::id())
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::config::resolve_run_plan;

    fn fixture_agent_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/hermes-shim-agent")
    }

    fn temp_process_agent_dir() -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "fabric-process-adapter-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create agent dir");
        fs::create_dir_all(root.join("adapters/process")).expect("create adapters dir");
        fs::write(
            root.join("agent.yaml"),
            r#"schema_version: fabric.agent/v1alpha1
metadata:
  name: process-test-agent
harness:
  adapter_id: acme.fabric.process
  settings:
    command: cat
models:
  default:
    provider: test
    model: test-model
runtime:
  mode: oneshot
  transport: cli
  input_schema: text
  output_schema: text
  artifacts: ./artifacts
"#,
        )
        .expect("write config");
        fs::write(
            root.join("adapters/process/fabric-adapter.json"),
            process_adapter_descriptor(),
        )
        .expect("write adapter descriptor");
        root
    }

    fn process_adapter_descriptor() -> &'static str {
        r#"{
  "adapter_id": "acme.fabric.process",
  "adapter_kind": "process"
}"#
    }

    #[test]
    fn prepare_environment_absolutizes_workspace() {
        let root =
            std::env::temp_dir().join(format!("fabric-workspace-abs-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("adapters/process")).expect("create agent dir");
        fs::write(
            root.join("agent.yaml"),
            r#"schema_version: fabric.agent/v1alpha1
metadata:
  name: workspace-abs-agent
harness:
  adapter_id: acme.fabric.process
  settings:
    command: cat
models:
  default:
    provider: test
    model: test-model
runtime:
  mode: oneshot
  transport: cli
  input_schema: text
  output_schema: text
  artifacts: ./artifacts
environment:
  provider: local
  workspace: ./repos/my-service
"#,
        )
        .expect("write config");
        fs::write(
            root.join("adapters/process/fabric-adapter.json"),
            process_adapter_descriptor(),
        )
        .expect("write adapter descriptor");

        let mut plan = resolve_run_plan(&root, None).expect("run plan");
        // Force a relative workspace to reproduce the pre-fix condition where an
        // adapter would re-join it onto the absolute config_root and double the path.
        plan.environment_plan
            .as_mut()
            .expect("environment plan")
            .workspace = Some(PathBuf::from("repos/my-service"));

        let environment = prepare_environment(&plan).expect("prepare environment");
        let workspace = environment.workspace.expect("workspace");
        assert!(
            workspace.is_absolute(),
            "prepared workspace must be absolute so adapters do not re-resolve it: {workspace:?}"
        );
        assert!(workspace.ends_with("repos/my-service"), "{workspace:?}");

        let _ = fs::remove_dir_all(&root);
    }

    fn artifact_content(result: &RunResult, name: &str) -> String {
        let artifact = result
            .artifacts
            .artifacts
            .iter()
            .find(|artifact| artifact.name == name)
            .unwrap_or_else(|| panic!("missing artifact {name}"));
        fs::read_to_string(&artifact.path).expect("read artifact")
    }

    fn run_command(cwd: &Path, command: &str, args: &[&str]) {
        let output = Command::new(command)
            .args(args)
            .current_dir(cwd)
            .output()
            .expect("run command");
        assert!(
            output.status.success(),
            "command failed: {command} {args:?}\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn process_adapter_passes_input_to_stdin() {
        let root = temp_process_agent_dir();
        let plan = resolve_run_plan(&root, None).expect("run plan");
        let result = run_plan(&plan, RunRequest::text("hello fabric")).expect("run result");

        assert_eq!(result.status, RunStatus::Succeeded);
        assert_eq!(result.output, Value::String("hello fabric".to_string()));
        assert_eq!(result.metadata.get("exit_code"), Some(&Value::from(0)));
        assert_eq!(artifact_content(&result, "stdout"), "hello fabric");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn run_promotes_relay_artifacts_into_artifact_manifest() {
        let root = std::env::temp_dir().join(format!(
            "fabric-relay-artifact-manifest-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("adapters/process")).expect("create adapters dir");
        let adapter_script = r#"
from pathlib import Path
import json
relay_dir = Path("artifacts/relay").resolve()
relay_dir.mkdir(parents=True, exist_ok=True)
atof = relay_dir / "events.atof.jsonl"
atif = relay_dir / "trajectory-runtime.atif.json"
atif_extra = relay_dir / "trajectory-child.atif.json"
atof.write_text('{"kind":"scope"}\n', encoding="utf-8")
atif.write_text('{"trajectory":true}', encoding="utf-8")
atif_extra.write_text('{"trajectory":"child"}', encoding="utf-8")
print(json.dumps({
    "response": "ok",
    "relay_artifacts": [
        {"kind": "atof", "path": str(atof)},
        {"kind": "atif", "path": str(atif)},
        {"kind": "atif", "path": str(atif_extra)}
    ]
}))
"#;
        let agent_config = serde_json::json!({
            "schema_version": "fabric.agent/v1alpha1",
            "metadata": {
                "name": "relay-artifact-test-agent",
            },
            "harness": {
                "adapter_id": "acme.fabric.process",
                "settings": {
                    "command": "python3",
                    "args": ["-c", adapter_script],
                },
            },
            "models": {
                "default": {
                    "provider": "test",
                    "model": "test-model",
                },
            },
            "runtime": {
                "mode": "oneshot",
                "transport": "cli",
                "input_schema": "text",
                "output_schema": "text",
                "artifacts": "./artifacts",
            },
        });
        fs::write(
            root.join("agent.yaml"),
            serde_yaml::to_string(&agent_config).expect("serialize agent config"),
        )
        .expect("write config");
        fs::write(
            root.join("adapters/process/fabric-adapter.json"),
            process_adapter_descriptor(),
        )
        .expect("write adapter descriptor");

        let plan = resolve_run_plan(&root, None).expect("run plan");
        let result = run_plan(&plan, RunRequest::text("collect relay")).expect("run result");

        assert_eq!(result.status, RunStatus::Succeeded);
        let atof = result
            .artifacts
            .artifacts
            .iter()
            .find(|artifact| artifact.name == "relay_atof" && artifact.kind == "atof")
            .expect("ATOF artifact promoted to manifest");
        let atif = result
            .artifacts
            .artifacts
            .iter()
            .find(|artifact| artifact.name == "relay_atif" && artifact.kind == "atif")
            .expect("ATIF artifact promoted to manifest");
        let atif_extra = result
            .artifacts
            .artifacts
            .iter()
            .find(|artifact| artifact.name == "relay_atif_2" && artifact.kind == "atif")
            .expect("second ATIF artifact promoted to manifest with unique name");
        assert!(atof.path.ends_with("events.atof.jsonl"));
        assert_eq!(atof.media_type.as_deref(), Some("application/x-ndjson"));
        assert!(atif.path.ends_with("trajectory-runtime.atif.json"));
        assert_eq!(atif.media_type.as_deref(), Some("application/json"));
        assert!(atif_extra.path.ends_with("trajectory-child.atif.json"));
        assert_eq!(atif_extra.media_type.as_deref(), Some("application/json"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn process_adapter_failure_returns_structured_error() {
        let root = std::env::temp_dir().join(format!(
            "fabric-process-adapter-failure-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("adapters/process")).expect("create adapters dir");
        fs::write(
            root.join("agent.yaml"),
            r#"schema_version: fabric.agent/v1alpha1
metadata:
  name: process-failure-agent
harness:
  adapter_id: acme.fabric.process
  settings:
    command: sh
    args:
      - -c
      - |
        echo adapter exploded >&2
        exit 7
models:
  default:
    provider: test
    model: test-model
runtime:
  mode: oneshot
  transport: cli
  input_schema: text
  output_schema: text
  artifacts: ./artifacts
"#,
        )
        .expect("write config");
        fs::write(
            root.join("adapters/process/fabric-adapter.json"),
            process_adapter_descriptor(),
        )
        .expect("write adapter descriptor");

        let plan = resolve_run_plan(&root, None).expect("run plan");
        let result = run_plan(&plan, RunRequest::text("hello fabric")).expect("run result");

        assert_eq!(result.status, RunStatus::Failed);
        let error = result.error.as_ref().expect("structured error");
        assert_eq!(error.stage, ErrorStage::Invoke);
        assert_eq!(error.code, "process_exit_nonzero");
        assert!(!error.retryable);
        assert!(error.message.contains("adapter exploded"));
        assert_eq!(
            error.metadata.get("adapter_runner"),
            Some(&Value::String("process".to_string()))
        );
        assert_eq!(error.metadata.get("exit_code"), Some(&Value::from(7)));
        assert_eq!(artifact_content(&result, "stderr"), "adapter exploded\n");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn python_adapter_runs_hermes_shim() {
        let plan = resolve_run_plan(fixture_agent_dir(), Some("env_local")).expect("run plan");
        let result = run_plan(&plan, RunRequest::text("review this")).expect("run result");

        assert_eq!(result.status, RunStatus::Succeeded);
        assert_eq!(result.adapter_kind, AdapterKind::Python);
        assert_eq!(
            result.output["harness"],
            Value::String("hermes".to_string())
        );
        assert_eq!(
            result.output["received"],
            Value::String("review this".to_string())
        );
        assert_eq!(
            result.metadata.get("adapter_runner"),
            Some(&Value::String("python".to_string()))
        );
    }

    #[test]
    fn run_collects_workspace_patch_artifact() {
        let root =
            std::env::temp_dir().join(format!("fabric-patch-artifact-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let workspace = root.join("workspace");
        fs::create_dir_all(&workspace).expect("create workspace");
        fs::write(workspace.join("bug.py"), "def answer():\n    return 41\n")
            .expect("write baseline");
        run_command(&workspace, "git", &["init", "-q"]);
        run_command(&workspace, "git", &["add", "bug.py"]);
        fs::write(
            root.join("agent.yaml"),
            r#"schema_version: fabric.agent/v1alpha1
metadata:
  name: patch-test-agent
harness:
  adapter_id: acme.fabric.process
  settings:
    command: python3
    args:
      - -c
      - |
        from pathlib import Path
        Path("bug.py").write_text("def answer():\n    return 42\n")
        Path("notes.txt").write_text("fixed by fabric\n")
        print('patched')
models:
  default:
    provider: test
    model: test-model
runtime:
  mode: oneshot
  transport: cli
  input_schema: text
  output_schema: text
  artifacts: ./artifacts
environment:
  provider: local
  workspace: ./workspace
  artifacts: ./artifacts
"#,
        )
        .expect("write config");
        fs::create_dir_all(root.join("adapters/process")).expect("create adapters dir");
        fs::write(
            root.join("adapters/process/fabric-adapter.json"),
            process_adapter_descriptor(),
        )
        .expect("write adapter descriptor");

        let plan = resolve_run_plan(&root, None).expect("run plan");
        let result = run_plan(&plan, RunRequest::text("fix the bug")).expect("run result");

        assert_eq!(result.status, RunStatus::Succeeded);
        let patch_artifact = result
            .artifacts
            .artifacts
            .iter()
            .find(|artifact| artifact.name == "workspace_patch")
            .expect("workspace patch artifact");
        let patch = fs::read_to_string(&patch_artifact.path).expect("read patch");
        assert!(patch.contains("-    return 41"));
        assert!(patch.contains("+    return 42"));
        assert!(patch.contains("new file mode"));
        assert!(patch.contains("notes.txt"));
        assert!(patch.contains("+fixed by fabric"));

        let _ = fs::remove_dir_all(root);
    }
}
