// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime invocation helpers.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
#[cfg(test)]
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::config::{
    AdapterKind, CapabilityKind, CapabilityPlan, CapabilityTarget, ControlLocation,
    EffectiveConfig, EnvironmentOwnership, RunPlan, TelemetryPlan,
};
use crate::error::{FabricError, Result};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
const ADAPTER_PYTHON_ENV: &str = "ADAPTER_PYTHON";
const VIRTUAL_ENV_ENV: &str = "VIRTUAL_ENV";

#[cfg(not(windows))]
const VENV_BIN_DIR: &str = "bin";
#[cfg(windows)]
const VENV_BIN_DIR: &str = "Scripts";

#[cfg(not(windows))]
const VENV_PYTHON: &str = "python";
#[cfg(windows)]
const VENV_PYTHON: &str = "python.exe";

#[cfg(not(windows))]
const DEFAULT_PYTHON: &str = "python3";
// `find_on_path` joins each PATH entry with this bare name and stats the result,
// so on Windows the fallback must name `python.exe` to be found on disk.
#[cfg(windows)]
const DEFAULT_PYTHON: &str = "python.exe";
#[cfg(test)]
static TEST_STOPPED_AGENTS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// A request passed to a Fabric-managed harness runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Default)]
pub struct RunRequest {
    /// Request id.
    pub request_id: String,
    /// Request payload for the harness.
    #[serde(default)]
    pub input: Value,
    /// Runtime context such as task, rollout, workflow, or caller metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub context: BTreeMap<String, Value>,
    /// Per-invocation overrides allowed by the resolved config.
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
    /// Stable machine-readable harness identifier used for this run.
    pub harness: String,
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
    /// Configuration resolution failed.
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
    /// Fabric-owned opaque binding for this runtime handle.
    pub runtime_binding: String,
    /// Agent name.
    pub agent_name: String,
    /// Stable machine-readable harness identifier.
    pub harness: String,
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
    /// Resolved typed configuration with explicit path context.
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
    let mut result = match invoke_runtime(plan, &runtime, request) {
        Ok(result) => result,
        Err(error) => {
            let _ = stop_runtime(plan, &runtime);
            return Err(error);
        }
    };
    result.events.extend(stop_runtime(plan, &runtime)?);
    Ok(result)
}

/// Resolve or attach to the execution environment context for a run plan.
pub fn prepare_environment(plan: &RunPlan) -> Result<EnvironmentHandle> {
    plan.validate_consistency()?;
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
            Some(plan.base_dir.clone()),
            plan.config
                .runtime
                .artifacts
                .as_ref()
                .map(|artifacts| resolve_path(&plan.base_dir, artifacts)),
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
    validate_blocked_tools_support(plan)?;
    let environment = prepare_environment(plan)?;
    match adapter_kind(plan) {
        AdapterKind::Process => ProcessAdapter.start(plan, environment),
        AdapterKind::Python => PythonAdapter.start(plan, environment),
        adapter_kind => Err(FabricError::UnsupportedRuntimeAdapter {
            harness: harness(plan),
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
    plan.validate_consistency()?;
    validate_blocked_tools_support(plan)?;
    validate_runtime_handle(plan, runtime)?;
    match adapter_kind(plan) {
        AdapterKind::Process => ProcessAdapter.invoke(plan, runtime, request),
        AdapterKind::Python => PythonAdapter.invoke(plan, runtime, request),
        adapter_kind => Err(FabricError::UnsupportedRuntimeAdapter {
            harness: harness(plan),
            adapter_kind,
        }),
    }
}

fn validate_blocked_tools_support(plan: &RunPlan) -> Result<()> {
    if let Some(route) = plan.capability_plan.routes.iter().find(|route| {
        route.kind == CapabilityKind::Tools && route.target == CapabilityTarget::Unsupported
    }) {
        return Err(FabricError::UnsupportedToolsPolicy {
            harness: harness(plan),
            reason: route.reason.clone(),
        });
    }
    Ok(())
}

/// Stop or detach from a harness runtime.
pub fn stop_runtime(plan: &RunPlan, runtime: &RuntimeHandle) -> Result<Vec<FabricEvent>> {
    plan.validate_consistency()?;
    validate_runtime_handle(plan, runtime)?;
    match runtime.adapter_kind {
        AdapterKind::Process => ProcessAdapter.stop(runtime),
        AdapterKind::Python => PythonAdapter.stop(runtime),
        adapter_kind => Err(FabricError::UnsupportedRuntimeAdapter {
            harness: runtime.harness.clone(),
            adapter_kind,
        }),
    }
}

fn validate_runtime_handle(plan: &RunPlan, runtime: &RuntimeHandle) -> Result<()> {
    let expected_binding = runtime_binding(&runtime.runtime_id, plan, &runtime.environment)?;
    expect_runtime_field(
        runtime,
        "runtime_binding",
        &expected_binding,
        &runtime.runtime_binding,
    )?;
    expect_runtime_field(runtime, "agent_name", &plan.agent_name, &runtime.agent_name)?;
    expect_runtime_field(runtime, "harness", &harness(plan), &runtime.harness)?;
    expect_runtime_field(
        runtime,
        "adapter_kind",
        &adapter_kind_name(adapter_kind(plan)),
        &adapter_kind_name(runtime.adapter_kind),
    )?;
    expect_runtime_field(
        runtime,
        "adapter_id",
        &optional_runtime_value(adapter_id(plan).as_deref()),
        &optional_runtime_value(runtime.adapter_id.as_deref()),
    )?;
    Ok(())
}

fn expect_runtime_field(
    runtime: &RuntimeHandle,
    field: &'static str,
    expected: &str,
    actual: &str,
) -> Result<()> {
    if expected == actual {
        return Ok(());
    }
    Err(FabricError::RuntimeHandleMismatch {
        field,
        expected: expected.to_string(),
        actual: actual.to_string(),
        runtime_id: runtime.runtime_id.clone(),
    })
}

#[derive(Serialize)]
struct RuntimeBindingMaterial<'a> {
    runtime_id: &'a str,
    environment_id: &'a str,
    plan: &'a RunPlan,
    environment: RuntimeEnvironmentBinding<'a>,
}

#[derive(Serialize)]
struct RuntimeEnvironmentBinding<'a> {
    provider: &'a str,
    control_location: ControlLocation,
    workspace: &'a Option<PathBuf>,
    artifacts: &'a Option<PathBuf>,
    ownership: EnvironmentOwnership,
    connection: &'a BTreeMap<String, Value>,
    metadata: &'a BTreeMap<String, Value>,
}

fn runtime_environment_binding(environment: &EnvironmentHandle) -> RuntimeEnvironmentBinding<'_> {
    RuntimeEnvironmentBinding {
        provider: &environment.provider,
        control_location: environment.control_location,
        workspace: &environment.workspace,
        artifacts: &environment.artifacts,
        ownership: environment.ownership,
        connection: &environment.connection,
        metadata: &environment.metadata,
    }
}

fn runtime_binding(
    runtime_id: &str,
    plan: &RunPlan,
    environment: &EnvironmentHandle,
) -> Result<String> {
    stable_hash(
        "fabric-runtime-binding",
        &RuntimeBindingMaterial {
            runtime_id,
            environment_id: &environment.environment_id,
            plan,
            environment: runtime_environment_binding(environment),
        },
    )
}

fn stable_hash<T: Serialize>(prefix: &str, value: &T) -> Result<String> {
    let bytes = serde_json::to_vec(value).map_err(FabricError::SerializeJson)?;
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Ok(format!("{prefix}-{hash:016x}"))
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
    module: String,
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

#[derive(Debug, Clone, PartialEq)]
struct PythonCommand {
    path: PathBuf,
    source: PythonSource,
}

/// Where a resolved Python interpreter came from. Drives clear preflight error
/// messages and how the interpreter is validated (concrete file vs. PATH lookup).
#[derive(Debug, Clone, PartialEq)]
enum PythonSource {
    /// `harness.settings.python`.
    Setting,
    /// `harness.settings.python_env` (the named environment variable).
    SettingEnv(String),
    /// The `ADAPTER_PYTHON` environment variable.
    AdapterPythonEnv,
    /// The active virtualenv (`VIRTUAL_ENV`).
    Virtualenv,
    /// An interpreter found next to the running Fabric executable.
    HostInterpreter,
    /// The bare `python3` command resolved off `PATH` (last resort).
    DefaultPython3,
}

impl PythonSource {
    fn describe(&self) -> String {
        match self {
            PythonSource::Setting => "harness.settings.python".to_string(),
            PythonSource::SettingEnv(name) => {
                format!("harness.settings.python_env (`{name}`)")
            }
            PythonSource::AdapterPythonEnv => {
                format!("`{ADAPTER_PYTHON_ENV}` environment variable")
            }
            PythonSource::Virtualenv => format!("active virtualenv (`{VIRTUAL_ENV_ENV}`)"),
            PythonSource::HostInterpreter => "Fabric host interpreter".to_string(),
            PythonSource::DefaultPython3 => format!("default `{DEFAULT_PYTHON}` on PATH"),
        }
    }
}

impl RuntimeAdapter for ProcessAdapter {
    fn start(&self, plan: &RunPlan, environment: EnvironmentHandle) -> Result<RuntimeHandle> {
        if environment.provider != "local" {
            return Err(FabricError::UnsupportedEnvironmentProvider {
                provider: environment.provider,
                adapter_kind: AdapterKind::Process,
            });
        }
        let runtime_id = new_id("runtime");
        let runtime_binding = runtime_binding(&runtime_id, plan, &environment)?;
        Ok(RuntimeHandle {
            runtime_id,
            runtime_binding,
            agent_name: plan.agent_name.clone(),
            harness: harness(plan),
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
        #[cfg(test)]
        TEST_STOPPED_AGENTS
            .lock()
            .expect("stop tracker")
            .push(runtime.agent_name.clone());
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
        preflight_python_adapter(plan)?;
        let runtime_id = new_id("runtime");
        let runtime_binding = runtime_binding(&runtime_id, plan, &environment)?;
        Ok(RuntimeHandle {
            runtime_id,
            runtime_binding,
            agent_name: plan.agent_name.clone(),
            harness: harness(plan),
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
        #[cfg(test)]
        TEST_STOPPED_AGENTS
            .lock()
            .expect("stop tracker")
            .push(runtime.agent_name.clone());
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
        .map(|path| resolve_path(&plan.base_dir, path))
        .or_else(|| runtime.environment.workspace.clone())
        .unwrap_or_else(|| plan.base_dir.clone());
    let mut artifacts = artifact_manifest(plan)?;
    let fabric_home = prepare_fabric_home(&artifacts, runtime, &invocation)?;
    let relay_config = prepare_relay_runtime_config(
        plan,
        runtime,
        &invocation,
        &request,
        &fabric_home,
        &mut artifacts,
    )?;
    let adapter_payload = fabric_adapter_payload(
        plan,
        runtime,
        &invocation,
        &request,
        &artifacts,
        relay_config.as_ref(),
    )?;
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
        format!("starting process adapter for {}", harness(plan)),
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
            &fabric_home,
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
            &fabric_home,
            "stderr",
            "log",
            "stderr.txt",
            &stderr,
            "text/plain",
        )?;
    }
    collect_workspace_artifacts(&mut artifacts, &fabric_home, runtime, &mut events)?;

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
        harness: harness(plan),
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
    let cwd = settings
        .cwd
        .as_ref()
        .map(|path| resolve_path(&plan.base_dir, path))
        .or_else(|| runtime.environment.workspace.clone())
        .unwrap_or_else(|| plan.base_dir.clone());

    let python = resolve_python_command(&plan.base_dir, &settings).path;
    let mut artifacts = artifact_manifest(plan)?;
    let fabric_home = prepare_fabric_home(&artifacts, runtime, &invocation)?;
    let relay_config = prepare_relay_runtime_config(
        plan,
        runtime,
        &invocation,
        &request,
        &fabric_home,
        &mut artifacts,
    )?;

    let mut command = Command::new(&python);
    command
        .arg("-m")
        .arg(&settings.module)
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
        format!("starting python adapter for {}", harness(plan)),
        BTreeMap::from([
            (
                "runtime_id".to_string(),
                Value::String(runtime.runtime_id.clone()),
            ),
            (
                "invocation_id".to_string(),
                Value::String(invocation.invocation_id.clone()),
            ),
            ("module".to_string(), Value::String(settings.module.clone())),
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
            &fabric_home,
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
            &fabric_home,
            "stderr",
            "log",
            "stderr.txt",
            &stderr,
            "text/plain",
        )?;
    }
    collect_workspace_artifacts(&mut artifacts, &fabric_home, runtime, &mut events)?;

    let mut metadata = BTreeMap::new();
    metadata.insert(
        "adapter_runner".to_string(),
        Value::String("python".to_string()),
    );
    metadata.insert(
        "python".to_string(),
        Value::String(python.to_string_lossy().into_owned()),
    );
    metadata.insert("module".to_string(), Value::String(settings.module));
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
        harness: harness(plan),
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

fn harness(plan: &RunPlan) -> String {
    plan.adapter_descriptor
        .as_ref()
        .map(|adapter| adapter.descriptor.harness.clone())
        .unwrap_or_else(|| "unknown".to_string())
}

fn adapter_kind(plan: &RunPlan) -> AdapterKind {
    plan.adapter_descriptor
        .as_ref()
        .map(|adapter| adapter.descriptor.adapter_kind)
        .unwrap_or(AdapterKind::Process)
}

fn adapter_kind_name(adapter_kind: AdapterKind) -> String {
    match adapter_kind {
        AdapterKind::Process => "process",
        AdapterKind::Http => "http",
        AdapterKind::Python => "python",
        AdapterKind::NativePlugin => "native_plugin",
    }
    .to_string()
}

fn optional_runtime_value(value: Option<&str>) -> String {
    value.unwrap_or("<none>").to_string()
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
        path: plan.base_dir.clone(),
        source,
    })
}

fn parse_process_settings(plan: &RunPlan) -> Result<ProcessAdapterSettings> {
    let value = Value::Object(merged_adapter_settings(plan));
    serde_json::from_value(value).map_err(|source| FabricError::InvalidProcessSettings {
        path: plan.base_dir.clone(),
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
        return &plan.base_dir;
    }
    plan.adapter_descriptor
        .as_ref()
        .map(|adapter| adapter.root.as_path())
        .unwrap_or(&plan.base_dir)
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
    effective_config.base_dir = absolute_path(effective_config.base_dir)?;
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
    metadata.insert(
        "telemetry_providers".to_string(),
        Value::Array(
            telemetry
                .providers
                .iter()
                .map(|provider| Value::String(provider.as_str().to_string()))
                .collect(),
        ),
    );
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

fn preflight_python_adapter(plan: &RunPlan) -> Result<()> {
    let settings = parse_python_settings(plan)?;
    let command = resolve_python_command(&plan.base_dir, &settings);
    validate_python_command(&command)
}

fn validate_python_command(command: &PythonCommand) -> Result<()> {
    let path = &command.path;
    // A single-component relative path (e.g. `python3`) is a bare command name
    // resolved off PATH; anything else is a concrete file path we can stat.
    let is_bare_command = !path.is_absolute() && path.components().count() == 1;
    if is_bare_command {
        if find_on_path(path).is_none() {
            return Err(FabricError::PythonInterpreterUnavailable {
                path: path.clone(),
                origin: command.source.describe(),
                reason: "not found on PATH".to_string(),
            });
        }
    } else if !path.is_file() {
        let reason = if path.exists() {
            "not a regular file"
        } else {
            "no such file"
        };
        return Err(FabricError::PythonInterpreterUnavailable {
            path: path.clone(),
            origin: command.source.describe(),
            reason: reason.to_string(),
        });
    }
    Ok(())
}

/// Locate a bare command name on `PATH`, returning the first matching file.
fn find_on_path(name: &Path) -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    std::env::split_paths(&paths)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

fn resolve_python_command(root: &Path, settings: &PythonAdapterSettings) -> PythonCommand {
    resolve_python_command_with_env(
        root,
        settings,
        |name| std::env::var_os(name),
        |path| path.is_file(),
        std::env::current_exe().ok(),
    )
}

fn resolve_python_command_with_env(
    root: &Path,
    settings: &PythonAdapterSettings,
    env: impl Fn(&str) -> Option<OsString>,
    exists: impl Fn(&Path) -> bool,
    current_exe: Option<PathBuf>,
) -> PythonCommand {
    if let Some(path) = settings.python.as_ref() {
        return PythonCommand {
            path: resolve_command_path(root, path),
            source: PythonSource::Setting,
        };
    }
    if let Some(env_name) = settings.python_env.as_ref() {
        // A set-but-empty variable is treated as unset so it falls through to
        // the shared fallback chain rather than yielding an empty path.
        if let Some(path) = env(env_name)
            && !path.as_os_str().is_empty()
        {
            return PythonCommand {
                path: resolve_command_path(root, Path::new(&path)),
                source: PythonSource::SettingEnv(env_name.clone()),
            };
        }
        return fallback_interpreter(&env, &exists, current_exe.as_deref());
    }
    if let Some(value) = env(ADAPTER_PYTHON_ENV)
        && !value.as_os_str().is_empty()
    {
        return PythonCommand {
            path: resolve_command_path(root, Path::new(&value)),
            source: PythonSource::AdapterPythonEnv,
        };
    }
    fallback_interpreter(&env, &exists, current_exe.as_deref())
}

/// With no interpreter explicitly configured, prefer (in order) the active
/// virtualenv, an interpreter next to the running Fabric executable, and only
/// then a bare `python3` off PATH. A preinstalled adapter is installed in the
/// caller's environment, so launching it with an unrelated `python3` off PATH
/// otherwise dies mid-run with an opaque ModuleNotFoundError (FABRIC-86).
fn fallback_interpreter(
    env: &impl Fn(&str) -> Option<OsString>,
    exists: &impl Fn(&Path) -> bool,
    current_exe: Option<&Path>,
) -> PythonCommand {
    if let Some(virtual_env) = env(VIRTUAL_ENV_ENV)
        && !virtual_env.as_os_str().is_empty()
    {
        // Inside an active virtualenv, always target its interpreter. If it is
        // missing, preflight surfaces a clear PythonInterpreterUnavailable
        // rather than silently falling back to an unrelated `python3`.
        return PythonCommand {
            path: Path::new(&virtual_env).join(VENV_BIN_DIR).join(VENV_PYTHON),
            source: PythonSource::Virtualenv,
        };
    }
    if let Some(exe) = current_exe
        && let Some(dir) = exe.parent()
    {
        let candidate = dir.join(VENV_PYTHON);
        if exists(&candidate) {
            return PythonCommand {
                path: candidate,
                source: PythonSource::HostInterpreter,
            };
        }
    }
    PythonCommand {
        path: PathBuf::from(DEFAULT_PYTHON),
        source: PythonSource::DefaultPython3,
    }
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
        .map(|path| resolve_path(&plan.base_dir, path))
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
    directory: &Path,
    name: &str,
    kind: &str,
    filename: &str,
    contents: &str,
    media_type: &str,
) -> Result<()> {
    if manifest.root.is_none() {
        return Ok(());
    }
    let path = directory.join(filename);
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
    artifact_directory: &Path,
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
        artifact_directory,
        "workspace_patch",
        "patch",
        "workspace.patch",
        &patch,
        "text/x-diff",
    )?;
    write_artifact(
        manifest,
        artifact_directory,
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
    artifact_directory: &Path,
    artifacts: &mut ArtifactManifest,
) -> Result<Option<RelayRuntimeConfig>> {
    let Some(telemetry) = plan.telemetry_plan.as_ref() else {
        return Ok(None);
    };
    if !telemetry.relay_enabled {
        return Ok(None);
    }
    if artifacts.root.is_none() {
        return Ok(None);
    }
    let relay_config = serde_json::json!({
        "schema_version": "fabric.relay/v1alpha1",
        "relay": {
            "enabled": true,
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
            "harness": harness(plan),
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
        artifact_directory,
        "relay_config",
        "telemetry_config",
        "relay-config.json",
        &contents,
        "application/json",
    )?;
    let path = absolute_path(artifact_directory.join("relay-config.json"))?;
    Ok(Some(RelayRuntimeConfig {
        path: path.clone(),
        env: BTreeMap::from([
            ("FABRIC_RELAY_ENABLED".to_string(), "true".to_string()),
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
    metadata.insert(
        "telemetry_providers".to_string(),
        Value::Array(
            telemetry
                .providers
                .iter()
                .map(|provider| Value::String(provider.as_str().to_string()))
                .collect(),
        ),
    );
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
    // process-backed runner spawns a fresh `fabric` process per call, resetting it
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
