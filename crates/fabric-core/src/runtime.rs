// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime invocation helpers.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs::File;
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, LazyLock, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::config::{
    ADAPTER_LIFECYCLE_CONTRACT_VERSION, AdapterKind, CapabilityKind, CapabilityPlan,
    CapabilityTarget, ControlLocation, EnvironmentOwnership, FabricConfig, RunPlan, TelemetryPlan,
};
use crate::error::{FabricError, Result};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
const ADAPTER_PYTHON_ENV: &str = "ADAPTER_PYTHON";
const VIRTUAL_ENV_ENV: &str = "VIRTUAL_ENV";
const LOCAL_HOST_START_TIMEOUT: Duration = Duration::from_secs(90);
// Protocol liveness backstop. Adapters remain responsible for normal request
// timeouts and should return a normalized response before this bound.
const LOCAL_HOST_INVOKE_TIMEOUT: Duration = Duration::from_secs(60 * 60);
const LOCAL_HOST_STOP_TIMEOUT: Duration = Duration::from_secs(10);
const LOCAL_HOST_EXIT_GRACE: Duration = Duration::from_secs(2);
const LOCAL_HOST_DIAGNOSTIC_LIMIT: usize = 16 * 1024;

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
static LOCAL_HOSTS: LazyLock<Mutex<BTreeMap<String, Arc<Mutex<LocalAdapterHost>>>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

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
    /// Stable agent name.
    pub agent_name: String,
    /// Absolute base directory used to resolve relative Fabric paths.
    pub base_dir: PathBuf,
    /// Complete typed Fabric config.
    pub config: FabricConfig,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AdapterLifecycleOperation {
    Start,
    Invoke,
    Stop,
}

impl AdapterLifecycleOperation {
    fn as_str(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::Invoke => "invoke",
            Self::Stop => "stop",
        }
    }

    fn error_stage(self) -> ErrorStage {
        match self {
            Self::Start => ErrorStage::Start,
            Self::Invoke => ErrorStage::Invoke,
            Self::Stop => ErrorStage::Stop,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct AdapterLifecycleStart {
    agent_name: String,
    base_dir: PathBuf,
    config: FabricConfig,
    runtime_context: RuntimeContext,
    capability_plan: CapabilityPlan,
    #[serde(skip_serializing_if = "Option::is_none")]
    telemetry_plan: Option<TelemetryPlan>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct AdapterLifecycleStop {
    runtime_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "operation", content = "payload", rename_all = "snake_case")]
enum AdapterLifecycleRequestKind {
    Start(AdapterLifecycleStart),
    Invoke(AdapterInvocation),
    Stop(AdapterLifecycleStop),
}

impl AdapterLifecycleRequestKind {
    fn operation(&self) -> AdapterLifecycleOperation {
        match self {
            Self::Start(_) => AdapterLifecycleOperation::Start,
            Self::Invoke(_) => AdapterLifecycleOperation::Invoke,
            Self::Stop(_) => AdapterLifecycleOperation::Stop,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct AdapterLifecycleRequest {
    contract_version: String,
    #[serde(flatten)]
    request: AdapterLifecycleRequestKind,
}

impl AdapterLifecycleRequest {
    fn new(request: AdapterLifecycleRequestKind) -> Self {
        Self {
            contract_version: ADAPTER_LIFECYCLE_CONTRACT_VERSION.to_string(),
            request,
        }
    }

    fn operation(&self) -> AdapterLifecycleOperation {
        self.request.operation()
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum AdapterLifecycleOutcome {
    Succeeded {
        #[serde(default)]
        output: Value,
    },
    Failed {
        error: ErrorInfo,
    },
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct AdapterLifecycleResponse {
    contract_version: String,
    operation: AdapterLifecycleOperation,
    outcome: AdapterLifecycleOutcome,
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
struct LocalHostAdapter;

#[derive(Debug, Clone)]
struct RelayRuntimeConfig {
    path: PathBuf,
    env: BTreeMap<String, String>,
}

enum RelayConfigCorrelation<'a> {
    Runtime,
    Invocation {
        invocation: &'a InvocationHandle,
        request: &'a RunRequest,
    },
}

struct LocalAdapterHost {
    child: Child,
    stdin: ChildStdin,
    responses: Receiver<std::result::Result<String, String>>,
    command: String,
    runtime_dir: PathBuf,
    stderr_path: PathBuf,
    stderr_offset: usize,
    artifacts: ArtifactManifest,
    relay_config: Option<RelayRuntimeConfig>,
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
    if uses_local_host(plan)? {
        return LocalHostAdapter.start(plan, environment);
    }
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
    validate_blocked_tools_support(plan)?;
    validate_runtime_handle(plan, runtime)?;
    if uses_local_host(plan)? {
        return LocalHostAdapter.invoke(plan, runtime, request);
    }
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
    validate_runtime_handle(plan, runtime)?;
    if uses_local_host(plan)? {
        return LocalHostAdapter.stop(runtime);
    }
    match runtime.adapter_kind {
        AdapterKind::Process => ProcessAdapter.stop(runtime),
        AdapterKind::Python => PythonAdapter.stop(runtime),
        adapter_kind => Err(FabricError::UnsupportedRuntimeAdapter {
            harness: runtime.harness.clone(),
            adapter_kind,
        }),
    }
}

fn uses_local_host(plan: &RunPlan) -> Result<bool> {
    let local_host = plan
        .adapter_descriptor
        .as_ref()
        .and_then(|adapter| adapter.descriptor.runtime.local_host.as_ref());
    let Some(local_host) = local_host else {
        return Ok(false);
    };
    if local_host.contract_version != ADAPTER_LIFECYCLE_CONTRACT_VERSION {
        return Err(FabricError::AdapterDescriptorUnsupported {
            adapter_id: adapter_id(plan).unwrap_or_else(|| harness(plan)),
            field: "runtime.local_host.contract_version",
            value: local_host.contract_version.clone(),
        });
    }
    Ok(true)
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

impl RuntimeAdapter for LocalHostAdapter {
    fn start(&self, plan: &RunPlan, environment: EnvironmentHandle) -> Result<RuntimeHandle> {
        if environment.provider != "local" {
            return Err(FabricError::UnsupportedEnvironmentProvider {
                provider: environment.provider,
                adapter_kind: adapter_kind(plan),
            });
        }
        if adapter_kind(plan) != AdapterKind::Python {
            return Err(FabricError::UnsupportedRuntimeAdapter {
                harness: harness(plan),
                adapter_kind: adapter_kind(plan),
            });
        }
        preflight_python_adapter(plan)?;

        let runtime_id = new_id("runtime");
        let runtime_binding = runtime_binding(&runtime_id, plan, &environment)?;
        let runtime = RuntimeHandle {
            runtime_id,
            runtime_binding,
            agent_name: plan.agent_name.clone(),
            harness: harness(plan),
            adapter_kind: adapter_kind(plan),
            adapter_id: adapter_id(plan),
            environment,
        };

        let start_request = RunRequest {
            request_id: new_id("runtime-start-request"),
            ..RunRequest::default()
        };
        let start_invocation = InvocationHandle {
            invocation_id: new_id("runtime-start"),
            request_id: start_request.request_id.clone(),
            runtime_id: runtime.runtime_id.clone(),
        };
        let mut artifacts = artifact_manifest(plan)?;
        let fabric_home = prepare_fabric_home(&artifacts, &runtime, &start_invocation)?;
        let relay_config = prepare_relay_runtime_config(
            plan,
            &runtime,
            RelayConfigCorrelation::Runtime,
            &fabric_home,
            &mut artifacts,
        )?;
        let AdapterInvocation {
            agent_name,
            base_dir,
            config,
            runtime_context,
            capability_plan,
            telemetry_plan,
            ..
        } = adapter_invocation(
            plan,
            &runtime,
            &start_invocation,
            &start_request,
            &artifacts,
            relay_config.as_ref(),
        )?;
        let request = AdapterLifecycleRequest::new(AdapterLifecycleRequestKind::Start(
            AdapterLifecycleStart {
                agent_name,
                base_dir,
                config,
                runtime_context,
                capability_plan,
                telemetry_plan,
            },
        ));
        let mut host = spawn_local_host(plan, &runtime, artifacts, relay_config)?;
        if let Err(error) = exchange_lifecycle_message(
            &mut host,
            &runtime.runtime_id,
            &request,
            LOCAL_HOST_START_TIMEOUT,
        ) {
            let _ = terminate_local_host(&mut host);
            let _ = remove_local_host_files(&host);
            return Err(error);
        }

        local_hosts().insert(runtime.runtime_id.clone(), Arc::new(Mutex::new(host)));
        Ok(runtime)
    }

    fn invoke(
        &self,
        plan: &RunPlan,
        runtime: &RuntimeHandle,
        request: RunRequest,
    ) -> Result<RunResult> {
        run_local_host_adapter(plan, runtime, request)
    }

    fn stop(&self, runtime: &RuntimeHandle) -> Result<Vec<FabricEvent>> {
        let Some(host) = local_hosts().remove(&runtime.runtime_id) else {
            return Ok(vec![local_host_stop_event(runtime, true, false)]);
        };
        let mut host = host.lock().unwrap_or_else(|error| error.into_inner());
        let request =
            AdapterLifecycleRequest::new(AdapterLifecycleRequestKind::Stop(AdapterLifecycleStop {
                runtime_id: runtime.runtime_id.clone(),
            }));
        let result = exchange_lifecycle_message(
            &mut host,
            &runtime.runtime_id,
            &request,
            LOCAL_HOST_STOP_TIMEOUT,
        );
        let termination = terminate_local_host(&mut host);
        let diagnostics = local_host_diagnostics(&host);
        let removal = remove_local_host_files(&host);
        let host_crashed = matches!(
            &result,
            Err(FabricError::AdapterLifecycleOperation { code, .. })
                if code == "host_crashed"
        );
        if !host_crashed {
            result?;
        }
        termination.map_err(|source| {
            lifecycle_error(
                AdapterLifecycleOperation::Stop,
                &runtime.runtime_id,
                "host_termination_failed",
                format!("persistent local adapter host could not be terminated: {source}"),
                diagnostics,
            )
        })?;
        removal.map_err(|source| {
            lifecycle_error(
                AdapterLifecycleOperation::Stop,
                &runtime.runtime_id,
                "host_cleanup_failed",
                format!("persistent local adapter host files could not be removed: {source}"),
                "",
            )
        })?;

        #[cfg(test)]
        TEST_STOPPED_AGENTS
            .lock()
            .expect("stop tracker")
            .push(runtime.agent_name.clone());
        Ok(vec![local_host_stop_event(runtime, false, host_crashed)])
    }
}

fn run_local_host_adapter(
    plan: &RunPlan,
    runtime: &RuntimeHandle,
    request: RunRequest,
) -> Result<RunResult> {
    run_local_host_adapter_with_timeout(plan, runtime, request, LOCAL_HOST_INVOKE_TIMEOUT)
}

fn run_local_host_adapter_with_timeout(
    plan: &RunPlan,
    runtime: &RuntimeHandle,
    mut request: RunRequest,
    invoke_timeout: Duration,
) -> Result<RunResult> {
    if request.request_id.is_empty() {
        request.request_id = new_id("request");
    }
    let invocation = InvocationHandle {
        invocation_id: new_id("invocation"),
        request_id: request.request_id.clone(),
        runtime_id: runtime.runtime_id.clone(),
    };
    let host = local_hosts()
        .get(&runtime.runtime_id)
        .cloned()
        .ok_or_else(|| {
            lifecycle_error(
                AdapterLifecycleOperation::Invoke,
                &runtime.runtime_id,
                "host_unavailable",
                "persistent local adapter host is not active",
                "",
            )
        })?;

    let exchange_result = {
        let mut host = host.lock().unwrap_or_else(|error| error.into_inner());
        let artifacts = host.artifacts.clone();
        let relay_config = host.relay_config.clone();
        let fabric_home = prepare_fabric_home(&artifacts, runtime, &invocation)?;
        let adapter_invocation = adapter_invocation(
            plan,
            runtime,
            &invocation,
            &request,
            &artifacts,
            relay_config.as_ref(),
        )?;
        let adapter_payload = serde_json::to_string_pretty(&adapter_invocation)
            .map_err(FabricError::SerializeJson)?;
        let fabric_invocation = write_fabric_invocation(&fabric_home, &adapter_payload)?;
        let lifecycle_request =
            AdapterLifecycleRequest::new(AdapterLifecycleRequestKind::Invoke(adapter_invocation));
        match exchange_lifecycle_message(
            &mut host,
            &runtime.runtime_id,
            &lifecycle_request,
            invoke_timeout,
        ) {
            Ok(output) => {
                let stderr = take_local_host_stderr(&mut host);
                Ok((
                    output,
                    stderr,
                    host.command.clone(),
                    host.child.id(),
                    artifacts,
                    relay_config,
                    fabric_home,
                    fabric_invocation,
                ))
            }
            Err(error) => Err(error),
        }
    };
    let (
        output,
        stderr,
        host_command,
        host_pid,
        mut artifacts,
        relay_config,
        fabric_home,
        fabric_invocation,
    ) = match exchange_result {
        Ok(result) => result,
        Err(error) => {
            if matches!(
                &error,
                FabricError::AdapterLifecycleOperation { code, .. } if code == "host_timeout"
            ) {
                invalidate_timed_out_local_host(&runtime.runtime_id, &host);
            }
            return Err(error);
        }
    };

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
            ("host_pid".to_string(), Value::from(host_pid)),
        ]),
    )];
    events.push(event_with_metadata(
        "invocation_start",
        format!("invoking persistent local host for {}", harness(plan)),
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
    let (status, error) = adapter_output_status(&output);
    events.push(event_with_metadata(
        "invocation_end",
        format!("persistent local host completed with status {status:?}"),
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
    let mut stdout = serde_json::to_string(&output).map_err(FabricError::SerializeJson)?;
    stdout.push('\n');
    write_artifact(
        &mut artifacts,
        &fabric_home,
        "stdout",
        "log",
        "stdout.txt",
        &stdout,
        "text/plain",
    )?;
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
    promote_relay_artifacts_to_manifest(&output, &mut artifacts);

    let metadata = BTreeMap::from([
        (
            "adapter_runner".to_string(),
            Value::String("persistent_local_host".to_string()),
        ),
        ("host_command".to_string(), Value::String(host_command)),
        ("host_pid".to_string(), Value::from(host_pid)),
        (
            "fabric_home".to_string(),
            Value::String(fabric_home.to_string_lossy().into_owned()),
        ),
        (
            "fabric_invocation".to_string(),
            Value::String(fabric_invocation.to_string_lossy().into_owned()),
        ),
        (
            "environment_provider".to_string(),
            Value::String(runtime.environment.provider.clone()),
        ),
    ]);
    Ok(RunResult {
        agent_name: plan.agent_name.clone(),
        harness: harness(plan),
        adapter_kind: adapter_kind(plan),
        adapter_id: adapter_id(plan),
        runtime_id: invocation.runtime_id,
        invocation_id: invocation.invocation_id,
        request_id: request.request_id,
        status,
        output,
        error,
        artifacts,
        telemetry: telemetry_ref(plan, relay_config.as_ref()),
        events,
        metadata,
    })
}

fn invalidate_timed_out_local_host(runtime_id: &str, expected_host: &Arc<Mutex<LocalAdapterHost>>) {
    {
        let mut hosts = local_hosts();
        if hosts
            .get(runtime_id)
            .is_some_and(|host| Arc::ptr_eq(host, expected_host))
        {
            hosts.remove(runtime_id);
        }
    }

    let mut host = expected_host
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let _ = terminate_local_host(&mut host);
    let _ = remove_local_host_files(&host);
}

fn adapter_output_status(output: &Value) -> (RunStatus, Option<ErrorInfo>) {
    let failed = output
        .as_object()
        .and_then(|output| output.get("failed"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if !failed {
        return (RunStatus::Succeeded, None);
    }

    let reported = output
        .as_object()
        .and_then(|output| output.get("error"))
        .and_then(Value::as_object);
    let code = reported
        .and_then(|error| error.get("code"))
        .and_then(Value::as_str)
        .unwrap_or("adapter_reported_failure")
        .to_string();
    let message = reported
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("adapter reported an invocation failure")
        .to_string();
    let retryable = reported
        .and_then(|error| error.get("retryable"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let metadata = reported
        .and_then(|error| error.get("metadata"))
        .and_then(Value::as_object)
        .map(|metadata| {
            metadata
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        })
        .unwrap_or_default();
    (
        RunStatus::Failed,
        Some(ErrorInfo {
            stage: ErrorStage::Invoke,
            code,
            message,
            retryable,
            metadata,
        }),
    )
}

fn local_host_stop_event(
    runtime: &RuntimeHandle,
    already_stopped: bool,
    host_crashed: bool,
) -> FabricEvent {
    event_with_metadata(
        "runtime_stop",
        format!("stopped runtime {}", runtime.runtime_id),
        BTreeMap::from([
            (
                "runtime_id".to_string(),
                Value::String(runtime.runtime_id.clone()),
            ),
            ("already_stopped".to_string(), Value::Bool(already_stopped)),
            ("host_crashed".to_string(), Value::Bool(host_crashed)),
        ]),
    )
}

fn local_hosts() -> std::sync::MutexGuard<'static, BTreeMap<String, Arc<Mutex<LocalAdapterHost>>>> {
    LOCAL_HOSTS
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

fn spawn_local_host(
    plan: &RunPlan,
    runtime: &RuntimeHandle,
    artifacts: ArtifactManifest,
    relay_config: Option<RelayRuntimeConfig>,
) -> Result<LocalAdapterHost> {
    let runtime_dir = std::env::temp_dir()
        .join("nemo-fabric")
        .join("hosts")
        .join(&runtime.runtime_id);
    std::fs::create_dir_all(&runtime_dir).map_err(|source| FabricError::Write {
        path: runtime_dir.clone(),
        source,
    })?;
    let stderr_path = runtime_dir.join("host.stderr.log");
    let stderr = File::create(&stderr_path).map_err(|source| FabricError::Write {
        path: stderr_path.clone(),
        source,
    })?;
    let (mut command, command_display) = match local_host_command(plan, runtime) {
        Ok(command) => command,
        Err(error) => {
            let _ = std::fs::remove_dir_all(&runtime_dir);
            return Err(error);
        }
    };
    command
        .env(
            "FABRIC_ADAPTER_LIFECYCLE_CONTRACT",
            ADAPTER_LIFECYCLE_CONTRACT_VERSION,
        )
        .env("FABRIC_RUNTIME_ID", &runtime.runtime_id)
        .env("FABRIC_HOME", &runtime_dir)
        .envs(relay_env(&relay_config))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::from(stderr));
    if let Some(root) = artifacts.root.as_ref() {
        command.env("FABRIC_ARTIFACTS", root);
    }
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(source) => {
            let _ = std::fs::remove_dir_all(&runtime_dir);
            return Err(FabricError::ProcessRunner {
                command: command_display,
                source,
            });
        }
    };
    let Some(stdin) = child.stdin.take() else {
        let _ = child.kill();
        let _ = child.wait();
        let _ = std::fs::remove_dir_all(&runtime_dir);
        return Err(lifecycle_error(
            AdapterLifecycleOperation::Start,
            &runtime.runtime_id,
            "host_io",
            "persistent local adapter host stdin was not available",
            "",
        ));
    };
    let Some(stdout) = child.stdout.take() else {
        let _ = child.kill();
        let _ = child.wait();
        let _ = std::fs::remove_dir_all(&runtime_dir);
        return Err(lifecycle_error(
            AdapterLifecycleOperation::Start,
            &runtime.runtime_id,
            "host_io",
            "persistent local adapter host stdout was not available",
            "",
        ));
    };
    let (sender, responses) = mpsc::channel();
    if let Err(source) = thread::Builder::new()
        .name(format!("fabric-host-{}", runtime.runtime_id))
        .spawn(move || {
            let mut stdout = BufReader::new(stdout);
            loop {
                let mut line = String::new();
                match stdout.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        while line.ends_with(['\n', '\r']) {
                            line.pop();
                        }
                        if sender.send(Ok(line)).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = sender.send(Err(error.to_string()));
                        break;
                    }
                }
            }
        })
    {
        let _ = child.kill();
        let _ = child.wait();
        let _ = std::fs::remove_dir_all(&runtime_dir);
        return Err(FabricError::ProcessRunner {
            command: command_display,
            source,
        });
    }
    Ok(LocalAdapterHost {
        child,
        stdin,
        responses,
        command: command_display,
        runtime_dir,
        stderr_path,
        stderr_offset: 0,
        artifacts,
        relay_config,
    })
}

fn local_host_command(plan: &RunPlan, runtime: &RuntimeHandle) -> Result<(Command, String)> {
    let settings = parse_python_settings(plan)?;
    let python = resolve_python_command(&plan.base_dir, &settings).path;
    let cwd = settings
        .cwd
        .as_ref()
        .map(|path| resolve_path(&plan.base_dir, path))
        .or_else(|| runtime.environment.workspace.clone())
        .unwrap_or_else(|| plan.base_dir.clone());
    let mut command = Command::new(&python);
    command
        .arg("-m")
        .arg(&settings.module)
        .args(&settings.args)
        .current_dir(cwd)
        .envs(&settings.env);
    Ok((
        command,
        format!("{} -m {}", python.to_string_lossy(), settings.module),
    ))
}

fn exchange_lifecycle_message(
    host: &mut LocalAdapterHost,
    runtime_id: &str,
    request: &AdapterLifecycleRequest,
    timeout: Duration,
) -> Result<Value> {
    let operation = request.operation();
    if let Some(status) = host.child.try_wait().map_err(|source| {
        lifecycle_error(
            operation,
            runtime_id,
            "host_io",
            format!("failed to inspect persistent local adapter host: {source}"),
            local_host_diagnostics(host),
        )
    })? {
        return Err(lifecycle_error(
            operation,
            runtime_id,
            "host_crashed",
            format!(
                "persistent local adapter host exited before {} ({status})",
                operation.as_str()
            ),
            local_host_diagnostics(host),
        ));
    }
    let mut message = serde_json::to_string(request).map_err(FabricError::SerializeJson)?;
    message.push('\n');
    if let Err(source) = host
        .stdin
        .write_all(message.as_bytes())
        .and_then(|()| host.stdin.flush())
    {
        let code = match host.child.try_wait() {
            Ok(Some(_)) => "host_crashed",
            _ => "host_io",
        };
        return Err(lifecycle_error(
            operation,
            runtime_id,
            code,
            format!(
                "failed to send {} to persistent local adapter host: {source}",
                operation.as_str()
            ),
            local_host_diagnostics(host),
        ));
    }

    let line = match host.responses.recv_timeout(timeout) {
        Ok(line) => line,
        Err(RecvTimeoutError::Timeout) => {
            return Err(lifecycle_error(
                operation,
                runtime_id,
                "host_timeout",
                format!(
                    "persistent local adapter host did not complete {} within {} ms",
                    operation.as_str(),
                    timeout.as_millis()
                ),
                local_host_diagnostics(host),
            ));
        }
        Err(RecvTimeoutError::Disconnected) => {
            return Err(lifecycle_error(
                operation,
                runtime_id,
                "host_crashed",
                format!(
                    "persistent local adapter host exited while processing {}",
                    operation.as_str()
                ),
                local_host_diagnostics(host),
            ));
        }
    }
    .map_err(|message| {
        lifecycle_error(
            operation,
            runtime_id,
            "host_io",
            message,
            local_host_diagnostics(host),
        )
    })?;
    let response: AdapterLifecycleResponse = serde_json::from_str(&line).map_err(|source| {
        lifecycle_error(
            operation,
            runtime_id,
            "protocol_error",
            format!("invalid lifecycle response: {source}"),
            local_host_diagnostics(host),
        )
    })?;
    if response.contract_version != ADAPTER_LIFECYCLE_CONTRACT_VERSION {
        return Err(lifecycle_error(
            operation,
            runtime_id,
            "protocol_version_mismatch",
            format!(
                "expected lifecycle contract `{ADAPTER_LIFECYCLE_CONTRACT_VERSION}` but host returned `{}`",
                response.contract_version
            ),
            local_host_diagnostics(host),
        ));
    }
    if response.operation != operation {
        return Err(lifecycle_error(
            operation,
            runtime_id,
            "protocol_error",
            format!(
                "expected `{}` response but host returned `{}`",
                operation.as_str(),
                response.operation.as_str()
            ),
            local_host_diagnostics(host),
        ));
    }
    match response.outcome {
        AdapterLifecycleOutcome::Succeeded { output } => Ok(output),
        AdapterLifecycleOutcome::Failed { error } => {
            if error.stage != operation.error_stage() {
                return Err(lifecycle_error(
                    operation,
                    runtime_id,
                    "protocol_error",
                    format!(
                        "{} failure reported the wrong lifecycle stage `{:?}`",
                        operation.as_str(),
                        error.stage
                    ),
                    local_host_diagnostics(host),
                ));
            }
            let mut diagnostics = local_host_diagnostics(host);
            if !error.metadata.is_empty()
                && let Ok(metadata) = serde_json::to_string(&error.metadata)
            {
                if !diagnostics.is_empty() {
                    diagnostics.push('\n');
                }
                diagnostics.push_str("adapter metadata: ");
                diagnostics.push_str(&metadata);
            }
            Err(lifecycle_error(
                operation,
                runtime_id,
                error.code,
                error.message,
                diagnostics,
            ))
        }
    }
}

fn lifecycle_error(
    operation: AdapterLifecycleOperation,
    runtime_id: &str,
    code: impl Into<String>,
    message: impl Into<String>,
    diagnostics: impl Into<String>,
) -> FabricError {
    FabricError::AdapterLifecycleOperation {
        operation: operation.as_str(),
        runtime_id: runtime_id.to_string(),
        code: code.into(),
        message: message.into(),
        diagnostics: diagnostics.into(),
    }
}

fn local_host_diagnostics(host: &LocalAdapterHost) -> String {
    let Ok(bytes) = std::fs::read(&host.stderr_path) else {
        return String::new();
    };
    let start = bytes.len().saturating_sub(LOCAL_HOST_DIAGNOSTIC_LIMIT);
    String::from_utf8_lossy(&bytes[start..]).trim().to_string()
}

fn take_local_host_stderr(host: &mut LocalAdapterHost) -> String {
    let Ok(bytes) = std::fs::read(&host.stderr_path) else {
        return String::new();
    };
    let start = host.stderr_offset.min(bytes.len());
    host.stderr_offset = bytes.len();
    String::from_utf8_lossy(&bytes[start..]).into_owned()
}

fn terminate_local_host(host: &mut LocalAdapterHost) -> std::io::Result<()> {
    let deadline = Instant::now() + LOCAL_HOST_EXIT_GRACE;
    loop {
        if host.child.try_wait()?.is_some() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }

    if let Err(source) = host.child.kill()
        && host.child.try_wait()?.is_none()
    {
        return Err(source);
    }
    host.child.wait()?;
    Ok(())
}

fn remove_local_host_files(host: &LocalAdapterHost) -> std::io::Result<()> {
    match std::fs::remove_dir_all(&host.runtime_dir) {
        Ok(()) => Ok(()),
        Err(source) if source.kind() == ErrorKind::NotFound => Ok(()),
        Err(source) => Err(source),
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
        RelayConfigCorrelation::Invocation {
            invocation: &invocation,
            request: &request,
        },
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
        RelayConfigCorrelation::Invocation {
            invocation: &invocation,
            request: &request,
        },
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
    Ok(AdapterInvocation {
        agent_name: plan.agent_name.clone(),
        base_dir: absolute_path(plan.base_dir.clone())?,
        config: plan.config.clone(),
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
    correlation: RelayConfigCorrelation<'_>,
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
    let mut fabric = serde_json::json!({
        "agent_name": plan.agent_name.clone(),
        "harness": harness(plan),
        "adapter_id": adapter_id(plan),
        "runtime_id": runtime.runtime_id.clone(),
        "adapter_outputs": telemetry.adapter_outputs.clone(),
    });
    if let RelayConfigCorrelation::Invocation {
        invocation,
        request,
    } = correlation
    {
        fabric["invocation_id"] = Value::String(invocation.invocation_id.clone());
        fabric["request_id"] = Value::String(request.request_id.clone());
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
        "fabric": fabric,
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
