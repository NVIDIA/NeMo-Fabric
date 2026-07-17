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
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::config::{
    ADAPTER_LIFECYCLE_CONTRACT_VERSION, AdapterKind, CapabilityKind, CapabilityPlan,
    CapabilityTarget, ControlLocation, EffectiveConfig, EnvironmentOwnership, ExecutionStrategy,
    RunPlan, RuntimeCapabilities, TelemetryPlan,
};
use crate::error::{FabricError, Result};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
const ADAPTER_PYTHON_ENV: &str = "ADAPTER_PYTHON";
const VIRTUAL_ENV_ENV: &str = "VIRTUAL_ENV";
const PERSISTENT_HOST_START_TIMEOUT: Duration = Duration::from_secs(10);
const PERSISTENT_HOST_STOP_TIMEOUT: Duration = Duration::from_secs(5);
const PERSISTENT_HOST_DIAGNOSTIC_LIMIT: usize = 16 * 1024;

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
static PERSISTENT_HOSTS: LazyLock<Mutex<BTreeMap<String, Arc<Mutex<PersistentHost>>>>> =
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
    /// Ordered profiles applied to this run.
    pub profiles: Vec<String>,
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
    /// Execution strategy selected for this runtime.
    pub execution_strategy: ExecutionStrategy,
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
    /// Execution strategy selected during planning.
    pub execution_strategy: ExecutionStrategy,
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

/// Operation exchanged over the versioned persistent-host adapter protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AdapterLifecycleOperation {
    /// Initialize one adapter-owned host for a Fabric runtime.
    Start,
    /// Execute one invocation against an initialized host.
    Invoke,
    /// Release the host and all runtime-owned resources.
    Stop,
}

impl AdapterLifecycleOperation {
    /// Stable serialized operation name.
    pub fn as_str(self) -> &'static str {
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

/// Start payload sent once when Fabric creates a persistent adapter host.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AdapterLifecycleStart {
    /// Runtime identity and prepared environment owned by this host.
    pub runtime: RuntimeHandle,
    /// Merged agent config and provenance for this runtime.
    pub effective_config: EffectiveConfig,
    /// Capability routing selected during planning.
    #[serde(default)]
    pub capability_plan: CapabilityPlan,
    /// Lifecycle capabilities selected during planning.
    #[serde(default)]
    pub capabilities: RuntimeCapabilities,
    /// Telemetry routing selected during planning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry_plan: Option<TelemetryPlan>,
}

/// Stop payload sent once when Fabric releases a persistent adapter host.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AdapterLifecycleStop {
    /// Runtime being stopped.
    pub runtime_id: String,
}

/// Typed operation payload carried by an adapter lifecycle request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "operation", content = "payload", rename_all = "snake_case")]
pub enum AdapterLifecycleRequestKind {
    /// Initialize the host.
    Start(AdapterLifecycleStart),
    /// Execute one invocation.
    Invoke(AdapterInvocation),
    /// Stop the host.
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

/// One newline-delimited request sent to a persistent adapter host.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AdapterLifecycleRequest {
    /// Lifecycle protocol version.
    pub contract_version: String,
    /// Typed lifecycle operation and payload.
    #[serde(flatten)]
    pub request: AdapterLifecycleRequestKind,
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

/// Outcome returned by a persistent adapter host lifecycle operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum AdapterLifecycleOutcome {
    /// The operation completed successfully.
    Succeeded {
        /// Operation output. Only invoke normally returns a non-null value.
        #[serde(default)]
        output: Value,
    },
    /// The operation failed with normalized lifecycle diagnostics.
    Failed {
        /// Structured failure reported by the adapter host.
        error: ErrorInfo,
    },
}

/// One newline-delimited response returned by a persistent adapter host.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AdapterLifecycleResponse {
    /// Lifecycle protocol version.
    pub contract_version: String,
    /// Operation completed by this response.
    pub operation: AdapterLifecycleOperation,
    /// Normalized success or failure outcome.
    pub outcome: AdapterLifecycleOutcome,
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
struct PersistentHostAdapter;

#[derive(Debug, Clone)]
struct RelayRuntimeConfig {
    path: PathBuf,
    env: BTreeMap<String, String>,
}

struct PersistentHost {
    child: Child,
    stdin: ChildStdin,
    responses: Receiver<std::result::Result<String, String>>,
    command: String,
    runtime_dir: PathBuf,
    stderr_path: PathBuf,
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
    validate_blocked_tools_support(plan)?;
    let environment = prepare_environment(plan)?;
    match plan.execution_strategy {
        ExecutionStrategy::PersistentLocalHost => {
            return PersistentHostAdapter.start(plan, environment);
        }
        ExecutionStrategy::RemoteService => return remote_service_unavailable(plan),
        ExecutionStrategy::ProcessPerInvocation => {}
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
    match plan.execution_strategy {
        ExecutionStrategy::PersistentLocalHost => {
            return PersistentHostAdapter.invoke(plan, runtime, request);
        }
        ExecutionStrategy::RemoteService => return remote_service_unavailable(plan),
        ExecutionStrategy::ProcessPerInvocation => {}
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
    match plan.execution_strategy {
        ExecutionStrategy::PersistentLocalHost => return PersistentHostAdapter.stop(runtime),
        ExecutionStrategy::RemoteService => return remote_service_unavailable(plan),
        ExecutionStrategy::ProcessPerInvocation => {}
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

fn remote_service_unavailable<T>(plan: &RunPlan) -> Result<T> {
    Err(FabricError::RuntimeStrategyUnavailable {
        adapter_id: adapter_id(plan).unwrap_or_else(|| plan.config.harness.adapter_id.clone()),
        strategy: ExecutionStrategy::RemoteService,
        reason: "the adapter does not provide a remote lifecycle transport",
    })
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
    expect_runtime_field(
        runtime,
        "execution_strategy",
        plan.execution_strategy.as_str(),
        runtime.execution_strategy.as_str(),
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
            execution_strategy: plan.execution_strategy,
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
            execution_strategy: plan.execution_strategy,
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

impl RuntimeAdapter for PersistentHostAdapter {
    fn start(&self, plan: &RunPlan, environment: EnvironmentHandle) -> Result<RuntimeHandle> {
        if environment.provider != "local" {
            return Err(FabricError::UnsupportedEnvironmentProvider {
                provider: environment.provider,
                adapter_kind: adapter_kind(plan),
            });
        }
        match adapter_kind(plan) {
            AdapterKind::Python => preflight_python_adapter(plan)?,
            AdapterKind::Process => {}
            adapter_kind => {
                return Err(FabricError::UnsupportedRuntimeAdapter {
                    harness: harness(plan),
                    adapter_kind,
                });
            }
        }

        let runtime_id = new_id("runtime");
        let runtime_binding = runtime_binding(&runtime_id, plan, &environment)?;
        let runtime = RuntimeHandle {
            runtime_id,
            runtime_binding,
            agent_name: plan.agent_name.clone(),
            harness: harness(plan),
            adapter_kind: adapter_kind(plan),
            adapter_id: adapter_id(plan),
            execution_strategy: plan.execution_strategy,
            environment,
        };
        let mut host = spawn_persistent_host(plan, &runtime)?;
        let request = AdapterLifecycleRequest::new(AdapterLifecycleRequestKind::Start(
            AdapterLifecycleStart {
                runtime: runtime.clone(),
                effective_config: adapter_effective_config(plan)?,
                capability_plan: plan.capability_plan.clone(),
                capabilities: plan.capabilities.clone(),
                telemetry_plan: plan.telemetry_plan.clone(),
            },
        ));
        if let Err(error) = exchange_lifecycle_message(
            &mut host,
            &runtime.runtime_id,
            &request,
            Some(PERSISTENT_HOST_START_TIMEOUT),
        ) {
            terminate_persistent_host(&mut host);
            remove_persistent_host_files(&host);
            return Err(error);
        }

        persistent_hosts().insert(runtime.runtime_id.clone(), Arc::new(Mutex::new(host)));
        Ok(runtime)
    }

    fn invoke(
        &self,
        plan: &RunPlan,
        runtime: &RuntimeHandle,
        request: RunRequest,
    ) -> Result<RunResult> {
        run_persistent_host_adapter(plan, runtime, request)
    }

    fn stop(&self, runtime: &RuntimeHandle) -> Result<Vec<FabricEvent>> {
        let Some(host) = persistent_hosts().remove(&runtime.runtime_id) else {
            return Ok(vec![persistent_host_stop_event(runtime, true)]);
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
            Some(PERSISTENT_HOST_STOP_TIMEOUT),
        );
        terminate_persistent_host(&mut host);
        remove_persistent_host_files(&host);
        result?;

        #[cfg(test)]
        TEST_STOPPED_AGENTS
            .lock()
            .expect("stop tracker")
            .push(runtime.agent_name.clone());
        Ok(vec![persistent_host_stop_event(runtime, false)])
    }
}

fn run_persistent_host_adapter(
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
    let adapter_invocation = adapter_invocation(
        plan,
        runtime,
        &invocation,
        &request,
        &artifacts,
        relay_config.as_ref(),
    )?;
    let adapter_payload =
        serde_json::to_string_pretty(&adapter_invocation).map_err(FabricError::SerializeJson)?;
    let fabric_invocation = write_fabric_invocation(&fabric_home, &adapter_payload)?;
    let lifecycle_request =
        AdapterLifecycleRequest::new(AdapterLifecycleRequestKind::Invoke(adapter_invocation));

    let host = persistent_hosts()
        .get(&runtime.runtime_id)
        .cloned()
        .ok_or_else(|| {
            lifecycle_error(
                AdapterLifecycleOperation::Invoke,
                &runtime.runtime_id,
                "host_unavailable",
                "persistent adapter host is not active",
                "",
            )
        })?;
    let (output, host_command) = {
        let mut host = host.lock().unwrap_or_else(|error| error.into_inner());
        let output =
            exchange_lifecycle_message(&mut host, &runtime.runtime_id, &lifecycle_request, None)?;
        (output, host.command.clone())
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
            (
                "execution_strategy".to_string(),
                Value::String(runtime.execution_strategy.as_str().to_string()),
            ),
        ]),
    )];
    events.push(event_with_metadata(
        "invocation_start",
        format!("invoking persistent adapter host for {}", harness(plan)),
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
        format!("persistent adapter host completed with status {status:?}"),
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
    collect_workspace_artifacts(&mut artifacts, &fabric_home, runtime, &mut events)?;
    promote_relay_artifacts_to_manifest(&output, &mut artifacts);

    let metadata = BTreeMap::from([
        (
            "adapter_runner".to_string(),
            Value::String("persistent_local_host".to_string()),
        ),
        ("host_command".to_string(), Value::String(host_command)),
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
        profiles: plan.profiles.clone(),
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

fn persistent_host_stop_event(runtime: &RuntimeHandle, already_stopped: bool) -> FabricEvent {
    event_with_metadata(
        "runtime_stop",
        format!("stopped runtime {}", runtime.runtime_id),
        BTreeMap::from([
            (
                "runtime_id".to_string(),
                Value::String(runtime.runtime_id.clone()),
            ),
            ("already_stopped".to_string(), Value::Bool(already_stopped)),
        ]),
    )
}

fn persistent_hosts() -> std::sync::MutexGuard<'static, BTreeMap<String, Arc<Mutex<PersistentHost>>>>
{
    PERSISTENT_HOSTS
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

fn spawn_persistent_host(plan: &RunPlan, runtime: &RuntimeHandle) -> Result<PersistentHost> {
    let runtime_dir = std::env::temp_dir()
        .join("nemo-fabric")
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
    let (mut command, command_display) = match persistent_host_command(plan, runtime) {
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
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::from(stderr));
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
            "persistent adapter host stdin was not available",
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
            "persistent adapter host stdout was not available",
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
    Ok(PersistentHost {
        child,
        stdin,
        responses,
        command: command_display,
        runtime_dir,
        stderr_path,
    })
}

fn persistent_host_command(plan: &RunPlan, runtime: &RuntimeHandle) -> Result<(Command, String)> {
    let cwd = |configured: Option<&Path>| {
        configured
            .map(|path| resolve_path(&plan.config_root, path))
            .or_else(|| runtime.environment.workspace.clone())
            .unwrap_or_else(|| plan.agent_root.clone())
    };
    match adapter_kind(plan) {
        AdapterKind::Python => {
            let settings = parse_python_settings(plan)?;
            let python = resolve_python_command(&plan.config_root, &settings).path;
            let mut command = Command::new(&python);
            command
                .arg("-m")
                .arg(&settings.module)
                .args(&settings.args)
                .current_dir(cwd(settings.cwd.as_deref()))
                .envs(&settings.env);
            Ok((
                command,
                format!("{} -m {}", python.to_string_lossy(), settings.module),
            ))
        }
        AdapterKind::Process => {
            let settings = parse_process_settings(plan)?;
            let command_path = resolve_command_path(
                adapter_setting_root(plan, "command"),
                Path::new(&settings.command),
            );
            let command_args = process_command_args(plan, &settings);
            let mut command = Command::new(&command_path);
            command
                .args(&command_args)
                .current_dir(cwd(settings.cwd.as_deref()))
                .envs(&settings.env);
            Ok((command, command_path.to_string_lossy().into_owned()))
        }
        adapter_kind => Err(FabricError::UnsupportedRuntimeAdapter {
            harness: harness(plan),
            adapter_kind,
        }),
    }
}

fn exchange_lifecycle_message(
    host: &mut PersistentHost,
    runtime_id: &str,
    request: &AdapterLifecycleRequest,
    timeout: Option<Duration>,
) -> Result<Value> {
    let operation = request.operation();
    if let Some(status) = host.child.try_wait().map_err(|source| {
        lifecycle_error(
            operation,
            runtime_id,
            "host_io",
            format!("failed to inspect persistent adapter host: {source}"),
            persistent_host_diagnostics(host),
        )
    })? {
        return Err(lifecycle_error(
            operation,
            runtime_id,
            "host_crashed",
            format!(
                "persistent adapter host exited before {} ({status})",
                operation.as_str()
            ),
            persistent_host_diagnostics(host),
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
                "failed to send {} to persistent adapter host: {source}",
                operation.as_str()
            ),
            persistent_host_diagnostics(host),
        ));
    }

    let line = match timeout {
        Some(timeout) => match host.responses.recv_timeout(timeout) {
            Ok(line) => line,
            Err(RecvTimeoutError::Timeout) => {
                return Err(lifecycle_error(
                    operation,
                    runtime_id,
                    "host_timeout",
                    format!(
                        "persistent adapter host did not complete {} within {} ms",
                        operation.as_str(),
                        timeout.as_millis()
                    ),
                    persistent_host_diagnostics(host),
                ));
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(lifecycle_error(
                    operation,
                    runtime_id,
                    "host_crashed",
                    format!(
                        "persistent adapter host exited while processing {}",
                        operation.as_str()
                    ),
                    persistent_host_diagnostics(host),
                ));
            }
        },
        None => host.responses.recv().map_err(|_| {
            lifecycle_error(
                operation,
                runtime_id,
                "host_crashed",
                format!(
                    "persistent adapter host exited while processing {}",
                    operation.as_str()
                ),
                persistent_host_diagnostics(host),
            )
        })?,
    }
    .map_err(|message| {
        lifecycle_error(
            operation,
            runtime_id,
            "host_io",
            message,
            persistent_host_diagnostics(host),
        )
    })?;
    let response: AdapterLifecycleResponse = serde_json::from_str(&line).map_err(|source| {
        lifecycle_error(
            operation,
            runtime_id,
            "protocol_error",
            format!("invalid lifecycle response: {source}"),
            persistent_host_diagnostics(host),
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
            persistent_host_diagnostics(host),
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
            persistent_host_diagnostics(host),
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
                    persistent_host_diagnostics(host),
                ));
            }
            let mut diagnostics = persistent_host_diagnostics(host);
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

fn persistent_host_diagnostics(host: &PersistentHost) -> String {
    let Ok(bytes) = std::fs::read(&host.stderr_path) else {
        return String::new();
    };
    let start = bytes.len().saturating_sub(PERSISTENT_HOST_DIAGNOSTIC_LIMIT);
    String::from_utf8_lossy(&bytes[start..]).trim().to_string()
}

fn terminate_persistent_host(host: &mut PersistentHost) {
    if matches!(host.child.try_wait(), Ok(None)) {
        let _ = host.child.kill();
    }
    let _ = host.child.wait();
}

fn remove_persistent_host_files(host: &PersistentHost) {
    let _ = std::fs::remove_dir_all(&host.runtime_dir);
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
        profiles: plan.profiles.clone(),
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
        .map(|path| resolve_path(&plan.config_root, path))
        .or_else(|| runtime.environment.workspace.clone())
        .unwrap_or_else(|| plan.agent_root.clone());

    let python = resolve_python_command(&plan.config_root, &settings).path;
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
        profiles: plan.profiles.clone(),
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
    Ok(AdapterInvocation {
        effective_config: adapter_effective_config(plan)?,
        execution_strategy: plan.execution_strategy,
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

fn adapter_effective_config(plan: &RunPlan) -> Result<EffectiveConfig> {
    let mut effective_config = plan.effective_config.clone();
    effective_config.agent_root = absolute_path(effective_config.agent_root)?;
    effective_config.config_path = absolute_path(effective_config.config_path)?;
    effective_config.config_root = absolute_path(effective_config.config_root)?;
    Ok(effective_config)
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
    let command = resolve_python_command(&plan.config_root, &settings);
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
            "profiles": plan.profiles.clone(),
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

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::config::resolve_run_plan;

    fn fixture_agent_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/hermes-shim-agent")
    }

    fn temp_process_agent_dir() -> PathBuf {
        let root = std::env::temp_dir().join(new_id("fabric-process-adapter-test"));
        process_agent_dir(root)
    }

    fn process_agent_dir(root: PathBuf) -> PathBuf {
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
  "contract_version": "fabric.adapter/v1alpha1",
  "adapter_id": "acme.fabric.process",
  "harness": "process",
  "adapter_kind": "process"
}"#
    }

    fn persistent_host_agent_dir(mode: &str) -> PathBuf {
        let root = std::env::temp_dir().join(new_id("fabric-persistent-host-test"));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("adapters/persistent")).expect("create adapters dir");
        fs::write(
            root.join("agent.yaml"),
            format!(
                r#"schema_version: fabric.agent/v1alpha1
metadata:
  name: persistent-host-test-agent
harness:
  adapter_id: acme.fabric.persistent
  settings:
    runtime_strategy: persistent_local_host
    command: python3
    script: ./fake_host.py
    env:
      FABRIC_FAKE_HOST_MODE: {mode}
models:
  default:
    provider: test
    model: test-model
runtime:
  input_schema: text
  output_schema: text
  artifacts: ./artifacts
"#
            ),
        )
        .expect("write config");
        fs::write(
            root.join("adapters/persistent/fabric-adapter.json"),
            r#"{
  "contract_version": "fabric.adapter/v1alpha1",
  "adapter_id": "acme.fabric.persistent",
  "harness": "persistent-test",
  "adapter_kind": "process",
  "execution": {
    "lifecycle_contract_version": "fabric.adapter.lifecycle/v1alpha1",
    "strategies": ["process_per_invocation", "persistent_local_host"]
  }
}"#,
        )
        .expect("write adapter descriptor");
        fs::write(
            root.join("fake_host.py"),
            r#"import json
import os
import sys

VERSION = "fabric.adapter.lifecycle/v1alpha1"
MODE = os.environ.get("FABRIC_FAKE_HOST_MODE", "success")
invocations = 0

def response(operation, *, output=None, error=None):
    if error is None:
        outcome = {"status": "succeeded", "output": output}
    else:
        outcome = {"status": "failed", "error": error}
    print(json.dumps({
        "contract_version": VERSION,
        "operation": operation,
        "outcome": outcome,
    }), flush=True)

def failure(stage, code, message):
    return {
        "stage": stage,
        "code": code,
        "message": message,
        "retryable": False,
    }

for line in sys.stdin:
    message = json.loads(line)
    operation = message["operation"]
    if operation == "start":
        if MODE == "start_failure":
            print("start diagnostic", file=sys.stderr, flush=True)
            response("start", error=failure("start", "fake_start", "start rejected"))
            continue
        response("start")
        if MODE == "crash_after_start":
            print("host crashed intentionally", file=sys.stderr, flush=True)
            sys.exit(17)
    elif operation == "invoke":
        invocations += 1
        if MODE == "invoke_failure":
            response("invoke", error=failure("invoke", "fake_invoke", "invoke rejected"))
            continue
        invocation = message["payload"]
        output = {
            "host_pid": os.getpid(),
            "invocation_count": invocations,
            "input": invocation["request"]["input"],
        }
        if MODE == "adapter_reported_failure":
            output.update({
                "failed": True,
                "error": {
                    "code": "fake_adapter_failure",
                    "message": "adapter rejected the invocation",
                    "retryable": True,
                    "metadata": {"source": "fake-host"},
                },
            })
        response("invoke", output=output)
    elif operation == "stop":
        if MODE == "stop_failure":
            response("stop", error=failure("stop", "fake_stop", "stop rejected"))
            sys.exit(18)
        response("stop")
        break
"#,
        )
        .expect("write fake host");
        root
    }

    fn stopped_agents() -> Vec<String> {
        TEST_STOPPED_AGENTS.lock().expect("stop tracker").clone()
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
        let mut plan = resolve_run_plan(&root, None).expect("run plan");
        plan.profiles = vec!["runtime".to_string(), "telemetry".to_string()];
        let result = run_plan(&plan, RunRequest::text("hello fabric")).expect("run result");

        assert_eq!(result.status, RunStatus::Succeeded);
        assert_eq!(result.output, Value::String("hello fabric".to_string()));
        assert_eq!(result.metadata.get("exit_code"), Some(&Value::from(0)));
        let result_json = serde_json::to_value(&result).expect("result json");
        assert!(result_json.get("profile").is_none());
        assert_eq!(
            result_json["profiles"],
            serde_json::json!(["runtime", "telemetry"])
        );
        assert_eq!(artifact_content(&result, "stdout"), "hello fabric");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn runtime_rejects_blocked_tools_when_adapter_cannot_enforce_them() {
        let root = temp_process_agent_dir();
        let config_path = root.join("agent.yaml");
        let mut config = fs::read_to_string(&config_path).expect("read config");
        config.push_str("tools:\n  blocked:\n    - shell\n");
        fs::write(&config_path, config).expect("write blocked tools config");
        fs::write(
            root.join("adapters/process/fabric-adapter.json"),
            r#"{
  "contract_version": "fabric.adapter/v1alpha1",
  "adapter_id": "acme.fabric.process",
  "harness": "process",
  "adapter_kind": "process",
  "config": {"accepts": ["tools"]}
}"#,
        )
        .expect("write generic tools descriptor");
        let plan = resolve_run_plan(&root, None).expect("run plan");

        let error = start_runtime(&plan).expect_err("unsupported tools policy must fail closed");

        assert!(matches!(error, FabricError::UnsupportedToolsPolicy { .. }));
        assert!(
            error
                .to_string()
                .contains("cannot enforce configured blocked tools")
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn independent_runtimes_use_distinct_artifact_paths() {
        let root = temp_process_agent_dir();
        let plan = resolve_run_plan(&root, None).expect("run plan");
        let first_runtime = start_runtime(&plan).expect("first runtime");
        let second_runtime = start_runtime(&plan).expect("second runtime");

        let first = invoke_runtime(&plan, &first_runtime, RunRequest::text("first runtime"))
            .expect("first invocation");
        let second = invoke_runtime(&plan, &second_runtime, RunRequest::text("second runtime"))
            .expect("second invocation");
        let first_stdout = first
            .artifacts
            .artifacts
            .iter()
            .find(|artifact| artifact.name == "stdout")
            .expect("first stdout artifact");
        let second_stdout = second
            .artifacts
            .artifacts
            .iter()
            .find(|artifact| artifact.name == "stdout")
            .expect("second stdout artifact");

        assert_eq!(first.artifacts.root, second.artifacts.root);
        assert_ne!(first_stdout.path, second_stdout.path);
        assert!(
            first_stdout.path.starts_with(
                root.join("artifacts")
                    .join(".fabric")
                    .join(&first_runtime.runtime_id)
                    .join(&first.invocation_id)
            )
        );
        assert!(
            second_stdout.path.starts_with(
                root.join("artifacts")
                    .join(".fabric")
                    .join(&second_runtime.runtime_id)
                    .join(&second.invocation_id)
            )
        );
        assert_eq!(artifact_content(&first, "stdout"), "first runtime");
        assert_eq!(artifact_content(&second, "stdout"), "second runtime");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn native_telemetry_skips_relay_and_reaches_adapter_payload() {
        let root = temp_process_agent_dir();
        let config_path = root.join("agent.yaml");
        fs::write(
            root.join("adapters/process/fabric-adapter.json"),
            r#"{
  "contract_version": "fabric.adapter/v1alpha1",
  "adapter_id": "acme.fabric.process",
  "harness": "process",
  "adapter_kind": "process",
  "telemetry": {
    "providers": {
      "native": {
        "outputs": ["otel"]
      }
    }
  }
}"#,
        )
        .expect("write adapter telemetry support");
        let mut config = fs::read_to_string(&config_path).expect("read agent config");
        config.push_str(
            r#"
telemetry:
  providers:
    native:
      config:
        exporter: test
relay:
  project: relay-project
  output_dir: ./relay-output
"#,
        );
        fs::write(&config_path, config).expect("write native telemetry config");
        let plan = resolve_run_plan(&root, None).expect("run plan");
        let runtime = start_runtime(&plan).expect("runtime");
        let request = RunRequest::text("hello fabric");
        let invocation = InvocationHandle {
            invocation_id: "invocation-native-telemetry".to_string(),
            request_id: request.request_id.clone(),
            runtime_id: runtime.runtime_id.clone(),
        };
        let mut artifacts = artifact_manifest(&plan).expect("artifact manifest");
        let artifact_directory =
            prepare_fabric_home(&artifacts, &runtime, &invocation).expect("fabric home");

        let relay = prepare_relay_runtime_config(
            &plan,
            &runtime,
            &invocation,
            &request,
            &artifact_directory,
            &mut artifacts,
        )
        .expect("prepare telemetry");
        let payload = adapter_invocation(
            &plan,
            &runtime,
            &invocation,
            &request,
            &artifacts,
            relay.as_ref(),
        )
        .expect("adapter invocation");
        let payload = serde_json::to_value(payload).expect("adapter payload json");

        assert!(relay.is_none());
        assert_eq!(
            payload["execution_strategy"],
            serde_json::json!("process_per_invocation")
        );
        assert!(
            !artifacts
                .artifacts
                .iter()
                .any(|artifact| artifact.name == "relay_config")
        );
        assert_eq!(
            payload["telemetry_plan"]["providers"],
            serde_json::json!(["native"])
        );
        assert_eq!(payload["telemetry_plan"]["relay_enabled"], false);
        assert_eq!(
            payload["effective_config"]["config"]["telemetry"]["providers"]["native"]["config"],
            serde_json::json!({"exporter": "test"})
        );
        assert!(payload["runtime_context"]["telemetry"].get("env").is_none());
        assert_eq!(
            payload["runtime_context"]["telemetry"]["metadata"]["telemetry_providers"],
            serde_json::json!(["native"])
        );
        assert!(
            payload["runtime_context"]["telemetry"]["metadata"]
                .get("relay_project")
                .is_none()
        );
        assert!(
            payload["runtime_context"]["telemetry"]["metadata"]
                .get("relay_output_dir")
                .is_none()
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn run_plan_stops_runtime_after_invoke_error() {
        let root = temp_process_agent_dir();
        let mut plan = resolve_run_plan(&root, None).expect("run plan");
        plan.agent_name = new_id("invoke-error-agent");
        plan.effective_config.agent_name = plan.agent_name.clone();
        plan.config.harness.settings.remove("command");
        let agent_name = plan.agent_name.clone();

        let error = run_plan(&plan, RunRequest::text("hello fabric")).expect_err("invoke error");

        assert!(
            error
                .to_string()
                .contains("invalid process adapter settings"),
            "{error}"
        );
        assert!(
            stopped_agents().contains(&agent_name),
            "run_plan must stop the started runtime for {agent_name}"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn persistent_host_reuses_one_process_and_stops_idempotently() {
        let root = persistent_host_agent_dir("success");
        let plan = resolve_run_plan(&root, None).expect("run plan");
        let runtime = start_runtime(&plan).expect("start persistent host");

        let first =
            invoke_runtime(&plan, &runtime, RunRequest::text("first")).expect("first invocation");
        let second =
            invoke_runtime(&plan, &runtime, RunRequest::text("second")).expect("second invocation");

        assert_eq!(first.output["host_pid"], second.output["host_pid"]);
        assert_eq!(first.output["invocation_count"], serde_json::json!(1));
        assert_eq!(second.output["invocation_count"], serde_json::json!(2));
        assert_eq!(first.output["input"], serde_json::json!("first"));
        assert_eq!(second.output["input"], serde_json::json!("second"));
        assert_eq!(
            first.metadata["adapter_runner"],
            serde_json::json!("persistent_local_host")
        );

        let first_stop = stop_runtime(&plan, &runtime).expect("first stop");
        let second_stop = stop_runtime(&plan, &runtime).expect("idempotent stop");
        assert_eq!(first_stop[0].metadata["already_stopped"], false);
        assert_eq!(second_stop[0].metadata["already_stopped"], true);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn persistent_host_start_failure_preserves_stage_and_diagnostics() {
        let root = persistent_host_agent_dir("start_failure");
        let plan = resolve_run_plan(&root, None).expect("run plan");

        let error = start_runtime(&plan).expect_err("start must fail");
        let message = error.to_string();
        assert!(message.contains("lifecycle start"), "{message}");
        assert!(message.contains("fake_start"), "{message}");
        assert!(message.contains("start diagnostic"), "{message}");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn persistent_host_invoke_failure_is_stopped_by_run_plan() {
        let root = persistent_host_agent_dir("invoke_failure");
        let mut plan = resolve_run_plan(&root, None).expect("run plan");
        plan.agent_name = new_id("persistent-invoke-error-agent");
        plan.effective_config.agent_name = plan.agent_name.clone();
        let agent_name = plan.agent_name.clone();

        let error = run_plan(&plan, RunRequest::text("fail")).expect_err("invoke must fail");
        let message = error.to_string();
        assert!(message.contains("lifecycle invoke"), "{message}");
        assert!(message.contains("fake_invoke"), "{message}");
        assert!(
            stopped_agents().contains(&agent_name),
            "run_plan must stop the persistent host after invocation failure"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn persistent_host_preserves_normalized_adapter_failure() {
        let root = persistent_host_agent_dir("adapter_reported_failure");
        let plan = resolve_run_plan(&root, None).expect("run plan");

        let result = run_plan(&plan, RunRequest::text("fail")).expect("normalized result");

        assert_eq!(result.status, RunStatus::Failed);
        assert_eq!(result.output["failed"], true);
        assert_eq!(
            result.error,
            Some(ErrorInfo {
                stage: ErrorStage::Invoke,
                code: "fake_adapter_failure".to_string(),
                message: "adapter rejected the invocation".to_string(),
                retryable: true,
                metadata: BTreeMap::from([("source".to_string(), serde_json::json!("fake-host"),)]),
            })
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn persistent_host_crash_rejects_new_invocations() {
        let root = persistent_host_agent_dir("crash_after_start");
        let plan = resolve_run_plan(&root, None).expect("run plan");
        let runtime = start_runtime(&plan).expect("start persistent host");

        let first = invoke_runtime(&plan, &runtime, RunRequest::text("first"))
            .expect_err("crashed host must reject invocation");
        let second = invoke_runtime(&plan, &runtime, RunRequest::text("second"))
            .expect_err("dead runtime handle must remain unusable");
        assert!(first.to_string().contains("host_crashed"), "{first}");
        assert!(second.to_string().contains("host_crashed"), "{second}");

        let stop = stop_runtime(&plan, &runtime).expect_err("crashed host cannot acknowledge stop");
        assert!(stop.to_string().contains("lifecycle stop"), "{stop}");
        stop_runtime(&plan, &runtime).expect("cleanup remains idempotent");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn persistent_host_stop_failure_cleans_up_before_retry() {
        let root = persistent_host_agent_dir("stop_failure");
        let plan = resolve_run_plan(&root, None).expect("run plan");
        let runtime = start_runtime(&plan).expect("start persistent host");

        let error = stop_runtime(&plan, &runtime).expect_err("stop must fail");
        let message = error.to_string();
        assert!(message.contains("lifecycle stop"), "{message}");
        assert!(message.contains("fake_stop"), "{message}");
        let retry = stop_runtime(&plan, &runtime).expect("cleanup retry is idempotent");
        assert_eq!(retry[0].metadata["already_stopped"], true);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn runtime_handle_exposes_single_opaque_binding() {
        let root = temp_process_agent_dir();
        let plan = resolve_run_plan(&root, None).expect("run plan");
        let runtime = start_runtime(&plan).expect("runtime");

        let value = serde_json::to_value(&runtime).expect("runtime json");
        assert!(value.get("runtime_binding").is_some());
        assert_eq!(
            value["execution_strategy"],
            serde_json::json!("process_per_invocation")
        );
        assert!(value.get("plan_fingerprint").is_none());
        assert!(value.get("environment_fingerprint").is_none());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn remote_service_never_falls_back_to_per_invocation_execution() {
        let root = temp_process_agent_dir();
        let mut plan = resolve_run_plan(&root, None).expect("run plan");
        plan.execution_strategy = ExecutionStrategy::RemoteService;

        let error = start_runtime(&plan).expect_err("remote transport is required");

        assert!(matches!(
            error,
            FabricError::RuntimeStrategyUnavailable {
                strategy: ExecutionStrategy::RemoteService,
                ..
            }
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn runtime_handle_without_binding_is_rejected_during_deserialization() {
        let root = temp_process_agent_dir();
        let plan = resolve_run_plan(&root, None).expect("run plan");
        let runtime = start_runtime(&plan).expect("runtime");
        let mut value = serde_json::to_value(&runtime).expect("runtime json");
        value
            .as_object_mut()
            .expect("runtime object")
            .remove("runtime_binding");
        let error = serde_json::from_value::<RuntimeHandle>(value)
            .expect_err("runtime binding must be required");

        assert!(
            error
                .to_string()
                .contains("missing field `runtime_binding`"),
            "{error}"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn runtime_handle_validation_is_independent_of_current_directory() {
        const CHILD_ENV: &str = "FABRIC_TEST_RUNTIME_HANDLE_CWD_CHILD";
        if std::env::var_os(CHILD_ENV).is_none() {
            let output = Command::new(std::env::current_exe().expect("current test executable"))
                .arg("runtime_handle_validation_is_independent_of_current_directory")
                .arg("--nocapture")
                .env(CHILD_ENV, "1")
                .output()
                .expect("run isolated cwd test");
            assert!(
                output.status.success(),
                "isolated cwd test failed\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        let parent = std::env::temp_dir().join(new_id("fabric-runtime-cwd-test"));
        let root = process_agent_dir(parent.join("agent"));
        fs::create_dir_all(parent.join("elsewhere")).expect("create alternate cwd");
        std::env::set_current_dir(&parent).expect("enter fixture parent");
        let plan = resolve_run_plan(Path::new("agent"), None).expect("relative run plan");
        let runtime = start_runtime(&plan).expect("runtime");

        std::env::set_current_dir(parent.join("elsewhere")).expect("change cwd");
        validate_runtime_handle(&plan, &runtime).expect("valid runtime handle");

        std::env::set_current_dir(std::env::temp_dir()).expect("leave fixture");
        let _ = fs::remove_dir_all(root.parent().expect("fixture parent"));
    }

    #[test]
    fn invoke_runtime_rejects_runtime_handle_from_different_plan() {
        let root = temp_process_agent_dir();
        let plan = resolve_run_plan(&root, None).expect("run plan");
        let runtime = start_runtime(&plan).expect("runtime");
        let mut other_plan = plan.clone();
        other_plan.agent_name = "other-agent".to_string();
        other_plan.effective_config.agent_name = "other-agent".to_string();

        let error = invoke_runtime(&other_plan, &runtime, RunRequest::text("hello fabric"))
            .expect_err("runtime mismatch");

        assert!(
            error
                .to_string()
                .contains("runtime handle does not match run plan"),
            "{error}"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn invoke_runtime_rejects_runtime_handle_from_mutated_adapter_settings() {
        let root = temp_process_agent_dir();
        let plan = resolve_run_plan(&root, None).expect("run plan");
        let runtime = start_runtime(&plan).expect("runtime");
        let mut other_plan = plan.clone();
        other_plan
            .config
            .harness
            .settings
            .insert("command".to_string(), Value::String("printf".to_string()));

        let error = invoke_runtime(&other_plan, &runtime, RunRequest::text("hello fabric"))
            .expect_err("runtime mismatch");

        assert!(
            error
                .to_string()
                .contains("runtime handle does not match run plan"),
            "{error}"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn invoke_runtime_rejects_mutated_runtime_environment() {
        let root = temp_process_agent_dir();
        let plan = resolve_run_plan(&root, None).expect("run plan");
        let mut runtime = start_runtime(&plan).expect("runtime");
        runtime.environment.workspace = Some(root.join("other-workspace"));

        let error = invoke_runtime(&plan, &runtime, RunRequest::text("hello fabric"))
            .expect_err("runtime mismatch");

        assert!(
            error
                .to_string()
                .contains("runtime handle does not match run plan"),
            "{error}"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn invoke_runtime_rejects_mutated_runtime_identity() {
        let root = temp_process_agent_dir();
        let plan = resolve_run_plan(&root, None).expect("run plan");
        let mut runtime = start_runtime(&plan).expect("runtime");
        runtime.harness = "other-harness".to_string();

        let error = invoke_runtime(&plan, &runtime, RunRequest::text("hello fabric"))
            .expect_err("runtime mismatch");

        assert!(
            error
                .to_string()
                .contains("runtime handle does not match run plan"),
            "{error}"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn invoke_runtime_rejects_mutated_execution_strategy() {
        let root = temp_process_agent_dir();
        let plan = resolve_run_plan(&root, None).expect("run plan");
        let mut runtime = start_runtime(&plan).expect("runtime");
        runtime.execution_strategy = ExecutionStrategy::PersistentLocalHost;

        let error = invoke_runtime(&plan, &runtime, RunRequest::text("hello fabric"))
            .expect_err("runtime mismatch");

        assert!(error.to_string().contains("execution_strategy"), "{error}");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn stop_runtime_rejects_runtime_handle_from_different_plan() {
        let root = temp_process_agent_dir();
        let plan = resolve_run_plan(&root, None).expect("run plan");
        let runtime = start_runtime(&plan).expect("runtime");
        let mut other_plan = plan.clone();
        other_plan.agent_name = "other-agent".to_string();
        other_plan.effective_config.agent_name = "other-agent".to_string();

        let error = stop_runtime(&other_plan, &runtime).expect_err("runtime mismatch");

        assert!(
            error
                .to_string()
                .contains("runtime handle does not match run plan"),
            "{error}"
        );

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
    fn adapter_runtime_context_contains_runtime_and_invocation_ids() {
        let root = std::env::temp_dir().join(format!(
            "fabric-runtime-context-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("adapters/process")).expect("create adapters dir");
        fs::write(
            root.join("agent.yaml"),
            r#"schema_version: fabric.agent/v1alpha1
metadata:
  name: runtime-context-agent
harness:
  adapter_id: acme.fabric.process
  settings:
    command: python3
    args:
      - -c
      - |
        import json
        import sys
        payload = json.load(sys.stdin)
        print(json.dumps(payload["runtime_context"], sort_keys=True))
    stdin_payload: fabric_request
models:
  default:
    provider: test
    model: test-model
runtime:
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
        let request = RunRequest::text("hello fabric");
        let result = run_plan(&plan, request).expect("run result");

        assert_eq!(result.status, RunStatus::Succeeded);
        assert!(result.output.get("session_id").is_none());
        assert_eq!(
            result.output["runtime_id"],
            Value::String(result.runtime_id.clone())
        );
        assert_eq!(
            result.output["invocation_id"],
            Value::String(result.invocation_id.clone())
        );

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

    fn python_settings() -> PythonAdapterSettings {
        PythonAdapterSettings {
            module: "test.adapter".to_string(),
            python: None,
            python_env: None,
            args: Vec::new(),
            cwd: None,
            env: BTreeMap::new(),
        }
    }

    #[test]
    fn python_command_uses_adapter_python_as_default_python_env() {
        let root = Path::new("/config");
        let settings = python_settings();
        let command = resolve_python_command_with_env(
            root,
            &settings,
            |name| (name == ADAPTER_PYTHON_ENV).then(|| OsString::from("venv/bin/python")),
            |_| false,
            None,
        );

        assert_eq!(command.path, root.join("venv/bin/python"));
        assert_eq!(command.source, PythonSource::AdapterPythonEnv);

        let command = resolve_python_command_with_env(root, &settings, |_| None, |_| false, None);
        assert_eq!(command.path, PathBuf::from(DEFAULT_PYTHON));
        assert_eq!(command.source, PythonSource::DefaultPython3);
    }

    #[test]
    fn explicit_python_settings_override_adapter_python() {
        let root = Path::new("/config");
        let mut settings = python_settings();
        settings.python_env = Some("CUSTOM_PYTHON".to_string());
        let command = resolve_python_command_with_env(
            root,
            &settings,
            |name| match name {
                "CUSTOM_PYTHON" => Some(OsString::from("/custom/python")),
                ADAPTER_PYTHON_ENV => Some(OsString::from("/adapter/python")),
                _ => None,
            },
            |_| false,
            None,
        );
        assert_eq!(command.path, PathBuf::from("/custom/python"));
        assert_eq!(
            command.source,
            PythonSource::SettingEnv("CUSTOM_PYTHON".to_string())
        );

        let command = resolve_python_command_with_env(
            root,
            &settings,
            |name| (name == ADAPTER_PYTHON_ENV).then(|| OsString::from("/adapter/python")),
            |_| false,
            None,
        );
        assert_eq!(command.path, PathBuf::from(DEFAULT_PYTHON));
        assert_eq!(command.source, PythonSource::DefaultPython3);

        settings.python = Some(PathBuf::from("/configured/python"));
        let command = resolve_python_command_with_env(
            root,
            &settings,
            |_| Some(OsString::from("/environment/python")),
            |_| false,
            None,
        );
        assert_eq!(command.path, PathBuf::from("/configured/python"));
        assert_eq!(command.source, PythonSource::Setting);
    }

    #[test]
    fn fallback_targets_active_virtualenv_even_when_missing() {
        // VIRTUAL_ENV wins over bare python3 and is returned even when its
        // interpreter is absent, so preflight reports a clear error instead of
        // silently falling back to an unrelated python3.
        let root = Path::new("/config");
        let settings = python_settings();
        let venv_python = Path::new("/venv").join(VENV_BIN_DIR).join(VENV_PYTHON);
        let command = resolve_python_command_with_env(
            root,
            &settings,
            |name| (name == VIRTUAL_ENV_ENV).then(|| OsString::from("/venv")),
            |_| false,
            None,
        );

        assert_eq!(command.path, venv_python);
        assert_eq!(command.source, PythonSource::Virtualenv);

        let error = validate_python_command(&command)
            .expect_err("missing virtualenv interpreter must fail preflight");
        assert!(matches!(
            error,
            FabricError::PythonInterpreterUnavailable { .. }
        ));
    }

    #[test]
    fn empty_environment_values_are_ignored() {
        let root = Path::new("/config");

        // ADAPTER_PYTHON set but empty -> ignored, falls through to python3.
        let command = resolve_python_command_with_env(
            root,
            &python_settings(),
            |name| (name == ADAPTER_PYTHON_ENV).then(OsString::new),
            |_| false,
            None,
        );
        assert_eq!(command.path, PathBuf::from(DEFAULT_PYTHON));
        assert_eq!(command.source, PythonSource::DefaultPython3);

        // python_env names a variable that is set but empty -> ignored.
        let mut settings = python_settings();
        settings.python_env = Some("CUSTOM_PYTHON".to_string());
        let command = resolve_python_command_with_env(
            root,
            &settings,
            |name| (name == "CUSTOM_PYTHON").then(OsString::new),
            |_| false,
            None,
        );
        assert_eq!(command.path, PathBuf::from(DEFAULT_PYTHON));
        assert_eq!(command.source, PythonSource::DefaultPython3);
    }

    #[cfg(windows)]
    #[test]
    fn windows_default_python_is_python_exe() {
        // find_on_path joins each PATH entry with the bare name and stats it, so
        // on Windows the fallback must name python.exe to resolve on disk.
        assert_eq!(DEFAULT_PYTHON, "python.exe");
        let command = resolve_python_command_with_env(
            Path::new("C:/config"),
            &python_settings(),
            |_| None,
            |_| false,
            None,
        );
        assert_eq!(command.path, PathBuf::from("python.exe"));
        assert_eq!(command.source, PythonSource::DefaultPython3);
    }

    #[test]
    fn fallback_uses_host_interpreter_next_to_executable() {
        let root = Path::new("/config");
        let settings = python_settings();
        let exe = PathBuf::from("/opt/fabric/bin/nemo-fabric");
        let sibling = PathBuf::from("/opt/fabric/bin").join(VENV_PYTHON);
        let expected = sibling.clone();
        let command = resolve_python_command_with_env(
            root,
            &settings,
            |_| None,
            move |path| path == expected.as_path(),
            Some(exe),
        );

        assert_eq!(command.path, sibling);
        assert_eq!(command.source, PythonSource::HostInterpreter);
    }

    #[test]
    fn adapter_python_preflight_requires_a_file() {
        let root = std::env::temp_dir().join(new_id("fabric-adapter-python-test"));
        fs::create_dir_all(&root).expect("create test directory");
        let interpreter = root.join("python");
        fs::write(&interpreter, "").expect("create interpreter file");

        validate_python_command(&PythonCommand {
            path: interpreter,
            source: PythonSource::AdapterPythonEnv,
        })
        .expect("file path should pass preflight");

        let error = validate_python_command(&PythonCommand {
            path: root.join("missing-python"),
            source: PythonSource::AdapterPythonEnv,
        })
        .expect_err("missing path should fail preflight");
        assert!(matches!(
            error,
            FabricError::PythonInterpreterUnavailable { .. }
        ));

        let error = validate_python_command(&PythonCommand {
            path: root.clone(),
            source: PythonSource::AdapterPythonEnv,
        })
        .expect_err("directory path should fail preflight");
        assert!(matches!(
            error,
            FabricError::PythonInterpreterUnavailable { .. }
        ));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn preflight_rejects_missing_virtualenv_interpreter() {
        // A resolved absolute interpreter that does not exist must fail preflight
        // with a clear error instead of surfacing as a mid-run subprocess crash.
        let error = validate_python_command(&PythonCommand {
            path: Path::new("/venv").join(VENV_BIN_DIR).join(VENV_PYTHON),
            source: PythonSource::Virtualenv,
        })
        .expect_err("missing virtualenv interpreter should fail preflight");
        assert!(matches!(
            error,
            FabricError::PythonInterpreterUnavailable { .. }
        ));
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
        assert_eq!(
            result.metadata.get("module"),
            Some(&Value::String(
                "nemo_fabric_test_adapters.hermes_shim.adapter".to_string()
            ))
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
