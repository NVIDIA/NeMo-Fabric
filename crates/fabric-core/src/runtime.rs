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
    AdapterKind, CapabilityKind, CapabilityPlan, CapabilityTarget, ControlLocation,
    EnvironmentOwnership, FabricConfig, RunPlan, TelemetryPlan,
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

/// Context generated for one invocation of a started runtime.
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

/// One invocation against an initialized adapter runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AdapterInvocation {
    /// Invocation context generated by Fabric.
    pub runtime_context: RuntimeContext,
    /// Typed caller request for this invocation.
    pub request: RunRequest,
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
    Start(Box<AdapterLifecycleStart>),
    Invoke(Box<AdapterInvocation>),
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
    #[serde(flatten)]
    request: AdapterLifecycleRequestKind,
}

impl AdapterLifecycleRequest {
    fn new(request: AdapterLifecycleRequestKind) -> Self {
        Self { request }
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

struct LocalHostAdapter;

#[derive(Debug, Clone)]
struct RelayRuntimeConfig {
    path: PathBuf,
    env: BTreeMap<String, String>,
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
    if uses_local_host(plan) {
        return LocalHostAdapter.start(plan, environment);
    }
    Err(FabricError::UnsupportedRuntimeAdapter {
        harness: harness(plan),
        adapter_kind: adapter_kind(plan),
    })
}

/// Invoke a started harness runtime.
pub fn invoke_runtime(
    plan: &RunPlan,
    runtime: &RuntimeHandle,
    request: RunRequest,
) -> Result<RunResult> {
    validate_blocked_tools_support(plan)?;
    validate_runtime_handle(plan, runtime)?;
    if uses_local_host(plan) {
        return LocalHostAdapter.invoke(plan, runtime, request);
    }
    Err(FabricError::UnsupportedRuntimeAdapter {
        harness: harness(plan),
        adapter_kind: adapter_kind(plan),
    })
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
    if uses_local_host(plan) {
        return LocalHostAdapter.stop(runtime);
    }
    Err(FabricError::UnsupportedRuntimeAdapter {
        harness: runtime.harness.clone(),
        adapter_kind: runtime.adapter_kind,
    })
}

fn uses_local_host(plan: &RunPlan) -> bool {
    matches!(
        adapter_kind(plan),
        AdapterKind::Process | AdapterKind::Python
    )
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

impl RuntimeAdapter for LocalHostAdapter {
    fn start(&self, plan: &RunPlan, environment: EnvironmentHandle) -> Result<RuntimeHandle> {
        if environment.provider != "local" {
            return Err(FabricError::UnsupportedEnvironmentProvider {
                provider: environment.provider,
                adapter_kind: adapter_kind(plan),
            });
        }
        if !matches!(
            adapter_kind(plan),
            AdapterKind::Process | AdapterKind::Python
        ) {
            return Err(FabricError::UnsupportedRuntimeAdapter {
                harness: harness(plan),
                adapter_kind: adapter_kind(plan),
            });
        }
        if adapter_kind(plan) == AdapterKind::Python {
            preflight_python_adapter(plan)?;
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
            environment,
        };

        let start_invocation = InvocationHandle {
            invocation_id: new_id("runtime-start"),
            request_id: new_id("runtime-start-request"),
            runtime_id: runtime.runtime_id.clone(),
        };
        let mut artifacts = artifact_manifest(plan)?;
        let fabric_home = prepare_fabric_home(&artifacts, &runtime, &start_invocation)?;
        let relay_config =
            prepare_relay_runtime_config(plan, &runtime, &fabric_home, &mut artifacts)?;
        let start = adapter_lifecycle_start(
            plan,
            &runtime,
            &start_invocation,
            &artifacts,
            relay_config.as_ref(),
        )?;
        let request =
            AdapterLifecycleRequest::new(AdapterLifecycleRequestKind::Start(Box::new(start)));
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
        let mut host_guard = host.lock().unwrap_or_else(|error| error.into_inner());
        let artifacts = host_guard.artifacts.clone();
        let relay_config = host_guard.relay_config.clone();
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
        let lifecycle_request = AdapterLifecycleRequest::new(AdapterLifecycleRequestKind::Invoke(
            Box::new(adapter_invocation),
        ));
        match exchange_lifecycle_message(
            &mut host_guard,
            &runtime.runtime_id,
            &lifecycle_request,
            invoke_timeout,
        ) {
            Ok(output) => {
                let stderr = take_local_host_stderr(&mut host_guard);
                Ok((
                    output,
                    stderr,
                    host_guard.command.clone(),
                    host_guard.child.id(),
                    artifacts,
                    relay_config,
                    fabric_home,
                    fabric_invocation,
                ))
            }
            Err(error) => {
                if matches!(
                    &error,
                    FabricError::AdapterLifecycleOperation { code, .. } if code == "host_timeout"
                ) {
                    invalidate_timed_out_local_host(&runtime.runtime_id, &host, &mut host_guard);
                }
                Err(error)
            }
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
    ) = exchange_result?;

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

fn invalidate_timed_out_local_host(
    runtime_id: &str,
    expected_host: &Arc<Mutex<LocalAdapterHost>>,
    host: &mut LocalAdapterHost,
) {
    {
        let mut hosts = local_hosts();
        if hosts
            .get(runtime_id)
            .is_some_and(|host| Arc::ptr_eq(host, expected_host))
        {
            hosts.remove(runtime_id);
        }
    }

    let _ = terminate_local_host(host);
    let _ = remove_local_host_files(host);
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
    match adapter_kind(plan) {
        AdapterKind::Process => process_local_host_command(plan, runtime),
        AdapterKind::Python => python_local_host_command(plan, runtime),
        adapter_kind => Err(FabricError::UnsupportedRuntimeAdapter {
            harness: harness(plan),
            adapter_kind,
        }),
    }
}

fn process_local_host_command(
    plan: &RunPlan,
    runtime: &RuntimeHandle,
) -> Result<(Command, String)> {
    let settings = parse_process_settings(plan)?;
    let command_path = resolve_command_path(
        adapter_setting_root(plan, "command"),
        Path::new(&settings.command),
    );
    let command_args = process_command_args(plan, &settings);
    let cwd = settings
        .cwd
        .as_ref()
        .map(|path| resolve_path(&plan.base_dir, path))
        .or_else(|| runtime.environment.workspace.clone())
        .unwrap_or_else(|| plan.base_dir.clone());
    let mut command = Command::new(&command_path);
    command
        .args(&command_args)
        .current_dir(cwd)
        .envs(&settings.env);
    let display = std::iter::once(command_path.to_string_lossy().into_owned())
        .chain(command_args)
        .collect::<Vec<_>>()
        .join(" ");
    Ok((command, display))
}

fn python_local_host_command(plan: &RunPlan, runtime: &RuntimeHandle) -> Result<(Command, String)> {
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

fn adapter_id(plan: &RunPlan) -> Option<String> {
    plan.adapter_descriptor
        .as_ref()
        .map(|adapter| adapter.descriptor.adapter_id.clone())
        .or_else(|| Some(plan.config.harness.adapter_id.clone()))
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

fn adapter_lifecycle_start(
    plan: &RunPlan,
    runtime: &RuntimeHandle,
    invocation: &InvocationHandle,
    artifacts: &ArtifactManifest,
    relay_config: Option<&RelayRuntimeConfig>,
) -> Result<AdapterLifecycleStart> {
    Ok(AdapterLifecycleStart {
        agent_name: plan.agent_name.clone(),
        base_dir: absolute_path(plan.base_dir.clone())?,
        config: plan.config.clone(),
        runtime_context: adapter_runtime_context(
            plan,
            runtime,
            invocation,
            artifacts,
            relay_config,
        ),
        capability_plan: plan.capability_plan.clone(),
        telemetry_plan: plan.telemetry_plan.clone(),
    })
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
        runtime_context: adapter_runtime_context(
            plan,
            runtime,
            invocation,
            artifacts,
            relay_config,
        ),
        request: request.clone(),
    })
}

fn adapter_runtime_context(
    plan: &RunPlan,
    runtime: &RuntimeHandle,
    invocation: &InvocationHandle,
    artifacts: &ArtifactManifest,
    relay_config: Option<&RelayRuntimeConfig>,
) -> RuntimeContext {
    RuntimeContext {
        runtime_id: runtime.runtime_id.clone(),
        invocation_id: invocation.invocation_id.clone(),
        request_id: invocation.request_id.clone(),
        environment: runtime.environment.clone(),
        artifacts: artifacts.clone(),
        telemetry: runtime_telemetry_context(plan, relay_config),
    }
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
    let fabric = serde_json::json!({
        "agent_name": plan.agent_name.clone(),
        "harness": harness(plan),
        "adapter_id": adapter_id(plan),
        "runtime_id": runtime.runtime_id.clone(),
        "adapter_outputs": telemetry.adapter_outputs.clone(),
    });
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
    // The atomic counter only differentiates ids within one Fabric process.
    // Include the process id so independently running Fabric processes cannot
    // collide when they generate ids in the same millisecond.
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
    use crate::config::{ResolveContext, resolve_run_plan_from_config};

    fn local_host_plan(mode: &str) -> (PathBuf, RunPlan) {
        local_host_plan_with_relay(mode, false)
    }

    fn local_host_plan_with_relay(mode: &str, relay: bool) -> (PathBuf, RunPlan) {
        let root = std::env::temp_dir().join(new_id("fabric-local-host-test"));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("adapters/local-host")).expect("create adapters dir");
        fs::write(
            root.join("adapters/local-host/fabric-adapter.json"),
            r#"{
  "contract_version": "fabric.adapter/v1alpha1",
  "adapter_id": "acme.fabric.local-host",
  "harness": "local-host-test",
  "adapter_kind": "python",
  "runner": {"module": "fake_host"},
  "telemetry": {
    "providers": {
      "relay": {"outputs": ["atif"]}
    }
  }
}"#,
        )
        .expect("write adapter descriptor");
        fs::write(
            root.join("fake_host.py"),
            r#"import json
import os
import sys

MODE = os.environ.get("FABRIC_FAKE_HOST_MODE", "success")
invocations = 0

def response(operation, *, output=None, error=None):
    outcome = (
        {"status": "succeeded", "output": output}
        if error is None
        else {"status": "failed", "error": error}
    )
    print(json.dumps({
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
            sys.exit(16)
        response("start")
        if MODE == "crash_after_start":
            print("host crashed intentionally", file=sys.stderr, flush=True)
            sys.exit(17)
    elif operation == "invoke":
        invocations += 1
        if MODE == "invoke_stderr":
            print(f"diagnostic-{invocations}", file=sys.stderr, flush=True)
        if MODE == "invoke_timeout":
            print("invoke accepted without response", file=sys.stderr, flush=True)
            continue
        if MODE == "invoke_failure":
            response("invoke", error=failure("invoke", "fake_invoke", "invoke rejected"))
            continue
        invocation = message["payload"]
        if set(invocation) != {"runtime_context", "request"}:
            response("invoke", error=failure(
                "invoke", "fake_invoke_shape", "invoke payload contains runtime config"
            ))
            continue
        output = {
            "host_pid": os.getpid(),
            "invocation_count": invocations,
            "input": invocation["request"]["input"],
            "runtime_id": invocation["runtime_context"]["runtime_id"],
            "invocation_id": invocation["runtime_context"]["invocation_id"],
            "request_id": invocation["runtime_context"]["request_id"],
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

        let mut config_value = serde_json::json!({
            "schema_version": "fabric.agent/v1alpha1",
            "metadata": {"name": "local-host-test-agent"},
            "harness": {
                "adapter_id": "acme.fabric.local-host",
                "resolution": "preinstalled",
                "settings": {
                    "python": "python3",
                    "cwd": ".",
                    "env": {"FABRIC_FAKE_HOST_MODE": mode},
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
        if relay {
            config_value["telemetry"] = serde_json::json!({
                "providers": {"relay": {}},
            });
            config_value["relay"] = serde_json::json!({
                "observability": {"atif": {"enabled": true}},
            });
        }
        let config: FabricConfig = serde_json::from_value(config_value).expect("typed config");
        let plan = resolve_run_plan_from_config(config, ResolveContext::new(&root))
            .expect("resolve local-host plan");
        (root, plan)
    }

    fn stopped_agents() -> Vec<String> {
        TEST_STOPPED_AGENTS.lock().expect("stop tracker").clone()
    }

    #[test]
    fn local_host_reuses_one_process_and_stops_idempotently() {
        let (root, plan) = local_host_plan("success");
        let runtime = start_runtime(&plan).expect("start local host");

        let first =
            invoke_runtime(&plan, &runtime, RunRequest::text("first")).expect("first invocation");
        let second =
            invoke_runtime(&plan, &runtime, RunRequest::text("second")).expect("second invocation");

        assert_eq!(first.output["host_pid"], second.output["host_pid"]);
        assert_eq!(first.output["invocation_count"], serde_json::json!(1));
        assert_eq!(second.output["invocation_count"], serde_json::json!(2));
        assert_eq!(first.output["input"], serde_json::json!("first"));
        assert_eq!(second.output["input"], serde_json::json!("second"));
        assert_eq!(first.metadata["host_pid"], second.metadata["host_pid"]);
        assert_eq!(
            first.metadata["adapter_runner"],
            serde_json::json!("persistent_local_host")
        );
        let stdout = first
            .artifacts
            .artifacts
            .iter()
            .find(|artifact| artifact.name == "stdout")
            .expect("stdout artifact");
        let captured: Value =
            serde_json::from_str(&fs::read_to_string(&stdout.path).expect("read stdout artifact"))
                .expect("parse stdout artifact");
        assert_eq!(captured, first.output);
        assert!(
            first
                .artifacts
                .artifacts
                .iter()
                .all(|artifact| artifact.name != "stderr")
        );

        let first_stop = stop_runtime(&plan, &runtime).expect("first stop");
        let second_stop = stop_runtime(&plan, &runtime).expect("idempotent stop");
        assert_eq!(first_stop[0].metadata["already_stopped"], false);
        assert_eq!(second_stop[0].metadata["already_stopped"], true);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn process_adapter_uses_the_same_persistent_local_host_protocol() {
        let (root, mut plan) = local_host_plan("success");
        let descriptor = plan
            .adapter_descriptor
            .as_mut()
            .expect("resolved adapter descriptor");
        descriptor.descriptor.adapter_kind = AdapterKind::Process;
        descriptor.descriptor.runner = serde_json::from_value(serde_json::json!({
            "command": "python3",
            "script": "../../fake_host.py",
        }))
        .expect("process runner settings");

        let runtime = start_runtime(&plan).expect("start process host");
        let first =
            invoke_runtime(&plan, &runtime, RunRequest::text("first")).expect("first invocation");
        let second =
            invoke_runtime(&plan, &runtime, RunRequest::text("second")).expect("second invocation");
        stop_runtime(&plan, &runtime).expect("stop process host");

        assert_eq!(first.output["host_pid"], second.output["host_pid"]);
        assert_eq!(first.output["invocation_count"], 1);
        assert_eq!(second.output["invocation_count"], 2);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn local_host_relay_config_is_runtime_scoped() {
        let (root, plan) = local_host_plan_with_relay("success", true);
        let runtime = start_runtime(&plan).expect("start local host");
        let host = local_hosts()
            .get(&runtime.runtime_id)
            .cloned()
            .expect("active local host");
        let relay_config_path = host
            .lock()
            .expect("local host")
            .relay_config
            .as_ref()
            .expect("Relay config")
            .path
            .clone();
        let relay_config: Value = serde_json::from_str(
            &fs::read_to_string(relay_config_path).expect("read Relay config"),
        )
        .expect("parse Relay config");

        assert_eq!(
            relay_config["fabric"]["runtime_id"],
            serde_json::json!(runtime.runtime_id)
        );
        assert!(relay_config["fabric"].get("invocation_id").is_none());
        assert!(relay_config["fabric"].get("request_id").is_none());

        let result =
            invoke_runtime(&plan, &runtime, RunRequest::text("first")).expect("invoke local host");
        assert_eq!(result.output["runtime_id"], result.runtime_id);
        assert_eq!(result.output["invocation_id"], result.invocation_id);
        assert_eq!(result.output["request_id"], result.request_id);

        stop_runtime(&plan, &runtime).expect("stop local host");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn local_host_start_failure_preserves_stage_and_diagnostics() {
        let (root, plan) = local_host_plan("start_failure");

        let error = start_runtime(&plan).expect_err("start must fail");
        let message = error.to_string();
        assert!(message.contains("lifecycle start"), "{message}");
        assert!(message.contains("fake_start"), "{message}");
        assert!(message.contains("start diagnostic"), "{message}");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn local_host_invoke_timeout_evicts_and_terminates_host() {
        let (root, plan) = local_host_plan("invoke_timeout");
        let runtime = start_runtime(&plan).expect("start local host");
        let host = local_hosts()
            .get(&runtime.runtime_id)
            .cloned()
            .expect("active local host");
        let runtime_dir = host.lock().expect("local host").runtime_dir.clone();

        let error = run_local_host_adapter_with_timeout(
            &plan,
            &runtime,
            RunRequest::text("hang"),
            Duration::from_millis(100),
        )
        .expect_err("unresponsive host must time out");

        assert!(matches!(
            &error,
            FabricError::AdapterLifecycleOperation {
                code,
                diagnostics,
                ..
            } if code == "host_timeout" && diagnostics.contains("invoke accepted without response")
        ));
        assert!(!local_hosts().contains_key(&runtime.runtime_id));
        assert!(
            host.lock()
                .expect("local host")
                .child
                .try_wait()
                .expect("inspect timed-out host")
                .is_some(),
            "timed-out host process must be terminated"
        );
        assert!(!runtime_dir.exists());

        let retry = invoke_runtime(&plan, &runtime, RunRequest::text("retry"))
            .expect_err("timed-out runtime must remain unavailable");
        assert!(matches!(
            retry,
            FabricError::AdapterLifecycleOperation { code, .. } if code == "host_unavailable"
        ));
        let stopped = stop_runtime(&plan, &runtime).expect("stop after timeout is idempotent");
        assert_eq!(stopped[0].metadata["already_stopped"], true);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn local_host_timeout_prevents_waiting_invocation_from_reusing_host() {
        let (root, plan) = local_host_plan("invoke_timeout");
        let runtime = start_runtime(&plan).expect("start local host");
        let host = local_hosts()
            .get(&runtime.runtime_id)
            .cloned()
            .expect("active local host");
        let stderr_path = host.lock().expect("local host").stderr_path.clone();

        let first_plan = plan.clone();
        let first_runtime = runtime.clone();
        let first = thread::spawn(move || {
            run_local_host_adapter_with_timeout(
                &first_plan,
                &first_runtime,
                RunRequest::text("first"),
                Duration::from_millis(500),
            )
        });
        let accepted_deadline = Instant::now() + Duration::from_secs(2);
        while !fs::read_to_string(&stderr_path)
            .expect("read host stderr")
            .contains("invoke accepted without response")
        {
            assert!(
                Instant::now() < accepted_deadline,
                "first invocation was not accepted"
            );
            thread::sleep(Duration::from_millis(10));
        }

        let second_plan = plan.clone();
        let second_runtime = runtime.clone();
        let second = thread::spawn(move || {
            run_local_host_adapter_with_timeout(
                &second_plan,
                &second_runtime,
                RunRequest::text("second"),
                Duration::from_millis(100),
            )
        });
        // The registry, this test, and the first invocation own three host
        // references. A fourth proves that the second invocation passed the
        // registry lookup and is waiting for the host mutex.
        let waiting_deadline = Instant::now() + Duration::from_secs(1);
        while Arc::strong_count(&host) < 4 {
            assert!(
                Instant::now() < waiting_deadline,
                "second invocation did not acquire the host reference"
            );
            thread::sleep(Duration::from_millis(10));
        }

        let first_error = first
            .join()
            .expect("first invocation thread")
            .expect_err("first invocation must time out");
        let second_error = second
            .join()
            .expect("second invocation thread")
            .expect_err("waiting invocation must not reuse the timed-out host");

        assert!(matches!(
            first_error,
            FabricError::AdapterLifecycleOperation { code, .. } if code == "host_timeout"
        ));
        assert!(matches!(
            second_error,
            FabricError::AdapterLifecycleOperation { code, .. } if code == "host_crashed"
        ));
        assert!(!local_hosts().contains_key(&runtime.runtime_id));
        let stopped = stop_runtime(&plan, &runtime).expect("stop after timeout is idempotent");
        assert_eq!(stopped[0].metadata["already_stopped"], true);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn local_host_captures_each_stderr_delta_once() {
        let (root, plan) = local_host_plan("invoke_stderr");
        let runtime = start_runtime(&plan).expect("start local host");

        let first =
            invoke_runtime(&plan, &runtime, RunRequest::text("first")).expect("first invocation");
        let second =
            invoke_runtime(&plan, &runtime, RunRequest::text("second")).expect("second invocation");
        let read_stderr = |result: &RunResult| {
            let artifact = result
                .artifacts
                .artifacts
                .iter()
                .find(|artifact| artifact.name == "stderr")
                .expect("stderr artifact");
            fs::read_to_string(&artifact.path).expect("read stderr artifact")
        };

        assert_eq!(read_stderr(&first), "diagnostic-1\n");
        assert_eq!(read_stderr(&second), "diagnostic-2\n");
        stop_runtime(&plan, &runtime).expect("stop local host");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn run_plan_stops_local_host_after_invoke_failure() {
        let (root, mut plan) = local_host_plan("invoke_failure");
        let agent_name = new_id("local-host-invoke-error-agent");
        plan.agent_name = agent_name.clone();
        plan.config.metadata.name = agent_name.clone();

        let error = run_plan(&plan, RunRequest::text("fail")).expect_err("invoke must fail");
        assert!(error.to_string().contains("fake_invoke"), "{error}");
        assert!(
            stopped_agents().contains(&agent_name),
            "run_plan must stop the local host after invocation failure"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn local_host_preserves_normalized_adapter_failure() {
        let (root, plan) = local_host_plan("adapter_reported_failure");

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
                metadata: BTreeMap::from([("source".to_string(), serde_json::json!("fake-host"))]),
            })
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn local_host_crash_is_terminal_for_runtime() {
        let (root, plan) = local_host_plan("crash_after_start");
        let runtime = start_runtime(&plan).expect("start local host");

        let first = invoke_runtime(&plan, &runtime, RunRequest::text("first"))
            .expect_err("crashed host must reject invocation");
        let second = invoke_runtime(&plan, &runtime, RunRequest::text("second"))
            .expect_err("dead runtime must remain unusable");
        assert!(first.to_string().contains("host_crashed"), "{first}");
        assert!(second.to_string().contains("host_crashed"), "{second}");

        let stopped = stop_runtime(&plan, &runtime).expect("crashed host cleanup");
        assert_eq!(stopped[0].metadata["host_crashed"], true);
        stop_runtime(&plan, &runtime).expect("cleanup remains idempotent");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn local_host_stop_failure_still_releases_host() {
        let (root, plan) = local_host_plan("stop_failure");
        let runtime = start_runtime(&plan).expect("start local host");

        let error = stop_runtime(&plan, &runtime).expect_err("stop must fail");
        assert!(error.to_string().contains("fake_stop"), "{error}");
        let retry = stop_runtime(&plan, &runtime).expect("cleanup retry is idempotent");
        assert_eq!(retry[0].metadata["already_stopped"], true);

        let _ = fs::remove_dir_all(root);
    }
}
