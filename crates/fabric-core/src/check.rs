// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime health checks and shared contracts.

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant, SystemTime};

use chrono::{DateTime, SecondsFormat, Utc};
use hmac::{Hmac, Mac};
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Deserializer, Serialize};
use sha2::Sha256;

use crate::{EnvironmentHandle, Result, RuntimeHandle};

const LIMIT: u64 = 16 * 1024;
static BINDINGS: LazyLock<Mutex<BTreeMap<String, Arc<Binding>>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// Version of the health request and report contract.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum CheckVersion {
    /// Health contract version `fabric.check/v1alpha1`.
    #[default]
    #[serde(rename = "fabric.check/v1alpha1")]
    V1Alpha1,
}

/// Agent health level.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CheckLevel {
    /// Observe the native process or session.
    #[default]
    Live,
    /// Observe the initialized request interface.
    Active,
    /// Observe required local model, tool, and connection setup.
    Ready,
}

impl CheckLevel {
    /// Stable wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Live => "live",
            Self::Active => "active",
            Self::Ready => "ready",
        }
    }
}

/// One attempted server or agent stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CheckStage {
    /// Server responsiveness.
    Server,
    /// Native session liveness.
    Live,
    /// Request interface availability.
    Active,
    /// Local setup readiness.
    Ready,
}

/// Health check outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CheckOutcome {
    /// All requested evidence passed.
    Passed,
    /// A confirmed problem was observed.
    Failed,
    /// Evidence is unavailable or invalidated.
    Unknown,
    /// The requested check is unimplemented.
    Unsupported,
}

/// Diagnostic codes for health observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CheckReason {
    /// The server is stopping.
    ServerStopping,
    /// Advertised capability evidence is unavailable.
    CapabilitiesUnverified,
    /// No matching runtime binding exists.
    RuntimeUnavailable,
    /// The owned process exited.
    AgentExited,
    /// The native session closed.
    SessionClosed,
    /// The native request interface is unavailable.
    EndpointUnavailable,
    /// Required local setup failed.
    HealthFailed,
    /// The shared deadline expired.
    ProbeTimeout,
    /// The probe did not provide valid evidence.
    ProbeError,
    /// Ownership or lifecycle changed while checking.
    ObservationChanged,
    /// The adapter does not implement health checks.
    UnsupportedAdapter,
    /// The requested level is unavailable.
    UnsupportedLevel,
    /// This environment does not implement the requested scope.
    UnsupportedScope,
}

/// Fixed evidence-source identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum CheckSource {
    /// Runtime server observation.
    #[serde(rename = "fabric.runtime_server")]
    Server,
    /// Runtime-server transport observation.
    #[serde(rename = "fabric.runtime_server.transport")]
    ServerTransport,
    /// Unobserved adapter transport or unsupported probe.
    #[serde(rename = "fabric.adapter")]
    Adapter,
    /// Fresh native Pi context observation.
    #[serde(rename = "fabric.adapter.pi.session")]
    PiSession,
    /// Pi initialization and interface guards.
    #[serde(rename = "fabric.adapter.pi.interface")]
    PiInterface,
    /// Pi model, tool, and connection inspection.
    #[serde(rename = "fabric.adapter.pi.setup")]
    PiSetup,
}

fn timeout_default() -> f64 {
    3.0
}
fn required_option<'de, D, T>(deserializer: D) -> std::result::Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}
fn required_fields(schema: &mut Schema) {
    if let Some(properties) = schema
        .get("properties")
        .and_then(serde_json::Value::as_object)
    {
        let names = properties
            .keys()
            .map(|name| serde_json::Value::String(name.clone()))
            .collect();
        schema.insert("required".into(), serde_json::Value::Array(names));
    }
}
fn timeout<'de, D: Deserializer<'de>>(deserializer: D) -> std::result::Result<f64, D::Error> {
    let value = f64::deserialize(deserializer)?;
    if !value.is_finite() || value <= 0.0 {
        return Err(serde::de::Error::custom(
            "check timeout must be positive and finite",
        ));
    }
    Ok(value)
}
fn runtime_id<'de, D: Deserializer<'de>>(deserializer: D) -> std::result::Result<String, D::Error> {
    let value = String::deserialize(deserializer)?;
    if value.trim().is_empty() || value.contains('\0') {
        return Err(serde::de::Error::custom("runtime ID must be nonempty"));
    }
    Ok(value)
}

/// Select server or runtime health; the environment is supplied separately.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "scope", rename_all = "snake_case", deny_unknown_fields)]
pub enum CheckRequest {
    /// Check runtime server responsiveness.
    Server {
        /// Version of this contract.
        #[serde(default)]
        schema_version: CheckVersion,
        /// One budget for all routing and probes.
        #[serde(default = "timeout_default", deserialize_with = "timeout")]
        #[schemars(extend("exclusiveMinimum" = 0))]
        timeout_seconds: f64,
    },
    /// Check the agent owned by one exact runtime ID.
    Runtime {
        /// Version of this contract.
        #[serde(default)]
        schema_version: CheckVersion,
        /// Runtime returned by startup in the selected environment.
        #[serde(deserialize_with = "runtime_id")]
        #[schemars(length(min = 1), regex(pattern = "\\S"))]
        runtime_id: String,
        /// Highest requested level; earlier levels run first.
        #[serde(default)]
        level: CheckLevel,
        /// One budget for all routing and probes.
        #[serde(default = "timeout_default", deserialize_with = "timeout")]
        #[schemars(extend("exclusiveMinimum" = 0))]
        timeout_seconds: f64,
    },
}

impl CheckRequest {
    /// Requested timeout, in seconds.
    pub fn timeout_seconds(&self) -> f64 {
        match self {
            Self::Server {
                timeout_seconds, ..
            }
            | Self::Runtime {
                timeout_seconds, ..
            } => *timeout_seconds,
        }
    }
    /// Requested runtime level; absent for server checks.
    pub fn level(&self) -> Option<CheckLevel> {
        match self {
            Self::Server { .. } => None,
            Self::Runtime { level, .. } => Some(*level),
        }
    }
}

/// Current runtime activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeActivity {
    /// No invocation is running.
    Idle,
    /// An invocation is running.
    Busy,
    /// Shutdown has begun.
    Stopping,
    /// Shutdown completed.
    Stopped,
    /// Current activity could not be verified.
    Unknown,
}

/// Runtime lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CheckRuntimeState {
    /// The bound runtime is active.
    Active,
    /// The runtime stopped.
    Stopped,
    /// Runtime operation failed.
    Failed,
    /// State could not be verified.
    Unknown,
}

/// Runtime state and activity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RuntimeObservation {
    /// Verified runtime state.
    pub state: CheckRuntimeState,
    /// Observed activity.
    pub activity: RuntimeActivity,
}

/// Server lifecycle observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ServerObservation {
    /// Server responsiveness or shutdown state.
    pub state: ServerState,
}

/// Server states supported by the health contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ServerState {
    /// The server replied.
    Responding,
    /// Shutdown began.
    Stopping,
    /// No server evidence is available.
    Unknown,
}

/// One adapter health probe using the remaining shared deadline budget.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AdapterCheckRequest {
    /// Exact runtime selected by the core.
    #[serde(deserialize_with = "runtime_id")]
    #[schemars(length(min = 1), regex(pattern = "\\S"))]
    pub runtime_id: String,
    /// One native level to observe.
    pub level: CheckLevel,
    /// Remaining shared deadline budget.
    #[serde(deserialize_with = "timeout")]
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub timeout_seconds: f64,
}

/// Result of one adapter health probe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(transform = required_fields)]
pub struct AdapterCheckResult {
    /// Probe outcome.
    pub outcome: CheckOutcome,
    /// Fixed diagnostic or null on pass.
    #[serde(deserialize_with = "required_option")]
    pub reason_code: Option<CheckReason>,
    /// Current agent activity.
    pub activity: RuntimeActivity,
    /// Fixed evidence source.
    pub source: CheckSource,
    /// Native observation time in UTC, or null if no observation was made.
    #[serde(deserialize_with = "required_option")]
    pub observed_at: Option<String>,
}

/// One attempted stage in evaluation order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(transform = required_fields)]
pub struct CheckObservation {
    /// Stage that was attempted.
    pub name: CheckStage,
    /// Stage outcome.
    pub outcome: CheckOutcome,
    /// Fixed diagnostic or null on pass.
    #[serde(deserialize_with = "required_option")]
    pub reason_code: Option<CheckReason>,
    /// Fixed evidence source.
    pub source: CheckSource,
    /// Fresh observation time, or null if unobserved.
    #[serde(deserialize_with = "required_option")]
    pub observed_at: Option<String>,
}

/// Health check report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(transform = required_fields)]
pub struct CheckReport {
    /// Version of this contract.
    pub schema_version: CheckVersion,
    /// Scope requested by the caller.
    pub scope: CheckScope,
    /// Selected runtime ID, or null for server scope.
    #[serde(deserialize_with = "required_option")]
    pub runtime_id: Option<String>,
    /// Defaulted level, or null for server scope.
    #[serde(deserialize_with = "required_option")]
    pub requested_level: Option<CheckLevel>,
    /// Aggregate outcome.
    pub outcome: CheckOutcome,
    /// Fixed aggregate diagnostic.
    #[serde(deserialize_with = "required_option")]
    pub reason_code: Option<CheckReason>,
    /// UTC completion timestamp.
    #[schemars(extend("format" = "date-time"))]
    pub completed_at: String,
    /// Elapsed time across all routing and probes.
    #[schemars(range(min = 0))]
    pub elapsed_ms: f64,
    /// Server evidence, absent for runtime scope.
    #[serde(deserialize_with = "required_option")]
    pub server: Option<ServerObservation>,
    /// Runtime evidence when the binding is verified.
    #[serde(deserialize_with = "required_option")]
    pub runtime: Option<RuntimeObservation>,
    /// Exactly the attempted stages, in order.
    pub checks: Vec<CheckObservation>,
}

/// Requested health scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CheckScope {
    /// Runtime server.
    Server,
    /// Agent in a selected runtime.
    Runtime,
}

pub(crate) fn unobserved_report(request: &CheckRequest, reason: CheckReason) -> CheckReport {
    let unsupported = matches!(
        reason,
        CheckReason::UnsupportedScope
            | CheckReason::UnsupportedLevel
            | CheckReason::UnsupportedAdapter
    );
    let server = matches!(request, CheckRequest::Server { .. });
    CheckReport {
        schema_version: CheckVersion::V1Alpha1,
        scope: if server {
            CheckScope::Server
        } else {
            CheckScope::Runtime
        },
        runtime_id: match request {
            CheckRequest::Runtime { runtime_id, .. } => Some(runtime_id.clone()),
            _ => None,
        },
        requested_level: request.level(),
        outcome: if unsupported {
            CheckOutcome::Unsupported
        } else {
            CheckOutcome::Unknown
        },
        reason_code: Some(reason),
        completed_at: utc_now(),
        elapsed_ms: 0.0,
        server: (server && !unsupported).then_some(ServerObservation {
            state: ServerState::Unknown,
        }),
        runtime: None,
        checks: if server && !unsupported {
            vec![CheckObservation {
                name: CheckStage::Server,
                outcome: CheckOutcome::Unknown,
                reason_code: Some(reason),
                source: CheckSource::ServerTransport,
                observed_at: None,
            }]
        } else {
            vec![]
        },
    }
}

fn valid_server_report(report: &CheckReport) -> bool {
    let completed = DateTime::parse_from_rfc3339(&report.completed_at).ok();
    if report.scope != CheckScope::Server
        || report.runtime_id.is_some()
        || report.requested_level.is_some()
        || report.runtime.is_some()
        || !report.elapsed_ms.is_finite()
        || report.elapsed_ms < 0.0
        || !completed
            .as_ref()
            .is_some_and(|time| time.offset().local_minus_utc() == 0)
        || report.checks.len() != 1
    {
        return false;
    }
    let Some(server) = &report.server else {
        return false;
    };
    let expected = match server.state {
        ServerState::Responding => (CheckOutcome::Passed, None),
        ServerState::Stopping => (CheckOutcome::Failed, Some(CheckReason::ServerStopping)),
        ServerState::Unknown
            if matches!(
                report.reason_code,
                Some(CheckReason::ProbeTimeout | CheckReason::ProbeError)
            ) =>
        {
            (CheckOutcome::Unknown, report.reason_code)
        }
        _ => return false,
    };
    let check = &report.checks[0];
    let observed = server.state != ServerState::Unknown;
    (report.outcome, report.reason_code) == expected
        && (check.outcome, check.reason_code) == expected
        && check.name == CheckStage::Server
        && check.source
            == if observed {
                CheckSource::Server
            } else {
                CheckSource::ServerTransport
            }
        && if observed {
            check
                .observed_at
                .as_ref()
                .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
                .is_some_and(|time| time.offset().local_minus_utc() == 0 && Some(time) <= completed)
        } else {
            check.observed_at.is_none()
        }
}

fn valid_remote_report(report: &CheckReport, request: &CheckRequest) -> bool {
    let CheckRequest::Runtime {
        runtime_id, level, ..
    } = request
    else {
        return valid_server_report(report);
    };
    let completed = DateTime::parse_from_rfc3339(&report.completed_at).ok();
    if report.scope != CheckScope::Runtime
        || report.runtime_id.as_ref() != Some(runtime_id)
        || report.requested_level != Some(*level)
        || report.server.is_some()
        || !report.elapsed_ms.is_finite()
        || report.elapsed_ms < 0.0
        || (report.outcome == CheckOutcome::Passed) != report.reason_code.is_none()
        || !completed
            .as_ref()
            .is_some_and(|time| time.offset().local_minus_utc() == 0)
    {
        return false;
    }
    let levels = [CheckStage::Live, CheckStage::Active, CheckStage::Ready];
    let count = match level {
        CheckLevel::Live => 1,
        CheckLevel::Active => 2,
        CheckLevel::Ready => 3,
        _ => return false,
    };
    if report.checks.len() > count {
        return false;
    }
    for (index, check) in report.checks.iter().enumerate() {
        if check.name != levels[index]
            || (check.outcome == CheckOutcome::Passed) != check.reason_code.is_none()
            || index + 1 < report.checks.len() && check.outcome != CheckOutcome::Passed
        {
            return false;
        }
        if let Some(observed) = &check.observed_at
            && !DateTime::parse_from_rfc3339(observed)
                .is_ok_and(|time| time.offset().local_minus_utc() == 0 && Some(time) <= completed)
        {
            return false;
        }
        if check.outcome == CheckOutcome::Passed
            && (check.observed_at.is_none()
                || check.source
                    != [
                        CheckSource::PiSession,
                        CheckSource::PiInterface,
                        CheckSource::PiSetup,
                    ][index])
        {
            return false;
        }
    }
    if report.outcome == CheckOutcome::Passed
        && (report.checks.len() != count
            || !report.runtime.as_ref().is_some_and(|runtime| {
                runtime.state == CheckRuntimeState::Active
                    && matches!(
                        runtime.activity,
                        RuntimeActivity::Idle | RuntimeActivity::Busy
                    )
            }))
    {
        return false;
    }
    if let Some(last) = report.checks.last() {
        let invalidated = report.outcome == CheckOutcome::Unknown
            && report.reason_code == Some(CheckReason::ObservationChanged)
            && report.runtime.is_none();
        if last.outcome != CheckOutcome::Passed
            && !invalidated
            && (report.outcome, report.reason_code) != (last.outcome, last.reason_code)
        {
            return false;
        }
        if last.outcome == CheckOutcome::Passed
            && report.outcome != CheckOutcome::Passed
            && !matches!(
                report.reason_code,
                Some(
                    CheckReason::ObservationChanged
                        | CheckReason::ProbeTimeout
                        | CheckReason::UnsupportedLevel
                )
            )
        {
            return false;
        }
    } else if report.outcome == CheckOutcome::Passed {
        return false;
    }
    true
}

fn probe_remote(
    environment: &EnvironmentHandle,
    request: &CheckRequest,
    started: Instant,
) -> CheckReport {
    let agent = if let CheckRequest::Runtime { runtime_id, .. } = request {
        let Some(agent) = crate::runtime::remote_check_agent(environment, runtime_id)
            .ok()
            .flatten()
        else {
            return unobserved_report(request, CheckReason::RuntimeUnavailable);
        };
        Some(agent)
    } else {
        None
    };
    let remaining = request.timeout_seconds() - started.elapsed().as_secs_f64();
    if remaining <= 0.0 {
        return unobserved_report(request, CheckReason::ProbeTimeout);
    }
    let command = crate::task_command::TaskCommand::Check {
        timeout_seconds: remaining,
        agent: agent.clone(),
        level: request.level(),
        runtime_id: match request {
            CheckRequest::Runtime { runtime_id, .. } => Some(runtime_id.clone()),
            _ => None,
        },
    };
    let response = crate::environment::command_runtime(
        environment,
        &command,
        remaining.ceil().clamp(1.0, 320.0) as u64,
    );
    let mut report = match response {
        Ok(crate::task_command::TaskCommandResponse::Returned { value }) => {
            serde_json::from_value::<CheckReport>(value)
                .ok()
                .filter(|report| valid_remote_report(report, request))
                .unwrap_or_else(|| unobserved_report(request, CheckReason::ProbeError))
        }
        _ => unobserved_report(request, CheckReason::ProbeError),
    };
    if started.elapsed().as_secs_f64() >= request.timeout_seconds() {
        report = unobserved_report(request, CheckReason::ProbeTimeout);
    }
    if let CheckRequest::Runtime { runtime_id, .. } = request
        && crate::runtime::remote_check_agent(environment, runtime_id)
            .ok()
            .flatten()
            != agent
    {
        report.outcome = CheckOutcome::Unknown;
        report.reason_code = Some(CheckReason::ObservationChanged);
        report.runtime = None;
    }
    report
}

struct Binding {
    runtime: RuntimeHandle,
    levels: Vec<CheckLevel>,
    generation: AtomicU64,
}

pub(crate) fn register(runtime: &RuntimeHandle, levels: Vec<CheckLevel>) {
    let mut bindings = BINDINGS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(previous) = bindings.insert(
        runtime.runtime_id.clone(),
        Arc::new(Binding {
            runtime: runtime.clone(),
            levels,
            generation: AtomicU64::new(0),
        }),
    ) {
        previous.generation.fetch_add(1, Ordering::SeqCst);
    }
}

pub(crate) fn invalidate_runtime(runtime: &RuntimeHandle) {
    let mut bindings = BINDINGS.lock().unwrap_or_else(|e| e.into_inner());
    if bindings
        .get(&runtime.runtime_id)
        .is_some_and(|b| b.runtime == *runtime)
        && let Some(binding) = bindings.remove(&runtime.runtime_id)
    {
        binding.generation.fetch_add(1, Ordering::SeqCst);
    }
}

pub(crate) fn invalidate_environment(environment: &EnvironmentHandle) {
    BINDINGS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|_, binding| {
            if binding.runtime.environment == *environment {
                binding.generation.fetch_add(1, Ordering::SeqCst);
                false
            } else {
                true
            }
        });
}

fn utc_now() -> String {
    DateTime::<Utc>::from(SystemTime::now()).to_rfc3339_opts(SecondsFormat::Nanos, true)
}

/// Check server or runtime health without acquiring an invocation or lifecycle-exchange lock.
///
/// Local runtimes use their authenticated adapter health channel. OpenShell
/// server and runtime checks execute `fabric-agent-ctl check` in the sandbox.
/// Local server checks and other providers return `unsupported_scope`.
/// The reserved operational level returns `unsupported_level` without probing.
/// Timeouts and ownership changes return health evidence without runtime teardown.
pub fn check_environment(
    environment: &EnvironmentHandle,
    request: &CheckRequest,
) -> Result<CheckReport> {
    let started = Instant::now();
    if !request.timeout_seconds().is_finite() || request.timeout_seconds() <= 0.0 {
        return Err(crate::FabricError::InvalidConfig {
            field: "check.timeout_seconds".into(),
            reason: "check timeout must be positive and finite".into(),
        });
    }
    if let CheckRequest::Runtime { runtime_id, .. } = request
        && (runtime_id.trim().is_empty() || runtime_id.contains('\0'))
    {
        return Err(crate::FabricError::InvalidConfig {
            field: "check.runtime_id".into(),
            reason: "runtime ID must be nonempty".into(),
        });
    }
    let mut report = CheckReport {
        schema_version: CheckVersion::V1Alpha1,
        scope: CheckScope::Server,
        runtime_id: None,
        requested_level: request.level(),
        outcome: CheckOutcome::Unsupported,
        reason_code: Some(CheckReason::UnsupportedScope),
        completed_at: utc_now(),
        elapsed_ms: 0.0,
        server: None,
        runtime: None,
        checks: vec![],
    };
    if let CheckRequest::Runtime {
        runtime_id, level, ..
    } = request
    {
        report.scope = CheckScope::Runtime;
        report.runtime_id = Some(runtime_id.clone());
        if environment.provider == "openshell" {
            report = probe_remote(environment, request, started);
        } else if environment.provider == "local" {
            let binding = BINDINGS
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(runtime_id)
                .map(|b| (b.clone(), b.generation.load(Ordering::SeqCst)));
            if let Some((binding, generation)) =
                binding.filter(|(b, _)| b.runtime.environment == *environment)
            {
                report.outcome = CheckOutcome::Passed;
                report.reason_code = None;
                for (probe, stage) in [
                    (CheckLevel::Live, CheckStage::Live),
                    (CheckLevel::Active, CheckStage::Active),
                    (CheckLevel::Ready, CheckStage::Ready),
                ] {
                    if !binding.levels.contains(&probe) {
                        report.outcome = CheckOutcome::Unsupported;
                        report.reason_code = Some(if binding.levels.is_empty() {
                            CheckReason::UnsupportedAdapter
                        } else {
                            CheckReason::UnsupportedLevel
                        });
                        break;
                    }
                    let remaining = request.timeout_seconds() - started.elapsed().as_secs_f64();
                    let result = if remaining <= 0.0 {
                        unobserved(CheckReason::ProbeTimeout)
                    } else {
                        probe_runtime(runtime_id, probe, remaining)
                    };
                    report.outcome = result.outcome;
                    report.reason_code = result.reason_code;
                    report.runtime = Some(RuntimeObservation {
                        state: match result.activity {
                            RuntimeActivity::Idle | RuntimeActivity::Busy => {
                                CheckRuntimeState::Active
                            }
                            RuntimeActivity::Stopped => CheckRuntimeState::Stopped,
                            _ => CheckRuntimeState::Unknown,
                        },
                        activity: result.activity,
                    });
                    report.checks.push(CheckObservation {
                        name: stage,
                        outcome: result.outcome,
                        reason_code: result.reason_code,
                        source: result.source,
                        observed_at: result.observed_at,
                    });
                    if result.outcome != CheckOutcome::Passed || probe == *level {
                        break;
                    }
                }
                if binding.generation.load(Ordering::SeqCst) != generation {
                    report.outcome = CheckOutcome::Unknown;
                    report.reason_code = Some(CheckReason::ObservationChanged);
                    report.runtime = None;
                }
            } else {
                report.outcome = CheckOutcome::Unknown;
                report.reason_code = Some(CheckReason::RuntimeUnavailable);
            }
        }
    }
    if matches!(request, CheckRequest::Server { .. }) && environment.provider == "openshell" {
        report = probe_remote(environment, request, started);
    }
    report.elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    if report.outcome == CheckOutcome::Passed
        && report.elapsed_ms >= request.timeout_seconds() * 1000.0
    {
        report.outcome = CheckOutcome::Unknown;
        report.reason_code = Some(CheckReason::ProbeTimeout);
        report.runtime = None;
    }
    report.completed_at = utc_now();
    Ok(report)
}

fn unobserved(reason: CheckReason) -> AdapterCheckResult {
    AdapterCheckResult {
        outcome: CheckOutcome::Unknown,
        reason_code: Some(reason),
        activity: RuntimeActivity::Unknown,
        source: CheckSource::Adapter,
        observed_at: None,
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Endpoint {
    protocol_version: CheckVersion,
    runtime_id: String,
    port: u16,
    token: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SignedResponse {
    payload: String,
    proof: String,
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(result, "{byte:02x}").expect("write to string");
    }
    result
}

fn health_mac(token: &str, direction: &str, nonce: &str, payload: &str) -> Hmac<Sha256> {
    let mut mac =
        Hmac::<Sha256>::new_from_slice(token.as_bytes()).expect("HMAC accepts any key size");
    for part in [direction, "\n", nonce, "\n", payload] {
        mac.update(part.as_bytes());
    }
    mac
}

fn verify_health_mac(
    token: &str,
    direction: &str,
    nonce: &str,
    payload: &str,
    proof: &str,
) -> bool {
    if proof.len() != 64 || !proof.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return false;
    }
    let mut bytes = [0; 32];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&proof[index * 2..index * 2 + 2], 16).expect("validated hex");
    }
    health_mac(token, direction, nonce, payload)
        .verify_slice(&bytes)
        .is_ok()
}

fn probe_runtime(runtime_id: &str, level: CheckLevel, timeout: f64) -> AdapterCheckResult {
    let started = Instant::now();
    let budget = Duration::try_from_secs_f64(timeout).unwrap_or(Duration::MAX);
    let remaining = || budget.saturating_sub(started.elapsed());
    let result = (|| -> std::io::Result<AdapterCheckResult> {
        let path = crate::runtime::local_host_directory(runtime_id).join("health.endpoint.json");
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        }
        let file = options.open(path)?;
        if !file.metadata()?.is_file() || file.metadata()?.len() > LIMIT {
            return Err(std::io::Error::other("invalid health endpoint file"));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let metadata = file.metadata()?;
            if metadata.uid() != unsafe { libc::getuid() } || metadata.mode() & 0o077 != 0 {
                return Err(std::io::Error::other("invalid health endpoint permissions"));
            }
        }
        let endpoint: Endpoint = serde_json::from_reader(file.take(LIMIT))?;
        if endpoint.runtime_id != runtime_id || endpoint.port == 0 || endpoint.token.len() != 64 {
            return Err(std::io::Error::other("invalid health endpoint"));
        }
        let _ = endpoint.protocol_version;
        let address = SocketAddr::from((Ipv4Addr::LOCALHOST, endpoint.port));
        let mut stream = TcpStream::connect_timeout(&address, remaining())?;
        stream.set_write_timeout(Some(remaining()))?;
        let mut challenge = [0; 32];
        getrandom::fill(&mut challenge)
            .map_err(|_| std::io::Error::other("health challenge unavailable"))?;
        let nonce = hex(&challenge);
        let payload = serde_json::to_string(&serde_json::json!({
            "runtime_id": runtime_id, "level": level,
            "timeout_seconds": remaining().as_secs_f64(),
        }))?;
        let proof = hex(&health_mac(&endpoint.token, "request", &nonce, &payload)
            .finalize()
            .into_bytes());
        let encoded = serde_json::to_vec(
            &serde_json::json!({"nonce": nonce, "payload": payload, "proof": proof}),
        )?;
        stream.write_all(&encoded)?;
        stream.write_all(b"\n")?;
        let mut line = Vec::new();
        loop {
            let budget = remaining();
            if budget.is_zero() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "health deadline",
                ));
            }
            stream.set_read_timeout(Some(budget))?;
            let mut chunk = [0; 1024];
            let count = stream.read(&mut chunk)?;
            if count == 0 {
                return Err(std::io::Error::other("truncated health response"));
            }
            line.extend_from_slice(&chunk[..count]);
            if line.len() > LIMIT as usize {
                return Err(std::io::Error::other("oversized health response"));
            }
            if let Some(end) = line.iter().position(|byte| *byte == b'\n') {
                if end + 1 != line.len() {
                    return Err(std::io::Error::other("invalid health framing"));
                }
                break;
            }
        }
        let response: SignedResponse = serde_json::from_slice(&line)?;
        if !verify_health_mac(
            &endpoint.token,
            "response",
            &nonce,
            &response.payload,
            &response.proof,
        ) {
            return Err(std::io::Error::other("unauthenticated health response"));
        }
        let observation: AdapterCheckResult = serde_json::from_str(&response.payload)?;
        let valid_time = observation.observed_at.as_ref().is_some_and(|value| {
            DateTime::parse_from_rfc3339(value).is_ok_and(|time| {
                time.offset().local_minus_utc() == 0
                    && time <= DateTime::<Utc>::from(SystemTime::now())
            })
        });
        let expected_source = match level {
            CheckLevel::Live => CheckSource::PiSession,
            CheckLevel::Active => CheckSource::PiInterface,
            CheckLevel::Ready => CheckSource::PiSetup,
            CheckLevel::Operational => CheckSource::Adapter,
        };
        if (observation.outcome == CheckOutcome::Passed) != observation.reason_code.is_none()
            || (observation.outcome == CheckOutcome::Passed
                && (!valid_time
                    || observation.source != expected_source
                    || !matches!(
                        observation.activity,
                        RuntimeActivity::Idle | RuntimeActivity::Busy
                    )))
            || observation.observed_at.is_some() && !valid_time
        {
            return Err(std::io::Error::other("invalid health observation"));
        }
        Ok(observation)
    })();
    if started.elapsed() >= budget {
        return unobserved(CheckReason::ProbeTimeout);
    }
    match result {
        Ok(result) => result,
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
            ) =>
        {
            unobserved(CheckReason::ProbeTimeout)
        }
        Err(_) => unobserved(CheckReason::ProbeError),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_does_not_precede_submillisecond_evidence() {
        let observation = DateTime::<Utc>::from(SystemTime::now());
        let completion = DateTime::parse_from_rfc3339(&utc_now()).unwrap();
        assert!(completion >= observation);
    }
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;

    static NEXT_TEST: AtomicU64 = AtomicU64::new(0);

    fn runtime() -> RuntimeHandle {
        let number = NEXT_TEST.fetch_add(1, Ordering::SeqCst);
        serde_json::from_value(serde_json::json!({
            "runtime_id": format!("health-unit-{}-{number}", std::process::id()),
            "runtime_binding": format!("binding-{number}"), "agent_name": "test", "harness": "pi",
            "adapter_kind": "process", "adapter_id": "nvidia.fabric.pi",
            "environment": {"environment_id": format!("health-env-{number}"), "provider": "local",
                "control_location": "external_control", "ownership": "caller_owned"}
        }))
        .unwrap()
    }

    fn request(runtime: &RuntimeHandle, level: CheckLevel, seconds: f64) -> CheckRequest {
        CheckRequest::Runtime {
            schema_version: CheckVersion::V1Alpha1,
            runtime_id: runtime.runtime_id.clone(),
            level,
            timeout_seconds: seconds,
        }
    }

    fn endpoint(runtime: &RuntimeHandle) -> TcpListener {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let directory = crate::runtime::local_host_directory(&runtime.runtime_id);
        std::fs::create_dir_all(&directory).unwrap();
        let endpoint = directory.join("health.endpoint.json");
        std::fs::write(
            &endpoint,
            serde_json::to_vec(&serde_json::json!({
                "protocol_version": "fabric.check/v1alpha1", "runtime_id": runtime.runtime_id,
                "port": listener.local_addr().unwrap().port(), "token": "1".repeat(64)
            }))
            .unwrap(),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(endpoint, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        listener
    }

    fn peer(
        runtime: &RuntimeHandle,
        count: usize,
        respond: impl Fn(usize, CheckLevel) -> serde_json::Value + Send + 'static,
    ) -> thread::JoinHandle<Vec<CheckLevel>> {
        peer_messages(runtime, count, move |index, level, nonce| {
            let payload = respond(index, level).to_string();
            let proof = hex(&health_mac(&"1".repeat(64), "response", nonce, &payload)
                .finalize()
                .into_bytes());
            serde_json::json!({"payload": payload, "proof": proof})
        })
    }

    fn peer_messages(
        runtime: &RuntimeHandle,
        count: usize,
        respond: impl Fn(usize, CheckLevel, &str) -> serde_json::Value + Send + 'static,
    ) -> thread::JoinHandle<Vec<CheckLevel>> {
        let listener = endpoint(runtime);
        listener.set_nonblocking(true).unwrap();
        thread::spawn(move || {
            let started = Instant::now();
            let mut calls = vec![];
            while calls.len() < count && started.elapsed() < Duration::from_secs(5) {
                let (mut socket, _) = match listener.accept() {
                    Ok(connection) => connection,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("test peer accept: {error}"),
                };
                socket.set_nonblocking(false).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(1)))
                    .unwrap();
                let mut line = String::new();
                BufReader::new(socket.try_clone().unwrap())
                    .read_line(&mut line)
                    .unwrap();
                assert!(
                    !line.contains(&"1".repeat(64)),
                    "endpoint secret crossed the wire"
                );
                let envelope: serde_json::Value = serde_json::from_str(&line).unwrap();
                let nonce = envelope["nonce"].as_str().unwrap();
                let payload = envelope["payload"].as_str().unwrap();
                assert!(verify_health_mac(
                    &"1".repeat(64),
                    "request",
                    nonce,
                    payload,
                    envelope["proof"].as_str().unwrap()
                ));
                let request: serde_json::Value = serde_json::from_str(payload).unwrap();
                let level = serde_json::from_value(request["level"].clone()).unwrap();
                let response = respond(calls.len(), level, nonce);
                calls.push(level);
                let _ = writeln!(socket, "{response}");
            }
            calls
        })
    }

    fn passed(level: CheckLevel) -> serde_json::Value {
        serde_json::json!({"outcome": "passed", "reason_code": null, "activity": "busy",
            "source": match level {
                CheckLevel::Live => "fabric.adapter.pi.session",
                CheckLevel::Active => "fabric.adapter.pi.interface",
                _ => "fabric.adapter.pi.setup",
            }, "observed_at": utc_now()})
    }

    fn cleanup(runtime: &RuntimeHandle) {
        invalidate_runtime(runtime);
        std::fs::remove_dir_all(crate::runtime::local_host_directory(&runtime.runtime_id)).unwrap();
    }

    #[test]
    fn probes_are_ordered_and_stop_at_the_first_nonpassing_level() {
        let runtime = runtime();
        register(
            &runtime,
            vec![CheckLevel::Live, CheckLevel::Active, CheckLevel::Ready],
        );
        let peer = peer(&runtime, 2, |_, level| {
            let mut result = passed(level);
            if level == CheckLevel::Active {
                result["outcome"] = "failed".into();
                result["reason_code"] = "endpoint_unavailable".into();
            }
            result
        });
        let report = check_environment(
            &runtime.environment,
            &request(&runtime, CheckLevel::Ready, 3.0),
        )
        .unwrap();
        assert_eq!(report.outcome, CheckOutcome::Failed);
        assert_eq!(report.checks.len(), 2);
        assert_eq!(
            peer.join().unwrap(),
            vec![CheckLevel::Live, CheckLevel::Active]
        );
        cleanup(&runtime);
    }

    #[test]
    fn a_probe_timeout_does_not_invalidate_or_terminate_the_runtime() {
        let runtime = runtime();
        register(&runtime, vec![CheckLevel::Live]);
        let peer = peer(&runtime, 2, |index, level| {
            if index == 0 {
                thread::sleep(Duration::from_millis(1250));
            }
            passed(level)
        });
        let report = check_environment(
            &runtime.environment,
            &request(&runtime, CheckLevel::Live, 1.0),
        )
        .unwrap();
        assert_eq!(report.reason_code, Some(CheckReason::ProbeTimeout));
        assert_eq!(report.runtime.unwrap().state, CheckRuntimeState::Unknown);
        let next = check_environment(
            &runtime.environment,
            &request(&runtime, CheckLevel::Live, 3.0),
        )
        .unwrap();
        assert_eq!(next.outcome, CheckOutcome::Passed);
        assert_eq!(peer.join().unwrap().len(), 2);
        cleanup(&runtime);
    }

    #[test]
    fn later_levels_use_the_remaining_budget() {
        let runtime = runtime();
        register(&runtime, vec![CheckLevel::Live, CheckLevel::Active]);
        let peer = peer(&runtime, 2, |_, level| {
            thread::sleep(Duration::from_millis(600));
            passed(level)
        });
        let report = check_environment(
            &runtime.environment,
            &request(&runtime, CheckLevel::Active, 1.0),
        )
        .unwrap();
        assert_eq!(report.reason_code, Some(CheckReason::ProbeTimeout));
        assert_eq!(report.checks.len(), 2);
        assert_eq!(
            peer.join().unwrap(),
            vec![CheckLevel::Live, CheckLevel::Active]
        );
        cleanup(&runtime);
    }

    #[test]
    fn unauthenticated_listeners_cannot_forge_healthy_observations() {
        let runtime = runtime();
        register(&runtime, vec![CheckLevel::Live]);
        let peer = peer_messages(&runtime, 1, |_, level, _| passed(level));
        let report = check_environment(
            &runtime.environment,
            &request(&runtime, CheckLevel::Live, 3.0),
        )
        .unwrap();
        assert_eq!(report.outcome, CheckOutcome::Unknown);
        assert_eq!(report.reason_code, Some(CheckReason::ProbeError));
        peer.join().unwrap();
        cleanup(&runtime);
    }

    #[test]
    fn signed_responses_cannot_be_replayed_for_a_new_challenge() {
        let runtime = runtime();
        register(&runtime, vec![CheckLevel::Live]);
        let previous = Mutex::new(None);
        let peer = peer_messages(&runtime, 2, move |index, level, nonce| {
            let mut previous = previous.lock().unwrap();
            if index == 0 {
                let payload = passed(level).to_string();
                let proof = hex(&health_mac(&"1".repeat(64), "response", nonce, &payload)
                    .finalize()
                    .into_bytes());
                *previous = Some(serde_json::json!({"payload": payload, "proof": proof}));
            }
            previous.clone().unwrap()
        });
        let request = request(&runtime, CheckLevel::Live, 3.0);
        assert_eq!(
            check_environment(&runtime.environment, &request)
                .unwrap()
                .outcome,
            CheckOutcome::Passed
        );
        let replayed = check_environment(&runtime.environment, &request).unwrap();
        assert_eq!(replayed.outcome, CheckOutcome::Unknown);
        assert_eq!(replayed.reason_code, Some(CheckReason::ProbeError));
        peer.join().unwrap();
        cleanup(&runtime);
    }

    #[test]
    fn partial_replies_cannot_restart_the_read_deadline() {
        let runtime = runtime();
        register(&runtime, vec![CheckLevel::Live]);
        let listener = endpoint(&runtime);
        listener.set_nonblocking(true).unwrap();
        let peer = thread::spawn(move || {
            let started = Instant::now();
            let mut socket = loop {
                match listener.accept() {
                    Ok((socket, _)) => break socket,
                    Err(error)
                        if error.kind() == std::io::ErrorKind::WouldBlock
                            && started.elapsed() < Duration::from_secs(2) =>
                    {
                        thread::sleep(Duration::from_millis(1));
                    }
                    Err(error) => panic!("test peer accept: {error}"),
                }
            };
            socket.set_nonblocking(false).unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(1)))
                .unwrap();
            let mut request = String::new();
            BufReader::new(socket.try_clone().unwrap())
                .read_line(&mut request)
                .unwrap();
            for chunk in [b"{".as_slice(), b"}".as_slice(), b"\n".as_slice()] {
                thread::sleep(Duration::from_millis(180));
                let _ = socket.write_all(chunk);
            }
        });
        let report = check_environment(
            &runtime.environment,
            &request(&runtime, CheckLevel::Live, 0.3),
        )
        .unwrap();
        assert_eq!(report.reason_code, Some(CheckReason::ProbeTimeout));
        assert!(
            report.elapsed_ms < 450.0,
            "partial response restarted the deadline"
        );
        peer.join().unwrap();
        cleanup(&runtime);
    }

    #[test]
    fn stop_and_environment_release_invalidate_in_flight_evidence() {
        for release_environment in [false, true] {
            let runtime = runtime();
            register(&runtime, vec![CheckLevel::Live]);
            let (entered_tx, entered_rx) = mpsc::channel();
            let (release_tx, release_rx) = mpsc::channel();
            let peer = peer(&runtime, 1, move |_, level| {
                entered_tx.send(()).unwrap();
                release_rx.recv_timeout(Duration::from_secs(2)).unwrap();
                passed(level)
            });
            let selected = runtime.clone();
            let checking = thread::spawn(move || {
                check_environment(
                    &selected.environment,
                    &request(&selected, CheckLevel::Live, 3.0),
                )
                .unwrap()
            });
            entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
            if release_environment {
                invalidate_environment(&runtime.environment);
            } else {
                invalidate_runtime(&runtime);
            }
            release_tx.send(()).unwrap();
            let report = checking.join().unwrap();
            assert_eq!(report.outcome, CheckOutcome::Unknown);
            assert_eq!(report.reason_code, Some(CheckReason::ObservationChanged));
            assert!(report.runtime.is_none());
            peer.join().unwrap();
            cleanup(&runtime);
        }
    }

    #[test]
    fn an_environment_binding_mismatch_never_probes() {
        let runtime = runtime();
        register(&runtime, vec![CheckLevel::Live]);
        let mut environment = runtime.environment.clone();
        environment.connection.insert("changed".into(), true.into());
        let report =
            check_environment(&environment, &request(&runtime, CheckLevel::Live, 3.0)).unwrap();
        assert_eq!(report.reason_code, Some(CheckReason::RuntimeUnavailable));
        assert!(report.checks.is_empty());
        invalidate_runtime(&runtime);
    }

    #[test]
    fn invalid_peer_evidence_is_a_safe_unknown_result() {
        let runtime = runtime();
        register(&runtime, vec![CheckLevel::Live]);
        let peer = peer(&runtime, 3, |index, level| {
            let mut result = passed(level);
            match index {
                0 => {
                    result.as_object_mut().unwrap().remove("reason_code");
                }
                1 => {
                    result["source"] = "PRIVATE".into();
                }
                _ => {
                    result["raw"] = "PRIVATE".into();
                }
            }
            result
        });
        for _ in 0..3 {
            let report = check_environment(
                &runtime.environment,
                &request(&runtime, CheckLevel::Live, 3.0),
            )
            .unwrap();
            assert_eq!(report.reason_code, Some(CheckReason::ProbeError));
            assert!(!serde_json::to_string(&report).unwrap().contains("PRIVATE"));
        }
        peer.join().unwrap();
        cleanup(&runtime);
    }
}
