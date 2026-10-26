// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Health check requests, reports, and shared contracts.

use crate::{EnvironmentHandle, Result};
use chrono::{DateTime, SecondsFormat, Utc};
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Deserializer, Serialize};
use std::time::{Instant, SystemTime};

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
    /// Adapter observation of native process, session, or service liveness.
    #[serde(rename = "fabric.adapter.liveness")]
    AdapterLiveness,
    /// Adapter observation of the selected request interface.
    #[serde(rename = "fabric.adapter.interface")]
    AdapterInterface,
    /// Adapter observation of required setup or native readiness.
    #[serde(rename = "fabric.adapter.readiness")]
    AdapterReadiness,
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

fn adapter_check_result_schema(schema: &mut Schema) {
    required_fields(schema);
    let properties = schema.get("properties").expect("result properties").clone();
    let required = schema.get("required").expect("result fields").clone();
    let mut passed = properties.clone();
    passed["outcome"] = serde_json::json!({"const": "passed"});
    passed["reason_code"] = serde_json::json!({"type": "null"});
    passed["observed_at"] = serde_json::json!({"type": "string"});
    let mut nonpassing = properties;
    nonpassing["outcome"] = serde_json::json!({"enum": ["failed", "unknown", "unsupported"]});
    nonpassing["reason_code"] = serde_json::json!({"$ref": "#/$defs/CheckReason"});
    schema.insert(
        "oneOf".into(),
        serde_json::json!([
            {"type": "object", "additionalProperties": false, "properties": passed, "required": required},
            {"type": "object", "additionalProperties": false, "properties": nonpassing, "required": required}
        ]),
    );
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
#[schemars(transform = adapter_check_result_schema)]
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

fn utc_now() -> String {
    DateTime::<Utc>::from(SystemTime::now()).to_rfc3339_opts(SecondsFormat::Nanos, true)
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

/// Check runtime server health through the command bridge.
///
/// OpenShell server checks map to `fabric-agent-ctl check` in the selected
/// sandbox. Local server checks and runtime checks return `unsupported_scope`.
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
    let mut report = unobserved_report(request, CheckReason::UnsupportedScope);
    if environment.provider == "openshell" && matches!(request, CheckRequest::Server { .. }) {
        let remaining = request.timeout_seconds() - started.elapsed().as_secs_f64();
        if remaining <= 0.0 {
            report = unobserved_report(request, CheckReason::ProbeTimeout);
        } else {
            let command = crate::task_command::TaskCommand::Check {
                timeout_seconds: remaining,
            };
            let response = crate::environment::command_runtime(
                environment,
                &command,
                remaining.ceil().clamp(1.0, 320.0) as u64,
            );
            report = match response {
                Ok(crate::task_command::TaskCommandResponse::Returned { value }) => {
                    serde_json::from_value::<CheckReport>(value)
                        .ok()
                        .filter(valid_server_report)
                        .unwrap_or_else(|| unobserved_report(request, CheckReason::ProbeError))
                }
                Err(crate::FabricError::EnvironmentProviderOperation { code, .. })
                    if code == "probe_timeout" =>
                {
                    unobserved_report(request, CheckReason::ProbeTimeout)
                }
                _ => unobserved_report(request, CheckReason::ProbeError),
            };
        }
        if started.elapsed().as_secs_f64() >= request.timeout_seconds() {
            report = unobserved_report(request, CheckReason::ProbeTimeout);
        }
    }
    report.completed_at = utc_now();
    report.elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    if report.outcome == CheckOutcome::Passed
        && report.elapsed_ms >= request.timeout_seconds() * 1000.0
    {
        report = unobserved_report(request, CheckReason::ProbeTimeout);
        report.completed_at = utc_now();
        report.elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    }
    Ok(report)
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
}
