// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One invocation's record: the start line before the request, the end line
//! after the invocation.
//!
//! Record I/O errors do not replace the invocation result. A failed create,
//! write or sync stops the record for that invocation and adds a
//! `record_write_failed` event to the result; nothing here returns a record
//! error to the caller.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use serde_json::Value;

use super::cgroup::CgroupReader;
use super::lines::{
    ContextValue, EndArtifact, EndError, EndLine, EndUsage, LimitType, LineBody, ModelName,
    NotRecorded, OomKills, RecordLine, RecordSchema, RecordVersion, StartLine, Stop, StopLimit,
    StopReason, cut_text, recorded_code,
};
use super::stop::{StopFacts, stop_cause};
use super::writer::{RECORD_FILE_NAME, RecordWriter};
use crate::config::RunPlan;
use crate::error::{FabricError, Result};
use crate::runtime::{
    ArtifactManifest, ArtifactRef, ErrorInfo, ErrorStage, HOST_CRASHED, HOST_TIMEOUT,
    INVALID_AGENT_RUN_RESULT, InvocationHandle, RunRequest, RunResult, RunStatus, RunUsage,
    RuntimeHandle, event_with_metadata, unique_artifact_name,
};
use crate::time::rfc3339_millis;

/// Name of the result artifact that points at the record.
const RECORD_ARTIFACT_NAME: &str = "record";
/// Kind of the result artifact that points at the record.
const RECORD_ARTIFACT_KIND: &str = "run_record";
/// Event added to the result when the record could not be written.
const RECORD_WRITE_FAILED: &str = "record_write_failed";
/// `RunRequest.context` keys whose string or integer values the start line
/// keeps: task identity and the Harbor trial link.
const RECORDED_CONTEXT_KEYS: [&str; 7] = [
    "task_id",
    "task_revision",
    "attempt",
    "step",
    "harbor_session_id",
    "harbor_context_id",
    "harbor_trial_name",
];
/// End-line error code for a failed Fabric file write: `adapter-invocation.json`
/// before the request, or a result artifact after the adapter returned.
const ARTIFACT_WRITE_FAILED: &str = "artifact_write_failed";
/// End-line error code for any other Fabric error after the record started.
const FABRIC_ERROR: &str = "fabric_error";
/// Error code an adapter reports when it stops at the deadline core gave it,
/// whether in a failed result or as a lifecycle error.
const ADAPTER_TIMEOUT_CODE: &str = "timeout";

/// What the start line needs from core's invocation path.
pub(crate) struct RecordStart<'a> {
    pub plan: &'a RunPlan,
    pub runtime: &'a RuntimeHandle,
    pub invocation: &'a InvocationHandle,
    pub request: &'a RunRequest,
    /// The invocation folder, `<artifact root>/.fabric/<runtime_id>/<invocation_id>`.
    pub fabric_home: &'a Path,
    /// The host timeout that applies to this invocation.
    pub timeout: Duration,
    /// Reads the container's out-of-memory kill count.
    pub cgroup: CgroupReader,
}

/// The adapter host's wait status, when it exited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HostExit {
    /// Exit code.
    pub exit_code: Option<i32>,
    /// Signal that ended the host (Unix only).
    pub exit_signal: Option<i32>,
}

impl HostExit {
    /// Read a wait status.
    pub(crate) fn from_status(status: std::process::ExitStatus) -> Self {
        #[cfg(unix)]
        let exit_signal = std::os::unix::process::ExitStatusExt::signal(&status);
        #[cfg(not(unix))]
        let exit_signal = None;
        Self {
            exit_code: status.code(),
            exit_signal,
        }
    }
}

/// The record of one running invocation.
#[derive(Debug)]
pub(crate) struct InvocationRecorder {
    runtime_id: String,
    invocation_id: String,
    request_id: String,
    path: PathBuf,
    started: Instant,
    writer: Option<RecordWriter>,
    wrote_start: bool,
    failure: Option<io::Error>,
    timeout: Duration,
    cgroup: CgroupReader,
    oom_kills_at_start: OomKills,
    exclusive_container: bool,
    exchange: Option<ExchangeEnd>,
}

/// What core saw when the exchange with the adapter host ended. Core reads
/// it under the host lock, before it stops a timed-out host and before a
/// later invocation can run, so every value describes this invocation.
#[derive(Debug, Clone)]
pub(crate) struct ExchangeEnd {
    /// Time from the start line to the end of the exchange, before core's
    /// own cleanup.
    elapsed_ms: u64,
    /// The container's out-of-memory kill count at that moment, or the
    /// larger count read after a crashed host's exit.
    oom_kills: OomKills,
    /// Whether the exchange raised an error.
    failed: bool,
    /// The host's wait status after a failed exchange, when it had exited.
    host_exit: Option<HostExit>,
}

impl ExchangeEnd {
    /// Mark the exchange failed, with the host's wait status when it exited.
    pub(crate) fn with_host_exit(self, host_exit: Option<HostExit>) -> Self {
        Self {
            failed: true,
            host_exit,
            ..self
        }
    }
}

impl InvocationRecorder {
    /// Create the record and write its start line.
    pub(crate) fn start(start: RecordStart<'_>) -> Self {
        let path = start.fabric_home.join(RECORD_FILE_NAME);
        let oom_kills_at_start = start.cgroup.oom_kills();
        let mut recorder = Self {
            runtime_id: start.invocation.runtime_id.clone(),
            invocation_id: start.invocation.invocation_id.clone(),
            request_id: start.invocation.request_id.clone(),
            started: Instant::now(),
            writer: None,
            wrote_start: false,
            failure: None,
            path,
            timeout: start.timeout,
            cgroup: start.cgroup.clone(),
            oom_kills_at_start,
            exclusive_container: exclusive_container(start.plan),
            exchange: None,
        };
        match RecordWriter::create(&recorder.path) {
            Ok(writer) => {
                recorder.writer = Some(writer);
                recorder.append(LineBody::Start(start_line(&start, oom_kills_at_start)));
                recorder.wrote_start = recorder.failure.is_none();
            }
            Err(error) => recorder.failure = Some(error),
        }
        recorder
    }

    /// The end values at `ended_at`, when the exchange with the host
    /// returned: the container's out-of-memory kill count, read now. Core
    /// calls this under the host lock.
    pub(crate) fn read_exchange_end(&self, ended_at: Instant) -> ExchangeEnd {
        ExchangeEnd {
            elapsed_ms: millis(ended_at.saturating_duration_since(self.started)),
            oom_kills: self.cgroup.oom_kills(),
            failed: false,
            host_exit: None,
        }
    }

    /// `end` with the larger of its out-of-memory kill count and a count
    /// read now. Core calls this after a crashed host's exit is observed:
    /// the kernel counts a kill after it sends SIGKILL, so a host killed for
    /// another process's allocation can exit before the count rises.
    pub(crate) fn recount_oom_kills(&self, end: ExchangeEnd) -> ExchangeEnd {
        let oom_kills = match (end.oom_kills, self.cgroup.oom_kills()) {
            (OomKills::Count(at_end), OomKills::Count(now)) => OomKills::Count(at_end.max(now)),
            (at_end, _) => at_end,
        };
        ExchangeEnd { oom_kills, ..end }
    }

    /// Keep what the exchange saw for the end line.
    pub(crate) fn exchange_ended(&mut self, end: ExchangeEnd) {
        self.exchange = Some(end);
    }

    /// Write and sync the end line for `outcome`, then return it. A result
    /// gains the `record` artifact when the start line was written, and a
    /// `record_write_failed` event when any write or the sync failed.
    pub(crate) fn finish(mut self, outcome: Result<RunResult>) -> Result<RunResult> {
        let elapsed_ms = millis(self.started.elapsed());
        // Core always records the exchange's end; reading here is a fallback.
        let oom_kills = self
            .exchange
            .as_ref()
            .map_or_else(|| self.cgroup.oom_kills(), |exchange| exchange.oom_kills);
        let stop = self.stop(&outcome, elapsed_ms, oom_kills);
        let end = match &outcome {
            Ok(result) => result_end_line(result, elapsed_ms, oom_kills, stop),
            Err(error) => error_end_line(error, elapsed_ms, oom_kills, stop),
        };
        self.append(LineBody::End(end));
        if let Some(writer) = &self.writer
            && let Err(error) = writer.sync()
        {
            self.failure = Some(error);
        }
        let mut result = outcome?;
        if self.wrote_start {
            let name = unique_artifact_name(&result.artifacts, RECORD_ARTIFACT_NAME);
            result.artifacts.artifacts.push(ArtifactRef {
                name,
                kind: RECORD_ARTIFACT_KIND.to_string(),
                path: self.path.clone(),
                media_type: Some("application/jsonl".to_string()),
                metadata: BTreeMap::new(),
            });
        }
        if let Some(error) = &self.failure {
            result.events.push(event_with_metadata(
                RECORD_WRITE_FAILED,
                format!("run record write failed; the record stops here: {error}"),
                BTreeMap::from([
                    (
                        "runtime_id".to_string(),
                        Value::from(self.runtime_id.as_str()),
                    ),
                    (
                        "invocation_id".to_string(),
                        Value::from(self.invocation_id.as_str()),
                    ),
                ]),
            ));
        }
        Ok(result)
    }

    /// The end line's `stop`, or `None` when it holds nothing (a success).
    fn stop(
        &self,
        outcome: &Result<RunResult>,
        elapsed_ms: u64,
        oom_kills_at_end: OomKills,
    ) -> Option<Stop> {
        let exchange = self.exchange.as_ref();
        let oom_rose = self.exclusive_container
            && matches!(
                (self.oom_kills_at_start, oom_kills_at_end),
                (OomKills::Count(start), OomKills::Count(end)) if end > start
            );
        // `forced` is set exactly when a timeout stopped the invocation.
        let (facts, forced) = match outcome {
            Ok(result) => {
                let failed = result.status == RunStatus::Failed;
                let adapter_timeout = failed
                    && result
                        .error
                        .as_ref()
                        .is_some_and(|error| error.code == ADAPTER_TIMEOUT_CODE);
                let facts = StopFacts {
                    status: result.status,
                    fabric_failed: false,
                    oom_rose,
                    timed_out: adapter_timeout,
                    host_exited: false,
                    agent_error: failed,
                };
                (facts, adapter_timeout.then_some(false))
            }
            // A raised error is never the agent's own failure: the adapter
            // contract returns a target failure as a `failed` result. An
            // out-of-memory rise outranks an error from the exchange itself,
            // such as an adapter that raised after its harness was killed;
            // core's own errors after a complete response keep rank 1.
            Err(error) => {
                let code = match error {
                    FabricError::AdapterLifecycleOperation { code, .. } => Some(code.as_str()),
                    _ => None,
                };
                let forced = code == Some(HOST_TIMEOUT);
                let timed_out = forced || code == Some(ADAPTER_TIMEOUT_CODE);
                let host_exited = code == Some(HOST_CRASHED);
                let exchange_error = exchange.is_some_and(|exchange| exchange.failed);
                let facts = StopFacts {
                    status: RunStatus::Failed,
                    fabric_failed: !(timed_out || host_exited || (exchange_error && oom_rose)),
                    oom_rose,
                    timed_out,
                    host_exited,
                    agent_error: false,
                };
                (facts, timed_out.then_some(forced))
            }
        };
        let usage = exchange.map_or(elapsed_ms, |exchange| exchange.elapsed_ms);
        let host_exit = exchange.and_then(|exchange| exchange.host_exit);
        let stop = Stop {
            cause: stop_cause(facts),
            reason: forced.map(|_| StopReason::Timeout),
            forced,
            limit: forced.map(|_| StopLimit {
                limit_type: LimitType::Time,
                value: millis(self.timeout),
                usage,
            }),
            exit_code: host_exit.and_then(|exit| exit.exit_code),
            exit_signal: host_exit.and_then(|exit| exit.exit_signal),
        };
        (stop != Stop::default()).then_some(stop)
    }

    /// Append one line. After the first failure the record stops.
    fn append(&mut self, body: LineBody) {
        let Some(writer) = self.writer.as_mut() else {
            return;
        };
        let line = RecordLine {
            schema: RecordSchema::RunRecord,
            version: RecordVersion::V1Alpha1,
            runtime_id: self.runtime_id.clone(),
            invocation_id: self.invocation_id.clone(),
            request_id: self.request_id.clone(),
            time: rfc3339_millis(SystemTime::now()),
            body,
        };
        if let Err(error) = writer.append(&line) {
            self.failure = Some(error);
            self.writer = None;
        }
    }
}

fn start_line(start: &RecordStart<'_>, container_oom_kills: OomKills) -> StartLine {
    StartLine {
        agent_name: cut_text(&start.runtime.agent_name),
        harness: cut_text(&start.runtime.harness),
        adapter_id: start.runtime.adapter_id.as_deref().map(cut_text),
        models: start
            .plan
            .config
            .models
            .iter()
            .map(|(role, model)| {
                (
                    role.clone(),
                    ModelName {
                        provider: cut_text(&model.provider),
                        model: cut_text(&model.model),
                    },
                )
            })
            .collect(),
        fabric_version: crate::version().to_string(),
        timeout_ms: millis(start.timeout),
        context: start
            .request
            .context
            .iter()
            .map(|(key, value)| (key.clone(), context_value(key, value)))
            .collect(),
        override_keys: override_keys(start.request.overrides.as_ref()),
        exclusive_container: exclusive_container(start.plan),
        container_oom_kills,
    }
}

fn exclusive_container(plan: &RunPlan) -> bool {
    plan.config
        .runtime
        .record
        .as_ref()
        .is_some_and(|record| record.exclusive_container)
}

fn millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// The sorted top-level key names of `overrides`; empty unless it is an object.
fn override_keys(overrides: Option<&Value>) -> Vec<String> {
    let Some(Value::Object(overrides)) = overrides else {
        return Vec::new();
    };
    let mut keys: Vec<String> = overrides.keys().map(|key| cut_text(key)).collect();
    keys.sort();
    keys
}

/// An allowlisted key keeps a string (cut) or integer value; any other key,
/// or an allowlisted key with another value type, is recorded by name only.
fn context_value(key: &str, value: &Value) -> ContextValue {
    if !RECORDED_CONTEXT_KEYS.contains(&key) {
        return ContextValue::NotRecorded(NotRecorded::VALUE);
    }
    match value {
        Value::String(text) => ContextValue::Text(cut_text(text)),
        Value::Number(number) => number
            .as_i64()
            .map_or(ContextValue::NotRecorded(NotRecorded::VALUE), |integer| {
                ContextValue::Integer(integer)
            }),
        _ => ContextValue::NotRecorded(NotRecorded::VALUE),
    }
}

fn result_end_line(
    result: &RunResult,
    elapsed_ms: u64,
    container_oom_kills: OomKills,
    stop: Option<Stop>,
) -> EndLine {
    EndLine {
        status: result.status,
        elapsed_ms,
        error: result.error.as_ref().map(result_error),
        usage: result.usage.as_ref().map(end_usage),
        artifacts: end_artifacts(&result.artifacts),
        container_oom_kills,
        stop,
    }
}

fn error_end_line(
    error: &FabricError,
    elapsed_ms: u64,
    container_oom_kills: OomKills,
    stop: Option<Stop>,
) -> EndLine {
    let (code, stage) = match error {
        FabricError::AdapterLifecycleOperation { code, .. } => (code.as_str(), ErrorStage::Invoke),
        // After the record starts, the check of the adapter's result
        // extensions raises these: an extension the descriptor does not
        // declare, or one that breaks its declared schema. Both are the same
        // fault as a rejected result.
        FabricError::AdapterCompatibility { .. } | FabricError::InvalidAdapterExtension { .. } => {
            (INVALID_AGENT_RUN_RESULT, ErrorStage::Invoke)
        }
        FabricError::Write { .. } => (ARTIFACT_WRITE_FAILED, ErrorStage::Artifact),
        _ => (FABRIC_ERROR, ErrorStage::Invoke),
    };
    EndLine {
        status: RunStatus::Failed,
        elapsed_ms,
        error: Some(EndError {
            code: recorded_code(code),
            stage,
            retryable: false,
        }),
        usage: None,
        artifacts: Vec::new(),
        container_oom_kills,
        stop,
    }
}

fn result_error(error: &ErrorInfo) -> EndError {
    EndError {
        code: recorded_code(&error.code),
        stage: error.stage,
        retryable: error.retryable,
    }
}

fn end_usage(usage: &RunUsage) -> EndUsage {
    EndUsage {
        input_tokens: usage.input_tokens,
        cached_input_tokens: usage.cached_input_tokens,
        input_tokens_include_cache: usage.input_tokens_include_cache,
        output_tokens: usage.output_tokens,
        total_tokens: usage.total_tokens,
        cost_usd: usage.cost_usd,
    }
}

fn end_artifacts(manifest: &ArtifactManifest) -> Vec<EndArtifact> {
    manifest
        .artifacts
        .iter()
        .map(|artifact| EndArtifact {
            name: cut_text(&artifact.name),
            kind: cut_text(&artifact.kind),
            media_type: artifact.media_type.as_deref().map(cut_text),
            path: manifest
                .root
                .as_deref()
                .and_then(|root| artifact.path.strip_prefix(root).ok())
                .map(|relative| relative.to_string_lossy().into_owned()),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FabricConfig, ResolveContext, resolve_run_plan_from_config};
    use crate::record::test_hooks::{self, Fault};

    #[test]
    fn rejected_result_extensions_read_invalid_agent_run_result() {
        let undeclared = FabricError::AdapterCompatibility {
            adapter_id: "demo".to_string(),
            field: "result.extensions".to_string(),
            reason: "not declared".to_string(),
        };
        let schema_invalid = FabricError::InvalidAdapterExtension {
            adapter_id: "demo".to_string(),
            descriptor_source: crate::config::DescriptorSource::Bundled,
            descriptor_path: std::path::PathBuf::from("adapter.json"),
            extension_path: "result.extensions.demo".to_string(),
            reason: "breaks the declared schema".to_string(),
        };
        for error in [undeclared, schema_invalid] {
            let line = error_end_line(&error, 1, OomKills::Count(0), None);
            assert_eq!(
                line.error.expect("end line error").code,
                INVALID_AGENT_RUN_RESULT,
                "{error}"
            );
        }
    }

    fn record_test_fixture() -> (RunPlan, RuntimeHandle, InvocationHandle) {
        let config: FabricConfig = serde_json::from_value(serde_json::json!({
            "schema_version": "fabric.agent/v1alpha1",
            "metadata": {"name": "demo"},
            "harness": {"adapter_id": "nvidia.fabric.hermes", "resolution": "preinstalled"},
            "runtime": {},
            "models": {"default": {"provider": "nvidia", "model": "nemotron"}},
        }))
        .expect("config");
        let plan = resolve_run_plan_from_config(config, ResolveContext::new("/tmp/fabric-base"))
            .expect("plan");
        let runtime: RuntimeHandle = serde_json::from_value(serde_json::json!({
            "runtime_id": "rt", "runtime_binding": "binding", "agent_name": "demo",
            "harness": "nvidia.fabric.hermes", "adapter_kind": "python",
            "adapter_id": "nvidia.fabric.hermes",
            "environment": {
                "environment_id": "env", "provider": "local",
                "control_location": "external_control", "ownership": "caller_owned",
            },
        }))
        .expect("runtime handle");
        let invocation = InvocationHandle {
            invocation_id: "inv".to_string(),
            request_id: "req".to_string(),
            runtime_id: "rt".to_string(),
        };
        (plan, runtime, invocation)
    }

    #[test]
    fn start_line_keeps_override_and_unknown_context_key_names_only() {
        let (plan, runtime, invocation) = record_test_fixture();
        let mut request = RunRequest::text("input");
        request.context = BTreeMap::from([
            ("task_id".to_string(), Value::from("t".repeat(300))),
            ("task_revision".to_string(), Value::from(1.5)),
            (
                "headers".to_string(),
                serde_json::json!({"authorization": "Bearer abc"}),
            ),
        ]);
        request.overrides = Some(serde_json::json!({"zeta": 1, "acme": {"api_key": "sk-123"}}));
        let start = RecordStart {
            plan: &plan,
            runtime: &runtime,
            invocation: &invocation,
            request: &request,
            fabric_home: Path::new("/unused"),
            timeout: Duration::from_secs(5),
            cgroup: CgroupReader::system(),
        };

        let line = serde_json::to_value(RecordLine {
            schema: RecordSchema::RunRecord,
            version: RecordVersion::V1Alpha1,
            runtime_id: "rt".to_string(),
            invocation_id: "inv".to_string(),
            request_id: "req".to_string(),
            time: "2026-10-14T18:01:10.205Z".to_string(),
            body: LineBody::Start(start_line(&start, OomKills::Count(0))),
        })
        .expect("start line");

        let schema = crate::schema::generate_schema(crate::schema::SchemaName::RunRecordLine)
            .expect("record line schema");
        let validator = jsonschema::validator_for(&schema).expect("valid schema");
        assert!(validator.is_valid(&line), "{line}");
        assert_eq!(line["override_keys"], serde_json::json!(["acme", "zeta"]));
        assert_eq!(
            line["context"]["headers"],
            serde_json::json!({"recorded": false})
        );
        assert_eq!(
            line["context"]["task_revision"],
            serde_json::json!({"recorded": false})
        );
        assert_eq!(
            line["context"]["task_id"]
                .as_str()
                .expect("task id")
                .chars()
                .count(),
            256
        );
        assert_eq!(line["timeout_ms"], 5_000);
        assert_eq!(
            line["models"],
            serde_json::json!({"default": {"provider": "nvidia", "model": "nemotron"}})
        );
        let text = line.to_string();
        assert!(
            !text.contains("sk-123") && !text.contains("Bearer"),
            "{text}"
        );
    }

    #[test]
    fn start_line_keeps_long_role_and_context_key_names_apart() {
        let (mut plan, runtime, invocation) = record_test_fixture();
        let shared = "k".repeat(300);
        let model = plan.config.models["default"].clone();
        for suffix in ["a", "b"] {
            plan.config
                .models
                .insert(format!("{shared}{suffix}"), model.clone());
        }
        let mut request = RunRequest::text("input");
        request.context = BTreeMap::from([
            (format!("{shared}a"), Value::from(1)),
            (format!("{shared}b"), Value::from(2)),
        ]);
        let start = RecordStart {
            plan: &plan,
            runtime: &runtime,
            invocation: &invocation,
            request: &request,
            fabric_home: Path::new("/unused"),
            timeout: Duration::from_secs(5),
            cgroup: CgroupReader::system(),
        };

        let line = start_line(&start, OomKills::Count(0));

        assert_eq!(line.models.len(), 3);
        assert_eq!(line.context.len(), 2);
        for suffix in ["a", "b"] {
            assert!(line.models.contains_key(&format!("{shared}{suffix}")));
            assert!(line.context.contains_key(&format!("{shared}{suffix}")));
        }
    }

    fn temp_folder(name: &str) -> PathBuf {
        let folder =
            std::env::temp_dir().join(format!("fabric-recorder-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&folder);
        std::fs::create_dir_all(&folder).expect("folder");
        folder
    }

    fn succeeded_result(invocation: &InvocationHandle) -> RunResult {
        serde_json::from_value(serde_json::json!({
            "agent_name": "a", "harness": "h", "adapter_kind": "python",
            "runtime_id": invocation.runtime_id, "invocation_id": invocation.invocation_id,
            "request_id": invocation.request_id, "status": "succeeded", "output": "done",
        }))
        .expect("result")
    }

    #[test]
    fn finish_names_the_record_artifact_apart_from_an_adapter_artifact_named_record() {
        let (plan, runtime, invocation) = record_test_fixture();
        let folder = temp_folder("unique-name");
        let request = RunRequest::text("input");
        let recorder = InvocationRecorder::start(RecordStart {
            plan: &plan,
            runtime: &runtime,
            invocation: &invocation,
            request: &request,
            fabric_home: &folder,
            timeout: Duration::from_secs(5),
            cgroup: CgroupReader::system(),
        });
        let mut result = succeeded_result(&invocation);
        result.artifacts.artifacts.push(ArtifactRef {
            name: RECORD_ARTIFACT_NAME.to_string(),
            kind: "report".to_string(),
            path: folder.join("report.json"),
            media_type: None,
            metadata: BTreeMap::new(),
        });

        let finished = recorder.finish(Ok(result)).expect("result");

        let names: Vec<&str> = finished
            .artifacts
            .artifacts
            .iter()
            .map(|artifact| artifact.name.as_str())
            .collect();
        assert_eq!(names, ["record", "record_2"]);
        assert_eq!(finished.artifacts.artifacts[1].kind, RECORD_ARTIFACT_KIND);
        let _ = std::fs::remove_dir_all(folder);
    }

    #[test]
    fn finish_after_a_failed_create_adds_the_event_and_no_record_artifact() {
        let (plan, runtime, invocation) = record_test_fixture();
        let folder = temp_folder("create-failure");
        std::fs::write(folder.join(RECORD_FILE_NAME), "planted").expect("planted file");
        let request = RunRequest::text("input");
        let recorder = InvocationRecorder::start(RecordStart {
            plan: &plan,
            runtime: &runtime,
            invocation: &invocation,
            request: &request,
            fabric_home: &folder,
            timeout: Duration::from_secs(5),
            cgroup: CgroupReader::system(),
        });
        let result = succeeded_result(&invocation);

        let finished = recorder.finish(Ok(result.clone())).expect("result");

        assert_eq!(finished.artifacts, result.artifacts);
        assert_eq!(finished.events.len(), 1);
        assert_eq!(finished.events[0].kind, RECORD_WRITE_FAILED);
        assert_eq!(finished.status, RunStatus::Succeeded);
        assert_eq!(
            std::fs::read_to_string(folder.join(RECORD_FILE_NAME)).expect("read"),
            "planted"
        );
        let _ = std::fs::remove_dir_all(folder);
    }

    fn failed_result_with_usage(invocation: &InvocationHandle) -> RunResult {
        serde_json::from_value(serde_json::json!({
            "agent_name": "a", "harness": "h", "adapter_kind": "python",
            "runtime_id": invocation.runtime_id, "invocation_id": invocation.invocation_id,
            "request_id": invocation.request_id, "status": "failed", "output": "partial",
            "error": {
                "stage": "invoke", "code": "adapter_failure",
                "message": "adapter gave up", "retryable": true,
            },
            "usage": {"input_tokens": 3, "output_tokens": 5, "cost_usd": 0.25},
        }))
        .expect("result")
    }

    /// Start and finish one record with `fault` injected, and return the
    /// finished outcome with the record file's lines.
    fn finish_with_fault(
        name: &str,
        fault: Fault,
        outcome: Result<RunResult>,
    ) -> (Result<RunResult>, Vec<Value>) {
        let (plan, runtime, invocation) = record_test_fixture();
        let folder = temp_folder(name);
        let request = RunRequest::text("input");
        test_hooks::FAULT.with(|slot| slot.set(Some(fault)));
        let recorder = InvocationRecorder::start(RecordStart {
            plan: &plan,
            runtime: &runtime,
            invocation: &invocation,
            request: &request,
            fabric_home: &folder,
            timeout: Duration::from_secs(5),
            cgroup: CgroupReader::system(),
        });
        let finished = recorder.finish(outcome);
        test_hooks::FAULT.with(|slot| slot.set(None));
        let lines = std::fs::read_to_string(folder.join(RECORD_FILE_NAME))
            .expect("read record")
            .lines()
            .map(|line| serde_json::from_str(line).expect("record line"))
            .collect();
        let _ = std::fs::remove_dir_all(folder);
        (finished, lines)
    }

    /// `finished` equals `original` apart from the artifacts and events that
    /// recording adds, and carries one `record_write_failed` event.
    fn assert_result_kept_with_failure_event(finished: &RunResult, original: &RunResult) {
        let mut without_record = finished.clone();
        without_record.artifacts = original.artifacts.clone();
        without_record.events = original.events.clone();
        assert_eq!(&without_record, original);
        let failures: Vec<_> = finished
            .events
            .iter()
            .filter(|event| event.kind == RECORD_WRITE_FAILED)
            .collect();
        assert_eq!(failures.len(), 1, "{:?}", finished.events);
        assert_eq!(
            failures[0].metadata["runtime_id"],
            original.runtime_id.as_str()
        );
        assert_eq!(
            failures[0].metadata["invocation_id"],
            original.invocation_id.as_str()
        );
    }

    fn has_record_artifact(result: &RunResult) -> bool {
        result
            .artifacts
            .artifacts
            .iter()
            .any(|artifact| artifact.kind == RECORD_ARTIFACT_KIND)
    }

    #[test]
    fn finish_after_a_failed_start_line_keeps_the_result_and_writes_nothing() {
        let (_, _, invocation) = record_test_fixture();
        let original = failed_result_with_usage(&invocation);

        let (finished, lines) =
            finish_with_fault("start-append", Fault::StartAppend, Ok(original.clone()));

        let finished = finished.expect("result");
        assert_result_kept_with_failure_event(&finished, &original);
        assert!(!has_record_artifact(&finished));
        assert!(lines.is_empty(), "{lines:?}");
    }

    #[test]
    fn finish_after_a_failed_end_line_keeps_the_result_and_the_start_line() {
        let (_, _, invocation) = record_test_fixture();
        let original = failed_result_with_usage(&invocation);

        let (finished, lines) =
            finish_with_fault("end-append", Fault::EndAppend, Ok(original.clone()));

        let finished = finished.expect("result");
        assert_result_kept_with_failure_event(&finished, &original);
        assert!(has_record_artifact(&finished));
        let kinds: Vec<&Value> = lines.iter().map(|line| &line["kind"]).collect();
        assert_eq!(kinds, ["start"]);
    }

    #[test]
    fn finish_after_a_failed_sync_keeps_the_result_and_both_lines() {
        let (_, _, invocation) = record_test_fixture();
        let original = failed_result_with_usage(&invocation);

        let (finished, lines) = finish_with_fault("sync", Fault::Sync, Ok(original.clone()));

        let finished = finished.expect("result");
        assert_result_kept_with_failure_event(&finished, &original);
        assert!(has_record_artifact(&finished));
        let kinds: Vec<&Value> = lines.iter().map(|line| &line["kind"]).collect();
        assert_eq!(kinds, ["start", "end"]);
        assert_eq!(lines[1]["usage"]["input_tokens"], 3);
    }

    #[test]
    fn finish_returns_the_invocation_error_when_recording_also_fails() {
        for fault in [Fault::StartAppend, Fault::EndAppend, Fault::Sync] {
            let error = FabricError::AdapterLifecycleOperation {
                operation: "invoke",
                runtime_id: "rt".to_string(),
                code: "host_timeout".to_string(),
                message: "adapter host did not answer".to_string(),
                diagnostics: String::new(),
            };
            let expected = error.to_string();

            let (finished, _) = finish_with_fault("error", fault, Err(error));

            let error = finished.expect_err("invocation error");
            assert_eq!(error.to_string(), expected, "{fault:?}");
            assert!(
                matches!(&error, FabricError::AdapterLifecycleOperation { code, .. } if code == "host_timeout"),
                "{fault:?}: {error}"
            );
        }
    }

    /// Run one invocation whose exchange with the host fails with `code` after
    /// the host died of SIGKILL, while a fake cgroup's `oom_kill` count rises
    /// from 4 to 5, and return its start and end lines. With `counted_late`,
    /// the count rises only after the exchange's end was read, and the end is
    /// recounted as core does for a crashed host.
    fn oom_kill_record(
        name: &str,
        exclusive_container: bool,
        code: &str,
        counted_late: bool,
    ) -> (Value, Value) {
        use crate::config::RecordConfig;
        use crate::record::cgroup::tests::{fake_root, memory_events};
        let (mut plan, runtime, invocation) = record_test_fixture();
        plan.config.runtime.record = Some(RecordConfig {
            enabled: true,
            exclusive_container,
        });
        let root = fake_root(name, "0::/\n", Some(&memory_events(4)));
        let folder = root.join("invocation");
        std::fs::create_dir_all(&folder).expect("invocation folder");
        let request = RunRequest::text("input");
        let mut recorder = InvocationRecorder::start(RecordStart {
            plan: &plan,
            runtime: &runtime,
            invocation: &invocation,
            request: &request,
            fabric_home: &folder,
            timeout: Duration::from_secs(5),
            cgroup: CgroupReader::with_root(&root),
        });
        let count_the_kill =
            || std::fs::write(root.join("sys/fs/cgroup/memory.events"), memory_events(5));
        if !counted_late {
            count_the_kill().expect("oom kill");
        }
        let mut end = recorder
            .read_exchange_end(Instant::now())
            .with_host_exit(Some(HostExit {
                exit_code: None,
                exit_signal: Some(9),
            }));
        if counted_late {
            count_the_kill().expect("oom kill");
            end = recorder.recount_oom_kills(end);
        }
        recorder.exchange_ended(end);

        let error = recorder
            .finish(Err(FabricError::AdapterLifecycleOperation {
                operation: "invoke",
                runtime_id: "rt".to_string(),
                code: code.to_string(),
                message: "host exited".to_string(),
                diagnostics: String::new(),
            }))
            .expect_err("error passes through");
        assert!(error.to_string().contains(code));
        let text = std::fs::read_to_string(folder.join(RECORD_FILE_NAME)).expect("record");
        let lines: Vec<Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).expect("json"))
            .collect();
        let _ = std::fs::remove_dir_all(root);
        let start = lines.first().expect("start line").clone();
        let end = lines.last().expect("end line").clone();
        assert_eq!(
            (&start["kind"], &end["kind"]),
            (&"start".into(), &"end".into())
        );
        (start, end)
    }

    #[test]
    fn oom_kill_rise_in_an_exclusive_container_reads_sandbox_failed() {
        let (start, end) = oom_kill_record("exclusive", true, HOST_CRASHED, false);

        assert_eq!(start["container_oom_kills"], 4);
        assert_eq!(start["exclusive_container"], true);
        assert_eq!(end["container_oom_kills"], 5);
        assert_eq!(
            end["stop"],
            serde_json::json!({"cause": "sandbox_failed", "exit_signal": 9})
        );
    }

    #[test]
    fn oom_kill_rise_in_a_shared_container_sets_no_sandbox_cause() {
        let (start, end) = oom_kill_record("shared", false, HOST_CRASHED, false);

        assert_eq!(start["exclusive_container"], false);
        assert_eq!(end["container_oom_kills"], 5);
        assert_eq!(
            end["stop"],
            serde_json::json!({"cause": "unclassified", "exit_signal": 9})
        );
    }

    #[test]
    fn oom_kill_rise_outranks_an_adapter_error_from_the_exchange() {
        let (_, end) = oom_kill_record(
            "adapter-error",
            true,
            "lifecycle_adapter_invoke_failed",
            false,
        );

        assert_eq!(
            end["stop"],
            serde_json::json!({"cause": "sandbox_failed", "exit_signal": 9})
        );
    }

    #[test]
    fn oom_kill_counted_after_the_crashed_host_exited_still_reads_sandbox_failed() {
        let (_, end) = oom_kill_record("counted-late", true, HOST_CRASHED, true);

        assert_eq!(end["container_oom_kills"], 5);
        assert_eq!(
            end["stop"],
            serde_json::json!({"cause": "sandbox_failed", "exit_signal": 9})
        );
    }

    #[test]
    fn oom_kill_counted_after_the_exchange_ended_stays_out_of_the_record() {
        use crate::config::RecordConfig;
        use crate::record::cgroup::tests::{fake_root, memory_events};
        let (mut plan, runtime, invocation) = record_test_fixture();
        plan.config.runtime.record = Some(RecordConfig {
            enabled: true,
            exclusive_container: true,
        });
        let root = fake_root("after-exchange", "0::/\n", Some(&memory_events(4)));
        let folder = root.join("invocation");
        std::fs::create_dir_all(&folder).expect("invocation folder");
        let request = RunRequest::text("input");
        let mut recorder = InvocationRecorder::start(RecordStart {
            plan: &plan,
            runtime: &runtime,
            invocation: &invocation,
            request: &request,
            fabric_home: &folder,
            timeout: Duration::from_secs(5),
            cgroup: CgroupReader::with_root(&root),
        });
        recorder.exchange_ended(recorder.read_exchange_end(Instant::now()));
        // Core released the host lock, so a later invocation on the same
        // runtime can be killed before this result is finished.
        std::fs::write(root.join("sys/fs/cgroup/memory.events"), memory_events(5))
            .expect("later oom kill");
        let mut result = succeeded_result(&invocation);
        result.status = RunStatus::Failed;

        recorder.finish(Ok(result)).expect("result");

        let text = std::fs::read_to_string(folder.join(RECORD_FILE_NAME)).expect("record");
        let _ = std::fs::remove_dir_all(root);
        let end: Value =
            serde_json::from_str(text.lines().last().expect("end line")).expect("json");
        assert_eq!(end["container_oom_kills"], 4, "{end}");
        assert_eq!(end["stop"], serde_json::json!({"cause": "agent_failed"}));
    }
}
