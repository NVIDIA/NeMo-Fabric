// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One invocation's record: the start line before the request, the end line
//! after the invocation.
//!
//! Recording never changes the run. A failed create, write or sync stops the
//! record for that invocation and adds a `record_write_failed` event to the
//! result; nothing here returns an error to the caller.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use serde_json::Value;

use super::lines::{
    ContextValue, EndArtifact, EndError, EndLine, EndUsage, LineBody, ModelName, NotRecorded,
    RecordLine, RecordSchema, RecordVersion, StartLine, cut_text, recorded_code,
};
use super::writer::{RECORD_FILE_NAME, RecordWriter};
use crate::config::RunPlan;
use crate::error::{FabricError, Result};
use crate::runtime::{
    ArtifactManifest, ArtifactRef, ErrorInfo, ErrorStage, INVALID_AGENT_RUN_RESULT,
    InvocationHandle, RunRequest, RunResult, RunStatus, RunUsage, RuntimeHandle,
    event_with_metadata, unique_artifact_name,
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
/// End-line error code for an artifact write that failed after the adapter returned.
const ARTIFACT_WRITE_FAILED: &str = "artifact_write_failed";
/// End-line error code for any other Fabric error after the request was sent.
const FABRIC_ERROR: &str = "fabric_error";

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
}

impl InvocationRecorder {
    /// Create the record and write its start line.
    pub(crate) fn start(start: RecordStart<'_>) -> Self {
        let path = start.fabric_home.join(RECORD_FILE_NAME);
        let mut recorder = Self {
            runtime_id: start.invocation.runtime_id.clone(),
            invocation_id: start.invocation.invocation_id.clone(),
            request_id: start.invocation.request_id.clone(),
            started: Instant::now(),
            writer: None,
            wrote_start: false,
            failure: None,
            path,
        };
        match RecordWriter::create(&recorder.path) {
            Ok(writer) => {
                recorder.writer = Some(writer);
                recorder.append(LineBody::Start(start_line(&start)));
                recorder.wrote_start = recorder.failure.is_none();
            }
            Err(error) => recorder.failure = Some(error),
        }
        recorder
    }

    /// Write and sync the end line for `outcome`, then return it. A result
    /// gains the `record` artifact when the start line was written, and a
    /// `record_write_failed` event when any write or the sync failed.
    pub(crate) fn finish(mut self, outcome: Result<RunResult>) -> Result<RunResult> {
        let elapsed_ms = u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX);
        let end = match &outcome {
            Ok(result) => result_end_line(result, elapsed_ms),
            Err(error) => error_end_line(error, elapsed_ms),
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

fn start_line(start: &RecordStart<'_>) -> StartLine {
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
                    cut_text(role),
                    ModelName {
                        provider: cut_text(&model.provider),
                        model: cut_text(&model.model),
                    },
                )
            })
            .collect(),
        fabric_version: crate::version().to_string(),
        timeout_ms: u64::try_from(start.timeout.as_millis()).unwrap_or(u64::MAX),
        context: start
            .request
            .context
            .iter()
            .map(|(key, value)| (cut_text(key), context_value(key, value)))
            .collect(),
        override_keys: override_keys(start.request.overrides.as_ref()),
    }
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

fn result_end_line(result: &RunResult, elapsed_ms: u64) -> EndLine {
    EndLine {
        status: result.status,
        elapsed_ms,
        error: result.error.as_ref().map(result_error),
        usage: result.usage.as_ref().map(end_usage),
        artifacts: end_artifacts(&result.artifacts),
    }
}

fn error_end_line(error: &FabricError, elapsed_ms: u64) -> EndLine {
    let (code, stage) = match error {
        FabricError::AdapterLifecycleOperation { code, .. } => (code.as_str(), ErrorStage::Invoke),
        // After the record starts, only the check of the adapter's result
        // extensions raises this, the same fault as a rejected result.
        FabricError::AdapterCompatibility { .. } => (INVALID_AGENT_RUN_RESULT, ErrorStage::Invoke),
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
        };

        let line = serde_json::to_value(RecordLine {
            schema: RecordSchema::RunRecord,
            version: RecordVersion::V1Alpha1,
            runtime_id: "rt".to_string(),
            invocation_id: "inv".to_string(),
            request_id: "req".to_string(),
            time: "2026-10-14T18:01:10.205Z".to_string(),
            body: LineBody::Start(start_line(&start)),
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
}
