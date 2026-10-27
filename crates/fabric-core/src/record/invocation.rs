// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit request-scoped ownership of a shared run journal.

use super::lines::{
    ContextValue, InvocationFinished, LineBody, ModelName, OutcomeScope, RecordLine, RecordSchema,
    RecordSource, RecordVersion, RequestStarted, Summary,
};
use super::writer::{AppendOutcome, MAX_DIAGNOSTIC_BYTES, RECORD_FILE_NAME, RecordWriter};
use crate::config::RunPlan;
use crate::error::{FabricError, Result};
use crate::runtime::{
    ArtifactRef, ErrorInfo, ErrorStage, InvocationHandle, RunRequest, RunResult, RunStatus,
    RuntimeHandle, event_with_metadata, new_id, unique_artifact_name,
};
use crate::time::rfc3339_millis;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

#[derive(Debug)]
struct RecorderState {
    writer: Option<RecordWriter>,
    recording_id: String,
    request_id: String,
    runtime_id: Option<String>,
    invocation_id: Option<String>,
    path: PathBuf,
    started: Instant,
    started_at: String,
    sequence: u64,
    invocation_started: Option<Instant>,
    adapter_exchange_elapsed_seconds: Option<f64>,
    scope: OutcomeScope,
    failure: Option<String>,
    closed: bool,
    diagnostic_omitted_bytes: u64,
}

/// Cloneable sink. One writer serializes producers without global ownership.
#[derive(Debug, Clone)]
pub(crate) struct RunRecorder {
    state: Arc<Mutex<RecorderState>>,
}

impl RunRecorder {
    pub(crate) fn start(plan: &RunPlan, request: &RunRequest, scope: OutcomeScope) -> Option<Self> {
        let root = super::run_record_folder(plan)?;
        let recording_id = new_id("recording");
        let directory = root.join("requests").join(&recording_id);
        let path = directory.join(RECORD_FILE_NAME);
        // Registration time includes directory preparation, validation and host
        // lock wait. It never shortens the adapter's own timeout budget.
        let started = Instant::now();
        let started_at = rfc3339_millis(SystemTime::now());
        let opened = std::fs::create_dir_all(&directory).and_then(|()| RecordWriter::create(&path));
        let (writer, failure) = match opened {
            Ok(writer) => (Some(writer), None),
            Err(error) => (None, Some(error.to_string())),
        };
        let recorder = Self {
            state: Arc::new(Mutex::new(RecorderState {
                writer,
                recording_id,
                request_id: request.request_id.clone(),
                runtime_id: None,
                invocation_id: None,
                path,
                started,
                started_at,
                sequence: 0,
                invocation_started: None,
                adapter_exchange_elapsed_seconds: None,
                scope,
                failure,
                closed: false,
                diagnostic_omitted_bytes: 0,
            })),
        };
        recorder.append(LineBody::RequestStarted(RequestStarted {
            outcome_scope: scope,
            agent_name: plan.agent_name.clone(),
            harness: plan
                .config
                .harness
                .as_ref()
                .map(|harness| harness.adapter_id.clone())
                .unwrap_or_else(|| "unknown".to_string()),
            context: request
                .context
                .iter()
                .map(|(key, value)| {
                    let kept = match key.as_str() {
                        "task_id" | "task_revision" | "attempt" | "step" | "harbor_session_id"
                        | "harbor_context_id" | "harbor_trial_name" => value
                            .as_str()
                            .map(|text| ContextValue::Text(text.chars().take(256).collect()))
                            .or_else(|| value.as_i64().map(ContextValue::Integer)),
                        _ => None,
                    };
                    (
                        key.clone(),
                        kept.unwrap_or(ContextValue::NotRecorded { recorded: false }),
                    )
                })
                .collect(),
            override_keys: request
                .overrides
                .as_ref()
                .and_then(|value| value.as_object())
                .map(|value| value.keys().cloned().collect())
                .unwrap_or_default(),
            models: plan
                .config
                .models
                .iter()
                .map(|(role, model)| {
                    (
                        role.clone(),
                        ModelName {
                            provider: model.provider.clone(),
                            model: model.model.clone(),
                        },
                    )
                })
                .collect(),
        }));
        Some(recorder)
    }

    /// Return the journal only after its first complete record was written.
    pub(crate) fn record_path(&self) -> Option<PathBuf> {
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        (state.sequence > 0).then(|| state.path.clone())
    }

    pub(crate) fn is_writable(&self) -> bool {
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.writer.is_some() && !state.closed
    }

    pub(crate) fn runtime_ready(&self, runtime: &RuntimeHandle) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.runtime_id = Some(runtime.runtime_id.clone());
        Self::append_locked(
            &mut state,
            LineBody::RuntimeReady {
                runtime_id: runtime.runtime_id.clone(),
            },
        );
    }

    pub(crate) fn set_runtime(&self, runtime: &RuntimeHandle) {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .runtime_id = Some(runtime.runtime_id.clone());
    }

    pub(crate) fn invocation_started(&self, invocation: &InvocationHandle, timeout: Duration) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.runtime_id = Some(invocation.runtime_id.clone());
        state.invocation_id = Some(invocation.invocation_id.clone());
        state.invocation_started = Some(Instant::now());
        Self::append_locked(
            &mut state,
            LineBody::InvocationStarted {
                timeout_seconds: timeout.as_secs_f64(),
            },
        );
    }

    pub(crate) fn invocation_finished(&self, outcome: &Result<RunResult>) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let Some(started) = state.invocation_started else {
            return;
        };
        let elapsed = started.elapsed().as_secs_f64();
        let (status, error, model_usage, system_stats) = match outcome {
            Ok(result) => (
                result.status,
                result.error.clone(),
                result.model_usage.clone(),
                result.system_stats.clone(),
            ),
            Err(error) => (
                RunStatus::Failed,
                Some(recorded_error(error, ErrorStage::Invoke)),
                None,
                None,
            ),
        };
        let exchange_elapsed = state.adapter_exchange_elapsed_seconds;
        Self::append_locked(
            &mut state,
            LineBody::InvocationFinished(Box::new(InvocationFinished {
                status,
                error,
                elapsed_seconds: elapsed,
                adapter_exchange_elapsed_seconds: exchange_elapsed,
                model_usage,
                system_stats,
            })),
        );
    }

    /// Freeze the already-measured exchange interval before endpoint/artifact I/O.
    pub(crate) fn exchange_ended(&self, elapsed_seconds: f64) {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .adapter_exchange_elapsed_seconds = Some(elapsed_seconds);
    }

    pub(crate) fn runtime_stopped(&self, outcome: &Result<Vec<crate::runtime::FabricEvent>>) {
        self.append(LineBody::RuntimeStopped {
            succeeded: outcome.is_ok(),
            error: outcome
                .as_ref()
                .err()
                .map(|error| recorded_error(error, ErrorStage::Stop)),
        });
    }

    /// The timestamp observes capture, not the original production of stderr.
    /// A warm host may produce background diagnostics outside any invocation.
    pub(crate) fn diagnostic(&self, stream: &str, message: &str) {
        // Bound chunks by UTF-8 bytes. JSON escaping can expand by six, so this
        // leaves room for the envelope without splitting a Unicode character.
        let chunk_limit = (MAX_DIAGNOSTIC_BYTES - 4096) / 6;
        let mut remaining = message;
        while !remaining.is_empty() {
            let mut end = remaining.len().min(chunk_limit);
            while !remaining.is_char_boundary(end) {
                end -= 1;
            }
            self.append(LineBody::Diagnostic {
                stream: stream.to_string(),
                scope: "runtime".to_string(),
                message: remaining[..end].to_string(),
            });
            remaining = &remaining[end..];
        }
    }

    pub(crate) fn diagnostic_retention(&self, omitted_bytes: u64, read_error: Option<String>) {
        self.append(LineBody::DiagnosticRetention {
            omitted_bytes,
            read_error,
        });
    }

    pub(crate) fn observes_runtime_lifecycle(&self) -> bool {
        matches!(
            self.state
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .scope,
            OutcomeScope::OneShotLifecycle
        )
    }

    fn append(&self, body: LineBody) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        Self::append_locked(&mut state, body);
    }

    fn append_locked(state: &mut RecorderState, body: LineBody) {
        if state.closed || state.writer.is_none() {
            return;
        }
        let line = RecordLine {
            schema: RecordSchema::Run,
            version: RecordVersion::V1Alpha1,
            recording_id: state.recording_id.clone(),
            request_id: state.request_id.clone(),
            sequence: state.sequence,
            time: rfc3339_millis(SystemTime::now()),
            elapsed_seconds: state.started.elapsed().as_secs_f64(),
            source: RecordSource::Fabric,
            runtime_id: state.runtime_id.clone(),
            invocation_id: state.invocation_id.clone(),
            body,
        };
        let appended = state.writer.as_mut().expect("writer checked").append(&line);
        match appended {
            Ok(AppendOutcome::Written) => state.sequence += 1,
            Ok(AppendOutcome::Omitted) => {
                if let LineBody::Diagnostic { message, .. } = &line.body {
                    state.diagnostic_omitted_bytes = state
                        .diagnostic_omitted_bytes
                        .saturating_add(message.len() as u64);
                }
                // Optional samples report rejected retention through their
                // existing series diagnostics. Neither refusal wrote bytes.
            }
            Err(error) => {
                state.failure = Some(error.to_string());
                // A partial line may be present. Never append valid data behind it.
                state.writer = None;
            }
        }
    }

    /// Freeze admission, append one terminal record and preserve the API outcome.
    pub(crate) fn finish(
        &self,
        mut outcome: Result<RunResult>,
        stage: ErrorStage,
    ) -> Result<RunResult> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if state.closed {
            return outcome;
        }
        if let Ok(result) = &mut outcome {
            state.runtime_id = Some(result.runtime_id.clone());
            state.invocation_id = Some(result.invocation_id.clone());
            if state.sequence > 0 {
                result.artifacts.artifacts.push(ArtifactRef {
                    name: unique_artifact_name(&result.artifacts, "run_log"),
                    kind: "run_log".to_string(),
                    path: state.path.clone(),
                    media_type: Some("application/x-ndjson".to_string()),
                    metadata: BTreeMap::from([(
                        "boundary".to_string(),
                        serde_json::to_value(state.scope).expect("scope"),
                    )]),
                });
            }
        }
        let summary = Summary {
            outcome_scope: state.scope,
            request_started_at: state.started_at.clone(),
            request_finished_at: rfc3339_millis(SystemTime::now()),
            elapsed_seconds: state.started.elapsed().as_secs_f64(),
            adapter_exchange_elapsed_seconds: state.adapter_exchange_elapsed_seconds,
            result: outcome.as_ref().ok().cloned(),
            error: outcome
                .as_ref()
                .err()
                .map(|error| recorded_error(error, stage)),
        };
        let omitted_bytes = state.diagnostic_omitted_bytes;
        if omitted_bytes != 0 {
            Self::append_locked(
                &mut state,
                LineBody::DiagnosticRetention {
                    omitted_bytes,
                    read_error: None,
                },
            );
        }
        Self::append_locked(&mut state, LineBody::Summary(Box::new(summary)));
        state.closed = true;
        if let Some(writer) = &state.writer
            && let Err(error) = writer.sync()
        {
            state.failure = Some(error.to_string());
        }
        if let Some(failure) = &state.failure {
            if let Ok(result) = &mut outcome {
                result.events.push(event_with_metadata(
                    "record_write_failed",
                    format!("could not finish run journal: {failure}"),
                    BTreeMap::from([(
                        "path".to_string(),
                        serde_json::Value::from(state.path.to_string_lossy().as_ref()),
                    )]),
                ));
            } else {
                eprintln!("nemo-fabric: run journal failed: {failure}");
            }
        }
        outcome
    }
}

fn recorded_error(error: &FabricError, stage: ErrorStage) -> ErrorInfo {
    let code = match error {
        FabricError::AdapterLifecycleOperation { code, .. } => code.clone(),
        FabricError::InvalidAdapterExtension { .. } => "invalid_agent_run_result".to_string(),
        FabricError::Write { .. } => "artifact_write_failed".to_string(),
        _ => "fabric_error".to_string(),
    };
    ErrorInfo {
        stage,
        code,
        message: error.to_string(),
        retryable: false,
        metadata: BTreeMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FabricConfig, ResolveContext, resolve_run_plan_from_config};
    use crate::record::test_hooks::{self, Fault};
    use serde_json::{Value, json};

    fn fixture() -> (RunPlan, RunRequest, RunResult) {
        let root = std::env::temp_dir().join(new_id("run-log-contract"));
        std::fs::create_dir_all(&root).unwrap();
        let config: FabricConfig = serde_json::from_value(json!({
            "schema_version":"fabric.agent/v1alpha1", "metadata":{"name":"demo"},
            "harness":{"adapter_id":"nvidia.fabric.hermes","resolution":"preinstalled"},
            "runtime":{"artifacts":root,"record":{"enabled":true}}
        }))
        .unwrap();
        let plan = resolve_run_plan_from_config(config, ResolveContext::new(&root)).unwrap();
        let mut request = RunRequest::text("test");
        request.request_id = "req".to_string();
        let result = serde_json::from_value(json!({
            "agent_name":"demo", "harness":"nvidia.fabric.hermes", "adapter_kind":"python",
            "runtime_id":"rt", "invocation_id":"inv", "request_id":"req",
            "status":"failed", "output":{"answer":"kept"},
            "usage":{"input_tokens":42,"output_tokens":0}
        }))
        .unwrap();
        (plan, request, result)
    }

    fn rows(recorder: &RunRecorder) -> Vec<Value> {
        let path = recorder.state.lock().unwrap().path.clone();
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn oversized_diagnostic_envelope_keeps_terminal_result_and_counts_omitted_bytes() {
        let (plan, mut request, mut result) = fixture();
        request.request_id = "r".repeat(70 * 1024);
        result.request_id = request.request_id.clone();
        let recorder = RunRecorder::start(&plan, &request, OutcomeScope::Invocation).unwrap();
        recorder.diagnostic("stderr", "first");
        recorder.diagnostic("stderr", "second");
        let returned = recorder
            .finish(Ok(result.clone()), ErrorStage::Invoke)
            .unwrap();
        let rows = rows(&recorder);
        assert_eq!(rows.last().unwrap()["kind"], "summary");
        assert_eq!(
            rows.last().unwrap()["data"]["result"],
            serde_json::to_value(&returned).unwrap()
        );
        assert_eq!(returned.status, result.status);
        assert_eq!(returned.usage, result.usage);
        assert!(
            !returned
                .events
                .iter()
                .any(|event| event.kind == "record_write_failed")
        );
        let retention: Vec<_> = rows
            .iter()
            .filter(|row| row["kind"] == "diagnostic_retention")
            .collect();
        assert_eq!(retention.len(), 1);
        assert_eq!(retention[0]["data"]["omitted_bytes"], 11);
        assert!(retention[0]["data"]["read_error"].is_null());
        assert!(rows.iter().all(|row| row["kind"] != "diagnostic"));
        for (sequence, row) in rows.iter().enumerate() {
            assert_eq!(row["sequence"], sequence as u64);
        }
    }

    #[test]
    fn terminal_result_retains_large_output_and_reported_zero() {
        let (plan, request, mut result) = fixture();
        result.output = Value::String("x".repeat(128 * 1024));
        let recorder = RunRecorder::start(&plan, &request, OutcomeScope::Invocation).unwrap();
        let returned = recorder.finish(Ok(result), ErrorStage::Invoke).unwrap();
        let rows = rows(&recorder);
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows[1]["data"]["result"],
            serde_json::to_value(returned).unwrap()
        );
        assert_eq!(rows[1]["data"]["result"]["usage"]["output_tokens"], 0);
    }

    #[test]
    fn concurrent_producers_share_sequence_and_stop_after_summary() {
        let (plan, request, result) = fixture();
        let recorder = RunRecorder::start(&plan, &request, OutcomeScope::Invocation).unwrap();
        let workers: Vec<_> = (0..4)
            .map(|worker| {
                let sink = recorder.clone();
                std::thread::spawn(move || {
                    for item in 0..20 {
                        sink.diagnostic("stderr", &format!("worker {worker}: {item}"));
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        recorder.finish(Ok(result), ErrorStage::Invoke).unwrap();
        recorder.diagnostic("stderr", "must not appear after terminal summary");
        let rows = rows(&recorder);
        assert_eq!(rows.len(), 82);
        assert_eq!(rows.last().unwrap()["kind"], "summary");
        for (sequence, row) in rows.iter().enumerate() {
            assert_eq!(row["sequence"], sequence as u64);
        }
    }

    #[test]
    fn failed_start_append_keeps_execution_without_claiming_a_saved_result() {
        let (plan, request, result) = fixture();
        test_hooks::FAULT.with(|slot| slot.set(Some(Fault::StartAppend)));
        let recorder = RunRecorder::start(&plan, &request, OutcomeScope::Invocation).unwrap();
        test_hooks::FAULT.with(|slot| slot.set(None));
        let returned = recorder
            .finish(Ok(result.clone()), ErrorStage::Invoke)
            .unwrap();
        assert_eq!(returned.output, result.output);
        assert_eq!(returned.usage, result.usage);
        assert!(returned.artifacts.artifacts.is_empty());
        assert!(
            returned
                .events
                .iter()
                .any(|event| event.kind == "record_write_failed")
        );
        assert!(rows(&recorder).is_empty());
    }

    #[test]
    fn failed_terminal_append_does_not_resume_or_change_execution() {
        let (plan, request, result) = fixture();
        let recorder = RunRecorder::start(&plan, &request, OutcomeScope::Invocation).unwrap();
        test_hooks::FAULT.with(|slot| slot.set(Some(Fault::EndAppend)));
        let returned = recorder
            .finish(Ok(result.clone()), ErrorStage::Invoke)
            .unwrap();
        test_hooks::FAULT.with(|slot| slot.set(None));
        recorder.diagnostic("stderr", "late");
        assert_eq!(returned.status, result.status);
        assert_eq!(returned.usage, result.usage);
        assert_eq!(returned.output, result.output);
        assert!(
            returned
                .events
                .iter()
                .any(|event| event.kind == "record_write_failed")
        );
        assert_eq!(rows(&recorder).len(), 1);
    }

    #[test]
    fn failed_sync_preserves_final_evidence_and_reports_durability_failure() {
        let (plan, request, result) = fixture();
        let recorder = RunRecorder::start(&plan, &request, OutcomeScope::Invocation).unwrap();
        test_hooks::FAULT.with(|slot| slot.set(Some(Fault::Sync)));
        let returned = recorder
            .finish(Ok(result.clone()), ErrorStage::Invoke)
            .unwrap();
        test_hooks::FAULT.with(|slot| slot.set(None));
        assert_eq!(returned.output, result.output);
        assert_eq!(rows(&recorder).last().unwrap()["kind"], "summary");
        assert!(
            returned
                .events
                .iter()
                .any(|event| event.kind == "record_write_failed")
        );
    }

    #[test]
    fn no_result_summary_keeps_outer_failure_without_inner_identity() {
        let (plan, request, _) = fixture();
        let recorder = RunRecorder::start(&plan, &request, OutcomeScope::OneShotLifecycle).unwrap();
        let error = FabricError::InvalidConfig {
            field: "test".to_string(),
            reason: "invalid".to_string(),
        };
        let original = error.to_string();
        assert_eq!(
            recorder
                .finish(Err(error), ErrorStage::Prepare)
                .unwrap_err()
                .to_string(),
            original
        );
        let rows = rows(&recorder);
        assert!(rows[1].get("runtime_id").is_none());
        assert!(rows[1]["data"]["result"].is_null());
        assert_eq!(rows[1]["data"]["error"]["stage"], "prepare");
    }

    #[test]
    fn exchange_time_freezes_before_invocation_finalization() {
        let (plan, request, result) = fixture();
        let recorder = RunRecorder::start(&plan, &request, OutcomeScope::Invocation).unwrap();
        let invocation = InvocationHandle {
            runtime_id: "rt".to_string(),
            invocation_id: "inv".to_string(),
            request_id: "req".to_string(),
        };
        recorder.invocation_started(&invocation, Duration::from_secs(60));
        recorder.exchange_ended(0.125);
        recorder.invocation_finished(&Ok(result.clone()));
        recorder.finish(Ok(result), ErrorStage::Invoke).unwrap();
        let rows = rows(&recorder);
        let finished = rows
            .iter()
            .find(|row| row["kind"] == "invocation_finished")
            .unwrap();
        assert_eq!(finished["data"]["adapter_exchange_elapsed_seconds"], 0.125);
        assert_eq!(
            rows.last().unwrap()["data"]["adapter_exchange_elapsed_seconds"],
            0.125
        );
    }
}
