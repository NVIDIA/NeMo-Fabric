// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Full finalized results and request-level evidence before a runtime exists.
//! These files may contain result output; the bounded invocation record does not.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use schemars::JsonSchema;
use serde::Serialize;
use serde_json::Value;

use crate::config::RunPlan;
use crate::runtime::{
    ArtifactRef, RunRequest, RunResult, event_with_metadata, new_id, unique_artifact_name,
};
use crate::time::rfc3339_millis;

pub(crate) const RESULT_FILE_NAME: &str = "run-result.json";
const WRITE_FAILED: &str = "record_write_failed";
const MAX_REQUEST_RECORD_BYTES: usize = 64 * 1024;

/// The recording root. This is known before startup and requires no inner IDs.
pub fn run_record_folder(plan: &RunPlan) -> Option<PathBuf> {
    if !plan.config.runtime.record_enabled() {
        return None;
    }
    plan.config
        .runtime
        .artifacts
        .as_ref()
        .map(|path| {
            if path.is_absolute() {
                path.clone()
            } else {
                plan.base_dir.join(path)
            }
        })
        .or_else(|| {
            plan.environment_plan
                .as_ref()
                .and_then(|environment| environment.artifacts.clone())
        })
        .map(|root| root.join(".fabric"))
}

fn private_new_file(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}

/// Publish a complete JSON file without replacing a file or a symlink.
/// A failed write may leave an explicitly named staging file, never a final file.
fn publish(path: &Path, value: &impl serde::Serialize) -> io::Result<()> {
    let bytes = serde_json::to_vec_pretty(value).map_err(io::Error::other)?;
    publish_bytes(path, &bytes)
}

fn publish_request(path: &Path, value: &RequestRecord) -> io::Result<()> {
    let bytes = serde_json::to_vec_pretty(value).map_err(io::Error::other)?;
    if bytes.len() > MAX_REQUEST_RECORD_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("request record exceeds {MAX_REQUEST_RECORD_BYTES} bytes"),
        ));
    }
    publish_bytes(path, &bytes)
}

fn publish_bytes(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let pending = path.with_extension(format!("{}.pending", new_id("write")));
    let mut file = private_new_file(&pending)?;
    let published = (|| {
        file.write_all(bytes)?;
        file.sync_all()?;
        // Unlike rename, this cannot replace an existing file or symlink.
        std::fs::hard_link(&pending, path)
    })();
    drop(file);
    // Only this operation's newly created staging path can be removed.
    if let Err(error) = cleanup_staging(&pending) {
        // Publication has its own outcome. A private staging-link cleanup
        // failure must not invalidate or mutate an already frozen result.
        eprintln!(
            "nemo-fabric: could not clean recording staging file {}: {error}",
            pending.display()
        );
    }
    published
}

fn cleanup_staging(path: &Path) -> io::Result<()> {
    #[cfg(test)]
    if CLEANUP_FAILURE.with(std::cell::Cell::get) {
        return Err(io::Error::other("injected staging cleanup failure"));
    }
    std::fs::remove_file(path)
}

#[cfg(test)]
thread_local! {
    static CLEANUP_FAILURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn add_failure(result: &mut RunResult, error: &io::Error, artifact: &str) {
    result.events.push(event_with_metadata(
        WRITE_FAILED,
        format!("could not save {artifact}: {error}"),
        BTreeMap::from([("artifact".to_string(), Value::from(artifact))]),
    ));
}

/// Save exactly the returned result after its invocation or one-shot lifecycle.
pub(crate) fn persist_result(plan: &RunPlan, mut result: RunResult, boundary: &str) -> RunResult {
    let Some(root) = run_record_folder(plan) else {
        return result;
    };
    let directory = root.join(&result.runtime_id).join(&result.invocation_id);
    let path = directory.join(RESULT_FILE_NAME);
    let name = unique_artifact_name(&result.artifacts, "run_result");
    let artifact = ArtifactRef {
        name: name.clone(),
        kind: "run_result".to_string(),
        path: path.clone(),
        media_type: Some("application/json".to_string()),
        metadata: BTreeMap::from([("boundary".to_string(), Value::from(boundary))]),
    };
    result.artifacts.artifacts.push(artifact);
    let saved = std::fs::create_dir_all(&directory).and_then(|()| publish(&path, &result));
    if let Err(error) = saved {
        result
            .artifacts
            .artifacts
            .retain(|artifact| artifact.name != name);
        add_failure(&mut result, &error, RESULT_FILE_NAME);
    }
    result
}

/// A request record does not imply a runtime or invocation was created.
#[derive(Debug, Serialize, JsonSchema)]
pub(crate) struct RequestRecord {
    schema: RequestSchema,
    version: super::lines::RecordVersion,
    request_id: String,
    #[schemars(extend("format" = "date-time"))]
    time: String,
    #[serde(flatten)]
    body: RequestBody,
}

#[derive(Debug, Serialize, JsonSchema)]
enum RequestSchema {
    #[serde(rename = "nemo.fabric.request_record")]
    RequestRecord,
}

#[derive(Debug, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum RequestBody {
    Start,
    End {
        #[serde(flatten)]
        outcome: RequestOutcome,
    },
}

#[derive(Debug, Serialize, JsonSchema)]
#[serde(tag = "state", rename_all = "snake_case")]
enum RequestOutcome {
    Returned {
        runtime_id: String,
        invocation_id: String,
        result: String,
    },
    NoResult {
        stage: RequestStage,
    },
}

#[derive(Debug, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RequestStage {
    Start,
    Invoke,
}

/// Bounded outer evidence for a request that can fail before inner IDs exist.
pub(crate) struct RequestRecorder {
    directory: Option<PathBuf>,
    request_id: String,
    failure: Option<io::Error>,
}

impl RequestRecorder {
    pub(crate) fn start(plan: &RunPlan, request: &RunRequest) -> Self {
        let directory =
            run_record_folder(plan).map(|root| root.join("requests").join(new_id("dispatch")));
        let mut recorder = Self {
            directory,
            request_id: request.request_id.clone(),
            failure: None,
        };
        if let Some(directory) = &recorder.directory {
            let value = recorder.line(RequestBody::Start);
            recorder.failure = std::fs::create_dir_all(directory)
                .and_then(|()| publish_request(&directory.join("request-start.json"), &value))
                .err();
        }
        recorder
    }

    pub(crate) fn returned(mut self, result: &mut RunResult) {
        self.finish(RequestOutcome::Returned {
            runtime_id: result.runtime_id.clone(),
            invocation_id: result.invocation_id.clone(),
            result: format!(
                "{}/{}/{}",
                result.runtime_id, result.invocation_id, RESULT_FILE_NAME
            ),
        });
        if let Some(error) = self.failure {
            add_failure(result, &error, "request record");
        }
    }

    pub(crate) fn no_result(mut self, stage: RequestStage) {
        self.finish(RequestOutcome::NoResult { stage });
        // No result exists to carry an event. Keep the execution error intact
        // and make the independent recording failure visible on stderr.
        if let Some(error) = self.failure {
            eprintln!("nemo-fabric: request recording failed: {error}");
        }
    }

    fn line(&self, body: RequestBody) -> RequestRecord {
        RequestRecord {
            schema: RequestSchema::RequestRecord,
            version: super::lines::RecordVersion::V1Alpha1,
            request_id: self.request_id.clone(),
            time: rfc3339_millis(SystemTime::now()),
            body,
        }
    }

    fn finish(&mut self, outcome: RequestOutcome) {
        let Some(directory) = &self.directory else {
            return;
        };
        let value = self.line(RequestBody::End { outcome });
        if let Err(error) = publish_request(&directory.join("request-end.json"), &value) {
            self.failure = Some(error);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FabricConfig, ResolveContext, resolve_run_plan_from_config};
    use serde_json::json;

    fn fixture() -> (PathBuf, RunPlan, RunResult) {
        let root = std::env::temp_dir().join(new_id("recording-result-test"));
        std::fs::create_dir_all(&root).expect("test root");
        let config: FabricConfig = serde_json::from_value(json!({
            "schema_version":"fabric.agent/v1alpha1", "metadata":{"name":"demo"},
            "harness":{"adapter_id":"nvidia.fabric.hermes","resolution":"preinstalled"},
            "runtime":{"artifacts":root,"record":{"enabled":true}}
        }))
        .expect("config");
        let plan = resolve_run_plan_from_config(config, ResolveContext::new(&root)).expect("plan");
        let result = serde_json::from_value(json!({
            "agent_name":"demo","harness":"nvidia.fabric.hermes","adapter_kind":"python",
            "runtime_id":"rt","invocation_id":"inv","request_id":"req",
            "status":"failed","output":{"answer":"kept"},
            "metadata":{"existing":"kept"},"usage":{"input_tokens":42,"output_tokens":0}
        }))
        .expect("result");
        (root, plan, result)
    }

    #[test]
    fn recorded_full_result_is_exact_and_keeps_failed_usage() {
        let (_, plan, result) = fixture();
        let returned = persist_result(&plan, result, "one_shot_lifecycle");
        let artifact = returned
            .artifacts
            .artifacts
            .iter()
            .find(|item| item.kind == "run_result")
            .expect("saved result artifact");
        let saved: Value =
            serde_json::from_slice(&std::fs::read(&artifact.path).expect("result file"))
                .expect("JSON");
        assert_eq!(
            saved,
            serde_json::to_value(&returned).expect("returned result")
        );
        assert_eq!(saved["usage"]["input_tokens"], 42);
        assert_eq!(saved["usage"]["output_tokens"], 0);
        assert_eq!(artifact.metadata["boundary"], "one_shot_lifecycle");
        assert!(
            std::fs::read_dir(artifact.path.parent().unwrap())
                .expect("directory")
                .all(|entry| !entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .ends_with(".pending"))
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&artifact.path)
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn published_result_remains_exact_when_owned_staging_cleanup_fails() {
        let (_, plan, result) = fixture();
        CLEANUP_FAILURE.with(|slot| slot.set(true));
        let returned = persist_result(&plan, result, "invocation");
        CLEANUP_FAILURE.with(|slot| slot.set(false));
        let artifact = returned
            .artifacts
            .artifacts
            .iter()
            .find(|item| item.kind == "run_result")
            .expect("publication remains successful");
        let saved: Value = serde_json::from_slice(&std::fs::read(&artifact.path).unwrap()).unwrap();
        assert_eq!(saved, serde_json::to_value(&returned).unwrap());
        assert!(
            returned
                .events
                .iter()
                .all(|event| event.kind != WRITE_FAILED)
        );
        assert_eq!(
            std::fs::read_dir(artifact.path.parent().unwrap())
                .unwrap()
                .count(),
            2
        );
    }

    #[test]
    fn existing_full_result_is_not_overwritten_and_execution_is_kept() {
        let (root, plan, result) = fixture();
        let folder = root.join(".fabric/rt/inv");
        std::fs::create_dir_all(&folder).expect("folder");
        let path = folder.join(RESULT_FILE_NAME);
        std::fs::write(&path, b"existing evidence").expect("existing file");
        let original = result.clone();
        let returned = persist_result(&plan, result, "invocation");
        assert_eq!(returned.status, original.status);
        assert_eq!(returned.usage, original.usage);
        assert_eq!(returned.output, original.output);
        assert_eq!(std::fs::read(&path).unwrap(), b"existing evidence");
        assert!(
            returned
                .artifacts
                .artifacts
                .iter()
                .all(|item| item.kind != "run_result")
        );
        assert!(
            returned
                .events
                .iter()
                .any(|event| event.kind == WRITE_FAILED)
        );
        assert_eq!(std::fs::read_dir(folder).unwrap().count(), 1);
    }

    #[test]
    fn outer_start_failure_keeps_request_identity_without_inner_ids_or_input() {
        let (root, plan, _) = fixture();
        let mut request = RunRequest::text("private input must not be recorded");
        request.request_id = "../../caller-id".to_string();
        let recorder = RequestRecorder::start(&plan, &request);
        recorder.no_result(RequestStage::Start);
        let folders: Vec<_> = std::fs::read_dir(root.join(".fabric/requests"))
            .unwrap()
            .collect();
        assert_eq!(folders.len(), 1);
        let folder = folders[0].as_ref().unwrap().path();
        let start: Value =
            serde_json::from_slice(&std::fs::read(folder.join("request-start.json")).unwrap())
                .unwrap();
        let end: Value =
            serde_json::from_slice(&std::fs::read(folder.join("request-end.json")).unwrap())
                .unwrap();
        assert_eq!(start["request_id"], "../../caller-id");
        assert_eq!(end["state"], "no_result");
        assert_eq!(end["stage"], "start");
        assert!(start.get("runtime_id").is_none());
        assert!(end.get("invocation_id").is_none());
        assert!(end.get("usage").is_none());
        assert!(start.get("input").is_none());
    }

    #[test]
    fn request_records_accept_the_byte_limit_and_reject_one_byte_more() {
        let (_, plan, _) = fixture();
        let empty = RequestRecorder {
            directory: None,
            request_id: String::new(),
            failure: None,
        };
        let overhead = serde_json::to_vec_pretty(&empty.line(RequestBody::Start))
            .unwrap()
            .len();
        let mut request = RunRequest::text("input");
        request.request_id = "r".repeat(64 * 1024 - overhead);
        let recorder = RequestRecorder::start(&plan, &request);
        assert!(recorder.failure.is_none());
        let path = recorder.directory.unwrap().join("request-start.json");
        assert_eq!(std::fs::metadata(path).unwrap().len(), 64 * 1024);

        request.request_id.push('r');
        let recorder = RequestRecorder::start(&plan, &request);
        assert!(recorder.failure.is_some());
        assert_eq!(
            std::fs::read_dir(recorder.directory.unwrap())
                .unwrap()
                .count(),
            0
        );
    }

    #[test]
    fn oversized_request_records_preserve_the_result_and_recording_diagnostic() {
        let (root, plan, mut result) = fixture();
        let mut request = RunRequest::text("private input");
        request.request_id = "r".repeat(16 * 1024 * 1024 + 1);
        result.request_id.clone_from(&request.request_id);
        let original = result.clone();
        let recorder = RequestRecorder::start(&plan, &request);
        let directory = recorder.directory.clone().expect("request directory");
        assert!(
            recorder.failure.is_some(),
            "oversized request record must be rejected"
        );
        recorder.returned(&mut result);
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 0);
        assert_eq!(result.status, original.status);
        assert_eq!(result.error, original.error);
        assert_eq!(result.usage, original.usage);
        assert_eq!(result.output, original.output);
        assert_eq!(result.request_id, original.request_id);
        assert!(result.events.iter().any(|event| event.kind == WRITE_FAILED));

        let returned = persist_result(&plan, result, "one_shot_lifecycle");
        let path = root.join(".fabric/rt/inv").join(RESULT_FILE_NAME);
        let saved: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(saved, serde_json::to_value(returned).unwrap());
    }

    #[test]
    #[cfg(unix)]
    fn atomic_publication_refuses_symlink_and_cleans_only_owned_stage() {
        let root = std::env::temp_dir().join(new_id("recording-link-test"));
        std::fs::create_dir_all(&root).unwrap();
        let original = root.join("original.json");
        std::fs::write(&original, "unchanged").unwrap();
        let target = root.join("result.json");
        std::os::unix::fs::symlink(&original, &target).unwrap();
        assert!(publish(&target, &json!({"new":"content"})).is_err());
        assert_eq!(std::fs::read_to_string(original).unwrap(), "unchanged");
        assert!(target.is_symlink());
        assert_eq!(std::fs::read_dir(root).unwrap().count(), 2);
    }
}
