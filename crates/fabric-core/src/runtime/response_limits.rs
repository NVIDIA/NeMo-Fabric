// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SDK-owned response budgets shared by host and task execution.

use super::*;
use crate::config::ResponseLimits;

pub(super) fn reject(
    result: &mut RunResult,
    limit: &str,
    configured: u64,
    observed: u64,
    asset: &str,
) {
    result.status = RunStatus::Failed;
    result.output = Value::Null;
    result.artifacts = ArtifactManifest::default();
    result.metadata.clear();
    result.events.clear();
    result.telemetry.clear();
    result.usage = None;
    result.execution = None;
    result.system_stats = None;
    result.resource_snapshot = None;
    result.error = Some(ErrorInfo {
        stage: ErrorStage::Invoke,
        code: "response_limit_exceeded".into(),
        message: format!(
            "runtime.response_limits.{limit}={configured} exceeded by {asset}: observed {observed}. Increase this limit and invoke again; the result was rejected without truncation."
        ),
        retryable: false,
        metadata: BTreeMap::from([
            ("limit".into(), Value::String(limit.into())),
            ("configured".into(), Value::from(configured)),
            ("observed".into(), Value::from(observed)),
            ("asset".into(), Value::String(asset.into())),
        ]),
    });
}

pub(super) fn enforce_response(result: &mut RunResult, limits: ResponseLimits) -> Result<()> {
    struct Counter(u64);
    impl Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self.0.saturating_add(bytes.len() as u64);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut counter = Counter(0);
    serde_json::to_writer(&mut counter, result).map_err(FabricError::SerializeJson)?;
    // Exempt only a small failed diagnostic, never arbitrary adapter output
    // that happens to use the same error code.
    if counter.0 <= 64 * 1024 && is_sdk_limit_diagnostic(result) {
        return Ok(());
    }
    if counter.0 > u64::from(limits.max_response_bytes) {
        reject(
            result,
            "max_response_bytes",
            limits.max_response_bytes.into(),
            counter.0,
            "RunResult",
        );
    }
    Ok(())
}

fn is_sdk_limit_diagnostic(result: &RunResult) -> bool {
    let Some(error) = &result.error else {
        return false;
    };
    if error.code != "response_limit_exceeded" {
        return false;
    }
    let (Some(limit), Some(configured), Some(observed), Some(asset)) = (
        error.metadata.get("limit").and_then(Value::as_str),
        error.metadata.get("configured").and_then(Value::as_u64),
        error.metadata.get("observed").and_then(Value::as_u64),
        error.metadata.get("asset").and_then(Value::as_str),
    ) else {
        return false;
    };
    if ![
        "max_response_bytes",
        "max_artifact_bytes",
        "max_artifact_files",
        "max_artifact_file_bytes",
    ]
    .contains(&limit)
        || configured == 0
        || configured > u64::from(u32::MAX)
        || observed <= configured
        || asset.is_empty()
        || asset.len() > 4096
        || asset.chars().any(char::is_control)
        || (limit == "max_response_bytes" && asset != "RunResult")
    {
        return false;
    }
    // Match the complete output of the one SDK rejection constructor, including
    // empty payload fields, exact diagnostic text/metadata and retry semantics.
    let mut canonical = result.clone();
    reject(&mut canonical, limit, configured, observed, asset);
    *result == canonical
}

pub(super) fn enforce_local_artifacts(
    result: &mut RunResult,
    limits: ResponseLimits,
) -> Result<()> {
    let mut paths = BTreeSet::new();
    let mut total = 0_u64;
    for artifact in &result.artifacts.artifacts {
        let relative = result
            .artifacts
            .root
            .as_ref()
            .and_then(|root| artifact.path.strip_prefix(root).ok())
            .unwrap_or(&artifact.path);
        if openshell::is_internal_sdk_artifact(&relative.to_string_lossy()) {
            continue;
        }
        if !paths.insert(artifact.path.clone()) {
            continue;
        }
        let asset = artifact.path.to_string_lossy().into_owned();
        if paths.len() as u64 > u64::from(limits.max_artifact_files) {
            reject(
                result,
                "max_artifact_files",
                limits.max_artifact_files.into(),
                paths.len() as u64,
                &asset,
            );
            return Ok(());
        }
        let metadata = std::fs::metadata(&artifact.path).map_err(|source| FabricError::Read {
            path: artifact.path.clone(),
            source,
        })?;
        // Directories and non-file references do not have transferable bytes.
        // The OpenShell collector additionally validates regular-file identity.
        if !metadata.is_file() {
            continue;
        }
        let bytes = metadata.len();
        if bytes > u64::from(limits.max_artifact_file_bytes) {
            reject(
                result,
                "max_artifact_file_bytes",
                limits.max_artifact_file_bytes.into(),
                bytes,
                &asset,
            );
            return Ok(());
        }
        total = total.saturating_add(bytes);
        if total > u64::from(limits.max_artifact_bytes) {
            reject(
                result,
                "max_artifact_bytes",
                limits.max_artifact_bytes.into(),
                total,
                &asset,
            );
            return Ok(());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result() -> RunResult {
        serde_json::from_value(serde_json::json!({
            "agent_name": "limits", "harness": "echo", "adapter_kind": "process",
            "runtime_id": "runtime-test", "invocation_id": "invocation-test",
            "request_id": "request-test", "status": "succeeded", "output": "é😀"
        }))
        .expect("result fixture")
    }

    #[test]
    fn response_bytes_below_at_above_use_compact_utf8_json() {
        let original = result();
        let bytes = serde_json::to_vec(&original).unwrap().len() as u32;
        for delta in [-1_i32, 0, 1] {
            let mut result = original.clone();
            let limits = ResponseLimits {
                max_response_bytes: bytes.checked_add_signed(delta).unwrap(),
                ..ResponseLimits::default()
            };
            enforce_response(&mut result, limits).unwrap();
            assert_eq!(result.status == RunStatus::Succeeded, delta >= 0);
            if delta < 0 {
                let error = result.error.as_ref().unwrap().clone();
                assert_eq!(error.metadata["asset"], "RunResult");
                assert_eq!(error.metadata["observed"], bytes);
                assert!(error.message.contains("max_response_bytes"));
                assert_eq!(result.output, Value::Null);
                // Even a one-byte budget must deliver an explicit rejection.
                enforce_response(
                    &mut result,
                    ResponseLimits {
                        max_response_bytes: 1,
                        ..limits
                    },
                )
                .unwrap();
                assert_eq!(result.error.as_ref().unwrap(), &error);
            }
        }
    }

    #[test]
    fn diagnostic_exemption_requires_the_exact_sdk_rejection_shape() {
        let mut original = result();
        reject(&mut original, "max_response_bytes", 1, 1000, "RunResult");
        assert!(is_sdk_limit_diagnostic(&original));
        for (field, value) in [
            ("usage", serde_json::json!({})),
            ("metadata", serde_json::json!({"payload": "x"})),
            (
                "events",
                serde_json::json!([{"event_id":"stop", "timestamp_millis":0, "kind":"stop", "message":"payload"}]),
            ),
            ("telemetry", serde_json::json!([{"relay_enabled":false}])),
            ("artifacts", serde_json::json!({"root":"/tmp"})),
        ] {
            let mut mapping = serde_json::to_value(&original).unwrap();
            mapping[field] = value;
            let mut result: RunResult = serde_json::from_value(mapping).unwrap();
            assert!(!is_sdk_limit_diagnostic(&result), "{field}");
            enforce_response(
                &mut result,
                ResponseLimits {
                    max_response_bytes: 1,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(is_sdk_limit_diagnostic(&result));
            assert_eq!(
                result.error.unwrap().metadata["limit"],
                "max_response_bytes"
            );
        }
        let mut result = original.clone();
        result
            .error
            .as_mut()
            .unwrap()
            .metadata
            .insert("extra".into(), Value::Bool(true));
        assert!(!is_sdk_limit_diagnostic(&result));
        let mut result = original;
        result.error.as_mut().unwrap().message = "arbitrary payload".into();
        assert!(!is_sdk_limit_diagnostic(&result));
    }

    #[test]
    fn adapter_error_code_cannot_exempt_oversized_output_or_usage() {
        let mut result = result();
        result.status = RunStatus::Failed;
        result.error = Some(ErrorInfo {
            stage: ErrorStage::Invoke,
            code: "response_limit_exceeded".into(),
            message: "adapter error".into(),
            retryable: false,
            metadata: BTreeMap::new(),
        });
        result.usage = Some(RunUsage {
            metadata: BTreeMap::from([("large".into(), Value::String("x".repeat(100_000)))]),
            ..RunUsage::default()
        });
        enforce_response(
            &mut result,
            ResponseLimits {
                max_response_bytes: 1,
                ..ResponseLimits::default()
            },
        )
        .unwrap();
        assert!(result.usage.is_none());
        assert!(result.output.is_null());
        assert!(serde_json::to_vec(&result).unwrap().len() < 2048);
        assert_eq!(
            result.error.unwrap().metadata["limit"],
            "max_response_bytes"
        );
    }

    #[test]
    fn each_artifact_budget_is_independent_and_inclusive() {
        let root = std::env::temp_dir().join(new_id("response-limits"));
        std::fs::create_dir_all(root.join(".fabric")).unwrap();
        let mut original = result();
        original.artifacts.root = Some(root.clone());
        for (name, bytes) in [
            ("first.png", 6),
            ("second.png", 4),
            (".fabric/stdout.txt", 100),
        ] {
            let path = root.join(name);
            std::fs::write(&path, vec![0_u8; bytes]).unwrap();
            original.artifacts.artifacts.push(ArtifactRef {
                name: name.into(),
                kind: "file".into(),
                path,
                media_type: None,
                metadata: BTreeMap::new(),
            });
        }
        // Repeated references count once.
        original
            .artifacts
            .artifacts
            .push(original.artifacts.artifacts[0].clone());
        for (field, boundary) in [
            ("max_artifact_bytes", 10_u32),
            ("max_artifact_file_bytes", 6),
            ("max_artifact_files", 2),
        ] {
            for delta in [-1_i32, 0, 1] {
                let mut limits = ResponseLimits::default();
                let value = boundary.checked_add_signed(delta).unwrap();
                match field {
                    "max_artifact_bytes" => limits.max_artifact_bytes = value,
                    "max_artifact_file_bytes" => limits.max_artifact_file_bytes = value,
                    _ => limits.max_artifact_files = value,
                }
                let mut result = original.clone();
                enforce_local_artifacts(&mut result, limits).unwrap();
                assert_eq!(
                    result.status == RunStatus::Succeeded,
                    delta >= 0,
                    "{field} {value}"
                );
                if delta < 0 {
                    let error = result.error.unwrap();
                    assert_eq!(error.metadata["limit"], field);
                    assert!(error.message.contains(".png"));
                    assert!(result.artifacts.artifacts.is_empty());
                }
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }
}
