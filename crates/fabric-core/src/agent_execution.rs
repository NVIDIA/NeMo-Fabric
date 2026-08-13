// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request and result structures exchanged with an adapter target.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use thiserror::Error;

/// Preview southbound invocation request.
///
/// The current local-host transport does not enforce this type. It will join
/// the negotiated adapter contract when typed invoke transport is implemented.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AgentRunRequest {
    /// Request payload for the adapter target.
    pub input: Value,
    /// Caller-provided task, rollout, workflow, or application context.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub context: BTreeMap<String, Value>,
    /// Adapter-owned request fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, Value>,
}

/// Completion status reported by an adapter target.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AgentRunStatus {
    /// The adapter target completed successfully.
    #[default]
    Succeeded,
    /// The adapter target completed with a failure.
    Failed,
    /// The adapter target cancelled the invocation.
    Cancelled,
}

/// Error reported by an adapter target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AgentRunError {
    /// Stable adapter error code.
    #[schemars(length(min = 1), regex(pattern = r"\S"))]
    pub code: String,
    /// Human-readable error message.
    #[schemars(length(min = 1), regex(pattern = r"\S"))]
    pub message: String,
    /// Whether the adapter considers the failure safe for a consumer-level retry.
    #[serde(default)]
    pub retryable: bool,
    /// Adapter-owned error fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, Value>,
}

/// One artifact produced by an adapter target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AgentArtifact {
    /// Logical artifact name.
    #[schemars(length(min = 1), regex(pattern = r"\S"))]
    pub name: String,
    /// Artifact kind.
    #[schemars(length(min = 1), regex(pattern = r"\S"))]
    pub kind: String,
    /// Path relative to the artifact root supplied in `RuntimeContext`.
    #[serde(deserialize_with = "deserialize_agent_artifact_path")]
    #[schemars(schema_with = "agent_artifact_path_schema")]
    pub path: PathBuf,
    /// Optional media type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1), regex(pattern = r"\S"))]
    pub media_type: Option<String>,
    /// Adapter-owned artifact fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, Value>,
}

/// Normalized model usage reported by an adapter target.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AgentUsage {
    /// Input tokens consumed by the invocation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub input_tokens: Option<u64>,
    /// Output tokens produced by the invocation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub output_tokens: Option<u64>,
    /// Total tokens reported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub total_tokens: Option<u64>,
    /// Invocation cost in US dollars when reported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0))]
    pub cost_usd: Option<f64>,
    /// Adapter-owned usage fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, Value>,
}

/// Preview southbound terminal result.
///
/// The current local-host transport does not decode this type. It will join
/// the negotiated adapter contract when typed invoke transport is implemented.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(transform = agent_run_result_schema)]
pub struct AgentRunResult {
    /// Adapter-target completion status.
    pub status: AgentRunStatus,
    /// Primary adapter-target output.
    pub output: Value,
    /// Adapter error when the invocation did not succeed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<AgentRunError>,
    /// Normalized model usage when reported by the adapter target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<AgentUsage>,
    /// Artifacts produced by the adapter target.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<AgentArtifact>,
    /// Adapter-owned result fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, Value>,
}

/// Invalid relationship or value within an adapter result.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AgentRunResultValidationError {
    /// A failed result omitted its required error.
    #[error("failed result requires an error")]
    FailedWithoutError,
    /// A successful result included an error.
    #[error("succeeded result must not include an error")]
    SucceededWithError,
    /// An artifact path was blank, absolute, or contained parent traversal.
    #[error("artifact path must be non-blank and relative, and contain no parent traversal: {0}")]
    InvalidArtifactPath(PathBuf),
}

impl AgentRunResult {
    /// Validate terminal status, error, and artifact invariants.
    pub fn validate(&self) -> std::result::Result<(), AgentRunResultValidationError> {
        match (self.status, self.error.as_ref()) {
            (AgentRunStatus::Failed, None) => {
                return Err(AgentRunResultValidationError::FailedWithoutError);
            }
            (AgentRunStatus::Succeeded, Some(_)) => {
                return Err(AgentRunResultValidationError::SucceededWithError);
            }
            _ => {}
        }
        if let Some(artifact) = self
            .artifacts
            .iter()
            .find(|artifact| !is_valid_agent_artifact_path(&artifact.path))
        {
            return Err(AgentRunResultValidationError::InvalidArtifactPath(
                artifact.path.clone(),
            ));
        }
        Ok(())
    }
}

fn is_valid_agent_artifact_path(path: &Path) -> bool {
    let raw = path.to_string_lossy();
    raw.chars().any(|character| !character.is_whitespace())
        && !path.is_absolute()
        && !raw.starts_with(['/', '\\'])
        && !raw
            .as_bytes()
            .get(0..2)
            .is_some_and(|prefix| prefix[0].is_ascii_alphabetic() && prefix[1] == b':')
        && !raw.split(['/', '\\']).any(|component| component == "..")
        && !path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
}

fn deserialize_agent_artifact_path<'de, D>(deserializer: D) -> Result<PathBuf, D::Error>
where
    D: Deserializer<'de>,
{
    let path = PathBuf::deserialize(deserializer)?;
    if !is_valid_agent_artifact_path(&path) {
        return Err(serde::de::Error::custom(
            "artifact path must be non-blank and relative, and contain no parent traversal",
        ));
    }
    Ok(path)
}

fn agent_artifact_path_schema(generator: &mut SchemaGenerator) -> Schema {
    let mut schema = String::json_schema(generator);
    schema.insert("minLength".into(), 1.into());
    schema.insert("pattern".into(), r"\S".into());
    schema.insert(
        "not".into(),
        serde_json::json!({
            "anyOf": [
                {"pattern": r"^[\\/]"},
                {"pattern": r"^[A-Za-z]:"},
                {"pattern": r"(^|[\\/])\.\.([\\/]|$)"}
            ]
        }),
    );
    schema
}

fn agent_run_result_schema(schema: &mut Schema) {
    schema.insert(
        "allOf".into(),
        serde_json::json!([
            {
                "if": {
                    "properties": {"status": {"const": "failed"}},
                    "required": ["status"]
                },
                "then": {
                    "properties": {"error": {"$ref": "#/$defs/AgentRunError"}},
                    "required": ["error"]
                }
            },
            {
                "if": {
                    "properties": {"status": {"const": "succeeded"}},
                    "required": ["status"]
                },
                "then": {"not": {"required": ["error"]}}
            }
        ]),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_terminal_status_and_error() {
        let failed_without_error = AgentRunResult {
            status: AgentRunStatus::Failed,
            output: Value::Null,
            error: None,
            usage: None,
            artifacts: Vec::new(),
            extensions: BTreeMap::new(),
        };
        assert_eq!(
            failed_without_error.validate(),
            Err(AgentRunResultValidationError::FailedWithoutError)
        );

        let error = AgentRunError {
            code: "target_error".to_string(),
            message: "target failed".to_string(),
            retryable: false,
            extensions: BTreeMap::new(),
        };
        let succeeded_with_error = AgentRunResult {
            status: AgentRunStatus::Succeeded,
            output: Value::Null,
            error: Some(error.clone()),
            usage: None,
            artifacts: Vec::new(),
            extensions: BTreeMap::new(),
        };
        assert_eq!(
            succeeded_with_error.validate(),
            Err(AgentRunResultValidationError::SucceededWithError)
        );

        for (status, error) in [
            (AgentRunStatus::Succeeded, None),
            (AgentRunStatus::Failed, Some(error.clone())),
            (AgentRunStatus::Cancelled, None),
            (AgentRunStatus::Cancelled, Some(error)),
        ] {
            let result = AgentRunResult {
                status,
                output: Value::Null,
                error,
                usage: None,
                artifacts: Vec::new(),
                extensions: BTreeMap::new(),
            };
            assert_eq!(result.validate(), Ok(()));
        }
    }

    #[test]
    fn rejects_unsafe_artifact_paths_during_deserialization() {
        for path in [
            "",
            " \t",
            "/tmp/output",
            "../output",
            "nested/../output",
            r"C:\tmp\output",
            r"C:output",
            r"\\server\output",
            r"nested\..\output",
        ] {
            let error = serde_json::from_value::<AgentArtifact>(serde_json::json!({
                "name": "output",
                "kind": "file",
                "path": path
            }))
            .expect_err("unsafe artifact path must fail");

            assert!(error.to_string().contains("artifact path must be"));
        }
    }

    #[test]
    fn validates_programmatically_constructed_artifact_paths() {
        let result = AgentRunResult {
            status: AgentRunStatus::Succeeded,
            output: Value::Null,
            error: None,
            usage: None,
            artifacts: vec![AgentArtifact {
                name: "output".to_string(),
                kind: "file".to_string(),
                path: PathBuf::from("../output"),
                media_type: None,
                extensions: BTreeMap::new(),
            }],
            extensions: BTreeMap::new(),
        };

        assert_eq!(
            result.validate(),
            Err(AgentRunResultValidationError::InvalidArtifactPath(
                PathBuf::from("../output")
            ))
        );
    }
}
