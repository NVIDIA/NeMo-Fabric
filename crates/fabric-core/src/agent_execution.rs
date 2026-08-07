// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request and result structures exchanged with an adapter target.

use std::collections::BTreeMap;
use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One invocation request projected southbound to an adapter target.
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
    pub input_tokens: Option<u64>,
    /// Output tokens produced by the invocation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    /// Total tokens reported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    /// Invocation cost in US dollars when reported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0))]
    pub cost_usd: Option<f64>,
    /// Adapter-owned usage fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, Value>,
}

/// Terminal result returned by an adapter target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
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

impl AgentRunResult {
    /// Validate the relationship between terminal status and adapter error.
    pub fn validate(&self) -> std::result::Result<(), &'static str> {
        match (self.status, self.error.as_ref()) {
            (AgentRunStatus::Failed, None) => Err("failed result requires an error"),
            (AgentRunStatus::Succeeded, Some(_)) => {
                Err("succeeded result must not include an error")
            }
            _ => Ok(()),
        }
    }
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
            Err("failed result requires an error")
        );

        let succeeded = AgentRunResult {
            status: AgentRunStatus::Succeeded,
            output: Value::Null,
            error: None,
            usage: None,
            artifacts: Vec::new(),
            extensions: BTreeMap::new(),
        };
        assert_eq!(succeeded.validate(), Ok(()));
    }
}
