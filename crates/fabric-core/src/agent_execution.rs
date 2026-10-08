// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request and result structures exchanged with an adapter target.

use std::collections::{BTreeMap, HashSet};
use std::path::{Component, Path, PathBuf};

use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use thiserror::Error;

/// Southbound invocation request passed to an adapter target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AgentRunRequest {
    /// Request payload for the adapter target.
    pub input: Value,
    /// UUID propagation root shared across conversation turns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relay_session_root: Option<String>,
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
    /// Cached input tokens consumed by the invocation, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub cached_input_tokens: Option<u64>,
    /// Input tokens written to the provider prompt cache, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub cache_write_input_tokens: Option<u64>,
    /// Whether input_tokens already includes cached_input_tokens and
    /// cache_write_input_tokens; absent means unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens_include_cache: Option<bool>,
    /// Output tokens produced by the invocation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub output_tokens: Option<u64>,
    /// Reasoning tokens produced by the invocation, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub reasoning_tokens: Option<u64>,
    /// Whether output_tokens already includes reasoning_tokens; absent means unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_include_reasoning: Option<bool>,
    /// Total tokens reported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub total_tokens: Option<u64>,
    /// Input tokens of the largest single model request in the invocation,
    /// including cached and cache-write input, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub peak_request_input_tokens: Option<u64>,
    /// Invocation cost in US dollars when reported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0))]
    pub cost_usd: Option<f64>,
    /// Usage broken out by the model that served the requests; empty when not
    /// reported. Each provider and model pair appears at most once, compared
    /// case-insensitively.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<AgentModelUsage>,
    /// Adapter-owned usage fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, Value>,
}

/// Usage of one model within one invocation.
///
/// Token fields have the same meaning as the matching `AgentUsage` fields,
/// restricted to requests served by this model.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AgentModelUsage {
    /// Model identifier as reported by the harness or provider response.
    #[schemars(length(min = 1), regex(pattern = r"\S"))]
    pub model: String,
    /// Provider or route that served the model, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1), regex(pattern = r"\S"))]
    pub provider: Option<String>,
    /// Input tokens consumed by this model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub input_tokens: Option<u64>,
    /// Cached input tokens consumed by this model, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub cached_input_tokens: Option<u64>,
    /// Input tokens this model wrote to the provider prompt cache, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub cache_write_input_tokens: Option<u64>,
    /// Whether input_tokens already includes cached_input_tokens and
    /// cache_write_input_tokens; absent means unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens_include_cache: Option<bool>,
    /// Output tokens produced by this model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub output_tokens: Option<u64>,
    /// Reasoning tokens produced by this model, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub reasoning_tokens: Option<u64>,
    /// Whether output_tokens already includes reasoning_tokens; absent means unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_include_reasoning: Option<bool>,
    /// Total tokens reported by the provider for this model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub total_tokens: Option<u64>,
    /// Input tokens of the largest single request served by this model,
    /// including cached and cache-write input, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub peak_request_input_tokens: Option<u64>,
    /// Cost in US dollars for this model when reported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0))]
    pub cost_usd: Option<f64>,
}

/// Southbound terminal result returned by an adapter target.
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
    /// A result string violated the non-blank contract.
    #[error("{0} must be a non-blank string")]
    BlankField(&'static str),
    /// An artifact path was blank, absolute, or contained parent traversal.
    #[error("artifact path must be non-blank and relative, and contain no parent traversal: {0}")]
    InvalidArtifactPath(PathBuf),
    /// Model usage repeated a provider and model pair, compared case-insensitively.
    #[error("usage.models must report each provider and model pair once: {0}")]
    DuplicateModelUsage(String),
    /// A cost was negative or not a finite number.
    #[error("{0} must be a finite number greater than or equal to 0")]
    InvalidCost(&'static str),
}

impl AgentRunResult {
    /// Validate terminal status, error, usage, and artifact invariants.
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
        if let Some(error) = &self.error {
            validate_nonblank(&error.code, "error.code")?;
            validate_nonblank(&error.message, "error.message")?;
        }
        if let Some(usage) = &self.usage {
            validate_model_usage(&usage.models)?;
        }
        for artifact in &self.artifacts {
            validate_nonblank(&artifact.name, "artifacts.name")?;
            validate_nonblank(&artifact.kind, "artifacts.kind")?;
            if let Some(media_type) = &artifact.media_type {
                validate_nonblank(media_type, "artifacts.media_type")?;
            }
            if !is_valid_agent_artifact_path(&artifact.path) {
                return Err(AgentRunResultValidationError::InvalidArtifactPath(
                    artifact.path.clone(),
                ));
            }
        }
        Ok(())
    }
}

fn validate_model_usage(
    models: &[AgentModelUsage],
) -> std::result::Result<(), AgentRunResultValidationError> {
    let mut reported = HashSet::with_capacity(models.len());
    for model_usage in models {
        validate_nonblank(&model_usage.model, "usage.models.model")?;
        if let Some(provider) = &model_usage.provider {
            validate_nonblank(provider, "usage.models.provider")?;
        }
        if let Some(cost_usd) = model_usage.cost_usd {
            validate_cost(cost_usd, "usage.models.cost_usd")?;
        }
        let key = (
            model_usage.provider.as_deref().map(str::to_lowercase),
            model_usage.model.to_lowercase(),
        );
        if !reported.insert(key) {
            let duplicate = match &model_usage.provider {
                Some(provider) => format!("provider {provider:?}, model {:?}", model_usage.model),
                None => format!("model {:?}", model_usage.model),
            };
            return Err(AgentRunResultValidationError::DuplicateModelUsage(
                duplicate,
            ));
        }
    }
    Ok(())
}

fn validate_cost(
    cost_usd: f64,
    field: &'static str,
) -> std::result::Result<(), AgentRunResultValidationError> {
    if cost_usd.is_finite() && cost_usd >= 0.0 {
        Ok(())
    } else {
        Err(AgentRunResultValidationError::InvalidCost(field))
    }
}

fn validate_nonblank(
    value: &str,
    field: &'static str,
) -> std::result::Result<(), AgentRunResultValidationError> {
    if is_nonblank(value) {
        Ok(())
    } else {
        Err(AgentRunResultValidationError::BlankField(field))
    }
}

fn is_nonblank(value: &str) -> bool {
    value.chars().any(|character| !character.is_whitespace())
}

fn is_valid_agent_artifact_path(path: &Path) -> bool {
    let raw = path.to_string_lossy();
    is_nonblank(&raw)
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
                "then": {"properties": {"error": {"type": "null"}}}
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

    #[test]
    fn rejects_blank_error_and_artifact_scalar_fields() {
        for (payload, field) in [
            (
                serde_json::json!({
                    "status": "failed",
                    "output": null,
                    "error": {"code": " \t", "message": "target failed"}
                }),
                "error.code",
            ),
            (
                serde_json::json!({
                    "status": "failed",
                    "output": null,
                    "error": {"code": "target_error", "message": " \t"}
                }),
                "error.message",
            ),
            (
                serde_json::json!({
                    "status": "succeeded",
                    "output": null,
                    "artifacts": [{"name": " \t", "kind": "file", "path": "output.txt"}]
                }),
                "artifacts.name",
            ),
            (
                serde_json::json!({
                    "status": "succeeded",
                    "output": null,
                    "artifacts": [{"name": "output", "kind": " \t", "path": "output.txt"}]
                }),
                "artifacts.kind",
            ),
            (
                serde_json::json!({
                    "status": "succeeded",
                    "output": null,
                    "artifacts": [{
                        "name": "output",
                        "kind": "file",
                        "path": "output.txt",
                        "media_type": " \t"
                    }]
                }),
                "artifacts.media_type",
            ),
        ] {
            let result = serde_json::from_value::<AgentRunResult>(payload)
                .expect("schema-constrained strings deserialize before runtime validation");
            assert_eq!(
                result.validate(),
                Err(AgentRunResultValidationError::BlankField(field))
            );
        }
    }

    fn result_with_usage(usage: Value) -> AgentRunResult {
        serde_json::from_value(serde_json::json!({
            "status": "succeeded",
            "output": null,
            "usage": usage
        }))
        .expect("usage payload deserializes")
    }

    fn model_usage(provider: Option<&str>, model: &str) -> AgentModelUsage {
        AgentModelUsage {
            model: model.to_string(),
            provider: provider.map(str::to_string),
            ..AgentModelUsage::default()
        }
    }

    fn result_with_models(models: Vec<AgentModelUsage>) -> AgentRunResult {
        AgentRunResult {
            status: AgentRunStatus::Succeeded,
            output: Value::Null,
            error: None,
            usage: Some(AgentUsage {
                models,
                ..AgentUsage::default()
            }),
            artifacts: Vec::new(),
            extensions: BTreeMap::new(),
        }
    }

    #[test]
    fn detailed_usage_round_trips_every_field() {
        let payload = serde_json::json!({
            "input_tokens": 1200,
            "cached_input_tokens": 800,
            "cache_write_input_tokens": 300,
            "input_tokens_include_cache": true,
            "output_tokens": 90,
            "reasoning_tokens": 40,
            "output_tokens_include_reasoning": false,
            "total_tokens": 1330,
            "peak_request_input_tokens": 700,
            "cost_usd": 0.5,
            "models": [{
                "model": "planner-model",
                "provider": "openai",
                "input_tokens": 1000,
                "cached_input_tokens": 700,
                "cache_write_input_tokens": 250,
                "input_tokens_include_cache": true,
                "output_tokens": 60,
                "reasoning_tokens": 30,
                "output_tokens_include_reasoning": true,
                "total_tokens": 1060,
                "peak_request_input_tokens": 600,
                "cost_usd": 0.375
            }, {
                "model": "summarizer-model"
            }],
            "extensions": {"source": "native"}
        });

        let usage: AgentUsage =
            serde_json::from_value(payload.clone()).expect("detailed usage deserializes");

        assert_eq!(
            usage,
            AgentUsage {
                input_tokens: Some(1200),
                cached_input_tokens: Some(800),
                cache_write_input_tokens: Some(300),
                input_tokens_include_cache: Some(true),
                output_tokens: Some(90),
                reasoning_tokens: Some(40),
                output_tokens_include_reasoning: Some(false),
                total_tokens: Some(1330),
                peak_request_input_tokens: Some(700),
                cost_usd: Some(0.5),
                models: vec![
                    AgentModelUsage {
                        model: "planner-model".to_string(),
                        provider: Some("openai".to_string()),
                        input_tokens: Some(1000),
                        cached_input_tokens: Some(700),
                        cache_write_input_tokens: Some(250),
                        input_tokens_include_cache: Some(true),
                        output_tokens: Some(60),
                        reasoning_tokens: Some(30),
                        output_tokens_include_reasoning: Some(true),
                        total_tokens: Some(1060),
                        peak_request_input_tokens: Some(600),
                        cost_usd: Some(0.375),
                    },
                    model_usage(None, "summarizer-model"),
                ],
                extensions: BTreeMap::from([("source".to_string(), serde_json::json!("native"),)]),
            }
        );
        assert_eq!(
            serde_json::to_value(&usage).expect("serialize usage"),
            payload
        );
        assert_eq!(result_with_usage(payload).validate(), Ok(()));
    }

    #[test]
    fn usage_without_detailed_fields_keeps_its_wire_shape() {
        let payload = serde_json::json!({
            "input_tokens": 3,
            "cached_input_tokens": 1,
            "input_tokens_include_cache": true,
            "output_tokens": 5,
            "total_tokens": 8,
            "cost_usd": 0.25
        });

        let result = result_with_usage(payload.clone());

        assert_eq!(result.validate(), Ok(()));
        let usage = result.usage.expect("usage");
        assert!(usage.models.is_empty());
        assert_eq!(
            serde_json::to_value(&usage).expect("serialize usage"),
            payload
        );
    }

    #[test]
    fn model_usage_rejects_unknown_fields() {
        for field in ["reasoning", "extensions"] {
            let error = serde_json::from_value::<AgentUsage>(serde_json::json!({
                "models": [{"model": "planner-model", field: {}}]
            }))
            .expect_err("model usage is a closed object");

            assert!(
                error
                    .to_string()
                    .contains(&format!("unknown field `{field}`")),
                "{error}"
            );
        }
    }

    #[test]
    fn rejects_blank_model_usage_identifiers() {
        for (models, field) in [
            (vec![model_usage(None, " \t")], "usage.models.model"),
            (
                vec![model_usage(Some(" \t"), "planner-model")],
                "usage.models.provider",
            ),
        ] {
            assert_eq!(
                result_with_models(models).validate(),
                Err(AgentRunResultValidationError::BlankField(field))
            );
        }
    }

    #[test]
    fn rejects_duplicate_model_usage_case_insensitively() {
        for (models, duplicate) in [
            (
                vec![
                    model_usage(None, "planner-model"),
                    model_usage(None, "Planner-Model"),
                ],
                r#"model "Planner-Model""#,
            ),
            (
                vec![
                    model_usage(Some("openai"), "planner-model"),
                    model_usage(Some("OpenAI"), "PLANNER-MODEL"),
                ],
                r#"provider "OpenAI", model "PLANNER-MODEL""#,
            ),
        ] {
            assert_eq!(
                result_with_models(models).validate(),
                Err(AgentRunResultValidationError::DuplicateModelUsage(
                    duplicate.to_string()
                ))
            );
        }
    }

    #[test]
    fn rejects_negative_or_non_finite_model_usage_cost() {
        for cost_usd in [-0.01, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let models = vec![AgentModelUsage {
                cost_usd: Some(cost_usd),
                ..model_usage(None, "planner-model")
            }];

            assert_eq!(
                result_with_models(models).validate(),
                Err(AgentRunResultValidationError::InvalidCost(
                    "usage.models.cost_usd"
                )),
                "cost_usd {cost_usd}"
            );
        }
    }

    #[test]
    fn accepts_zero_model_usage_cost() {
        let models = vec![AgentModelUsage {
            cost_usd: Some(0.0),
            ..model_usage(None, "planner-model")
        }];

        assert_eq!(result_with_models(models).validate(), Ok(()));
    }

    #[test]
    fn accepts_one_model_served_by_distinct_providers() {
        let result = result_with_models(vec![
            model_usage(None, "planner-model"),
            model_usage(Some("openai"), "planner-model"),
            model_usage(Some("azure"), "planner-model"),
        ]);

        assert_eq!(result.validate(), Ok(()));
    }
}
