// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration projected southbound to an adapter target.

use std::collections::BTreeMap;
use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::InstructionMode;

/// Extensible block types within the southbound agent configuration.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum AgentConfigExtensionPoint {
    /// Root `AgentConfig.extensions`.
    Root,
    /// `AgentHarnessConfig.extensions`.
    Harness,
    /// `AgentModelConfig.extensions` for every named model role.
    Model,
    /// `AgentInstructionsConfig.extensions`.
    Instructions,
    /// `AgentInstructionConfig.extensions` for every instruction value.
    Instruction,
    /// `AgentRuntimeConfig.extensions`.
    Runtime,
    /// `AgentSkillConfig.extensions`.
    Skills,
    /// `AgentMcpConfig.extensions`.
    Mcp,
    /// `AgentMcpServerConfig.extensions` for every named server.
    McpServer,
    /// `AgentToolsConfig.extensions`.
    Tools,
    /// `AgentToolDefinition.extensions` for every named definition.
    ToolDefinition,
    /// `AgentWorkflowConfig.extensions`.
    Workflow,
    /// `AgentWorkflowEntrypointConfig.extensions`.
    WorkflowEntrypoint,
}

impl AgentConfigExtensionPoint {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Root => "root",
            Self::Harness => "harness",
            Self::Model => "model",
            Self::Instructions => "instructions",
            Self::Instruction => "instruction",
            Self::Runtime => "runtime",
            Self::Skills => "skills",
            Self::Mcp => "mcp",
            Self::McpServer => "mcp_server",
            Self::Tools => "tools",
            Self::ToolDefinition => "tool_definition",
            Self::Workflow => "workflow",
            Self::WorkflowEntrypoint => "workflow_entrypoint",
        }
    }
}

/// Configuration projected southbound to one adapter target.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AgentConfig {
    /// Adapter-owned target settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub harness: Option<AgentHarnessConfig>,
    /// Named model roles applied by the adapter target.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub models: BTreeMap<String, AgentModelConfig>,
    /// Normalized instructions applied by the adapter target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<AgentInstructionsConfig>,
    /// Adapter-applied runtime behavior.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<AgentRuntimeConfig>,
    /// Skills made available to the adapter target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skills: Option<AgentSkillConfig>,
    /// MCP servers routed to the adapter target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp: Option<AgentMcpConfig>,
    /// Named tool definitions and effective tool policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<AgentToolsConfig>,
    /// Custom agent or workflow selection and construction settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workflow: Option<AgentWorkflowConfig>,
    /// Adapter-owned fields validated against the selected adapter descriptor.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, Value>,
}

/// Adapter-owned target settings projected from `FabricConfig.harness`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AgentHarnessConfig {
    /// Target-specific settings validated by the selected adapter descriptor.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub settings: serde_json::Map<String, Value>,
    /// Adapter-owned harness fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, Value>,
}

/// Configuration for one named model role projected to an adapter target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AgentModelConfig {
    /// Model provider identifier.
    #[schemars(length(min = 1))]
    pub provider: String,
    /// Provider model identifier.
    #[schemars(length(min = 1))]
    pub model: String,
    /// Environment variable containing the provider credential.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// Optional model temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Optional provider API base URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Provider-specific model settings.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub settings: serde_json::Map<String, Value>,
    /// Adapter-owned model fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, Value>,
}

/// One normalized instruction value projected to an adapter target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AgentInstructionConfig {
    /// Instruction text.
    #[schemars(length(min = 1), regex(pattern = r"\S"))]
    pub content: String,
    /// How the instruction is applied.
    #[serde(default)]
    pub mode: InstructionMode,
    /// Adapter-owned instruction fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, Value>,
}

/// Normalized instructions projected to an adapter target.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AgentInstructionsConfig {
    /// System instructions for the selected adapter target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<AgentInstructionConfig>,
    /// Adapter-owned instruction categories.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, Value>,
}

/// Runtime behavior applied by an adapter target.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AgentRuntimeConfig {
    /// Maximum number of agent turns allowed for one invocation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub max_turns: Option<u32>,
    /// Adapter-owned runtime fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, Value>,
}

/// Skill paths made available to an adapter target.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AgentSkillConfig {
    /// Skill paths resolved for the task environment.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub paths: Vec<PathBuf>,
    /// Adapter-owned skill fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, Value>,
}

/// Named MCP servers routed to an adapter target.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AgentMcpConfig {
    /// MCP servers keyed by normalized server name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub servers: BTreeMap<String, AgentMcpServerConfig>,
    /// Adapter-owned MCP fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, Value>,
}

/// One MCP server routed to an adapter target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AgentMcpServerConfig {
    /// MCP transport identifier.
    #[schemars(length(min = 1), regex(pattern = r"\S"))]
    pub transport: String,
    /// MCP server URL or process command, depending on transport.
    #[schemars(length(min = 1), regex(pattern = r"\S"))]
    pub url: String,
    /// Arguments passed to an MCP stdio process.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    /// Environment variables passed to an MCP stdio process.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    /// MCP tool names to expose. `None` exposes every discovered tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<Vec<String>>,
    /// MCP tool names blocked after applying the optional allowlist.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocked_tools: Vec<String>,
    /// Adapter-owned MCP server fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, Value>,
}

/// One named tool or tool-group definition resolved by an adapter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AgentToolDefinition {
    /// Resolution semantics declared by the selected adapter descriptor.
    #[schemars(length(min = 1), regex(pattern = r"\S"))]
    pub kind: String,
    /// Executable or factory reference interpreted under `kind`.
    #[schemars(length(min = 1), regex(pattern = r"\S"))]
    pub r#ref: String,
    /// Definition-specific construction settings.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub settings: serde_json::Map<String, Value>,
    /// Adapter-owned tool-definition fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, Value>,
}

/// Named tool definitions and effective adapter-target tool policy.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AgentToolsConfig {
    /// Tool and tool-group definitions keyed by normalized name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub definitions: BTreeMap<String, AgentToolDefinition>,
    /// Named tools to expose. `None` preserves the adapter-target default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<Vec<String>>,
    /// Named tools to block.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocked: Vec<String>,
    /// Adapter-owned tool fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, Value>,
}

/// Adapter-declared resolution semantics for one custom agent or workflow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AgentWorkflowEntrypointConfig {
    /// Resolution semantics declared by the selected adapter descriptor.
    #[schemars(length(min = 1), regex(pattern = r"\S"))]
    pub kind: String,
    /// Executable or factory reference interpreted under `kind`.
    #[schemars(length(min = 1), regex(pattern = r"\S"))]
    pub r#ref: String,
    /// Adapter-owned entry-point fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, Value>,
}

/// Custom agent or workflow selection and construction settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AgentWorkflowConfig {
    /// Entry point resolved by the selected adapter.
    pub entrypoint: AgentWorkflowEntrypointConfig,
    /// Agent-specific construction settings.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub settings: serde_json::Map<String, Value>,
    /// Adapter-owned workflow fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, Value>,
}
