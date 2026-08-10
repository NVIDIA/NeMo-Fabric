// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration projected southbound to an adapter target.

use std::collections::BTreeMap;
use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::{
    AdapterConfigField, AdapterDescriptor, CapabilityPlan, FabricConfig, InstructionMode,
};

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
    #[schemars(range(min = 1, max = u32::MAX))]
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
    /// MCP server URL for network transports or executable for stdio.
    #[schemars(length(min = 1), regex(pattern = r"\S"))]
    pub url: String,
    /// Command-line arguments passed to an MCP stdio process.
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

/// Project a resolved northbound config into the selected adapter target contract.
pub(crate) fn project_agent_config(
    config: &FabricConfig,
    capability_plan: &CapabilityPlan,
    descriptor: Option<&AdapterDescriptor>,
) -> AgentConfig {
    let accepts = |field: AdapterConfigField| {
        descriptor.is_some_and(|descriptor| descriptor.config.accepts.contains(&field))
    };

    let models = if accepts(AdapterConfigField::Models) {
        config
            .models
            .iter()
            .map(|(name, model)| {
                (
                    name.clone(),
                    AgentModelConfig {
                        provider: model.provider.clone(),
                        model: model.model.clone(),
                        api_key_env: model.api_key_env.clone(),
                        temperature: model.temperature,
                        base_url: model.base_url.clone(),
                        settings: model.settings.clone(),
                        extensions: model.extensions.clone(),
                    },
                )
            })
            .collect()
    } else {
        BTreeMap::new()
    };

    let instructions = config.instructions.as_ref().and_then(|instructions| {
        let system = instructions.system.as_ref().and_then(|system| {
            accepts(AdapterConfigField::SystemInstructions).then(|| AgentInstructionConfig {
                content: system.content.clone(),
                mode: system.mode,
                extensions: system.extensions.clone(),
            })
        });
        (system.is_some() || !instructions.extensions.is_empty()).then(|| AgentInstructionsConfig {
            system,
            extensions: instructions.extensions.clone(),
        })
    });

    let max_turns = accepts(AdapterConfigField::MaxTurns)
        .then_some(config.runtime.max_turns)
        .flatten();
    let runtime = (max_turns.is_some() || !config.runtime.extensions.is_empty()).then(|| {
        AgentRuntimeConfig {
            max_turns,
            extensions: config.runtime.extensions.clone(),
        }
    });

    let skills = config.skills.as_ref().and_then(|skills| {
        (!capability_plan.native.skill_paths.is_empty() || !skills.extensions.is_empty()).then(
            || AgentSkillConfig {
                paths: capability_plan.native.skill_paths.clone(),
                extensions: skills.extensions.clone(),
            },
        )
    });

    let mcp = config.mcp.as_ref().and_then(|mcp| {
        let servers = capability_plan
            .native
            .mcp_servers
            .iter()
            .map(|(name, server)| {
                (
                    name.clone(),
                    AgentMcpServerConfig {
                        transport: server.transport.clone(),
                        url: server.url.clone(),
                        args: server.args.clone(),
                        env: server.env.clone(),
                        allowed_tools: server.allowed_tools.clone(),
                        blocked_tools: server.blocked_tools.clone(),
                        extensions: server.extensions.clone(),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        (!servers.is_empty() || !mcp.extensions.is_empty()).then(|| AgentMcpConfig {
            servers,
            extensions: mcp.extensions.clone(),
        })
    });

    let tools = config.tools.as_ref().and_then(|tools| {
        let definitions = if accepts(AdapterConfigField::ToolDefinitions) {
            tools
                .definitions
                .iter()
                .map(|(name, definition)| {
                    (
                        name.clone(),
                        AgentToolDefinition {
                            kind: definition.kind.clone(),
                            r#ref: definition.r#ref.clone(),
                            settings: definition.settings.clone(),
                            extensions: definition.extensions.clone(),
                        },
                    )
                })
                .collect()
        } else {
            BTreeMap::new()
        };
        let enabled = accepts(AdapterConfigField::EnabledTools)
            .then(|| tools.enabled.clone())
            .flatten();
        let blocked = if accepts(AdapterConfigField::BlockedTools) {
            tools.blocked.clone()
        } else {
            Vec::new()
        };
        (enabled.is_some()
            || !blocked.is_empty()
            || !definitions.is_empty()
            || !tools.extensions.is_empty())
        .then(|| AgentToolsConfig {
            definitions,
            enabled,
            blocked,
            extensions: tools.extensions.clone(),
        })
    });

    let workflow = config
        .workflow
        .as_ref()
        .map(|workflow| AgentWorkflowConfig {
            entrypoint: AgentWorkflowEntrypointConfig {
                kind: workflow.entrypoint.kind.clone(),
                r#ref: workflow.entrypoint.r#ref.clone(),
                extensions: workflow.entrypoint.extensions.clone(),
            },
            settings: workflow.settings.clone(),
            extensions: workflow.extensions.clone(),
        });

    let harness = (!config.harness.settings.is_empty() || !config.harness.extensions.is_empty())
        .then(|| AgentHarnessConfig {
            settings: config.harness.settings.clone(),
            extensions: config.harness.extensions.clone(),
        });

    AgentConfig {
        harness,
        models,
        instructions,
        runtime,
        skills,
        mcp,
        tools,
        workflow,
        extensions: config.extensions.clone(),
    }
}
