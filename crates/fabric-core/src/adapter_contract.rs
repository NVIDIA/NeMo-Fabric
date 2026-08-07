// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared southbound adapter contract metadata.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Southbound adapter contract version supported by this core.
pub const ADAPTER_CONTRACT_VERSION: &str = "fabric.adapter/v1alpha2";

/// Extensible block types within the southbound adapter contract.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum AdapterExtensionPoint {
    /// Root `AgentConfig.extensions`.
    AgentConfig,
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
    /// `AgentRunRequest.extensions`.
    RunRequest,
    /// `AgentRunResult.extensions`.
    RunResult,
    /// `AgentRunError.extensions`.
    RunError,
    /// `AgentArtifact.extensions`.
    Artifact,
    /// `AgentUsage.extensions`.
    Usage,
}

impl AdapterExtensionPoint {
    pub(crate) const ALL: [Self; 18] = [
        Self::AgentConfig,
        Self::Harness,
        Self::Model,
        Self::Instructions,
        Self::Instruction,
        Self::Runtime,
        Self::Skills,
        Self::Mcp,
        Self::McpServer,
        Self::Tools,
        Self::ToolDefinition,
        Self::Workflow,
        Self::WorkflowEntrypoint,
        Self::RunRequest,
        Self::RunResult,
        Self::RunError,
        Self::Artifact,
        Self::Usage,
    ];

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::AgentConfig => "agent_config",
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
            Self::RunRequest => "run_request",
            Self::RunResult => "run_result",
            Self::RunError => "run_error",
            Self::Artifact => "artifact",
            Self::Usage => "usage",
        }
    }
}
