// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core config and runtime contract for NeMo Fabric.

pub mod adapter_contract;
pub mod agent_config;
pub mod agent_execution;
pub mod config;
pub mod doctor;
pub mod error;
pub mod runtime;
pub mod schema;

pub use adapter_contract::{ADAPTER_CONTRACT_VERSION, AdapterExtensionPoint};
pub use agent_execution::{
    AgentArtifact, AgentRunError, AgentRunRequest, AgentRunResult, AgentRunResultValidationError,
    AgentRunStatus, AgentUsage,
};
pub use config::{
    AdapterConfigField, AdapterConfigInput, AdapterConfigSupport, AdapterDescriptor,
    AdapterDescriptorSource, AdapterKind, AdapterRequirements, AdapterTelemetryProviderSupport,
    AdapterTelemetrySupport, AgentConfig, AgentHarnessConfig, AgentInstructionConfig,
    AgentInstructionsConfig, AgentMcpConfig, AgentMcpServerConfig, AgentModelConfig,
    AgentRuntimeConfig, AgentSkillConfig, AgentToolDefinition, AgentToolsConfig,
    AgentWorkflowConfig, AgentWorkflowEntrypointConfig, CapabilityPlan, ControlLocation,
    EnvironmentConfig, EnvironmentOwnership, EnvironmentPlan, FabricConfig, HarnessConfig,
    InstructionConfig, InstructionMode, InstructionsConfig, McpAuthenticationConfig, McpConfig,
    McpExposure, McpServerPlan, McpTransport, MetadataConfig, ModelConfig,
    OAuthTokenEndpointAuthMethod, ResolutionStrategy, ResolveContext, ResolvedAdapterDescriptor,
    RunPlan, RuntimeCapabilities, RuntimeConfig, SkillConfig, TelemetryConfig, TelemetryPlan,
    TelemetryProvider, TelemetryProviderConfig, ToolDefinitionConfig, ToolsConfig, WorkflowConfig,
    WorkflowEntrypointConfig, load_adapter_descriptor, resolve_diagnostic_plan_from_config,
    resolve_diagnostic_plan_from_config_with_adapter_directories, resolve_run_plan_from_config,
    resolve_run_plan_from_config_with_adapter_directories,
};
pub use doctor::{DoctorCheck, DoctorReport, DoctorStatus, doctor_plan};
pub use error::{FabricError, Result};
pub use runtime::{
    AdapterInvocation, ArtifactManifest, ArtifactRef, EnvironmentHandle, ErrorInfo, ErrorStage,
    FabricEvent, InvocationHandle, RunRequest, RunResult, RunStatus, RuntimeContext, RuntimeHandle,
    RuntimeTelemetryContext, TelemetryRef, invoke_runtime, prepare_environment, run_plan,
    start_runtime, stop_runtime,
};
pub use schema::{
    SchemaName, generate_all_schemas, generate_schema, generate_schema_json, write_schema_snapshots,
};

/// Returns the crate version compiled into this build.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
