// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core config and runtime contract for NeMo Fabric.

pub mod adapter_contract;
pub mod agent_config;
pub mod agent_execution;
pub mod check;
pub mod config;
pub mod doctor;
mod endpoint;
mod environment;
pub mod error;
mod measurement;
pub mod model_usage;
pub mod pricing;
pub mod resource_snapshot;
pub mod runtime;
pub mod schema;
pub mod system_stats;
mod task_command;
mod time;
pub mod usage_accounting;
mod usage_normalization;
pub mod what_ran;

pub use adapter_contract::{ADAPTER_CONTRACT_VERSION, AdapterExtensionPoint};
pub use agent_execution::{
    AgentArtifact, AgentRunError, AgentRunRequest, AgentRunResult, AgentRunResultValidationError,
    AgentRunStatus, AgentUsage,
};
pub use check::{
    AdapterCheckRequest, AdapterCheckResult, CheckLevel, CheckObservation, CheckOutcome,
    CheckReason, CheckReport, CheckRequest, CheckRuntimeState, CheckScope, CheckSource, CheckStage,
    CheckVersion, RuntimeActivity, RuntimeObservation, ServerObservation, ServerState,
    check_environment, check_service,
};
pub use config::{
    AdapterConfigField, AdapterConfigSupport, AdapterDescriptor, AdapterInspection, AdapterKind,
    AdapterRequirements, AdapterTarget, AdapterTargetDescriptor, AdapterTargetType,
    AdapterTelemetryProviderSupport, AdapterTelemetrySupport, AgentConfig, AgentHarnessConfig,
    AgentInstructionConfig, AgentInstructionsConfig, AgentMcpConfig, AgentMcpServerConfig,
    AgentModelConfig, AgentRuntimeConfig, AgentSkillConfig, AgentToolDefinition, AgentToolsConfig,
    AgentWorkflowConfig, AgentWorkflowEntrypointConfig, CapabilityPlan, ControlLocation,
    DescriptorProvenance, DescriptorSource, DiscoveryConfig, EnvironmentConfig,
    EnvironmentOwnership, EnvironmentPlan, FabricConfig, HarnessConfig, InstructionConfig,
    InstructionMode, InstructionsConfig, McpAuthenticationConfig, McpConfig, McpExposure,
    McpServerPlan, McpTransport, MetadataConfig, ModelConfig, OAuthTokenEndpointAuthMethod,
    ResolutionStrategy, ResolveContext, ResolvedAdapterDescriptor, ResolvedAdapterTargetDescriptor,
    RunPlan, RuntimeCapabilities, RuntimeConfig, SkillConfig, TelemetryConfig, TelemetryPlan,
    TelemetryProvider, TelemetryProviderConfig, ToolDefinitionConfig, ToolsConfig, WorkflowConfig,
    WorkflowEntrypointConfig, WorkflowTargetSpec, inspect_adapter_config, load_adapter_descriptor,
    load_adapter_target_descriptor, resolve_diagnostic_plan_from_config,
    resolve_diagnostic_plan_from_config_with_adapter_directories, resolve_run_plan_from_config,
    resolve_run_plan_from_config_with_adapter_directories,
};
pub use doctor::{DoctorCheck, DoctorReport, DoctorStatus, doctor_plan};
pub use error::{FabricError, Result};
pub use resource_snapshot::RuntimeResourceSnapshot;
pub use runtime::{
    AdapterInvocation, ArtifactManifest, ArtifactRef, EnvironmentHandle, EnvironmentReference,
    ErrorInfo, ErrorStage, FabricEvent, InvocationHandle, OpenAiChatCompletionChunk,
    OpenAiChatCompletionChunkChoice, OpenAiChatCompletionChunkDelta,
    OpenAiChatCompletionChunkObject, OpenAiStreamHost, OpenAiStreamInvocation, OpenAiStreamProfile,
    OpenAiStreamProtocolVersion, OpenAiStreamRecord, OpenAiStreamSink, OpenAiStreamTransport,
    RunRequest, RunResult, RunStatus, RunUsage, RuntimeContext, RuntimeHandle,
    RuntimeTelemetryContext, ServiceHandle, ServiceOwnership, ServiceReference, TelemetryRef,
    attach_environment, attach_service, invoke_openai_stream, invoke_runtime, prepare_environment,
    prepare_service, release_environment, release_service, run_plan, runtime_resource_snapshot,
    start_runtime, start_runtime_in, start_runtime_with_service, stop_runtime,
};
pub use schema::{
    SchemaName, generate_all_schemas, generate_schema, generate_schema_json, write_schema_snapshots,
};
pub use system_stats::{
    CgroupObservation, CgroupSnapshot, CgroupStats, GaugeSummary, InvocationSystemStats,
    Measurement, ResourceObservation, SampleSummary, SystemStatsConfig, SystemStatsMetricGroup,
    SystemStatsSamplingConfig,
};

/// Returns the crate version compiled into this build.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
