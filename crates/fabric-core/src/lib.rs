// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core config and runtime contract for NeMo Fabric.

pub mod config;
pub mod doctor;
pub mod error;
pub mod runtime;
pub mod schema;

pub use config::{
    AdapterConfigSupport, AdapterDescriptor, AdapterDescriptorSource, AdapterKind,
    AdapterRequirements, AdapterTelemetrySupport, CapabilityPlan, ControlLocation, EffectiveConfig,
    EnvironmentConfig, EnvironmentOwnership, EnvironmentPlan, FabricConfig, FabricDocument,
    HarnessConfig, McpConfig, McpExposure, McpServerPlan, MetadataConfig, ModelConfig,
    ProfileConfig, ResolutionStrategy, ResolveContext, ResolvedAdapterDescriptor, RunPlan,
    RuntimeCapabilities, RuntimeConfig, SkillConfig, TelemetryConfig, TelemetryPlan,
    TelemetryProvider, load_adapter_descriptor, load_fabric_document, resolve_effective_config,
    resolve_effective_config_from_config, resolve_effective_config_with_profiles, resolve_run_plan,
    resolve_run_plan_from_config, resolve_run_plan_from_effective_config,
    resolve_run_plan_with_profiles, validate_agent_directory,
};
pub use doctor::{DoctorCheck, DoctorReport, DoctorStatus, doctor_plan};
pub use error::{FabricError, Result};
pub use runtime::{
    AdapterInvocation, ArtifactManifest, ArtifactRef, EnvironmentHandle, ErrorInfo, ErrorStage,
    FabricEvent, InvocationHandle, RunRequest, RunResult, RunStatus, RuntimeContext, RuntimeHandle,
    RuntimeTelemetryContext, SessionHandle, SessionStatus, StartedSession, TelemetryRef,
    invoke_runtime, prepare_environment, run_plan, start_runtime, start_session, stop_runtime,
};
pub use schema::{
    SchemaName, generate_all_schemas, generate_schema, generate_schema_json, write_schema_snapshots,
};

/// Returns the crate version compiled into this build.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
