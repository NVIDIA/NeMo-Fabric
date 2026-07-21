// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core config and runtime contract for NeMo Fabric.

pub mod config;
pub mod doctor;
pub mod error;
pub mod runtime;
pub mod schema;

pub use config::{
    ADAPTER_CONTRACT_VERSION, AdapterConfigSupport, AdapterDescriptor, AdapterDescriptorSource,
    AdapterKind, AdapterLocalHostSupport, AdapterRequirements, AdapterRuntimeSupport,
    AdapterTelemetryProviderSupport, AdapterTelemetrySupport, CapabilityPlan, ControlLocation,
    EnvironmentConfig, EnvironmentOwnership, EnvironmentPlan, FabricConfig, HarnessConfig,
    McpConfig, McpExposure, McpServerPlan, MetadataConfig, ModelConfig, ResolutionStrategy,
    ResolveContext, ResolvedAdapterDescriptor, RunPlan, RuntimeCapabilities, RuntimeConfig,
    SkillConfig, TelemetryConfig, TelemetryPlan, TelemetryProvider, TelemetryProviderConfig,
    load_adapter_descriptor, resolve_run_plan_from_config,
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
