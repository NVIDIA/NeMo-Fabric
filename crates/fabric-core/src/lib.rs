//! Core config and runtime contract for NeMo Fabric.

pub mod config;
pub mod doctor;
pub mod error;
pub mod runtime;
pub mod schema;

pub use config::{
    AdapterArtifactSupport, AdapterConfigSupport, AdapterDescriptor, AdapterDescriptorSource,
    AdapterKind, AdapterRequirements, AdapterResolutionConfig, AdapterTelemetrySupport,
    CapabilityPlan, ControlLocation, EnvironmentConfig, EnvironmentOwnership, EnvironmentPlan,
    FabricConfig, FabricDocument, HarnessConfig, McpConfig, McpExposure, McpServerPlan,
    MetadataConfig, ModelConfig, ProfileConfig, ResolutionStrategy, ResolvedAdapterDescriptor,
    RunPlan, RuntimeConfig, RuntimeMode, SkillConfig, TelemetryConfig, TelemetryPlan, Transport,
    load_adapter_descriptor, load_fabric_document, resolve_run_plan,
    resolve_run_plan_with_profiles,
};
pub use doctor::{DoctorCheck, DoctorReport, DoctorStatus, doctor_plan};
pub use error::{FabricError, Result};
pub use runtime::{
    ArtifactManifest, ArtifactRef, EnvironmentHandle, ErrorInfo, FabricEvent, InvocationHandle,
    RunRequest, RunResult, RunStatus, RuntimeHandle, TelemetryRef, invoke_runtime,
    prepare_environment, run_plan, start_runtime, stop_runtime,
};
pub use schema::{
    SchemaName, generate_all_schemas, generate_schema, generate_schema_json, write_schema_snapshots,
};

/// Returns the crate version compiled into this build.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
