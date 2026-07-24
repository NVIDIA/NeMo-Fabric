// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fabric config models and loading helpers.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{FabricError, Result};

/// Adapter descriptor contract version supported by this core.
pub const ADAPTER_CONTRACT_VERSION: &str = "fabric.adapter/v1alpha1";

/// Versioned NVIDIA NeMo Fabric agent config.
///
/// NeMo Fabric-owned fields apply uniformly, while adapter-translated fields are
/// validated against the selected adapter descriptor. See the
/// [configuration compatibility matrix](/sdk/python#normalized-configuration-compatibility).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FabricConfig {
    /// Config schema version.
    pub schema_version: String,
    /// Human-readable metadata.
    pub metadata: MetadataConfig,
    /// Harness selection and harness-specific settings.
    pub harness: HarnessConfig,
    /// Model aliases.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub models: BTreeMap<String, ModelConfig>,
    /// Portable system instructions for the selected harness.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// Maximum number of harness turns within one invocation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub max_turns: Option<u32>,
    /// Runtime input/output contract.
    pub runtime: RuntimeConfig,
    /// Environment where the harness or its tools execute.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment: Option<EnvironmentConfig>,
    /// Tool capability configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolsConfig>,
    /// Skill capability configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skills: Option<SkillConfig>,
    /// MCP capability configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp: Option<McpConfig>,
    /// Telemetry configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry: Option<TelemetryConfig>,
    /// First-class NeMo Relay integration configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relay: Option<RelayConfig>,
    /// Additive fields not yet recognized by this core version.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

/// Harness-neutral tool capability configuration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ToolsConfig {
    /// Adapter-native tool names to block.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocked: Vec<String>,
    /// Adapter-native toolset selection and blocking policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub toolsets: Option<ToolsetConfig>,
    /// Additive tool configuration fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

/// Harness-neutral toolset capability configuration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ToolsetConfig {
    /// Toolsets to expose. `None` preserves the adapter default; an empty list exposes none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<Vec<String>>,
    /// Toolsets to block.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocked: Vec<String>,
    /// Additive toolset configuration fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

/// Human-readable metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MetadataConfig {
    /// Agent/config name.
    pub name: String,
    /// Optional description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Additive metadata fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

/// Harness selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct HarnessConfig {
    /// Adapter implementation id.
    pub adapter_id: String,
    /// Selected install or availability strategy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<ResolutionStrategy>,
    /// Harness-specific settings.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub settings: serde_json::Map<String, Value>,
    /// Additive normalized harness fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

/// Language-neutral adapter descriptor for a harness integration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AdapterDescriptor {
    /// Adapter descriptor contract version.
    #[schemars(length(min = 1))]
    pub contract_version: String,
    /// Unique id for this adapter implementation.
    #[schemars(length(min = 1))]
    pub adapter_id: String,
    /// Stable machine-readable harness identifier implemented by this adapter.
    #[schemars(length(min = 1))]
    pub harness: String,
    /// Adapter implementation kind.
    pub adapter_kind: AdapterKind,
    /// Generic runner defaults consumed by the selected runtime adapter.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub runner: serde_json::Map<String, Value>,
    /// Runtime requirements.
    #[serde(default)]
    pub requirements: AdapterRequirements,
    /// Fabric config areas this adapter consumes or generates.
    #[serde(default)]
    pub config: AdapterConfigSupport,
    /// Telemetry support declared by this adapter.
    #[serde(default)]
    pub telemetry: AdapterTelemetrySupport,
    /// Runtime lifecycle operations supported by this adapter.
    #[serde(default)]
    pub capabilities: RuntimeCapabilities,
    /// Additive adapter descriptor fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

/// Where Fabric resolved an adapter descriptor from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AdapterDescriptorSource {
    /// Descriptor maintained in this Fabric repository.
    Repository,
    /// Descriptor registered by the agent package or local development config.
    Local,
}

/// Adapter descriptor selected for a run plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResolvedAdapterDescriptor {
    /// Registry source for this descriptor.
    pub source: AdapterDescriptorSource,
    /// Path to the adapter descriptor.
    pub path: PathBuf,
    /// Directory used to resolve descriptor-local runner and requirement paths.
    pub root: PathBuf,
    /// Adapter-owned compatibility and capability metadata.
    pub descriptor: AdapterDescriptor,
}

#[derive(Debug, Clone)]
struct AdapterRegistryEntry {
    source: AdapterDescriptorSource,
    path: PathBuf,
    root: PathBuf,
    descriptor: AdapterDescriptor,
}

#[derive(Debug, Clone, Default)]
struct AdapterRegistry {
    entries: BTreeMap<String, AdapterRegistryEntry>,
}

impl AdapterRegistry {
    fn from_config(
        _config: &FabricConfig,
        base_dir: &Path,
        additional_directories: &[PathBuf],
    ) -> Result<Self> {
        let mut registry = Self::default();
        registry.register_repository_directory(&repository_adapter_dir())?;
        for directory in additional_directories {
            registry.register_local_directory(directory)?;
        }
        registry.register_local_directory(&base_dir.join("adapters"))?;
        Ok(registry)
    }

    fn register_repository_directory(&mut self, directory: &Path) -> Result<()> {
        self.register_directory_tree(directory, AdapterDescriptorSource::Repository)
    }

    fn register_local_directory(&mut self, directory: &Path) -> Result<()> {
        self.register_directory_tree(directory, AdapterDescriptorSource::Local)
    }

    fn register_directory_tree(
        &mut self,
        directory: &Path,
        source: AdapterDescriptorSource,
    ) -> Result<()> {
        if !directory.is_dir() {
            return Ok(());
        }
        let entries = fs::read_dir(directory).map_err(|source| FabricError::Read {
            path: directory.to_path_buf(),
            source,
        })?;
        for entry in entries {
            let entry = entry.map_err(|source| FabricError::Read {
                path: directory.to_path_buf(),
                source,
            })?;
            let path = entry.path();
            if path.is_dir() {
                self.register_directory_tree(&path, source)?;
                continue;
            }
            if !is_adapter_descriptor_file(&path) {
                continue;
            }
            let descriptor = load_adapter_descriptor(&path)?;
            self.register_descriptor(path, source, descriptor)?;
        }
        Ok(())
    }

    fn register_descriptor(
        &mut self,
        path: PathBuf,
        source: AdapterDescriptorSource,
        descriptor: AdapterDescriptor,
    ) -> Result<()> {
        validate_adapter_descriptor_shape(&descriptor, &path)?;
        let path = path.canonicalize().unwrap_or(path);
        let root = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        self.entries.insert(
            descriptor.adapter_id.clone(),
            AdapterRegistryEntry {
                source,
                path,
                root,
                descriptor,
            },
        );
        Ok(())
    }

    fn get(&self, adapter_id: &str) -> Option<&AdapterRegistryEntry> {
        self.entries.get(adapter_id)
    }

    fn ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.entries.keys().cloned().collect();
        ids.sort();
        ids
    }
}

fn is_adapter_descriptor_file(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    name == "fabric-adapter.json"
}

fn repository_adapter_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("adapters")
}

/// Adapter install or availability strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionStrategy {
    /// Harness is already available in the prepared environment.
    Preinstalled,
    /// Environment image already contains the harness and dependencies.
    ImageProvided,
    /// Adapter may install a Python package with pip or uv.
    PipUv,
    /// Adapter may install a Node package.
    Npm,
    /// Adapter may install from source.
    Source,
    /// Adapter connects to an already-running service.
    Service,
    /// Adapter is installed through a harness-native plugin manager.
    NativePlugin,
}

/// Where Fabric control code runs relative to the environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ControlLocation {
    /// Fabric runs on the host/control plane and starts or connects to the harness in the environment.
    ExternalControl,
    /// Fabric runs inside the prepared environment with the harness.
    InEnvControl,
}

/// Whether Fabric owns the underlying environment resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EnvironmentOwnership {
    /// The caller or a surrounding system owns the environment resource.
    CallerOwned,
    /// Fabric created or leased the environment resource and may release it.
    FabricOwned,
}

/// Adapter runtime requirements.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AdapterRequirements {
    /// Required binaries.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub binaries: Vec<String>,
    /// Required environment variables.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub env: Vec<String>,
    /// Required files.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<PathBuf>,
    /// Required services.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub services: Vec<String>,
    /// Required harness plugin hooks.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub plugin_hooks: Vec<String>,
    /// Additive requirement fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

/// Adapter config support.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AdapterConfigSupport {
    /// Normalized NVIDIA NeMo Fabric config fields accepted by this adapter.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub accepts: Vec<AdapterConfigField>,
    /// Harness-native files generated by this adapter.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub generates: Vec<PathBuf>,
    /// Additive adapter config-support fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

/// Adapter-translated normalized NVIDIA NeMo Fabric configuration fields.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
pub enum AdapterConfigField {
    /// Normalized model selection and credentials.
    #[serde(rename = "models")]
    Models,
    /// Custom model endpoint.
    #[serde(rename = "models.base_url")]
    ModelBaseUrl,
    /// Model temperature.
    #[serde(rename = "models.temperature")]
    ModelTemperature,
    /// Portable system instructions.
    #[serde(rename = "system_prompt")]
    SystemPrompt,
    /// Per-invocation harness turn limit.
    #[serde(rename = "max_turns")]
    MaxTurns,
    /// Adapter-native tool names to block.
    #[serde(rename = "tools.blocked")]
    BlockedTools,
    /// Harness-defined toolsets to expose.
    #[serde(rename = "tools.toolsets.enabled")]
    EnabledToolsets,
    /// Harness-defined toolsets to block.
    #[serde(rename = "tools.toolsets.blocked")]
    BlockedToolsets,
    /// Harness-native MCP servers.
    #[serde(rename = "mcp")]
    Mcp,
    /// Harness-native skills.
    #[serde(rename = "skills")]
    Skills,
    /// Typed telemetry configuration. Exact support remains declared by telemetry provider.
    #[serde(rename = "telemetry")]
    Telemetry,
}

/// Adapter telemetry support.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AdapterTelemetrySupport {
    /// Provider-specific telemetry capabilities supported by this adapter.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub providers: BTreeMap<TelemetryProvider, AdapterTelemetryProviderSupport>,
    /// Additive adapter telemetry fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

/// Telemetry capabilities for one adapter-supported provider.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AdapterTelemetryProviderSupport {
    /// Telemetry outputs the adapter can produce or forward for this provider.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<String>,
    /// Integration modes implemented by the adapter for this provider.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub integration_modes: Vec<String>,
    /// Additive provider capability fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

/// Source context used when resolving an in-memory Fabric config.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolveContext {
    /// Base directory used to resolve relative Fabric paths.
    pub base_dir: PathBuf,
}

impl ResolveContext {
    /// Build a context with an explicit base directory.
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }
}

/// Adapter implementation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AdapterKind {
    /// Launch and supervise a persistent adapter process.
    Process,
    /// Connect to a service or HTTP-backed harness.
    Http,
    /// Launch and supervise a persistent Python adapter host.
    Python,
    /// Delegate to a harness-native plugin package.
    NativePlugin,
}

/// Model configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ModelConfig {
    /// Model provider name.
    pub provider: String,
    /// Provider model identifier.
    pub model: String,
    /// Optional temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Optional environment variable containing an API key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// Optional provider endpoint URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Provider-specific settings.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub settings: serde_json::Map<String, Value>,
    /// Additive normalized model fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

/// Runtime input/output contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RuntimeConfig {
    /// Input schema label.
    #[serde(default = "default_input_schema")]
    pub input_schema: String,
    /// Output schema label.
    #[serde(default = "default_output_schema")]
    pub output_schema: String,
    /// Artifact directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifacts: Option<PathBuf>,
    /// Maximum duration of one invocation in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("exclusiveMinimum" = 0.0))]
    pub timeout_seconds: Option<f64>,
    /// Additive normalized runtime fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

fn default_input_schema() -> String {
    "text".to_string()
}

fn default_output_schema() -> String {
    "text".to_string()
}

/// Execution environment configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EnvironmentConfig {
    /// Environment provider, for example `local`, `docker`, `opensandbox`, or `k8s`.
    pub provider: String,
    /// Where Fabric control code runs relative to the environment.
    #[serde(default = "default_control_location")]
    pub control_location: ControlLocation,
    /// Whether Fabric owns the environment resource.
    #[serde(default = "default_environment_ownership")]
    pub ownership: EnvironmentOwnership,
    /// Workspace path inside or outside the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace: Option<PathBuf>,
    /// Artifact path inside or outside the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifacts: Option<PathBuf>,
    /// Environment variables visible to the harness and its tools.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    /// Provider connection metadata, such as server URL, credential reference, or namespace.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub connection: serde_json::Map<String, Value>,
    /// Consumer-provided environment metadata.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub metadata: serde_json::Map<String, Value>,
    /// Provider-specific settings.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub settings: serde_json::Map<String, Value>,
    /// Additive normalized environment fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

fn default_control_location() -> ControlLocation {
    ControlLocation::InEnvControl
}

fn default_environment_ownership() -> EnvironmentOwnership {
    EnvironmentOwnership::CallerOwned
}

/// Skill capability configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Default)]
pub struct SkillConfig {
    /// Skill paths resolved relative to the config base directory.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub paths: Vec<PathBuf>,
    /// Additive skill fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

/// MCP capability configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Default)]
pub struct McpConfig {
    /// Named MCP servers.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub servers: BTreeMap<String, McpServerConfig>,
    /// Additive MCP fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

/// MCP server configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct McpServerConfig {
    /// MCP transport.
    pub transport: String,
    /// MCP server URL or process command, depending on transport.
    pub url: String,
    /// How Fabric exposes the MCP capability to the harness.
    pub exposure: McpExposure,
    /// Additive MCP server fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

/// MCP exposure strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum McpExposure {
    /// Map into harness-native MCP config through the selected adapter.
    HarnessNative,
    /// Fabric manages MCP and exposes basic tools/actions.
    FabricManaged,
}

/// Telemetry configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TelemetryConfig {
    /// Telemetry providers enabled for this run.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub providers: BTreeMap<TelemetryProvider, TelemetryProviderConfig>,
    /// Additive telemetry fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

/// Provider-specific telemetry configuration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TelemetryProviderConfig {
    /// Provider-specific pass-through config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<Value>,
    /// Additive provider fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

/// NeMo Relay integration configuration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RelayConfig {
    /// Optional project name for Relay backends.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    /// Optional Relay output directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_dir: Option<PathBuf>,
    /// Relay observability component configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observability: Option<RelayObservabilityConfig>,
    /// Additional Relay plugin components.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub components: Vec<RelayComponentConfig>,
    /// Relay plugin validation policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy: Option<RelayConfigPolicy>,
    /// Additive Relay fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

/// Generic NeMo Relay plugin component configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RelayComponentConfig {
    /// Registered Relay plugin kind.
    pub kind: String,
    /// Whether this Relay component should be activated.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// Component-local Relay plugin config.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub config: BTreeMap<String, Value>,
    /// Additive component fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

/// NeMo Relay observability component configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RelayObservabilityConfig {
    /// Relay observability config version.
    #[serde(default = "default_relay_config_version")]
    pub version: u32,
    /// ATOF export configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub atof: Option<RelayAtofConfig>,
    /// ATIF export configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub atif: Option<RelayAtifConfig>,
    /// OpenTelemetry export configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opentelemetry: Option<RelayOtlpConfig>,
    /// OpenInference export configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openinference: Option<RelayOtlpConfig>,
    /// Relay config validation policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy: Option<RelayConfigPolicy>,
    /// Additive observability fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

impl Default for RelayObservabilityConfig {
    fn default() -> Self {
        Self {
            version: default_relay_config_version(),
            atof: None,
            atif: None,
            opentelemetry: None,
            openinference: None,
            policy: None,
            extensions: BTreeMap::new(),
        }
    }
}

/// Relay ATOF export configuration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RelayAtofConfig {
    /// Whether ATOF export is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// ATOF file and stream sinks.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sinks: Vec<RelayAtofSinkConfig>,
    /// Additive ATOF fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

/// Relay ATOF sink configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RelayAtofSinkConfig {
    /// Write ATOF records to a local file.
    File {
        /// Directory used for ATOF files.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output_directory: Option<PathBuf>,
        /// ATOF file name.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        /// File write mode.
        #[serde(default)]
        mode: RelayAtofMode,
        /// Additive file sink fields.
        #[serde(default, flatten)]
        extensions: BTreeMap<String, Value>,
    },
    /// Send ATOF records to a remote stream.
    Stream {
        /// Stream URL.
        url: String,
        /// Stream transport.
        #[serde(default)]
        transport: RelayAtofStreamTransport,
        /// Static stream headers.
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        headers: BTreeMap<String, String>,
        /// Environment-variable-backed stream headers.
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        header_env: BTreeMap<String, String>,
        /// Request timeout in milliseconds.
        #[serde(default = "default_relay_timeout_millis")]
        timeout_millis: u64,
        /// Field-name handling policy.
        #[serde(default)]
        field_name_policy: RelayAtofStreamFieldNamePolicy,
        /// Optional stream sink name.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// Additive stream sink fields.
        #[serde(default, flatten)]
        extensions: BTreeMap<String, Value>,
    },
}

/// Relay ATIF export configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RelayAtifConfig {
    /// Whether ATIF export is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Agent name written into ATIF.
    #[serde(default = "default_relay_atif_agent_name")]
    pub agent_name: String,
    /// Agent version written into ATIF.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_version: Option<String>,
    /// Model name written into ATIF.
    #[serde(default = "default_relay_atif_model_name")]
    pub model_name: String,
    /// Tool definitions written into ATIF.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_definitions: Option<Vec<Value>>,
    /// Extra ATIF metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Value>,
    /// Directory used for ATIF files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_directory: Option<PathBuf>,
    /// ATIF file name template.
    #[serde(default = "default_relay_atif_filename_template")]
    pub filename_template: String,
    /// Optional ATIF remote storage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage: Option<Vec<RelayAtifStorageConfig>>,
    /// Additive ATIF fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

impl Default for RelayAtifConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            agent_name: default_relay_atif_agent_name(),
            agent_version: None,
            model_name: default_relay_atif_model_name(),
            tool_definitions: None,
            extra: None,
            output_directory: None,
            filename_template: default_relay_atif_filename_template(),
            storage: None,
            extensions: BTreeMap::new(),
        }
    }
}

/// Relay ATIF remote storage configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RelayAtifStorageConfig {
    /// Upload ATIF artifacts to HTTP storage.
    Http {
        /// HTTP storage endpoint.
        #[serde(default)]
        endpoint: String,
        /// Static HTTP headers.
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        headers: BTreeMap<String, String>,
        /// Environment-variable-backed HTTP headers.
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        header_env: BTreeMap<String, String>,
        /// Request timeout in milliseconds.
        #[serde(default = "default_relay_timeout_millis")]
        timeout_millis: u64,
        /// Additive HTTP storage fields.
        #[serde(default, flatten)]
        extensions: BTreeMap<String, Value>,
    },
    /// Upload ATIF artifacts to S3-compatible storage.
    S3 {
        /// S3 bucket name.
        #[serde(default)]
        bucket: String,
        /// Optional S3 object key prefix.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        key_prefix: Option<String>,
        /// AWS access key id.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        access_key_id: Option<String>,
        /// Environment variable containing the AWS secret access key.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        secret_access_key_var: Option<String>,
        /// Environment variable containing the AWS session token.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        session_token_var: Option<String>,
        /// AWS region.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        region: Option<String>,
        /// S3-compatible endpoint URL.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        endpoint_url: Option<String>,
        /// Allow HTTP endpoints for S3-compatible storage.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        allow_http: Option<bool>,
        /// Additive S3 storage fields.
        #[serde(default, flatten)]
        extensions: BTreeMap<String, Value>,
    },
}

/// Relay OpenTelemetry/OpenInference export configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RelayOtlpConfig {
    /// Whether OTLP export is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// OTLP transport.
    #[serde(default)]
    pub transport: RelayOtlpTransport,
    /// OTLP endpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// OTLP headers.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    /// OTLP resource attributes.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub resource_attributes: BTreeMap<String, String>,
    /// OTLP service name.
    #[serde(default = "default_relay_service_name")]
    pub service_name: String,
    /// OTLP service namespace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_namespace: Option<String>,
    /// OTLP service version.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_version: Option<String>,
    /// OTLP instrumentation scope.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instrumentation_scope: Option<String>,
    /// Request timeout in milliseconds.
    #[serde(default = "default_relay_timeout_millis")]
    pub timeout_millis: u64,
    /// Additive OTLP fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

impl Default for RelayOtlpConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            transport: RelayOtlpTransport::default(),
            endpoint: None,
            headers: BTreeMap::new(),
            resource_attributes: BTreeMap::new(),
            service_name: default_relay_service_name(),
            service_namespace: None,
            service_version: None,
            instrumentation_scope: None,
            timeout_millis: default_relay_timeout_millis(),
            extensions: BTreeMap::new(),
        }
    }
}

/// Relay validation policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RelayConfigPolicy {
    /// Policy for unknown components.
    #[serde(default)]
    pub unknown_component: RelayUnsupportedBehavior,
    /// Policy for unknown fields.
    #[serde(default)]
    pub unknown_field: RelayUnsupportedBehavior,
    /// Policy for unsupported values.
    #[serde(default = "default_relay_unsupported_value_behavior")]
    pub unsupported_value: RelayUnsupportedBehavior,
}

impl Default for RelayConfigPolicy {
    fn default() -> Self {
        Self {
            unknown_component: RelayUnsupportedBehavior::default(),
            unknown_field: RelayUnsupportedBehavior::default(),
            unsupported_value: default_relay_unsupported_value_behavior(),
        }
    }
}

/// Relay unsupported/unknown config handling.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RelayUnsupportedBehavior {
    /// Ignore the unsupported or unknown value.
    Ignore,
    /// Warn on the unsupported or unknown value.
    #[default]
    Warn,
    /// Error on the unsupported or unknown value.
    Error,
}

/// Relay ATOF file mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RelayAtofMode {
    /// Append to an existing ATOF file.
    #[default]
    Append,
    /// Overwrite an existing ATOF file.
    Overwrite,
}

/// Relay ATOF stream transport.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RelayAtofStreamTransport {
    /// HTTP POST transport.
    #[default]
    HttpPost,
    /// WebSocket transport.
    Websocket,
    /// NDJSON transport.
    Ndjson,
}

/// Relay ATOF stream field-name policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RelayAtofStreamFieldNamePolicy {
    /// Preserve field names.
    #[default]
    Preserve,
    /// Replace dots in field names.
    ReplaceDots,
}

/// Relay OTLP transport.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RelayOtlpTransport {
    /// OTLP HTTP binary transport.
    #[default]
    HttpBinary,
    /// OTLP gRPC transport.
    Grpc,
}

/// Telemetry runtime provider.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum TelemetryProvider {
    /// Use NeMo Relay for telemetry integration.
    #[default]
    Relay,
    /// Let the selected adapter handle telemetry natively.
    Native,
}

impl TelemetryProvider {
    /// Return the stable configuration value for this provider.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Relay => "relay",
            Self::Native => "native",
        }
    }
}

fn default_relay_config_version() -> u32 {
    2
}

fn default_enabled() -> bool {
    true
}

fn default_relay_atif_agent_name() -> String {
    "NeMo Relay".to_string()
}

fn default_relay_atif_model_name() -> String {
    "unknown".to_string()
}

fn default_relay_atif_filename_template() -> String {
    "nemo-relay-atif-{session_id}.json".to_string()
}

fn default_relay_timeout_millis() -> u64 {
    3000
}

fn default_relay_service_name() -> String {
    "nemo-relay".to_string()
}

fn default_relay_unsupported_value_behavior() -> RelayUnsupportedBehavior {
    RelayUnsupportedBehavior::Error
}

/// Load an adapter descriptor from JSON package metadata.
pub fn load_adapter_descriptor(path: impl AsRef<Path>) -> Result<AdapterDescriptor> {
    let path = path.as_ref();
    let descriptor = read_json(path)?;
    validate_adapter_descriptor_shape(&descriptor, path)?;
    Ok(descriptor)
}

fn validate_config(config: &FabricConfig) -> Result<()> {
    if config.harness.adapter_id.trim().is_empty() {
        return Err(FabricError::UnknownAdapter {
            adapter_id: config.harness.adapter_id.clone(),
            available: Vec::new(),
        });
    }
    if config.max_turns == Some(0) {
        return invalid_config("max_turns", "must be greater than zero");
    }
    if let Some(timeout) = config.runtime.timeout_seconds
        && (!timeout.is_finite()
            || timeout <= 0.0
            || std::time::Duration::try_from_secs_f64(timeout).is_err())
    {
        return invalid_config(
            "runtime.timeout_seconds",
            "must be a finite number greater than zero",
        );
    }
    for (role, model) in &config.models {
        if model.provider.trim().is_empty()
            || model.provider.trim() != model.provider
            || model.provider.to_ascii_lowercase() != model.provider
        {
            return invalid_config(
                format!("models.{role}.provider"),
                "must be a non-empty lowercase identifier",
            );
        }
        if model.model.trim().is_empty() {
            return invalid_config(format!("models.{role}.model"), "must be a non-empty string");
        }
        if model
            .api_key_env
            .as_ref()
            .is_some_and(|name| name.trim().is_empty())
        {
            return invalid_config(
                format!("models.{role}.api_key_env"),
                "must be a non-empty string",
            );
        }
        if let Some(base_url) = &model.base_url
            && base_url.trim().is_empty()
        {
            return invalid_config(
                format!("models.{role}.base_url"),
                "must be a non-empty string",
            );
        }
        if model.settings.contains_key("base_url") {
            return invalid_config(
                format!("models.{role}.settings.base_url"),
                "is not supported; use models.<role>.base_url",
            );
        }
    }
    if let Some(environment) = &config.environment {
        for name in environment.env.keys() {
            if name.trim().is_empty() {
                return invalid_config("environment.env", "variable names must not be empty");
            }
        }
    }
    if let Some(tools) = &config.tools {
        validate_names("tools.blocked", &tools.blocked)?;
        if let Some(toolsets) = &tools.toolsets {
            if let Some(enabled) = &toolsets.enabled {
                validate_names("tools.toolsets.enabled", enabled)?;
                if let Some(name) = enabled.iter().find(|name| toolsets.blocked.contains(name)) {
                    return invalid_config(
                        "tools.toolsets",
                        format!("`{name}` cannot be both enabled and blocked"),
                    );
                }
            }
            validate_names("tools.toolsets.blocked", &toolsets.blocked)?;
        }
    }
    Ok(())
}

fn validate_names(field: &str, names: &[String]) -> Result<()> {
    if names.iter().any(|name| name.trim().is_empty()) {
        return invalid_config(field, "entries must be non-empty strings");
    }
    Ok(())
}

fn invalid_config<T>(field: impl Into<String>, reason: impl Into<String>) -> Result<T> {
    Err(FabricError::InvalidConfig {
        field: field.into(),
        reason: reason.into(),
    })
}

/// Resolve a typed Fabric config into a runnable plan.
///
/// Callers provide an already-composed typed config and the explicit base
/// directory used for resolving relative paths.
pub fn resolve_run_plan_from_config(
    config: FabricConfig,
    context: ResolveContext,
) -> Result<RunPlan> {
    resolve_run_plan_from_config_with_adapter_directories(config, context, &[])
}

/// Resolve a typed Fabric config with additional adapter descriptor directories.
///
/// This is an internal integration surface for hosts that know about
/// environment-specific package data directories. Callers should otherwise use
/// [`resolve_run_plan_from_config`].
#[doc(hidden)]
pub fn resolve_run_plan_from_config_with_adapter_directories(
    config: FabricConfig,
    context: ResolveContext,
    adapter_directories: &[PathBuf],
) -> Result<RunPlan> {
    validate_config(&config)?;
    let supplied_base_dir = context.base_dir;
    let base_dir = std::path::absolute(&supplied_base_dir)
        .map(normalize_path)
        .map_err(|source| FabricError::ResolveBaseDirectory {
            path: supplied_base_dir,
            source,
        })?;
    resolve_run_plan(config, base_dir, adapter_directories)
}

fn read_json<T>(path: &Path) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let raw = std::fs::read_to_string(path).map_err(|source| FabricError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&raw).map_err(|source| FabricError::ParseJson {
        path: path.to_path_buf(),
        source,
    })
}

fn resolve_run_plan(
    config: FabricConfig,
    base_dir: PathBuf,
    adapter_directories: &[PathBuf],
) -> Result<RunPlan> {
    let adapter_descriptor = resolve_adapter_descriptor(&config, &base_dir, adapter_directories)?;
    let descriptor = adapter_descriptor
        .as_ref()
        .map(|adapter| &adapter.descriptor);
    validate_adapter_config_compatibility(&config, descriptor)?;
    let resolution = resolve_resolution(&config, descriptor)?;
    let environment_plan = resolve_environment_plan(&config, &base_dir);
    validate_control_location(descriptor, environment_plan.as_ref())?;
    let capability_plan = resolve_capability_plan(&config, &base_dir, adapter_descriptor.as_ref());
    validate_capability_plan_compatibility(&capability_plan, descriptor)?;
    let capabilities = resolve_runtime_capabilities(&config, descriptor);
    let telemetry_plan = resolve_telemetry_plan(&config, descriptor)?;
    Ok(RunPlan {
        agent_name: config.metadata.name.clone(),
        base_dir,
        config,
        adapter_descriptor,
        resolution,
        environment_plan,
        capability_plan,
        capabilities,
        telemetry_plan,
    })
}

fn validate_capability_plan_compatibility(
    capability_plan: &CapabilityPlan,
    descriptor: Option<&AdapterDescriptor>,
) -> Result<()> {
    let Some(route) = capability_plan
        .routes
        .iter()
        .find(|route| route.target == CapabilityTarget::Unsupported)
    else {
        return Ok(());
    };
    Err(FabricError::AdapterCompatibility {
        adapter_id: descriptor
            .map(|descriptor| descriptor.adapter_id.clone())
            .unwrap_or_else(|| "unknown".to_string()),
        field: route.config_field(),
        reason: route.reason.clone(),
    })
}

fn validate_adapter_config_compatibility(
    config: &FabricConfig,
    descriptor: Option<&AdapterDescriptor>,
) -> Result<()> {
    let Some(descriptor) = descriptor else {
        return Ok(());
    };
    let accepts = |field: AdapterConfigField| descriptor.config.accepts.contains(&field);
    let incompatible = |field: String, reason: String| {
        Err(FabricError::AdapterCompatibility {
            adapter_id: descriptor.adapter_id.clone(),
            reason,
            field,
        })
    };

    if config.system_prompt.is_some() && !accepts(AdapterConfigField::SystemPrompt) {
        return incompatible(
            "system_prompt".to_string(),
            "the adapter does not declare an equivalent native mapping".to_string(),
        );
    }
    if config.max_turns.is_some() && !accepts(AdapterConfigField::MaxTurns) {
        return incompatible(
            "max_turns".to_string(),
            "the adapter does not declare an equivalent native mapping".to_string(),
        );
    }
    if !config.models.is_empty() && !accepts(AdapterConfigField::Models) {
        return incompatible(
            "models".to_string(),
            "the adapter does not consume normalized model configuration".to_string(),
        );
    }
    let selected_model = match (config.models.get_key_value("default"), config.models.len()) {
        (Some(model), _) => Some(model),
        (None, 0) => None,
        (None, 1) => config.models.first_key_value(),
        (None, _) => {
            return incompatible(
                "models".to_string(),
                "multiple model roles are configured and no default role selects one".to_string(),
            );
        }
    };
    if let Some((role, model)) = selected_model {
        if model.base_url.is_some() && !accepts(AdapterConfigField::ModelBaseUrl) {
            return incompatible(
                format!("models.{role}.base_url"),
                "the adapter does not declare custom endpoint support".to_string(),
            );
        }
        if model.temperature.is_some() && !accepts(AdapterConfigField::ModelTemperature) {
            return incompatible(
                format!("models.{role}.temperature"),
                "the adapter does not declare an equivalent native mapping".to_string(),
            );
        }
    }
    Ok(())
}

fn resolve_adapter_descriptor(
    config: &FabricConfig,
    base_dir: &Path,
    adapter_directories: &[PathBuf],
) -> Result<Option<ResolvedAdapterDescriptor>> {
    let adapter_id = &config.harness.adapter_id;
    let registry = AdapterRegistry::from_config(config, base_dir, adapter_directories)?;
    let Some(entry) = registry.get(adapter_id) else {
        return Err(FabricError::UnknownAdapter {
            adapter_id: adapter_id.clone(),
            available: registry.ids(),
        });
    };
    validate_adapter_descriptor(config, adapter_id, &entry.descriptor, entry.path.clone())?;
    Ok(Some(ResolvedAdapterDescriptor {
        source: entry.source,
        path: entry.path.clone(),
        root: entry.root.clone(),
        descriptor: entry.descriptor.clone(),
    }))
}

fn validate_adapter_descriptor(
    _config: &FabricConfig,
    expected_adapter_id: &str,
    descriptor: &AdapterDescriptor,
    path: PathBuf,
) -> Result<()> {
    if descriptor.adapter_id != expected_adapter_id {
        return Err(FabricError::AdapterDescriptorMismatch {
            path,
            field: "adapter_id",
            expected: expected_adapter_id.to_string(),
            actual: descriptor.adapter_id.clone(),
        });
    }
    Ok(())
}

fn validate_adapter_descriptor_shape(descriptor: &AdapterDescriptor, path: &Path) -> Result<()> {
    if descriptor.contract_version.trim().is_empty() {
        return invalid_adapter_descriptor(path, "contract_version must not be empty");
    }
    if descriptor.contract_version != ADAPTER_CONTRACT_VERSION {
        return Err(FabricError::AdapterDescriptorUnsupported {
            adapter_id: descriptor.adapter_id.clone(),
            field: "contract_version",
            value: descriptor.contract_version.clone(),
        });
    }
    if descriptor.adapter_id.trim().is_empty() {
        return invalid_adapter_descriptor(path, "adapter_id must not be empty");
    }
    if descriptor.harness.trim().is_empty() {
        return invalid_adapter_descriptor(path, "harness must not be empty");
    }
    Ok(())
}

fn invalid_adapter_descriptor<T>(path: &Path, message: impl Into<String>) -> Result<T> {
    Err(FabricError::InvalidAdapterDescriptor {
        path: path.to_path_buf(),
        message: message.into(),
    })
}

fn validate_control_location(
    _adapter_descriptor: Option<&AdapterDescriptor>,
    _environment_plan: Option<&EnvironmentPlan>,
) -> Result<()> {
    Ok(())
}

fn resolve_runtime_capabilities(
    _config: &FabricConfig,
    descriptor: Option<&AdapterDescriptor>,
) -> RuntimeCapabilities {
    let implemented_runtime = descriptor.is_some_and(|descriptor| {
        matches!(
            descriptor.adapter_kind,
            AdapterKind::Process | AdapterKind::Python
        )
    });
    let descriptor_capabilities = descriptor
        .map(|descriptor| descriptor.capabilities.clone())
        .unwrap_or_default();
    RuntimeCapabilities {
        service: implemented_runtime && descriptor_capabilities.service,
        streaming: implemented_runtime && descriptor_capabilities.streaming,
        updates: implemented_runtime && descriptor_capabilities.updates,
        cancellation: implemented_runtime && descriptor_capabilities.cancellation,
        metadata: descriptor_capabilities.metadata,
    }
}

fn resolve_resolution(
    config: &FabricConfig,
    _adapter_descriptor: Option<&AdapterDescriptor>,
) -> Result<Option<ResolutionStrategy>> {
    Ok(config.harness.resolution)
}

fn resolve_environment_plan(config: &FabricConfig, base_dir: &Path) -> Option<EnvironmentPlan> {
    let environment = config.environment.as_ref()?;
    Some(EnvironmentPlan {
        provider: environment.provider.clone(),
        control_location: environment.control_location,
        ownership: environment.ownership,
        workspace: environment
            .workspace
            .as_ref()
            .map(|workspace| resolve_path(base_dir, workspace)),
        artifacts: environment
            .artifacts
            .as_ref()
            .or(config.runtime.artifacts.as_ref())
            .map(|artifacts| resolve_path(base_dir, artifacts)),
        env: environment.env.clone(),
        connection: environment.connection.clone(),
        metadata: environment.metadata.clone(),
        settings: environment.settings.clone(),
    })
}

fn resolve_capability_plan(
    config: &FabricConfig,
    base_dir: &Path,
    adapter_descriptor: Option<&ResolvedAdapterDescriptor>,
) -> CapabilityPlan {
    let accepts = |field: AdapterConfigField| {
        adapter_descriptor
            .map(|adapter| adapter.descriptor.config.accepts.contains(&field))
            .unwrap_or(false)
    };
    let skill_paths: Vec<PathBuf> = config
        .skills
        .as_ref()
        .map(|skills| {
            skills
                .paths
                .iter()
                .map(|path| resolve_path(base_dir, path))
                .collect()
        })
        .unwrap_or_default();
    let skills_are_native = !skill_paths.is_empty() && accepts(AdapterConfigField::Skills);
    let mcp_servers: BTreeMap<String, McpServerPlan> = config
        .mcp
        .as_ref()
        .map(|mcp| {
            mcp.servers
                .iter()
                .map(|(name, server)| {
                    (
                        name.clone(),
                        McpServerPlan {
                            transport: server.transport.clone(),
                            url: server.url.clone(),
                            exposure: server.exposure,
                        },
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    let blocked_tools = config
        .tools
        .as_ref()
        .map(|tools| tools.blocked.clone())
        .unwrap_or_default();
    let toolsets = config
        .tools
        .as_ref()
        .and_then(|tools| tools.toolsets.clone());
    let blocked_tools_configured = !blocked_tools.is_empty();
    let enabled_toolsets_configured = toolsets
        .as_ref()
        .is_some_and(|toolsets| toolsets.enabled.is_some());
    let blocked_toolsets_configured = toolsets
        .as_ref()
        .is_some_and(|toolsets| !toolsets.blocked.is_empty());
    let tools_configured =
        blocked_tools_configured || enabled_toolsets_configured || blocked_toolsets_configured;
    let mut native = CapabilityTargetPlan::default();
    let managed = CapabilityTargetPlan::default();
    let mut unsupported = CapabilityTargetPlan::default();
    let mut routes = Vec::new();

    if blocked_tools_configured {
        if accepts(AdapterConfigField::BlockedTools) {
            native.tools_configured = true;
            routes.push(CapabilityRoute {
                kind: CapabilityKind::Tools,
                name: "tools.blocked".to_string(),
                target: CapabilityTarget::HarnessNative,
                reason: "selected adapter explicitly supports the Fabric blocked-tools policy"
                    .to_string(),
            });
        } else {
            unsupported.tools_configured = true;
            routes.push(CapabilityRoute {
                kind: CapabilityKind::Tools,
                name: "tools.blocked".to_string(),
                target: CapabilityTarget::Unsupported,
                reason: "selected adapter does not explicitly declare blocked-tools policy support and Fabric-managed enforcement is not implemented".to_string(),
            });
        }
    }
    for (configured, support, field, description) in [
        (
            enabled_toolsets_configured,
            AdapterConfigField::EnabledToolsets,
            "tools.toolsets.enabled",
            "enabled-toolsets selection",
        ),
        (
            blocked_toolsets_configured,
            AdapterConfigField::BlockedToolsets,
            "tools.toolsets.blocked",
            "blocked-toolsets policy",
        ),
    ] {
        if !configured {
            continue;
        }
        if accepts(support) {
            native.tools_configured = true;
            routes.push(CapabilityRoute {
                kind: CapabilityKind::Tools,
                name: field.to_string(),
                target: CapabilityTarget::HarnessNative,
                reason: format!("selected adapter explicitly supports the Fabric {description}"),
            });
        } else {
            unsupported.tools_configured = true;
            routes.push(CapabilityRoute {
                kind: CapabilityKind::Tools,
                name: field.to_string(),
                target: CapabilityTarget::Unsupported,
                reason: format!(
                    "selected adapter does not explicitly declare {description} support"
                ),
            });
        }
    }

    if !skill_paths.is_empty() {
        if skills_are_native {
            native.skill_paths = skill_paths.clone();
            routes.push(CapabilityRoute {
                kind: CapabilityKind::Skills,
                name: "skills".to_string(),
                target: CapabilityTarget::HarnessNative,
                reason: "selected adapter accepts Fabric skills config".to_string(),
            });
        } else {
            unsupported.skill_paths = skill_paths.clone();
            routes.push(CapabilityRoute {
                kind: CapabilityKind::Skills,
                name: "skills".to_string(),
                target: CapabilityTarget::Unsupported,
                reason: "selected adapter does not declare native skills support and Fabric-managed skills are not implemented".to_string(),
            });
        }
    }

    for (name, server) in &mcp_servers {
        let can_map_native = accepts(AdapterConfigField::Mcp)
            && matches!(server.exposure, McpExposure::HarnessNative);
        if can_map_native {
            native.mcp_servers.insert(name.clone(), server.clone());
            routes.push(CapabilityRoute {
                kind: CapabilityKind::Mcp,
                name: name.clone(),
                target: CapabilityTarget::HarnessNative,
                reason: format!(
                    "MCP server uses {} exposure and adapter accepts mcp",
                    mcp_exposure_name(server.exposure)
                ),
            });
        } else {
            unsupported.mcp_servers.insert(name.clone(), server.clone());
            routes.push(CapabilityRoute {
                kind: CapabilityKind::Mcp,
                name: name.clone(),
                target: CapabilityTarget::Unsupported,
                reason: match server.exposure {
                    McpExposure::FabricManaged => {
                        "MCP server explicitly requests Fabric-managed exposure but Fabric-managed MCP is not implemented".to_string()
                    }
                    _ => "selected adapter does not declare native MCP support and Fabric-managed MCP is not implemented".to_string(),
                },
            });
        }
    }

    CapabilityPlan {
        tools: ToolsPlan {
            blocked: blocked_tools,
            toolsets,
        },
        tools_configured,
        skill_paths,
        mcp_servers,
        native,
        managed,
        unsupported,
        routes,
    }
}

fn mcp_exposure_name(exposure: McpExposure) -> &'static str {
    match exposure {
        McpExposure::HarnessNative => "harness_native",
        McpExposure::FabricManaged => "fabric_managed",
    }
}

fn resolve_telemetry_plan(
    config: &FabricConfig,
    adapter_descriptor: Option<&AdapterDescriptor>,
) -> Result<Option<TelemetryPlan>> {
    let Some(telemetry) = config.telemetry.as_ref() else {
        return Ok(None);
    };
    if telemetry.providers.is_empty() {
        return Ok(None);
    }
    let relay_provider = telemetry.providers.get(&TelemetryProvider::Relay);
    let native_provider = telemetry.providers.get(&TelemetryProvider::Native);
    let relay = config.relay.as_ref();
    let relay_enabled = relay_provider.is_some();
    let providers = [TelemetryProvider::Relay, TelemetryProvider::Native]
        .into_iter()
        .filter(|provider| telemetry.providers.contains_key(provider))
        .collect::<Vec<_>>();
    if let Some(descriptor) = adapter_descriptor {
        for provider in &providers {
            if !descriptor.telemetry.providers.contains_key(provider) {
                return Err(FabricError::AdapterDescriptorUnsupported {
                    adapter_id: descriptor.adapter_id.clone(),
                    field: "telemetry.providers",
                    value: provider.as_str().to_string(),
                });
            }
        }
    }
    let adapter_outputs = adapter_descriptor
        .map(|descriptor| {
            providers
                .iter()
                .filter_map(|provider| descriptor.telemetry.providers.get(provider))
                .flat_map(|support| support.outputs.iter().cloned())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect()
        })
        .unwrap_or_default();
    Ok(Some(TelemetryPlan {
        providers,
        relay_enabled,
        relay_project: relay_enabled
            .then(|| relay.and_then(|relay| relay.project.clone()))
            .flatten(),
        relay_output_dir: relay_enabled
            .then(|| relay.and_then(|relay| relay.output_dir.clone()))
            .flatten(),
        relay_config: relay_enabled
            .then(|| resolve_relay_plugin_config(relay))
            .flatten(),
        native_config: native_provider.and_then(|provider| provider.config.clone()),
        adapter_outputs,
    }))
}

fn resolve_relay_plugin_config(relay: Option<&RelayConfig>) -> Option<Value> {
    let relay = relay?;
    let mut components = Vec::new();

    if let Some(observability) = relay.observability.as_ref() {
        components.push(serde_json::json!({
            "kind": "observability",
            "enabled": true,
            "config": serde_json::to_value(observability).unwrap_or(Value::Null),
        }));
    }

    for component in &relay.components {
        components.push(serde_json::to_value(component).unwrap_or(Value::Null));
    }

    if !components.is_empty() || relay.policy.is_some() {
        let mut plugin_config = serde_json::json!({
            "version": 1,
            "components": components,
        });
        if let Some(policy) = relay.policy.as_ref()
            && let Some(object) = plugin_config.as_object_mut()
        {
            object.insert(
                "policy".to_string(),
                serde_json::to_value(policy).unwrap_or(Value::Null),
            );
        }
        return Some(plugin_config);
    }

    None
}

fn resolve_path(root: &Path, path: &Path) -> PathBuf {
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    normalize_path(resolved)
}

fn normalize_path(path: PathBuf) -> PathBuf {
    path.components()
        .filter(|component| !matches!(component, std::path::Component::CurDir))
        .collect()
}

/// Resolved Fabric run plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RunPlan {
    /// Stable agent name.
    pub agent_name: String,
    /// Base directory used to resolve relative Fabric paths.
    pub base_dir: PathBuf,
    /// Complete typed Fabric config.
    pub config: FabricConfig,
    /// Adapter descriptor resolved for this plan, when configured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_descriptor: Option<ResolvedAdapterDescriptor>,
    /// Selected install or availability strategy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<ResolutionStrategy>,
    /// Resolved environment plan.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment_plan: Option<EnvironmentPlan>,
    /// Resolved capability configuration.
    #[serde(default)]
    pub capability_plan: CapabilityPlan,
    /// Lifecycle behavior implemented by the selected runtime path.
    pub capabilities: RuntimeCapabilities,
    /// Resolved telemetry pass-through plan.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry_plan: Option<TelemetryPlan>,
}

/// Lifecycle behavior implemented by a resolved runtime path.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RuntimeCapabilities {
    /// Whether the selected runtime supports service lifecycle operations.
    #[serde(default)]
    pub service: bool,
    /// Whether invocations can emit progressive output.
    #[serde(default)]
    pub streaming: bool,
    /// Whether a running runtime can accept config updates.
    #[serde(default)]
    pub updates: bool,
    /// Whether an in-flight invocation can be cancelled.
    #[serde(default)]
    pub cancellation: bool,
    /// Additional adapter-specific capability metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

/// Resolved environment plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EnvironmentPlan {
    /// Environment provider.
    pub provider: String,
    /// Fabric control location.
    pub control_location: ControlLocation,
    /// Environment resource ownership.
    pub ownership: EnvironmentOwnership,
    /// Resolved workspace path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace: Option<PathBuf>,
    /// Resolved artifact path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifacts: Option<PathBuf>,
    /// Environment variables visible to the harness and its tools.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    /// Provider connection metadata.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub connection: serde_json::Map<String, Value>,
    /// Consumer-provided environment metadata.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub metadata: serde_json::Map<String, Value>,
    /// Provider-specific settings.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub settings: serde_json::Map<String, Value>,
}

/// Resolved capability configuration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CapabilityPlan {
    /// Normalized tool policy.
    #[serde(default)]
    pub tools: ToolsPlan,
    /// Whether tool configuration was provided.
    #[serde(default)]
    pub tools_configured: bool,
    /// Resolved skill paths.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skill_paths: Vec<PathBuf>,
    /// MCP server exposure plan.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub mcp_servers: BTreeMap<String, McpServerPlan>,
    /// Capabilities mapped into the harness-native surface.
    #[serde(default)]
    pub native: CapabilityTargetPlan,
    /// Capabilities that Fabric must expose or manage outside the native harness config.
    #[serde(default)]
    pub managed: CapabilityTargetPlan,
    /// Capabilities that are configured but not executable by this Fabric build.
    #[serde(default)]
    pub unsupported: CapabilityTargetPlan,
    /// Routing decisions made while planning the configured capabilities.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub routes: Vec<CapabilityRoute>,
}

/// Normalized tool policy for a run.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ToolsPlan {
    /// Adapter-native tool names to block.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocked: Vec<String>,
    /// Adapter-native toolset selection and blocking policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub toolsets: Option<ToolsetConfig>,
}

/// Capabilities routed to one target.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CapabilityTargetPlan {
    /// Whether tool configuration was provided for this target.
    #[serde(default)]
    pub tools_configured: bool,
    /// Resolved skill paths for this target.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skill_paths: Vec<PathBuf>,
    /// MCP servers for this target.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub mcp_servers: BTreeMap<String, McpServerPlan>,
}

/// One capability routing decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CapabilityRoute {
    /// Capability kind.
    pub kind: CapabilityKind,
    /// Capability name.
    pub name: String,
    /// Routing target.
    pub target: CapabilityTarget,
    /// Human-readable reason for the selected route.
    pub reason: String,
}

impl CapabilityRoute {
    pub(crate) fn config_field(&self) -> String {
        match self.kind {
            CapabilityKind::Mcp => format!("mcp.servers.{}", self.name),
            CapabilityKind::Skills => "skills".to_string(),
            CapabilityKind::Tools => self.name.clone(),
        }
    }
}

/// Capability kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityKind {
    /// Tool config.
    Tools,
    /// Skill paths.
    Skills,
    /// MCP server.
    Mcp,
}

/// Capability routing target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityTarget {
    /// Adapter maps the capability into harness-native config.
    HarnessNative,
    /// Fabric exposes or manages the capability around the harness.
    FabricManaged,
    /// Capability is configured but no executable surface exists.
    Unsupported,
}

/// Resolved MCP server exposure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct McpServerPlan {
    /// MCP transport.
    pub transport: String,
    /// MCP URL or command.
    pub url: String,
    /// Exposure strategy.
    pub exposure: McpExposure,
}

/// Resolved telemetry plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TelemetryPlan {
    /// Telemetry providers selected for this run.
    pub providers: Vec<TelemetryProvider>,
    /// Whether Relay is enabled.
    pub relay_enabled: bool,
    /// Relay project, when configured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relay_project: Option<String>,
    /// Relay output directory, when configured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relay_output_dir: Option<PathBuf>,
    /// Relay pass-through config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relay_config: Option<Value>,
    /// Native telemetry pass-through config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_config: Option<Value>,
    /// Telemetry outputs declared by the selected adapter descriptor.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub adapter_outputs: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typed_config(adapter_id: &str) -> FabricConfig {
        serde_json::from_value(serde_json::json!({
            "schema_version": "fabric.agent/v1alpha1",
            "metadata": {"name": "typed-agent"},
            "harness": {
                "adapter_id": adapter_id,
                "resolution": "preinstalled"
            },
            "runtime": {},
            "environment": {
                "provider": "local",
                "workspace": "workspace"
            },
            "skills": {"paths": ["skills/review"]}
        }))
        .expect("typed config")
    }

    fn repository_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    #[test]
    fn relay_observability_uses_v2_typed_atof_sinks() {
        let observability: RelayObservabilityConfig = serde_json::from_value(serde_json::json!({
            "atof": {
                "enabled": true,
                "sinks": [
                    {
                        "type": "file",
                        "output_directory": "artifacts/relay",
                        "filename": "events.atof.jsonl",
                        "mode": "overwrite"
                    },
                    {
                        "type": "stream",
                        "url": "http://localhost:4319/events",
                        "transport": "ndjson",
                        "header_env": {"authorization": "RELAY_AUTHORIZATION"},
                        "name": "live-events"
                    }
                ]
            }
        }))
        .expect("Relay v2 observability config");

        let value = serde_json::to_value(observability).expect("serialized observability");
        assert_eq!(value["version"], 2);
        assert_eq!(value["atof"]["sinks"][0]["type"], "file");
        assert_eq!(value["atof"]["sinks"][1]["type"], "stream");
    }

    #[test]
    fn resolves_complete_typed_config_with_explicit_base_dir() {
        let base_dir = repository_root();
        let plan = resolve_run_plan_from_config(
            typed_config("nvidia.fabric.hermes"),
            ResolveContext::new(&base_dir),
        )
        .expect("typed plan");
        let base_dir = std::path::absolute(base_dir).expect("absolute base directory");

        assert_eq!(plan.agent_name, "typed-agent");
        assert_eq!(plan.base_dir, base_dir);
        assert_eq!(
            plan.capability_plan.skill_paths,
            vec![plan.base_dir.join("skills/review")]
        );
        assert_eq!(
            plan.adapter_descriptor
                .as_ref()
                .map(|adapter| adapter.descriptor.adapter_id.as_str()),
            Some("nvidia.fabric.hermes")
        );
    }

    #[test]
    fn relative_base_dir_becomes_absolute_before_deriving_paths() {
        let plan = resolve_run_plan_from_config(
            typed_config("nvidia.fabric.hermes"),
            ResolveContext::new("."),
        )
        .expect("typed plan");
        let expected_base = std::env::current_dir().expect("current directory");

        assert_eq!(plan.base_dir, expected_base);
        assert_eq!(
            plan.environment_plan
                .as_ref()
                .and_then(|environment| environment.workspace.as_ref()),
            Some(&expected_base.join("workspace"))
        );
        assert_eq!(
            plan.capability_plan.skill_paths,
            vec![expected_base.join("skills/review")]
        );
    }

    #[test]
    fn typed_config_round_trip_has_no_file_provenance() {
        let plan = resolve_run_plan_from_config(
            typed_config("nvidia.fabric.hermes"),
            ResolveContext::new("/tmp/fabric-base"),
        )
        .expect("run plan");
        let value = serde_json::to_value(plan).expect("run plan JSON");

        assert_eq!(value["base_dir"], "/tmp/fabric-base");
        assert_eq!(value["config"]["metadata"]["name"], "typed-agent");
        assert!(value.get("effective_config").is_none());
    }

    #[test]
    fn normalized_fields_survive_planning() {
        let mut config = typed_config("nvidia.fabric.hermes");
        config.system_prompt = Some("Be concise.".to_string());
        config.max_turns = Some(7);
        config.runtime.timeout_seconds = Some(12.5);
        config.environment.as_mut().expect("environment").env =
            BTreeMap::from([("VISIBLE".to_string(), "yes".to_string())]);
        config.models.insert(
            "default".to_string(),
            ModelConfig {
                provider: "nvidia".to_string(),
                model: "nvidia/test".to_string(),
                temperature: Some(0.2),
                api_key_env: Some("NVIDIA_API_KEY".to_string()),
                base_url: Some("https://models.example/v1".to_string()),
                settings: serde_json::Map::new(),
                extensions: BTreeMap::new(),
            },
        );
        config.tools = Some(ToolsConfig {
            blocked: Vec::new(),
            toolsets: Some(ToolsetConfig {
                enabled: Some(vec!["terminal".to_string()]),
                blocked: vec!["browser".to_string()],
                extensions: BTreeMap::new(),
            }),
            extensions: BTreeMap::new(),
        });

        let plan =
            resolve_run_plan_from_config(config, ResolveContext::new("/tmp/fabric-normalized"))
                .expect("normalized plan");

        assert_eq!(plan.config.system_prompt.as_deref(), Some("Be concise."));
        assert_eq!(plan.config.max_turns, Some(7));
        assert_eq!(plan.config.runtime.timeout_seconds, Some(12.5));
        assert_eq!(
            plan.environment_plan
                .as_ref()
                .and_then(|environment| environment.env.get("VISIBLE")),
            Some(&"yes".to_string())
        );
        assert_eq!(
            plan.capability_plan
                .tools
                .toolsets
                .as_ref()
                .and_then(|toolsets| toolsets.enabled.as_ref()),
            Some(&vec!["terminal".to_string()])
        );
        assert!(plan.capability_plan.routes.iter().any(|route| {
            route.name == "tools.toolsets.enabled"
                && route.target == CapabilityTarget::HarnessNative
        }));
    }

    #[test]
    fn unsupported_normalized_scalar_reports_adapter_config_incompatibility() {
        for adapter_id in ["nvidia.fabric.codex", "nvidia.fabric.langchain.deepagents"] {
            let mut config = typed_config(adapter_id);
            config.max_turns = Some(3);

            let error = resolve_run_plan_from_config(
                config,
                ResolveContext::new("/tmp/fabric-incompatible"),
            )
            .expect_err("adapter does not advertise max_turns");

            assert!(matches!(
                error,
                FabricError::AdapterCompatibility {
                    adapter_id: actual,
                    field,
                    ..
                } if actual == adapter_id && field == "max_turns"
            ));
        }
    }

    #[test]
    fn unsupported_model_temperature_reports_canonical_field() {
        for (adapter_id, provider) in [
            ("nvidia.fabric.claude", "anthropic"),
            ("nvidia.fabric.codex", "openai"),
        ] {
            let mut config = typed_config(adapter_id);
            config.models.insert(
                "review".to_string(),
                ModelConfig {
                    provider: provider.to_string(),
                    model: "test-model".to_string(),
                    temperature: Some(0.2),
                    api_key_env: None,
                    base_url: None,
                    settings: serde_json::Map::new(),
                    extensions: BTreeMap::new(),
                },
            );

            let error = resolve_run_plan_from_config(
                config,
                ResolveContext::new("/tmp/fabric-temperature"),
            )
            .expect_err("adapter does not advertise model temperature");

            assert!(matches!(
                error,
                FabricError::AdapterCompatibility {
                    adapter_id: actual,
                    field,
                    ..
                } if actual == adapter_id && field == "models.review.temperature"
            ));
        }
    }

    #[test]
    fn unsupported_toolsets_report_canonical_field() {
        for adapter_id in [
            "nvidia.fabric.claude",
            "nvidia.fabric.codex",
            "nvidia.fabric.langchain.deepagents",
        ] {
            let mut config = typed_config(adapter_id);
            config.tools = Some(ToolsConfig {
                blocked: Vec::new(),
                toolsets: Some(ToolsetConfig {
                    enabled: Some(vec!["terminal".to_string()]),
                    blocked: Vec::new(),
                    extensions: BTreeMap::new(),
                }),
                extensions: BTreeMap::new(),
            });

            let error =
                resolve_run_plan_from_config(config, ResolveContext::new("/tmp/fabric-toolsets"))
                    .expect_err("adapter does not advertise enabled toolsets");

            assert!(matches!(
                error,
                FabricError::AdapterCompatibility {
                    adapter_id: actual,
                    field,
                    ..
                } if actual == adapter_id && field == "tools.toolsets.enabled"
            ));
        }
    }

    #[test]
    fn model_settings_rejects_normalized_base_url_alias() {
        let mut config = typed_config("nvidia.fabric.hermes");
        config.models.insert(
            "default".to_string(),
            ModelConfig {
                provider: "nvidia".to_string(),
                model: "nvidia/test".to_string(),
                temperature: None,
                api_key_env: None,
                base_url: None,
                settings: serde_json::Map::from_iter([(
                    "base_url".to_string(),
                    Value::String("https://legacy.example/v1".to_string()),
                )]),
                extensions: BTreeMap::new(),
            },
        );

        let error =
            resolve_run_plan_from_config(config, ResolveContext::new("/tmp/fabric-model-alias"))
                .expect_err("normalized aliases must be rejected");

        assert!(matches!(
            error,
            FabricError::InvalidConfig { field, reason }
                if field == "models.default.settings.base_url"
                    && reason.contains("models.<role>.base_url")
        ));
    }

    #[test]
    fn sole_named_model_is_selected_without_forcing_default_role() {
        let mut config = typed_config("nvidia.fabric.claude");
        config.models.insert(
            "review".to_string(),
            ModelConfig {
                provider: "anthropic".to_string(),
                model: "claude-test".to_string(),
                temperature: None,
                api_key_env: None,
                base_url: None,
                settings: serde_json::Map::new(),
                extensions: BTreeMap::new(),
            },
        );

        resolve_run_plan_from_config(config, ResolveContext::new("/tmp/fabric-model-role"))
            .expect("sole named model");
    }

    #[test]
    fn multiple_models_require_an_explicit_default_role() {
        let mut config = typed_config("nvidia.fabric.claude");
        for role in ["fast", "slow"] {
            config.models.insert(
                role.to_string(),
                ModelConfig {
                    provider: "anthropic".to_string(),
                    model: format!("claude-{role}"),
                    temperature: None,
                    api_key_env: None,
                    base_url: None,
                    settings: serde_json::Map::new(),
                    extensions: BTreeMap::new(),
                },
            );
        }

        let error =
            resolve_run_plan_from_config(config, ResolveContext::new("/tmp/fabric-model-role"))
                .expect_err("ambiguous model roles");

        assert!(matches!(
            error,
            FabricError::AdapterCompatibility { field, reason, .. }
                if field == "models" && reason.contains("no default role")
        ));
    }

    #[test]
    fn tool_and_toolset_policies_are_routed_independently() {
        let mut config = typed_config("nvidia.fabric.hermes");
        config.tools = Some(ToolsConfig {
            blocked: Vec::new(),
            toolsets: Some(ToolsetConfig {
                enabled: Some(Vec::new()),
                blocked: vec!["browser".to_string()],
                extensions: BTreeMap::new(),
            }),
            extensions: BTreeMap::new(),
        });

        let plan = resolve_run_plan_from_config(config, ResolveContext::new("/tmp/fabric-tools"))
            .expect("tool capability plan");

        assert!(plan.capability_plan.routes.iter().any(|route| {
            route.name == "tools.toolsets.enabled"
                && route.target == CapabilityTarget::HarnessNative
        }));
        assert!(plan.capability_plan.routes.iter().any(|route| {
            route.name == "tools.toolsets.blocked"
                && route.target == CapabilityTarget::HarnessNative
        }));
    }

    #[test]
    fn unsupported_tool_policy_fails_during_planning() {
        let mut config = typed_config("nvidia.fabric.hermes");
        config.tools = Some(ToolsConfig {
            blocked: vec!["Bash".to_string()],
            toolsets: None,
            extensions: BTreeMap::new(),
        });

        let error = resolve_run_plan_from_config(
            config,
            ResolveContext::new("/tmp/fabric-unsupported-tools"),
        )
        .expect_err("Hermes does not support per-tool blocking");

        assert!(matches!(
            error,
            FabricError::AdapterCompatibility {
                adapter_id,
                field,
                ..
            } if adapter_id == "nvidia.fabric.hermes" && field == "tools.blocked"
        ));
    }

    #[test]
    fn unsupported_mcp_reports_canonical_config_path() {
        let mut config = typed_config("nvidia.fabric.claude");
        config.mcp = Some(McpConfig {
            servers: BTreeMap::from([(
                "docs".to_string(),
                McpServerConfig {
                    transport: "streamable-http".to_string(),
                    url: "https://mcp.example".to_string(),
                    exposure: McpExposure::FabricManaged,
                    extensions: BTreeMap::new(),
                },
            )]),
            extensions: BTreeMap::new(),
        });

        let error = resolve_run_plan_from_config(
            config,
            ResolveContext::new("/tmp/fabric-unsupported-mcp"),
        )
        .expect_err("Fabric-managed MCP is not implemented");

        assert!(matches!(
            error,
            FabricError::AdapterCompatibility {
                adapter_id,
                field,
                ..
            } if adapter_id == "nvidia.fabric.claude" && field == "mcp.servers.docs"
        ));
    }

    #[test]
    fn overlapping_toolset_policy_is_invalid() {
        let mut config = typed_config("nvidia.fabric.hermes");
        config.tools = Some(ToolsConfig {
            blocked: Vec::new(),
            toolsets: Some(ToolsetConfig {
                enabled: Some(vec!["browser".to_string()]),
                blocked: vec!["browser".to_string()],
                extensions: BTreeMap::new(),
            }),
            extensions: BTreeMap::new(),
        });

        let error =
            resolve_run_plan_from_config(config, ResolveContext::new("/tmp/fabric-invalid-tools"))
                .expect_err("overlapping toolset policy");

        assert!(matches!(
            error,
            FabricError::InvalidConfig { field, .. } if field == "tools.toolsets"
        ));
    }

    #[test]
    fn loads_and_validates_json_adapter_descriptor() {
        let descriptor =
            load_adapter_descriptor(repository_root().join("adapters/hermes/fabric-adapter.json"))
                .expect("adapter descriptor");

        assert_eq!(descriptor.contract_version, ADAPTER_CONTRACT_VERSION);
        assert_eq!(descriptor.adapter_kind, AdapterKind::Python);
    }

    #[test]
    fn resolves_additional_adapter_directory_before_agent_local_override() {
        struct RemoveDirOnDrop(PathBuf);

        impl Drop for RemoveDirOnDrop {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        let root = std::env::temp_dir().join(format!(
            "nemo-fabric-adapter-discovery-{}",
            std::process::id()
        ));
        let _cleanup = RemoveDirOnDrop(root.clone());
        let installed_directory = root.join("installed");
        let base_dir = root.join("agent");
        let installed_descriptor = installed_directory.join("stopgap/fabric-adapter.json");
        let local_descriptor = base_dir.join("adapters/stopgap/fabric-adapter.json");
        let descriptor = |module: &str| {
            serde_json::json!({
                "contract_version": ADAPTER_CONTRACT_VERSION,
                "adapter_id": "test.fabric.installed",
                "harness": "installed-test",
                "adapter_kind": "python",
                "runner": {"module": module},
                "config": {"accepts": ["skills"]}
            })
        };

        std::fs::create_dir_all(installed_descriptor.parent().expect("installed parent"))
            .expect("create installed adapter directory");
        std::fs::write(
            &installed_descriptor,
            serde_json::to_vec_pretty(&descriptor("installed.adapter")).expect("descriptor JSON"),
        )
        .expect("write installed descriptor");

        let plan = resolve_run_plan_from_config_with_adapter_directories(
            typed_config("test.fabric.installed"),
            ResolveContext::new(&base_dir),
            std::slice::from_ref(&installed_directory),
        )
        .expect("installed adapter plan");
        let expected_installed_descriptor = installed_descriptor
            .canonicalize()
            .expect("canonical installed descriptor");
        assert_eq!(
            plan.adapter_descriptor
                .as_ref()
                .map(|adapter| adapter.path.as_path()),
            Some(expected_installed_descriptor.as_path())
        );

        std::fs::create_dir_all(local_descriptor.parent().expect("local parent"))
            .expect("create local adapter directory");
        std::fs::write(
            &local_descriptor,
            serde_json::to_vec_pretty(&descriptor("local.adapter")).expect("descriptor JSON"),
        )
        .expect("write local descriptor");

        let plan = resolve_run_plan_from_config_with_adapter_directories(
            typed_config("test.fabric.installed"),
            ResolveContext::new(&base_dir),
            &[installed_directory],
        )
        .expect("agent-local adapter plan");
        let resolved = plan.adapter_descriptor.expect("resolved adapter");
        assert_eq!(
            resolved.path,
            local_descriptor
                .canonicalize()
                .expect("canonical local descriptor")
        );
        assert_eq!(
            resolved.descriptor.runner["module"],
            serde_json::json!("local.adapter")
        );
    }
}
