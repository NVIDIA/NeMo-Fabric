// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fabric config models and loading helpers.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{FabricError, Result};

const AGENT_YAML: &str = "agent.yaml";
/// Adapter descriptor contract version supported by this core.
pub const ADAPTER_CONTRACT_VERSION: &str = "fabric.adapter/v1alpha1";

/// A loaded Fabric document with resolved source path and agent root.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FabricDocument {
    /// Unified Fabric agent config.
    FabricConfig {
        /// Path to the config file.
        path: PathBuf,
        /// Root used for resolving relative paths.
        root: PathBuf,
        /// Parsed config.
        config: FabricConfig,
    },
}

/// Versioned Fabric agent config.
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
    /// Runtime input/output contract.
    pub runtime: RuntimeConfig,
    /// Environment where the harness or its tools execute.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment: Option<EnvironmentConfig>,
    /// Tool capability configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Value>,
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
    /// Optional profile discovery config.
    #[serde(default, skip_serializing_if = "ProfileRegistryConfig::is_empty")]
    pub profiles: ProfileRegistryConfig,
    /// Additive fields not yet recognized by this core version.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

/// Profile discovery config for curated package profiles.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ProfileRegistryConfig {
    /// Directories searched when a caller selects a profile by name.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub directories: Vec<PathBuf>,
    /// Additive profile-discovery fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

impl ProfileRegistryConfig {
    fn is_empty(&self) -> bool {
        self.directories.is_empty() && self.extensions.is_empty()
    }
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
    fn from_config(_config: &FabricConfig, config_root: &Path) -> Result<Self> {
        let mut registry = Self::default();
        registry.register_repository_directory(&repository_adapter_dir())?;
        registry.register_local_directory(&config_root.join("adapters"))?;
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
    /// Fabric config areas accepted by this adapter.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub accepts: Vec<String>,
    /// Harness-native files generated by this adapter.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub generates: Vec<PathBuf>,
    /// Additive adapter config-support fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
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

/// Profile config applied on top of a Fabric config.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ProfileConfig {
    /// Optional profile schema version.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema_version: Option<String>,
    /// Optional profile name used for directory discovery.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Optional profile description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Raw config fields recursively merged over the base config.
    #[serde(default, flatten)]
    pub overlay: BTreeMap<String, Value>,
}

#[derive(JsonSchema)]
#[allow(dead_code)]
struct ProfileConfigSchema {
    /// Optional profile schema version.
    schema_version: Option<String>,
    /// Optional profile name used for directory discovery.
    name: Option<String>,
    /// Optional profile description.
    description: Option<String>,
    /// Partial harness overlay.
    harness: Option<BTreeMap<String, Value>>,
    /// Partial model overlays by alias.
    models: Option<BTreeMap<String, Value>>,
    /// Partial runtime overlay.
    runtime: Option<BTreeMap<String, Value>>,
    /// Partial environment overlay.
    environment: Option<BTreeMap<String, Value>>,
    /// Tool capability overlay.
    tools: Option<Value>,
    /// Partial skill overlay.
    skills: Option<BTreeMap<String, Value>>,
    /// Partial MCP overlay.
    mcp: Option<BTreeMap<String, Value>>,
    /// Partial telemetry overlay.
    telemetry: Option<BTreeMap<String, Value>>,
    /// Partial Relay integration overlay.
    relay: Option<BTreeMap<String, Value>>,
    /// Additive config overlays.
    #[serde(flatten)]
    extensions: BTreeMap<String, Value>,
}

impl JsonSchema for ProfileConfig {
    fn schema_name() -> Cow<'static, str> {
        "ProfileConfig".into()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        ProfileConfigSchema::json_schema(generator)
    }
}

/// Source context used when resolving an in-memory Fabric config.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolveContext {
    /// Root used to resolve agent package paths.
    pub agent_root: PathBuf,
    /// Path recorded as config provenance in the run plan.
    pub config_path: PathBuf,
    /// Root used to resolve config-local paths.
    pub config_root: PathBuf,
}

impl ResolveContext {
    /// Build a context for an agent package root.
    pub fn from_agent_root(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self {
            agent_root: root.clone(),
            config_path: root.join(AGENT_YAML),
            config_root: root,
        }
    }

    /// Build a context for a config file and its config root.
    pub fn from_config_path(path: impl Into<PathBuf>, root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self {
            agent_root: root.clone(),
            config_path: path.into(),
            config_root: root,
        }
    }
}

/// Adapter implementation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AdapterKind {
    /// Launch and supervise a CLI process.
    Process,
    /// Connect to a service or HTTP-backed harness.
    Http,
    /// Call a Python SDK/plugin adapter.
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
    /// Skill paths relative to the agent root.
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
    /// Directory used for ATOF files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_directory: Option<PathBuf>,
    /// ATOF file name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    /// File write mode.
    #[serde(default)]
    pub mode: RelayAtofMode,
    /// Optional remote ATOF endpoints.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub endpoints: Vec<RelayAtofEndpointConfig>,
    /// Additive ATOF fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
}

/// Relay ATOF endpoint configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RelayAtofEndpointConfig {
    /// Endpoint URL.
    pub url: String,
    /// Endpoint transport.
    #[serde(default)]
    pub transport: RelayAtofEndpointTransport,
    /// Endpoint headers.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    /// Request timeout in milliseconds.
    #[serde(default = "default_relay_timeout_millis")]
    pub timeout_millis: u64,
    /// Field-name handling policy.
    #[serde(default)]
    pub field_name_policy: RelayAtofEndpointFieldNamePolicy,
    /// Additive endpoint fields.
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, Value>,
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

/// Relay ATOF endpoint transport.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RelayAtofEndpointTransport {
    /// HTTP POST transport.
    #[default]
    HttpPost,
    /// WebSocket transport.
    Websocket,
    /// NDJSON transport.
    Ndjson,
}

/// Relay ATOF endpoint field-name policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RelayAtofEndpointFieldNamePolicy {
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
    1
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

/// Load a Fabric document from an agent directory or single agent config.
pub fn load_fabric_document(path: impl AsRef<Path>) -> Result<FabricDocument> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(FabricError::PathNotFound(path.to_path_buf()));
    }
    if path.is_dir() {
        return load_directory(path);
    }
    load_file(path)
}

/// Validate an agent directory or config, including discoverable profile YAMLs.
pub fn validate_agent_directory(path: impl AsRef<Path>) -> Result<()> {
    match load_fabric_document(path)? {
        FabricDocument::FabricConfig { root, config, .. } => {
            discover_profiles(&config, &root)?;
        }
    }

    Ok(())
}

/// Load an adapter descriptor from JSON package metadata.
pub fn load_adapter_descriptor(path: impl AsRef<Path>) -> Result<AdapterDescriptor> {
    let path = path.as_ref();
    let descriptor = read_json(path)?;
    validate_adapter_descriptor_shape(&descriptor, path)?;
    Ok(descriptor)
}

/// Resolve an agent directory or single agent config into merged effective config.
pub fn resolve_effective_config(
    path: impl AsRef<Path>,
    profile: Option<&str>,
) -> Result<EffectiveConfig> {
    let profiles: Vec<String> = profile.into_iter().map(str::to_string).collect();
    resolve_effective_config_with_profiles(path, &profiles)
}

/// Resolve an agent directory or single agent config with ordered profiles into
/// merged effective config.
pub fn resolve_effective_config_with_profiles(
    path: impl AsRef<Path>,
    profiles: &[String],
) -> Result<EffectiveConfig> {
    match load_fabric_document(path)? {
        FabricDocument::FabricConfig { path, root, config } => {
            let agent_name = config.metadata.name.clone();
            let (profile_configs, selected_profiles) =
                load_profile_configs(&agent_name, &config, &root, profiles)?;
            resolve_effective_config_from_config_with_profile_names(
                config,
                &profile_configs,
                selected_profiles,
                ResolveContext::from_config_path(path, root),
            )
        }
    }
}

/// Resolve typed config/profile overlays into merged effective config.
pub fn resolve_effective_config_from_config(
    config: FabricConfig,
    profiles: &[ProfileConfig],
    context: ResolveContext,
) -> Result<EffectiveConfig> {
    let selected_profiles = profiles
        .iter()
        .enumerate()
        .map(|(index, profile)| {
            profile
                .name
                .clone()
                .unwrap_or_else(|| format!("profile_{}", index + 1))
        })
        .collect();
    resolve_effective_config_from_config_with_profile_names(
        config,
        profiles,
        selected_profiles,
        context,
    )
}

fn load_directory(path: &Path) -> Result<FabricDocument> {
    let agent = path.join(AGENT_YAML);
    if agent.exists() {
        return load_fabric_config(&agent);
    }
    Err(FabricError::MissingEntrypoint(path.to_path_buf()))
}

fn load_file(path: &Path) -> Result<FabricDocument> {
    match path.file_name().and_then(|name| name.to_str()) {
        Some(AGENT_YAML) => load_fabric_config(path),
        Some(name) if name.ends_with(".yaml") || name.ends_with(".yml") => load_fabric_config(path),
        _ => Err(FabricError::UnsupportedExtension(path.to_path_buf())),
    }
}

/// Resolve an agent directory or single agent config into a runnable plan.
pub fn resolve_run_plan(path: impl AsRef<Path>, profile: Option<&str>) -> Result<RunPlan> {
    let profiles: Vec<String> = profile.into_iter().map(str::to_string).collect();
    resolve_run_plan_with_profiles(path, &profiles)
}

/// Resolve an agent directory or single agent config with ordered profile application.
pub fn resolve_run_plan_with_profiles(
    path: impl AsRef<Path>,
    profiles: &[String],
) -> Result<RunPlan> {
    resolve_run_plan_from_effective_config(resolve_effective_config_with_profiles(path, profiles)?)
}

/// Resolve a typed Fabric config and typed profile overlays into a runnable plan.
///
/// This is the SDK-facing entrypoint. File and YAML loading stays outside this
/// function; callers provide already-typed configs and the path context used for
/// resolving relative paths.
pub fn resolve_run_plan_from_config(
    config: FabricConfig,
    profiles: &[ProfileConfig],
    context: ResolveContext,
) -> Result<RunPlan> {
    resolve_run_plan_from_effective_config(resolve_effective_config_from_config(
        config, profiles, context,
    )?)
}

fn load_fabric_config(path: &Path) -> Result<FabricDocument> {
    let config = read_yaml::<FabricConfig>(path)?;
    let root = parent_or_current(path);
    Ok(FabricDocument::FabricConfig {
        path: path.to_path_buf(),
        root,
        config,
    })
}

fn read_yaml<T>(path: &Path) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let raw = std::fs::read_to_string(path).map_err(|source| FabricError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    serde_yaml::from_str(&raw).map_err(|source| FabricError::ParseYaml {
        path: path.to_path_buf(),
        source,
    })
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

fn parent_or_current(path: &Path) -> PathBuf {
    path.parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn load_profile_configs(
    agent_name: &str,
    config: &FabricConfig,
    config_root: &Path,
    profiles: &[String],
) -> Result<(Vec<ProfileConfig>, Vec<String>)> {
    if profiles.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let discovered = discover_profiles(&config, config_root)?;
    let mut profile_configs = Vec::new();
    let mut selected_profiles = Vec::new();
    for profile_name in profiles {
        let profile_path =
            if let Some(path) = resolve_profile_file_reference(config_root, &profile_name) {
                path
            } else if let Some(path) = discovered.get(profile_name.as_str()) {
                path.clone()
            } else {
                return Err(FabricError::UnknownProfile {
                    profile: profile_name.clone(),
                    agent: agent_name.to_string(),
                    available: discovered.keys().cloned().collect(),
                });
            };
        let profile_config = read_yaml::<ProfileConfig>(&profile_path)?;
        profile_configs.push(profile_config);
        selected_profiles.push(profile_name.clone());
    }
    Ok((profile_configs, selected_profiles))
}

fn resolve_effective_config_from_config_with_profile_names(
    config: FabricConfig,
    profiles: &[ProfileConfig],
    selected_profiles: Vec<String>,
    context: ResolveContext,
) -> Result<EffectiveConfig> {
    let mut effective = config;
    for profile in profiles {
        apply_profile_config(&mut effective, profile)?;
    }
    let config = into_effective_config(effective);
    Ok(EffectiveConfig {
        agent_name: config.metadata.name.clone(),
        profiles: selected_profiles,
        agent_root: context.agent_root,
        config_path: context.config_path,
        config_root: context.config_root,
        config,
    })
}

/// Resolve execution planning metadata from merged effective config.
pub fn resolve_run_plan_from_effective_config(
    effective_config: EffectiveConfig,
) -> Result<RunPlan> {
    let config = effective_config.config.clone();
    let config_root = effective_config.config_root.clone();
    let adapter_descriptor = resolve_adapter_descriptor(&config, &config_root)?;
    let descriptor = adapter_descriptor
        .as_ref()
        .map(|adapter| &adapter.descriptor);
    let resolution = resolve_resolution(&config, descriptor)?;
    let environment_plan = resolve_environment_plan(&config, &config_root);
    validate_control_location(descriptor, environment_plan.as_ref())?;
    let capability_plan =
        resolve_capability_plan(&config, &config_root, adapter_descriptor.as_ref());
    let capabilities = resolve_runtime_capabilities(&config, descriptor);
    let telemetry_plan = resolve_telemetry_plan(&config, descriptor)?;
    Ok(RunPlan {
        agent_name: effective_config.agent_name.clone(),
        profiles: effective_config.profiles.clone(),
        adapter_descriptor,
        resolution,
        environment_plan,
        capability_plan,
        capabilities,
        telemetry_plan,
        agent_root: effective_config.agent_root.clone(),
        config_path: effective_config.config_path.clone(),
        config_root,
        config,
        effective_config,
    })
}

fn into_effective_config(mut config: FabricConfig) -> FabricConfig {
    config.profiles = ProfileRegistryConfig::default();
    config
}

fn resolve_profile_file_reference(config_root: &Path, profile: &str) -> Option<PathBuf> {
    let path = Path::new(profile);
    let cwd_relative = normalize_path(path.to_path_buf());
    if cwd_relative.is_file() {
        return Some(cwd_relative);
    }
    let config_relative = resolve_path(config_root, path);
    if config_relative.is_file() {
        return Some(config_relative);
    }
    None
}

fn discover_profiles(
    config: &FabricConfig,
    config_root: &Path,
) -> Result<BTreeMap<String, PathBuf>> {
    let mut profiles: BTreeMap<String, PathBuf> = BTreeMap::new();
    for directory in &config.profiles.directories {
        let directory = resolve_path(config_root, directory);
        if !directory.exists() {
            println!(
                "warning: profile directory does not exist: {}",
                directory.display()
            );
            continue;
        }
        let entries = std::fs::read_dir(&directory).map_err(|source| FabricError::Read {
            path: directory.clone(),
            source,
        })?;
        for entry in entries {
            let path = entry
                .map_err(|source| FabricError::Read {
                    path: directory.clone(),
                    source,
                })?
                .path();
            if !is_yaml_file(&path) {
                continue;
            }
            let profile_config = read_yaml::<ProfileConfig>(&path)?;
            let Some(name) = profile_config.name.clone().or_else(|| {
                path.file_stem()
                    .and_then(|stem| stem.to_str())
                    .map(str::to_string)
            }) else {
                continue;
            };
            let profile_path = normalize_path(path);
            if profiles.contains_key(&name) {
                return Err(FabricError::ProfileError {
                    path: profile_path,
                    message: format!("duplicate profile `{name}`"),
                });
            }
            profiles.insert(name, profile_path);
        }
    }
    Ok(profiles)
}

fn is_yaml_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| matches!(extension, "yaml" | "yml"))
}

fn apply_profile_config(config: &mut FabricConfig, profile: &ProfileConfig) -> Result<()> {
    let mut merged = serde_json::to_value(&*config).map_err(FabricError::SerializeJson)?;
    let overlay = Value::Object(profile.overlay.clone().into_iter().collect());
    merge_json(&mut merged, overlay);
    *config = serde_json::from_value(merged).map_err(FabricError::SerializeJson)?;
    Ok(())
}

fn merge_json(base: &mut Value, overlay: Value) {
    match (base, overlay) {
        (Value::Object(base), Value::Object(overlay)) => {
            for (key, value) in overlay {
                if let Some(current) = base.get_mut(&key) {
                    merge_json(current, value);
                } else {
                    base.insert(key, value);
                }
            }
        }
        (base, overlay) => *base = overlay,
    }
}

fn resolve_adapter_descriptor(
    config: &FabricConfig,
    config_root: &Path,
) -> Result<Option<ResolvedAdapterDescriptor>> {
    let adapter_id = &config.harness.adapter_id;
    let registry = AdapterRegistry::from_config(config, config_root)?;
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

fn resolve_environment_plan(config: &FabricConfig, config_root: &Path) -> Option<EnvironmentPlan> {
    let environment = config.environment.as_ref()?;
    Some(EnvironmentPlan {
        provider: environment.provider.clone(),
        control_location: environment.control_location,
        ownership: environment.ownership,
        workspace: environment
            .workspace
            .as_ref()
            .map(|workspace| resolve_path(config_root, workspace)),
        artifacts: environment
            .artifacts
            .as_ref()
            .or(config.runtime.artifacts.as_ref())
            .map(|artifacts| resolve_path(config_root, artifacts)),
        connection: environment.connection.clone(),
        metadata: environment.metadata.clone(),
        settings: environment.settings.clone(),
    })
}

fn resolve_capability_plan(
    config: &FabricConfig,
    config_root: &Path,
    adapter_descriptor: Option<&ResolvedAdapterDescriptor>,
) -> CapabilityPlan {
    let accepts = |area: &str| {
        adapter_descriptor
            .map(|adapter| {
                adapter
                    .descriptor
                    .config
                    .accepts
                    .iter()
                    .any(|accepted| accepted == area)
            })
            .unwrap_or(false)
    };
    let skill_paths: Vec<PathBuf> = config
        .skills
        .as_ref()
        .map(|skills| {
            skills
                .paths
                .iter()
                .map(|path| resolve_path(config_root, path))
                .collect()
        })
        .unwrap_or_default();
    let skills_are_native = !skill_paths.is_empty() && accepts("skills");
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
    let tools_are_native = config.tools.is_some() && accepts("tools");
    let mut native = CapabilityTargetPlan::default();
    let managed = CapabilityTargetPlan::default();
    let mut unsupported = CapabilityTargetPlan::default();
    let mut routes = Vec::new();

    if config.tools.is_some() {
        if tools_are_native {
            native.tools_configured = true;
            routes.push(CapabilityRoute {
                kind: CapabilityKind::Tools,
                name: "tools".to_string(),
                target: CapabilityTarget::HarnessNative,
                reason: "selected adapter accepts Fabric tools config".to_string(),
            });
        } else {
            unsupported.tools_configured = true;
            routes.push(CapabilityRoute {
                kind: CapabilityKind::Tools,
                name: "tools".to_string(),
                target: CapabilityTarget::Unsupported,
                reason: "selected adapter does not declare native tools support and Fabric-managed tools are not implemented".to_string(),
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
        let can_map_native =
            accepts("mcp") && matches!(server.exposure, McpExposure::HarnessNative);
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
        tools_configured: config.tools.is_some(),
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

/// Merged Fabric config after applying selected profiles.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EffectiveConfig {
    /// Stable agent name.
    pub agent_name: String,
    /// Ordered selected profiles.
    pub profiles: Vec<String>,
    /// Root used to resolve agent package paths.
    pub agent_root: PathBuf,
    /// Resolved Fabric config path.
    pub config_path: PathBuf,
    /// Root used to resolve config-local paths.
    pub config_root: PathBuf,
    /// Merged Fabric config with authoring-time profile discovery removed.
    pub config: FabricConfig,
}

/// Resolved Fabric run plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RunPlan {
    /// Merged config and provenance used as the base for this plan.
    pub effective_config: EffectiveConfig,
    /// Stable agent name.
    pub agent_name: String,
    /// Ordered selected profiles.
    pub profiles: Vec<String>,
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
    /// Root used to resolve agent package paths.
    pub agent_root: PathBuf,
    /// Resolved Fabric config path.
    pub config_path: PathBuf,
    /// Root used to resolve config-local paths.
    pub config_root: PathBuf,
    /// Resolved Fabric profile config.
    pub config: FabricConfig,
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
    /// Routing decisions made while resolving the effective config.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub routes: Vec<CapabilityRoute>,
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

    fn file_config_agent_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/file-config-agent")
    }

    fn example_adapter_descriptor_path() -> PathBuf {
        repository_adapter_dir().join("hermes/fabric-adapter.json")
    }

    #[test]
    fn typed_profiles_merge_partial_objects_and_preserve_extensions() {
        let config: FabricConfig = serde_yaml::from_str(
            r#"
schema_version: fabric.agent/v1alpha1
metadata:
  name: demo
harness:
  adapter_id: nvidia.fabric.hermes
  settings:
    workspace: ./workspace
runtime:
  input_schema: chat
  output_schema: message
tools:
  enabled: [base]
  cleared:
    value: true
future_top_level:
  base: true
  nested:
    first: 1
"#,
        )
        .expect("base config");
        let profile: ProfileConfig = serde_yaml::from_str(
            r#"
schema_version: fabric.profile/v1alpha1
name: overlay
harness:
  settings:
    timeout_seconds: 30
runtime:
  input_schema: prompt
tools:
  enabled: [profile]
  cleared: null
future_top_level:
  profile: true
  nested:
    second: 2
"#,
        )
        .expect("partial profile");

        let effective = resolve_effective_config_from_config(
            config,
            &[profile],
            ResolveContext::from_agent_root("."),
        )
        .expect("effective config");
        let value = serde_json::to_value(&effective.config).expect("config json");

        assert_eq!(effective.profiles, ["overlay"]);
        assert_eq!(value["runtime"]["input_schema"], "prompt");
        assert_eq!(value["harness"]["settings"]["workspace"], "./workspace");
        assert_eq!(value["harness"]["settings"]["timeout_seconds"], 30);
        assert_eq!(value["tools"]["enabled"], serde_json::json!(["profile"]));
        assert!(value["tools"]["cleared"].is_null());
        assert_eq!(value["future_top_level"]["base"], true);
        assert_eq!(value["future_top_level"]["profile"], true);
        assert_eq!(value["future_top_level"]["nested"]["first"], 1);
        assert_eq!(value["future_top_level"]["nested"]["second"], 2);
    }

    #[test]
    fn runtime_uses_stable_defaults_for_omitted_optional_fields() {
        let config: FabricConfig = serde_yaml::from_str(
            r#"
schema_version: fabric.agent/v1alpha1
metadata:
  name: demo
harness:
  adapter_id: nvidia.fabric.hermes
runtime:
"#,
        )
        .expect("minimal config");

        assert_eq!(config.runtime.input_schema, "text");
        assert_eq!(config.runtime.output_schema, "text");
    }

    #[test]
    fn relay_telemetry_provider_is_enabled_by_presence() {
        let config: FabricConfig = serde_yaml::from_str(
            r#"
schema_version: fabric.agent/v1alpha1
metadata:
  name: demo
harness:
  adapter_id: nvidia.fabric.hermes
runtime:
telemetry:
  providers:
    relay: {}
"#,
        )
        .expect("config with relay telemetry provider");

        let telemetry = config.telemetry.as_ref().expect("telemetry config");
        let plan = resolve_telemetry_plan(&config, None)
            .expect("resolve telemetry plan")
            .expect("telemetry plan");

        assert!(telemetry.providers.contains_key(&TelemetryProvider::Relay));
        assert_eq!(plan.providers, vec![TelemetryProvider::Relay]);
        assert!(plan.relay_enabled);
        assert_eq!(plan.relay_config, None);
    }

    #[test]
    fn native_telemetry_provider_skips_relay_config() {
        let config: FabricConfig = serde_yaml::from_str(
            r#"
schema_version: fabric.agent/v1alpha1
metadata:
  name: demo
harness:
  adapter_id: nvidia.fabric.hermes
runtime:
telemetry:
  providers:
    native:
      config:
        exporter: test
relay:
  project: relay-project
  output_dir: ./relay-output
"#,
        )
        .expect("config with native telemetry provider");

        let plan = resolve_telemetry_plan(&config, None)
            .expect("resolve telemetry plan")
            .expect("telemetry plan");

        assert_eq!(plan.providers, vec![TelemetryProvider::Native]);
        assert!(!plan.relay_enabled);
        assert_eq!(plan.relay_project, None);
        assert_eq!(plan.relay_output_dir, None);
        assert_eq!(plan.relay_config, None);
        assert_eq!(
            plan.native_config,
            Some(serde_json::json!({"exporter": "test"}))
        );
    }

    #[test]
    fn relay_telemetry_config_generates_relay_plugin_config() {
        let config: FabricConfig = serde_yaml::from_str(
            r#"
schema_version: fabric.agent/v1alpha1
metadata:
  name: demo
harness:
  adapter_id: nvidia.fabric.hermes.sdk
runtime:
telemetry:
  providers:
    relay: {}
relay:
  project: typed-project
  output_dir: ./typed-relay
  observability:
    atof:
      enabled: true
      output_directory: ./typed-relay
      filename: events.atof.jsonl
      mode: overwrite
    atif:
      enabled: true
      output_directory: ./typed-relay
      filename_template: trajectory-{session_id}.atif.json
      agent_name: code-review-agent
    opentelemetry:
      enabled: true
      endpoint: http://localhost:4318/v1/traces
  components:
    - kind: switchyard
      enabled: true
      config:
        route: canary
  policy:
    unknown_component: error
"#,
        )
        .expect("config with typed relay telemetry");

        let plan = resolve_telemetry_plan(&config, None)
            .expect("resolve telemetry plan")
            .expect("telemetry plan");
        let relay_config = plan.relay_config.expect("relay plugin config");

        assert_eq!(plan.relay_project.as_deref(), Some("typed-project"));
        assert_eq!(plan.relay_output_dir, Some(PathBuf::from("./typed-relay")));
        assert_eq!(relay_config["version"], serde_json::json!(1));
        assert_eq!(
            relay_config["components"][0]["kind"],
            serde_json::json!("observability")
        );
        assert_eq!(
            relay_config["components"][0]["config"]["atof"]["mode"],
            serde_json::json!("overwrite")
        );
        assert_eq!(
            relay_config["components"][0]["config"]["atif"]["agent_name"],
            serde_json::json!("code-review-agent")
        );
        assert_eq!(
            relay_config["components"][0]["config"]["opentelemetry"]["endpoint"],
            serde_json::json!("http://localhost:4318/v1/traces")
        );
        assert_eq!(
            relay_config["components"][1],
            serde_json::json!({
                "kind": "switchyard",
                "enabled": true,
                "config": {"route": "canary"}
            })
        );
        assert_eq!(
            relay_config["policy"]["unknown_component"],
            serde_json::json!("error")
        );
    }

    #[test]
    fn telemetry_provider_rejects_unknown_provider_keys() {
        let result = serde_yaml::from_str::<TelemetryConfig>(
            r#"
providers:
  unsupported: {}
"#,
        );

        assert!(result.is_err());
    }

    #[test]
    fn telemetry_plan_rejects_provider_not_declared_by_adapter() {
        let config: FabricConfig = serde_yaml::from_str(
            r#"
schema_version: fabric.agent/v1alpha1
metadata:
  name: demo
harness:
  adapter_id: nvidia.fabric.claude
runtime:
telemetry:
  providers:
    relay: {}
"#,
        )
        .expect("config with relay telemetry provider");
        let descriptor: AdapterDescriptor = serde_json::from_value(serde_json::json!({
            "contract_version": ADAPTER_CONTRACT_VERSION,
            "adapter_id": "nvidia.fabric.claude",
            "harness": "claude",
            "adapter_kind": "python"
        }))
        .expect("adapter descriptor");

        let error = resolve_telemetry_plan(&config, Some(&descriptor))
            .expect_err("unsupported relay provider");

        assert!(matches!(
            error,
            FabricError::AdapterDescriptorUnsupported {
                adapter_id,
                field: "telemetry.providers",
                value,
            } if adapter_id == "nvidia.fabric.claude" && value == "relay"
        ));
    }

    #[test]
    fn telemetry_plan_uses_outputs_for_selected_adapter_providers() {
        let config: FabricConfig = serde_yaml::from_str(
            r#"
schema_version: fabric.agent/v1alpha1
metadata:
  name: demo
harness:
  adapter_id: nvidia.fabric.codex.cli
runtime:
telemetry:
  providers:
    relay: {}
    native: {}
"#,
        )
        .expect("config with relay and native telemetry providers");
        let descriptor: AdapterDescriptor = serde_json::from_value(serde_json::json!({
            "contract_version": ADAPTER_CONTRACT_VERSION,
            "adapter_id": "nvidia.fabric.codex.cli",
            "harness": "codex",
            "adapter_kind": "python",
            "telemetry": {
                "providers": {
                    "relay": {
                        "outputs": ["atif", "otel"],
                        "integration_modes": ["hooks", "gateway"]
                    },
                    "native": {
                        "outputs": ["otel"]
                    }
                }
            }
        }))
        .expect("adapter descriptor");

        let plan = resolve_telemetry_plan(&config, Some(&descriptor))
            .expect("resolve telemetry plan")
            .expect("telemetry plan");

        assert_eq!(
            plan.providers,
            vec![TelemetryProvider::Relay, TelemetryProvider::Native]
        );
        assert_eq!(plan.adapter_outputs, vec!["atif", "otel"]);
    }

    #[test]
    fn loads_adapter_descriptor() {
        let descriptor =
            load_adapter_descriptor(example_adapter_descriptor_path()).expect("adapter descriptor");

        assert_eq!(descriptor.contract_version, ADAPTER_CONTRACT_VERSION);
        assert_eq!(descriptor.adapter_id, "nvidia.fabric.hermes");
        assert_eq!(descriptor.harness, "hermes");
        assert_eq!(descriptor.adapter_kind, AdapterKind::Python);
        let descriptor_json = serde_json::to_value(&descriptor).expect("descriptor json");
        assert_eq!(
            descriptor_json.get("harness").and_then(Value::as_str),
            Some("hermes")
        );
        assert!(descriptor_json.get("harness_type").is_none());
        assert_eq!(
            descriptor.runner.get("module").and_then(Value::as_str),
            Some("nemo_fabric_adapters.hermes.adapter")
        );
        assert_eq!(
            descriptor.runner.get("callable").and_then(Value::as_str),
            Some("run")
        );
        assert!(descriptor.config.accepts.contains(&"telemetry".to_string()));
        let relay = descriptor
            .telemetry
            .providers
            .get(&TelemetryProvider::Relay)
            .expect("relay telemetry support");
        assert!(relay.outputs.contains(&"atif".to_string()));
    }

    #[test]
    fn resolves_base_config_from_agent_directory() {
        let plan = resolve_run_plan(file_config_agent_dir(), None).expect("run plan");

        assert_eq!(plan.agent_name, "code-review-agent");
        assert!(plan.profiles.is_empty());
        let plan_json = serde_json::to_value(&plan).expect("plan json");
        assert_eq!(plan_json["profiles"], serde_json::json!([]));
        assert_eq!(
            plan_json["capabilities"],
            serde_json::json!({
                "service": false,
                "streaming": false,
                "updates": false,
                "cancellation": false
            })
        );
        assert_eq!(plan.config.harness.adapter_id, "nvidia.fabric.hermes");
        assert_eq!(
            plan.adapter_descriptor
                .as_ref()
                .map(|adapter| adapter.descriptor.adapter_id.as_str()),
            Some("nvidia.fabric.hermes")
        );
        assert_eq!(
            plan.adapter_descriptor
                .as_ref()
                .map(|adapter| adapter.descriptor.adapter_kind),
            Some(AdapterKind::Python)
        );
        assert_eq!(
            plan.adapter_descriptor
                .as_ref()
                .map(|adapter| adapter.source),
            Some(AdapterDescriptorSource::Repository)
        );
        assert_eq!(plan.resolution, Some(ResolutionStrategy::Preinstalled));
        assert_eq!(
            plan.environment_plan
                .as_ref()
                .map(|environment| environment.provider.as_str()),
            Some("local")
        );
        assert_eq!(plan.capability_plan.skill_paths.len(), 1);
        assert!(plan.capability_plan.mcp_servers.contains_key("github"));
        assert_eq!(plan.capability_plan.native.skill_paths.len(), 1);
        assert!(
            plan.capability_plan
                .native
                .mcp_servers
                .contains_key("github")
        );
        assert!(plan.capability_plan.managed.skill_paths.is_empty());
        assert!(plan.capability_plan.managed.mcp_servers.is_empty());
        assert!(plan.config.profiles.directories.is_empty());
        assert_eq!(
            plan.config
                .telemetry
                .as_ref()
                .map(|telemetry| telemetry.providers.is_empty()),
            None
        );
    }

    #[test]
    fn resolves_hermes_adapter_descriptor() {
        let plan = resolve_run_plan(file_config_agent_dir(), Some("hermes")).expect("run plan");
        let adapter = plan
            .adapter_descriptor
            .as_ref()
            .expect("configured adapter");

        assert_eq!(adapter.source, AdapterDescriptorSource::Repository);
        assert_eq!(adapter.descriptor.adapter_id, "nvidia.fabric.hermes");
        assert_eq!(adapter.descriptor.adapter_kind, AdapterKind::Python);
        assert!(adapter.root.ends_with("adapters/hermes"));
    }

    #[test]
    fn resolves_package_local_custom_adapter_descriptor() {
        let root =
            std::env::temp_dir().join(format!("fabric-local-adapter-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("adapters/reviewer-process")).expect("create adapters");
        std::fs::write(
            root.join("agent.yaml"),
            r#"schema_version: fabric.agent/v1alpha1
metadata:
  name: reviewer-agent
harness:
  adapter_id: acme.fabric.reviewer.process
models:
  default:
    provider: test
    model: test-model
runtime:
  input_schema: text
  output_schema: message
environment:
  provider: local
"#,
        )
        .expect("write agent config");
        std::fs::write(
            root.join("adapters/reviewer-process/fabric-adapter.json"),
            r#"{
  "contract_version": "fabric.adapter/v1alpha1",
  "adapter_id": "acme.fabric.reviewer.process",
  "harness": "reviewer",
  "adapter_kind": "process"
}"#,
        )
        .expect("write adapter descriptor");

        let plan = resolve_run_plan(&root, None).expect("run plan");
        let adapter = plan
            .adapter_descriptor
            .as_ref()
            .expect("configured adapter");

        assert_eq!(adapter.source, AdapterDescriptorSource::Local);
        assert_eq!(
            adapter.descriptor.adapter_id,
            "acme.fabric.reviewer.process"
        );
        assert_eq!(adapter.descriptor.adapter_kind, AdapterKind::Process);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn resolves_env_profile_from_agent_directory() {
        let plan =
            resolve_run_plan(file_config_agent_dir(), Some("env_opensandbox")).expect("run plan");

        assert_eq!(plan.profiles, vec!["env_opensandbox"]);
        assert!(plan.config_path.ends_with("agent.yaml"));
        assert_eq!(
            plan.config
                .environment
                .as_ref()
                .map(|environment| environment.provider.as_str()),
            Some("opensandbox")
        );
    }

    #[test]
    fn resolves_mcp_profile_from_agent_directory() {
        let plan = resolve_run_plan(file_config_agent_dir(), Some("mcp_github")).expect("run plan");

        assert_eq!(plan.profiles, vec!["mcp_github"]);
        let plan_json = serde_json::to_value(&plan).expect("plan json");
        assert!(plan_json.get("profile").is_none());
        assert_eq!(
            plan.config.mcp.as_ref().map(|mcp| mcp.servers.len()),
            Some(1)
        );
        assert!(plan.capability_plan.native.mcp_servers.is_empty());
        assert!(plan.capability_plan.managed.mcp_servers.is_empty());
        assert!(
            plan.capability_plan.routes.iter().any(
                |route| route.name == "github" && route.target == CapabilityTarget::Unsupported
            )
        );
        assert!(
            plan.capability_plan
                .unsupported
                .mcp_servers
                .contains_key("github")
        );
    }

    #[test]
    fn resolves_ordered_profiles_from_agent_directory() {
        let profiles = vec!["env_local".to_string(), "mcp_github".to_string()];
        let plan =
            resolve_run_plan_with_profiles(file_config_agent_dir(), &profiles).expect("run plan");

        assert_eq!(plan.profiles, profiles);
        assert_eq!(
            plan.environment_plan
                .as_ref()
                .map(|environment| environment.provider.as_str()),
            Some("local")
        );
        assert_eq!(
            plan.telemetry_plan
                .as_ref()
                .map(|telemetry| telemetry.relay_enabled),
            Some(true)
        );
        assert!(plan.capability_plan.managed.mcp_servers.is_empty());
        assert!(
            plan.capability_plan
                .unsupported
                .mcp_servers
                .contains_key("github")
        );
    }

    #[test]
    fn resolves_in_memory_config_with_typed_profiles() {
        let FabricDocument::FabricConfig { config, root, .. } =
            load_fabric_document(file_config_agent_dir()).expect("agent config");
        let profile = read_yaml::<ProfileConfig>(&root.join("profiles/mcp-github.yaml"))
            .expect("profile config");

        let plan = resolve_run_plan_from_config(
            config,
            &[profile],
            ResolveContext::from_agent_root(root.clone()),
        )
        .expect("run plan");

        assert_eq!(plan.profiles, vec!["mcp_github"]);
        assert!(plan.config_path.ends_with("agent.yaml"));
        assert!(plan.config.profiles.directories.is_empty());
        assert!(plan.capability_plan.managed.mcp_servers.is_empty());
        assert!(
            plan.capability_plan
                .unsupported
                .mcp_servers
                .contains_key("github")
        );
        assert_eq!(plan.config_root, root);
    }

    #[test]
    fn unsupported_capabilities_do_not_claim_fabric_managed_execution() {
        let root = std::env::temp_dir().join(format!(
            "fabric-unsupported-capability-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("adapters/minimal")).expect("create adapters");
        std::fs::create_dir_all(root.join("skills/review")).expect("create skills");
        std::fs::write(
            root.join("agent.yaml"),
            r#"schema_version: fabric.agent/v1alpha1
metadata:
  name: unsupported-capability-agent
harness:
  adapter_id: acme.fabric.minimal
models:
  default:
    provider: test
    model: test-model
runtime:
  input_schema: text
  output_schema: text
tools:
  - name: shell
skills:
  paths:
    - ./skills/review
mcp:
  servers:
    github:
      transport: streamable-http
      url: http://example.invalid/mcp
      exposure: fabric_managed
"#,
        )
        .expect("write agent config");
        std::fs::write(
            root.join("adapters/minimal/fabric-adapter.json"),
            r#"{
  "contract_version": "fabric.adapter/v1alpha1",
  "adapter_id": "acme.fabric.minimal",
  "harness": "minimal",
  "adapter_kind": "process"
}"#,
        )
        .expect("write adapter descriptor");

        let plan = resolve_run_plan(&root, None).expect("run plan");

        assert!(!plan.capability_plan.managed.tools_configured);
        assert!(plan.capability_plan.managed.skill_paths.is_empty());
        assert!(plan.capability_plan.managed.mcp_servers.is_empty());
        assert!(plan.capability_plan.unsupported.tools_configured);
        assert_eq!(plan.capability_plan.unsupported.skill_paths.len(), 1);
        assert!(
            plan.capability_plan
                .unsupported
                .mcp_servers
                .contains_key("github")
        );
        assert!(
            plan.capability_plan
                .routes
                .iter()
                .all(|route| route.target == CapabilityTarget::Unsupported),
            "{:?}",
            plan.capability_plan.routes
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn later_profiles_override_earlier_profiles() {
        let profiles = vec!["env_opensandbox".to_string(), "env_local".to_string()];
        let plan =
            resolve_run_plan_with_profiles(file_config_agent_dir(), &profiles).expect("run plan");

        assert_eq!(plan.profiles, profiles);
        assert_eq!(
            plan.environment_plan
                .as_ref()
                .map(|environment| environment.provider.as_str()),
            Some("local")
        );
        assert_eq!(
            plan.environment_plan
                .as_ref()
                .and_then(|environment| environment.workspace.as_ref())
                .map(|path| path.ends_with("repos/my-service")),
            Some(true)
        );
        assert_eq!(
            plan.telemetry_plan
                .as_ref()
                .map(|telemetry| telemetry.relay_enabled),
            None
        );

        let profiles = vec!["env_local".to_string(), "env_opensandbox".to_string()];
        let plan =
            resolve_run_plan_with_profiles(file_config_agent_dir(), &profiles).expect("run plan");

        assert_eq!(
            plan.environment_plan
                .as_ref()
                .map(|environment| environment.provider.as_str()),
            Some("opensandbox")
        );
        assert_eq!(
            plan.telemetry_plan
                .as_ref()
                .map(|telemetry| telemetry.relay_enabled),
            Some(true)
        );
    }

    #[test]
    fn resolves_hermes_profile_from_agent_directory() {
        let plan = resolve_run_plan(file_config_agent_dir(), Some("hermes")).expect("run plan");

        assert_eq!(plan.profiles, vec!["hermes"]);
        assert_eq!(plan.config.harness.adapter_id, "nvidia.fabric.hermes");
        assert_eq!(
            plan.adapter_descriptor
                .as_ref()
                .map(|adapter| adapter.descriptor.adapter_id.as_str()),
            Some("nvidia.fabric.hermes")
        );
        assert_eq!(
            plan.adapter_descriptor
                .as_ref()
                .map(|adapter| adapter.descriptor.adapter_kind),
            Some(AdapterKind::Python)
        );
        assert_eq!(
            plan.adapter_descriptor
                .as_ref()
                .map(|adapter| adapter.source),
            Some(AdapterDescriptorSource::Repository)
        );
        assert_eq!(plan.resolution, Some(ResolutionStrategy::Preinstalled));
        assert_eq!(
            plan.telemetry_plan
                .as_ref()
                .map(|telemetry| telemetry.relay_enabled),
            None
        );
        assert!(
            plan.capability_plan
                .native
                .mcp_servers
                .contains_key("github")
        );
        assert!(plan.capability_plan.managed.mcp_servers.is_empty());
        assert_eq!(plan.capability_plan.native.skill_paths.len(), 1);
        assert!(plan.capability_plan.managed.skill_paths.is_empty());
    }

    #[test]
    fn resolves_direct_profile_path_from_agent_directory() {
        let plan = resolve_run_plan(file_config_agent_dir(), Some("./profiles/hermes.yaml"))
            .expect("run plan");

        assert_eq!(plan.profiles, vec!["./profiles/hermes.yaml"]);
        assert_eq!(plan.config.harness.adapter_id, "nvidia.fabric.hermes");
        assert_eq!(
            plan.adapter_descriptor
                .as_ref()
                .map(|adapter| adapter.descriptor.adapter_id.as_str()),
            Some("nvidia.fabric.hermes")
        );
    }

    #[test]
    fn errors_for_unknown_manifest_profile() {
        let error = resolve_run_plan(file_config_agent_dir(), Some("missing")).expect_err("error");

        assert!(matches!(error, FabricError::UnknownProfile { .. }));
    }

    #[test]
    fn rejects_malformed_adapter_descriptor() {
        let root = std::env::temp_dir().join(format!(
            "fabric-invalid-adapter-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("adapters/invalid-process")).expect("create adapters");
        std::fs::write(
            root.join("agent.yaml"),
            r#"schema_version: fabric.agent/v1alpha1
metadata:
  name: invalid-adapter-agent
harness:
  adapter_id: acme.fabric.invalid.process
models:
  default:
    provider: test
    model: test-model
runtime:
  input_schema: text
  output_schema: message
environment:
  provider: local
"#,
        )
        .expect("write agent config");
        std::fs::write(
            root.join("adapters/invalid-process/fabric-adapter.json"),
            r#"{
  "contract_version": "fabric.adapter/v1alpha1",
  "adapter_id": "   ",
  "harness": "invalid",
  "adapter_kind": "process"
}"#,
        )
        .expect("write adapter descriptor");

        let error = resolve_run_plan(&root, None).expect_err("invalid descriptor");
        assert!(matches!(
            error,
            FabricError::InvalidAdapterDescriptor { message, .. }
                if message.contains("adapter_id")
        ));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn rejects_empty_adapter_id_when_loading_descriptor_directly() {
        let root = std::env::temp_dir().join(format!(
            "fabric-invalid-adapter-shape-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("create temp root");
        let descriptor_path = root.join("fabric-adapter.json");
        std::fs::write(
            &descriptor_path,
            r#"{
  "contract_version": "fabric.adapter/v1alpha1",
  "adapter_id": "",
  "harness": "invalid",
  "adapter_kind": "process"
}"#,
        )
        .expect("write adapter descriptor");

        let error = load_adapter_descriptor(&descriptor_path).expect_err("invalid descriptor");
        assert!(matches!(
            error,
            FabricError::InvalidAdapterDescriptor { message, .. }
                if message.contains("adapter_id")
        ));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn rejects_unsupported_adapter_contract_version() {
        let root = std::env::temp_dir().join(format!(
            "fabric-invalid-adapter-contract-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("create temp root");
        let descriptor_path = root.join("fabric-adapter.json");
        std::fs::write(
            &descriptor_path,
            r#"{
  "contract_version": "fabric.adapter/v9",
  "adapter_id": "acme.fabric.future",
  "harness": "future",
  "adapter_kind": "process"
}"#,
        )
        .expect("write adapter descriptor");

        let error = load_adapter_descriptor(&descriptor_path).expect_err("invalid descriptor");
        assert!(matches!(
            error,
            FabricError::AdapterDescriptorUnsupported {
                field,
                value,
                ..
            } if field == "contract_version" && value == "fabric.adapter/v9"
        ));

        let _ = std::fs::remove_dir_all(root);
    }
}
