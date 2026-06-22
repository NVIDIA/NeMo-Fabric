// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fabric config models and loading helpers.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{FabricError, Result};

const AGENT_YAML: &str = "agent.yaml";

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
    /// Runtime mode and input/output contract.
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
    /// Optional profile discovery config.
    #[serde(default, skip_serializing_if = "ProfileRegistryConfig::is_empty")]
    pub profiles: ProfileRegistryConfig,
}

/// Profile discovery config for curated package profiles.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ProfileRegistryConfig {
    /// Directories searched when a caller selects a profile by name.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub directories: Vec<PathBuf>,
}

impl ProfileRegistryConfig {
    fn is_empty(&self) -> bool {
        self.directories.is_empty()
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
}

/// Language-neutral adapter descriptor for a harness integration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AdapterDescriptor {
    /// Unique id for this adapter implementation.
    pub adapter_id: String,
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
}

/// Adapter telemetry support.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AdapterTelemetrySupport {
    /// Telemetry outputs supported by this adapter.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supports: Vec<String>,
}

/// Profile config applied on top of a Fabric config.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Default)]
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
    /// Harness overrides.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub harness: Option<HarnessConfig>,
    /// Model aliases to add or replace.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub models: BTreeMap<String, ModelConfig>,
    /// Runtime override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<RuntimeConfig>,
    /// Environment override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment: Option<EnvironmentConfig>,
    /// Tool capability override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Value>,
    /// Skill capability override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skills: Option<SkillConfig>,
    /// MCP capability override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp: Option<McpConfig>,
    /// Telemetry override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry: Option<TelemetryConfig>,
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
}

/// Runtime mode and input/output contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RuntimeConfig {
    /// Runtime mode.
    pub mode: RuntimeMode,
    /// Transport used to operate the harness.
    pub transport: Transport,
    /// Input schema label.
    pub input_schema: String,
    /// Output schema label.
    pub output_schema: String,
    /// Artifact directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifacts: Option<PathBuf>,
}

/// Runtime lifecycle mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeMode {
    /// Request is the lifecycle boundary.
    Oneshot,
    /// Long-running process or service is the lifecycle boundary.
    Service,
    /// Session is the lifecycle boundary.
    Session,
}

/// Runtime transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Transport {
    /// In-process library/SDK call.
    Library,
    /// CLI process.
    Cli,
    /// HTTP service.
    Http,
    /// Harness-native plugin surface.
    NativePlugin,
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
    /// Provider connection metadata, such as server URL, session id, or namespace.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub connection: serde_json::Map<String, Value>,
    /// Consumer-provided environment metadata.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub metadata: serde_json::Map<String, Value>,
    /// Provider-specific settings.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub settings: serde_json::Map<String, Value>,
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
}

/// MCP capability configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Default)]
pub struct McpConfig {
    /// Named MCP servers.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub servers: BTreeMap<String, McpServerConfig>,
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
    /// Whether telemetry is enabled for this run. Relay is the Phase 1 telemetry path.
    #[serde(default)]
    pub enabled: bool,
    /// Telemetry mode, for example `sdk`, `gateway`, or `external`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// Optional project name for telemetry backends.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    /// Optional telemetry output directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_dir: Option<PathBuf>,
    /// Pass-through telemetry backend config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<Value>,
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
            Ok(resolve_effective_config_from_config_with_profile_names(
                config,
                &profile_configs,
                selected_profiles,
                ResolveContext::from_config_path(path, root),
            ))
        }
    }
}

/// Resolve typed config/profile overlays into merged effective config.
pub fn resolve_effective_config_from_config(
    config: FabricConfig,
    profiles: &[ProfileConfig],
    context: ResolveContext,
) -> EffectiveConfig {
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
    ))
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
) -> EffectiveConfig {
    let mut effective = config;
    for profile in profiles {
        apply_profile_config(&mut effective, profile);
    }
    let config = into_effective_config(effective);
    let profile = if selected_profiles.len() == 1 {
        selected_profiles.first().cloned()
    } else {
        None
    };
    EffectiveConfig {
        agent_name: config.metadata.name.clone(),
        profile,
        profiles: selected_profiles,
        agent_root: context.agent_root,
        config_path: context.config_path,
        config_root: context.config_root,
        config,
    }
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
    let telemetry_plan = resolve_telemetry_plan(&config, descriptor);
    Ok(RunPlan {
        agent_name: effective_config.agent_name.clone(),
        profile: effective_config.profile.clone(),
        profiles: effective_config.profiles.clone(),
        adapter_descriptor,
        resolution,
        environment_plan,
        capability_plan,
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

fn apply_profile_config(config: &mut FabricConfig, profile: &ProfileConfig) {
    if let Some(harness) = &profile.harness {
        config.harness = harness.clone();
    }
    for (name, model) in &profile.models {
        config.models.insert(name.clone(), model.clone());
    }
    if let Some(runtime) = &profile.runtime {
        config.runtime = runtime.clone();
    }
    if let Some(environment) = &profile.environment {
        config.environment = Some(environment.clone());
    }
    if let Some(tools) = &profile.tools {
        config.tools = Some(tools.clone());
    }
    if let Some(skills) = &profile.skills {
        config.skills = Some(skills.clone());
    }
    if let Some(mcp) = &profile.mcp {
        config.mcp = Some(mcp.clone());
    }
    if let Some(telemetry) = &profile.telemetry {
        config.telemetry = Some(telemetry.clone());
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
    if descriptor.adapter_id.trim().is_empty() {
        return invalid_adapter_descriptor(path, "adapter_id must not be empty");
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
    let mut managed = CapabilityTargetPlan::default();
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
            managed.tools_configured = true;
            routes.push(CapabilityRoute {
                kind: CapabilityKind::Tools,
                name: "tools".to_string(),
                target: CapabilityTarget::FabricManaged,
                reason: "selected adapter does not declare native tools support".to_string(),
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
            managed.skill_paths = skill_paths.clone();
            routes.push(CapabilityRoute {
                kind: CapabilityKind::Skills,
                name: "skills".to_string(),
                target: CapabilityTarget::FabricManaged,
                reason: "selected adapter does not declare native skills support".to_string(),
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
            managed.mcp_servers.insert(name.clone(), server.clone());
            routes.push(CapabilityRoute {
                kind: CapabilityKind::Mcp,
                name: name.clone(),
                target: CapabilityTarget::FabricManaged,
                reason: match server.exposure {
                    McpExposure::FabricManaged => {
                        "MCP server explicitly requests Fabric-managed exposure".to_string()
                    }
                    _ => "selected adapter does not declare native MCP support".to_string(),
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
) -> Option<TelemetryPlan> {
    let telemetry = config.telemetry.as_ref()?;
    Some(TelemetryPlan {
        relay_enabled: telemetry.enabled,
        relay_mode: telemetry.mode.clone(),
        relay_project: telemetry.project.clone(),
        relay_output_dir: telemetry.output_dir.clone(),
        relay_config: telemetry.config.clone(),
        adapter_outputs: adapter_descriptor
            .map(|descriptor| descriptor.telemetry.supports.clone())
            .unwrap_or_default(),
    })
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
    /// Selected profile when exactly one profile is applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    /// Ordered selected profiles.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
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
    /// Selected profile when exactly one profile is applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    /// Ordered selected profiles.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
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
    /// Whether Relay is enabled.
    pub relay_enabled: bool,
    /// Relay mode, when configured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relay_mode: Option<String>,
    /// Relay project, when configured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relay_project: Option<String>,
    /// Relay output directory, when configured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relay_output_dir: Option<PathBuf>,
    /// Relay pass-through config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relay_config: Option<Value>,
    /// Telemetry outputs declared by the selected adapter descriptor.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub adapter_outputs: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_agent_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/code-review-agent")
    }

    fn example_adapter_descriptor_path() -> PathBuf {
        repository_adapter_dir().join("hermes-sdk/fabric-adapter.json")
    }

    #[test]
    fn loads_adapter_descriptor() {
        let descriptor =
            load_adapter_descriptor(example_adapter_descriptor_path()).expect("adapter descriptor");

        assert_eq!(descriptor.adapter_id, "nvidia.fabric.hermes.sdk");
        assert_eq!(descriptor.adapter_kind, AdapterKind::Python);
        assert_eq!(
            descriptor.runner.get("module").and_then(Value::as_str),
            Some("nemo_fabric_adapters.hermes_sdk.adapter")
        );
        assert_eq!(
            descriptor.runner.get("callable").and_then(Value::as_str),
            Some("run")
        );
        assert!(descriptor.config.accepts.contains(&"telemetry".to_string()));
        assert!(descriptor.telemetry.supports.contains(&"relay".to_string()));
    }

    #[test]
    fn resolves_base_config_from_agent_directory() {
        let plan = resolve_run_plan(example_agent_dir(), None).expect("run plan");

        assert_eq!(plan.agent_name, "code-review-agent");
        assert_eq!(plan.profile.as_deref(), None);
        assert_eq!(plan.config.harness.adapter_id, "nvidia.fabric.hermes.sdk");
        assert_eq!(
            plan.adapter_descriptor
                .as_ref()
                .map(|adapter| adapter.descriptor.adapter_id.as_str()),
            Some("nvidia.fabric.hermes.sdk")
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
                .map(|telemetry| telemetry.enabled),
            Some(false)
        );
    }

    #[test]
    fn resolves_hermes_sdk_adapter_descriptor() {
        let plan = resolve_run_plan(example_agent_dir(), Some("hermes_sdk")).expect("run plan");
        let adapter = plan
            .adapter_descriptor
            .as_ref()
            .expect("configured adapter");

        assert_eq!(adapter.source, AdapterDescriptorSource::Repository);
        assert_eq!(adapter.descriptor.adapter_id, "nvidia.fabric.hermes.sdk");
        assert_eq!(adapter.descriptor.adapter_kind, AdapterKind::Python);
        assert!(adapter.root.ends_with("adapters/hermes-sdk"));
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
  mode: oneshot
  transport: cli
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
  "adapter_id": "acme.fabric.reviewer.process",
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
            resolve_run_plan(example_agent_dir(), Some("env_opensandbox")).expect("run plan");

        assert_eq!(plan.profile.as_deref(), Some("env_opensandbox"));
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
        let plan = resolve_run_plan(example_agent_dir(), Some("mcp_github")).expect("run plan");

        assert_eq!(plan.profile.as_deref(), Some("mcp_github"));
        assert_eq!(plan.profiles, vec!["mcp_github"]);
        assert_eq!(
            plan.config.mcp.as_ref().map(|mcp| mcp.servers.len()),
            Some(1)
        );
        assert!(plan.capability_plan.native.mcp_servers.is_empty());
        assert!(
            plan.capability_plan
                .managed
                .mcp_servers
                .contains_key("github")
        );
        assert!(
            plan.capability_plan
                .routes
                .iter()
                .any(|route| route.name == "github"
                    && route.target == CapabilityTarget::FabricManaged)
        );
    }

    #[test]
    fn resolves_ordered_profiles_from_agent_directory() {
        let profiles = vec!["env_local".to_string(), "mcp_github".to_string()];
        let plan =
            resolve_run_plan_with_profiles(example_agent_dir(), &profiles).expect("run plan");

        assert_eq!(plan.profile, None);
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
        assert!(
            plan.capability_plan
                .managed
                .mcp_servers
                .contains_key("github")
        );
    }

    #[test]
    fn resolves_in_memory_config_with_typed_profiles() {
        let FabricDocument::FabricConfig { config, root, .. } =
            load_fabric_document(example_agent_dir()).expect("agent config");
        let profile = read_yaml::<ProfileConfig>(&root.join("profiles/mcp-github.yaml"))
            .expect("profile config");

        let plan = resolve_run_plan_from_config(
            config,
            &[profile],
            ResolveContext::from_agent_root(root.clone()),
        )
        .expect("run plan");

        assert_eq!(plan.profile.as_deref(), Some("mcp_github"));
        assert_eq!(plan.profiles, vec!["mcp_github"]);
        assert!(plan.config_path.ends_with("agent.yaml"));
        assert!(plan.config.profiles.directories.is_empty());
        assert!(
            plan.capability_plan
                .managed
                .mcp_servers
                .contains_key("github")
        );
        assert_eq!(plan.config_root, root);
    }

    #[test]
    fn later_profiles_override_earlier_profiles() {
        let profiles = vec!["env_opensandbox".to_string(), "env_local".to_string()];
        let plan =
            resolve_run_plan_with_profiles(example_agent_dir(), &profiles).expect("run plan");

        assert_eq!(plan.profile, None);
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
            Some(false)
        );

        let profiles = vec!["env_local".to_string(), "env_opensandbox".to_string()];
        let plan =
            resolve_run_plan_with_profiles(example_agent_dir(), &profiles).expect("run plan");

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
    fn resolves_hermes_sdk_profile_from_agent_directory() {
        let plan = resolve_run_plan(example_agent_dir(), Some("hermes_sdk")).expect("run plan");

        assert_eq!(plan.profile.as_deref(), Some("hermes_sdk"));
        assert_eq!(plan.profiles, vec!["hermes_sdk"]);
        assert_eq!(plan.config.harness.adapter_id, "nvidia.fabric.hermes.sdk");
        assert_eq!(
            plan.adapter_descriptor
                .as_ref()
                .map(|adapter| adapter.descriptor.adapter_id.as_str()),
            Some("nvidia.fabric.hermes.sdk")
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
            Some(false)
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
        let plan = resolve_run_plan(example_agent_dir(), Some("./profiles/hermes-sdk.yaml"))
            .expect("run plan");

        assert_eq!(plan.profile.as_deref(), Some("./profiles/hermes-sdk.yaml"));
        assert_eq!(plan.config.harness.adapter_id, "nvidia.fabric.hermes.sdk");
        assert_eq!(
            plan.adapter_descriptor
                .as_ref()
                .map(|adapter| adapter.descriptor.adapter_id.as_str()),
            Some("nvidia.fabric.hermes.sdk")
        );
    }

    #[test]
    fn errors_for_unknown_manifest_profile() {
        let error = resolve_run_plan(example_agent_dir(), Some("missing")).expect_err("error");

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
  mode: oneshot
  transport: cli
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
  "adapter_id": "   ",
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
  "adapter_id": "",
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
}
