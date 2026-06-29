// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error types for Fabric core.

use std::path::PathBuf;

use crate::config::AdapterKind;

/// Core Fabric result type.
pub type Result<T> = std::result::Result<T, FabricError>;

/// Errors raised by Fabric config loading and validation.
#[derive(Debug, thiserror::Error)]
pub enum FabricError {
    /// The requested path does not exist.
    #[error("path does not exist: {0}")]
    PathNotFound(PathBuf),
    /// A directory did not contain the expected Fabric entrypoint.
    #[error("directory does not contain agent.yaml: {0}")]
    MissingEntrypoint(PathBuf),
    /// A file extension is not recognized for Fabric config loading.
    #[error("unsupported config file extension for {0}")]
    UnsupportedExtension(PathBuf),
    /// A requested profile is not present in an agent manifest.
    #[error("unknown profile `{profile}` for agent `{agent}`; available profiles: {available:?}")]
    UnknownProfile {
        /// Requested profile.
        profile: String,
        /// Agent name.
        agent: String,
        /// Available profile names.
        available: Vec<String>,
    },
    /// A profile config failed validation.
    #[error("profile error in {path}: {message}")]
    ProfileError {
        /// Profile path.
        path: PathBuf,
        /// Validation message.
        message: String,
    },
    /// A requested adapter id is not present in the agent config.
    #[error("unknown adapter `{adapter_id}`; available adapters: {available:?}")]
    UnknownAdapter {
        /// Requested adapter id.
        adapter_id: String,
        /// Available adapter ids.
        available: Vec<String>,
    },
    /// An adapter descriptor did not match the selected harness config.
    #[error(
        "adapter descriptor mismatch in {path}: `{field}` expected `{expected}` but found `{actual}`"
    )]
    AdapterDescriptorMismatch {
        /// Adapter descriptor path.
        path: PathBuf,
        /// Mismatched field.
        field: &'static str,
        /// Expected value.
        expected: String,
        /// Actual value.
        actual: String,
    },
    /// An adapter descriptor does not support a selected config value.
    #[error("adapter `{adapter_id}` does not support `{field}` value `{value}`")]
    AdapterDescriptorUnsupported {
        /// Adapter id.
        adapter_id: String,
        /// Unsupported field.
        field: &'static str,
        /// Unsupported value.
        value: String,
    },
    /// An adapter descriptor is malformed.
    #[error("invalid adapter descriptor in {path}: {message}")]
    InvalidAdapterDescriptor {
        /// Adapter descriptor path.
        path: PathBuf,
        /// Validation message.
        message: String,
    },
    /// A requested schema is not known.
    #[error("unknown schema `{schema}`; available schemas: {available:?}")]
    UnknownSchema {
        /// Requested schema name.
        schema: String,
        /// Available schema names.
        available: Vec<String>,
    },
    /// A profile path resolved to an agent manifest instead of a profile config.
    #[error("profile `{profile}` did not resolve to a Fabric config: {path}")]
    ProfileTargetNotConfig {
        /// Profile name.
        profile: String,
        /// Resolved path.
        path: PathBuf,
    },
    /// A profile was requested for a single profile config.
    #[error("profile selection is only supported for agent manifests: {0}")]
    ProfileSelectionNotSupported(PathBuf),
    /// Runtime invocation is not supported for the selected adapter.
    #[error(
        "runtime invocation is not implemented for harness `{harness}` with adapter `{adapter_kind:?}`"
    )]
    UnsupportedRuntimeAdapter {
        /// Harness type.
        harness: String,
        /// Adapter kind.
        adapter_kind: AdapterKind,
    },
    /// A runtime handle was used with a different run plan than the one that created it.
    #[error(
        "runtime handle does not match run plan for `{field}`: expected `{expected}` but found `{actual}` (runtime `{runtime_id}`)"
    )]
    RuntimeHandleMismatch {
        /// Mismatched runtime handle field.
        field: &'static str,
        /// Expected value from the run plan.
        expected: String,
        /// Actual value from the runtime handle.
        actual: String,
        /// Runtime handle id.
        runtime_id: String,
    },
    /// An environment provider is not runnable for the selected adapter in this POC.
    #[error("environment provider `{provider}` is not implemented for adapter `{adapter_kind:?}`")]
    UnsupportedEnvironmentProvider {
        /// Environment provider.
        provider: String,
        /// Adapter kind.
        adapter_kind: AdapterKind,
    },
    /// Process adapter settings were invalid.
    #[error("invalid process adapter settings for {path}: {source}")]
    InvalidProcessSettings {
        /// Config path.
        path: PathBuf,
        /// Underlying JSON parse error.
        source: serde_json::Error,
    },
    /// Python adapter settings were invalid.
    #[error("invalid python adapter settings for {path}: {source}")]
    InvalidPythonSettings {
        /// Config path.
        path: PathBuf,
        /// Underlying JSON parse error.
        source: serde_json::Error,
    },
    /// A process runner failed to start or complete.
    #[error("process runner failed for `{command}`: {source}")]
    ProcessRunner {
        /// Command being run.
        command: String,
        /// Underlying IO error.
        source: std::io::Error,
    },
    /// JSON serialization failed.
    #[error("failed to serialize JSON: {0}")]
    SerializeJson(serde_json::Error),
    /// Filesystem read failed.
    #[error("failed to read {path}: {source}")]
    Read {
        /// File path.
        path: PathBuf,
        /// Underlying IO error.
        source: std::io::Error,
    },
    /// Filesystem write failed.
    #[error("failed to write {path}: {source}")]
    Write {
        /// File path.
        path: PathBuf,
        /// Underlying IO error.
        source: std::io::Error,
    },
    /// YAML parse failed.
    #[error("failed to parse YAML in {path}: {source}")]
    ParseYaml {
        /// File path.
        path: PathBuf,
        /// Underlying YAML error.
        source: serde_yaml::Error,
    },
    /// JSON parse failed.
    #[error("failed to parse JSON in {path}: {source}")]
    ParseJson {
        /// File path.
        path: PathBuf,
        /// Underlying JSON error.
        source: serde_json::Error,
    },
}
