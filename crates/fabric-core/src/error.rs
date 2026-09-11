// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error types for NeMo Fabric core.

use std::path::PathBuf;

use crate::config::{AdapterKind, DescriptorSource};

/// Core NeMo Fabric result type.
pub type Result<T> = std::result::Result<T, FabricError>;

/// Errors raised by NeMo Fabric config loading and validation.
#[derive(Debug, thiserror::Error)]
pub enum FabricError {
    /// The base directory could not be resolved to an absolute path.
    #[error("failed to resolve base directory {path}: {source}")]
    ResolveBaseDirectory {
        /// Base directory supplied by the caller.
        path: PathBuf,
        /// Underlying path-resolution error.
        source: std::io::Error,
    },
    /// The requested path does not exist.
    #[error("path does not exist: {0}")]
    PathNotFound(PathBuf),
    /// A requested adapter id is not present in the agent config.
    #[error("unknown adapter `{adapter_id}`; available adapters: {available:?}")]
    UnknownAdapter {
        /// Requested adapter id.
        adapter_id: String,
        /// Available adapter ids.
        available: Vec<String>,
    },
    /// A requested Adapter Target Descriptor id was not discovered.
    #[error("unknown adapter target `{target_id}`; available targets: {available:?}")]
    UnknownAdapterTarget {
        /// Requested target id.
        target_id: String,
        /// Available target ids.
        available: Vec<String>,
    },
    /// More than one distinct descriptor record was discovered for one id.
    #[error("ambiguous {descriptor_kind} descriptor `{id}`; conflicting paths: {paths:?}")]
    AmbiguousDescriptor {
        /// Human-readable descriptor category.
        descriptor_kind: &'static str,
        /// Conflicting descriptor id.
        id: String,
        /// Every conflicting descriptor path.
        paths: Vec<PathBuf>,
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
    /// An Adapter Target Descriptor is malformed.
    #[error("invalid adapter target descriptor in {path}: {message}")]
    InvalidAdapterTargetDescriptor {
        /// Adapter Target Descriptor path.
        path: PathBuf,
        /// Validation message.
        message: String,
    },
    /// Adapter-owned harness settings do not satisfy the resolved descriptor schema.
    #[error(
        "invalid harness settings for adapter `{adapter_id}` from {descriptor_source:?} descriptor {descriptor_path} at `{settings_path}`: {reason}"
    )]
    InvalidHarnessSettings {
        /// Selected adapter id.
        adapter_id: String,
        /// Registry source of the selected descriptor.
        descriptor_source: DescriptorSource,
        /// Path to the selected descriptor.
        descriptor_path: PathBuf,
        /// Canonical path to the invalid setting.
        settings_path: String,
        /// Schema validation failure.
        reason: String,
    },
    /// Adapter-owned workflow configuration does not satisfy the resolved descriptor schema.
    #[error(
        "invalid workflow for adapter `{adapter_id}` from {descriptor_source:?} descriptor {descriptor_path} at `{workflow_path}`: {reason}"
    )]
    InvalidWorkflow {
        /// Selected adapter id.
        adapter_id: String,
        /// Registry source of the selected descriptor.
        descriptor_source: DescriptorSource,
        /// Path to the selected descriptor.
        descriptor_path: PathBuf,
        /// Canonical path to the invalid workflow field.
        workflow_path: String,
        /// Schema validation failure.
        reason: String,
    },
    /// A normalized tool definition does not satisfy the resolved descriptor schema.
    #[error(
        "invalid tool definition for adapter `{adapter_id}` from {descriptor_source:?} descriptor {descriptor_path} at `{definition_path}`: {reason}"
    )]
    InvalidToolDefinition {
        /// Selected adapter id.
        adapter_id: String,
        /// Registry source of the selected descriptor.
        descriptor_source: DescriptorSource,
        /// Path to the selected descriptor.
        descriptor_path: PathBuf,
        /// Canonical path to the invalid definition field.
        definition_path: String,
        /// Schema validation failure.
        reason: String,
    },
    /// Adapter-owned extensions do not satisfy a descriptor extension schema.
    #[error(
        "invalid adapter extension for adapter `{adapter_id}` from {descriptor_source:?} descriptor {descriptor_path} at `{extension_path}`: {reason}"
    )]
    InvalidAdapterExtension {
        /// Selected adapter id.
        adapter_id: String,
        /// Registry source of the selected descriptor.
        descriptor_source: DescriptorSource,
        /// Path to the selected descriptor.
        descriptor_path: PathBuf,
        /// Canonical path to the invalid extension field.
        extension_path: String,
        /// Schema validation failure.
        reason: String,
    },
    /// A normalized Fabric config field is invalid.
    #[error("invalid Fabric configuration at `{field}`: {reason}")]
    InvalidConfig {
        /// Canonical configuration path.
        field: String,
        /// Validation failure.
        reason: String,
    },
    /// A valid normalized field cannot be implemented by the selected adapter.
    #[error("adapter `{adapter_id}` cannot implement configuration at `{field}`: {reason}")]
    AdapterCompatibility {
        /// Selected adapter id.
        adapter_id: String,
        /// Canonical configuration path.
        field: String,
        /// Compatibility failure.
        reason: String,
    },
    /// A requested schema is not known.
    #[error("unknown schema `{schema}`; available schemas: {available:?}")]
    UnknownSchema {
        /// Requested schema name.
        schema: String,
        /// Available schema names.
        available: Vec<String>,
    },
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
    /// A requested runtime capability is not implemented by the selected adapter.
    #[error("adapter `{adapter_id}` does not support runtime capability `{capability}`")]
    UnsupportedRuntimeCapability {
        /// Selected adapter id or harness name.
        adapter_id: String,
        /// Requested capability.
        capability: &'static str,
    },
    /// The SDK-provided native streaming transport is invalid.
    #[error("invalid OpenAI stream transport at `{field}`: {reason}")]
    InvalidOpenAiStreamTransport {
        /// Invalid transport field.
        field: &'static str,
        /// Validation failure without credential material.
        reason: &'static str,
    },
    /// A persistent local-host lifecycle operation failed.
    #[error(
        "adapter lifecycle {operation} failed for runtime `{runtime_id}` ({code}): {message}{diagnostics_suffix}",
        diagnostics_suffix = if diagnostics.is_empty() {
            String::new()
        } else {
            format!("; diagnostics: {diagnostics}")
        }
    )]
    AdapterLifecycleOperation {
        /// Lifecycle operation that failed.
        operation: &'static str,
        /// Runtime whose host failed.
        runtime_id: String,
        /// Stable failure code.
        code: String,
        /// Human-readable failure message.
        message: String,
        /// Bounded adapter-host diagnostics.
        diagnostics: String,
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
    /// An environment provider is not runnable for the selected adapter.
    #[error("environment provider `{provider}` is not implemented for adapter `{adapter_kind:?}`")]
    UnsupportedEnvironmentProvider {
        /// Environment provider.
        provider: String,
        /// Adapter kind.
        adapter_kind: AdapterKind,
    },
    /// An environment provider process could not complete an operation.
    #[error("environment provider `{provider}` failed during `{operation}` ({code}): {message}")]
    EnvironmentProviderOperation {
        /// Environment provider.
        provider: String,
        /// Lifecycle operation.
        operation: String,
        /// Stable provider or host error code.
        code: String,
        /// Sanitized failure detail.
        message: String,
    },
    /// A non-local runtime was started without an explicitly prepared environment handle.
    #[error(
        "environment provider `{provider}` requires an explicit EnvironmentHandle; call prepare_environment() and then start_runtime_in()"
    )]
    EnvironmentHandleRequired {
        /// Environment provider selected by the run plan.
        provider: String,
    },
    /// A prepared environment handle does not match the run plan that is binding to it.
    #[error(
        "environment handle does not match run plan for `{field}`: expected `{expected}` but found `{actual}` (environment `{environment_id}`)"
    )]
    EnvironmentHandleMismatch {
        /// Mismatched environment field.
        field: &'static str,
        /// Expected value from the run plan.
        expected: String,
        /// Actual value from the environment handle.
        actual: String,
        /// Environment handle id.
        environment_id: String,
    },
    /// An environment already has an active Fabric runtime session.
    #[error(
        "environment `{environment_id}` is already bound to runtime `{runtime_id}`; stop that runtime before starting another session"
    )]
    EnvironmentInUse {
        /// Environment whose exclusive session slot is occupied.
        environment_id: String,
        /// Active runtime holding the slot.
        runtime_id: String,
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
    /// The resolved Python adapter interpreter could not be used.
    #[error(
        "python adapter interpreter {path} (from {origin}) is unusable: {reason}; \
         set the `ADAPTER_PYTHON` environment variable to a valid interpreter"
    )]
    PythonInterpreterUnavailable {
        /// Resolved interpreter path.
        path: PathBuf,
        /// Human-readable description of where the interpreter was resolved from.
        origin: String,
        /// Why the interpreter cannot be used.
        reason: String,
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
    /// JSON parse failed.
    #[error("failed to parse JSON in {path}: {source}")]
    ParseJson {
        /// File path.
        path: PathBuf,
        /// Underlying JSON error.
        source: serde_json::Error,
    },
}
