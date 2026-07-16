// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plan diagnostics for Fabric.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::{
    AdapterKind, CapabilityKind, CapabilityTarget, ControlLocation, EffectiveConfig,
    EnvironmentOwnership, ModelCredentialRef, ModelEndpointRef, ResolutionStrategy, RunPlan,
    resolve_run_plan_from_effective_config,
};
use crate::error::{FabricError, Result};

/// Diagnostic status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DoctorStatus {
    /// Check passed.
    Pass,
    /// Check is informational or partially supported.
    Warn,
    /// Check failed.
    Fail,
}

/// Diagnostic check result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DoctorCheck {
    /// Stable check name.
    pub name: String,
    /// Check status.
    pub status: DoctorStatus,
    /// Human-readable detail.
    pub message: String,
    /// Optional structured metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

/// Diagnostic report for a resolved run plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DoctorReport {
    /// Agent name.
    pub agent_name: String,
    /// Ordered profiles applied to the inspected plan.
    pub profiles: Vec<String>,
    /// Overall status.
    pub status: DoctorStatus,
    /// Checks.
    pub checks: Vec<DoctorCheck>,
}

/// Inspect a resolved run plan without mutating the environment.
pub fn doctor_plan(plan: &RunPlan) -> DoctorReport {
    let mut checks = Vec::new();
    checks.push(check_adapter_descriptor(plan));
    if let Some(check) = check_model_binding(plan) {
        checks.push(check);
    }
    checks.push(check_resolution(plan));
    checks.extend(check_runtime_execution_surface(plan));
    checks.push(check_environment_context(plan));
    checks.extend(check_capability_routes(plan));
    checks.extend(check_requirements(plan));
    let status = checks.iter().fold(DoctorStatus::Pass, |status, check| {
        worst(status, check.status)
    });
    DoctorReport {
        agent_name: plan.agent_name.clone(),
        profiles: plan.profiles.clone(),
        status,
        checks,
    }
}

/// Diagnose an effective config using the same planning path as execution.
pub fn doctor_effective_config(effective_config: EffectiveConfig) -> Result<DoctorReport> {
    let agent_name = effective_config.agent_name.clone();
    let profiles = effective_config.profiles.clone();
    match resolve_run_plan_from_effective_config(effective_config) {
        Ok(plan) => Ok(doctor_plan(&plan)),
        Err(FabricError::ModelCompatibility {
            adapter_id,
            role,
            provider,
            wire_protocol,
            reason,
        }) => {
            let mut metadata = BTreeMap::new();
            metadata.insert("adapter_id".to_string(), Value::String(adapter_id));
            metadata.insert("role".to_string(), Value::String(role));
            if let Some(provider) = provider {
                metadata.insert("provider".to_string(), Value::String(provider));
            }
            if let Some(wire_protocol) = wire_protocol {
                metadata.insert("wire_protocol".to_string(), Value::String(wire_protocol));
            }
            Ok(DoctorReport {
                agent_name,
                profiles,
                status: DoctorStatus::Fail,
                checks: vec![check_with_metadata(
                    "model.compatibility",
                    DoctorStatus::Fail,
                    reason,
                    metadata,
                )],
            })
        }
        Err(error) => Err(error),
    }
}

fn check_model_binding(plan: &RunPlan) -> Option<DoctorCheck> {
    let binding = plan.model_binding.as_ref()?;
    let mut metadata = BTreeMap::new();
    metadata.insert("role".to_string(), Value::String(binding.role.clone()));
    metadata.insert(
        "provider".to_string(),
        Value::String(binding.provider.clone()),
    );
    metadata.insert(
        "model_id".to_string(),
        Value::String(binding.model_id.clone()),
    );
    metadata.insert(
        "wire_protocol".to_string(),
        Value::String(binding.wire_protocol.clone()),
    );
    metadata.insert(
        "endpoint".to_string(),
        Value::String(
            match binding.endpoint_ref {
                ModelEndpointRef::ProviderDefault => "provider_default",
                ModelEndpointRef::Configured { .. } => "configured",
            }
            .to_string(),
        ),
    );
    metadata.insert(
        "credential".to_string(),
        Value::String(
            match binding.credential_ref {
                ModelCredentialRef::HarnessManaged => "harness_managed",
                ModelCredentialRef::Environment { .. } => "environment",
            }
            .to_string(),
        ),
    );
    Some(check_with_metadata(
        "model.compatibility",
        DoctorStatus::Pass,
        format!(
            "resolved model role `{}` through `{}`",
            binding.role, binding.wire_protocol
        ),
        metadata,
    ))
}

fn check_adapter_descriptor(plan: &RunPlan) -> DoctorCheck {
    if let Some(adapter) = &plan.adapter_descriptor {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "source".to_string(),
            Value::String(format!("{:?}", adapter.source).to_lowercase()),
        );
        return check_with_metadata(
            "adapter_descriptor",
            DoctorStatus::Pass,
            format!(
                "resolved {} adapter descriptor `{}`",
                format!("{:?}", adapter.source).to_lowercase(),
                adapter.descriptor.adapter_id
            ),
            metadata,
        );
    }
    check(
        "adapter_descriptor",
        DoctorStatus::Fail,
        "adapter id was configured but no adapter descriptor was resolved",
    )
}

fn check_resolution(plan: &RunPlan) -> DoctorCheck {
    let Some(resolution) = plan.resolution else {
        return check(
            "resolution",
            DoctorStatus::Warn,
            "no resolution strategy selected",
        );
    };
    let status = match resolution {
        ResolutionStrategy::Preinstalled | ResolutionStrategy::ImageProvided => DoctorStatus::Pass,
        ResolutionStrategy::PipUv
        | ResolutionStrategy::Npm
        | ResolutionStrategy::Source
        | ResolutionStrategy::Service
        | ResolutionStrategy::NativePlugin => DoctorStatus::Warn,
    };
    let message = match status {
        DoctorStatus::Pass => format!("selected resolution strategy `{resolution:?}`"),
        DoctorStatus::Warn if matches!(resolution, ResolutionStrategy::Service) => {
            "selected resolution strategy `service` is modeled but not implemented by Fabric runtime execution".to_string()
        }
        DoctorStatus::Warn => format!(
            "selected resolution strategy `{resolution:?}` is declared but not executed by this POC"
        ),
        DoctorStatus::Fail => unreachable!("resolution check never fails directly"),
    };
    check("resolution", status, message)
}

fn check_runtime_execution_surface(plan: &RunPlan) -> Vec<DoctorCheck> {
    let mut checks = Vec::new();
    let Some(adapter) = &plan.adapter_descriptor else {
        return checks;
    };
    match adapter.descriptor.adapter_kind {
        AdapterKind::Http | AdapterKind::NativePlugin => checks.push(check(
            "runtime.adapter",
            DoctorStatus::Warn,
            format!(
                "`{}` adapter runtime dispatch is not implemented",
                adapter_kind_name(adapter.descriptor.adapter_kind)
            ),
        )),
        AdapterKind::Process | AdapterKind::Python => {}
    }
    checks
}

fn check_environment_context(plan: &RunPlan) -> DoctorCheck {
    let Some(environment) = &plan.environment_plan else {
        return check(
            "environment",
            DoctorStatus::Warn,
            "no environment configured; using caller-owned local context",
        );
    };
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "provider".to_string(),
        Value::String(environment.provider.clone()),
    );
    metadata.insert(
        "control_location".to_string(),
        Value::String(control_location_name(environment.control_location).to_string()),
    );
    metadata.insert(
        "ownership".to_string(),
        Value::String(ownership_name(environment.ownership).to_string()),
    );
    check_with_metadata(
        "environment",
        DoctorStatus::Pass,
        format!(
            "resolved {} {} environment context",
            ownership_name(environment.ownership),
            environment.provider
        ),
        metadata,
    )
}

fn check_capability_routes(plan: &RunPlan) -> Vec<DoctorCheck> {
    plan.capability_plan
        .routes
        .iter()
        .filter(|route| route.target == CapabilityTarget::Unsupported)
        .map(|route| {
            let status = if route.kind == CapabilityKind::Tools {
                DoctorStatus::Fail
            } else {
                DoctorStatus::Warn
            };
            check(
                "capability.unsupported",
                status,
                format!(
                    "{:?} capability `{}` is configured but not executable: {}",
                    route.kind, route.name, route.reason
                ),
            )
        })
        .collect()
}

fn check_requirements(plan: &RunPlan) -> Vec<DoctorCheck> {
    match plan.resolution {
        Some(ResolutionStrategy::ImageProvided) => return check_image_provided_requirements(plan),
        Some(
            ResolutionStrategy::PipUv
            | ResolutionStrategy::Npm
            | ResolutionStrategy::Source
            | ResolutionStrategy::Service
            | ResolutionStrategy::NativePlugin,
        ) => {
            return vec![check(
                "requirements.resolution",
                DoctorStatus::Warn,
                format!(
                    "requirements are declared for `{}` but this POC does not execute that resolution strategy",
                    resolution_name(plan.resolution)
                ),
            )];
        }
        Some(ResolutionStrategy::Preinstalled) | None => {
            if let Some(check) = non_local_preinstalled_check(plan) {
                return vec![check];
            }
        }
    }
    let Some(adapter) = &plan.adapter_descriptor else {
        return Vec::new();
    };
    let descriptor = &adapter.descriptor;
    let mut checks = Vec::new();
    for binary in &descriptor.requirements.binaries {
        let requirement = binary_requirement(plan, binary);
        checks.push(if requirement.available {
            check(
                "requirement.binary",
                DoctorStatus::Pass,
                requirement.pass_message,
            )
        } else {
            check(
                "requirement.binary",
                DoctorStatus::Fail,
                requirement.fail_message,
            )
        });
    }
    for env in &descriptor.requirements.env {
        checks.push(if std::env::var_os(env).is_some() {
            check(
                "requirement.env",
                DoctorStatus::Pass,
                format!("environment variable `{env}` is set"),
            )
        } else {
            check(
                "requirement.env",
                DoctorStatus::Fail,
                format!("environment variable `{env}` is not set"),
            )
        });
    }
    for file in &descriptor.requirements.files {
        let path = resolve_path(&adapter.root, file);
        checks.push(if path.exists() {
            check(
                "requirement.file",
                DoctorStatus::Pass,
                format!("file `{}` exists", path.display()),
            )
        } else {
            check(
                "requirement.file",
                DoctorStatus::Fail,
                format!("file `{}` does not exist", path.display()),
            )
        });
    }
    for service in &descriptor.requirements.services {
        checks.push(check(
            "requirement.service",
            DoctorStatus::Warn,
            format!("service requirement `{service}` is declared but not probed by this POC"),
        ));
    }
    for hook in &descriptor.requirements.plugin_hooks {
        checks.push(check(
            "requirement.plugin_hook",
            DoctorStatus::Warn,
            format!("plugin hook `{hook}` is declared but not probed by this POC"),
        ));
    }
    checks
}

fn check_image_provided_requirements(plan: &RunPlan) -> Vec<DoctorCheck> {
    let image = plan
        .environment_plan
        .as_ref()
        .and_then(|environment| environment.settings.get("image"))
        .and_then(Value::as_str);
    let Some(image) = image else {
        return vec![check(
            "requirements.image",
            DoctorStatus::Warn,
            "`image_provided` selected but no environment image is configured",
        )];
    };
    vec![check(
        "requirements.image",
        DoctorStatus::Pass,
        format!("requirements are expected from environment image `{image}`"),
    )]
}

fn non_local_preinstalled_check(plan: &RunPlan) -> Option<DoctorCheck> {
    let environment = plan.environment_plan.as_ref()?;
    if environment.provider == "local" {
        return None;
    }
    Some(check(
        "requirements.environment",
        DoctorStatus::Warn,
        format!(
            "`preinstalled` requirements are expected inside `{}` and are not probed by this POC",
            environment.provider
        ),
    ))
}

fn control_location_name(control_location: ControlLocation) -> &'static str {
    match control_location {
        ControlLocation::ExternalControl => "external_control",
        ControlLocation::InEnvControl => "in_env_control",
    }
}

fn ownership_name(ownership: EnvironmentOwnership) -> &'static str {
    match ownership {
        EnvironmentOwnership::CallerOwned => "caller_owned",
        EnvironmentOwnership::FabricOwned => "fabric_owned",
    }
}

fn resolution_name(resolution: Option<ResolutionStrategy>) -> &'static str {
    match resolution {
        Some(ResolutionStrategy::Preinstalled) => "preinstalled",
        Some(ResolutionStrategy::ImageProvided) => "image_provided",
        Some(ResolutionStrategy::PipUv) => "pip_uv",
        Some(ResolutionStrategy::Npm) => "npm",
        Some(ResolutionStrategy::Source) => "source",
        Some(ResolutionStrategy::Service) => "service",
        Some(ResolutionStrategy::NativePlugin) => "native_plugin",
        None => "unspecified",
    }
}

fn adapter_kind_name(adapter_kind: AdapterKind) -> &'static str {
    match adapter_kind {
        AdapterKind::Process => "process",
        AdapterKind::Http => "http",
        AdapterKind::Python => "python",
        AdapterKind::NativePlugin => "native_plugin",
    }
}

fn command_available(binary: &str) -> bool {
    let path = Path::new(binary);
    if path.components().count() > 1 {
        return path.is_file();
    }
    let Some(paths) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&paths).any(|dir| dir.join(binary).is_file())
}

struct BinaryRequirement {
    available: bool,
    pass_message: String,
    fail_message: String,
}

fn binary_requirement(plan: &RunPlan, binary: &str) -> BinaryRequirement {
    let setting_key = binary_command_setting_key(binary);
    if let Some(Value::String(command)) = plan.config.harness.settings.get(&setting_key) {
        let command_path = resolve_command(&plan.config_root, command);
        let display = command_path.to_string_lossy().into_owned();
        return BinaryRequirement {
            available: command_available(&display),
            pass_message: format!(
                "binary `{binary}` resolved from harness setting `{setting_key}` as `{display}`"
            ),
            fail_message: format!(
                "binary `{binary}` resolved from harness setting `{setting_key}` as `{display}` but was not found"
            ),
        };
    }
    BinaryRequirement {
        available: command_available(binary),
        pass_message: format!("binary `{binary}` is available on PATH"),
        fail_message: format!("binary `{binary}` was not found on PATH"),
    }
}

fn binary_command_setting_key(binary: &str) -> String {
    let normalized: String = binary
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '_'
            }
        })
        .collect();
    format!("{normalized}_command")
}

fn resolve_command(root: &Path, command: &str) -> PathBuf {
    let path = Path::new(command);
    if path.is_absolute() || path.components().count() == 1 {
        return path.to_path_buf();
    }
    root.join(path)
}

fn resolve_path(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    root.join(path)
}

fn check(name: impl Into<String>, status: DoctorStatus, message: impl Into<String>) -> DoctorCheck {
    DoctorCheck {
        name: name.into(),
        status,
        message: message.into(),
        metadata: BTreeMap::new(),
    }
}

fn check_with_metadata(
    name: impl Into<String>,
    status: DoctorStatus,
    message: impl Into<String>,
    metadata: BTreeMap<String, Value>,
) -> DoctorCheck {
    DoctorCheck {
        name: name.into(),
        status,
        message: message.into(),
        metadata,
    }
}

fn worst(left: DoctorStatus, right: DoctorStatus) -> DoctorStatus {
    match (left, right) {
        (DoctorStatus::Fail, _) | (_, DoctorStatus::Fail) => DoctorStatus::Fail,
        (DoctorStatus::Warn, _) | (_, DoctorStatus::Warn) => DoctorStatus::Warn,
        _ => DoctorStatus::Pass,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::Value;

    use super::*;
    use crate::config::{
        AdapterKind, CapabilityKind, CapabilityRoute, CapabilityTarget, FabricConfig,
        ResolutionStrategy, ResolveContext, resolve_effective_config_from_config, resolve_run_plan,
        resolve_run_plan_from_effective_config,
    };

    fn file_config_agent_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/file-config-agent")
    }

    fn claude_effective_config(model: &str) -> EffectiveConfig {
        let config: FabricConfig = serde_yaml::from_str(&format!(
            r#"
schema_version: fabric.agent/v1alpha1
metadata:
  name: doctor-model-test
harness:
  adapter_id: nvidia.fabric.claude
models:
  default:
{model}
runtime: {{}}
"#
        ))
        .expect("Claude config");
        resolve_effective_config_from_config(config, &[], ResolveContext::from_agent_root("."))
            .expect("effective config")
    }

    #[test]
    fn doctor_reports_the_same_resolved_model_binding_as_plan() {
        let effective = claude_effective_config(
            "    provider: anthropic\n    model: anthropic/claude-sonnet-4-5",
        );
        let plan = resolve_run_plan_from_effective_config(effective.clone()).expect("run plan");

        let report = doctor_effective_config(effective).expect("doctor report");

        let binding = plan.model_binding.expect("model binding");
        let check = report
            .checks
            .iter()
            .find(|check| check.name == "model.compatibility")
            .expect("model compatibility check");
        assert_eq!(check.status, DoctorStatus::Pass);
        assert_eq!(check.metadata["provider"], binding.provider);
        assert_eq!(check.metadata["model_id"], binding.model_id);
        assert_eq!(check.metadata["wire_protocol"], binding.wire_protocol);
    }

    #[test]
    fn doctor_turns_planning_model_incompatibility_into_structured_failure() {
        let effective = claude_effective_config(
            "    provider: private-cloud\n    model: private-cloud/claude-sonnet-4-5",
        );

        let report = doctor_effective_config(effective).expect("doctor report");

        assert_eq!(report.status, DoctorStatus::Fail);
        assert_eq!(report.checks.len(), 1);
        let check = &report.checks[0];
        assert_eq!(check.name, "model.compatibility");
        assert_eq!(check.status, DoctorStatus::Fail);
        assert_eq!(check.metadata["provider"], "private-cloud");
        assert!(check.message.contains("custom endpoint"));
    }

    #[test]
    fn image_provided_uses_environment_image_instead_of_host_requirements() {
        let mut plan = resolve_run_plan(file_config_agent_dir(), None).expect("run plan");
        plan.resolution = Some(ResolutionStrategy::ImageProvided);
        plan.environment_plan
            .as_mut()
            .expect("environment")
            .settings
            .insert(
                "image".to_string(),
                Value::String("fabric-hermes:latest".to_string()),
            );

        let report = doctor_plan(&plan);

        assert_eq!(report.status, DoctorStatus::Pass);
        assert!(report.checks.iter().any(|check| {
            check.name == "requirements.image" && check.message.contains("fabric-hermes:latest")
        }));
        assert!(
            !report
                .checks
                .iter()
                .any(|check| check.name == "requirement.binary")
        );
    }

    #[test]
    fn preinstalled_non_local_environment_does_not_probe_host_requirements() {
        let plan =
            resolve_run_plan(file_config_agent_dir(), Some("env_opensandbox")).expect("run plan");

        let report = doctor_plan(&plan);

        assert_eq!(report.status, DoctorStatus::Warn);
        let report_json = serde_json::to_value(&report).expect("doctor report json");
        assert!(report_json.get("profile").is_none());
        assert_eq!(
            report_json["profiles"],
            serde_json::json!(["env_opensandbox"])
        );
        assert!(report.checks.iter().any(|check| {
            check.name == "requirements.environment" && check.message.contains("opensandbox")
        }));
        assert!(
            !report
                .checks
                .iter()
                .any(|check| check.name == "requirement.binary")
        );
    }

    #[test]
    fn binary_requirement_can_use_harness_command_setting() {
        let mut plan = resolve_run_plan(file_config_agent_dir(), Some("codex")).expect("run plan");
        plan.adapter_descriptor
            .as_mut()
            .expect("adapter descriptor")
            .descriptor
            .requirements
            .binaries = vec!["fabric-doctor-test".to_string()];
        plan.config.harness.settings.insert(
            "fabric_doctor_test_command".to_string(),
            Value::String(
                std::env::current_exe()
                    .expect("current executable")
                    .to_string_lossy()
                    .into_owned(),
            ),
        );

        let report = doctor_plan(&plan);

        assert!(report.checks.iter().any(|check| {
            check.name == "requirement.binary"
                && check.status == DoctorStatus::Pass
                && check.message.contains("fabric_doctor_test_command")
        }));
    }

    #[test]
    fn doctor_reports_http_execution_as_modeled_not_implemented() {
        let mut plan = resolve_run_plan(file_config_agent_dir(), None).expect("run plan");
        plan.resolution = Some(ResolutionStrategy::Service);
        plan.adapter_descriptor
            .as_mut()
            .expect("adapter descriptor")
            .descriptor
            .adapter_kind = AdapterKind::Http;

        let report = doctor_plan(&plan);

        assert_eq!(report.status, DoctorStatus::Warn);
        assert!(report.checks.iter().any(|check| {
            check.name == "runtime.adapter"
                && check.status == DoctorStatus::Warn
                && check
                    .message
                    .contains("runtime dispatch is not implemented")
                && check.message.contains("http")
        }));
        assert!(report.checks.iter().any(|check| {
            check.name == "resolution"
                && check.status == DoctorStatus::Warn
                && check.message.contains("modeled but not implemented")
                && check.message.contains("service")
        }));
    }

    #[test]
    fn unsupported_blocked_tools_fail_doctor() {
        let mut plan = resolve_run_plan(file_config_agent_dir(), None).expect("run plan");
        plan.capability_plan.routes.push(CapabilityRoute {
            kind: CapabilityKind::Tools,
            name: "blocked".to_string(),
            target: CapabilityTarget::Unsupported,
            reason: "adapter does not accept tools".to_string(),
        });

        let report = doctor_plan(&plan);

        assert_eq!(report.status, DoctorStatus::Fail);
        assert!(report.checks.iter().any(|check| {
            check.name == "capability.unsupported"
                && check.status == DoctorStatus::Fail
                && check.message.contains("adapter does not accept tools")
        }));
    }
}
