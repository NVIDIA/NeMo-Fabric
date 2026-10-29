// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Selected software evidence. Optional evidence never controls invocation success.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Maximum serialized size of one software report (64 KiB).
pub const MAX_REPORT_BYTES: usize = 64 * 1024;
/// Maximum installation contexts in one report.
pub const MAX_ENVIRONMENTS: usize = 16;
/// Maximum selected entries across one report.
pub const MAX_ENTRIES: usize = 128;

/// Availability of one expected observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceStatus {
    /// A value was captured.
    Available,
    /// The expected source could not supply a value.
    Unavailable,
    /// This collector has no supported source.
    Unsupported,
    /// Collection was disabled or omitted.
    NotCollected,
    /// The observation does not apply.
    NotApplicable,
}

/// Availability of an installation inventory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum InventoryStatus {
    /// The selected report was received.
    Available,
    /// Expected evidence was unavailable.
    Unavailable,
    /// This boundary has no supported reporter.
    Unsupported,
    /// Collection did not run.
    NotCollected,
}

/// Whether evidence was observed by its source or supplied by an owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceOrigin {
    /// Read directly by the named observer.
    Observed,
    /// Supplied by a component or environment owner.
    Reported,
}

/// A full version or an explicit reason why it could not be captured.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(transform = version_observation_schema)]
pub struct VersionObservation {
    /// Availability of this value.
    pub status: EvidenceStatus,
    /// Full value, present only when available; never shortened or hashed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    /// Observer and method, such as `adapter:importlib.metadata`.
    #[schemars(schema_with = "nonblank_string_schema")]
    pub source: String,
    /// How the named source obtained this observation.
    pub origin: EvidenceOrigin,
    /// Original RFC 3339 UTC capture time; forwarding does not change it.
    #[schemars(schema_with = "utc_timestamp_schema")]
    pub observed_at: String,
    /// Required when a value is missing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl VersionObservation {
    fn valid(&self) -> bool {
        let nonblank = |s: &str| !s.trim().is_empty();
        nonblank(&self.source)
            && utc_timestamp(&self.observed_at)
            && if self.status == EvidenceStatus::Available {
                self.value.as_deref().is_some_and(nonblank) && self.reason.is_none()
            } else {
                self.value.is_none() && self.reason.as_deref().is_some_and(nonblank)
            }
    }
}

/// One selected component in an installation inventory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SoftwareEntry {
    /// Component category, such as `fabric_sdk`, `adapter`, or `harness`.
    #[schemars(schema_with = "nonblank_string_schema")]
    pub role: String,
    /// `python_distribution`, `npm_package`, `executable`, or `component`.
    #[schemars(schema_with = "nonblank_string_schema")]
    pub kind: String,
    /// Distribution, executable, or component name.
    #[schemars(schema_with = "nonblank_string_schema")]
    pub name: String,
    /// Full version observation.
    pub version: VersionObservation,
}

/// An observed installation/resolution context, never inferred from equal versions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(transform = software_environment_schema)]
pub struct SoftwareEnvironment {
    /// Opaque result-local join key; has no machine or role semantics.
    #[schemars(schema_with = "nonblank_string_schema")]
    pub context_ref: String,
    /// Components using this installation, such as `caller_sdk` or `adapter`.
    #[schemars(schema_with = "nonempty_strings_schema")]
    pub participants: Vec<String>,
    /// Report availability. `not_applicable` is not an inventory status.
    pub status: InventoryStatus,
    /// Why coverage is unavailable or incomplete.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Every selected target is represented, including unavailable versions.
    pub complete: bool,
    /// Selected software only; this is not a dependency scan.
    pub entries: Vec<SoftwareEntry>,
}

/// Existing sandbox/provider report, without any new service discovery.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SandboxSoftwareReport {
    /// Installation contexts served by this sandbox.
    #[schemars(schema_with = "nonempty_strings_schema")]
    pub context_refs: Vec<String>,
    /// Provider name.
    pub provider: VersionObservation,
    /// Gateway/service version, not a guest package version.
    pub service_version: VersionObservation,
}

/// Existing environment-owner report of a container engine server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ContainerSoftwareReport {
    /// Installation contexts served by this engine.
    #[schemars(schema_with = "nonempty_strings_schema")]
    pub context_refs: Vec<String>,
    /// Engine name.
    pub engine: VersionObservation,
    /// Server version, never a CLI or API version.
    pub engine_version: VersionObservation,
}

/// Conditional infrastructure software reports.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InfrastructureSoftware {
    /// Included only when an owner supplied a sandbox report.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox: Option<SandboxSoftwareReport>,
    /// Included only when an owner supplied an engine report.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<ContainerSoftwareReport>,
}

/// Selected software evidence included inline in a RunResult.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExecutionEvidence {
    /// One inventory per observed installation/resolution context.
    #[schemars(length(max = 16))]
    pub software_environments: Vec<SoftwareEnvironment>,
    /// Existing provider/engine reports, omitted when none were supplied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub infrastructure: Option<InfrastructureSoftware>,
}

impl ExecutionEvidence {
    /// Check coverage/value invariants and numeric report bounds.
    pub fn valid(&self) -> bool {
        let refs: std::collections::BTreeSet<_> = self
            .software_environments
            .iter()
            .map(|env| env.context_ref.as_str())
            .collect();
        let referenced = |keys: &[String]| {
            !keys.is_empty() && keys.iter().all(|key| refs.contains(key.as_str()))
        };
        self.software_environments.len() <= MAX_ENVIRONMENTS
            && refs.len() == self.software_environments.len()
            && self
                .software_environments
                .iter()
                .map(|env| env.entries.len())
                .sum::<usize>()
                <= MAX_ENTRIES
            && self.software_environments.iter().all(|env| {
                !env.context_ref.trim().is_empty()
                    && !env.participants.is_empty()
                    && env.participants.iter().all(|p| !p.trim().is_empty())
                    && ((env.status == InventoryStatus::Available && env.complete)
                        || env.reason.as_deref().is_some_and(|s| !s.trim().is_empty()))
                    && (env.status == InventoryStatus::Available
                        || (!env.complete && env.entries.is_empty()))
                    && env.entries.iter().all(|entry| {
                        !entry.role.trim().is_empty()
                            && !entry.kind.trim().is_empty()
                            && !entry.name.trim().is_empty()
                            && entry.version.valid()
                    })
            })
            && self.infrastructure.as_ref().is_none_or(|i| {
                i.sandbox.as_ref().is_none_or(|s| {
                    referenced(&s.context_refs) && s.provider.valid() && s.service_version.valid()
                }) && i.container.as_ref().is_none_or(|c| {
                    referenced(&c.context_refs) && c.engine.valid() && c.engine_version.valid()
                })
            })
            && serde_json::to_vec(self).is_ok_and(|bytes| bytes.len() <= MAX_REPORT_BYTES)
    }

    /// Decode optional untrusted evidence without replacing the original outcome.
    /// Receivers call this even when older intermediaries omitted the report.
    pub fn receive(value: Option<serde_json::Value>, participant: &str) -> Self {
        let missing = value.is_none();
        value
            .and_then(|value| serde_json::from_value::<Self>(value).ok())
            .filter(Self::valid)
            .unwrap_or_else(|| {
                Self::gap(
                    participant,
                    if missing {
                        "report_missing"
                    } else {
                        "report_invalid_or_over_budget"
                    },
                )
            })
    }

    /// Explicit incomplete evidence at an implemented receiving boundary.
    pub fn gap(participant: &str, reason: &str) -> Self {
        Self {
            software_environments: vec![SoftwareEnvironment {
                context_ref: "env-0".into(),
                participants: vec![participant.into()],
                status: InventoryStatus::Unavailable,
                reason: Some(reason.into()),
                complete: false,
                entries: vec![],
            }],
            infrastructure: None,
        }
    }
}

fn utc_timestamp(s: &str) -> bool {
    let Some(s) = s.strip_suffix('Z').or_else(|| s.strip_suffix("+00:00")) else {
        return false;
    };
    let b = s.as_bytes();
    if !(b.len() >= 19
        && b[4] == b'-'
        && b[7] == b'-'
        && b[10] == b'T'
        && b[13] == b':'
        && b[16] == b':'
        && b[..19]
            .iter()
            .enumerate()
            .all(|(i, c)| [4, 7, 10, 13, 16].contains(&i) || c.is_ascii_digit())
        && (b.len() == 19
            || (b[19] == b'.' && b.len() > 20 && b[20..].iter().all(u8::is_ascii_digit))))
    {
        return false;
    }
    let number = |range: std::ops::Range<usize>| {
        b[range]
            .iter()
            .fold(0u32, |n, c| n * 10 + u32::from(c - b'0'))
    };
    let year = number(0..4);
    let month = number(5..7);
    let day = number(8..10);
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = match month {
        2 => {
            if leap {
                29
            } else {
                28
            }
        }
        4 | 6 | 9 | 11 => 30,
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        _ => 0,
    };
    year > 0
        && day > 0
        && day <= days
        && number(11..13) < 24
        && number(14..16) < 60
        && number(17..19) < 60
}

/// Default-on collector control and explicit reports forwarded in existing dispatch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProvenanceConfig {
    /// False skips host and guest collector work and omits result execution evidence.
    #[serde(default = "enabled_by_default")]
    pub enabled: bool,
    /// Caller/owner snapshots, captured at initialization and forwarded unchanged.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_report"
    )]
    pub reports: Option<ExecutionEvidence>,
}

pub(crate) fn deserialize_optional_report<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<ExecutionEvidence>, D::Error> {
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(value.map(|value| ExecutionEvidence::receive(Some(value), "reporter")))
}

fn enabled_by_default() -> bool {
    true
}
impl Default for ProvenanceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            reports: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn receiving_optional_evidence_preserves_gaps_and_rejects_malformed_values() {
        for value in [
            None,
            Some(json!({"software_environments": "broken"})),
            Some(json!({})),
        ] {
            let report = ExecutionEvidence::receive(value, "guest_sdk");
            assert!(!report.software_environments[0].complete);
            assert!(report.valid());
        }
        let report = ExecutionEvidence::gap("adapter", "report_missing");
        assert_eq!(
            ExecutionEvidence::receive(Some(serde_json::to_value(&report).unwrap()), "adapter"),
            report
        );
    }

    #[test]
    fn golden_report_preserves_full_versions_and_original_times() {
        let value: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/provenance/report.json"
        ))
        .unwrap();
        let report = ExecutionEvidence::receive(Some(value.clone()), "adapter");
        assert!(report.valid());
        assert_eq!(serde_json::to_value(report).unwrap(), value);
        assert!(!utc_timestamp("2026-02-30T12:00:00Z"));
        assert!(!utc_timestamp("2026-10-27T99:00:00Z"));
    }

    #[test]
    fn disable_survives_serialization_and_omission_defaults_on() {
        let config: ProvenanceConfig = serde_json::from_value(json!({"enabled": false})).unwrap();
        assert_eq!(
            serde_json::to_value(&config).unwrap(),
            json!({"enabled": false})
        );
        assert!(
            serde_json::from_value::<ProvenanceConfig>(json!({}))
                .unwrap()
                .enabled
        );
    }
}

impl ExecutionEvidence {
    /// Join distinct source contexts, preserving observations and rebasing local refs.
    pub fn append(&mut self, report: Self) {
        let mut report = if report.valid() {
            report
        } else {
            Self::gap("reporter", "report_invalid_or_over_budget")
        };
        let mut refs = std::collections::BTreeMap::new();
        for env in &mut report.software_environments {
            let next = format!("env-{}", self.software_environments.len());
            refs.insert(env.context_ref.clone(), next.clone());
            env.context_ref = next;
            self.software_environments.push(env.clone());
        }
        if let Some(mut infrastructure) = report.infrastructure {
            if let Some(sandbox) = &mut infrastructure.sandbox {
                for key in &mut sandbox.context_refs {
                    *key = refs[key].clone();
                }
            }
            if let Some(container) = &mut infrastructure.container {
                for key in &mut container.context_refs {
                    *key = refs[key].clone();
                }
            }
            let existing = self.infrastructure.get_or_insert_with(Default::default);
            if (existing.sandbox.is_some() && infrastructure.sandbox.is_some())
                || (existing.container.is_some() && infrastructure.container.is_some())
            {
                *self = Self::gap("reporter", "conflicting_infrastructure_reports");
                return;
            }
            if infrastructure.sandbox.is_some() {
                existing.sandbox = infrastructure.sandbox;
            }
            if infrastructure.container.is_some() {
                existing.container = infrastructure.container;
            }
        }
        if self.software_environments.len() > MAX_ENVIRONMENTS {
            self.software_environments.truncate(MAX_ENVIRONMENTS - 1);
            let mut gap = Self::gap("additional_contexts", "report_context_limit")
                .software_environments
                .remove(0);
            gap.context_ref = format!("env-{}", MAX_ENVIRONMENTS - 1);
            self.software_environments.push(gap);
        }
        while self
            .software_environments
            .iter()
            .map(|env| env.entries.len())
            .sum::<usize>()
            > MAX_ENTRIES
            || serde_json::to_vec(self).is_ok_and(|bytes| bytes.len() > MAX_REPORT_BYTES)
        {
            let Some(env) = self.software_environments.last_mut() else {
                break;
            };
            if env.entries.pop().is_some() {
                env.complete = false;
                env.reason = Some("report_truncated_by_entry_or_byte_limit".into());
            } else if self.software_environments.len() > 1 {
                self.software_environments.pop();
                let env = self.software_environments.last_mut().unwrap();
                env.complete = false;
                env.reason = Some("additional_contexts_over_budget".into());
            } else {
                *self = Self::gap("reporter", "combined_report_over_budget");
                return;
            }
        }
        let refs: std::collections::BTreeSet<_> = self
            .software_environments
            .iter()
            .map(|e| e.context_ref.as_str())
            .collect();
        if let Some(infrastructure) = &mut self.infrastructure {
            // Never keep a dangling join after dropping an over-budget context.
            if infrastructure
                .sandbox
                .as_ref()
                .is_some_and(|s| s.context_refs.iter().any(|r| !refs.contains(r.as_str())))
            {
                infrastructure.sandbox = None;
            }
            if infrastructure
                .container
                .as_ref()
                .is_some_and(|s| s.context_refs.iter().any(|r| !refs.contains(r.as_str())))
            {
                infrastructure.container = None;
            }
        }
    }
}

/// Capture the executing compiled core once at runtime initialization.
/// No package discovery, executable command or dependency walk is performed.
pub(crate) fn core_snapshot(config: &ProvenanceConfig) -> Option<ExecutionEvidence> {
    if !config.enabled {
        return None;
    }
    let mut report = config.reports.clone().unwrap_or_default();
    report.append(ExecutionEvidence {
        software_environments: vec![SoftwareEnvironment {
            context_ref: "env-0".into(),
            participants: vec!["core".into()],
            status: InventoryStatus::Available,
            reason: None,
            complete: true,
            entries: vec![SoftwareEntry {
                role: "fabric_core".into(),
                kind: "component".into(),
                name: "nemo-fabric-core".into(),
                version: VersionObservation {
                    status: EvidenceStatus::Available,
                    value: Some(crate::version().into()),
                    source: "core:version".into(),
                    origin: EvidenceOrigin::Observed,
                    observed_at: crate::time::rfc3339_millis(std::time::SystemTime::now()),
                    reason: None,
                },
            }],
        }],
        infrastructure: None,
    });
    Some(report)
}

fn utc_timestamp_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
    schemars::json_schema!({"type": "string", "format": "date-time", "pattern": "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(?:\\.\\d+)?(?:Z|\\+00:00)$"})
}

fn nonblank_string_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
    schemars::json_schema!({"type": "string", "minLength": 1, "pattern": "\\S"})
}

fn nonempty_strings_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
    schemars::json_schema!({"type": "array", "minItems": 1,
        "items": {"type": "string", "minLength": 1, "pattern": "\\S"}})
}

fn version_observation_schema(schema: &mut schemars::Schema) {
    schema.insert("allOf".into(), serde_json::json!([{
        "if": {"properties": {"status": {"const": "available"}}, "required": ["status"]},
        "then": {"required": ["value"], "properties": {"value": {"type": "string", "minLength": 1, "pattern": "\\S"}, "reason": {"type": "null"}}},
        "else": {"required": ["reason"], "properties": {"reason": {"type": "string", "minLength": 1, "pattern": "\\S"}, "value": {"type": "null"}}}
    }]));
}

fn software_environment_schema(schema: &mut schemars::Schema) {
    schema.insert("allOf".into(), serde_json::json!([
        {"if": {"properties": {"status": {"const": "available"}}, "required": ["status"]},
         "else": {"properties": {"complete": {"const": false}, "entries": {"maxItems": 0}}}},
        {"if": {"anyOf": [ {"properties": {"status": {"not": {"const": "available"}}}, "required": ["status"]},
             {"properties": {"complete": {"const": false}}, "required": ["complete"]} ]},
         "then": {"required": ["reason"], "properties": {"reason": {"type": "string", "minLength": 1, "pattern": "\\S"}}}}
    ]));
}
