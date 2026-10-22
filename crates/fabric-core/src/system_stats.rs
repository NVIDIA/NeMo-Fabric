// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Invocation measurements collected by NeMo Fabric in the executing runtime.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{FabricError, Result};
use crate::resource_snapshot::RuntimeResourceSnapshot;

pub mod cgroup;
mod sampling;

pub use cgroup::{CgroupObservation, CgroupSnapshot, Measurement};
pub use sampling::{GaugeSummary, SampleSummary};
pub(crate) use sampling::{StatsSession, SystemReader};

/// Resource groups supported by the native invocation collector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum SystemStatsMetricGroup {
    /// Adapter host process-tree CPU, resident memory, and process count.
    #[serde(rename = "process")]
    Process,
    /// Cgroup CPU capacity, usage, and throttling.
    #[serde(rename = "cgroup.cpu")]
    CgroupCpu,
    /// Cgroup memory capacity, usage, swap, and OOM kills.
    #[serde(rename = "cgroup.memory")]
    CgroupMemory,
    /// Cgroup CPU, memory, and I/O pressure counters.
    #[serde(rename = "cgroup.pressure")]
    CgroupPressure,
}

/// Native invocation collection, independent of Relay and recording.
/// Omitted configuration collects all supported groups at the endpoints.
/// Explicit null disables invocation stats, preserving the existing final
/// `resource_snapshot` contract. Periodic sampling is independently opt-in.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SystemStatsConfig {
    /// Groups to collect and persist. Omission selects all supported groups.
    #[serde(default = "default_groups")]
    #[schemars(length(min = 1), extend("uniqueItems" = true))]
    pub collect: Vec<SystemStatsMetricGroup>,
    /// Optional bounded periodic collection; absence means endpoints only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<SystemStatsSamplingConfig>,
}

fn default_groups() -> Vec<SystemStatsMetricGroup> {
    use SystemStatsMetricGroup::*;
    vec![Process, CgroupCpu, CgroupMemory, CgroupPressure]
}

impl Default for SystemStatsConfig {
    fn default() -> Self {
        Self {
            collect: default_groups(),
            sampling: None,
        }
    }
}

pub(crate) fn default_system_stats() -> Option<SystemStatsConfig> {
    Some(SystemStatsConfig::default())
}

impl SystemStatsConfig {
    pub(crate) fn includes(&self, group: SystemStatsMetricGroup) -> bool {
        self.collect.contains(&group)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.collect.is_empty()
            || self
                .collect
                .iter()
                .enumerate()
                .any(|(i, group)| self.collect[..i].contains(group))
        {
            return Err(FabricError::InvalidConfig {
                field: "system_stats.collect".into(),
                reason:
                    "must contain at least one group, without duplicates; use null to disable stats"
                        .into(),
            });
        }
        if let Some(sampling) = &self.sampling {
            sampling.validate()?;
        }
        Ok(())
    }
}

/// Bounded periodic collection within an enabled invocation stats configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SystemStatsSamplingConfig {
    /// Target sample cadence, in milliseconds. Actual cadence is reported.
    #[serde(default = "default_interval")]
    #[schemars(range(min = 100, max = 60000))]
    pub interval_ms: u32,
    /// Maximum retained readings, including both endpoints. Intermediate
    /// readings are dropped at this bound; the final reading is preserved.
    #[serde(default = "default_max_samples")]
    #[schemars(range(min = 2, max = 100000))]
    pub max_samples: u32,
}

fn default_interval() -> u32 {
    1000
}
fn default_max_samples() -> u32 {
    10000
}

impl Default for SystemStatsSamplingConfig {
    fn default() -> Self {
        Self {
            interval_ms: default_interval(),
            max_samples: default_max_samples(),
        }
    }
}

impl SystemStatsSamplingConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        for (field, valid, reason) in [
            (
                "interval_ms",
                (100..=60000).contains(&self.interval_ms),
                "must be between 100 and 60000",
            ),
            (
                "max_samples",
                (2..=100000).contains(&self.max_samples),
                "must be between 2 and 100000",
            ),
        ] {
            if !valid {
                return Err(FabricError::InvalidConfig {
                    field: format!("system_stats.sampling.{field}"),
                    reason: reason.into(),
                });
            }
        }
        Ok(())
    }
}

/// One attempt to observe the adapter host process tree. Missing readings are
/// explicit; an unavailable observation never stands for zero resource usage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ResourceObservation {
    /// The process group was explicitly excluded from collection.
    NotCollected,
    /// The process tree was read successfully.
    Available {
        /// Timestamped cumulative counters and instantaneous gauges.
        snapshot: RuntimeResourceSnapshot,
    },
    /// The reader supports this runtime, but could not read its process tree.
    Unavailable {
        /// Why this observation could not be collected.
        reason: String,
    },
    /// The runtime's provider or operating system has no reader.
    Unsupported {
        /// Which collection capability is missing.
        reason: String,
    },
}

impl ResourceObservation {
    pub(crate) fn from_result(result: Result<RuntimeResourceSnapshot>) -> Self {
        match result {
            Ok(snapshot) => Self::Available { snapshot },
            Err(FabricError::ResourceObservationUnsupported { reason, .. }) => {
                Self::Unsupported { reason }
            }
            Err(FabricError::ResourceObservationUnavailable { reason, .. }) => {
                Self::Unavailable { reason }
            }
            Err(error) => Self::Unavailable {
                reason: error.to_string(),
            },
        }
    }

    pub(crate) fn snapshot(&self) -> Option<&RuntimeResourceSnapshot> {
        match self {
            Self::Available { snapshot } => Some(snapshot),
            Self::NotCollected | Self::Unavailable { .. } | Self::Unsupported { .. } => None,
        }
    }
}

/// System measurements around one invocation, collected inside the executing
/// runner. Process-tree observations cover its adapter host and descendants,
/// excluding the orchestrator and remote model/service processes. The separate
/// cgroup observations may include other workloads in the same cgroup.
///
/// Endpoints are instantaneous readings, not peaks or time averages. CPU time
/// is cumulative since runtime startup; its difference is only an estimate
/// because process reaping and reparenting can change which work is visible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct InvocationSystemStats {
    /// Requested groups, including selected groups unavailable on this platform.
    #[serde(default = "default_groups")]
    pub collected_groups: Vec<SystemStatsMetricGroup>,
    /// Monotonic elapsed seconds for the adapter exchange. Excludes runtime
    /// startup, endpoint collection, artifact preparation, and teardown.
    /// Optional sampling runs concurrently; its effect is not subtracted.
    #[schemars(range(min = 0.0))]
    pub elapsed_seconds: f64,
    /// Observation immediately before the adapter exchange.
    pub before: ResourceObservation,
    /// Observation immediately after the adapter exchange, before another
    /// invocation can acquire this runtime's host lock.
    pub after: ResourceObservation,
    /// Nonnegative CPU counter difference, in seconds. Absent if an endpoint
    /// is missing, the runtime identity differs, or its CPU counter decreased.
    /// It can undercount work whose processes leave the tree between reads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0))]
    pub cpu_time_seconds_estimate: Option<f64>,
    /// Before/after readings for the adapter host's cgroup. This scope may
    /// contain other workloads and overlaps the process tree; never add them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cgroup: Option<CgroupStats>,
    /// Bounded sample reductions and collection quality when sampling is enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampled: Option<SampleSummary>,
}

/// Native cgroup observations and counter deltas over an invocation. Values
/// identify the adapter host's cgroup, without claiming container exclusivity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CgroupStats {
    /// Reading immediately before invocation.
    pub before: CgroupObservation,
    /// Reading immediately after invocation.
    pub after: CgroupObservation,
    /// CPU usage delta in microseconds, or an explicit unknown reason.
    pub cpu_usage_microseconds: Measurement<u64>,
    /// CPU throttled time delta in microseconds, or an explicit unknown reason.
    pub cpu_throttled_microseconds: Measurement<u64>,
    /// OOM-kill counter delta, or an explicit unknown reason.
    pub oom_kills: Measurement<u64>,
}

impl CgroupStats {
    pub(crate) fn new(before: CgroupObservation, after: CgroupObservation) -> Self {
        Self {
            cpu_usage_microseconds: cgroup::delta(&before, &after, |s| &s.cpu_usage_microseconds),
            cpu_throttled_microseconds: cgroup::delta(&before, &after, |s| {
                &s.cpu_throttled_microseconds
            }),
            oom_kills: cgroup::delta(&before, &after, |s| &s.oom_kills),
            before,
            after,
        }
    }
}

pub(crate) fn persist_summary(
    stats: &InvocationSystemStats,
    ids: &crate::InvocationHandle,
    home: &std::path::Path,
) -> std::result::Result<crate::ArtifactRef, ()> {
    use std::io::Write;
    let temporary = home.join("system-stats-summary.json.tmp");
    let path = home.join("system-stats-summary.json");
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|_| ())?;
    serde_json::to_writer_pretty(
        &mut file,
        &serde_json::json!({
            "schema_version": "fabric.system_stats/v1alpha1",
            "runtime_id": ids.runtime_id, "invocation_id": ids.invocation_id,
            "request_id": ids.request_id, "system_stats": stats,
        }),
    )
    .map_err(|_| ())?;
    file.write_all(b"\n").map_err(|_| ())?;
    drop(file);
    std::fs::rename(&temporary, &path).map_err(|_| ())?;
    Ok(crate::ArtifactRef {
        name: "system_stats_summary".into(),
        kind: "system_stats_summary".into(),
        path,
        media_type: Some("application/json".into()),
        metadata: Default::default(),
    })
}

impl InvocationSystemStats {
    pub(crate) fn new(
        before: ResourceObservation,
        after: ResourceObservation,
        elapsed_seconds: f64,
    ) -> Self {
        let cpu_time_seconds_estimate =
            before
                .snapshot()
                .zip(after.snapshot())
                .and_then(|(before, after)| {
                    let delta = after.cpu_time_seconds - before.cpu_time_seconds;
                    (before.runtime_id == after.runtime_id
                        && before.started_at == after.started_at
                        && delta.is_finite()
                        && delta >= 0.0)
                        .then_some(delta)
                });
        Self {
            collected_groups: default_groups(),
            elapsed_seconds,
            before,
            after,
            cpu_time_seconds_estimate,
            cgroup: None,
            sampled: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(cpu: f64) -> ResourceObservation {
        ResourceObservation::Available {
            snapshot: RuntimeResourceSnapshot {
                runtime_id: "runtime-1".into(),
                started_at: "2026-10-21T00:00:00.000Z".into(),
                observed_at: "2026-10-21T00:01:00.000Z".into(),
                resident_memory_bytes: 100,
                process_count: 1,
                cpu_time_seconds: cpu,
            },
        }
    }

    #[test]
    fn cpu_estimate_excludes_runtime_history_and_does_not_clamp_counter_resets() {
        let stats = InvocationSystemStats::new(observation(12.0), observation(14.5), 3.0);
        assert_eq!(stats.cpu_time_seconds_estimate, Some(2.5));
        let unchanged = InvocationSystemStats::new(observation(12.0), observation(12.0), 3.0);
        assert_eq!(unchanged.cpu_time_seconds_estimate, Some(0.0));
        let reset = InvocationSystemStats::new(observation(12.0), observation(2.0), 3.0);
        assert_eq!(reset.cpu_time_seconds_estimate, None);
    }

    #[test]
    fn missing_or_different_runtime_observations_have_no_cpu_estimate() {
        let mut replaced = observation(14.0);
        if let ResourceObservation::Available { snapshot } = &mut replaced {
            snapshot.started_at = "2026-10-21T00:00:30.000Z".into();
        }
        for after in [
            replaced,
            ResourceObservation::Unavailable {
                reason: "host exited".into(),
            },
            ResourceObservation::Unsupported {
                reason: "no OS reader".into(),
            },
        ] {
            let stats = InvocationSystemStats::new(observation(12.0), after, 3.0);
            assert_eq!(stats.cpu_time_seconds_estimate, None);
            let json = serde_json::to_value(&stats).unwrap();
            assert!(json.get("cpu_time_seconds_estimate").is_none());
            assert_eq!(
                serde_json::from_value::<InvocationSystemStats>(json).unwrap(),
                stats
            );
        }
    }
}
