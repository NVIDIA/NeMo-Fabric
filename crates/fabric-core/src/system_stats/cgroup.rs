// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native, read-only cgroup v2 observations of the executing adapter host.
//! A cgroup is a scope, not a claim of exclusive ownership by this invocation.

// Discovery checks extend Maryam Najafian's reader in NVIDIA/NeMo-Fabric#426.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::time::SystemTime;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::SystemStatsMetricGroup;
use crate::resource_snapshot::LocalHostProcess;
use crate::time::rfc3339_millis;

/// A measured value or its availability. Unlimited is distinct from unknown
/// and is used only for limits. No missing measurement is replaced with zero.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Measurement<T> {
    /// Its metric group was explicitly excluded from collection.
    NotCollected,
    /// A value was read successfully.
    Available {
        /// Observed numeric value, with units defined by its containing field.
        value: T,
    },
    /// No limit is configured in the visible hierarchy.
    Unlimited,
    /// The source could not be read or its counter changed epoch.
    Unavailable {
        /// Safe reason the reading is unknown.
        reason: String,
    },
    /// The operating system or controller does not expose this measurement.
    Unsupported {
        /// Missing platform or controller capability.
        reason: String,
    },
}

impl<T> Measurement<T> {
    pub(crate) fn value(&self) -> Option<&T> {
        if let Self::Available { value } = self {
            Some(value)
        } else {
            None
        }
    }

    fn unavailable(reason: impl Into<String>) -> Self {
        Self::Unavailable {
            reason: reason.into(),
        }
    }
}

/// One observation of the cgroup containing the adapter host. Its counters
/// include the cgroup's descendants and may include other workloads. Limits
/// cover only visible ancestors; ancestors hidden by namespaces are unknown.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CgroupSnapshot {
    /// Fixed source identifier, `linux_cgroup_v2`.
    pub source: String,
    /// Fixed scope identifier, `adapter_host_cgroup`.
    pub scope: String,
    /// Device/inode identity. A different identity invalidates counter deltas.
    pub target: String,
    /// UTC observation timestamp.
    #[schemars(extend("format" = "date-time"))]
    pub observed_at: String,
    /// Minimum of effective cpuset size and visible ancestor CPU quotas.
    /// This is a visible upper bound, not reserved or exclusively usable CPU.
    pub visible_cpu_limit_cores: Measurement<f64>,
    /// Minimum memory.max among visible ancestors, in bytes.
    pub visible_memory_limit_bytes: Measurement<u64>,
    /// Hierarchically accounted memory, distinct from process RSS, in bytes.
    pub memory_current_bytes: Measurement<u64>,
    /// Provider high-water mark in bytes, since cgroup creation or the last
    /// external reset. Never an invocation-specific peak.
    pub memory_peak_bytes: Measurement<u64>,
    /// Current swap usage in bytes.
    pub swap_current_bytes: Measurement<u64>,
    /// Cumulative cgroup CPU time, in microseconds.
    pub cpu_usage_microseconds: Measurement<u64>,
    /// Cumulative CPU throttled time, in microseconds.
    pub cpu_throttled_microseconds: Measurement<u64>,
    /// Cumulative OOM kills in this cgroup hierarchy.
    pub oom_kills: Measurement<u64>,
    /// Cumulative pressure-stall microseconds: cpu_some, cpu_full,
    /// memory_some, memory_full, io_some, and io_full.
    pub pressure_stall_microseconds: BTreeMap<String, Measurement<u64>>,
}

/// A cgroup reading, or why the correctly targeted scope was unavailable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CgroupObservation {
    /// All cgroup groups were explicitly excluded from collection.
    NotCollected,
    /// The adapter host's cgroup was located and read.
    Available {
        /// Snapshot of the resolved cgroup identity.
        snapshot: CgroupSnapshot,
    },
    /// No cgroup v2 reader is supported for this platform/provider.
    Unsupported {
        /// Missing platform or provider capability.
        reason: String,
    },
    /// The target moved, exited, or could not be resolved safely.
    Unavailable {
        /// Safe reason targeting or collection failed.
        reason: String,
    },
}

impl CgroupObservation {
    pub(crate) fn snapshot(&self) -> Option<&CgroupSnapshot> {
        if let Self::Available { snapshot } = self {
            Some(snapshot)
        } else {
            None
        }
    }
}

pub(crate) fn observe(
    host: Option<&LocalHostProcess>,
    local: bool,
    groups: &[SystemStatsMetricGroup],
) -> CgroupObservation {
    if !groups.iter().any(|g| *g != SystemStatsMetricGroup::Process) {
        return CgroupObservation::NotCollected;
    }
    if !local || !cfg!(target_os = "linux") {
        return CgroupObservation::Unsupported {
            reason: "native cgroup v2 reader requires local Linux execution".into(),
        };
    }
    let Some(host) = host.filter(|host| host.is_current()) else {
        return CgroupObservation::Unavailable {
            reason: "adapter host is not active".into(),
        };
    };
    match read_host(Path::new("/"), host.pid(), groups) {
        Ok(snapshot) if host.is_current() => CgroupObservation::Available { snapshot },
        Ok(_) => CgroupObservation::Unavailable {
            reason: "adapter host changed during collection".into(),
        },
        Err(reason) => CgroupObservation::Unavailable { reason },
    }
}

fn read_text(path: &Path, limit: u64) -> Result<String, String> {
    let mut text = String::new();
    File::open(path)
        .and_then(|file| file.take(limit + 1).read_to_string(&mut text))
        .map_err(|_| "source is missing, unreadable, or not UTF-8".to_string())?;
    if text.len() as u64 > limit {
        return Err("source exceeds collection bound".into());
    }
    Ok(text)
}

fn membership(text: &str) -> Result<&Path, String> {
    let path = text
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .ok_or("cgroup v2 membership is unavailable")?;
    let path = Path::new(path);
    if !path.is_absolute()
        || path
            .components()
            .any(|c| !matches!(c, Component::RootDir | Component::Normal(_)))
    {
        return Err("cgroup membership is outside the visible hierarchy".into());
    }
    Ok(path)
}

fn unescape_mount(text: &str) -> PathBuf {
    PathBuf::from(
        text.replace("\\040", " ")
            .replace("\\011", "\t")
            .replace("\\012", "\n")
            .replace("\\134", "\\"),
    )
}

fn locate(root: &Path, member: &Path, mountinfo: &str) -> Result<(PathBuf, PathBuf), String> {
    let mut found = None;
    for line in mountinfo.lines() {
        let Some((prefix, suffix)) = line.split_once(" - ") else {
            continue;
        };
        if suffix.split_whitespace().next() != Some("cgroup2") {
            continue;
        }
        let fields: Vec<_> = prefix.split_whitespace().collect();
        if fields.len() < 5 {
            continue;
        }
        let mount_root = unescape_mount(fields[3]);
        let mount = unescape_mount(fields[4]);
        if !mount.is_absolute()
            || mount
                .components()
                .any(|c| matches!(c, Component::ParentDir))
        {
            continue;
        }
        if let Ok(relative) = member.strip_prefix(&mount_root) {
            let visible_root = root.join(mount.strip_prefix("/").unwrap());
            found = Some((visible_root.join(relative), visible_root));
            break;
        }
    }
    found.ok_or_else(|| "cgroup v2 mount does not expose the adapter host's group".into())
}

#[cfg(unix)]
fn identity(path: &Path) -> Result<String, String> {
    use std::os::unix::fs::MetadataExt;
    let metadata = std::fs::metadata(path).map_err(|_| "cgroup identity is unreadable")?;
    Ok(format!("{}:{}", metadata.dev(), metadata.ino()))
}
#[cfg(not(unix))]
fn identity(_: &Path) -> Result<String, String> {
    Err("cgroup identity requires Unix".into())
}

fn missing_source<T>(path: &Path) -> Option<Measurement<T>> {
    match std::fs::metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Some(Measurement::Unsupported {
                reason: "kernel or controller does not expose this measurement".into(),
            })
        }
        _ => None,
    }
}

fn number(path: &Path) -> Measurement<u64> {
    if let Some(missing) = missing_source(path) {
        return missing;
    }
    match read_text(path, 4096).and_then(|text| {
        text.trim()
            .parse()
            .map_err(|_| "invalid unsigned counter".into())
    }) {
        Ok(value) => Measurement::Available { value },
        Err(reason) => Measurement::unavailable(reason),
    }
}

fn counter(path: &Path, key: &str) -> Measurement<u64> {
    if let Some(missing) = missing_source(path) {
        return missing;
    }
    let parsed = read_text(path, 16384).and_then(|text| {
        text.lines()
            .find_map(|line| {
                let (name, value) = line.split_once(' ')?;
                (name == key)
                    .then(|| value.trim().parse::<u64>().ok())
                    .flatten()
            })
            .ok_or_else(|| format!("counter {key} is unavailable"))
    });
    match parsed {
        Ok(value) => Measurement::Available { value },
        Err(reason) => Measurement::unavailable(reason),
    }
}

fn cpus(text: &str) -> Result<u64, String> {
    let mut total = 0_u64;
    let mut previous_end = None;
    for range in text.trim().split(',') {
        let (start, end) = range.split_once('-').unwrap_or((range, range));
        let start: u32 = start.parse().map_err(|_| "invalid cpuset")?;
        let end: u32 = end.parse().map_err(|_| "invalid cpuset")?;
        if end < start || previous_end.is_some_and(|previous| start <= previous) {
            return Err("invalid cpuset ordering".into());
        }
        total += u64::from(end) - u64::from(start) + 1;
        previous_end = Some(end);
    }
    Ok(total)
}

fn limits(
    directory: &Path,
    visible_root: &Path,
    cpu_selected: bool,
    memory_selected: bool,
) -> (Measurement<f64>, Measurement<u64>) {
    let mut cpu = Ok(None::<f64>);
    let mut cpu_source_found = false;
    let cpuset = directory.join("cpuset.cpus.effective");
    if cpu_selected
        && !matches!(std::fs::metadata(&cpuset), Err(error) if error.kind() == std::io::ErrorKind::NotFound)
    {
        cpu = read_text(&cpuset, 16384)
            .and_then(|text| cpus(&text))
            .map(|count| Some(count as f64));
    }
    let mut memory = Ok(None::<u64>);
    let mut memory_source_found = false;
    let mut current = directory;
    for depth in 0..64 {
        for (name, cpu_file) in [("cpu.max", true), ("memory.max", false)] {
            if (cpu_file && !cpu_selected) || (!cpu_file && !memory_selected) {
                continue;
            }
            let file = current.join(name);
            if matches!(std::fs::metadata(&file), Err(error) if error.kind() == std::io::ErrorKind::NotFound)
            {
                continue;
            }
            let result = read_text(&file, 4096);
            if cpu_file {
                cpu_source_found = true;
                match result.and_then(|text| {
                    let values: Vec<_> = text.split_whitespace().collect();
                    if values.len() != 2 {
                        return Err("invalid cpu.max".into());
                    }
                    let period = values[1]
                        .parse::<u64>()
                        .ok()
                        .filter(|p| *p > 0)
                        .ok_or("invalid CPU period")?;
                    if values[0] == "max" {
                        return Ok(None);
                    }
                    let quota = values[0]
                        .parse::<u64>()
                        .ok()
                        .filter(|q| *q > 0)
                        .ok_or("invalid CPU quota")?;
                    Ok(Some(quota as f64 / period as f64))
                }) {
                    Ok(Some(quota)) => {
                        cpu = cpu.map(|limit| Some(limit.map_or(quota, |old| old.min(quota))))
                    }
                    Ok(None) => {}
                    Err(reason) => cpu = Err(reason),
                }
            } else {
                memory_source_found = true;
                match result.and_then(|text| {
                    if text.trim() == "max" {
                        Ok(None)
                    } else {
                        text.trim()
                            .parse::<u64>()
                            .map(Some)
                            .map_err(|_| "invalid memory.max".into())
                    }
                }) {
                    Ok(Some(limit)) => {
                        memory = memory.map(|old| Some(old.map_or(limit, |old| old.min(limit))))
                    }
                    Ok(None) => {}
                    Err(reason) => memory = Err(reason),
                }
            }
        }
        if current == visible_root {
            break;
        }
        let Some(parent) = current.parent().filter(|p| p.starts_with(visible_root)) else {
            break;
        };
        current = parent;
        if depth == 63 {
            cpu = Err("visible cgroup ancestry exceeds bound".into());
            memory = Err("visible cgroup ancestry exceeds bound".into());
        }
    }
    let cpu = match cpu {
        _ if !cpu_selected => Measurement::NotCollected,
        Ok(Some(value)) => Measurement::Available { value },
        Ok(None) if cpu_source_found => Measurement::Unlimited,
        Ok(None) => Measurement::Unsupported {
            reason: "kernel or controller does not expose this measurement".into(),
        },
        Err(reason) => Measurement::unavailable(reason),
    };
    let memory = match memory {
        _ if !memory_selected => Measurement::NotCollected,
        Ok(Some(value)) => Measurement::Available { value },
        Ok(None) if memory_source_found => Measurement::Unlimited,
        Ok(None) => Measurement::Unsupported {
            reason: "kernel or controller does not expose this measurement".into(),
        },
        Err(reason) => Measurement::unavailable(reason),
    };
    (cpu, memory)
}

fn pressure(directory: &Path, controller: &str, category: &str) -> Measurement<u64> {
    if let Some(missing) = missing_source(&directory.join(format!("{controller}.pressure"))) {
        return missing;
    }
    let result =
        read_text(&directory.join(format!("{controller}.pressure")), 4096).and_then(|text| {
            text.lines()
                .find(|line| line.split_whitespace().next() == Some(category))
                .and_then(|line| {
                    line.split_whitespace()
                        .find_map(|field| field.strip_prefix("total=")?.parse().ok())
                })
                .ok_or_else(|| "pressure total is unavailable".into())
        });
    match result {
        Ok(value) => Measurement::Available { value },
        Err(reason) => Measurement::unavailable(reason),
    }
}

fn read_host(
    root: &Path,
    pid: u32,
    groups: &[SystemStatsMetricGroup],
) -> Result<CgroupSnapshot, String> {
    let cpu_selected = groups.contains(&SystemStatsMetricGroup::CgroupCpu);
    let memory_selected = groups.contains(&SystemStatsMetricGroup::CgroupMemory);
    let pressure_selected = groups.contains(&SystemStatsMetricGroup::CgroupPressure);
    let member_file = root.join(format!("proc/{pid}/cgroup"));
    let member = read_text(&member_file, 16384)?;
    let mounts = read_text(&root.join("proc/self/mountinfo"), 1_048_576)?;
    let (directory, visible_root) = locate(root, membership(&member)?, &mounts)?;
    let target = identity(&directory)?;
    let (visible_cpu_limit_cores, visible_memory_limit_bytes) =
        limits(&directory, &visible_root, cpu_selected, memory_selected);
    let snapshot = CgroupSnapshot {
        source: "linux_cgroup_v2".into(),
        scope: "adapter_host_cgroup".into(),
        target: target.clone(),
        observed_at: rfc3339_millis(SystemTime::now()),
        visible_cpu_limit_cores,
        visible_memory_limit_bytes,
        memory_current_bytes: if memory_selected {
            number(&directory.join("memory.current"))
        } else {
            Measurement::NotCollected
        },
        memory_peak_bytes: if memory_selected {
            number(&directory.join("memory.peak"))
        } else {
            Measurement::NotCollected
        },
        swap_current_bytes: if memory_selected {
            number(&directory.join("memory.swap.current"))
        } else {
            Measurement::NotCollected
        },
        cpu_usage_microseconds: if cpu_selected {
            counter(&directory.join("cpu.stat"), "usage_usec")
        } else {
            Measurement::NotCollected
        },
        cpu_throttled_microseconds: if cpu_selected {
            counter(&directory.join("cpu.stat"), "throttled_usec")
        } else {
            Measurement::NotCollected
        },
        oom_kills: if memory_selected {
            counter(&directory.join("memory.events"), "oom_kill")
        } else {
            Measurement::NotCollected
        },
        pressure_stall_microseconds: ["cpu", "memory", "io"]
            .into_iter()
            .flat_map(|controller| {
                let directory = &directory;
                ["some", "full"].into_iter().map(move |kind| {
                    (
                        format!("{controller}_{kind}"),
                        if pressure_selected {
                            pressure(&directory, controller, kind)
                        } else {
                            Measurement::NotCollected
                        },
                    )
                })
            })
            .collect(),
    };
    if read_text(&member_file, 16384)? != member || identity(&directory)? != target {
        return Err("adapter host cgroup changed during collection".into());
    }
    Ok(snapshot)
}

pub(crate) fn delta(
    before: &CgroupObservation,
    after: &CgroupObservation,
    field: fn(&CgroupSnapshot) -> &Measurement<u64>,
) -> Measurement<u64> {
    for observation in [before, after] {
        if let CgroupObservation::Unsupported { reason } = observation {
            return Measurement::Unsupported {
                reason: reason.clone(),
            };
        }
    }
    if let Some((before, after)) = before.snapshot().zip(after.snapshot()) {
        if before.target != after.target {
            return Measurement::unavailable("cgroup target changed");
        }
        if matches!(
            (field(before), field(after)),
            (Measurement::NotCollected, Measurement::NotCollected)
        ) {
            return Measurement::NotCollected;
        }
        for measurement in [field(before), field(after)] {
            if let Measurement::Unsupported { .. } = measurement {
                return measurement.clone();
            }
        }
        if let Some((before, after)) = field(before).value().zip(field(after).value()) {
            return after
                .checked_sub(*before)
                .map(|value| Measurement::Available { value })
                .unwrap_or_else(|| Measurement::unavailable("counter decreased or reset"));
        }
    }
    Measurement::unavailable("counter endpoints are unavailable")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_observations_preserve_counter_availability() {
        let unsupported = CgroupObservation::Unsupported {
            reason: "no cgroup reader".into(),
        };
        let unavailable = CgroupObservation::Unavailable {
            reason: "host exited".into(),
        };
        for (before, after) in [(&unsupported, &unavailable), (&unavailable, &unsupported)] {
            assert_eq!(
                delta(before, after, |s| &s.cpu_usage_microseconds),
                Measurement::Unsupported {
                    reason: "no cgroup reader".into()
                }
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn unsupported_counter_sources_preserve_availability() {
        let root = fixture("unsupported-counter");
        let available = CgroupObservation::Available {
            snapshot: read_host(&root, 42, &super::super::default_groups()).unwrap(),
        };
        std::fs::remove_file(root.join("sys/fs/cgroup/parent/child/cpu.stat")).unwrap();
        let unsupported = CgroupObservation::Available {
            snapshot: read_host(&root, 42, &super::super::default_groups()).unwrap(),
        };
        for (before, after) in [(&available, &unsupported), (&unsupported, &available)] {
            assert!(matches!(
                delta(before, after, |s| &s.cpu_usage_microseconds),
                Measurement::Unsupported { .. }
            ));
        }
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn cpuset_ranges_are_counted_without_duplicates_or_unsafe_paths() {
        assert_eq!(cpus("0-3,8,10-11\n").unwrap(), 7);
        for invalid in ["", "1-0", "0-3,2", "0,0", "x"] {
            assert!(cpus(invalid).is_err());
        }
        assert!(membership("0::/../../host\n").is_err());
        assert!(
            locate(
                Path::new("/"),
                Path::new("/outside"),
                "1 2 0:1 /container /sys/fs/cgroup rw - cgroup2 cgroup rw"
            )
            .is_err()
        );
    }

    #[cfg(unix)]
    fn fixture(name: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("fabric-cgroup-stats-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let group = root.join("sys/fs/cgroup/parent/child");
        std::fs::create_dir_all(&group).unwrap();
        std::fs::create_dir_all(root.join("proc/42")).unwrap();
        std::fs::create_dir_all(root.join("proc/self")).unwrap();
        std::fs::write(root.join("proc/42/cgroup"), "0::/parent/child\n").unwrap();
        std::fs::write(
            root.join("proc/self/mountinfo"),
            "1 2 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
        )
        .unwrap();
        for (file, contents) in [
            ("cpuset.cpus.effective", "0-3"),
            ("cpu.max", "max 100000"),
            ("memory.max", "max"),
            ("cpu.stat", "usage_usec 123456\nthrottled_usec 456\n"),
            ("memory.current", "80"),
            ("memory.peak", "400"),
            ("memory.swap.current", "0"),
            ("memory.events", "oom_kill 2\n"),
            (
                "cpu.pressure",
                "some avg10=0.00 total=123\nfull avg10=0.00 total=12\n",
            ),
        ] {
            std::fs::write(group.join(file), contents).unwrap();
        }
        std::fs::write(group.parent().unwrap().join("cpu.max"), "50000 100000").unwrap();
        std::fs::write(group.parent().unwrap().join("memory.max"), "1024").unwrap();
        root
    }

    #[cfg(unix)]
    #[test]
    fn selecting_memory_skips_cpu_and_pressure_sources() {
        let root = fixture("memory-selection");
        let group = root.join("sys/fs/cgroup/parent/child");
        for name in [
            "cpu.max",
            "cpuset.cpus.effective",
            "cpu.stat",
            "cpu.pressure",
        ] {
            std::fs::write(group.join(name), "invalid").unwrap();
        }
        let snapshot = read_host(&root, 42, &[SystemStatsMetricGroup::CgroupMemory]).unwrap();
        assert!(matches!(
            snapshot.memory_current_bytes,
            Measurement::Available { value: 80 }
        ));
        assert!(matches!(
            snapshot.visible_memory_limit_bytes,
            Measurement::Available { value: 1024 }
        ));
        assert!(matches!(
            snapshot.visible_cpu_limit_cores,
            Measurement::NotCollected
        ));
        assert!(matches!(
            snapshot.cpu_usage_microseconds,
            Measurement::NotCollected
        ));
        assert!(
            snapshot
                .pressure_stall_microseconds
                .values()
                .all(|v| matches!(v, Measurement::NotCollected))
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn missing_controller_files_do_not_hide_visible_ancestor_quotas() {
        let root = fixture("missing-controllers");
        let group = root.join("sys/fs/cgroup/parent/child");
        for name in ["cpu.max", "cpuset.cpus.effective", "memory.max"] {
            std::fs::remove_file(group.join(name)).unwrap();
        }
        let snapshot = read_host(&root, 42, &super::super::default_groups()).unwrap();
        assert!(
            matches!(snapshot.visible_cpu_limit_cores, Measurement::Available { value } if value == 0.5)
        );
        assert!(matches!(
            snapshot.visible_memory_limit_bytes,
            Measurement::Available { value: 1024 }
        ));
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn explicit_max_limits_remain_unlimited() {
        let root = fixture("unlimited-limits");
        let group = root.join("sys/fs/cgroup/parent/child");
        std::fs::remove_file(group.join("cpuset.cpus.effective")).unwrap();
        for name in ["cpu.max", "memory.max"] {
            std::fs::remove_file(group.parent().unwrap().join(name)).unwrap();
        }
        let snapshot = read_host(&root, 42, &super::super::default_groups()).unwrap();
        assert!(matches!(
            snapshot.visible_cpu_limit_cores,
            Measurement::Unlimited
        ));
        assert!(matches!(
            snapshot.visible_memory_limit_bytes,
            Measurement::Unlimited
        ));
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn missing_limit_sources_are_unsupported() {
        let root = fixture("absent-limits");
        let group = root.join("sys/fs/cgroup/parent/child");
        std::fs::remove_file(group.join("cpuset.cpus.effective")).unwrap();
        for directory in [&group, group.parent().unwrap()] {
            for name in ["cpu.max", "memory.max"] {
                std::fs::remove_file(directory.join(name)).unwrap();
            }
        }
        let snapshot = read_host(&root, 42, &super::super::default_groups()).unwrap();
        assert!(matches!(
            snapshot.visible_cpu_limit_cores,
            Measurement::Unsupported { .. }
        ));
        assert!(matches!(
            snapshot.visible_memory_limit_bytes,
            Measurement::Unsupported { .. }
        ));
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn reads_visible_ancestor_limits_and_keeps_missing_metrics_explicit() {
        let root = fixture("limits");
        let snapshot = read_host(&root, 42, &super::super::default_groups()).unwrap();
        assert_eq!(snapshot.visible_cpu_limit_cores.value(), Some(&0.5));
        assert_eq!(snapshot.visible_memory_limit_bytes.value(), Some(&1024));
        assert_eq!(snapshot.memory_current_bytes.value(), Some(&80));
        assert_eq!(snapshot.memory_peak_bytes.value(), Some(&400));
        assert_eq!(snapshot.cpu_usage_microseconds.value(), Some(&123456));
        assert_eq!(
            snapshot.pressure_stall_microseconds["cpu_some"].value(),
            Some(&123)
        );
        assert!(matches!(
            snapshot.pressure_stall_microseconds["io_some"],
            Measurement::Unsupported { .. }
        ));
        assert_eq!(snapshot.scope, "adapter_host_cgroup");
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn counter_resets_and_cgroup_changes_never_turn_into_zero_delta() {
        let root = fixture("delta");
        let before = CgroupObservation::Available {
            snapshot: read_host(&root, 42, &super::super::default_groups()).unwrap(),
        };
        std::fs::write(
            root.join("sys/fs/cgroup/parent/child/cpu.stat"),
            "usage_usec 4\n",
        )
        .unwrap();
        let mut after = CgroupObservation::Available {
            snapshot: read_host(&root, 42, &super::super::default_groups()).unwrap(),
        };
        assert!(matches!(
            delta(&before, &after, |s| &s.cpu_usage_microseconds),
            Measurement::Unavailable { .. }
        ));
        if let CgroupObservation::Available { snapshot } = &mut after {
            snapshot.target = "different".into();
        }
        assert!(matches!(
            delta(&before, &after, |s| &s.oom_kills),
            Measurement::Unavailable { .. }
        ));
        let _ = std::fs::remove_dir_all(root);
    }
}
