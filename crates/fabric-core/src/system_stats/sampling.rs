// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded resource series and reductions. The single process-wide ticker,
//! registrations, fork check, and read-outside-lock lifecycle adapt
//! Maryam Najafian's sampler in PR #426, independently of invocation recording.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, Once};
use std::thread;
use std::time::{Duration, Instant};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::{
    CgroupObservation, Measurement, ResourceObservation, SystemStatsMetricGroup,
    SystemStatsSamplingConfig, cgroup,
};
use crate::resource_snapshot::{LocalHostProcess, snapshot_local_host};
use crate::runtime::{ArtifactRef, InvocationHandle};

#[derive(Clone)]
pub(crate) struct SystemReader {
    pub runtime_id: String,
    pub host: Option<LocalHostProcess>,
    pub local: bool,
    pub collect: Vec<SystemStatsMetricGroup>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Reading {
    pub adapter_tree: ResourceObservation,
    pub cgroup: CgroupObservation,
    pub collection_seconds: f64,
}

impl SystemReader {
    pub(crate) fn read(&self) -> Reading {
        let started = Instant::now();
        let collected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let resource = if !self.collect.contains(&SystemStatsMetricGroup::Process) {
                ResourceObservation::NotCollected
            } else {
                ResourceObservation::from_result(if self.local {
                    snapshot_local_host(&self.runtime_id, self.host.as_ref())
                } else {
                    Err(crate::FabricError::ResourceObservationUnsupported {
                        runtime_id: self.runtime_id.clone(),
                        reason: "provider has no local resource reader".into(),
                    })
                })
            };
            (
                resource,
                cgroup::observe(self.host.as_ref(), self.local, &self.collect),
            )
        }));
        let (adapter_tree, cgroup) = collected.unwrap_or_else(|_| {
            (
                if self.collect.contains(&SystemStatsMetricGroup::Process) {
                    ResourceObservation::Unavailable {
                        reason: "resource collector panicked".into(),
                    }
                } else {
                    ResourceObservation::NotCollected
                },
                if self
                    .collect
                    .iter()
                    .any(|group| *group != SystemStatsMetricGroup::Process)
                {
                    CgroupObservation::Unavailable {
                        reason: "resource collector panicked".into(),
                    }
                } else {
                    CgroupObservation::NotCollected
                },
            )
        });
        Reading {
            adapter_tree,
            cgroup,
            collection_seconds: started.elapsed().as_secs_f64(),
        }
    }
}

/// A sampled gauge over the invocation window. Min/max cover retained valid
/// samples. Mean holds the previous value only across adjacent valid samples
/// no more than twice the requested cadence apart. Missing intervals never
/// contribute zero. No sampled maximum claims a true process high-water mark.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct GaugeSummary {
    /// `byte` for RSS or `process` for process count.
    pub unit: String,
    /// Final endpoint, absent if its reading failed.
    pub final_value: Option<u64>,
    /// Minimum retained value.
    pub min: Option<u64>,
    /// Maximum retained value, not an unsampled high-water mark.
    pub sampled_peak: Option<u64>,
    /// Hold-last-value integral divided by covered seconds, or unknown.
    pub time_weighted_mean: Option<f64>,
    /// Number of valid retained readings for this metric.
    pub valid_samples: u32,
    /// Seconds covered by valid adjacent readings.
    pub covered_seconds: f64,
    /// Covered seconds divided by invocation duration.
    pub covered_fraction: f64,
}

/// Resource series retention, reductions, and collection quality. Resource
/// samples remain in the referenced JSONL artifact for interval selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SampleSummary {
    /// Requested cadence; actual timing is retained in sample offsets.
    pub target_interval_ms: u32,
    /// Retention bound, including endpoints.
    pub max_samples: u32,
    /// Retained samples, including unavailable readings and endpoints.
    pub sample_count: u32,
    /// Samples skipped because collection fell behind or retention was full.
    pub dropped_samples: u64,
    /// Maximum observed interval between retained readings.
    pub largest_gap_seconds: f64,
    /// Total time spent collecting retained observations, including endpoints.
    pub collection_seconds: f64,
    /// Whether collection or retention prevented a complete series.
    pub truncated: bool,
    /// Safe collection/persistence diagnostics. They never change agent status.
    pub diagnostics: Vec<String>,
    /// Series artifact, absent when its file could not be created or fully written.
    pub samples: Option<ArtifactRef>,
    /// Whether retained readings kept one cgroup identity with nondecreasing
    /// counters and no gaps over twice cadence. This does not prove exclusivity.
    pub cgroup_target_continuous: bool,
    /// Adapter-host tree resident-memory reductions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resident_memory_bytes: Option<GaugeSummary>,
    /// Adapter-host tree live-process-count reductions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub process_count: Option<GaugeSummary>,
    /// Estimated CPU counter delta, invalidated by observed resets, missing
    /// endpoints, target changes, or collection gaps larger than twice cadence.
    pub cpu_time_seconds_estimate: Measurement<f64>,
    /// 100% means one fully occupied CPU core; this is not normalized by quota.
    pub cpu_percent_one_core: Measurement<f64>,
}

#[derive(Default)]
struct Gauge {
    last: Option<(f64, u64)>,
    min: Option<u64>,
    peak: Option<u64>,
    count: u32,
    covered: f64,
    integral: f64,
}

impl Gauge {
    fn push(&mut self, t: f64, value: Option<u64>, max_gap: f64) {
        if let Some(value) = value {
            if let Some((previous_t, previous)) = self.last {
                let dt = t - previous_t;
                if dt >= 0.0 && dt <= max_gap {
                    self.covered += dt;
                    self.integral += previous as f64 * dt;
                }
            }
            self.min = Some(self.min.map_or(value, |old| old.min(value)));
            self.peak = Some(self.peak.map_or(value, |old| old.max(value)));
            self.count += 1;
        }
        self.last = value.map(|value| (t, value));
    }

    fn finish(&self, unit: &str, elapsed: f64) -> GaugeSummary {
        GaugeSummary {
            unit: unit.into(),
            final_value: self.last.map(|(_, value)| value),
            min: self.min,
            sampled_peak: self.peak,
            time_weighted_mean: (self.covered > 0.0).then(|| self.integral / self.covered),
            valid_samples: self.count,
            covered_seconds: self.covered,
            covered_fraction: if elapsed > 0.0 {
                (self.covered / elapsed).min(1.0)
            } else {
                0.0
            },
        }
    }
}

// Sequence numbers let finalization flush the last accepted sample itself.
// A ticker delayed after accepting a sample can subsequently retry that write
// without duplicating it or placing it after the final endpoint.
struct SampleFile {
    writer: Option<File>,
    next: u32,
    failed: bool,
}
impl SampleFile {
    fn write(&mut self, sequence: u32, record: &serde_json::Value) {
        if sequence < self.next {
            return;
        }
        if sequence != self.next {
            self.failed = true;
            self.writer = None;
            return;
        }
        self.next += 1;
        if let Some(writer) = &mut self.writer {
            if serde_json::to_writer(&mut *writer, record).is_err()
                || writer.write_all(b"\n").is_err()
            {
                self.failed = true;
                self.writer = None;
            }
        }
    }
}
type Pending = (Arc<Mutex<SampleFile>>, u32, serde_json::Value);
fn flush_pending(pending: Option<Pending>) {
    if let Some((file, sequence, record)) = pending {
        file.lock()
            .unwrap_or_else(|e| e.into_inner())
            .write(sequence, &record);
    }
}

struct SampleState {
    config: SystemStatsSamplingConfig,
    ids: InvocationHandle,
    started: Instant,
    next_due: Instant,
    closed: bool,
    file: Option<Arc<Mutex<SampleFile>>>,
    pending: Option<serde_json::Value>,
    path: Option<PathBuf>,
    diagnostics: Vec<String>,
    count: u32,
    dropped: u64,
    largest_gap: f64,
    last_t: f64,
    collection: f64,
    rss: Gauge,
    processes: Gauge,
    cpu_first: Option<f64>,
    cpu_last: Option<f64>,
    cpu_invalid: bool,
    target: Option<(String, String)>,
    cgroup_target: Option<String>,
    cgroup_counters: Option<[Option<u64>; 3]>,
    cgroup_invalid: bool,
    reader: SystemReader,
}

impl SampleState {
    fn push(&mut self, t: f64, reading: &Reading) {
        let maximum_gap = f64::from(self.config.interval_ms) / 500.0;
        if self.count > 0 {
            self.largest_gap = self.largest_gap.max(t - self.last_t);
            self.cpu_invalid |= t - self.last_t > maximum_gap;
        }
        self.last_t = t;
        self.count += 1;
        self.collection += reading.collection_seconds;
        let snapshot = reading.adapter_tree.snapshot();
        self.rss
            .push(t, snapshot.map(|s| s.resident_memory_bytes), maximum_gap);
        self.processes
            .push(t, snapshot.map(|s| s.process_count), maximum_gap);
        if let Some(snapshot) = snapshot {
            let target = (snapshot.runtime_id.clone(), snapshot.started_at.clone());
            if let Some(previous) = &self.target {
                self.cpu_invalid |= *previous != target;
            }
            self.target = Some(target);
            if let Some(last) = self.cpu_last {
                self.cpu_invalid |= snapshot.cpu_time_seconds < last;
            }
            if self.count == 1 {
                self.cpu_first = Some(snapshot.cpu_time_seconds);
            }
            self.cpu_last = Some(snapshot.cpu_time_seconds);
        } else {
            self.cpu_last = None;
            self.cpu_invalid = true;
        }
        if let Some(snapshot) = reading.cgroup.snapshot() {
            if let Some(target) = &self.cgroup_target {
                self.cgroup_invalid |= *target != snapshot.target;
            }
            let counters = [
                snapshot.cpu_usage_microseconds.value().copied(),
                snapshot.cpu_throttled_microseconds.value().copied(),
                snapshot.oom_kills.value().copied(),
            ];
            if let Some(previous) = self.cgroup_counters {
                self.cgroup_invalid |= previous
                    .into_iter()
                    .zip(counters)
                    .any(|(a, b)| a.zip(b).is_some_and(|(a, b)| b < a));
            }
            self.cgroup_counters = Some(counters);
            self.cgroup_target = Some(snapshot.target.clone());
        } else {
            self.cgroup_invalid = true;
        }
        if self.file.is_some() {
            self.pending = Some(serde_json::json!({
                "schema_version": "fabric.system_stats/v1alpha1",
                "runtime_id": self.ids.runtime_id, "invocation_id": self.ids.invocation_id,
                "request_id": self.ids.request_id, "offset_seconds": t,
                "adapter_tree": reading.adapter_tree, "cgroup": reading.cgroup,
                "collection_seconds": reading.collection_seconds,
            }));
        }
    }

    fn pending(&self) -> Option<Pending> {
        self.file
            .as_ref()
            .zip(self.pending.as_ref())
            .map(|(file, record)| (Arc::clone(file), self.count - 1, record.clone()))
    }

    fn summary(&self, elapsed: f64) -> SampleSummary {
        let delta = self
            .cpu_first
            .zip(self.cpu_last)
            .filter(|_| !self.cpu_invalid)
            .map(|(first, last)| last - first)
            .filter(|value| value.is_finite() && *value >= 0.0);
        let cpu_time_seconds_estimate = delta
            .map(|value| Measurement::Available { value })
            .unwrap_or(Measurement::Unavailable {
                reason: "counter endpoints, continuity, or identity are unavailable".into(),
            });
        let cpu_percent_one_core = delta
            .filter(|_| elapsed > 0.0)
            .map(|value| value / elapsed * 100.0)
            .filter(|value| value.is_finite())
            .map(|value| Measurement::Available { value })
            .unwrap_or(Measurement::Unavailable {
                reason: "CPU delta or duration is unavailable".into(),
            });
        let selected_process = self
            .reader
            .collect
            .contains(&SystemStatsMetricGroup::Process);
        SampleSummary {
            target_interval_ms: self.config.interval_ms,
            max_samples: self.config.max_samples,
            sample_count: self.count,
            dropped_samples: self.dropped,
            largest_gap_seconds: self.largest_gap,
            collection_seconds: self.collection,
            truncated: self.dropped > 0 || !self.diagnostics.is_empty(),
            diagnostics: self.diagnostics.clone(),
            samples: self.path.as_ref().map(|path| ArtifactRef {
                name: "system_stats_samples".into(),
                kind: "system_stats_samples".into(),
                path: path.clone(),
                media_type: Some("application/x-ndjson".into()),
                metadata: BTreeMap::new(),
            }),
            cgroup_target_continuous: !self.cgroup_invalid
                && self.largest_gap <= f64::from(self.config.interval_ms) / 500.0,
            resident_memory_bytes: self
                .reader
                .collect
                .contains(&SystemStatsMetricGroup::Process)
                .then(|| self.rss.finish("byte", elapsed)),
            process_count: self
                .reader
                .collect
                .contains(&SystemStatsMetricGroup::Process)
                .then(|| self.processes.finish("process", elapsed)),
            cpu_time_seconds_estimate: if selected_process {
                cpu_time_seconds_estimate
            } else {
                Measurement::NotCollected
            },
            cpu_percent_one_core: if selected_process {
                cpu_percent_one_core
            } else {
                Measurement::NotCollected
            },
        }
    }
}

static TICKER: LazyLock<Arc<SampleTicker>> = LazyLock::new(|| {
    Arc::new(SampleTicker {
        records: Mutex::new(BTreeMap::new()),
        wake: Condvar::new(),
        next: AtomicU64::new(1),
        once: Once::new(),
        running: AtomicBool::new(false),
        process: AtomicU32::new(0),
    })
});

struct SampleTicker {
    records: Mutex<BTreeMap<u64, Arc<Mutex<SampleState>>>>,
    wake: Condvar,
    next: AtomicU64,
    once: Once,
    running: AtomicBool,
    process: AtomicU32,
}

impl SampleTicker {
    fn start(self: &Arc<Self>) -> bool {
        // A forked child does not inherit the sampling thread. Check before
        // acquiring the ticker's mutex, which may have been locked at fork.
        let pid = self.process.load(Ordering::Acquire);
        if pid != 0 && pid != std::process::id() {
            return false;
        }
        self.once.call_once(|| {
            self.process.store(std::process::id(), Ordering::Release);
            let ticker = Arc::downgrade(self);
            let spawned = thread::Builder::new()
                .name("fabric-system-stats".into())
                .spawn(move || {
                    loop {
                        let Some(ticker) = ticker.upgrade() else {
                            return;
                        };
                        let mut records = ticker.records.lock().unwrap_or_else(|e| e.into_inner());
                        while records.is_empty() {
                            records = ticker.wake.wait(records).unwrap_or_else(|e| e.into_inner());
                        }
                        let active: Vec<_> = records.values().cloned().collect();
                        drop(records);
                        for record in active {
                            let read = {
                                let mut state = record.lock().unwrap_or_else(|e| e.into_inner());
                                if state.closed || Instant::now() < state.next_due {
                                    continue;
                                }
                                let interval =
                                    Duration::from_millis(u64::from(state.config.interval_ms));
                                let missed = Instant::now()
                                    .saturating_duration_since(state.next_due)
                                    .as_nanos()
                                    / interval.as_nanos();
                                state.dropped = state
                                    .dropped
                                    .saturating_add(missed.min(u64::MAX as u128) as u64);
                                state.next_due = Instant::now() + interval;
                                if state.count >= state.config.max_samples - 1 {
                                    state.dropped = state.dropped.saturating_add(1);
                                    continue;
                                }
                                state.reader.clone()
                            };
                            // A slow read never holds the host, registry, or state lock.
                            let reading = read.read();
                            let mut state = record.lock().unwrap_or_else(|e| e.into_inner());
                            let pending = if !state.closed {
                                let t = state.started.elapsed().as_secs_f64();
                                state.push(t, &reading);
                                state.pending()
                            } else {
                                None
                            };
                            drop(state);
                            flush_pending(pending);
                        }
                        let records = ticker.records.lock().unwrap_or_else(|e| e.into_inner());
                        let _ = ticker.wake.wait_timeout(records, Duration::from_millis(50));
                    }
                });
            self.running.store(spawned.is_ok(), Ordering::Release);
        });
        self.running.load(Ordering::Acquire)
    }
}

/// Per-invocation registration. Dropping it closes the window without waiting
/// for in-flight OS reads; their late results are discarded.
pub(crate) struct StatsSession {
    state: Arc<Mutex<SampleState>>,
    id: u64,
    ticker: Arc<SampleTicker>,
}

impl StatsSession {
    pub(crate) fn prepare(
        config: &SystemStatsSamplingConfig,
        ids: &InvocationHandle,
        reader: SystemReader,
        before: &Reading,
        home: &Path,
    ) -> Self {
        let path = home.join("system-stats.jsonl");
        let opened = OpenOptions::new().write(true).create_new(true).open(&path);
        let (file, path, mut diagnostics) = match opened {
            Ok(file) => (
                Some(Arc::new(Mutex::new(SampleFile {
                    writer: Some(file),
                    next: 0,
                    failed: false,
                }))),
                Some(path),
                Vec::new(),
            ),
            Err(_) => (None, None, vec!["sample_file_unavailable".into()]),
        };
        let ticker = Arc::clone(&TICKER);
        if !ticker.start() {
            diagnostics.push("sampling_thread_unavailable".into());
        }
        let started = Instant::now();
        let mut state = SampleState {
            config: config.clone(),
            ids: ids.clone(),
            started,
            next_due: started + Duration::from_millis(u64::from(config.interval_ms)),
            closed: false,
            file,
            pending: None,
            path,
            diagnostics,
            count: 0,
            dropped: 0,
            largest_gap: 0.0,
            last_t: 0.0,
            collection: 0.0,
            rss: Gauge::default(),
            processes: Gauge::default(),
            cpu_first: None,
            cpu_last: None,
            cpu_invalid: false,
            target: None,
            cgroup_target: None,
            cgroup_counters: None,
            cgroup_invalid: false,
            reader,
        };
        state.push(0.0, before);
        flush_pending(state.pending());
        Self {
            state: Arc::new(Mutex::new(state)),
            id: ticker.next.fetch_add(1, Ordering::Relaxed),
            ticker,
        }
    }

    pub(crate) fn start(&self, started: Instant) {
        if !self.ticker.running.load(Ordering::Acquire) {
            return;
        }
        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.started = started;
            state.next_due = started + Duration::from_millis(u64::from(state.config.interval_ms));
        }
        self.ticker
            .records
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(self.id, self.state.clone());
        self.ticker.wake.notify_one();
    }

    pub(crate) fn close(&self) {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).closed = true;
        // No registered state is inherited in a child with a different PID.
        if self.ticker.process.load(Ordering::Acquire) == std::process::id() {
            self.ticker
                .records
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&self.id);
        }
    }

    pub(crate) fn finish(&self, elapsed: f64, after: &Reading) -> SampleSummary {
        self.close();
        let pending = self
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pending();
        flush_pending(pending);
        let pending = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.push(elapsed, after);
            state.pending()
        };
        flush_pending(pending);
        let file = self
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .file
            .clone();
        let failed = file.is_some_and(|file| {
            let mut file = file.lock().unwrap_or_else(|e| e.into_inner());
            file.writer.take();
            file.failed
        });
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if failed {
            state.diagnostics.push("sample_write_failed".into());
            state.path = None;
        }
        state.summary(elapsed)
    }
}

impl Drop for StatsSession {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RuntimeResourceSnapshot;

    #[test]
    fn gauge_peak_and_mean_capture_a_spike_that_endpoints_miss() {
        let mut gauge = Gauge::default();
        for (time, memory) in [(0.0, 80), (1.0, 400), (2.0, 80)] {
            gauge.push(time, Some(memory), 2.0);
        }
        let summary = gauge.finish("MiB", 2.0);
        assert_eq!(summary.final_value, Some(80));
        assert_eq!(summary.min, Some(80));
        assert_eq!(summary.sampled_peak, Some(400));
        assert_eq!(summary.time_weighted_mean, Some(240.0));
        assert_eq!(summary.covered_fraction, 1.0);
    }

    #[test]
    fn missing_readings_and_long_gaps_do_not_become_zero_or_coverage() {
        let mut gauge = Gauge::default();
        gauge.push(0.0, Some(80), 2.0);
        gauge.push(1.0, None, 2.0);
        gauge.push(2.0, Some(400), 2.0);
        gauge.push(3.0, Some(80), 2.0);
        gauge.push(8.0, Some(80), 2.0);
        let summary = gauge.finish("MiB", 8.0);
        assert_eq!(summary.covered_seconds, 1.0);
        assert_eq!(summary.covered_fraction, 0.125);
        assert_eq!(summary.time_weighted_mean, Some(400.0));
    }

    fn reading(cpu: f64) -> Reading {
        Reading {
            adapter_tree: ResourceObservation::Available {
                snapshot: RuntimeResourceSnapshot {
                    runtime_id: "runtime-1".into(),
                    started_at: "2026-10-21T00:00:00.000Z".into(),
                    observed_at: "2026-10-21T00:01:00.000Z".into(),
                    resident_memory_bytes: 80,
                    process_count: 1,
                    cpu_time_seconds: cpu,
                },
            },
            cgroup: CgroupObservation::Unsupported {
                reason: "test".into(),
            },
            collection_seconds: 0.01,
        }
    }

    fn session(name: &str, baseline: &Reading) -> (PathBuf, StatsSession) {
        let root = std::env::temp_dir().join(format!("fabric-stats-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let session = StatsSession::prepare(
            &SystemStatsSamplingConfig::default(),
            &InvocationHandle {
                runtime_id: "runtime-1".into(),
                invocation_id: "invocation-1".into(),
                request_id: "request-1".into(),
            },
            SystemReader {
                runtime_id: "runtime-1".into(),
                host: None,
                local: false,
                collect: super::super::default_groups(),
            },
            baseline,
            &root,
        );
        (root, session)
    }

    #[test]
    fn finishing_flushes_an_accepted_sample_before_the_endpoint_exactly_once() {
        let (root, session) = session("flush-order", &reading(1.0));
        let pending = {
            let mut state = session.state.lock().unwrap();
            state.push(1.0, &reading(2.0));
            state.pending()
        };
        let summary = session.finish(2.0, &reading(3.0));
        // Simulate the ticker resuming after finalization already flushed it.
        flush_pending(pending);
        let content = std::fs::read_to_string(root.join("system-stats.jsonl")).unwrap();
        let records: Vec<serde_json::Value> = content
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(records.len(), 3);
        assert_eq!(
            records
                .iter()
                .map(|row| row["offset_seconds"].as_f64().unwrap())
                .collect::<Vec<_>>(),
            vec![0.0, 1.0, 2.0]
        );
        assert_eq!(summary.cpu_time_seconds_estimate.value(), Some(&2.0));
        assert_eq!(summary.collection_seconds, 0.03);
        drop(session);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn failed_sample_write_keeps_reductions_without_publishing_partial_artifact() {
        let (root, session) = session("write-failure", &reading(1.0));
        let path = root.join("system-stats.jsonl");
        let file = session.state.lock().unwrap().file.clone().unwrap();
        file.lock().unwrap().writer = Some(File::open(&path).unwrap());
        let summary = session.finish(1.0, &reading(2.0));
        assert!(summary.samples.is_none());
        assert_eq!(summary.sample_count, 2);
        assert_eq!(summary.cpu_time_seconds_estimate.value(), Some(&1.0));
        assert_eq!(summary.resident_memory_bytes.unwrap().valid_samples, 2);
        assert!(summary.truncated);
        assert_eq!(summary.diagnostics, vec!["sample_write_failed"]);
        assert_eq!(std::fs::read_to_string(path).unwrap().lines().count(), 1);
        drop(session);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn observed_counter_reset_invalidates_a_positive_endpoint_difference() {
        let (root, session) = session("counter-reset", &reading(1.0));
        session.state.lock().unwrap().push(1.0, &reading(0.5));
        let summary = session.finish(2.0, &reading(3.0));
        assert!(matches!(
            summary.cpu_time_seconds_estimate,
            Measurement::Unavailable { .. }
        ));
        drop(session);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn unsupported_counter_does_not_invalidate_other_cgroup_counters() {
        let cgroup_reading = |cpu| {
            let mut sample = reading(cpu);
            sample.cgroup = CgroupObservation::Available {
                snapshot: super::super::CgroupSnapshot {
                    source: "linux_cgroup_v2".into(),
                    scope: "adapter_host_cgroup".into(),
                    target: "device:inode".into(),
                    observed_at: "2026-10-21T00:01:00.000Z".into(),
                    visible_cpu_limit_cores: Measurement::Unlimited,
                    visible_memory_limit_bytes: Measurement::Unlimited,
                    memory_current_bytes: Measurement::Available { value: 80 },
                    memory_peak_bytes: Measurement::Available { value: 400 },
                    swap_current_bytes: Measurement::Available { value: 0 },
                    cpu_usage_microseconds: Measurement::Available {
                        value: (cpu * 1_000_000.0) as u64,
                    },
                    cpu_throttled_microseconds: Measurement::Unsupported {
                        reason: "controller does not expose throttling".into(),
                    },
                    oom_kills: Measurement::Available { value: 0 },
                    pressure_stall_microseconds: BTreeMap::new(),
                },
            };
            sample
        };
        let (root, session) = session("partial-cgroup-counters", &cgroup_reading(1.0));
        session
            .state
            .lock()
            .unwrap()
            .push(1.0, &cgroup_reading(2.0));
        let summary = session.finish(2.0, &cgroup_reading(3.0));
        assert!(summary.cgroup_target_continuous);
        drop(session);
        let _ = std::fs::remove_dir_all(root);
    }
}
