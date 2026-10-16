// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One process-wide thread that appends a sample line to every active record.
//!
//! The thread starts with the first recorded invocation and serves every
//! runtime in the process, so N parallel runtimes cost one wake-up per
//! interval, not N threads. Each tick reads each record's runtime snapshot,
//! which never waits for the runtime's host lock.

use std::collections::BTreeMap;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, Once, Weak};
use std::thread;
use std::time::{Duration, Instant};

use super::file::{RecordFile, lock};
use super::lines::{LineBody, Resources, SAMPLE_INTERVAL_MS, SampleLine};
use crate::error::Result;
use crate::resource_snapshot::RuntimeResourceSnapshot;

/// Reads a runtime's resource snapshot without waiting for its host lock.
pub(crate) type ReadResources = Arc<dyn Fn() -> Result<RuntimeResourceSnapshot> + Send + Sync>;

static GLOBAL: LazyLock<Arc<SampleTicker>> =
    LazyLock::new(|| SampleTicker::new(Duration::from_millis(SAMPLE_INTERVAL_MS)));

/// One active record the ticker samples.
struct ActiveRecord {
    file: Arc<Mutex<RecordFile>>,
    started: Instant,
    read: ReadResources,
}

impl ActiveRecord {
    /// Read the snapshot, then append the sample line. The read runs without
    /// the file lock, so a slow read never delays the end line. A read that
    /// panics is written as unavailable, so the record keeps its heartbeat.
    fn sample(&self) {
        let resources = panic::catch_unwind(AssertUnwindSafe(|| Resources::from(&(self.read)())))
            .unwrap_or(Resources::UNAVAILABLE);
        let sample = SampleLine {
            elapsed_ms: u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX),
            resources,
        };
        lock(&self.file).append_sample(LineBody::Sample(sample));
    }
}

/// Appends sample lines to active records at a fixed interval.
pub(crate) struct SampleTicker {
    interval: Duration,
    records: Mutex<BTreeMap<u64, Arc<ActiveRecord>>>,
    next_id: AtomicU64,
    thread_started: Once,
    running: AtomicBool,
    /// The process that spawned the thread. A forked child inherits
    /// `running` but not the thread.
    spawned_by: AtomicU32,
}

impl SampleTicker {
    /// A ticker with its own interval; [`Self::start`] starts its thread.
    pub(crate) fn new(interval: Duration) -> Arc<Self> {
        Arc::new(Self {
            interval,
            records: Mutex::new(BTreeMap::new()),
            next_id: AtomicU64::new(0),
            thread_started: Once::new(),
            running: AtomicBool::new(false),
            spawned_by: AtomicU32::new(0),
        })
    }

    /// The process-wide ticker, sampling every [`SAMPLE_INTERVAL_MS`].
    pub(crate) fn global() -> Arc<Self> {
        GLOBAL.clone()
    }

    /// Start the thread once, and return the sample interval when it runs
    /// in this process. `None` means the operating system refused the thread,
    /// or this process is a forked child of the one that started it, so
    /// records get no sample lines.
    pub(crate) fn start(self: &Arc<Self>) -> Option<Duration> {
        self.thread_started.call_once(|| {
            // The thread holds a weak reference and ends when the ticker is
            // dropped. The process-wide ticker is never dropped.
            let ticker = Arc::downgrade(self);
            let interval = self.interval;
            let spawned = thread::Builder::new()
                .name("fabric-record-samples".to_string())
                .spawn(move || run(&ticker, interval));
            self.spawned_by.store(std::process::id(), Ordering::Relaxed);
            self.running.store(spawned.is_ok(), Ordering::Release);
        });
        self.interval_in(std::process::id())
    }

    /// The sample interval when the thread runs in process `pid`.
    fn interval_in(&self, pid: u32) -> Option<Duration> {
        (self.running.load(Ordering::Acquire) && self.spawned_by.load(Ordering::Relaxed) == pid)
            .then_some(self.interval)
    }

    /// Sample `file` until the returned registration is dropped.
    pub(crate) fn register(
        self: &Arc<Self>,
        file: Arc<Mutex<RecordFile>>,
        started: Instant,
        read: ReadResources,
    ) -> Registration {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.lock_records().insert(
            id,
            Arc::new(ActiveRecord {
                file,
                started,
                read,
            }),
        );
        Registration {
            ticker: Arc::clone(self),
            id,
        }
    }

    /// Append one sample line to every active record.
    fn tick(&self) {
        let active: Vec<Arc<ActiveRecord>> = self.lock_records().values().cloned().collect();
        for record in active {
            record.sample();
        }
    }

    fn lock_records(&self) -> MutexGuard<'_, BTreeMap<u64, Arc<ActiveRecord>>> {
        self.records
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
}

/// The ticker thread: tick on a fixed schedule until the ticker is gone.
/// Ticks are scheduled from the first deadline, so the time a tick takes
/// does not add up across ticks.
fn run(ticker: &Weak<SampleTicker>, interval: Duration) {
    let mut next = Instant::now() + interval;
    loop {
        thread::sleep(next.saturating_duration_since(Instant::now()));
        let Some(ticker) = ticker.upgrade() else {
            return;
        };
        ticker.tick();
        drop(ticker);
        next += interval;
        let now = Instant::now();
        if next < now {
            // A tick longer than the interval skips the missed ticks.
            next = now + interval;
        }
    }
}

/// Keeps a record in the ticker; dropping it stops sampling that record.
pub(crate) struct Registration {
    ticker: Arc<SampleTicker>,
    id: u64,
}

impl Drop for Registration {
    fn drop(&mut self) {
        self.ticker.lock_records().remove(&self.id);
    }
}

impl std::fmt::Debug for Registration {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Registration")
            .field("id", &self.id)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_returns_no_interval_in_a_process_that_did_not_spawn_the_thread() {
        let ticker = SampleTicker::new(Duration::from_secs(3600));

        assert_eq!(ticker.start(), Some(Duration::from_secs(3600)));
        // A forked child keeps the parent's memory, so it sees a started
        // thread, but it has another PID and no thread.
        assert_eq!(ticker.interval_in(std::process::id().wrapping_add(1)), None);
    }
}
