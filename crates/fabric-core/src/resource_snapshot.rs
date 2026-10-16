// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resource snapshots of a runtime's local adapter host and its descendants.
//!
//! A snapshot reads the process tree rooted at the adapter host process that
//! NeMo Fabric started for a runtime. Each read visits a parent before its
//! children. CPU time adds every process's own CPU time and the CPU time of the
//! children it has already waited for, so work done by finished descendants
//! still counts. A child reaped between the read of its parent and its own
//! read is missed by that one snapshot and is never counted twice. CPU time
//! can drop between snapshots when a descendant is reparented out of the tree,
//! and on macOS while an exited descendant is not yet waited for, because
//! macOS cannot list an unreaped process.
//!
//! Linux reads `/proc/<pid>/stat` and macOS reads `proc_pid_rusage`. Neither
//! reads command lines, environment variables, or open files.

use std::time::{SystemTime, UNIX_EPOCH};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{FabricError, Result};

const MILLIS_PER_DAY: i64 = 86_400_000;
const NANOS_PER_SECOND: f64 = 1_000_000_000.0;

/// Timestamped resource usage of one runtime's local adapter host process tree.
///
/// Memory, process count, and CPU time cover the adapter host process and its
/// live descendants. For a runtime started on a service, the shared service
/// process is not included. Values Fabric cannot read are returned as an error,
/// never as zero.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RuntimeResourceSnapshot {
    /// Runtime this snapshot describes.
    pub runtime_id: String,
    /// When Fabric started the runtime's adapter host, as an RFC 3339 UTC timestamp.
    #[schemars(extend("format" = "date-time"))]
    pub started_at: String,
    /// When Fabric read these values, as an RFC 3339 UTC timestamp.
    #[schemars(extend("format" = "date-time"))]
    pub observed_at: String,
    /// Sum of the resident set sizes of the live processes, in bytes. Pages
    /// shared between processes count once per process.
    pub resident_memory_bytes: u64,
    /// Live processes in the tree, including the adapter host. Exited processes
    /// that are not yet reaped are not counted.
    pub process_count: u64,
    /// CPU time consumed by the tree since the adapter host started, in seconds,
    /// including exited descendants (on macOS, only once they are waited for).
    /// It can drop when a descendant leaves the tree, and on macOS while an
    /// exited descendant is not yet waited for.
    #[schemars(range(min = 0.0))]
    pub cpu_time_seconds: f64,
}

/// The adapter host process of one runtime, as recorded when the runtime started.
#[derive(Debug, Clone)]
pub(crate) struct LocalHostProcess {
    pid: u32,
    /// Start time of the process at `pid`, which tells the host apart from a
    /// later process that reuses its PID.
    identity: Option<u64>,
    started_at: String,
}

impl LocalHostProcess {
    /// Record the host `pid`, started by Fabric at `started_at`.
    pub(crate) fn new(pid: u32, started_at: SystemTime) -> Self {
        Self {
            pid,
            identity: platform::start_identity(pid),
            started_at: rfc3339_millis(unix_millis(started_at)),
        }
    }
}

/// Read a snapshot of `host`, or explain why none can be read.
///
/// `host` is `None` when the runtime is not active in this process.
pub(crate) fn snapshot_local_host(
    runtime_id: &str,
    host: Option<&LocalHostProcess>,
) -> Result<RuntimeResourceSnapshot> {
    if !platform::SUPPORTED {
        return Err(FabricError::ResourceObservationUnsupported {
            runtime_id: runtime_id.to_string(),
            reason: format!(
                "no process reader for operating system `{}`",
                std::env::consts::OS
            ),
        });
    }
    let host =
        host.ok_or_else(|| unavailable(runtime_id, "runtime is not active in this process"))?;
    let read = host
        .identity
        .and_then(|identity| platform::read_tree(host.pid, identity))
        .ok_or_else(|| {
            unavailable(
                runtime_id,
                "adapter host process has exited or the process table cannot be read",
            )
        })?;
    Ok(RuntimeResourceSnapshot {
        runtime_id: runtime_id.to_string(),
        started_at: host.started_at.clone(),
        observed_at: rfc3339_millis(unix_millis(SystemTime::now())),
        resident_memory_bytes: read.resident_bytes,
        process_count: read.process_count,
        cpu_time_seconds: read.cpu_nanos as f64 / NANOS_PER_SECOND,
    })
}

fn unavailable(runtime_id: &str, reason: &str) -> FabricError {
    FabricError::ResourceObservationUnavailable {
        runtime_id: runtime_id.to_string(),
        reason: reason.to_string(),
    }
}

/// One read of a live process tree.
#[cfg_attr(
    not(any(target_os = "linux", target_os = "macos")),
    allow(
        dead_code,
        reason = "only the Linux and macOS readers build a tree read"
    )
)]
#[derive(Debug, Default)]
struct TreeRead {
    cpu_nanos: u64,
    resident_bytes: u64,
    process_count: u64,
}

fn unix_millis(time: SystemTime) -> i64 {
    match time.duration_since(UNIX_EPOCH) {
        Ok(after) => i64::try_from(after.as_millis()).unwrap_or(i64::MAX),
        Err(before) => -i64::try_from(before.duration().as_millis()).unwrap_or(i64::MAX),
    }
}

/// Format Unix milliseconds as RFC 3339 UTC with three fraction digits.
///
/// Core has no date crate, so this converts days to a civil date with Howard
/// Hinnant's `civil_from_days` algorithm.
fn rfc3339_millis(unix_millis: i64) -> String {
    let days = unix_millis.div_euclid(MILLIS_PER_DAY);
    let millis_of_day = unix_millis.rem_euclid(MILLIS_PER_DAY);
    let (year, month, day) = civil_from_days(days);
    let seconds_of_day = millis_of_day / 1000;
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        seconds_of_day / 3600,
        seconds_of_day / 60 % 60,
        seconds_of_day % 60,
        millis_of_day % 1000,
    )
}

fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

#[cfg(target_os = "linux")]
mod platform {
    //! `/proc` reads: one pass lists every process's parent, then the tree is
    //! read from the root down, each parent before its children.

    use std::collections::{BTreeMap, VecDeque};
    use std::sync::OnceLock;

    use super::TreeRead;

    pub(super) const SUPPORTED: bool = true;

    /// The fields of `/proc/<pid>/stat` a snapshot needs.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) struct Stat {
        pub zombie: bool,
        pub ppid: u32,
        /// `utime + stime + cutime + cstime`, in clock ticks.
        pub cpu_ticks: u64,
        /// `starttime`, in clock ticks since boot.
        pub start: u64,
        pub rss_pages: u64,
    }

    /// Parse a `/proc/<pid>/stat` line. Fields are counted after the last `)`,
    /// because the command name can itself hold spaces and parentheses.
    pub(super) fn parse_stat(text: &str) -> Option<Stat> {
        let (_, rest) = text.rsplit_once(')')?;
        let fields: Vec<&str> = rest.split_whitespace().collect();
        let number = |index: usize| fields.get(index)?.parse::<u64>().ok();
        let ticks = [11, 12, 13, 14]
            .into_iter()
            .map(&number)
            .sum::<Option<u64>>()?;
        Some(Stat {
            zombie: fields.first() == Some(&"Z"),
            ppid: u32::try_from(number(1)?).ok()?,
            cpu_ticks: ticks,
            start: number(19)?,
            rss_pages: number(21)?,
        })
    }

    fn read_stat(pid: u32) -> Option<Stat> {
        parse_stat(&std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?)
    }

    pub(super) fn start_identity(pid: u32) -> Option<u64> {
        read_stat(pid).map(|stat| stat.start)
    }

    fn sysconf(name: libc::c_int) -> u64 {
        // SAFETY: sysconf has no memory-safety preconditions.
        u64::try_from(unsafe { libc::sysconf(name) }).unwrap_or(0)
    }

    fn clock_ticks_per_second() -> u64 {
        static TICKS: OnceLock<u64> = OnceLock::new();
        *TICKS.get_or_init(|| sysconf(libc::_SC_CLK_TCK).max(1))
    }

    fn page_size() -> u64 {
        static PAGE: OnceLock<u64> = OnceLock::new();
        *PAGE.get_or_init(|| sysconf(libc::_SC_PAGESIZE))
    }

    /// Every process's children, each with its start time, or `None` when
    /// `/proc` cannot be listed.
    fn children_by_parent() -> Option<BTreeMap<u32, Vec<(u32, u64)>>> {
        let mut children: BTreeMap<u32, Vec<(u32, u64)>> = BTreeMap::new();
        for entry in std::fs::read_dir("/proc").ok()?.flatten() {
            let Some(pid) = entry
                .file_name()
                .to_str()
                .and_then(|name| name.parse().ok())
            else {
                continue;
            };
            if let Some(stat) = read_stat(pid) {
                children
                    .entry(stat.ppid)
                    .or_default()
                    .push((pid, stat.start));
            }
        }
        Some(children)
    }

    /// Read the tree under `root`, or `None` when the root has exited or its
    /// PID now names a process with another start time than `identity`.
    pub(super) fn read_tree(root: u32, identity: u64) -> Option<TreeRead> {
        let children = children_by_parent()?;
        let mut read = TreeRead::default();
        let mut cpu_ticks: u64 = 0;
        let mut rss_pages: u64 = 0;
        let mut pending = VecDeque::from([(root, identity)]);
        while let Some((pid, expected_start)) = pending.pop_front() {
            let stat = read_stat(pid).filter(|stat| stat.start == expected_start);
            if pid == root && stat.is_none_or(|stat| stat.zombie) {
                return None;
            }
            let Some(stat) = stat else {
                continue;
            };
            cpu_ticks = cpu_ticks.saturating_add(stat.cpu_ticks);
            if !stat.zombie {
                rss_pages = rss_pages.saturating_add(stat.rss_pages);
                read.process_count += 1;
            }
            // A child never starts before its parent; one that seems to reuses a PID.
            let own = children.get(&pid).into_iter().flatten();
            pending.extend(own.filter(|(_, start)| *start >= stat.start).copied());
        }
        read.cpu_nanos = ticks_to_nanos(cpu_ticks, clock_ticks_per_second());
        read.resident_bytes = rss_pages.saturating_mul(page_size());
        Some(read)
    }

    pub(super) fn ticks_to_nanos(ticks: u64, ticks_per_second: u64) -> u64 {
        let nanos = u128::from(ticks) * 1_000_000_000 / u128::from(ticks_per_second);
        u64::try_from(nanos).unwrap_or(u64::MAX)
    }
}

#[cfg(target_os = "macos")]
mod platform {
    //! `proc_pid_rusage` reads, from the root down, each parent before its children.

    use std::collections::{BTreeMap, VecDeque};
    use std::sync::OnceLock;

    use super::TreeRead;

    pub(super) const SUPPORTED: bool = true;

    fn rusage(pid: u32) -> Option<libc::rusage_info_v2> {
        let pid = libc::c_int::try_from(pid).ok()?;
        // SAFETY: rusage_info_v2 is plain data; zero is a valid initial value.
        let mut info: libc::rusage_info_v2 = unsafe { std::mem::zeroed() };
        // SAFETY: `info` is a writable rusage_info_v2, the layout RUSAGE_INFO_V2 names.
        let status = unsafe {
            libc::proc_pid_rusage(
                pid,
                libc::RUSAGE_INFO_V2,
                (&raw mut info).cast::<libc::rusage_info_t>(),
            )
        };
        (status == 0).then_some(info)
    }

    /// `mach_timebase_info_data_t`.
    #[repr(C)]
    struct MachTimebaseInfo {
        numer: u32,
        denom: u32,
    }

    // libc marks its binding deprecated in favour of the `mach2` crate; one
    // system call does not justify a new dependency.
    unsafe extern "C" {
        fn mach_timebase_info(info: *mut MachTimebaseInfo) -> libc::c_int;
    }

    /// Nanoseconds per Mach absolute time unit, as `(numer, denom)`, or `None`
    /// when the kernel does not report it.
    fn timebase() -> Option<(u64, u64)> {
        static TIMEBASE: OnceLock<Option<(u64, u64)>> = OnceLock::new();
        *TIMEBASE.get_or_init(|| {
            let mut info = MachTimebaseInfo { numer: 0, denom: 0 };
            // SAFETY: `info` is a writable mach_timebase_info_data_t.
            let status = unsafe { mach_timebase_info(&mut info) };
            (status == 0 && info.denom != 0).then(|| (u64::from(info.numer), u64::from(info.denom)))
        })
    }

    /// Convert Mach absolute time units, which `rusage_info` CPU times use on
    /// Apple silicon, to nanoseconds.
    pub(super) fn mach_to_nanos(units: u64, (numer, denom): (u64, u64)) -> u64 {
        let nanos = u128::from(units) * u128::from(numer) / u128::from(denom);
        u64::try_from(nanos).unwrap_or(u64::MAX)
    }

    pub(super) fn start_identity(pid: u32) -> Option<u64> {
        rusage(pid).map(|info| info.ri_proc_start_abstime)
    }

    /// Every live process's children. One `proc_listallpids` call and one
    /// `proc_pidinfo` per process: `proc_listchildpids` scans the whole
    /// process table on each call, which costs too much for a wide tree.
    /// `PROC_PIDTBSDINFO` fails for a zombie, so a zombie is never listed.
    /// Returns `None` when the process table cannot be listed.
    fn children_by_parent() -> Option<BTreeMap<u32, Vec<u32>>> {
        let mut pids = vec![0 as libc::pid_t; 1024];
        let count = loop {
            let bytes = libc::c_int::try_from(std::mem::size_of_val(pids.as_slice())).ok()?;
            // SAFETY: the buffer is writable and sized in bytes as libproc requires.
            let count = unsafe { libc::proc_listallpids(pids.as_mut_ptr().cast(), bytes) };
            let count = usize::try_from(count).ok().filter(|count| *count > 0)?;
            if count < pids.len() {
                break count;
            }
            pids.resize(pids.len() * 2, 0);
        };
        let mut children: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for &pid in &pids[..count] {
            if let (Ok(pid), Some(ppid)) = (u32::try_from(pid), parent_pid(pid)) {
                children.entry(ppid).or_default().push(pid);
            }
        }
        Some(children)
    }

    /// The parent of a live process, or `None` for a zombie or a missing process.
    fn parent_pid(pid: libc::pid_t) -> Option<u32> {
        // SAFETY: proc_bsdinfo is plain data; zero is a valid initial value.
        let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
        let size = libc::c_int::try_from(std::mem::size_of::<libc::proc_bsdinfo>()).ok()?;
        // SAFETY: `info` is writable and `size` bytes long, as PROC_PIDTBSDINFO needs.
        let written = unsafe {
            libc::proc_pidinfo(pid, libc::PROC_PIDTBSDINFO, 0, (&raw mut info).cast(), size)
        };
        (written == size).then_some(info.pbi_ppid)
    }

    /// Read the tree under `root`, or `None` when the root has exited or its
    /// PID now names a process with another start time than `identity`. A
    /// child that seems to start before its parent reuses a PID and is left out.
    pub(super) fn read_tree(root: u32, identity: u64) -> Option<TreeRead> {
        parent_pid(libc::pid_t::try_from(root).ok()?)?;
        let root_info = rusage(root).filter(|info| info.ri_proc_start_abstime == identity)?;
        let timebase = timebase()?;
        let children = children_by_parent()?;
        let mut read = TreeRead::default();
        let mut cpu_units: u64 = 0;
        let mut resident: u64 = 0;
        let mut pending = VecDeque::from([(root, root_info)]);
        while let Some((pid, info)) = pending.pop_front() {
            cpu_units = cpu_units
                .saturating_add(info.ri_user_time)
                .saturating_add(info.ri_system_time)
                .saturating_add(info.ri_child_user_time)
                .saturating_add(info.ri_child_system_time);
            resident = resident.saturating_add(info.ri_resident_size);
            read.process_count += 1;
            let start = info.ri_proc_start_abstime;
            let own = children.get(&pid).into_iter().flatten();
            pending.extend(
                own.filter_map(|child| rusage(*child).map(|info| (*child, info)))
                    // Skip a child that exited after it was listed, and one
                    // that seems to start before its parent (a reused PID).
                    .filter(|(_, info)| {
                        info.ri_proc_exit_abstime == 0 && info.ri_proc_start_abstime >= start
                    }),
            );
        }
        read.cpu_nanos = mach_to_nanos(cpu_units, timebase);
        read.resident_bytes = resident;
        Some(read)
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod platform {
    use super::TreeRead;

    pub(super) const SUPPORTED: bool = false;

    pub(super) fn start_identity(_pid: u32) -> Option<u64> {
        None
    }

    pub(super) fn read_tree(_root: u32, _identity: u64) -> Option<TreeRead> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_millis_formats_fixed_vectors() {
        assert_eq!(rfc3339_millis(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(
            rfc3339_millis(1_792_000_870_205),
            "2026-10-14T18:01:10.205Z"
        );
        assert_eq!(rfc3339_millis(951_782_400_000), "2000-02-29T00:00:00.000Z");
        assert_eq!(
            rfc3339_millis(4_107_542_399_999),
            "2100-02-28T23:59:59.999Z"
        );
        assert_eq!(rfc3339_millis(-1), "1969-12-31T23:59:59.999Z");
    }

    #[test]
    fn snapshot_local_host_without_active_host_is_unavailable_or_unsupported() {
        let error = snapshot_local_host("runtime-1", None).expect_err("no host");

        if platform::SUPPORTED {
            assert!(matches!(
                error,
                FabricError::ResourceObservationUnavailable { ref runtime_id, .. }
                    if runtime_id == "runtime-1"
            ));
        } else {
            assert!(matches!(
                error,
                FabricError::ResourceObservationUnsupported { .. }
            ));
        }
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn snapshot_local_host_reads_this_process_tree() {
        let host = LocalHostProcess::new(std::process::id(), SystemTime::now());

        let snapshot = snapshot_local_host("runtime-1", Some(&host)).expect("snapshot");

        assert_eq!(snapshot.runtime_id, "runtime-1");
        assert!(snapshot.process_count >= 1);
        assert!(snapshot.resident_memory_bytes > 0);
        // Linux counts CPU in 10 ms ticks, so a young test process can read zero.
        assert!(snapshot.cpu_time_seconds.is_finite() && snapshot.cpu_time_seconds >= 0.0);
        assert!(snapshot.observed_at >= snapshot.started_at);
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn snapshot_local_host_with_reused_pid_identity_is_unavailable() {
        let mut host = LocalHostProcess::new(std::process::id(), SystemTime::now());
        host.identity = host.identity.map(|identity| identity.wrapping_add(1));

        let error = snapshot_local_host("runtime-1", Some(&host)).expect_err("other process");

        assert!(matches!(
            error,
            FabricError::ResourceObservationUnavailable { .. }
        ));
    }

    /// A host that exited but is not yet reaped is not a live process.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn snapshot_local_host_with_unreaped_exited_host_is_unavailable() {
        let mut child = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn child");
        let host = LocalHostProcess::new(child.id(), SystemTime::now());
        child.kill().expect("kill child");
        std::thread::sleep(std::time::Duration::from_millis(200));

        let error = snapshot_local_host("runtime-1", Some(&host)).expect_err("zombie host");
        child.wait().expect("reap child");

        assert!(matches!(
            error,
            FabricError::ResourceObservationUnavailable { .. }
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_stat_parse_reads_cpu_with_waited_children_start_time_and_ticks() {
        let stat = "42 (a (weird) name) S 7 42 42 0 -1 4194560 100 0 0 0 30 20 5 4 20 0 1 0 \
                    9001 1000000 250 18446744073709551615";

        let parsed = platform::parse_stat(stat).expect("stat parses");

        assert_eq!(parsed.ppid, 7);
        assert_eq!(parsed.cpu_ticks, 30 + 20 + 5 + 4);
        assert_eq!(parsed.start, 9001);
        assert_eq!(parsed.rss_pages, 250);
        assert!(!parsed.zombie);
        assert_eq!(platform::ticks_to_nanos(150, 100), 1_500_000_000);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_mach_units_convert_with_the_timebase() {
        // Apple silicon's timebase: 125 ns per 3 units.
        assert_eq!(platform::mach_to_nanos(24_000_000, (125, 3)), 1_000_000_000);
        assert_eq!(
            platform::mach_to_nanos(1_000_000_000, (1, 1)),
            1_000_000_000
        );
    }
}
