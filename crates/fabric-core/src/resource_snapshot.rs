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
//! A live process in the tree that cannot be read, or a process table that
//! cannot be listed in full, makes the snapshot unavailable instead of
//! leaving the process out.
//!
//! Linux reads `/proc/<pid>/stat` and macOS reads `proc_pid_rusage`. To find
//! the tree, the macOS reader, and the Linux reader on a kernel without
//! `/proc/<pid>/task/<tid>/children`, also read other processes' parent IDs.
//! A snapshot exposes only its own fields: never command lines, environment
//! variables, open files, or processes outside the tree.

use std::time::{SystemTime, UNIX_EPOCH};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{FabricError, Result};

const MILLIS_PER_DAY: i64 = 86_400_000;
const NANOS_PER_SECOND: f64 = 1_000_000_000.0;

/// Reasons a tree read returns when the snapshot is unavailable.
#[cfg_attr(
    not(any(target_os = "linux", target_os = "macos")),
    allow(dead_code, reason = "only the Linux and macOS readers read a tree")
)]
mod reason {
    pub(super) const HOST_EXITED: &str = "adapter host process has exited";
    pub(super) const PROCESS_UNREADABLE: &str =
        "a live process in the runtime's process tree cannot be read";
    pub(super) const TABLE_UNREADABLE: &str = "the process table cannot be read in full";
}

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
    let identity = host.identity.ok_or_else(|| {
        unavailable(
            runtime_id,
            "adapter host process could not be read when the runtime started",
        )
    })?;
    let read = platform::read_tree(host.pid, identity)
        .map_err(|reason| unavailable(runtime_id, reason))?;
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
    //! `/proc` reads, from the root down, each parent before its children.
    //! Each process's children come from `/proc/<pid>/task/<tid>/children`,
    //! so a read visits only the runtime's own processes. A kernel built
    //! without those files falls back to one pass over every process's parent.

    use std::collections::{BTreeMap, VecDeque};
    use std::sync::OnceLock;

    use super::{TreeRead, reason};

    pub(super) const SUPPORTED: bool = true;

    /// Why one process could not be read.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) enum ReadError {
        /// The process exited.
        Gone,
        /// The process exists but cannot be read, or its record cannot be parsed.
        Unreadable,
    }

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

    /// Times a process's children are read while waiting for two reads that agree.
    const MAX_CHILDREN_READS: usize = 4;

    /// The `/proc` reads a tree read makes. Tests script them.
    pub(super) trait Procfs {
        /// Whether the kernel provides `/proc/<pid>/task/<tid>/children`.
        fn has_children_files(&self) -> bool;
        /// The children of every thread of `pid`, sorted.
        fn children(&self, pid: u32) -> Result<Vec<u32>, ReadError>;
        /// Every PID in `/proc`.
        fn pids(&self) -> Result<Vec<u32>, ReadError>;
        fn stat(&self, pid: u32) -> Result<Stat, ReadError>;
    }

    struct Live;

    impl Procfs for Live {
        fn has_children_files(&self) -> bool {
            static PRESENT: OnceLock<bool> = OnceLock::new();
            *PRESENT.get_or_init(|| {
                let pid = std::process::id();
                std::path::Path::new(&format!("/proc/{pid}/task/{pid}/children")).exists()
            })
        }

        fn children(&self, pid: u32) -> Result<Vec<u32>, ReadError> {
            read_until_stable(|| children_once(pid))
        }

        fn pids(&self) -> Result<Vec<u32>, ReadError> {
            let mut pids = Vec::new();
            for entry in std::fs::read_dir("/proc").map_err(read_error)? {
                let name = entry.map_err(read_error)?.file_name();
                if let Some(pid) = name.to_str().and_then(|name| name.parse().ok()) {
                    pids.push(pid);
                }
            }
            Ok(pids)
        }

        fn stat(&self, pid: u32) -> Result<Stat, ReadError> {
            let text = std::fs::read_to_string(format!("/proc/{pid}/stat")).map_err(read_error)?;
            parse_stat(&text).ok_or(ReadError::Unreadable)
        }
    }

    /// One read of the children files of every thread of `pid`. A thread that
    /// exited during the read is skipped; its children move to another thread.
    fn children_once(pid: u32) -> Result<Vec<u32>, ReadError> {
        let mut children = Vec::new();
        for entry in std::fs::read_dir(format!("/proc/{pid}/task")).map_err(read_error)? {
            let tid = entry.map_err(read_error)?.file_name();
            let path = format!("/proc/{pid}/task/{}/children", tid.to_string_lossy());
            let text = match std::fs::read_to_string(path).map_err(read_error) {
                Ok(text) => text,
                Err(ReadError::Gone) => continue,
                Err(unreadable) => return Err(unreadable),
            };
            for child in text.split_whitespace() {
                children.push(child.parse().map_err(|_| ReadError::Unreadable)?);
            }
        }
        children.sort_unstable();
        Ok(children)
    }

    /// Read a process's children until two reads in a row agree. The kernel
    /// can skip a live child when a sibling is reaped during the read, and a
    /// thread's children move when it exits; either changes the next read.
    pub(super) fn read_until_stable(
        mut read: impl FnMut() -> Result<Vec<u32>, ReadError>,
    ) -> Result<Vec<u32>, ReadError> {
        let mut last = read()?;
        for _ in 1..MAX_CHILDREN_READS {
            let next = read()?;
            if next == last {
                return Ok(next);
            }
            last = next;
        }
        Err(ReadError::Unreadable)
    }

    /// `ENOENT` and `ESRCH` mean the process exited; anything else means it
    /// cannot be read.
    fn read_error(error: std::io::Error) -> ReadError {
        if error.kind() == std::io::ErrorKind::NotFound || error.raw_os_error() == Some(libc::ESRCH)
        {
            ReadError::Gone
        } else {
            ReadError::Unreadable
        }
    }

    pub(super) fn start_identity(pid: u32) -> Option<u64> {
        Live.stat(pid).ok().map(|stat| stat.start)
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

    pub(super) fn read_tree(root: u32, identity: u64) -> Result<TreeRead, &'static str> {
        read_tree_with(&Live, root, identity, clock_ticks_per_second(), page_size())
    }

    /// Every process's children, for a kernel without children files. A
    /// process that exited during the pass is left out; one that cannot be
    /// read leaves the listing incomplete.
    fn children_by_parent(procfs: &impl Procfs) -> Result<BTreeMap<u32, Vec<u32>>, &'static str> {
        let pids = procfs.pids().map_err(|_| reason::TABLE_UNREADABLE)?;
        let mut children: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for pid in pids {
            match procfs.stat(pid) {
                Ok(stat) => children.entry(stat.ppid).or_default().push(pid),
                Err(ReadError::Gone) => {}
                Err(ReadError::Unreadable) => return Err(reason::TABLE_UNREADABLE),
            }
        }
        Ok(children)
    }

    /// Read the tree under `root`. Fails when the root exited or its PID now
    /// names a process with another start time than `identity`, or when a
    /// live process in the tree cannot be read.
    pub(super) fn read_tree_with(
        procfs: &impl Procfs,
        root: u32,
        identity: u64,
        ticks_per_second: u64,
        page_size: u64,
    ) -> Result<TreeRead, &'static str> {
        let listing = if procfs.has_children_files() {
            None
        } else {
            Some(children_by_parent(procfs)?)
        };
        let mut read = TreeRead::default();
        let mut cpu_ticks: u64 = 0;
        let mut rss_pages: u64 = 0;
        // Each entry is a PID and the parent it was listed under (none for the root).
        let mut pending = VecDeque::from([(root, None)]);
        while let Some((pid, parent)) = pending.pop_front() {
            // List the children before the stat read: a stat that still names
            // the expected process shows the listed children were its own.
            let children = match &listing {
                Some(listing) => Ok(listing.get(&pid).cloned().unwrap_or_default()),
                None => procfs.children(pid),
            };
            let found = children.and_then(|children| Ok((procfs.stat(pid)?, children)));
            let expected = |stat: &Stat| match parent {
                None => stat.start == identity && !stat.zombie,
                Some(parent) => stat.ppid == parent,
            };
            let (stat, children) = match found {
                Ok((stat, children)) if expected(&stat) => (stat, children),
                // An exited child, or one whose PID now names another process.
                Ok(_) | Err(ReadError::Gone) if parent.is_some() => continue,
                Ok(_) | Err(ReadError::Gone) => return Err(reason::HOST_EXITED),
                Err(ReadError::Unreadable) => return Err(reason::PROCESS_UNREADABLE),
            };
            cpu_ticks = cpu_ticks.saturating_add(stat.cpu_ticks);
            if !stat.zombie {
                rss_pages = rss_pages.saturating_add(stat.rss_pages);
                read.process_count += 1;
            }
            pending.extend(children.into_iter().map(|child| (child, Some(pid))));
        }
        read.cpu_nanos = ticks_to_nanos(cpu_ticks, ticks_per_second);
        read.resident_bytes = rss_pages.saturating_mul(page_size);
        Ok(read)
    }

    pub(super) fn ticks_to_nanos(ticks: u64, ticks_per_second: u64) -> u64 {
        let nanos = u128::from(ticks) * 1_000_000_000 / u128::from(ticks_per_second);
        u64::try_from(nanos).unwrap_or(u64::MAX)
    }
}

#[cfg(target_os = "macos")]
mod platform {
    //! `libproc` reads. One pass lists every process's parent and start time,
    //! then the tree is read from the root down, each parent before its
    //! children.

    use std::collections::{BTreeMap, VecDeque};
    use std::sync::OnceLock;

    use super::{TreeRead, reason};

    pub(super) const SUPPORTED: bool = true;

    /// Why one process could not be read.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) enum ReadError {
        /// The process exited, or is a zombie the call cannot describe.
        Gone,
        /// The process exists but this process may not read it.
        Denied,
    }

    /// A process's parent and start time, which together identify it.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) struct Parent {
        pub ppid: u32,
        /// `pbi_start_tvsec` and `pbi_start_tvusec`, or `None` for another
        /// user's process, whose parent only the short record shows.
        pub started: Option<(u64, u64)>,
    }

    /// `struct proc_bsdshortinfo` from `<sys/proc_info.h>`, which the `libc`
    /// crate version in use does not bind: PID, parent PID, then 56 bytes of
    /// group, status, name, flags, and user and group IDs that the reader ignores.
    #[repr(C)]
    struct ProcBsdShortInfo {
        pid: u32,
        ppid: u32,
        rest: [u32; 14],
    }

    const PROC_PIDT_SHORTBSDINFO: libc::c_int = 13;

    /// The `proc_pid_rusage` fields a snapshot needs, CPU times in Mach units.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) struct Usage {
        pub cpu_units: u64,
        pub resident_bytes: u64,
        /// `ri_proc_start_abstime`.
        pub start: u64,
        pub exited: bool,
    }

    /// The process reads a tree read makes. Tests script them.
    pub(super) trait Libproc {
        /// Every PID, or `None` when the process table cannot be listed.
        fn pids(&self) -> Option<Vec<u32>>;
        fn parent(&self, pid: u32) -> Result<Parent, ReadError>;
        fn usage(&self, pid: u32) -> Result<Usage, ReadError>;
    }

    struct Live;

    impl Libproc for Live {
        fn pids(&self) -> Option<Vec<u32>> {
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
            Some(
                pids[..count]
                    .iter()
                    .filter_map(|pid| u32::try_from(*pid).ok())
                    .collect(),
            )
        }

        /// `PROC_PIDTBSDINFO` fails for a zombie, so a zombie reads as gone.
        /// It is denied for another user's process, whose parent the short
        /// record still shows.
        fn parent(&self, pid: u32) -> Result<Parent, ReadError> {
            let pid = libc::pid_t::try_from(pid).map_err(|_| ReadError::Gone)?;
            // SAFETY: proc_bsdinfo is plain data; zero is a valid initial value.
            let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
            let size = libc::c_int::try_from(std::mem::size_of::<libc::proc_bsdinfo>())
                .map_err(|_| ReadError::Denied)?;
            // SAFETY: `info` is writable and `size` bytes long, as PROC_PIDTBSDINFO needs.
            let written = unsafe {
                libc::proc_pidinfo(pid, libc::PROC_PIDTBSDINFO, 0, (&raw mut info).cast(), size)
            };
            if written == size {
                return Ok(Parent {
                    ppid: info.pbi_ppid,
                    started: Some((info.pbi_start_tvsec, info.pbi_start_tvusec)),
                });
            }
            match last_read_error() {
                ReadError::Denied => short_parent(pid),
                gone => Err(gone),
            }
        }

        fn usage(&self, pid: u32) -> Result<Usage, ReadError> {
            let pid = libc::c_int::try_from(pid).map_err(|_| ReadError::Gone)?;
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
            if status != 0 {
                return Err(last_read_error());
            }
            Ok(Usage {
                cpu_units: info
                    .ri_user_time
                    .saturating_add(info.ri_system_time)
                    .saturating_add(info.ri_child_user_time)
                    .saturating_add(info.ri_child_system_time),
                resident_bytes: info.ri_resident_size,
                start: info.ri_proc_start_abstime,
                exited: info.ri_proc_exit_abstime != 0,
            })
        }
    }

    fn short_parent(pid: libc::pid_t) -> Result<Parent, ReadError> {
        let mut info = ProcBsdShortInfo {
            pid: 0,
            ppid: 0,
            rest: [0; 14],
        };
        let size = libc::c_int::try_from(std::mem::size_of::<ProcBsdShortInfo>())
            .map_err(|_| ReadError::Denied)?;
        // SAFETY: `info` is writable and `size` bytes long, as PROC_PIDT_SHORTBSDINFO needs.
        let written = unsafe {
            libc::proc_pidinfo(pid, PROC_PIDT_SHORTBSDINFO, 0, (&raw mut info).cast(), size)
        };
        if written == size {
            return Ok(Parent {
                ppid: info.ppid,
                started: None,
            });
        }
        Err(last_read_error())
    }

    /// Classify the error of the libproc call that just failed.
    fn last_read_error() -> ReadError {
        match std::io::Error::last_os_error().raw_os_error() {
            Some(libc::ESRCH) => ReadError::Gone,
            _ => ReadError::Denied,
        }
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
        Live.usage(pid).ok().map(|usage| usage.start)
    }

    pub(super) fn read_tree(root: u32, identity: u64) -> Result<TreeRead, &'static str> {
        let timebase = timebase().ok_or(reason::PROCESS_UNREADABLE)?;
        read_tree_with(&Live, root, identity, timebase)
    }

    /// Child PIDs, each with its parent record, by parent PID.
    type ChildrenByParent = BTreeMap<u32, Vec<(u32, Parent)>>;

    /// Every live process's children, each with its parent record. One
    /// `proc_listallpids` call and one `proc_pidinfo` per process:
    /// `proc_listchildpids` scans the whole process table on each call, which
    /// costs too much for a wide tree. A process that exited during the pass
    /// is left out; one that cannot be read leaves the listing incomplete.
    fn children_by_parent(procs: &impl Libproc) -> Result<ChildrenByParent, &'static str> {
        let pids = procs.pids().ok_or(reason::TABLE_UNREADABLE)?;
        let mut children = ChildrenByParent::new();
        for pid in pids {
            match procs.parent(pid) {
                Ok(parent) => children.entry(parent.ppid).or_default().push((pid, parent)),
                Err(ReadError::Gone) => {}
                Err(ReadError::Denied) => return Err(reason::TABLE_UNREADABLE),
            }
        }
        Ok(children)
    }

    /// Read the tree under `root`. Fails when the root exited or its PID now
    /// names a process with another start time than `identity`, or when a
    /// live process in the tree cannot be read.
    pub(super) fn read_tree_with(
        procs: &impl Libproc,
        root: u32,
        identity: u64,
        timebase: (u64, u64),
    ) -> Result<TreeRead, &'static str> {
        let children = children_by_parent(procs)?;
        // Read after the listing: a root that is still the same live process
        // was alive during the whole listing, so its listed children are its own.
        let root_usage = match procs.usage(root) {
            Ok(usage) if usage.start == identity && !usage.exited => usage,
            Ok(_) | Err(ReadError::Gone) => return Err(reason::HOST_EXITED),
            Err(ReadError::Denied) => return Err(reason::PROCESS_UNREADABLE),
        };
        let mut read = TreeRead::default();
        let mut cpu_units: u64 = 0;
        let mut pending = VecDeque::from([(root, root_usage)]);
        while let Some((pid, usage)) = pending.pop_front() {
            cpu_units = cpu_units.saturating_add(usage.cpu_units);
            read.resident_bytes = read.resident_bytes.saturating_add(usage.resident_bytes);
            read.process_count += 1;
            for &(child, listed) in children.get(&pid).into_iter().flatten() {
                if let Some(usage) = listed_child_usage(procs, child, listed)? {
                    pending.push_back((child, usage));
                }
            }
        }
        read.cpu_nanos = mach_to_nanos(cpu_units, timebase);
        Ok(read)
    }

    /// The usage of `child`, or `None` when it exited or its PID no longer
    /// names the listed process. A process that still has the listed parent
    /// and start time after the usage read was the listed one throughout, so
    /// the usage is its own. Fails when the child is live but cannot be read.
    fn listed_child_usage(
        procs: &impl Libproc,
        child: u32,
        listed: Parent,
    ) -> Result<Option<Usage>, &'static str> {
        if listed.started.is_none() {
            return Err(reason::PROCESS_UNREADABLE);
        }
        let usage = match procs.usage(child) {
            Ok(usage) if !usage.exited => usage,
            Ok(_) | Err(ReadError::Gone) => return Ok(None),
            Err(ReadError::Denied) => return Err(reason::PROCESS_UNREADABLE),
        };
        match procs.parent(child) {
            Ok(now) if now == listed => Ok(Some(usage)),
            Ok(now) if now.ppid == listed.ppid && now.started.is_none() => {
                Err(reason::PROCESS_UNREADABLE)
            }
            Ok(_) | Err(ReadError::Gone) => Ok(None),
            Err(ReadError::Denied) => Err(reason::PROCESS_UNREADABLE),
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod platform {
    use super::TreeRead;

    pub(super) const SUPPORTED: bool = false;

    pub(super) fn start_identity(_pid: u32) -> Option<u64> {
        None
    }

    pub(super) fn read_tree(_root: u32, _identity: u64) -> Result<TreeRead, &'static str> {
        Err("no process reader for this operating system")
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

    /// Scripted `libproc` answers for the macOS tree walk.
    #[cfg(target_os = "macos")]
    #[derive(Default)]
    struct ScriptedLibproc {
        pids: Option<Vec<u32>>,
        /// Answers to `parent`, first the listing's, then the later checks'.
        parents: std::collections::BTreeMap<
            u32,
            std::cell::RefCell<Vec<std::result::Result<platform::Parent, platform::ReadError>>>,
        >,
        usages: std::collections::BTreeMap<
            u32,
            std::result::Result<platform::Usage, platform::ReadError>,
        >,
    }

    #[cfg(target_os = "macos")]
    impl ScriptedLibproc {
        fn process(
            mut self,
            pid: u32,
            parents: Vec<std::result::Result<platform::Parent, platform::ReadError>>,
            usage: std::result::Result<platform::Usage, platform::ReadError>,
        ) -> Self {
            self.pids.get_or_insert_with(Vec::new).push(pid);
            self.parents.insert(pid, std::cell::RefCell::new(parents));
            self.usages.insert(pid, usage);
            self
        }
    }

    #[cfg(target_os = "macos")]
    impl platform::Libproc for ScriptedLibproc {
        fn pids(&self) -> Option<Vec<u32>> {
            self.pids.clone()
        }

        fn parent(&self, pid: u32) -> std::result::Result<platform::Parent, platform::ReadError> {
            let mut answers = self.parents.get(&pid).expect("scripted pid").borrow_mut();
            if answers.len() > 1 {
                answers.remove(0)
            } else {
                answers[0]
            }
        }

        fn usage(&self, pid: u32) -> std::result::Result<platform::Usage, platform::ReadError> {
            self.usages[&pid]
        }
    }

    #[cfg(target_os = "macos")]
    fn mac_parent(ppid: u32, started: u64) -> platform::Parent {
        platform::Parent {
            ppid,
            started: Some((started, 0)),
        }
    }

    #[cfg(target_os = "macos")]
    fn mac_usage(start: u64, cpu_units: u64) -> platform::Usage {
        platform::Usage {
            cpu_units,
            resident_bytes: 1000,
            start,
            exited: false,
        }
    }

    /// The host (PID 10) has one listed child (PID 11). After the listing the
    /// child exits and an unrelated, newer process takes PID 11.
    #[cfg(target_os = "macos")]
    #[test]
    fn macos_tree_read_skips_child_pid_reused_after_listing() {
        let procs = ScriptedLibproc::default()
            .process(10, vec![Ok(mac_parent(1, 100))], Ok(mac_usage(5, 7)))
            .process(
                11,
                vec![Ok(mac_parent(10, 200)), Ok(mac_parent(1, 300))],
                Ok(mac_usage(9, 1_000_000)),
            );

        let read = platform::read_tree_with(&procs, 10, 5, (1, 1)).expect("tree read");

        assert_eq!(read.process_count, 1);
        assert_eq!(read.cpu_nanos, 7);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_tree_read_counts_child_that_is_still_the_listed_process() {
        let procs = ScriptedLibproc::default()
            .process(10, vec![Ok(mac_parent(1, 100))], Ok(mac_usage(5, 7)))
            .process(11, vec![Ok(mac_parent(10, 200))], Ok(mac_usage(9, 3)));

        let read = platform::read_tree_with(&procs, 10, 5, (1, 1)).expect("tree read");

        assert_eq!(read.process_count, 2);
        assert_eq!(read.cpu_nanos, 10);
        assert_eq!(read.resident_bytes, 2000);
    }

    #[cfg(target_os = "macos")]
    fn mac_tree_with_child(
        child_parents: Vec<std::result::Result<platform::Parent, platform::ReadError>>,
        child_usage: std::result::Result<platform::Usage, platform::ReadError>,
    ) -> std::result::Result<TreeRead, &'static str> {
        let procs = ScriptedLibproc::default()
            .process(10, vec![Ok(mac_parent(1, 100))], Ok(mac_usage(5, 7)))
            .process(11, child_parents, child_usage);
        platform::read_tree_with(&procs, 10, 5, (1, 1))
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_tree_read_with_unreadable_live_child_is_unavailable() {
        let denied_usage = mac_tree_with_child(
            vec![Ok(mac_parent(10, 200))],
            Err(platform::ReadError::Denied),
        );
        let other_user = mac_tree_with_child(
            vec![Ok(platform::Parent {
                ppid: 10,
                started: None,
            })],
            Ok(mac_usage(9, 3)),
        );
        let denied_recheck = mac_tree_with_child(
            vec![Ok(mac_parent(10, 200)), Err(platform::ReadError::Denied)],
            Ok(mac_usage(9, 3)),
        );

        assert_eq!(denied_usage.unwrap_err(), reason::PROCESS_UNREADABLE);
        assert_eq!(other_user.unwrap_err(), reason::PROCESS_UNREADABLE);
        assert_eq!(denied_recheck.unwrap_err(), reason::PROCESS_UNREADABLE);
    }

    /// The child is listed, then exits before its usage read or before the
    /// identity check that follows it.
    #[cfg(target_os = "macos")]
    #[test]
    fn macos_tree_read_skips_child_reaped_between_reads() {
        let before_usage = mac_tree_with_child(
            vec![Ok(mac_parent(10, 200))],
            Err(platform::ReadError::Gone),
        )
        .expect("tree read");
        let before_check = mac_tree_with_child(
            vec![Ok(mac_parent(10, 200)), Err(platform::ReadError::Gone)],
            Ok(mac_usage(9, 3)),
        )
        .expect("tree read");
        let exited = mac_tree_with_child(
            vec![Ok(mac_parent(10, 200))],
            Ok(platform::Usage {
                exited: true,
                ..mac_usage(9, 3)
            }),
        )
        .expect("tree read");

        for read in [before_usage, before_check, exited] {
            assert_eq!(read.process_count, 1);
            assert_eq!(read.cpu_nanos, 7);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_tree_read_with_incomplete_process_table_is_unavailable() {
        let unlisted = ScriptedLibproc::default();
        let unreadable_other = ScriptedLibproc::default()
            .process(10, vec![Ok(mac_parent(1, 100))], Ok(mac_usage(5, 7)))
            .process(
                50,
                vec![Err(platform::ReadError::Denied)],
                Err(platform::ReadError::Denied),
            );

        assert_eq!(
            platform::read_tree_with(&unlisted, 10, 5, (1, 1)).unwrap_err(),
            reason::TABLE_UNREADABLE
        );
        assert_eq!(
            platform::read_tree_with(&unreadable_other, 10, 5, (1, 1)).unwrap_err(),
            reason::TABLE_UNREADABLE
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_tree_read_of_exited_or_unreadable_host_is_unavailable() {
        let host = |usage| {
            let procs = ScriptedLibproc::default().process(10, vec![Ok(mac_parent(1, 100))], usage);
            platform::read_tree_with(&procs, 10, 5, (1, 1)).unwrap_err()
        };

        assert_eq!(host(Err(platform::ReadError::Gone)), reason::HOST_EXITED);
        assert_eq!(host(Ok(mac_usage(6, 7))), reason::HOST_EXITED);
        assert_eq!(
            host(Ok(platform::Usage {
                exited: true,
                ..mac_usage(5, 7)
            })),
            reason::HOST_EXITED
        );
        assert_eq!(
            host(Err(platform::ReadError::Denied)),
            reason::PROCESS_UNREADABLE
        );
    }

    /// Scripted `/proc` answers for the Linux tree walk.
    #[cfg(target_os = "linux")]
    #[derive(Default)]
    struct ScriptedProcfs {
        /// Children by parent from children files, or `None` for a kernel
        /// without them.
        children: Option<
            std::collections::BTreeMap<u32, std::result::Result<Vec<u32>, platform::ReadError>>,
        >,
        pids: Vec<u32>,
        unlisted: bool,
        /// Answers to `stat`, first the listing's, then the walk's.
        stats: std::collections::BTreeMap<
            u32,
            std::cell::RefCell<Vec<std::result::Result<platform::Stat, platform::ReadError>>>,
        >,
    }

    #[cfg(target_os = "linux")]
    impl ScriptedProcfs {
        fn process(
            mut self,
            pid: u32,
            stats: Vec<std::result::Result<platform::Stat, platform::ReadError>>,
        ) -> Self {
            self.pids.push(pid);
            self.stats.insert(pid, std::cell::RefCell::new(stats));
            self
        }
    }

    #[cfg(target_os = "linux")]
    impl platform::Procfs for ScriptedProcfs {
        fn has_children_files(&self) -> bool {
            self.children.is_some()
        }

        fn children(&self, pid: u32) -> std::result::Result<Vec<u32>, platform::ReadError> {
            let children = self.children.as_ref().expect("children files");
            children.get(&pid).cloned().unwrap_or(Ok(Vec::new()))
        }

        fn pids(&self) -> std::result::Result<Vec<u32>, platform::ReadError> {
            if self.unlisted {
                return Err(platform::ReadError::Unreadable);
            }
            Ok(self.pids.clone())
        }

        fn stat(&self, pid: u32) -> std::result::Result<platform::Stat, platform::ReadError> {
            let mut answers = self.stats.get(&pid).expect("scripted pid").borrow_mut();
            if answers.len() > 1 {
                answers.remove(0)
            } else {
                answers[0]
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn linux_stat(ppid: u32, start: u64, cpu_ticks: u64) -> platform::Stat {
        platform::Stat {
            zombie: false,
            ppid,
            cpu_ticks,
            start,
            rss_pages: 1,
        }
    }

    /// The host is PID 10 started at tick 5; its child is PID 11.
    #[cfg(target_os = "linux")]
    fn linux_tree_with_child(
        child: Vec<std::result::Result<platform::Stat, platform::ReadError>>,
    ) -> std::result::Result<TreeRead, &'static str> {
        let procfs = ScriptedProcfs::default()
            .process(10, vec![Ok(linux_stat(1, 5, 7))])
            .process(11, child);
        platform::read_tree_with(&procfs, 10, 5, 1, 1)
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_tree_read_with_unreadable_live_child_is_unavailable() {
        let read = linux_tree_with_child(vec![
            Ok(linux_stat(10, 9, 3)),
            Err(platform::ReadError::Unreadable),
        ]);

        assert_eq!(read.unwrap_err(), reason::PROCESS_UNREADABLE);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_tree_read_skips_child_reaped_or_replaced_between_reads() {
        let reaped = linux_tree_with_child(vec![
            Ok(linux_stat(10, 9, 3)),
            Err(platform::ReadError::Gone),
        ])
        .expect("tree read");
        let replaced =
            linux_tree_with_child(vec![Ok(linux_stat(10, 9, 3)), Ok(linux_stat(1, 12, 900))])
                .expect("tree read");
        let counted = linux_tree_with_child(vec![Ok(linux_stat(10, 9, 3))]).expect("tree read");

        for read in [reaped, replaced] {
            assert_eq!(read.process_count, 1);
            assert_eq!(read.cpu_nanos, 7_000_000_000);
        }
        assert_eq!(counted.process_count, 2);
        assert_eq!(counted.cpu_nanos, 10_000_000_000);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_tree_read_with_incomplete_process_table_is_unavailable() {
        let unlisted = ScriptedProcfs {
            unlisted: true,
            ..ScriptedProcfs::default()
        };
        let unreadable_other = ScriptedProcfs::default()
            .process(10, vec![Ok(linux_stat(1, 5, 7))])
            .process(50, vec![Err(platform::ReadError::Unreadable)]);

        assert_eq!(
            platform::read_tree_with(&unlisted, 10, 5, 1, 1).unwrap_err(),
            reason::TABLE_UNREADABLE
        );
        assert_eq!(
            platform::read_tree_with(&unreadable_other, 10, 5, 1, 1).unwrap_err(),
            reason::TABLE_UNREADABLE
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_tree_read_of_exited_or_unreadable_host_is_unavailable() {
        let host = |walk| {
            let procfs = ScriptedProcfs::default().process(10, vec![Ok(linux_stat(1, 5, 7)), walk]);
            platform::read_tree_with(&procfs, 10, 5, 1, 1).unwrap_err()
        };

        assert_eq!(host(Err(platform::ReadError::Gone)), reason::HOST_EXITED);
        assert_eq!(host(Ok(linux_stat(1, 6, 7))), reason::HOST_EXITED);
        assert_eq!(
            host(Ok(platform::Stat {
                zombie: true,
                ..linux_stat(1, 5, 7)
            })),
            reason::HOST_EXITED
        );
        assert_eq!(
            host(Err(platform::ReadError::Unreadable)),
            reason::PROCESS_UNREADABLE
        );
    }

    /// A kernel with children files: the host is PID 10 started at tick 5 and
    /// lists child PID 11. The process table is never listed.
    #[cfg(target_os = "linux")]
    fn linux_children_tree(
        host_children: std::result::Result<Vec<u32>, platform::ReadError>,
        child: std::result::Result<platform::Stat, platform::ReadError>,
    ) -> std::result::Result<TreeRead, &'static str> {
        let mut procfs = ScriptedProcfs {
            children: Some(std::collections::BTreeMap::from([(10, host_children)])),
            unlisted: true,
            ..ScriptedProcfs::default()
        }
        .process(10, vec![Ok(linux_stat(1, 5, 7))])
        .process(11, vec![child]);
        procfs.pids.clear();
        platform::read_tree_with(&procfs, 10, 5, 1, 1)
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_tree_read_with_children_files_reads_only_the_tree() {
        let read = linux_children_tree(Ok(vec![11]), Ok(linux_stat(10, 9, 3))).expect("tree read");

        assert_eq!(read.process_count, 2);
        assert_eq!(read.cpu_nanos, 10_000_000_000);
        assert_eq!(read.resident_bytes, 2);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_tree_read_with_children_files_skips_reaped_or_replaced_child() {
        let reaped =
            linux_children_tree(Ok(vec![11]), Err(platform::ReadError::Gone)).expect("tree read");
        let replaced =
            linux_children_tree(Ok(vec![11]), Ok(linux_stat(1, 12, 900))).expect("tree read");

        for read in [reaped, replaced] {
            assert_eq!(read.process_count, 1);
            assert_eq!(read.cpu_nanos, 7_000_000_000);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_tree_read_with_children_files_and_unreadable_process_is_unavailable() {
        let child = linux_children_tree(Ok(vec![11]), Err(platform::ReadError::Unreadable));
        let host_children = linux_children_tree(
            Err(platform::ReadError::Unreadable),
            Ok(linux_stat(10, 9, 3)),
        );
        let host_gone =
            linux_children_tree(Err(platform::ReadError::Gone), Ok(linux_stat(10, 9, 3)));

        assert_eq!(child.unwrap_err(), reason::PROCESS_UNREADABLE);
        assert_eq!(host_children.unwrap_err(), reason::PROCESS_UNREADABLE);
        assert_eq!(host_gone.unwrap_err(), reason::HOST_EXITED);
    }

    /// The second read finds a child the first skipped; the third agrees.
    #[cfg(target_os = "linux")]
    #[test]
    fn linux_children_read_waits_for_two_reads_that_agree() {
        let mut reads = vec![vec![11, 13], vec![11, 12, 13], vec![11, 12, 13]].into_iter();
        let mut changing = (0..).map(|n| vec![n]);

        let stable = platform::read_until_stable(|| Ok(reads.next().expect("read")));
        let unstable = platform::read_until_stable(|| Ok(changing.next().expect("read")));

        assert_eq!(stable, Ok(vec![11, 12, 13]));
        assert_eq!(unstable, Err(platform::ReadError::Unreadable));
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
