// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resource snapshots of a runtime's local adapter host and its descendants.
//!
//! A snapshot reads the process tree rooted at the adapter host process that
//! NeMo Fabric started for a runtime, each parent before its children. CPU
//! time adds every process's own CPU time and the CPU time of the children it
//! has already waited for. An exited child that is not yet waited for counts
//! through its own CPU time, so work done by finished descendants still
//! counts. A child reaped between the read of its parent and its own read is
//! missed by that one snapshot and is never counted twice. CPU time can drop
//! between snapshots when a descendant is reparented out of the tree.
//!
//! Linux reads `/proc/<pid>/task/<tid>/children` and `/proc/<pid>/stat`. On a
//! kernel without those children files, it reads every process's parent ID
//! once per snapshot to find the tree. macOS reads `proc_listchildpids` and
//! `proc_pid_rusage`. A snapshot exposes only its own fields: never command
//! lines, environment variables, open files, or processes outside the tree.
//!
//! A live process in the tree that cannot be read, or a process table that
//! the Linux fallback cannot read in full, makes the snapshot unavailable
//! instead of leaving the process out. On Linux with `/proc` mounted
//! `hidepid=invisible`, another user's descendant is invisible and is not
//! counted.

use std::time::SystemTime;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{FabricError, Result};
use crate::time::rfc3339_millis;

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
    /// including exited descendants. It can drop when a descendant leaves the
    /// tree.
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
            started_at: rfc3339_millis(started_at),
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
        observed_at: rfc3339_millis(SystemTime::now()),
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

/// The tree walk the Linux and macOS readers share. It is platform-neutral so
/// its tests run on every platform.
#[cfg_attr(
    not(any(target_os = "linux", target_os = "macos")),
    allow(dead_code, reason = "only the Linux and macOS readers walk a tree")
)]
mod tree {
    use std::collections::VecDeque;

    use super::TreeRead;

    pub(super) const HOST_EXITED: &str = "adapter host process has exited";
    pub(super) const PROCESS_UNREADABLE: &str =
        "a live process in the runtime's process tree cannot be read";

    /// Why a process, or the list of its children, could not be read.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) enum ReadError {
        /// The process exited and was reaped.
        Gone,
        /// The process exists but cannot be read, for this reason.
        Unavailable(&'static str),
    }

    /// One process as a tree read sees it.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) struct Process {
        /// Parent PID, when the platform reads it in the same record as the
        /// rest of these fields.
        pub ppid: Option<u32>,
        /// Start time, in the clock that [`Children::listed_at`] uses.
        pub start: u64,
        /// Whether the process exited and is not yet reaped.
        pub exited: bool,
        /// The process's own CPU time plus that of the children it waited for.
        pub cpu_nanos: u64,
        pub resident_bytes: u64,
    }

    /// The children of one process, listed at one instant.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(super) struct Children {
        pub pids: Vec<u32>,
        /// When the list was read, in the clock that [`Process::start`] uses.
        /// Only a process read without its parent PID is checked against it.
        pub listed_at: u64,
    }

    /// The reads a tree read makes. Each platform implements them; tests
    /// script them.
    pub(super) trait ProcessTable {
        fn children(&self, pid: u32) -> Result<Children, ReadError>;
        fn read(&self, pid: u32) -> Result<Process, ReadError>;
    }

    /// The process that listed a child, as the tree read saw it.
    #[derive(Debug, Clone, Copy)]
    struct Parent {
        pid: u32,
        /// The parent's start time when the tree read counted it.
        start: u64,
        /// When the parent's children were listed.
        listed_at: u64,
    }

    /// Whether `process`, listed as a child of `parent` and read afterwards,
    /// is still a child of that same parent process.
    ///
    /// With a parent PID in the record, the parent is read again after the
    /// child. A parent that still has the start time it was counted with
    /// held its PID throughout, so the child's parent PID names it and not a
    /// later process that reused its PID. A parent that exited no longer
    /// shows that, so its child is not counted, like a reparented child.
    ///
    /// Without a parent PID, a process that started before the list was read
    /// already held its PID then, so it is the listed process.
    fn is_listed_child(
        table: &impl ProcessTable,
        process: &Process,
        parent: Parent,
    ) -> Result<bool, &'static str> {
        match process.ppid {
            Some(ppid) if ppid != parent.pid => Ok(false),
            Some(_) => match table.read(parent.pid) {
                Ok(now) => Ok(now.start == parent.start),
                Err(ReadError::Gone) => Ok(false),
                Err(ReadError::Unavailable(reason)) => Err(reason),
            },
            None => Ok(process.start < parent.listed_at),
        }
    }

    /// Read the tree under `root`. Fails when the root exited or its PID now
    /// names a process with another start time than `identity`, or when a
    /// live process in the tree cannot be read.
    pub(super) fn read_tree_with(
        table: &impl ProcessTable,
        root: u32,
        identity: u64,
    ) -> Result<TreeRead, &'static str> {
        let mut read = TreeRead::default();
        // Each entry is a PID and, for a child, the parent that listed it.
        let mut pending = VecDeque::from([(root, None)]);
        while let Some((pid, parent)) = pending.pop_front() {
            // List the children before reading the process: a read that still
            // shows the expected process proves the listed children were its own.
            let found = table
                .children(pid)
                .and_then(|children| Ok((table.read(pid)?, children)));
            let (process, children) = match (found, parent) {
                (Ok((process, children)), None) if process.start == identity && !process.exited => {
                    (process, children)
                }
                (Ok(_) | Err(ReadError::Gone), None) => return Err(HOST_EXITED),
                (Ok((process, children)), Some(parent)) => {
                    if !is_listed_child(table, &process, parent)? {
                        // Its PID now names another process, or it left the tree.
                        continue;
                    }
                    (process, children)
                }
                // A reaped child.
                (Err(ReadError::Gone), Some(_)) => continue,
                (Err(ReadError::Unavailable(reason)), _) => return Err(reason),
            };
            read.cpu_nanos = read.cpu_nanos.saturating_add(process.cpu_nanos);
            if !process.exited {
                read.resident_bytes = read.resident_bytes.saturating_add(process.resident_bytes);
                read.process_count += 1;
            }
            let parent = Parent {
                pid,
                start: process.start,
                listed_at: children.listed_at,
            };
            pending.extend(children.pids.into_iter().map(|child| (child, Some(parent))));
        }
        Ok(read)
    }
}

#[cfg(target_os = "linux")]
mod platform {
    //! `/proc` reads. Each process's children come from
    //! `/proc/<pid>/task/<tid>/children`, so a read visits only the runtime's
    //! own processes. A kernel built without those files falls back to one
    //! pass over every process's parent per snapshot.

    use std::collections::BTreeMap;
    use std::sync::OnceLock;

    use super::TreeRead;
    use super::tree::{
        Children, PROCESS_UNREADABLE, Process, ProcessTable, ReadError, read_tree_with,
    };

    pub(super) const SUPPORTED: bool = true;
    pub(super) const TABLE_UNREADABLE: &str = "the process table cannot be read in full";
    pub(super) const TREE_CHANGING: &str =
        "the runtime's process tree kept changing during the read";
    /// Times a process's children are read while waiting for two reads that agree.
    const MAX_CHILDREN_READS: usize = 4;

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

    /// Where a tree read finds each process's children.
    pub(super) enum ChildSource {
        /// `/proc/<pid>/task/<tid>/children`.
        ChildrenFiles,
        /// Children by parent PID, from one pass over every process.
        Scanned(BTreeMap<u32, Vec<u32>>),
    }

    struct Live {
        children: ChildSource,
        ticks_per_second: u64,
        page_size: u64,
    }

    impl ProcessTable for Live {
        fn children(&self, pid: u32) -> Result<Children, ReadError> {
            let pids = match &self.children {
                ChildSource::ChildrenFiles => read_until_stable(|| children_once(pid))?,
                ChildSource::Scanned(by_parent) => by_parent.get(&pid).cloned().unwrap_or_default(),
            };
            // Each read checks the parent PID in `stat`, so the list time is unused.
            Ok(Children { pids, listed_at: 0 })
        }

        fn read(&self, pid: u32) -> Result<Process, ReadError> {
            let stat = stat(pid)?;
            Ok(Process {
                ppid: Some(stat.ppid),
                start: stat.start,
                exited: stat.zombie,
                cpu_nanos: ticks_to_nanos(stat.cpu_ticks, self.ticks_per_second),
                resident_bytes: stat.rss_pages.saturating_mul(self.page_size),
            })
        }
    }

    fn stat(pid: u32) -> Result<Stat, ReadError> {
        let text = std::fs::read_to_string(format!("/proc/{pid}/stat")).map_err(read_error)?;
        parse_stat(&text).ok_or(ReadError::Unavailable(PROCESS_UNREADABLE))
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
                children.push(
                    child
                        .parse()
                        .map_err(|_| ReadError::Unavailable(PROCESS_UNREADABLE))?,
                );
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
        Err(ReadError::Unavailable(TREE_CHANGING))
    }

    /// Every process's children, for a kernel without children files. A
    /// process that exited during the pass is left out; one whose parent
    /// cannot be read leaves the listing incomplete.
    pub(super) fn children_by_parent(
        pids: Vec<u32>,
        parent_of: impl Fn(u32) -> Result<u32, ReadError>,
    ) -> Result<BTreeMap<u32, Vec<u32>>, &'static str> {
        let mut children: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for pid in pids {
            match parent_of(pid) {
                Ok(ppid) => children.entry(ppid).or_default().push(pid),
                Err(ReadError::Gone) => {}
                Err(ReadError::Unavailable(_)) => return Err(TABLE_UNREADABLE),
            }
        }
        Ok(children)
    }

    fn all_pids() -> Result<Vec<u32>, &'static str> {
        let entries = std::fs::read_dir("/proc").map_err(|_| TABLE_UNREADABLE)?;
        let mut pids = Vec::new();
        for entry in entries {
            let name = entry.map_err(|_| TABLE_UNREADABLE)?.file_name();
            if let Some(pid) = name.to_str().and_then(|name| name.parse().ok()) {
                pids.push(pid);
            }
        }
        Ok(pids)
    }

    /// `ENOENT` and `ESRCH` mean the process exited; anything else means it
    /// cannot be read.
    fn read_error(error: std::io::Error) -> ReadError {
        if error.kind() == std::io::ErrorKind::NotFound || error.raw_os_error() == Some(libc::ESRCH)
        {
            ReadError::Gone
        } else {
            ReadError::Unavailable(PROCESS_UNREADABLE)
        }
    }

    pub(super) fn start_identity(pid: u32) -> Option<u64> {
        stat(pid).ok().map(|stat| stat.start)
    }

    fn sysconf(name: libc::c_int) -> u64 {
        // SAFETY: sysconf has no memory-safety preconditions.
        u64::try_from(unsafe { libc::sysconf(name) }).unwrap_or(0)
    }

    fn has_children_files() -> bool {
        static PRESENT: OnceLock<bool> = OnceLock::new();
        *PRESENT.get_or_init(|| {
            let pid = std::process::id();
            std::path::Path::new(&format!("/proc/{pid}/task/{pid}/children")).exists()
        })
    }

    /// Children by parent PID, from one pass over every process.
    pub(super) fn scan_children() -> Result<ChildSource, &'static str> {
        let by_parent = children_by_parent(all_pids()?, |pid| stat(pid).map(|stat| stat.ppid))?;
        Ok(ChildSource::Scanned(by_parent))
    }

    pub(super) fn read_tree(root: u32, identity: u64) -> Result<TreeRead, &'static str> {
        let children = if has_children_files() {
            ChildSource::ChildrenFiles
        } else {
            scan_children()?
        };
        read_tree_from(children, root, identity)
    }

    pub(super) fn read_tree_from(
        children: ChildSource,
        root: u32,
        identity: u64,
    ) -> Result<TreeRead, &'static str> {
        static TICKS: OnceLock<u64> = OnceLock::new();
        static PAGE: OnceLock<u64> = OnceLock::new();
        let live = Live {
            children,
            ticks_per_second: *TICKS.get_or_init(|| sysconf(libc::_SC_CLK_TCK).max(1)),
            page_size: *PAGE.get_or_init(|| sysconf(libc::_SC_PAGESIZE)),
        };
        read_tree_with(&live, root, identity)
    }

    pub(super) fn ticks_to_nanos(ticks: u64, ticks_per_second: u64) -> u64 {
        let nanos = u128::from(ticks) * 1_000_000_000 / u128::from(ticks_per_second);
        u64::try_from(nanos).unwrap_or(u64::MAX)
    }
}

#[cfg(target_os = "macos")]
mod platform {
    //! `libproc` reads. `proc_listchildpids` lists a process's children,
    //! including exited ones that are not yet reaped, and `proc_pid_rusage`
    //! reads any of them.

    use std::sync::OnceLock;

    use super::TreeRead;
    use super::tree::{
        Children, PROCESS_UNREADABLE, Process, ProcessTable, ReadError, read_tree_with,
    };

    pub(super) const SUPPORTED: bool = true;
    /// Child PIDs the first list call has room for. A full buffer doubles it.
    const INITIAL_LISTED_CHILDREN: usize = 64;

    struct Live {
        timebase: (u64, u64),
    }

    impl ProcessTable for Live {
        fn children(&self, pid: u32) -> Result<Children, ReadError> {
            let ppid = libc::pid_t::try_from(pid).map_err(|_| ReadError::Gone)?;
            let mut pids = vec![0 as libc::pid_t; INITIAL_LISTED_CHILDREN];
            loop {
                let bytes = libc::c_int::try_from(std::mem::size_of_val(pids.as_slice()))
                    .map_err(|_| ReadError::Unavailable(PROCESS_UNREADABLE))?;
                // SAFETY: mach_absolute_time has no preconditions.
                let listed_at = unsafe { mach_absolute_time() };
                // SAFETY: the buffer is writable and sized in bytes as libproc requires.
                let count =
                    unsafe { libc::proc_listchildpids(ppid, pids.as_mut_ptr().cast(), bytes) };
                let count = usize::try_from(count)
                    .map_err(|_| ReadError::Unavailable(PROCESS_UNREADABLE))?;
                // A full buffer can mean the list was cut short.
                if count < pids.len() {
                    pids.truncate(count);
                    let pids = pids
                        .into_iter()
                        .filter_map(|pid| u32::try_from(pid).ok())
                        .collect();
                    return Ok(Children { pids, listed_at });
                }
                pids.resize(pids.len() * 2, 0);
            }
        }

        fn read(&self, pid: u32) -> Result<Process, ReadError> {
            let info = usage(pid)?;
            Ok(Process {
                ppid: None,
                start: info.ri_proc_start_abstime,
                exited: info.ri_proc_exit_abstime != 0,
                cpu_nanos: mach_to_nanos(
                    info.ri_user_time
                        .saturating_add(info.ri_system_time)
                        .saturating_add(info.ri_child_user_time)
                        .saturating_add(info.ri_child_system_time),
                    self.timebase,
                ),
                resident_bytes: info.ri_resident_size,
            })
        }
    }

    /// `proc_pid_rusage` of `pid`. CPU and start times are in Mach absolute
    /// time units.
    fn usage(pid: u32) -> Result<libc::rusage_info_v2, ReadError> {
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
        if status == 0 {
            return Ok(info);
        }
        match std::io::Error::last_os_error().raw_os_error() {
            Some(libc::ESRCH) => Err(ReadError::Gone),
            _ => Err(ReadError::Unavailable(PROCESS_UNREADABLE)),
        }
    }

    /// `mach_timebase_info_data_t`.
    #[repr(C)]
    struct MachTimebaseInfo {
        numer: u32,
        denom: u32,
    }

    // libc marks these bindings deprecated in favour of the `mach2` crate; two
    // system calls do not justify a new dependency.
    unsafe extern "C" {
        fn mach_timebase_info(info: *mut MachTimebaseInfo) -> libc::c_int;
        /// The clock that `ri_proc_start_abstime` uses.
        fn mach_absolute_time() -> u64;
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
        usage(pid).ok().map(|info| info.ri_proc_start_abstime)
    }

    pub(super) fn read_tree(root: u32, identity: u64) -> Result<TreeRead, &'static str> {
        let timebase = timebase().ok_or(PROCESS_UNREADABLE)?;
        read_tree_with(&Live { timebase }, root, identity)
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
    use std::collections::BTreeMap;
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    use std::time::{Duration, Instant};

    use super::tree::{
        Children, HOST_EXITED, PROCESS_UNREADABLE, Process, ProcessTable, ReadError, read_tree_with,
    };
    use super::*;

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

        let deadline = Instant::now() + Duration::from_secs(5);
        let mut result = snapshot_local_host("runtime-1", Some(&host));
        while result.is_ok() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
            result = snapshot_local_host("runtime-1", Some(&host));
        }
        child.wait().expect("reap child");

        assert!(matches!(
            result,
            Err(FabricError::ResourceObservationUnavailable { ref reason, .. })
                if reason == HOST_EXITED
        ));
    }

    /// The host is `sleep`, which never waits for children. Before its `exec`,
    /// the shell starts a child that burns CPU and exits, so the child stays
    /// unreaped under the host.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn snapshot_local_host_counts_cpu_of_exited_unreaped_child() {
        let mut host_process = std::process::Command::new("/bin/sh")
            .args([
                "-c",
                "(i=0; while [ $i -lt 100000 ]; do i=$((i+1)); done) & exec sleep 30",
            ])
            .spawn()
            .expect("spawn host");
        let host = LocalHostProcess::new(host_process.id(), SystemTime::now());
        let deadline = Instant::now() + Duration::from_secs(30);
        let snapshot_until = |done: fn(u64) -> bool| loop {
            let snapshot = snapshot_local_host("runtime-1", Some(&host)).expect("snapshot");
            if done(snapshot.process_count) || Instant::now() > deadline {
                return snapshot;
            }
            std::thread::sleep(Duration::from_millis(20));
        };

        let running = snapshot_until(|count| count >= 2);
        let exited = snapshot_until(|count| count == 1);
        host_process.kill().expect("kill host");
        host_process.wait().expect("reap host");

        assert_eq!(running.process_count, 2, "child never seen: {running:?}");
        assert_eq!(exited.process_count, 1, "child still running: {exited:?}");
        assert!(
            exited.cpu_time_seconds >= 0.1,
            "exited child's CPU missing: {exited:?}"
        );
    }

    /// When a scripted table read the listed children.
    const LISTED_AT: u64 = 100;
    /// The scripted host: PID 10, started at time 5.
    const HOST: u32 = 10;
    const HOST_START: u64 = 5;

    /// Scripted answers for the platform-neutral tree walk. A PID without
    /// answers has exited.
    #[derive(Default)]
    struct ScriptedTable {
        children: BTreeMap<u32, std::result::Result<Vec<u32>, ReadError>>,
        reads: BTreeMap<u32, std::result::Result<Process, ReadError>>,
    }

    impl ScriptedTable {
        fn process(
            mut self,
            pid: u32,
            children: std::result::Result<Vec<u32>, ReadError>,
            read: std::result::Result<Process, ReadError>,
        ) -> Self {
            self.children.insert(pid, children);
            self.reads.insert(pid, read);
            self
        }

        fn read_host_tree(&self) -> std::result::Result<TreeRead, &'static str> {
            read_tree_with(self, HOST, HOST_START)
        }
    }

    impl ProcessTable for ScriptedTable {
        fn children(&self, pid: u32) -> std::result::Result<Children, ReadError> {
            let pids = self
                .children
                .get(&pid)
                .cloned()
                .unwrap_or(Err(ReadError::Gone))?;
            Ok(Children {
                pids,
                listed_at: LISTED_AT,
            })
        }

        fn read(&self, pid: u32) -> std::result::Result<Process, ReadError> {
            self.reads
                .get(&pid)
                .copied()
                .unwrap_or(Err(ReadError::Gone))
        }
    }

    /// How a platform shows that a listed child is still the listed process.
    #[derive(Debug, Clone, Copy)]
    enum Lineage {
        /// Linux: the parent PID in the same record.
        ParentPid,
        /// macOS: a start time before the list was read.
        StartedBefore,
    }

    impl Lineage {
        /// A live process listed under `parent`, with 1000 resident bytes.
        fn child(self, parent: u32, cpu_nanos: u64) -> Process {
            let (ppid, start) = match self {
                Self::ParentPid => (Some(parent), LISTED_AT + 1),
                Self::StartedBefore => (None, LISTED_AT - 1),
            };
            Process {
                ppid,
                start,
                exited: false,
                cpu_nanos,
                resident_bytes: 1000,
            }
        }

        /// A process that took the PID of a child of `parent` after the list
        /// was read.
        fn replacement(self, parent: u32) -> Process {
            match self {
                Self::ParentPid => Process {
                    ppid: Some(1),
                    ..self.child(parent, 900)
                },
                Self::StartedBefore => Process {
                    start: LISTED_AT,
                    ..self.child(parent, 900)
                },
            }
        }

        fn host(self, cpu_nanos: u64) -> Process {
            Process {
                start: HOST_START,
                ..self.child(1, cpu_nanos)
            }
        }
    }

    const LINEAGES: [Lineage; 2] = [Lineage::ParentPid, Lineage::StartedBefore];

    #[test]
    fn tree_read_counts_live_descendants_and_cpu_of_exited_ones() {
        for lineage in LINEAGES {
            let table = ScriptedTable::default()
                .process(HOST, Ok(vec![11]), Ok(lineage.host(7)))
                .process(11, Ok(vec![12, 13]), Ok(lineage.child(HOST, 3)))
                .process(12, Ok(vec![]), Ok(lineage.child(11, 2)))
                .process(
                    13,
                    Ok(vec![]),
                    Ok(Process {
                        exited: true,
                        ..lineage.child(11, 4)
                    }),
                );

            let read = table.read_host_tree().expect("tree read");

            assert_eq!(read.process_count, 3, "{lineage:?}");
            assert_eq!(read.cpu_nanos, 7 + 3 + 2 + 4, "{lineage:?}");
            assert_eq!(read.resident_bytes, 3000, "{lineage:?}");
        }
    }

    #[test]
    fn tree_read_skips_child_reaped_or_replaced_after_listing() {
        for lineage in LINEAGES {
            let with_child = |children, read| {
                ScriptedTable::default()
                    .process(HOST, Ok(vec![11]), Ok(lineage.host(7)))
                    .process(11, children, read)
                    .read_host_tree()
                    .expect("tree read")
            };
            let reaped = with_child(Ok(vec![]), Err(ReadError::Gone));
            let reaped_before_its_list =
                with_child(Err(ReadError::Gone), Ok(lineage.child(HOST, 3)));
            let replaced = with_child(Ok(vec![]), Ok(lineage.replacement(HOST)));

            for read in [reaped, reaped_before_its_list, replaced] {
                assert_eq!(read.process_count, 1, "{lineage:?}");
                assert_eq!(read.cpu_nanos, 7, "{lineage:?}");
            }
        }
    }

    /// How the Linux reader lists children.
    #[derive(Debug, Clone, Copy)]
    enum Listing {
        /// `/proc/<pid>/task/<tid>/children`, read when the walk reaches the process.
        ChildrenFiles,
        /// One pass over every process's parent, made before the walk.
        Scan,
    }

    const LISTINGS: [Listing; 2] = [Listing::ChildrenFiles, Listing::Scan];

    /// A table in which other processes take some PIDs once the walk has
    /// read `reused_after`: `before` answers until then and `after` from then
    /// on. A scan lists children from `before`, because it ran before the walk.
    struct PidReuseTable {
        before: ScriptedTable,
        after: ScriptedTable,
        reused_after: u32,
        reused: std::cell::Cell<bool>,
        listing: Listing,
    }

    impl PidReuseTable {
        fn current(&self) -> &ScriptedTable {
            if self.reused.get() {
                &self.after
            } else {
                &self.before
            }
        }
    }

    impl ProcessTable for PidReuseTable {
        fn children(&self, pid: u32) -> std::result::Result<Children, ReadError> {
            match self.listing {
                Listing::ChildrenFiles => self.current().children(pid),
                Listing::Scan => self.before.children(pid),
            }
        }

        fn read(&self, pid: u32) -> std::result::Result<Process, ReadError> {
            let read = self.current().read(pid);
            if pid == self.reused_after {
                self.reused.set(true);
            }
            read
        }
    }

    /// The host's child 11 and grandchild 12 exit right after the walk reads
    /// 11. `after_11` is what PID 11 then reads as; PID 12 is an unrelated
    /// process whose parent PID is 11 and which used 900 ns of CPU time.
    fn read_tree_after_reuse(
        listing: Listing,
        after_11: std::result::Result<Process, ReadError>,
    ) -> std::result::Result<TreeRead, &'static str> {
        let lineage = Lineage::ParentPid;
        let before = ScriptedTable::default()
            .process(HOST, Ok(vec![11]), Ok(lineage.host(7)))
            .process(11, Ok(vec![12]), Ok(lineage.child(HOST, 3)))
            .process(12, Ok(vec![]), Ok(lineage.child(11, 2)));
        let after = ScriptedTable::default()
            .process(HOST, Ok(vec![]), Ok(lineage.host(7)))
            .process(11, Ok(vec![12]), after_11)
            .process(12, Ok(vec![]), Ok(lineage.child(11, 900)));
        let table = PidReuseTable {
            before,
            after,
            reused_after: 11,
            reused: std::cell::Cell::new(false),
            listing,
        };
        read_tree_with(&table, HOST, HOST_START)
    }

    #[test]
    fn tree_read_skips_child_whose_parent_pid_was_reused() {
        for listing in LISTINGS {
            let unrelated_parent = Process {
                ppid: Some(1),
                start: LISTED_AT + 50,
                ..Lineage::ParentPid.child(1, 500)
            };
            for after_11 in [Ok(unrelated_parent), Err(ReadError::Gone)] {
                let read = read_tree_after_reuse(listing, after_11).expect("tree read");

                assert_eq!(read.process_count, 2, "{listing:?} {after_11:?}");
                assert_eq!(read.cpu_nanos, 7 + 3, "{listing:?} {after_11:?}");
            }
        }
    }

    #[test]
    fn tree_read_with_parent_unreadable_after_child_read_is_unavailable() {
        for listing in LISTINGS {
            let read = read_tree_after_reuse(listing, Err(ReadError::Unavailable("hidden")));

            assert_eq!(read.unwrap_err(), "hidden", "{listing:?}");
        }
    }

    #[test]
    fn tree_read_with_unreadable_live_child_is_unavailable() {
        let with_child = |children, read| {
            ScriptedTable::default()
                .process(HOST, Ok(vec![11]), Ok(Lineage::ParentPid.host(7)))
                .process(11, children, read)
                .read_host_tree()
        };

        let unreadable = with_child(Ok(vec![]), Err(ReadError::Unavailable(PROCESS_UNREADABLE)));
        let unlisted = with_child(
            Err(ReadError::Unavailable("children cannot be listed")),
            Ok(Lineage::ParentPid.child(HOST, 3)),
        );

        assert_eq!(unreadable.unwrap_err(), PROCESS_UNREADABLE);
        assert_eq!(unlisted.unwrap_err(), "children cannot be listed");
    }

    #[test]
    fn tree_read_of_exited_replaced_or_unreadable_host_is_unavailable() {
        let host = |children, read| {
            ScriptedTable::default()
                .process(HOST, children, read)
                .read_host_tree()
                .unwrap_err()
        };
        let live = Lineage::ParentPid.host(7);

        assert_eq!(host(Ok(vec![]), Err(ReadError::Gone)), HOST_EXITED);
        assert_eq!(host(Err(ReadError::Gone), Ok(live)), HOST_EXITED);
        assert_eq!(
            host(
                Ok(vec![]),
                Ok(Process {
                    start: HOST_START + 1,
                    ..live
                })
            ),
            HOST_EXITED
        );
        assert_eq!(
            host(
                Ok(vec![]),
                Ok(Process {
                    exited: true,
                    ..live
                })
            ),
            HOST_EXITED
        );
        assert_eq!(
            host(Ok(vec![]), Err(ReadError::Unavailable(PROCESS_UNREADABLE))),
            PROCESS_UNREADABLE
        );
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

    /// The second read finds a child the first skipped; the third agrees.
    #[cfg(target_os = "linux")]
    #[test]
    fn linux_children_read_waits_for_two_reads_that_agree() {
        let mut reads = vec![vec![11, 13], vec![11, 12, 13], vec![11, 12, 13]].into_iter();
        let mut changing = (0..).map(|n| vec![n]);

        let stable = platform::read_until_stable(|| Ok(reads.next().expect("read")));
        let unstable = platform::read_until_stable(|| Ok(changing.next().expect("read")));

        assert_eq!(stable, Ok(vec![11, 12, 13]));
        assert_eq!(
            unstable,
            Err(ReadError::Unavailable(platform::TREE_CHANGING))
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_process_scan_groups_children_and_needs_every_parent() {
        let parents = BTreeMap::from([
            (10, Ok(1)),
            (11, Ok(10)),
            (12, Ok(10)),
            (13, Err(ReadError::Gone)),
        ]);
        let parent_of = |pid| parents[&pid];

        let scanned = platform::children_by_parent(vec![10, 11, 12, 13], parent_of).expect("scan");
        let incomplete = platform::children_by_parent(vec![10, 50], |pid| match pid {
            50 => Err(ReadError::Unavailable(PROCESS_UNREADABLE)),
            _ => Ok(1),
        });

        assert_eq!(scanned, BTreeMap::from([(1, vec![10]), (10, vec![11, 12])]));
        assert_eq!(incomplete.unwrap_err(), platform::TABLE_UNREADABLE);
    }

    /// The fallback for a kernel without children files reads the tree too.
    #[cfg(target_os = "linux")]
    #[test]
    fn linux_tree_read_with_process_scan_reads_this_process_tree() {
        let host = LocalHostProcess::new(std::process::id(), SystemTime::now());
        let children = platform::scan_children().expect("scan");

        let read = platform::read_tree_from(children, host.pid, host.identity.expect("identity"))
            .expect("tree read");

        assert!(read.process_count >= 1);
        assert!(read.resident_bytes > 0);
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
