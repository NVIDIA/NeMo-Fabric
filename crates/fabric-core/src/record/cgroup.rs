// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The container's out-of-memory kill count: cgroup v2 `memory.events`
//! `oom_kill`, read at the start and end lines of an invocation.
//!
//! The count describes Fabric's container only when Fabric owns its cgroup:
//! `/proc/self/cgroup` has the single v2 line `0::/`, and the cgroup's
//! `memory.events` reads with an `oom_kill` line. The kernel creates
//! `memory.events` only in a non-root cgroup with the memory controller, so a
//! true root cgroup and rootless containers without memory delegation fail
//! that check. Every other case reads unavailable with the failed check's
//! reason.

use std::path::{Path, PathBuf};

use super::lines::{OomKills, OomKillsUnavailable};

/// Reads cgroup v2 files under a root, `/` on a real system.
#[derive(Debug, Clone)]
pub(crate) struct CgroupReader {
    root: PathBuf,
}

impl CgroupReader {
    /// The running system's `/proc/self/cgroup` and `/sys/fs/cgroup`.
    pub(crate) fn system() -> Self {
        Self {
            root: PathBuf::from("/"),
        }
    }

    /// A fake cgroupfs: `<root>/proc/self/cgroup` and `<root>/sys/fs/cgroup`.
    #[cfg(test)]
    pub(crate) fn with_root(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The owned cgroup's `oom_kill` count, or why it is unavailable.
    pub(crate) fn oom_kills(&self) -> OomKills {
        match self.read_oom_kills() {
            Ok(count) => OomKills::Count(count),
            Err(reason) => OomKills::Unavailable { reason },
        }
    }

    fn read_oom_kills(&self) -> Result<u64, OomKillsUnavailable> {
        if !cfg!(target_os = "linux") && self.root == Path::new("/") {
            return Err(OomKillsUnavailable::Platform);
        }
        let membership = std::fs::read_to_string(self.root.join("proc/self/cgroup"))
            .map_err(|_| OomKillsUnavailable::NoOwnedCgroup)?;
        let lines: Vec<&str> = membership.lines().filter(|line| !line.is_empty()).collect();
        let [line] = lines.as_slice() else {
            return Err(OomKillsUnavailable::CgroupV1);
        };
        let Some(path) = line.strip_prefix("0::") else {
            return Err(OomKillsUnavailable::CgroupV1);
        };
        if path != "/" {
            return Err(OomKillsUnavailable::NoOwnedCgroup);
        }
        let events = std::fs::read_to_string(self.root.join("sys/fs/cgroup/memory.events"))
            .map_err(|_| OomKillsUnavailable::NoOwnedCgroup)?;
        events
            .lines()
            .find_map(|line| line.strip_prefix("oom_kill "))
            .and_then(|count| count.trim().parse().ok())
            .ok_or(OomKillsUnavailable::NoOwnedCgroup)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A fake cgroupfs root with `membership` in `/proc/self/cgroup` and, when
    /// given, the cgroup's `memory.events`.
    pub(crate) fn fake_root(name: &str, membership: &str, memory_events: Option<&str>) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("fabric-cgroup-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("proc/self")).expect("proc");
        std::fs::create_dir_all(root.join("sys/fs/cgroup")).expect("cgroup");
        std::fs::write(root.join("proc/self/cgroup"), membership).expect("membership");
        if let Some(events) = memory_events {
            std::fs::write(root.join("sys/fs/cgroup/memory.events"), events).expect("events");
        }
        root
    }

    /// `memory.events` of a cgroup whose `oom_kill` count is `count`.
    pub(crate) fn memory_events(count: u64) -> String {
        format!("low 0\nhigh 0\nmax 2\noom 1\noom_kill {count}\noom_group_kill 0\n")
    }

    fn unavailable(reason: OomKillsUnavailable) -> OomKills {
        OomKills::Unavailable { reason }
    }

    #[test]
    fn oom_kills_owned_cgroup_v2_reads_memory_events() {
        let root = fake_root("owned", "0::/\n", Some(&memory_events(1)));

        assert_eq!(
            CgroupReader::with_root(&root).oom_kills(),
            OomKills::Count(1)
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn oom_kills_cgroup_v1_reads_cgroup_v1() {
        let root = fake_root(
            "v1",
            "12:memory:/docker/abc\n11:cpu:/docker/abc\n",
            Some(&memory_events(1)),
        );

        assert_eq!(
            CgroupReader::with_root(&root).oom_kills(),
            unavailable(OomKillsUnavailable::CgroupV1)
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn oom_kills_named_cgroup_missing_events_or_no_oom_kill_line_reads_no_owned_cgroup() {
        let named = fake_root(
            "named",
            "0::/user.slice/session-2.scope\n",
            Some(&memory_events(1)),
        );
        // Rootless containers without memory delegation, and a true root
        // cgroup, have no `memory.events`.
        let no_events = fake_root("no-events", "0::/\n", None);
        let no_oom_kill = fake_root("no-oom-kill", "0::/\n", Some("low 0\nhigh 0\n"));

        for root in [&named, &no_events, &no_oom_kill] {
            assert_eq!(
                CgroupReader::with_root(root).oom_kills(),
                unavailable(OomKillsUnavailable::NoOwnedCgroup),
                "{}",
                root.display()
            );
        }
        for root in [named, no_events, no_oom_kill] {
            let _ = std::fs::remove_dir_all(root);
        }
    }
}
