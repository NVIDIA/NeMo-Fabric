// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The run record file: created once, owner-only, appended one line per write.
//!
//! Each line is one `write_all` on an unbuffered file, so a SIGKILL of Fabric
//! or the adapter host keeps every line whose write returned. Only the end
//! line is synced, once, so the record survives a crash of the machine only
//! up to its last synced line.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

use super::lines::RecordLine;

/// File name of the record inside the invocation folder.
pub(crate) const RECORD_FILE_NAME: &str = "run-record.jsonl";
/// Maximum serialized JSON size of one line, excluding its newline.
const MAX_LINE_BYTES: usize = 64 * 1024;

/// Appends lines to one invocation's record.
#[derive(Debug)]
pub(crate) struct RecordWriter {
    file: File,
}

impl RecordWriter {
    /// Create the record at `path`. `create_new` refuses an existing file or
    /// symbolic link at `path`, and on Unix the file is mode 0600.
    pub(crate) fn create(path: &Path) -> io::Result<Self> {
        #[cfg(test)]
        std::thread::sleep(test_hooks::CREATE_DELAY.with(std::cell::Cell::get));
        let mut options = OpenOptions::new();
        options.append(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        Ok(Self {
            file: options.open(path)?,
        })
    }

    /// Append `line` and its newline with one write.
    pub(crate) fn append(&mut self, line: &RecordLine) -> io::Result<()> {
        #[cfg(test)]
        test_hooks::fail_append(&line.body)?;
        let mut bytes = serde_json::to_vec(line).map_err(io::Error::other)?;
        if bytes.len() > MAX_LINE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("run record line exceeds {MAX_LINE_BYTES} bytes"),
            ));
        }
        bytes.push(b'\n');
        self.file.write_all(&bytes)
    }

    /// Flush the file's data and metadata to stable storage.
    pub(crate) fn sync(&self) -> io::Result<()> {
        #[cfg(test)]
        test_hooks::fail_if(test_hooks::Fault::Sync)?;
        self.file.sync_all()
    }
}

/// Per-thread hooks that make record I/O slow or fail in tests.
#[cfg(test)]
pub(crate) mod test_hooks {
    use std::cell::Cell;
    use std::io;
    use std::time::Duration;

    use super::super::lines::LineBody;

    /// A record operation that a test makes fail.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum Fault {
        StartAppend,
        EndAppend,
        Sync,
    }

    thread_local! {
        /// Delay added before each record file is created on this thread.
        pub(crate) static CREATE_DELAY: Cell<Duration> = const { Cell::new(Duration::ZERO) };
        /// Operation that fails on this thread.
        pub(crate) static FAULT: Cell<Option<Fault>> = const { Cell::new(None) };
    }

    pub(super) fn fail_append(body: &LineBody) -> io::Result<()> {
        match body {
            LineBody::Start(_) => fail_if(Fault::StartAppend),
            LineBody::End(_) => fail_if(Fault::EndAppend),
        }
    }

    pub(super) fn fail_if(fault: Fault) -> io::Result<()> {
        if FAULT.with(Cell::get) == Some(fault) {
            return Err(io::Error::other(format!("injected {fault:?} failure")));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_folder(name: &str) -> std::path::PathBuf {
        let folder = std::env::temp_dir().join(format!(
            "fabric-record-writer-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&folder).expect("create temp folder");
        folder
    }

    #[test]
    fn record_writer_refuses_an_existing_file_or_a_symlink() {
        let folder = temp_folder("excl");
        let path = folder.join(RECORD_FILE_NAME);
        std::fs::write(&path, "old").expect("seed file");

        let error = RecordWriter::create(&path).expect_err("existing file");

        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read_to_string(&path).expect("read"), "old");
        #[cfg(unix)]
        {
            let linked = folder.join("linked");
            std::fs::create_dir_all(&linked).expect("linked folder");
            let target = folder.join("target");
            std::os::unix::fs::symlink(&target, linked.join(RECORD_FILE_NAME)).expect("symlink");
            assert!(RecordWriter::create(&linked.join(RECORD_FILE_NAME)).is_err());
            assert!(!target.exists());
        }
    }

    #[cfg(unix)]
    #[test]
    fn record_writer_new_file_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let folder = temp_folder("mode");
        let path = folder.join(RECORD_FILE_NAME);

        RecordWriter::create(&path).expect("create");
        let mode = std::fs::metadata(&path).expect("stat").permissions().mode();

        assert_eq!(mode & 0o777, 0o600);
    }

    #[test]
    fn line_byte_limit_accepts_exact_boundary_and_rejects_oversize_without_appending() {
        use crate::record::lines::{EndLine, LineBody, RecordSchema, RecordVersion, ResourceScope};
        use crate::runtime::RunStatus;

        let folder = temp_folder("line-size");
        let path = folder.join(RECORD_FILE_NAME);
        let mut writer = RecordWriter::create(&path).expect("writer");
        let mut line = RecordLine {
            schema: RecordSchema::RunRecord,
            version: RecordVersion::V1Alpha1,
            runtime_id: "rt".to_string(),
            invocation_id: "inv".to_string(),
            request_id: "req".to_string(),
            time: "2026-10-21T00:00:00.000Z".to_string(),
            body: LineBody::End(Box::new(EndLine {
                outcome_scope: "invocation",
                result: None,
                status: RunStatus::Succeeded,
                elapsed_ms: 0,
                exchange_elapsed_ms: None,
                resource_scope: ResourceScope::RuntimeAdapterHostProcessTree,
                resources_before: crate::ResourceObservation::NotCollected,
                resources: crate::ResourceObservation::NotCollected,
                model_usage: None,
                error: None,
                usage: None,
                artifacts: Vec::new(),
            })),
        };
        let overhead = serde_json::to_vec(&line).expect("line JSON").len();
        line.request_id
            .push_str(&"a".repeat(MAX_LINE_BYTES - overhead));
        assert_eq!(serde_json::to_vec(&line).unwrap().len(), MAX_LINE_BYTES);
        writer.append(&line).expect("exact limit is valid");
        let before = std::fs::read(&path).expect("written line");
        assert_eq!(before.len(), MAX_LINE_BYTES + 1);
        assert_eq!(before.last(), Some(&b'\n'));

        line.request_id.push('a');
        let error = writer.append(&line).expect_err("one byte too large");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }
}
