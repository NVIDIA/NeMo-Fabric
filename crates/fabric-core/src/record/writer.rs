// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exclusive append writer. A failed write disables its owning recorder.

use super::lines::{LineBody, RecordLine};
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

pub(crate) const RECORD_FILE_NAME: &str = "run.jsonl";
pub(super) const MAX_DIAGNOSTIC_BYTES: usize = 64 * 1024;

/// A size refusal writes no bytes; I/O failures remain ordinary errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AppendOutcome {
    Written,
    Omitted,
}

#[derive(Debug)]
pub(super) struct RecordWriter {
    file: File,
}

impl RecordWriter {
    pub(super) fn create(path: &Path) -> io::Result<Self> {
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

    pub(super) fn append(&mut self, line: &RecordLine) -> io::Result<AppendOutcome> {
        #[cfg(test)]
        test_hooks::fail_append(&line.body)?;
        // Lifecycle/result evidence is retained without the old journal's 64 KiB
        // cap. Diagnostic producers bound each message before it reaches here.
        let mut bytes = serde_json::to_vec(line).map_err(io::Error::other)?;
        if matches!(line.body, LineBody::Diagnostic { .. })
            && bytes.len().saturating_add(1) > MAX_DIAGNOSTIC_BYTES
        {
            return Ok(AppendOutcome::Omitted);
        }
        bytes.push(b'\n');
        self.file.write_all(&bytes)?;
        Ok(AppendOutcome::Written)
    }

    pub(super) fn sync(&self) -> io::Result<()> {
        #[cfg(test)]
        test_hooks::fail_if(test_hooks::Fault::Sync)?;
        self.file.sync_all()
    }
}

#[cfg(test)]
pub(crate) mod test_hooks {
    use super::super::lines::LineBody;
    use std::cell::Cell;
    use std::io;
    use std::time::Duration;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum Fault {
        StartAppend,
        EndAppend,
        Sync,
    }
    thread_local! {
        pub(crate) static CREATE_DELAY: Cell<Duration> = const { Cell::new(Duration::ZERO) };
        pub(crate) static INVOCATION_START_DELAY: Cell<Duration> = const { Cell::new(Duration::ZERO) };
        pub(crate) static FAULT: Cell<Option<Fault>> = const { Cell::new(None) };
    }
    pub(super) fn fail_append(body: &LineBody) -> io::Result<()> {
        if matches!(body, LineBody::InvocationStarted { .. }) {
            std::thread::sleep(INVOCATION_START_DELAY.with(Cell::get));
        }
        match body {
            LineBody::RequestStarted(_) => fail_if(Fault::StartAppend),
            LineBody::Summary(_) => fail_if(Fault::EndAppend),
            _ => Ok(()),
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
    use crate::runtime::new_id;

    #[test]
    fn optional_record_limit_includes_the_terminating_newline() {
        use crate::record::lines::{RecordSchema, RecordSource, RecordVersion};
        let folder = std::env::temp_dir().join(new_id("record-byte-limit"));
        std::fs::create_dir_all(&folder).unwrap();
        let bodies = [LineBody::Diagnostic {
            stream: "stderr".into(),
            scope: "runtime".into(),
            message: String::new(),
        }];
        for (index, body) in bodies.into_iter().enumerate() {
            let path = folder.join(format!("optional-{index}.jsonl"));
            let mut writer = RecordWriter::create(&path).unwrap();
            let mut line = RecordLine {
                schema: RecordSchema::Run,
                version: RecordVersion::V1Alpha1,
                recording_id: "recording".into(),
                request_id: "request".into(),
                sequence: 0,
                time: "2026-10-25T00:00:00.000Z".into(),
                elapsed_seconds: 0.0,
                source: RecordSource::Fabric,
                runtime_id: None,
                invocation_id: None,
                body,
            };
            let overhead = serde_json::to_vec(&line).unwrap().len() + 1;
            let capacity = MAX_DIAGNOSTIC_BYTES - overhead;
            let fill = |line: &mut RecordLine, size| match &mut line.body {
                LineBody::Diagnostic { message, .. } => *message = "x".repeat(size),

                _ => panic!("expected optional record"),
            };
            fill(&mut line, capacity);
            assert_eq!(writer.append(&line).unwrap(), AppendOutcome::Written);
            assert_eq!(
                std::fs::metadata(&path).unwrap().len(),
                MAX_DIAGNOSTIC_BYTES as u64
            );
            fill(&mut line, capacity + 1);
            assert_eq!(writer.append(&line).unwrap(), AppendOutcome::Omitted);
            assert_eq!(
                std::fs::metadata(&path).unwrap().len(),
                MAX_DIAGNOSTIC_BYTES as u64
            );
        }
    }

    #[test]
    fn writer_refuses_existing_files_and_symlinks() {
        let folder = std::env::temp_dir().join(new_id("record-writer"));
        std::fs::create_dir_all(&folder).unwrap();
        let path = folder.join(RECORD_FILE_NAME);
        std::fs::write(&path, b"existing evidence").unwrap();
        assert_eq!(
            RecordWriter::create(&path).unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"existing evidence");
        #[cfg(unix)]
        {
            let link = folder.join("linked.jsonl");
            std::os::unix::fs::symlink(&path, &link).unwrap();
            assert!(RecordWriter::create(&link).is_err());
        }
    }

    #[test]
    #[cfg(unix)]
    fn writer_creates_owner_only_file() {
        use std::os::unix::fs::PermissionsExt;
        let folder = std::env::temp_dir().join(new_id("record-private"));
        std::fs::create_dir_all(&folder).unwrap();
        let path = folder.join(RECORD_FILE_NAME);
        let _writer = RecordWriter::create(&path).unwrap();
        assert_eq!(
            std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
}
