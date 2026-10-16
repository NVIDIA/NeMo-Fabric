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

/// Appends lines to one invocation's record.
#[derive(Debug)]
pub(crate) struct RecordWriter {
    file: File,
}

impl RecordWriter {
    /// Create the record at `path`. `create_new` refuses an existing file or
    /// symbolic link at `path`, and on Unix the file is mode 0600.
    pub(crate) fn create(path: &Path) -> io::Result<Self> {
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
        let mut bytes = serde_json::to_vec(line).map_err(io::Error::other)?;
        bytes.push(b'\n');
        self.file.write_all(&bytes)
    }

    /// Flush the file's data and metadata to stable storage.
    pub(crate) fn sync(&self) -> io::Result<()> {
        self.file.sync_all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_folder(name: &str) -> std::path::PathBuf {
        let folder = std::env::temp_dir().join(format!(
            "fabric-record-writer-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&folder);
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
        let _ = std::fs::remove_dir_all(folder);
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
        let _ = std::fs::remove_dir_all(folder);
    }
}
