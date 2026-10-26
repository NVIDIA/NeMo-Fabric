// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Observe the host's existing stderr file without changing its writer or stdout
// protocol. Timestamps describe capture; the source does not supply emission times.

use super::RunRecorder;
use std::fs::{File, Metadata};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const DIAGNOSTIC_BYTE_BUDGET: u64 = 1024 * 1024;
const READ_CHUNK_BYTES: usize = 4096;
const POLL_INTERVAL: Duration = Duration::from_millis(25);

/// Request-scoped capture of runtime stderr, including possible background work.
/// It stops at request return and never waits for the host process to exit.
#[derive(Debug)]
pub(crate) struct DiagnosticObserver {
    stop: Option<Sender<()>>,
    worker: Option<JoinHandle<()>>,
    recorder: RunRecorder,
}

impl DiagnosticObserver {
    pub(crate) fn start(path: PathBuf, offset: u64, recorder: RunRecorder) -> Self {
        // Open synchronously so fast startup failure/cleanup cannot retire the
        // path before the observer owns a descriptor for this exact source.
        let tail = match DiagnosticTail::open(&path, offset) {
            Ok(tail) => tail,
            Err(error) => {
                recorder.diagnostic_retention(0, Some(error));
                return Self {
                    stop: None,
                    worker: None,
                    recorder,
                };
            }
        };
        let (sender, receiver) = mpsc::channel();
        let sink = recorder.clone();
        match thread::Builder::new()
            .name("fabric-diagnostics".to_string())
            .spawn(move || observe(path, tail, sink, receiver))
        {
            Ok(worker) => Self {
                stop: Some(sender),
                worker: Some(worker),
                recorder,
            },
            Err(error) => {
                recorder.diagnostic_retention(0, Some(format!("observer_start_failed: {error}")));
                Self {
                    stop: None,
                    worker: None,
                    recorder,
                }
            }
        }
    }

    pub(crate) fn finish(mut self) {
        self.stop_and_join();
    }

    fn stop_and_join(&mut self) {
        if let Some(sender) = self.stop.take() {
            // A disconnected worker has already finished and recorded its outcome.
            let _ = sender.send(());
        }
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            self.recorder
                .diagnostic_retention(0, Some("observer_panicked".to_string()));
        }
    }
}

impl Drop for DiagnosticObserver {
    fn drop(&mut self) {
        self.stop_and_join();
    }
}

fn observe(path: PathBuf, mut tail: DiagnosticTail, recorder: RunRecorder, receiver: Receiver<()>) {
    loop {
        if let Err(error) = tail.drain(&path, &mut |message| recorder.diagnostic("stderr", message))
        {
            tail.read_error = Some(error);
            break;
        }
        match receiver.recv_timeout(POLL_INTERVAL) {
            Err(RecvTimeoutError::Timeout) => continue,
            Ok(()) | Err(RecvTimeoutError::Disconnected) => {
                // One finite snapshot, not a loop chasing a still-running host.
                if let Err(error) =
                    tail.drain(&path, &mut |message| recorder.diagnostic("stderr", message))
                {
                    tail.read_error = Some(error);
                }
                break;
            }
        }
    }
    tail.finish();
    recorder.diagnostic_retention(tail.omitted_bytes, tail.read_error);
}

struct DiagnosticTail {
    file: File,
    identity: Metadata,
    offset: u64,
    source_bytes_read: u64,
    pending_utf8: Vec<u8>,
    omitted_bytes: u64,
    read_error: Option<String>,
}

impl DiagnosticTail {
    fn open(path: &Path, offset: u64) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|error| format!("open_failed: {error}"))?;
        let identity = file
            .metadata()
            .map_err(|error| format!("metadata_failed: {error}"))?;
        if identity.len() < offset {
            return Err("source_truncated_before_capture".to_string());
        }
        file.seek(SeekFrom::Start(offset))
            .map_err(|error| format!("seek_failed: {error}"))?;
        Ok(Self {
            file,
            identity,
            offset,
            source_bytes_read: 0,
            pending_utf8: Vec::with_capacity(READ_CHUNK_BYTES + 3),
            omitted_bytes: 0,
            read_error: None,
        })
    }

    fn fail(&mut self, error: String) {
        if self.read_error.is_none() {
            self.read_error = Some(error);
        }
    }

    fn drain(&mut self, path: &Path, emit: &mut impl FnMut(&str)) -> Result<(), String> {
        let metadata = self
            .file
            .metadata()
            .map_err(|error| format!("metadata_failed: {error}"))?;
        match std::fs::metadata(path) {
            Ok(path_metadata) if source_replaced(&self.identity, &path_metadata) => {
                return Err("source_replaced".to_string());
            }
            Ok(_) => {}
            // Owned runtime cleanup can remove the path before final drain.
            // The open descriptor still supplies the original source bytes.
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("source_metadata_failed: {error}")),
        }
        let end = metadata.len();
        if end < self.offset {
            return Err("source_truncated".to_string());
        }
        let mut buffer = [0_u8; READ_CHUNK_BYTES];
        while self.offset < end && self.source_bytes_read < DIAGNOSTIC_BYTE_BUDGET {
            let available =
                (end - self.offset).min(DIAGNOSTIC_BYTE_BUDGET - self.source_bytes_read);
            let count = usize::try_from(available.min(READ_CHUNK_BYTES as u64))
                .expect("bounded read length");
            let read = match self.file.read(&mut buffer[..count]) {
                Ok(0) => return Err("source_changed_during_read".to_string()),
                Ok(read) => read,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(format!("read_failed: {error}")),
            };
            self.offset += read as u64;
            self.source_bytes_read += read as u64;
            self.pending_utf8.extend_from_slice(&buffer[..read]);
            self.emit_complete_utf8(emit);
        }
        if self.offset < end {
            self.omitted_bytes = self.omitted_bytes.saturating_add(end - self.offset);
            self.offset = end;
            self.file
                .seek(SeekFrom::Start(end))
                .map_err(|error| format!("seek_failed: {error}"))?;
        }
        Ok(())
    }

    fn emit_complete_utf8(&mut self, emit: &mut impl FnMut(&str)) {
        while !self.pending_utf8.is_empty() {
            match std::str::from_utf8(&self.pending_utf8) {
                Ok(message) => {
                    emit(message);
                    self.pending_utf8.clear();
                }
                Err(error) => {
                    let valid = error.valid_up_to();
                    if valid != 0 {
                        // The decoder established this exact prefix as valid UTF-8.
                        emit(
                            std::str::from_utf8(&self.pending_utf8[..valid])
                                .expect("validated UTF-8 prefix"),
                        );
                        self.pending_utf8.drain(..valid);
                    }
                    if let Some(invalid) = error.error_len() {
                        self.pending_utf8.drain(..invalid);
                        self.omitted_bytes = self.omitted_bytes.saturating_add(invalid as u64);
                        self.fail("invalid_utf8".to_string());
                    } else {
                        // At most three bytes belong to an unfinished code point.
                        break;
                    }
                }
            }
        }
    }

    fn finish(&mut self) {
        if !self.pending_utf8.is_empty() {
            self.omitted_bytes = self
                .omitted_bytes
                .saturating_add(self.pending_utf8.len() as u64);
            self.pending_utf8.clear();
            self.fail("incomplete_utf8_at_capture_end".to_string());
        }
    }
}

#[cfg(unix)]
fn source_replaced(initial: &Metadata, current: &Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    initial.dev() != current.dev() || initial.ino() != current.ino()
}

#[cfg(not(unix))]
fn source_replaced(_initial: &Metadata, _current: &Metadata) -> bool {
    // Fabric owns this runtime file. External replacement is unsupported;
    // portable continuity checks still detect observed shrinkage.
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn source(contents: &[u8]) -> (PathBuf, File) {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "fabric-diagnostic-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)
            .expect("private fixture");
        file.write_all(contents).expect("source bytes");
        (path, file)
    }

    #[test]
    fn tail_preserves_code_points_split_across_reads_and_poll_boundaries() {
        let (path, mut file) = source(b"previous\n");
        let mut tail = DiagnosticTail::open(&path, 9).expect("tail");
        let mut messages = String::new();
        let emoji = "\u{1f680}".as_bytes();
        file.write_all(&vec![b'x'; READ_CHUNK_BYTES - 1])
            .expect("prefix");
        file.write_all(&emoji[..2]).expect("partial code point");
        tail.drain(&path, &mut |message| messages.push_str(message))
            .expect("first drain");
        assert_eq!(messages, "x".repeat(READ_CHUNK_BYTES - 1));
        file.write_all(&emoji[2..]).expect("completed code point");
        tail.drain(&path, &mut |message| messages.push_str(message))
            .expect("second drain");
        tail.finish();
        assert_eq!(
            messages,
            format!("{}\u{1f680}", "x".repeat(READ_CHUNK_BYTES - 1))
        );
        assert_eq!(tail.omitted_bytes, 0);
        assert_eq!(tail.read_error, None);
    }

    #[test]
    fn source_byte_budget_counts_later_bytes_without_reading_or_changing_them() {
        let bytes = vec![b'x'; DIAGNOSTIC_BYTE_BUDGET as usize + 31];
        let (path, mut file) = source(&bytes);
        let mut tail = DiagnosticTail::open(&path, 0).expect("tail");
        let mut captured = 0;
        tail.drain(&path, &mut |message| captured += message.len())
            .expect("capture");
        file.write_all(b"later diagnostic").expect("later bytes");
        tail.drain(&path, &mut |message| captured += message.len())
            .expect("later capture");
        tail.finish();
        assert_eq!(captured, DIAGNOSTIC_BYTE_BUDGET as usize);
        assert_eq!(tail.omitted_bytes, 31 + b"later diagnostic".len() as u64);
        assert_eq!(
            std::fs::metadata(&path).expect("source retained").len(),
            bytes.len() as u64 + b"later diagnostic".len() as u64
        );
    }

    #[test]
    fn open_source_remains_readable_after_its_path_is_retired() {
        let (path, mut file) = source(b"before cleanup\n");
        let mut tail = DiagnosticTail::open(&path, 0).expect("tail");
        let retired = path.with_extension("retired");
        std::fs::rename(&path, &retired).expect("retire source path");
        file.write_all(b"last shutdown diagnostic\n")
            .expect("final bytes");
        let mut captured = String::new();
        tail.drain(&path, &mut |message| captured.push_str(message))
            .expect("descriptor remains readable");
        assert_eq!(captured, "before cleanup\nlast shutdown diagnostic\n");
        assert_eq!(tail.read_error, None);
    }

    #[test]
    fn source_shrink_stops_capture_without_replaying_previous_bytes() {
        let (path, file) = source(b"first");
        let mut tail = DiagnosticTail::open(&path, 0).expect("tail");
        let mut messages = String::new();
        tail.drain(&path, &mut |message| messages.push_str(message))
            .expect("first capture");
        file.set_len(0).expect("truncate source");
        assert_eq!(
            tail.drain(&path, &mut |message| messages.push_str(message)),
            Err("source_truncated".to_string())
        );
        assert_eq!(messages, "first");
    }
}
