// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The open record file that the invocation and the sample thread share.

use std::io;
use std::path::Path;
use std::sync::{Mutex, MutexGuard};
use std::time::SystemTime;

use super::lines::{LineBody, RecordLine, RecordSchema, RecordVersion};
use super::writer::RecordWriter;
use crate::time::rfc3339_millis;

/// One invocation's record file. After the first failed write it stops.
#[derive(Debug)]
pub(crate) struct RecordFile {
    pub runtime_id: String,
    pub invocation_id: String,
    pub request_id: String,
    writer: Option<RecordWriter>,
    /// The first create, write or sync failure.
    pub failure: Option<io::Error>,
    /// Set when the sampling window closes, at the exchange's end or at the
    /// end line; no sample line follows it.
    sampling_closed: bool,
}

impl RecordFile {
    /// Create the file at `path`. A failed create is kept in `failure`, and
    /// every later append is skipped.
    pub(crate) fn create(
        path: &Path,
        runtime_id: String,
        invocation_id: String,
        request_id: String,
    ) -> Self {
        let (writer, failure) = match RecordWriter::create(path) {
            Ok(writer) => (Some(writer), None),
            Err(error) => (None, Some(error)),
        };
        Self {
            runtime_id,
            invocation_id,
            request_id,
            writer,
            failure,
            sampling_closed: false,
        }
    }

    /// Append one line.
    pub(crate) fn append(&mut self, body: LineBody) {
        let Some(writer) = self.writer.as_mut() else {
            return;
        };
        let line = RecordLine {
            schema: RecordSchema::RunRecord,
            version: RecordVersion::V1Alpha1,
            runtime_id: self.runtime_id.clone(),
            invocation_id: self.invocation_id.clone(),
            request_id: self.request_id.clone(),
            time: rfc3339_millis(SystemTime::now()),
            body,
        };
        if let Err(error) = writer.append(&line) {
            self.failure = Some(error);
            self.writer = None;
        }
    }

    /// Append a sample line, unless the sampling window is closed.
    pub(crate) fn append_sample(&mut self, body: LineBody) {
        if !self.sampling_closed {
            self.append(body);
        }
    }

    /// Close the sampling window. A sample whose read finishes later is
    /// discarded.
    pub(crate) fn close_sampling(&mut self) {
        self.sampling_closed = true;
    }

    /// Append the end line, sync the file and close it. No sample line
    /// follows, and a sample tick that still holds this record keeps no open
    /// file handle.
    pub(crate) fn append_end(&mut self, body: LineBody) {
        self.sampling_closed = true;
        self.append(body);
        if let Some(writer) = self.writer.take()
            && let Err(error) = writer.sync()
        {
            self.failure = Some(error);
        }
    }
}

/// Lock a shared record file. A panic elsewhere while it was held leaves the
/// file usable: every write is one append.
pub(crate) fn lock(file: &Mutex<RecordFile>) -> MutexGuard<'_, RecordFile> {
    file.lock().unwrap_or_else(|error| error.into_inner())
}
