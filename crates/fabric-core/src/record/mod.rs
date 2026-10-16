// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The opt-in per-invocation run record, `run-record.jsonl`.
//!
//! When `runtime.record.enabled` is set, core appends one start line before it
//! sends the request to the adapter host and one end line after the invocation
//! returns or fails. Recording never changes the run: a failed write stops the
//! record and becomes a `record_write_failed` event on the result.
//!
//! Everything here is crate-private. The line format is public only through
//! the `run-record-line` JSON Schema, version `v1alpha1`.

mod invocation;
mod lines;
mod writer;

pub(crate) use invocation::{InvocationRecorder, RecordStart};
pub(crate) use lines::RecordLine;
