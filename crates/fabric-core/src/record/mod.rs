// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The opt-in per-invocation run record, `run-record.jsonl`.
//!
//! When `runtime.record.enabled` is set, core appends one start line before it
//! sends the request to the adapter host and one end line after the invocation
//! returns or fails. Recording is synchronous and best-effort: record I/O
//! errors do not replace the invocation result. A failed write stops the
//! record and becomes a `record_write_failed` event on the result, and slow
//! storage delays the invocation.
//!
//! Everything here is crate-private. The line format is public only through
//! the `run-record-line` JSON Schema, version `v1alpha1`.

mod cgroup;
mod invocation;
mod lines;
mod stop;
mod writer;

pub(crate) use cgroup::CgroupReader;
pub(crate) use invocation::{HostExit, InvocationRecorder, RecordStart};
pub(crate) use lines::RecordLine;
#[cfg(test)]
pub(crate) use writer::test_hooks;
