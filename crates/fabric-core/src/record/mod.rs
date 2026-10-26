// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One best-effort, append-only `run.jsonl` per recorded SDK operation.
//!
//! Recording starts before validation and startup. I/O failures never replace
//! execution outcomes. The terminal summary preserves the returned result.

mod invocation;
mod lines;
mod results;
mod writer;

pub(crate) use invocation::RunRecorder;
pub(crate) use lines::{OutcomeScope, RecordLine};
pub use results::run_record_folder;
#[cfg(test)]
pub(crate) use writer::test_hooks;
