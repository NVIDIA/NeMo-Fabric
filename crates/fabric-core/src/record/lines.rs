// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Line types of `run-record.jsonl`, schema `nemo.fabric.run_record` version
//! `v1alpha1`.
//!
//! Every field is typed. A line holds identifiers, names, numbers, closed codes
//! and the values of a few allowlisted request context keys. It never holds a
//! value outside that allowlist: no prompt, output, error message or metadata.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::Serialize;

use crate::runtime::{ErrorStage, RunStatus};

/// Longest value a line keeps from config or the request, in characters.
/// Model role and context key names are kept whole, so map keys stay distinct.
const MAX_VALUE_CHARS: usize = 256;
/// Longest adapter error code a line keeps.
const MAX_CODE_CHARS: usize = 128;
/// Error code written in place of a code outside `^[a-z0-9_.:-]{1,128}$`.
const UNRECOGNIZED_CODE: &str = "unrecognized";

/// One line of a run record.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[schemars(title = "RunRecordLine")]
pub(crate) struct RecordLine {
    /// Always `nemo.fabric.run_record`.
    pub schema: RecordSchema,
    /// Line format version. `v1alpha1` may change until a record reader ships.
    pub version: RecordVersion,
    /// Runtime id.
    pub runtime_id: String,
    /// Invocation id.
    pub invocation_id: String,
    /// Request id.
    pub request_id: String,
    /// UTC wall-clock time the line was written, RFC 3339 with milliseconds.
    #[schemars(extend("format" = "date-time"))]
    pub time: String,
    /// Line kind and its fields.
    #[serde(flatten)]
    pub body: LineBody,
}

/// The record's schema name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
pub(crate) enum RecordSchema {
    /// `nemo.fabric.run_record`.
    #[serde(rename = "nemo.fabric.run_record")]
    RunRecord,
}

/// The record's line format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
pub(crate) enum RecordVersion {
    /// `v1alpha1`.
    #[serde(rename = "v1alpha1")]
    V1Alpha1,
}

/// A line's kind, with the fields that kind carries.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum LineBody {
    /// Written once, before core sends the request to the adapter host.
    Start(StartLine),
    /// Written once, after the invocation returned a result or failed.
    End(EndLine),
}

/// Fields of the start line.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub(crate) struct StartLine {
    /// Config name (`metadata.name`).
    pub agent_name: String,
    /// Harness identifier.
    pub harness: String,
    /// Adapter implementation id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_id: Option<String>,
    /// Configured models by role: provider and model name only.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub models: BTreeMap<String, ModelName>,
    /// NeMo Fabric version that wrote the record.
    pub fabric_version: String,
    /// Host invocation timeout that applied, in milliseconds.
    pub timeout_ms: u64,
    /// `RunRequest.context`. Allowlisted identity and Harbor link keys keep a
    /// string or integer value; every other key is listed by name only.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub context: BTreeMap<String, ContextValue>,
    /// Top-level key names of `RunRequest.overrides`, sorted. Values are never
    /// recorded.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub override_keys: Vec<String>,
    /// `runtime.record.exclusive_container`: whether a rise in
    /// `container_oom_kills` can set the `sandbox_failed` cause.
    pub exclusive_container: bool,
    /// The container's cgroup v2 `oom_kill` count when the invocation started.
    pub container_oom_kills: OomKills,
}

/// One configured model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
pub(crate) struct ModelName {
    /// Model provider name.
    pub provider: String,
    /// Provider model identifier.
    pub model: String,
}

/// A request context value as the start line keeps it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(untagged)]
pub(crate) enum ContextValue {
    /// An allowlisted key's string value, cut to 256 characters.
    Text(String),
    /// An allowlisted key's integer value.
    Integer(i64),
    /// Any other key or value: the key name only.
    NotRecorded(NotRecorded),
}

/// Marks a key whose value the record leaves out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
pub(crate) struct NotRecorded {
    /// Always `false`.
    #[schemars(extend("const" = false))]
    pub recorded: bool,
}

impl NotRecorded {
    pub(crate) const VALUE: Self = Self { recorded: false };
}

/// Fields of the end line.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub(crate) struct EndLine {
    /// Final status. A Fabric lifecycle error, such as a host timeout, reads `failed`.
    pub status: RunStatus,
    /// Monotonic time from the start line to the end line, in milliseconds.
    pub elapsed_ms: u64,
    /// Error code, stage and retryable flag, without the message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<EndError>,
    /// Numeric usage counters, without adapter metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<EndUsage>,
    /// Result artifacts, with paths relative to the artifact root.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<EndArtifact>,
    /// The container's cgroup v2 `oom_kill` count when the invocation ended.
    pub container_oom_kills: OomKills,
    /// Why the invocation stopped; absent when it holds nothing, as on success.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Stop>,
}

/// The cgroup v2 `memory.events` `oom_kill` count of the container Fabric
/// runs in, or why it could not be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(untagged)]
pub(crate) enum OomKills {
    /// The count.
    Count(u64),
    /// The count could not be read.
    Unavailable {
        /// Why the count could not be read.
        reason: OomKillsUnavailable,
    },
}

/// Why the container's `oom_kill` count could not be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OomKillsUnavailable {
    /// The operating system has no cgroup v2 source.
    Platform,
    /// The host uses cgroup v1.
    CgroupV1,
    /// Fabric does not run at the root of its own cgroup, or the cgroup files
    /// cannot be read.
    NoOwnedCgroup,
}

/// Why an invocation stopped.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, JsonSchema)]
pub(crate) struct Stop {
    /// Fabric's failure cause; absent on a cancel alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cause: Option<StopCause>,
    /// What stopped the invocation, when a limit did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<StopReason>,
    /// Whether Fabric force-stopped the adapter host. After a forced stop on
    /// Linux and macOS, `exit_signal` is usually the signal Fabric sent (9),
    /// unless the host exited by itself during the 2-second grace period.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forced: Option<bool>,
    /// The limit that ended the invocation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<StopLimit>,
    /// The adapter host's exit code, when it exited.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    /// The signal that ended the adapter host, when one did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_signal: Option<i32>,
}

/// Fabric's own closed list of failure causes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StopCause {
    /// Core or the adapter host raised an error other than a timeout or a
    /// host exit, such as an invalid adapter result or an adapter crash
    /// inside the host. An error from the exchange with the host during an
    /// out-of-memory rise in an exclusive container reads `SandboxFailed`.
    FabricFailed,
    /// The container's `oom_kill` count rose and
    /// `runtime.record.exclusive_container` is set.
    SandboxFailed,
    /// A limit ended the invocation, such as Fabric's timeout.
    AgentExhaustedBudget,
    /// The adapter host exited, or nothing else matched.
    Unclassified,
    /// The adapter returned a failed result.
    AgentFailed,
}

/// What stopped an invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StopReason {
    /// The invocation reached its time limit.
    Timeout,
}

/// The limit that ended an invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
pub(crate) struct StopLimit {
    /// The kind of limit.
    #[serde(rename = "type")]
    pub limit_type: LimitType,
    /// The configured limit, in milliseconds for `time`: the invoke timeout
    /// that applied, `runtime.timeout_seconds` or 1 hour when it is unset.
    /// Core gives the adapter an earlier deadline, the timeout less
    /// 10 seconds and at least half of it.
    pub value: u64,
    /// What the invocation used when it stopped, in the same unit. For a
    /// forced stop, the time until core's timeout fired.
    pub usage: u64,
}

/// Kinds of limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LimitType {
    /// Wall-clock time.
    Time,
}

/// The end line's error.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
pub(crate) struct EndError {
    /// Error code matching `^[a-z0-9_.:-]{1,128}$`, else `unrecognized`.
    #[schemars(regex(pattern = r"^[a-z0-9_.:-]{1,128}$"))]
    pub code: String,
    /// Lifecycle stage where the failure surfaced.
    pub stage: ErrorStage,
    /// Whether a consumer-level retry is considered safe.
    pub retryable: bool,
}

/// The end line's usage: the numeric `RunUsage` fields.
#[derive(Debug, Clone, Default, PartialEq, Serialize, JsonSchema)]
pub(crate) struct EndUsage {
    /// Input tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    /// Cached input tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_input_tokens: Option<u64>,
    /// Whether `input_tokens` includes `cached_input_tokens`; absent means unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens_include_cache: Option<bool>,
    /// Output tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    /// Total tokens reported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    /// Reported cost in US dollars.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0))]
    pub cost_usd: Option<f64>,
}

/// One result artifact on the end line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
pub(crate) struct EndArtifact {
    /// Logical artifact name.
    pub name: String,
    /// Artifact kind.
    pub kind: String,
    /// Media type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    /// Path relative to the artifact root; absent for a path outside it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

/// Cut `text` to [`MAX_VALUE_CHARS`] characters: 255 plus "…".
pub(crate) fn cut_text(text: &str) -> String {
    if text.chars().count() <= MAX_VALUE_CHARS {
        return text.to_string();
    }
    let mut cut: String = text.chars().take(MAX_VALUE_CHARS - 1).collect();
    cut.push('…');
    cut
}

/// An error code as the end line keeps it: unchanged when it matches
/// `^[a-z0-9_.:-]{1,128}$`, else [`UNRECOGNIZED_CODE`], because a code outside
/// that pattern may be a message.
pub(crate) fn recorded_code(code: &str) -> String {
    let valid = (1..=MAX_CODE_CHARS).contains(&code.len())
        && code.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"_.:-".contains(&byte)
        });
    if valid {
        code.to_string()
    } else {
        UNRECOGNIZED_CODE.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cut_text_over_the_limit_keeps_255_chars_and_an_ellipsis() {
        let cut = cut_text(&"é".repeat(1000));

        assert_eq!(cut.chars().count(), MAX_VALUE_CHARS);
        assert!(cut.ends_with('…'));
        assert_eq!(cut_text("short"), "short");
        assert_eq!(cut_text(&"a".repeat(256)), "a".repeat(256));
    }

    #[test]
    fn recorded_code_keeps_only_codes_in_the_pattern() {
        for code in [
            "timeout",
            "host_timeout",
            "provider.rate_limit",
            "http:429",
            "a-b",
        ] {
            assert_eq!(recorded_code(code), code);
        }
        for code in [
            "",
            "Timeout",
            "rate limit",
            "error: key sk-123",
            "über",
            &"a".repeat(129),
        ] {
            assert_eq!(recorded_code(code), UNRECOGNIZED_CODE, "{code}");
        }
        assert_eq!(recorded_code(&"a".repeat(128)), "a".repeat(128));
    }
}
