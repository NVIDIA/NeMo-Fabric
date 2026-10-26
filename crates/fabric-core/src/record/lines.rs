// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed lines in the shared, request-scoped run journal.

use crate::model_usage::ModelUsage;
use crate::runtime::{ErrorInfo, RunResult, RunStatus};
use crate::system_stats::InvocationSystemStats;
use schemars::JsonSchema;
use serde::Serialize;
use std::collections::BTreeMap;

/// One independently parseable line of a recorded request.
#[derive(Debug, Clone, Serialize, JsonSchema)]
#[schemars(title = "RunRecordLine")]
pub(crate) struct RecordLine {
    pub schema: RecordSchema,
    pub version: RecordVersion,
    pub recording_id: String,
    pub request_id: String,
    pub sequence: u64,
    #[schemars(extend("format" = "date-time"))]
    pub time: String,
    #[schemars(range(min = 0.0))]
    pub elapsed_seconds: f64,
    pub source: RecordSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub invocation_id: Option<String>,
    #[serde(flatten)]
    pub body: LineBody,
}

#[derive(Debug, Clone, Copy, Serialize, JsonSchema)]
pub(crate) enum RecordSchema {
    #[serde(rename = "nemo.fabric.run")]
    Run,
}

#[derive(Debug, Clone, Copy, Serialize, JsonSchema)]
pub(crate) enum RecordVersion {
    #[serde(rename = "v1alpha1")]
    V1Alpha1,
}

#[derive(Debug, Clone, Copy, Serialize, JsonSchema)]
pub(crate) enum RecordSource {
    #[serde(rename = "fabric")]
    Fabric,
}

/// The operation whose return boundary closes this journal.
#[derive(Debug, Clone, Copy, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OutcomeScope {
    Invocation,
    OneShotLifecycle,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
pub(crate) enum LineBody {
    RequestStarted(RequestStarted),
    RuntimeReady {
        runtime_id: String,
    },
    InvocationStarted {
        timeout_seconds: f64,
    },
    InvocationFinished(Box<InvocationFinished>),
    RuntimeStopped {
        succeeded: bool,
        error: Option<ErrorInfo>,
    },
    Diagnostic {
        stream: String,
        scope: String,
        message: String,
    },
    DiagnosticRetention {
        omitted_bytes: u64,
        read_error: Option<String>,
    },
    ResourceSample(Box<ResourceSample>),
    Summary(Box<Summary>),
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub(crate) struct RequestStarted {
    pub outcome_scope: OutcomeScope,
    pub agent_name: String,
    pub harness: String,
    pub models: BTreeMap<String, ModelName>,
    pub context: BTreeMap<String, ContextValue>,
    pub override_keys: Vec<String>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub(crate) struct ModelName {
    pub provider: String,
    pub model: String,
}

/// Evidence at invocation return is independent from later owned shutdown.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub(crate) struct InvocationFinished {
    pub status: RunStatus,
    pub error: Option<ErrorInfo>,
    pub elapsed_seconds: f64,
    pub adapter_exchange_elapsed_seconds: Option<f64>,
    pub model_usage: Option<ModelUsage>,
    pub system_stats: Option<InvocationSystemStats>,
}

/// Authoritative final evidence for this recorded operation.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub(crate) struct Summary {
    pub outcome_scope: OutcomeScope,
    pub request_started_at: String,
    pub request_finished_at: String,
    pub elapsed_seconds: f64,
    pub adapter_exchange_elapsed_seconds: Option<f64>,
    /// Independent resource evidence retained when execution returned no result.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_stats: Option<InvocationSystemStats>,
    pub result: Option<RunResult>,
    pub error: Option<ErrorInfo>,
}

/// Unknown context values are intentionally not copied into request evidence.
#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(untagged)]
pub(crate) enum ContextValue {
    Text(String),
    Integer(i64),
    NotRecorded { recorded: bool },
}

/// A sampled observation. Its offset is relative to the adapter exchange;
/// the envelope's elapsed time is relative to request registration.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub(crate) struct ResourceSample {
    pub offset_seconds: f64,
    pub adapter_tree: crate::system_stats::ResourceObservation,
    pub cgroup: crate::system_stats::CgroupObservation,
    pub collection_seconds: f64,
}
