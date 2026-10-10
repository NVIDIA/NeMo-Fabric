// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private serialization model for the runtime-control wire protocol.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub(crate) const PROTOCOL_VERSION: &str = "fabric.runtime-control.v1alpha1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RuntimeControlOperation {
    Start,
    Invoke,
    Stop,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RuntimeAdapterProcess {
    pub(crate) command: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) cwd: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub(crate) env: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum RuntimeControlCommand {
    Start {
        process: RuntimeAdapterProcess,
        lifecycle: Value,
    },
    Invoke {
        lifecycle: Value,
    },
    Stop {
        lifecycle: Value,
    },
}

impl RuntimeControlCommand {
    pub(crate) const fn operation(&self) -> RuntimeControlOperation {
        match self {
            Self::Start { .. } => RuntimeControlOperation::Start,
            Self::Invoke { .. } => RuntimeControlOperation::Invoke,
            Self::Stop { .. } => RuntimeControlOperation::Stop,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct RuntimeControlRequest {
    pub(crate) protocol_version: String,
    pub(crate) operation_id: String,
    pub(crate) environment_id: String,
    pub(crate) runtime_id: String,
    pub(crate) timeout_seconds: u64,
    #[serde(flatten)]
    pub(crate) command: RuntimeControlCommand,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RuntimeControlFailure {
    pub(crate) code: String,
    pub(crate) message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct RuntimeControlResponse {
    pub(crate) protocol_version: String,
    pub(crate) operation_id: String,
    pub(crate) environment_id: String,
    pub(crate) runtime_id: String,
    pub(crate) operation: RuntimeControlOperation,
    #[serde(flatten)]
    pub(crate) outcome: RuntimeControlOutcome,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum RuntimeControlOutcome {
    Succeeded { output: Value },
    Failed { error: RuntimeControlFailure },
}
