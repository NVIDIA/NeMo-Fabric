// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private command contract between the host runtime and environment provider.
//!
//! `TaskCommand` names the task-side start, invoke, and stop operations while
//! carrying the full public SDK config or request as JSON. The provider returns
//! either the SDK value or a normalized diagnostic through `TaskCommandResponse`.
//! Command execution and task-runtime state live outside this module.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub(crate) enum TaskCommand {
    Start {
        agent: String,
        config: Value,
    },
    Invoke {
        agent: String,
        runtime_id: String,
        request: Value,
    },
    Stop {
        agent: String,
        runtime_id: String,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum TaskCommandResponse {
    Returned { value: Value },
    Rejected { error: TaskDiagnostic },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TaskDiagnostic {
    pub(crate) code: String,
    pub(crate) stage: String,
    pub(crate) effects: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdk_invocation_matches_the_provider_golden_contract() {
        let expected: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/openshell_echo/invoke-command.json"
        ))
        .expect("golden contract");
        let command = TaskCommand::Invoke {
            agent: "echo".to_string(),
            runtime_id: "runtime-task".to_string(),
            request: expected["request"].clone(),
        };
        assert_eq!(
            serde_json::to_value(command).expect("command JSON"),
            expected
        );
    }
}
