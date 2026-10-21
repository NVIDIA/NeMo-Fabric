// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public health request and no-probe behavior.

use nemo_fabric_core::{
    CheckOutcome, CheckReason, CheckRequest, EnvironmentHandle, check_environment,
};
use serde_json::json;

fn environment() -> EnvironmentHandle {
    serde_json::from_value(json!({
        "environment_id": "check-test-environment",
        "provider": "local",
        "control_location": "external_control",
        "ownership": "caller_owned"
    }))
    .unwrap()
}

#[test]
fn check_request_defaults_and_invalid_combinations() {
    let request: CheckRequest = serde_json::from_value(json!({
        "scope": "runtime", "runtime_id": "selected"
    }))
    .unwrap();
    assert_eq!(request.level().unwrap().as_str(), "live");
    assert_eq!(request.timeout_seconds(), 3.0);
    for value in [
        json!({"scope": "server", "runtime_id": "selected"}),
        json!({"scope": "server", "level": "live"}),
        json!({"scope": "runtime"}),
        json!({"scope": "runtime", "runtime_id": ""}),
        json!({"scope": "runtime", "runtime_id": " "}),
        json!({"scope": "runtime", "runtime_id": "selected", "extra": "PRIVATE"}),
        json!({"scope": "server", "timeout_seconds": 0}),
        json!({"scope": "server", "timeout_seconds": true}),
        json!({"scope": "server", "schema_version": "wrong"}),
    ] {
        assert!(serde_json::from_value::<CheckRequest>(value).is_err());
    }
    assert!(
        serde_json::from_str::<CheckRequest>(r#"{"scope":"server","scope":"server"}"#).is_err()
    );
}

#[test]
fn runtime_scope_and_local_server_are_unsupported() {
    let request =
        serde_json::from_value(json!({"scope":"runtime", "runtime_id":"missing"})).unwrap();
    let report = check_environment(&environment(), &request).unwrap();
    assert_eq!(report.outcome, CheckOutcome::Unsupported);
    assert_eq!(report.reason_code, Some(CheckReason::UnsupportedScope));
    assert!(report.checks.is_empty());
    assert!(report.runtime.is_none());
    assert!(report.server.is_none());
    let request = serde_json::from_value(json!({"scope":"server"})).unwrap();
    let report = check_environment(&environment(), &request).unwrap();
    assert_eq!(report.outcome, CheckOutcome::Unsupported);
    assert_eq!(report.reason_code, Some(CheckReason::UnsupportedScope));
    assert!(report.checks.is_empty());
}

#[test]
fn operational_is_not_a_valid_health_level() {
    assert!(
        serde_json::from_value::<CheckRequest>(json!({
            "scope":"runtime", "runtime_id":"missing", "level":"operational"
        }))
        .is_err()
    );
}
