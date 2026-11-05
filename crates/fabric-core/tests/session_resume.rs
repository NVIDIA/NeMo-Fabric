// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public session envelope validation.

use nemo_fabric_core::SessionRef;
use serde_json::json;

#[test]
fn session_reference_round_trips_without_exposing_native_identity_in_debug() {
    let reference: SessionRef = serde_json::from_value(json!({
        "schema_version": "fabric.session_ref/v1alpha1",
        "adapter_id": "nvidia.fabric.hermes",
        "session_id": "private-native-session"
    }))
    .expect("valid reference");
    reference.validate().expect("valid reference");
    assert_eq!(
        serde_json::to_value(&reference).unwrap()["session_id"],
        "private-native-session"
    );
    assert!(!format!("{reference:?}").contains("private-native-session"));
}

#[test]
fn session_reference_rejects_unknown_fields_and_unsupported_versions() {
    assert!(
        serde_json::from_value::<SessionRef>(json!({
            "schema_version": "fabric.session_ref/v1alpha1",
            "adapter_id": "nvidia.fabric.hermes",
            "session_id": "native-session",
            "path": "/native/home"
        }))
        .is_err()
    );
    let reference: SessionRef = serde_json::from_value(json!({
        "schema_version": "unsupported",
        "adapter_id": "nvidia.fabric.hermes",
        "session_id": "native-session"
    }))
    .unwrap();
    assert!(reference.validate().is_err());
    assert!(
        serde_json::from_value::<SessionRef>(json!({
            "mode": "create", "session_ref": {}
        }))
        .is_err()
    );
}

#[test]
fn session_reference_schema_rejects_unsupported_or_blank_identity() {
    let schema =
        nemo_fabric_core::schema::generate_schema(nemo_fabric_core::schema::SchemaName::SessionRef)
            .unwrap();
    for change in [
        json!({"schema_version": "unsupported"}),
        json!({"adapter_id": " "}),
        json!({"session_id": ""}),
    ] {
        let mut reference = json!({"schema_version": "fabric.session_ref/v1alpha1", "adapter_id": "nvidia.fabric.hermes", "session_id": "native-session"});
        reference
            .as_object_mut()
            .unwrap()
            .extend(change.as_object().unwrap().clone());
        assert!(!jsonschema::is_valid(&schema, &reference));
    }
}
