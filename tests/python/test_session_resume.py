# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The public session selector preserves native identity across runtimes."""

import pytest

from nemo_fabric import RunPlan, SessionRef
from nemo_fabric.errors import FabricConfigError


def test_session_reference_round_trip_and_explicit_selection():
    reference = SessionRef.from_mapping(
        {
            "schema_version": "fabric.session_ref/v1alpha1",
            "adapter_id": "nvidia.fabric.hermes",
            "session_id": "native-session",
        }
    )

    assert SessionRef.from_mapping(reference.to_mapping()) == reference


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": "unsupported"},
        {"adapter_id": ""},
        {"session_id": " "},
        {"session_id": 7},
        {"native_path": "/some/native/home"},
    ],
)
def test_session_reference_rejects_invalid_envelopes(change):
    with pytest.raises(FabricConfigError):
        SessionRef.from_mapping(
            {
                "schema_version": "fabric.session_ref/v1alpha1",
                "adapter_id": "nvidia.fabric.hermes",
                "session_id": "native-session",
                **change,
            }
        )


def test_session_reference_diagnostics_do_not_disclose_native_identifier():
    reference = SessionRef.from_mapping(
        {
            "schema_version": "fabric.session_ref/v1alpha1",
            "adapter_id": "nvidia.fabric.hermes",
            "session_id": "private-native-session",
        }
    )

    assert "private-native-session" not in repr(reference)


def test_run_plan_normalizes_the_session_reference():
    plan = RunPlan.from_mapping(
        {
            "agent_name": "demo",
            "base_dir": ".",
            "config": {
                "metadata": {"name": "demo"},
                "harness": {"adapter_id": "nvidia.fabric.hermes"},
            },
            "adapter": {"adapter_id": "nvidia.fabric.hermes", "adapter_kind": "python"},
            "session_ref": {
                "schema_version": "fabric.session_ref/v1alpha1",
                "adapter_id": "nvidia.fabric.hermes",
                "session_id": "original-session",
            },
        }
    )
    assert isinstance(plan.session_ref, SessionRef)
    assert plan.session_ref.session_id == "original-session"
    assert "session_ref" not in plan.extra_fields


@pytest.mark.parametrize(
    "value",
    [{"mode": "create"}, {"mode": "ephemeral"}, {"mode": "resume", "session_ref": {}}],
)
def test_runtime_start_rejects_mode_wrappers(value):
    from nemo_fabric.client import _with_session_ref

    with pytest.raises(FabricConfigError, match="SessionRef"):
        _with_session_ref(None, value)


@pytest.mark.parametrize(
    "value, expected",
    [(True, True), (False, False), ("false", False), (1, False), (None, False)],
)
def test_native_error_retryability_requires_a_boolean(value, expected):
    from nemo_fabric.errors import _runtime_error

    error = RuntimeError("Native operation failed")
    error.code = "native_failure"
    error.retryable = value
    normalized = _runtime_error(error, stage="start")
    assert normalized.code == "native_failure"
    assert normalized.retryable is expected
