# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Malformed optional reports cannot replace successful or failed outcomes."""

import pytest
from nemo_fabric import RunResult


@pytest.mark.parametrize(
    "report", ["broken", [], 42, True, {"software_environments": "broken"}]
)
@pytest.mark.parametrize("status", ["succeeded", "failed"])
def test_optional_report_failure_preserves_terminal_outcome(report, status):
    original = {
        "agent_name": "example",
        "harness": "example",
        "adapter_kind": "python",
        "runtime_id": "runtime",
        "invocation_id": "invocation",
        "request_id": "request",
        "status": status,
        "output": {"partial": True},
        "usage": {"input_tokens": 9},
    }
    if status == "failed":
        original["error"] = {
            "stage": "invoke",
            "code": "test_failure",
            "message": "original failure",
        }
    decoded = RunResult.from_mapping({**original, "execution": report})
    assert decoded.status == status
    assert decoded.to_mapping()["output"] == original["output"]
    assert decoded.usage.input_tokens == 9
    assert not decoded.execution.software_environments[0].complete
    if status == "failed":
        assert decoded.error.code == "test_failure"


def test_typed_snapshot_optional_attributes_and_explicit_forwarding_are_consistent():
    from nemo_fabric import ExecutionEvidence, ProvenanceConfig

    snapshot = ExecutionEvidence.from_mapping(
        {
            "software_environments": [
                {
                    "context_ref": "env-0",
                    "participants": ["adapter"],
                    "status": "available",
                    "complete": True,
                    "entries": [
                        {
                            "role": "harness",
                            "kind": "component",
                            "name": "example",
                            "version": {
                                "status": "unavailable",
                                "reason": "no_existing_report",
                                "source": "adapter:startup",
                                "origin": "observed",
                                "observed_at": "2026-10-27T12:00:00Z",
                            },
                        }
                    ],
                }
            ]
        }
    )
    assert snapshot.infrastructure is None
    assert snapshot.software_environments[0].reason is None
    assert snapshot.software_environments[0].entries[0].version.value is None
    assert "infrastructure" not in snapshot.to_mapping()
    assert (
        "value"
        not in snapshot.to_mapping()["software_environments"][0]["entries"][0][
            "version"
        ]
    )
    assert (
        ProvenanceConfig(reports=snapshot).reports.to_mapping() == snapshot.to_mapping()
    )


def test_sdk_evidence_round_trip_without_the_adapter_contract():
    import subprocess
    import sys

    script = """
import sys
from types import SimpleNamespace

def reject_adapter_contract(fullname, path=None, target=None):
    if fullname == "nemo_fabric_adapter_contract" or fullname.startswith("nemo_fabric_adapter_contract."):
        raise ModuleNotFoundError("Adapter contract import is blocked")
    return None

sys.meta_path.insert(0, SimpleNamespace(find_spec=reject_adapter_contract))
from nemo_fabric import ExecutionEvidence, ProvenanceConfig, RunResult
from nemo_fabric_evidence.evidence import ExecutionEvidence as SharedReport

report = SharedReport.gap("caller_sdk", "no_existing_report")
assert isinstance(ProvenanceConfig(reports=report).reports, SharedReport)
assert ProvenanceConfig(reports=report).reports.to_mapping() == report.to_mapping()
snapshot = ExecutionEvidence.from_mapping(report.to_mapping())
assert ProvenanceConfig(reports=snapshot).reports.to_mapping() == report.to_mapping()
result = RunResult.from_mapping({
    "agent_name": "example", "harness": "example", "adapter_kind": "python",
    "runtime_id": "r", "invocation_id": "i", "request_id": "q",
    "status": "succeeded", "output": "ok", "execution": report.to_mapping(),
})
assert result.execution.to_mapping() == report.to_mapping()
assert not any(name.startswith("nemo_fabric_adapter_contract") for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True)
