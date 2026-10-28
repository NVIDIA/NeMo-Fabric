# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional software reports preserve availability, full versions and wire bounds."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from nemo_fabric_evidence.evidence import ExecutionEvidence
from nemo_fabric_evidence.evidence import VersionObservation

FIXTURE = Path(__file__).parents[1] / "fixtures/provenance/report.json"


def test_selected_report_matches_rust_schema_and_round_trips():
    report = json.loads(FIXTURE.read_text())
    schema = json.loads(
        (
            Path(__file__).parents[2]
            / "schemas/adapter-contract/execution-evidence.schema.json"
        ).read_text()
    )
    Draft202012Validator(schema).validate(report)
    assert ExecutionEvidence.from_mapping(report).to_mapping() == report
    assert (
        len(report["software_environments"][0]["entries"][0]["version"]["value"]) > 256
    )


@pytest.mark.parametrize(
    "patch",
    [
        {"value": None},
        {"reason": "guessed"},
        {"observed_at": "2026-02-30T12:00:00Z"},
        {"origin": "configured"},
        {"observed_at": "today"},
        {"status": "unavailable"},
    ],
)
def test_invalid_observation_is_rejected(patch):
    value = {
        "status": "available",
        "value": "1.0+build",
        "source": "adapter:metadata",
        "origin": "observed",
        "observed_at": "2026-10-27T12:00:00Z",
    }
    with pytest.raises(ValueError):
        VersionObservation.from_mapping({**value, **patch})


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"software_environments": "broken"},
        {"software_environments": [], "extra": True},
        "x" * 70000,
    ],
    ids=["missing", "empty", "malformed-inventory", "extra-field", "oversized-report"],
)
def test_receiving_optional_malformed_evidence_preserves_explicit_gap(value):
    gap = ExecutionEvidence.receive(value, "guest_sdk")
    assert gap.software_environments[0].complete is False
    assert gap.software_environments[0].participants == ["guest_sdk"]
    assert gap.software_environments[0].reason


def test_invalid_context_links_and_over_budget_entries_are_gaps():
    report = json.loads(FIXTURE.read_text())
    report["software_environments"] *= 17
    assert (
        not ExecutionEvidence.receive(report, "adapter")
        .software_environments[0]
        .complete
    )


def test_schema_rejects_availability_contradictions_and_accepts_utc_offset():
    report = json.loads(FIXTURE.read_text())
    schema = json.loads(
        (
            Path(__file__).parents[2]
            / "schemas/adapter-contract/execution-evidence.schema.json"
        ).read_text()
    )
    validator = Draft202012Validator(schema)
    report["software_environments"][0]["entries"][0]["version"]["reason"] = (
        "contradictory"
    )
    assert not validator.is_valid(report)
    del report["software_environments"][0]["entries"][0]["version"]["reason"]
    report["software_environments"][0]["entries"][0]["version"]["observed_at"] = (
        "2026-10-27T12:00:00+00:00"
    )
    assert validator.is_valid(report)
    assert ExecutionEvidence.from_mapping(report).to_mapping() == report


@pytest.mark.parametrize(
    "path",
    [
        ("context_ref",),
        ("participants", 0),
        ("entries", 0, "role"),
        ("entries", 0, "kind"),
        ("entries", 0, "name"),
        ("entries", 0, "version", "source"),
    ],
)
def test_schema_and_python_reject_blank_inventory_identity(path):
    report = json.loads(FIXTURE.read_text())
    destination = report["software_environments"][0]
    for key in path[:-1]:
        destination = destination[key]
    destination[path[-1]] = "   "
    schema = json.loads(
        (
            Path(__file__).parents[2]
            / "schemas/adapter-contract/execution-evidence.schema.json"
        ).read_text()
    )
    assert not Draft202012Validator(schema).is_valid(report)
    with pytest.raises(ValueError):
        ExecutionEvidence.from_mapping(report)


def test_schema_requires_participants_and_infrastructure_context_references():
    report = json.loads(FIXTURE.read_text())
    schema = json.loads(
        (
            Path(__file__).parents[2]
            / "schemas/adapter-contract/execution-evidence.schema.json"
        ).read_text()
    )
    validator = Draft202012Validator(schema)
    observation = report["software_environments"][0]["entries"][0]["version"]
    report["infrastructure"] = {
        "sandbox": {
            "context_refs": [],
            "provider": observation,
            "service_version": observation,
        }
    }
    assert not validator.is_valid(report)
    report["infrastructure"]["sandbox"]["context_refs"] = ["env-0"]
    assert validator.is_valid(report)
    report["software_environments"][0]["participants"] = []
    assert not validator.is_valid(report)


def test_adapter_evidence_and_validation_reexports_preserve_type_identity():
    from nemo_fabric_evidence.codec import ContractValidationError
    from nemo_fabric_evidence.models import ContractModel
    from nemo_fabric_adapter_contract import evidence
    from nemo_fabric_adapter_contract.codec import (
        ContractValidationError as LegacyError,
    )
    from nemo_fabric_adapter_contract.models import ContractModel as LegacyModel

    assert evidence.ExecutionEvidence is ExecutionEvidence
    assert evidence.VersionObservation is VersionObservation
    assert LegacyError is ContractValidationError
    assert LegacyModel is ContractModel
    report = evidence.ExecutionEvidence.from_mapping(json.loads(FIXTURE.read_text()))
    assert isinstance(report, ExecutionEvidence)
    assert isinstance(report, LegacyModel)
