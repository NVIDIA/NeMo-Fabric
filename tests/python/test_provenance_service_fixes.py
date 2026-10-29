# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retain startup observations across embedded and bounded report boundaries."""

import pytest

from nemo_fabric_evidence.evidence import ExecutionEvidence
from nemo_fabric_evidence.metadata import observation
from nemo_fabric_adapters.common import provenance


def _entry(name):
    return {
        "role": "harness",
        "kind": "executable",
        "name": name,
        "version": observation(
            "1.0", "service:startup", "2026-10-27T12:00:00Z"
        ).to_mapping(),
    }


def _environment(ref="env-0", participant="adapter", entries=()):
    return {
        "context_ref": ref,
        "participants": [participant],
        "status": "available",
        "complete": True,
        "entries": list(entries),
    }


def test_passive_observation_survives_an_unavailable_inventory():
    report = ExecutionEvidence.gap("service", "report_missing").to_mapping()
    provenance.enable_capture(True)
    try:
        provenance.record_version(
            "server", "2.0", "service:startup", participant="service"
        )
        result = provenance.refresh_passive(report)
        decoded = ExecutionEvidence.from_mapping(result)
        assert decoded.software_environments[0].reason == "report_missing"
        observed = decoded.software_environments[1]
        assert observed.participants == ["service"]
        assert observed.entries[0].version.value == "2.0"
        assert not observed.complete
        assert observed.reason == "selected_passive_observations_only"
    finally:
        provenance.enable_capture(False)


@pytest.mark.parametrize("limit", ["entries", "bytes", "contexts"])
def test_passive_refresh_bounds_without_discarding_existing_inventory(limit):
    entries = [
        _entry(f"component-{i}") for i in range(128 if limit == "entries" else 1)
    ]
    environments = [_environment(entries=entries)]
    if limit == "contexts":
        environments.extend(
            _environment(f"env-{i}", f"other-{i}") for i in range(1, 16)
        )
    report = {"software_environments": environments}
    ExecutionEvidence.from_mapping(report)
    provenance.enable_capture(True)
    try:
        provenance.record_version(
            "new-component",
            "x" * (65500 if limit == "bytes" else 1),
            "service:startup",
            participant="service" if limit == "contexts" else "adapter",
        )
        result = ExecutionEvidence.from_mapping(provenance.refresh_passive(report))
        assert result.software_environments[0].entries[0].name == "component-0"
        assert not all(
            env.status == "unavailable" for env in result.software_environments
        )
        assert any(not env.complete for env in result.software_environments)
        assert len(result.software_environments) <= 16
        assert sum(len(env.entries) for env in result.software_environments) <= 128
    finally:
        provenance.enable_capture(False)


@pytest.mark.parametrize("cached", [False, True])
def test_supplied_fabric_owns_guest_observations_only(monkeypatch, tmp_path, cached):
    from nemo_fabric import Fabric, FabricConfig
    from nemo_fabric.runtime_server.state import RuntimeServer
    from nemo_fabric import provenance as sdk_provenance

    calls = []

    def snapshot(participant):
        calls.append(participant)
        return ExecutionEvidence.from_mapping(
            {
                "software_environments": [
                    _environment(participant=participant, entries=[_entry("sdk")])
                ]
            }
        )

    monkeypatch.setattr(sdk_provenance, "sdk_snapshot", snapshot)
    fabric = Fabric()
    config = FabricConfig.from_mapping(
        {"metadata": {"name": "example"}, "harness": {"adapter_id": "test.adapter"}}
    )
    if cached:
        fabric._with_provenance(config)
    original = fabric._software_snapshot
    external = ExecutionEvidence.from_mapping(
        {
            "software_environments": [
                _environment(participant="caller_sdk", entries=[_entry("external")])
            ]
        }
    )
    config.provenance.reports = external
    server = RuntimeServer(directory=tmp_path, fabric=fabric)
    result = server.fabric._with_provenance(config).provenance.reports
    assert result.software_environments[0].participants == ["caller_sdk"]
    assert result.software_environments[1].participants == ["guest_sdk"]
    assert (
        result.software_environments[1].entries[0].version.observed_at
        == "2026-10-27T12:00:00Z"
    )
    assert calls == (["caller_sdk"] if cached else ["guest_sdk"])
    if original is not None:
        assert original.software_environments[0].participants == ["caller_sdk"]
