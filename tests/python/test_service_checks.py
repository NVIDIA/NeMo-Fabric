# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gateway services can be checked before any Fabric runtime exists."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_fabric import (
    CheckReport,
    CheckRequest,
    Fabric,
    FabricConfigError,
    ServiceHandle,
)


@pytest.fixture(name="check_service")
def check_service_fixture():
    return ServiceHandle.from_mapping(
        {
            "service_id": "service-1",
            "service_binding": "test-service-binding",
            "adapter_id": "nvidia.fabric.openclaw",
            "service_type": "openclaw_gateway",
            "ownership": "caller_owned",
        }
    )


@pytest.fixture(name="service_check_report")
def service_check_report_fixture():
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": "fabric.check/v1alpha1",
        "scope": "service",
        "service_id": "service-1",
        "runtime_id": None,
        "requested_level": "ready",
        "outcome": "passed",
        "reason_code": None,
        "completed_at": now,
        "elapsed_ms": 0,
        "server": None,
        "runtime": None,
        "checks": [
            {
                "name": level,
                "outcome": "passed",
                "reason_code": None,
                "source": source,
                "observed_at": now,
            }
            for level, source in [
                ("live", "fabric.adapter.openclaw.gateway"),
                ("active", "fabric.adapter.openclaw.interface"),
                ("ready", "fabric.adapter.openclaw.readiness"),
            ]
        ],
    }


def test_service_check_defaults_without_a_runtime():
    request = CheckRequest(scope="service", service_id="service-1")
    assert request.level == "live"
    assert request.timeout_seconds == 3
    assert request.to_mapping() == {
        "schema_version": "fabric.check/v1alpha1",
        "scope": "service",
        "service_id": "service-1",
        "level": "live",
        "timeout_seconds": 3,
    }


@pytest.mark.parametrize(
    "fields",
    [
        {"scope": "service"},
        {"scope": "service", "service_id": " "},
        {"scope": "service", "service_id": "service\u0000id"},
        {"scope": "service", "service_id": "service-1", "runtime_id": "runtime-1"},
        {"scope": "service", "service_id": "service-1", "runtime_id": None},
        {"scope": "service", "service_id": "service-1", "level": None},
        {"scope": "service", "service_id": "service-1", "level": "operational"},
        {"scope": "runtime", "runtime_id": "runtime-1", "service_id": "service-1"},
        {"scope": "server", "service_id": "service-1"},
    ],
)
def test_service_targets_reject_ambiguous_or_invalid_requests(fields):
    with pytest.raises(ValidationError):
        CheckRequest(**fields)


def test_service_report_requires_no_fabric_runtime(service_check_report):
    report = CheckReport.from_mapping(service_check_report)
    assert report.outcome == "passed"
    assert report.service_id == "service-1"
    assert report.runtime_id is None
    assert report.runtime is None
    assert report.to_mapping() == service_check_report


@pytest.mark.parametrize(
    "changes",
    [
        {"service_id": None},
        {"service_id": " "},
        {"runtime_id": "runtime-1"},
        {"runtime": {"state": "active", "activity": "idle"}},
        {"requested_level": None},
        {"checks": []},
    ],
)
def test_service_reports_reject_invalid_evidence(service_check_report, changes):
    with pytest.raises(FabricConfigError):
        CheckReport.from_mapping({**service_check_report, **changes})


async def test_sdk_checks_a_registered_service_without_an_environment(
    check_service, service_check_report, monkeypatch
):
    native = MagicMock()

    def check(service_json, request_json):
        assert json.loads(service_json) == check_service.to_mapping()
        request = json.loads(request_json)
        assert request["scope"] == "service"
        assert request["service_id"] == "service-1"
        assert 0 < request["timeout_seconds"] <= 3
        return json.dumps(service_check_report)

    native.check_service.side_effect = check
    monkeypatch.setattr("nemo_fabric.client._native", native)

    report = await Fabric().check(
        service=check_service,
        request=CheckRequest(scope="service", service_id="service-1", level="ready"),
    )

    assert report.outcome == "passed"
    native.check_environment.assert_not_called()


async def test_sdk_rejects_a_service_report_for_a_different_gateway(
    check_service, service_check_report, monkeypatch
):
    service_check_report["service_id"] = "another-service"
    native = MagicMock()
    native.check_service.return_value = json.dumps(service_check_report)
    monkeypatch.setattr("nemo_fabric.client._native", native)

    with pytest.raises(FabricConfigError, match="requested target"):
        await Fabric().check(
            service=check_service,
            request=CheckRequest(
                scope="service", service_id="service-1", level="ready"
            ),
        )


async def test_sdk_rejects_a_handle_for_a_different_service(check_service, monkeypatch):
    native = MagicMock()
    monkeypatch.setattr("nemo_fabric.client._native", native)

    with pytest.raises(FabricConfigError):
        await Fabric().check(
            service=check_service,
            request=CheckRequest(scope="service", service_id="another-service"),
        )
    native.check_service.assert_not_called()
