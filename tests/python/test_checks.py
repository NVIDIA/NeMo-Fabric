# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK health request and report validation."""

import asyncio
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_fabric import (
    CheckReport,
    CheckRequest,
    EnvironmentHandle,
    Fabric,
    FabricConfigError,
)


@pytest.fixture(name="check_environment")
def check_environment_fixture():
    return EnvironmentHandle.from_mapping(
        {
            "environment_id": "environment-1",
            "provider": "local",
            "control_location": "external_control",
            "ownership": "caller_owned",
        }
    )


@pytest.fixture(name="check_report")
def check_report_fixture():
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": "fabric.check/v1alpha1",
        "scope": "runtime",
        "runtime_id": "runtime-1",
        "requested_level": "live",
        "outcome": "passed",
        "reason_code": None,
        "completed_at": now,
        "elapsed_ms": 0,
        "server": None,
        "runtime": {"state": "active", "activity": "busy"},
        "checks": [
            {
                "name": "live",
                "outcome": "passed",
                "reason_code": None,
                "source": "fabric.adapter.pi.session",
                "observed_at": now,
            }
        ],
    }


@pytest.mark.parametrize(
    "fields",
    [
        {"scope": "server", "runtime_id": "runtime-1"},
        {"scope": "server", "level": "live"},
        {"scope": "runtime"},
        {"scope": "runtime", "runtime_id": " "},
        {"scope": "runtime", "runtime_id": "runtime-1", "level": "operational"},
        {"scope": "server", "timeout_seconds": True},
        {"scope": "server", "timeout_seconds": 0},
        {"scope": "server", "timeout_seconds": float("nan")},
        {"scope": "server", "timeout_seconds": float("inf")},
        {"scope": "server", "extra": "PRIVATE"},
    ],
)
def test_check_request_rejects_invalid_combinations(fields):
    with pytest.raises(ValidationError):
        CheckRequest(**fields)


def test_request_defaults_and_immutable_report(check_report):
    request = CheckRequest(scope="runtime", runtime_id="runtime-1")
    assert request.level == "live"
    assert request.timeout_seconds == 3
    report = CheckReport.from_mapping(check_report)
    detached = report.to_mapping()
    detached["checks"][0]["source"] = "PRIVATE"
    assert report.to_mapping()["checks"][0]["source"] == "fabric.adapter.pi.session"


def test_report_fields_cannot_be_reassigned_or_deleted(check_report):
    report = CheckReport.from_mapping(check_report)
    with pytest.raises(AttributeError):
        report.outcome = "failed"
    with pytest.raises(AttributeError):
        del report.outcome


async def test_sdk_checks_use_one_client_timer(
    check_environment, check_report, monkeypatch
):
    native = MagicMock()

    def check(environment_json, request_json):
        assert json.loads(environment_json)["environment_id"] == "environment-1"
        assert 0 < json.loads(request_json)["timeout_seconds"] <= 3
        time.sleep(0.03)
        return json.dumps(check_report)

    native.check_environment.side_effect = check
    monkeypatch.setattr("nemo_fabric.client._native", native)
    report = await Fabric().check(
        environment=check_environment,
        request=CheckRequest(scope="runtime", runtime_id="runtime-1"),
    )
    assert isinstance(report, CheckReport)
    assert report.outcome == "passed"
    assert report.elapsed_ms >= 30
    assert report.to_mapping()["checks"] == check_report["checks"]


async def test_expired_request_returns_timeout_without_probing(
    check_environment, monkeypatch
):
    """Use a scoped clock to exercise an expired deadline."""
    native = MagicMock()
    monkeypatch.setattr("nemo_fabric.client._native", native)
    ticks = iter([0.0])
    clock = SimpleNamespace(monotonic=lambda: next(ticks, 1.0))
    monkeypatch.setattr("nemo_fabric.client.time", clock)
    monkeypatch.setattr("nemo_fabric.check.time", clock)
    report = await Fabric().check(
        environment=check_environment,
        request=CheckRequest(
            scope="runtime",
            runtime_id="runtime-1",
            level="live",
            timeout_seconds=0.5,
        ),
    )
    assert report.outcome == "unknown"
    assert report.reason_code == "probe_timeout"
    assert report.checks == []
    native.check_environment.assert_not_called()


async def test_ownership_change_overrides_a_probe_disconnected_during_stop(
    check_environment, check_report, monkeypatch
):
    check_report.update(
        outcome="unknown", reason_code="observation_changed", runtime=None
    )
    check_report["checks"][0].update(
        outcome="unknown",
        reason_code="probe_error",
        source="fabric.adapter",
        observed_at=None,
    )
    native = MagicMock()
    native.check_environment.return_value = json.dumps(check_report)
    monkeypatch.setattr("nemo_fabric.client._native", native)
    report = await Fabric().check(
        environment=check_environment,
        request=CheckRequest(scope="runtime", runtime_id="runtime-1"),
    )
    assert report.outcome == "unknown"
    assert report.reason_code == "observation_changed"
    assert report.runtime is None
    assert report.checks[0]["reason_code"] == "probe_error"


async def test_report_finalization_cannot_pass_after_the_deadline(
    check_environment, check_report, monkeypatch
):
    native = MagicMock()
    native.check_environment.return_value = json.dumps(check_report)
    monkeypatch.setattr("nemo_fabric.client._native", native)
    normalize = CheckReport._normalize

    def delayed(cls, data):
        time.sleep(0.04)
        return normalize(data)

    monkeypatch.setattr(CheckReport, "_normalize", classmethod(delayed))
    started = time.monotonic()
    report = await Fabric().check(
        environment=check_environment,
        request=CheckRequest(
            scope="runtime", runtime_id="runtime-1", timeout_seconds=0.06
        ),
    )
    assert report.outcome != "passed" or time.monotonic() - started < 0.06


async def test_sdk_timeout_does_not_use_invocation_teardown(
    check_environment, check_report, monkeypatch
):
    native = MagicMock()

    def check(*args):
        time.sleep(0.08)
        return json.dumps(check_report)

    native.check_environment.side_effect = check
    monkeypatch.setattr("nemo_fabric.client._native", native)
    report = await asyncio.wait_for(
        Fabric().check(
            environment=check_environment,
            request=CheckRequest(
                scope="runtime", runtime_id="runtime-1", timeout_seconds=0.01
            ),
        ),
        0.05,
    )
    assert report.outcome == "unknown"
    assert report.reason_code == "probe_timeout"
    native.stop_runtime.assert_not_called()
    native.invoke_runtime.assert_not_called()
    await asyncio.sleep(0.09)


async def test_health_does_not_queue_behind_invocation_workers(
    check_environment, check_report, monkeypatch
):
    native = MagicMock()
    native.check_environment.return_value = json.dumps(check_report)
    monkeypatch.setattr("nemo_fabric.client._native", native)
    entered, release = threading.Event(), threading.Event()

    def invocation():
        entered.set()
        assert release.wait(5)

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        loop.set_default_executor(executor)
        busy = loop.run_in_executor(None, invocation)
        try:
            async with asyncio.timeout(1):
                while not entered.is_set():
                    await asyncio.sleep(0.001)
            report = await Fabric().check(
                environment=check_environment,
                request=CheckRequest(
                    scope="runtime", runtime_id="runtime-1", timeout_seconds=0.25
                ),
            )
            assert report.outcome == "passed"
            assert not busy.done()
        finally:
            release.set()
            await busy


@pytest.mark.parametrize(
    "field,value",
    [
        ("extra", "PRIVATE"),
        ("runtime_id", "wrong-target"),
        ("requested_level", "ready"),
        ("elapsed_ms", -1),
    ],
)
async def test_sdk_rejects_invalid_or_mistargeted_reports(
    check_environment, check_report, monkeypatch, field, value
):
    native = MagicMock()
    check_report[field] = value
    native.check_environment.return_value = json.dumps(check_report)
    monkeypatch.setattr("nemo_fabric.client._native", native)
    with pytest.raises((FabricConfigError, ValueError)):
        await Fabric().check(
            environment=check_environment,
            request=CheckRequest(scope="runtime", runtime_id="runtime-1"),
        )
