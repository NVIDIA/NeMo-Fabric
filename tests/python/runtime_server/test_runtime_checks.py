# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-targeted health checks through the SDK."""

import asyncio
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_fabric import CheckReport, EnvironmentHandle
from nemo_fabric.runtime_server import cli, protocol, transport
from nemo_fabric.runtime_server.health import check_response

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Unix command bridge")


@pytest.fixture(name="runtime_check_report")
def runtime_check_report_fixture():
    now = datetime.now(timezone.utc).isoformat()
    return CheckReport.from_mapping(
        {
            "schema_version": "fabric.check/v1alpha1",
            "scope": "runtime",
            "runtime_id": "runtime-1",
            "requested_level": "ready",
            "outcome": "passed",
            "reason_code": None,
            "completed_at": now,
            "elapsed_ms": 0,
            "server": None,
            "runtime": {"state": "active", "activity": "busy"},
            "checks": [
                {
                    "name": level,
                    "outcome": "passed",
                    "reason_code": None,
                    "source": source,
                    "observed_at": now,
                }
                for level, source in [
                    ("live", "fabric.adapter.pi.session"),
                    ("active", "fabric.adapter.pi.interface"),
                    ("ready", "fabric.adapter.pi.setup"),
                ]
            ],
        }
    )


@pytest.mark.parametrize("level", ["live", "active", "ready"])
def test_runtime_check_flags_map_to_sdk_levels(level):
    request = cli.parse_command(
        ["check", "--agent", "researcher", "--runtime-id", "runtime-1", f"--{level}"]
    )
    assert request["level"] == level
    assert request["agent"] == "researcher"
    assert request["runtime_id"] == "runtime-1"


async def test_busy_runtime_check_dispatches_without_its_invocation_lock(
    host, host_request, mock_fabric, runtime_check_report
):
    await host.handle(host_request("start"))
    mock_fabric.check = AsyncMock(return_value=runtime_check_report)
    owned = host.runtimes["runtime-1"]
    async with owned.lock:
        response = await asyncio.wait_for(
            host.handle(
                host_request(
                    "check", runtime_id="runtime-1", level="ready", timeout_seconds=0.5
                )
            ),
            0.2,
        )
    assert response["status"] == "succeeded"
    assert response["changed"] is False
    assert response["result"]["runtime"]["activity"] == "busy"
    request = mock_fabric.check.await_args.kwargs["request"]
    assert isinstance(
        mock_fabric.check.await_args.kwargs["environment"], EnvironmentHandle
    )
    assert request.level == "ready"
    assert request.runtime_id == "runtime-1"
    assert 0 < request.timeout_seconds <= 0.5
    protocol.validate_response(response, "check")


async def test_missing_runtime_does_not_dispatch(host, host_request, mock_fabric):
    mock_fabric.check = AsyncMock()
    missing = await host.handle(host_request("check", runtime_id="missing"))
    assert missing["result"]["reason_code"] == "runtime_unavailable"
    mock_fabric.check.assert_not_awaited()


def test_removed_operational_level_is_rejected_by_cli_and_protocol(host_request):
    with pytest.raises(protocol.ProtocolError):
        cli.parse_command(
            [
                "check",
                "--agent",
                "researcher",
                "--runtime-id",
                "runtime-1",
                "--operational",
            ]
        )
    with pytest.raises(protocol.ProtocolError):
        protocol.validate_request(
            host_request("check", runtime_id="runtime-1", level="operational")
        )


async def test_runtime_check_deadline_does_not_stop_the_runtime(
    host, host_request, mock_fabric, mock_runtime, runtime_check_report
):
    await host.handle(host_request("start"))

    async def check(**kwargs):
        await asyncio.sleep(1)
        return runtime_check_report

    mock_fabric.check = AsyncMock(side_effect=check)
    response = await asyncio.wait_for(
        host.handle(
            host_request(
                "check", runtime_id="runtime-1", level="ready", timeout_seconds=0.01
            )
        ),
        0.1,
    )
    assert response["result"]["reason_code"] == "probe_timeout"
    mock_runtime.stop.assert_not_awaited()


async def test_runtime_transport_timeout_preserves_the_runtime_target(
    host_request, monkeypatch
):
    monkeypatch.setattr(
        asyncio, "open_unix_connection", AsyncMock(side_effect=TimeoutError)
    )
    response = await transport.client(
        host_request("check", runtime_id="runtime-1", level="ready")
    )
    assert response["result"]["scope"] == "runtime"
    assert response["result"]["runtime_id"] == "runtime-1"
    assert response["result"]["requested_level"] == "ready"
    assert response["result"]["reason_code"] == "probe_timeout"
    protocol.validate_response(response, "check")


async def test_runtime_transport_rejects_another_targets_evidence(
    host_request, runtime_check_report, monkeypatch
):
    report = runtime_check_report.to_mapping()
    report["runtime_id"] = "other-runtime"
    reader = MagicMock(spec=asyncio.StreamReader)
    reader.readline = AsyncMock(
        return_value=protocol.encode_response(check_response(report))
    )
    writer = MagicMock(spec=asyncio.StreamWriter)
    writer.drain = AsyncMock()
    monkeypatch.setattr(
        asyncio, "open_unix_connection", AsyncMock(return_value=(reader, writer))
    )
    response = await transport.client(
        host_request("check", runtime_id="runtime-1", level="ready")
    )
    assert response["result"] is None
    assert response["error"]["code"] == "invalid_response"


async def test_stop_invalidates_runtime_health_in_flight(
    host, host_request, mock_fabric, runtime_check_report
):
    await host.handle(host_request("start"))
    entered, release = asyncio.Event(), asyncio.Event()

    async def check(**kwargs):
        entered.set()
        await release.wait()
        return runtime_check_report

    mock_fabric.check = AsyncMock(side_effect=check)
    checking = asyncio.create_task(
        host.handle(host_request("check", runtime_id="runtime-1", level="ready"))
    )
    try:
        await asyncio.wait_for(entered.wait(), 0.5)
        assert (await host.handle(host_request("stop")))["status"] == "succeeded"
        release.set()
        result = await checking
        assert result["result"]["outcome"] == "unknown"
        assert result["result"]["reason_code"] == "observation_changed"
        assert result["result"]["runtime"] is None
    finally:
        release.set()
        await checking
