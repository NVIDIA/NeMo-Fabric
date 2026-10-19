# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server health is fresh, bounded, independent of agents, and read only."""

import asyncio
import copy
import json
import math
import sys
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

if sys.platform == "win32":
    pytest.skip(
        "The Unix command bridge requires a Unix platform", allow_module_level=True
    )

from nemo_fabric.runtime_server import cli, protocol, transport


def test_server_check_command_defaults_and_timeout():
    assert cli.parse_command(["check"]) == {
        "interface_version": 2,
        "operation": "check",
    }
    assert cli.parse_command(["check", "--timeout-seconds", "0.25"]) == {
        "interface_version": 2,
        "operation": "check",
        "timeout_seconds": 0.25,
    }


@pytest.mark.parametrize(
    "flags",
    [
        ["--agent", "researcher"],
        ["--runtime-id", "r"],
        ["--agent", "researcher", "--runtime-id", "r"],
        ["--timeout-seconds", "0"],
        ["--timeout-seconds", "-1"],
        ["--timeout-seconds", "nan"],
        ["--timeout-seconds", "inf"],
        ["--timeout-seconds", "1e309"],
        ["--timeout-seconds", "PRIVATE"],
        ["--timeout-seconds", "1", "--timeout-seconds", "2"],
    ],
)
def test_invalid_server_check_does_not_connect(flags, monkeypatch, capsys):
    mock_client = AsyncMock()
    monkeypatch.setattr(transport, "client", mock_client)
    assert cli.main(["check", *flags]) == 1
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    response = json.loads(captured.out)
    assert response["operation"] == "check"
    assert response["changed"] is False
    assert response["error"]["code"] == "invalid_request"
    assert response["error"]["effects"] == "none"
    assert "PRIVATE" not in captured.out
    mock_client.assert_not_awaited()


@pytest.mark.parametrize(
    "fields",
    [
        {"runtime_id": "r"},
        {"level": "ready"},
        {"agent": "researcher"},
        {"surprise": "PRIVATE"},
        *(
            {"timeout_seconds": value}
            for value in (0, -1, True, "3", None, math.inf, math.nan, 10**400)
        ),
    ],
)
async def test_invalid_wire_check_has_no_effects(
    host, host_request, mock_fabric, fields
):
    response = await host.handle(host_request("check", **fields))
    assert response["error"]["code"] == "invalid_request"
    assert response["changed"] is False
    assert response["error"]["effects"] == "none"
    mock_fabric.start_runtime.assert_not_awaited()


@pytest.mark.parametrize("stopping", [False, True])
async def test_check_reports_current_server_state_without_a_runtime(
    host, host_request, mock_fabric, stopping
):
    host.shutting_down = stopping
    before = datetime.now(timezone.utc)
    response = await host.handle(host_request("check"))
    after = datetime.now(timezone.utc)
    assert response["operation"] == "check"
    assert response["status"] == ("failed" if stopping else "succeeded")
    assert response["changed"] is False
    report = response["result"]
    assert set(report) == {
        "schema_version",
        "scope",
        "runtime_id",
        "requested_level",
        "outcome",
        "reason_code",
        "completed_at",
        "elapsed_ms",
        "server",
        "runtime",
        "checks",
    }
    assert report["schema_version"] == "fabric.check/v1alpha1"
    assert report["scope"] == "server"
    assert (
        report["runtime_id"] is report["requested_level"] is report["runtime"] is None
    )
    assert report["outcome"] == ("failed" if stopping else "passed")
    reason = "server_stopping" if stopping else None
    assert report["reason_code"] == reason
    assert report["server"] == {"state": "stopping" if stopping else "responding"}
    assert report["elapsed_ms"] >= 0
    assert before <= datetime.fromisoformat(report["completed_at"]) <= after
    assert len(report["checks"]) == 1
    observation = report["checks"][0]
    assert set(observation) == {
        "name",
        "outcome",
        "reason_code",
        "source",
        "observed_at",
    }
    assert observation["name"] == "server"
    assert observation["outcome"] == report["outcome"]
    assert observation["reason_code"] == reason
    assert observation["source"] == "fabric.runtime_server"
    assert before <= datetime.fromisoformat(observation["observed_at"]) <= after
    protocol.validate_response(response, "check")
    assert host.runtimes == {}
    assert host.has_used_runtime is False
    assert host.start_uncertain is False
    mock_fabric.start_runtime.assert_not_awaited()


async def test_server_check_does_not_wait_for_lifecycle_locks(
    host, host_request, mock_runtime
):
    await host.handle(host_request("start"))
    async with host._start_lock, host.runtimes["runtime-1"].lock:
        response = await asyncio.wait_for(host.handle(host_request("check")), 0.1)
    assert response["result"]["outcome"] == "passed"
    mock_runtime.invoke.assert_not_awaited()
    mock_runtime.stop.assert_not_awaited()


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "wrong"),
        ("scope", "runtime"),
        ("runtime_id", "PRIVATE"),
        ("requested_level", "ready"),
        ("runtime", {}),
        ("outcome", "failed"),
        ("reason_code", "PRIVATE"),
        ("elapsed_ms", -1),
        ("elapsed_ms", math.inf),
        ("elapsed_ms", 10**400),
        ("elapsed_ms", True),
        ("completed_at", "yesterday"),
        ("completed_at", "2026-10-19T12:00:00"),
        ("server", {"state": "unknown"}),
        ("checks", []),
        ("extra", "PRIVATE"),
    ],
)
async def test_malformed_server_report_is_rejected(host, host_request, field, value):
    response = await host.handle(host_request("check"))
    response["result"][field] = value
    with pytest.raises(protocol.ProtocolError) as error:
        protocol.validate_response(response, "check")
    assert error.value.code == "invalid_response"


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "ready"),
        ("outcome", "unknown"),
        ("reason_code", "PRIVATE"),
        ("source", "PRIVATE"),
        ("observed_at", None),
        ("extra", "PRIVATE"),
    ],
)
async def test_malformed_server_observation_is_rejected(
    host, host_request, field, value
):
    response = await host.handle(host_request("check"))
    response["result"]["checks"][0][field] = value
    with pytest.raises(protocol.ProtocolError):
        protocol.validate_response(response, "check")


@pytest.mark.parametrize("stopping", [False, True])
async def test_check_cli_preserves_complete_report(
    host, host_request, monkeypatch, capsys, stopping
):
    host.shutting_down = stopping
    response = await host.handle(host_request("check"))
    # main owns an event loop, so call main from a worker thread.
    monkeypatch.setattr(
        transport, "client", AsyncMock(return_value=copy.deepcopy(response))
    )
    assert await asyncio.to_thread(cli.main, ["check"]) == (1 if stopping else 0)
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    assert json.loads(captured.out) == response


@pytest.mark.parametrize("stopping", [False, True])
async def test_check_report_timing_covers_the_full_client_call(
    host, host_request, socket_path, monkeypatch, stopping
):
    host.shutting_down = stopping
    observations = []
    validated_at = []
    connect = asyncio.open_unix_connection
    validate = transport.validate_response

    async def delayed_connect(*args, **kwargs):
        await asyncio.sleep(0.02)
        return await connect(*args, **kwargs)

    def delayed_validation(*args):
        validate(*args)
        time.sleep(0.02)
        validated_at.append(datetime.now(timezone.utc))

    async def peer(reader, writer):
        await reader.readline()
        await asyncio.sleep(0.02)
        response = await host.handle(host_request("check"))
        observations.extend(copy.deepcopy(response["result"]["checks"]))
        await asyncio.sleep(0.02)
        writer.write(protocol.encode_response(response))
        await writer.drain()
        writer.close()

    monkeypatch.setattr(asyncio, "open_unix_connection", delayed_connect)
    monkeypatch.setattr(transport, "validate_response", delayed_validation)
    server = await asyncio.start_unix_server(peer, path=socket_path)
    async with server:
        started = time.monotonic()
        response = await transport.client(
            host_request("check"), socket_path=socket_path
        )
    elapsed_ms = (time.monotonic() - started) * 1000
    completed_at = datetime.now(timezone.utc)
    report = response["result"]
    assert report["outcome"] == ("failed" if stopping else "passed")
    assert 80 <= report["elapsed_ms"] <= elapsed_ms
    assert (
        validated_at[0]
        <= datetime.fromisoformat(report["completed_at"])
        <= completed_at
    )
    assert report["checks"] == observations
    protocol.validate_response(response, "check")


async def test_unreachable_server_check_has_no_unknown_invocation_effects(
    socket_path, host_request
):
    response = await transport.client(host_request("check"), socket_path=socket_path)
    assert response["operation"] == "check"
    assert response["status"] == "failed"
    assert response["result"] is None
    assert response["changed"] is False
    assert response["error"]["effects"] == "none"
    assert response["error"]["code"] == "host_unavailable"


def test_oversized_check_response_cannot_claim_unknown_invocation_effects():
    response = protocol.envelope("check", {"private": "x" * protocol.RESPONSE_LIMIT})
    result = json.loads(protocol.encode_response(response))
    assert result["status"] == "failed"
    assert result["changed"] is False
    assert result["result"] is None
    assert result["error"]["code"] == "response_too_large"
    assert result["error"]["effects"] == "none"


@pytest.mark.parametrize("delay_stage", ["connect", "decode"])
async def test_check_deadline_covers_connection_and_response_validation(
    host, host_request, monkeypatch, delay_stage
):
    response = await host.handle(host_request("check"))
    mock_reader = AsyncMock(spec=asyncio.StreamReader)
    mock_reader.readline.return_value = protocol.encode_response(response)
    mock_writer = MagicMock(spec=asyncio.StreamWriter)
    mock_writer.drain = AsyncMock()

    async def connect(*args, **kwargs):
        if delay_stage == "connect":
            await asyncio.sleep(0.03)
        return mock_reader, mock_writer

    async def read_line():
        await asyncio.sleep(0.03)
        return protocol.encode_response(response)

    def validate(*args):
        time.sleep(0.06)

    monkeypatch.setattr(asyncio, "open_unix_connection", connect)
    if delay_stage == "connect":
        mock_reader.readline.side_effect = read_line
    else:
        monkeypatch.setattr(transport, "validate_response", validate)
    result = await transport.client(host_request("check", timeout_seconds=0.05))
    assert result["result"]["outcome"] == "unknown"
    assert result["result"]["reason_code"] == "probe_timeout"
    mock_writer.close.assert_called_once()


async def test_check_default_deadline_is_three_seconds(host_request, monkeypatch):
    assert transport.CHECK_SECONDS == 3
    monkeypatch.setattr(transport, "CHECK_SECONDS", 0.01)

    async def connect(*args, **kwargs):
        await asyncio.sleep(1)

    monkeypatch.setattr(asyncio, "open_unix_connection", connect)
    response = await asyncio.wait_for(transport.client(host_request("check")), 0.5)
    assert response["result"]["reason_code"] == "probe_timeout"


async def test_check_timeout_uses_request_budget_without_cancelling_server_work(
    host, host_request, socket_path
):
    received, release, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []

    async def peer(reader, writer):
        calls.append(await reader.readline())
        received.set()
        try:
            await release.wait()
            response = await host.handle(host_request("check"))
            writer.write(protocol.encode_response(response))
            await writer.drain()
        finally:
            writer.close()
            completed.set()

    server = await asyncio.start_unix_server(peer, path=socket_path)
    async with server:
        try:
            response = await asyncio.wait_for(
                transport.client(
                    host_request("check", timeout_seconds=0.03), socket_path=socket_path
                ),
                0.5,
            )
            assert received.is_set()
            assert response["status"] == "failed"
            assert response["changed"] is False
            assert response["result"]["outcome"] == "unknown"
            assert response["result"]["reason_code"] == "probe_timeout"
            assert response["result"]["elapsed_ms"] >= 30
            assert response["result"]["server"] == {"state": "unknown"}
            assert response["error"]["effects"] == "none"
            assert len(calls) == 1
            protocol.validate_response(response, "check")
        finally:
            release.set()
            await asyncio.wait_for(completed.wait(), 1)


@pytest.mark.parametrize(
    "reply", [b"", b"{}\n", b'{"x":1,"x":2}\n', b'{"operation":"check"}']
)
async def test_unreadable_check_response_is_a_safe_transport_failure(
    socket_path, host_request, reply
):
    async def peer(reader, writer):
        await reader.readline()
        writer.write(reply)
        await writer.drain()
        writer.close()

    server = await asyncio.start_unix_server(peer, path=socket_path)
    async with server:
        response = await transport.client(
            host_request("check"), socket_path=socket_path
        )
    assert response["status"] == "failed"
    assert response["result"] is None
    assert response["changed"] is False
    assert response["error"]["code"] == "invalid_response"
    assert response["error"]["effects"] == "none"
