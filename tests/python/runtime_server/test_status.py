# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only runtime discovery through the SDK and the command bridge."""

import asyncio
import json
import sys
from unittest.mock import AsyncMock

import pytest

from nemo_fabric import RuntimeStatus
from nemo_fabric.runtime_server import cli, protocol, transport
from nemo_fabric.runtime_server.state import RuntimeServer


@pytest.mark.parametrize("extra", [[], ["--live"], ["--config", "private.json"]])
def test_status_command_requires_only_agent_and_runtime(extra):
    arguments = ["status", "--agent", "researcher", "--runtime-id", "runtime-1", *extra]
    if extra:
        with pytest.raises(protocol.ProtocolError):
            cli.parse_command(arguments)
    else:
        assert cli.parse_command(arguments) == {
            "interface_version": 2,
            "operation": "status",
            "agent": "researcher",
            "runtime_id": "runtime-1",
        }


@pytest.mark.parametrize("state", [RuntimeStatus.ACTIVE, RuntimeStatus.FAILED])
async def test_status_server_delegates_to_sdk_without_invocation_lock(
    blocking_adapter_config, tmp_path, monkeypatch, state
):
    host = RuntimeServer(directory=tmp_path)
    config = blocking_adapter_config.to_mapping()
    start = await host.handle(
        {
            "interface_version": 2,
            "operation": "start",
            "agent": "blocking",
            "config": config,
        }
    )
    assert start["status"] == "succeeded", start
    runtime_id = start["result"]["runtime_id"]
    owned = host.runtimes[runtime_id]
    owned.runtime._status = state
    sdk_status = AsyncMock(wraps=host.fabric.runtime_status)
    monkeypatch.setattr(host.fabric, "runtime_status", sdk_status)
    try:
        async with owned.lock:
            response = await asyncio.wait_for(
                host.handle(
                    {
                        "interface_version": 2,
                        "operation": "status",
                        "agent": "blocking",
                        "runtime_id": runtime_id,
                    }
                ),
                0.5,
            )
        assert response["status"] == "succeeded", response
        assert response["changed"] is False
        assert response["result"]["status"] == state
        assert response["result"]["server_state"] == "running"
        assert response["result"]["handle"]["runtime_id"] == runtime_id
        sdk_status.assert_awaited_once_with(
            runtime_id=runtime_id, agent_name="blocking"
        )
        assert not (tmp_path / "entered").exists()
        assert not (tmp_path / "stopped").exists()
    finally:
        await host.shutdown()


@pytest.mark.skipif(sys.platform == "win32", reason="Unix command bridge")
async def test_status_process_bridge_reads_stopped_record_and_does_not_start_server(
    blocking_adapter_config, tmp_path, socket_path
):
    host = RuntimeServer(directory=tmp_path)
    ready, stop = asyncio.Event(), asyncio.Event()
    serving = asyncio.create_task(
        transport.serve(
            host,
            socket_path=socket_path,
            ready_event=ready,
            stop_event=stop,
        )
    )
    await asyncio.wait_for(ready.wait(), 5)
    try:
        started = await host.handle(
            {
                "interface_version": 2,
                "operation": "start",
                "agent": "blocking",
                "config": blocking_adapter_config.to_mapping(),
            }
        )
        assert started["status"] == "succeeded", started
        request = {
            "interface_version": 2,
            "operation": "status",
            "agent": "blocking",
            "runtime_id": started["result"]["runtime_id"],
        }
        active = await transport.client(request, socket_path=socket_path)
        assert active["status"] == "succeeded", active
        assert active["result"]["status"] == "active"
        assert (await host.handle({**request, "operation": "stop"}))[
            "status"
        ] == "succeeded"
        stopped = await transport.client(request, socket_path=socket_path)
        assert stopped["result"]["status"] == "stopped"
        assert stopped["result"]["server_state"] == "stopped"
        assert stopped["result"]["handle"] is None
        wrong = await transport.client(
            {**request, "agent": "other"}, socket_path=socket_path
        )
        assert wrong["error"]["code"] == "wrong_agent"
        missing = await transport.client(
            {**request, "runtime_id": "missing"}, socket_path=socket_path
        )
        assert missing["error"]["code"] == "runtime_unavailable"
    finally:
        stop.set()
        await asyncio.wait_for(serving, 5)
    missing_server = await transport.client(request, socket_path=socket_path)
    assert missing_server["error"]["effects"] == "none"
    assert missing_server["changed"] is False
    assert not socket_path.exists()


def test_oversized_status_response_has_no_effects():
    response = protocol.envelope("status", {"private": "x" * protocol.RESPONSE_LIMIT})
    decoded = json.loads(protocol.encode_response(response))
    assert decoded["changed"] is False
    assert decoded["error"]["effects"] == "none"


@pytest.mark.parametrize(
    "field,value",
    [
        ("changed", True),
        ("changed", None),
        ("result", {"status": "active"}),
        (
            "error",
            {
                "code": "runtime_unavailable",
                "stage": "status",
                "message": "PRIVATE",
                "effects": "unknown",
            },
        ),
    ],
)
def test_status_rejects_malformed_or_effectful_responses(field, value):
    response = protocol.envelope(
        "status",
        {
            "runtime_id": "runtime-1",
            "agent_name": "researcher",
            "status": "stopped",
            "handle": None,
            "server_state": "stopped",
        },
    )
    response[field] = value
    if field == "error":
        response["status"] = "failed"
        response["result"] = None
    with pytest.raises(protocol.ProtocolError):
        protocol.validate_response(response, "status")


@pytest.mark.skipif(sys.platform == "win32", reason="Unix socket transport")
@pytest.mark.parametrize(
    "field,value", [("runtime_id", "other"), ("agent_name", "other")]
)
async def test_status_transport_rejects_a_different_target(socket_path, field, value):
    request = {
        "interface_version": 2,
        "operation": "status",
        "agent": "researcher",
        "runtime_id": "runtime-1",
    }

    async def answer(reader, writer):
        await reader.readline()
        report = {
            "runtime_id": "runtime-1",
            "agent_name": "researcher",
            "status": "stopped",
            "handle": None,
            "server_state": "stopped",
            field: value,
        }
        writer.write(protocol.encode_response(protocol.envelope("status", report)))
        await writer.drain()
        writer.close()

    server = await asyncio.start_unix_server(answer, path=socket_path)
    async with server:
        response = await transport.client(request, socket_path=socket_path)
    assert response["status"] == "failed"
    assert response["error"]["code"] == "invalid_response"
    assert response["error"]["effects"] == "none"
    assert response["changed"] is False
