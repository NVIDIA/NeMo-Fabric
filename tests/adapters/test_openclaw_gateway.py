# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the persistent session connection against a WebSocket server."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from websockets.asyncio.server import serve

from nemo_fabric_adapters.common.lifecycle import LifecycleError
from nemo_fabric_adapters.openclaw import gateway


async def test_session_calls_share_one_authenticated_connection(
    unused_tcp_port, caplog
):
    caplog.set_level("DEBUG", logger="websockets.client")
    connections = []
    requests = []

    async def respond(socket):
        connections.append(socket)
        await socket.send(
            json.dumps(
                {
                    "type": "event",
                    "event": "connect.challenge",
                    "payload": {"nonce": "challenge"},
                }
            )
        )
        async for message in socket:
            request = json.loads(message)
            requests.append(request)
            if request["method"] == "connect":
                assert request["params"]["auth"] == {"token": "private-test-token"}
                assert request["params"]["client"]["mode"] == "backend"
                result = {"type": "hello-ok", "protocol": 4}
            else:
                result = {
                    "method": request["method"],
                    "history": "private-test-history",
                }
            await socket.send(json.dumps({"type": "event", "event": "tick"}))
            await socket.send(
                json.dumps(
                    {"type": "res", "id": request["id"], "ok": True, "payload": result}
                )
            )

    async with serve(respond, "127.0.0.1", unused_tcp_port):
        client = gateway.SessionClient(
            f"http://127.0.0.1:{unused_tcp_port}", "private-test-token"
        )
        try:
            for _ in range(7):
                assert await client.call(
                    "sessions.describe", {"key": "selected"}, timeout=2
                ) == {"method": "sessions.describe", "history": "private-test-history"}
        finally:
            await client.close()
        assert "private-test-token" not in caplog.text
        assert "private-test-history" not in caplog.text
        assert len(connections) == 1
        assert len(requests) == 8
        await asyncio.wait_for(connections[0].wait_closed(), 2)


@pytest.mark.parametrize(
    "failure", ["authentication", "rpc", "timeout", "cancel", "disconnect", "invalid"]
)
async def test_session_failures_close_connection_without_replay(
    unused_tcp_port, failure
):
    requests = []
    received = asyncio.Event()

    async def respond(socket):
        await socket.send(
            json.dumps(
                {
                    "type": "event",
                    "event": "connect.challenge",
                    "payload": {"nonce": "challenge"},
                }
            )
        )
        async for message in socket:
            request = json.loads(message)
            requests.append(request["method"])
            if request["method"] == "connect":
                await socket.send(
                    json.dumps(
                        {
                            "type": "res",
                            "id": request["id"],
                            "ok": failure != "authentication",
                            "payload": {"type": "hello-ok", "protocol": 4},
                            "error": {"message": "secret-server-detail"},
                        }
                    )
                )
                continue
            received.set()
            if failure in {"timeout", "cancel"}:
                await socket.wait_closed()
            elif failure == "disconnect":
                await socket.close()
            elif failure == "invalid":
                await socket.send("not-json-secret-server-detail")
            else:
                await socket.send(
                    json.dumps(
                        {
                            "type": "res",
                            "id": request["id"],
                            "ok": False,
                            "error": {"message": "secret-server-detail"},
                        }
                    )
                )

    async with serve(respond, "127.0.0.1", unused_tcp_port):
        client = gateway.SessionClient(
            f"http://127.0.0.1:{unused_tcp_port}", "private-test-token"
        )
        try:
            task = asyncio.create_task(
                client.call(
                    "sessions.abort",
                    {"key": "selected"},
                    timeout=0.1 if failure == "timeout" else 2,
                )
            )
            if failure == "cancel":
                await asyncio.wait_for(received.wait(), 2)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(LifecycleError) as caught:
                    await task
                assert "secret-server-detail" not in str(caught.value)
                assert "private-test-token" not in str(caught.value)
            with pytest.raises(LifecycleError):
                await client.call("sessions.abort", {"key": "selected"}, timeout=2)
            assert requests.count("connect") == 1
            assert requests.count("sessions.abort") <= 1
        finally:
            await client.close()


async def test_runtime_session_checks_do_not_spawn_cli_processes():
    from nemo_fabric_adapters.openclaw import adapter

    runtime = adapter.OpenClawRuntime()
    runtime._session_client = AsyncMock(spec=gateway.SessionClient)
    runtime._session_client.call.return_value = {"session": None}
    try:
        assert await runtime._session_call(
            "sessions.describe", {"key": "selected"}
        ) == {"session": None}
        runtime._session_client.call.assert_awaited_once()
    finally:
        await runtime.stop()


async def test_gateway_redirect_never_receives_authentication(unused_tcp_port_factory):
    original_port = unused_tcp_port_factory()
    redirected_port = unused_tcp_port_factory()
    redirected = []

    async def observe(socket):
        redirected.append(socket)
        await socket.close()

    async def redirect(connection, request):
        response = connection.respond(302, "Moved")
        response.headers["Location"] = f"ws://127.0.0.1:{redirected_port}/"
        return response

    async with serve(observe, "127.0.0.1", redirected_port):
        async with serve(observe, "127.0.0.1", original_port, process_request=redirect):
            client = gateway.SessionClient(
                f"http://127.0.0.1:{original_port}", "private-test-token"
            )
            try:
                with pytest.raises(LifecycleError):
                    await client.call(
                        "sessions.describe", {"key": "selected"}, timeout=2
                    )
                assert redirected == []
            finally:
                await client.close()
