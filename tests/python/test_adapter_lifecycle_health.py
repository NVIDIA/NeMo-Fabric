# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Health-control tests for the shared Python adapter host."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from nemo_fabric_adapter_contract.models import AdapterHealthRequest
from nemo_fabric_adapter_contract.models import AdapterHealthResult
from nemo_fabric_adapter_contract.models import AdapterReadiness
from nemo_fabric_adapter_contract.models import RuntimeReadiness
from nemo_fabric_adapters.common.lifecycle import _adapter_call
from nemo_fabric_adapters.common.lifecycle import _adapter_health
from nemo_fabric_adapters.common.lifecycle import _close_health_server
from nemo_fabric_adapters.common.lifecycle import _handle_start
from nemo_fabric_adapters.common.lifecycle import _handle_stop
from nemo_fabric_adapters.common.lifecycle import _HostState
from nemo_fabric_adapters.common.lifecycle import _start_health_server


def _runtime_mock(*, health: AsyncMock | None = None) -> MagicMock:
    methods = ["start", "invoke", "stop"]
    if health is not None:
        methods.append("health")
    runtime = MagicMock(spec_set=methods)
    runtime.start = AsyncMock()
    runtime.invoke = AsyncMock(side_effect=AssertionError("not used"))
    runtime.stop = AsyncMock()
    if health is not None:
        runtime.health = health
    return runtime


async def _check_health(control: dict[str, Any], runtime_id: str) -> dict[str, Any]:
    reader, writer = await asyncio.open_connection(control["host"], control["port"])
    writer.write(
        json.dumps(
            {
                "protocol_version": control["protocol_version"],
                "token": control["token"],
                "runtime_id": runtime_id,
                "timeout_millis": 1_000,
            }
        ).encode()
        + b"\n"
    )
    await writer.drain()
    response = json.loads(await reader.readline())
    writer.close()
    await writer.wait_closed()
    return response


async def test_python_health_control_reports_busy_without_lifecycle_channel():
    state = _HostState(
        runtime=_runtime_mock(),
        runtime_id="runtime-1",
        invoking=True,
    )
    output = await _start_health_server(state)
    control = output["health_control"]
    try:
        response = await _check_health(control, "runtime-1")
    finally:
        await _close_health_server(state)

    assert response["runtime_id"] == "runtime-1"
    assert response["result"]["readiness"] == {
        "state": "not_ready",
        "reason_code": "invocation_in_progress",
    }
    assert [check["status"] for check in response["result"]["checks"]] == [
        "unsupported",
        "unsupported",
    ]


async def test_python_adapter_health_hook_timeout_is_data():
    async def slow_health(request: AdapterHealthRequest):
        del request
        await asyncio.sleep(1)
        raise AssertionError("health hook should time out")

    state = _HostState(
        runtime=_runtime_mock(health=AsyncMock(side_effect=slow_health)),
        runtime_id="runtime-1",
    )

    result = await _adapter_health(
        state,
        AdapterHealthRequest(runtime_id="runtime-1", timeout_millis=51),
    )

    assert state.failed is False
    assert result.readiness is not None
    assert result.readiness.state.value == "ready"
    assert result.checks[-1].reason_code == "adapter_health_timed_out"


async def test_python_adapter_health_preserves_busy_runtime_readiness():
    readiness = AdapterHealthResult(
        readiness=AdapterReadiness(
            state=RuntimeReadiness.READY,
            reason_code="concurrent_invocations_supported",
        )
    )
    state = _HostState(
        runtime=_runtime_mock(health=AsyncMock(return_value=readiness)),
        runtime_id="runtime-1",
        invoking=True,
    )

    result = await _adapter_health(
        state,
        AdapterHealthRequest(runtime_id="runtime-1", timeout_millis=1_000),
    )

    assert result.readiness is not None
    assert result.readiness.state is RuntimeReadiness.READY
    assert result.readiness.reason_code == "concurrent_invocations_supported"


async def test_python_health_reports_stop_in_progress():
    stop_started = asyncio.Event()
    release_stop = asyncio.Event()

    async def blocking_stop():
        stop_started.set()
        await release_stop.wait()

    runtime = _runtime_mock()
    runtime.stop.side_effect = blocking_stop
    state = _HostState(runtime=runtime, runtime_id="runtime-1")
    control = (await _start_health_server(state))["health_control"]
    stopping = asyncio.create_task(_handle_stop(state, runtime))
    await stop_started.wait()

    try:
        response = await _check_health(control, "runtime-1")

        assert response["result"]["readiness"] == {
            "state": "not_ready",
            "reason_code": "stop_in_progress",
        }
    finally:
        release_stop.set()
        await stopping


async def test_python_host_skips_health_server_when_capability_is_disabled():
    state = _HostState()
    runtime = _runtime_mock()

    response = await _handle_start(
        state,
        lambda: runtime,
        {"health_enabled": False},
        "runtime-1",
        None,
    )

    assert response["outcome"]["output"] is None
    assert state.health_server is None
    assert state.runtime is not None
    await _handle_stop(state, state.runtime)


async def test_python_close_health_server_drains_connections_before_wait_closed():
    events: list[str] = []
    connection_started = asyncio.Event()
    release_connection = asyncio.Event()

    async def connection():
        connection_started.set()
        try:
            await release_connection.wait()
        finally:
            events.append("connection_closed")

    task = asyncio.create_task(connection())
    await connection_started.wait()
    server = MagicMock()
    server.close.side_effect = lambda: events.append("server_closed")

    async def wait_closed():
        assert task.done()
        events.append("server_waited")

    server.wait_closed = AsyncMock(side_effect=wait_closed)
    state = _HostState(health_server=server, health_token="secret")
    state.health_tasks.add(task)

    await _close_health_server(state)

    assert events == ["server_closed", "connection_closed", "server_waited"]
    assert state.health_server is None
    assert state.health_token is None
    assert not state.health_tasks


async def test_adapter_calls_do_not_rebind_stdout_when_they_overlap():
    original_stdout = sys.stdout
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release_first = asyncio.Event()
    release_second = asyncio.Event()

    async def first_call():
        first_started.set()
        await release_first.wait()

    async def second_call():
        second_started.set()
        await release_second.wait()

    try:
        first = asyncio.create_task(_adapter_call("first", first_call))
        await first_started.wait()
        second = asyncio.create_task(_adapter_call("second", second_call))
        await second_started.wait()
        release_first.set()
        await first
        release_second.set()
        await second

        assert sys.stdout is original_stdout
    finally:
        sys.stdout = original_stdout
