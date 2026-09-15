# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Health-control tests for the shared Python adapter host."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from nemo_fabric_adapter_contract.models import AdapterHealthRequest
from nemo_fabric_adapters.common.lifecycle import _adapter_health
from nemo_fabric_adapters.common.lifecycle import _close_health_server
from nemo_fabric_adapters.common.lifecycle import _HostState
from nemo_fabric_adapters.common.lifecycle import _start_health_server


class _Runtime:
    async def start(self, payload: dict[str, Any]):
        del payload

    async def invoke(self, request: Any, context: Any):
        del request, context
        raise AssertionError("not used")

    async def stop(self):
        pass


class _SlowHealthRuntime(_Runtime):
    async def health(self, request: AdapterHealthRequest):
        del request
        await asyncio.sleep(1)
        raise AssertionError("health hook should time out")


async def test_python_health_control_reports_busy_without_lifecycle_channel():
    state = _HostState(
        runtime=_Runtime(),
        runtime_id="runtime-1",
        invoking=True,
    )
    output = await _start_health_server(state)
    control = output["health_control"]
    try:
        reader, writer = await asyncio.open_connection(
            control["host"], control["port"]
        )
        writer.write(
            json.dumps(
                {
                    "protocol_version": control["protocol_version"],
                    "token": control["token"],
                    "runtime_id": "runtime-1",
                    "timeout_millis": 1_000,
                }
            ).encode()
            + b"\n"
        )
        await writer.drain()
        response = json.loads(await reader.readline())
        writer.close()
        await writer.wait_closed()
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
    state = _HostState(runtime=_SlowHealthRuntime(), runtime_id="runtime-1")

    result = await _adapter_health(
        state,
        AdapterHealthRequest(runtime_id="runtime-1", timeout_millis=51),
    )

    assert state.failed is False
    assert result.readiness is not None
    assert result.readiness.state.value == "ready"
    assert result.checks[-1].reason_code == "adapter_health_timed_out"
