# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for Relay-backed SDK streaming."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from nemo_fabric import (
    Fabric,
    FabricCapabilityError,
    FabricConfig,
    FabricStateError,
    HarnessConfig,
    InvokeStream,
    MetadataConfig,
    RelayAtofConfig,
    RelayAtofEndpointConfig,
    RelayObservabilityConfig,
    RunResult,
)
from nemo_fabric import client as client_mod
from nemo_fabric.streaming import _AtofStreamListener


def _config(*, relay: bool = False) -> FabricConfig:
    config = FabricConfig(
        metadata=MetadataConfig(name="demo"),
        harness=HarnessConfig(adapter_id="test.fabric.shim"),
    )
    if relay:
        config.enable_relay(
            observability=RelayObservabilityConfig(
                atof=RelayAtofConfig(
                    endpoints=[
                        RelayAtofEndpointConfig(url="https://example.com/events")
                    ]
                )
            )
        )
    return config


def _plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "agent_name": "demo",
        "base_dir": ".",
        "config": config,
        "adapter_descriptor": {
            "descriptor": {
                "adapter_id": "test.fabric.shim",
                "harness": "hermes",
                "adapter_kind": "python",
            }
        },
        "capabilities": {
            "service": False,
            "streaming": False,
            "updates": False,
            "cancellation": False,
        },
    }


def _runtime() -> dict[str, Any]:
    return {
        "runtime_id": "runtime-1",
        "runtime_binding": "fabric-runtime-binding-test",
        "agent_name": "demo",
        "harness": "hermes",
        "adapter_kind": "python",
        "adapter_id": "test.fabric.shim",
        "environment": {
            "environment_id": "environment-1",
            "provider": "local",
            "control_location": "external_control",
            "ownership": "caller_owned",
        },
    }


def _result(request: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "agent_name": "demo",
        "harness": "hermes",
        "adapter_kind": "python",
        "adapter_id": "test.fabric.shim",
        "runtime_id": runtime["runtime_id"],
        "invocation_id": f"invocation-{request['request_id']}",
        "request_id": request["request_id"],
        "status": "succeeded",
        "output": {"response": "done"},
        "artifacts": {"artifacts": []},
        "events": [],
    }


@pytest.fixture(name="mock_native")
def mock_native_fixture() -> MagicMock:
    mock_native = MagicMock()
    mock_native.plan_config.side_effect = lambda config_json, base_dir: json.dumps(
        _plan(json.loads(config_json))
    )
    mock_native.start_runtime.return_value = json.dumps(_runtime())
    mock_native.invoke_runtime.side_effect = (
        lambda plan_json, runtime_json, request_json: json.dumps(
            _result(json.loads(request_json), json.loads(runtime_json))
        )
    )
    mock_native.stop_runtime.return_value = "[]"
    return mock_native


@pytest.fixture(name="native_client")
def native_client_fixture(
    monkeypatch: pytest.MonkeyPatch,
    mock_native: MagicMock,
) -> Fabric:
    monkeypatch.setattr(client_mod, "_native", mock_native)
    return Fabric()


async def _post_chunked(url: str, records: list[dict[str, Any]]) -> None:
    host_port = url.removeprefix("http://").split("/", 1)[0]
    host, port = host_port.split(":", 1)
    reader, writer = await asyncio.open_connection(host, int(port))
    writer.write(
        b"POST /atof HTTP/1.1\r\n"
        + f"Host: {host_port}\r\n".encode()
        + b"Transfer-Encoding: chunked\r\n"
        + b"Content-Type: application/x-ndjson\r\n\r\n"
    )
    for record in records:
        payload = json.dumps(record).encode() + b"\n"
        writer.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
        await writer.drain()
    writer.write(b"0\r\n\r\n")
    await writer.drain()
    assert await reader.readline() == b"HTTP/1.1 200 OK\r\n"
    writer.close()
    await writer.wait_closed()


async def _wait_for(event: threading.Event, timeout: float = 2.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not event.is_set() and loop.time() < deadline:
        await asyncio.sleep(0.001)
    return event.is_set()


async def test_start_runtime_injects_stream_endpoint_without_mutating_config(
    native_client: Fabric,
    mock_native: MagicMock,
):
    config = _config(relay=True)

    runtime = await native_client.start_runtime(config)

    planned = json.loads(mock_native.plan_config.call_args.args[0])
    endpoints = planned["relay"]["observability"]["atof"]["endpoints"]
    assert endpoints[0] == {
        "url": "https://example.com/events",
        "transport": "http_post",
        "headers": {},
        "timeout_millis": 3000,
        "field_name_policy": "preserve",
    }
    assert endpoints[1]["transport"] == "ndjson"
    assert endpoints[1]["url"].startswith("http://127.0.0.1:")
    assert runtime.supports_streaming is True
    assert len(config.relay.observability.atof.endpoints) == 1

    await runtime.stop()


async def test_invoke_stream_yields_raw_records_and_returns_result_out_of_band(
    native_client: Fabric,
    mock_native: MagicMock,
):
    runtime = await native_client.start_runtime(_config(relay=True))
    endpoint = json.loads(mock_native.plan_config.call_args.args[0])["relay"][
        "observability"
    ]["atof"]["endpoints"][-1]["url"]
    records = [
        {"type": "scope", "uuid": "scope-1", "name": "request"},
        {"type": "mark", "uuid": "mark-1", "parent_uuid": "scope-1"},
    ]

    stream = runtime.invoke_stream(input="hello")
    assert isinstance(stream, InvokeStream)
    await _post_chunked(endpoint, records)
    streamed = [record async for record in stream]
    result = await stream.result()

    assert streamed == records
    assert isinstance(result, RunResult)
    assert result.output["response"] == "done"
    assert all(not isinstance(record, RunResult) for record in streamed)
    await runtime.stop()


async def test_stream_must_be_finalized_before_another_turn(
    native_client: Fabric,
    mock_native: MagicMock,
):
    runtime = await native_client.start_runtime(_config(relay=True))
    endpoint = json.loads(mock_native.plan_config.call_args.args[0])["relay"][
        "observability"
    ]["atof"]["endpoints"][-1]["url"]
    stream = runtime.invoke_stream(input="first")
    await _post_chunked(endpoint, [{"uuid": "first"}])

    async for record in stream:
        assert record == {"uuid": "first"}
        break

    with pytest.raises(FabricStateError, match="streaming invocation is active"):
        runtime.invoke_stream(input="second")
    with pytest.raises(FabricStateError, match="cannot stop while a turn is in flight"):
        await runtime.stop()

    await stream.aclose()
    second = runtime.invoke_stream(input="second")
    assert [record async for record in second] == []
    assert (await second.result()).status == "succeeded"
    await runtime.stop()


async def test_invoke_stream_requires_relay(
    native_client: Fabric,
):
    runtime = await native_client.start_runtime(_config())

    assert runtime.supports_streaming is False
    with pytest.raises(
        FabricCapabilityError,
        match="requires Relay telemetry",
    ) as caught:
        runtime.invoke_stream(input="hello")

    assert caught.value.code == "streaming_unavailable"
    assert caught.value.details == {"capability": "streaming"}
    await runtime.stop()


async def test_context_manager_finalizes_unconsumed_stream(
    native_client: Fabric,
):
    async with await native_client.start_runtime(_config(relay=True)) as runtime:
        stream = runtime.invoke_stream(input="hello")

    assert (await stream.result()).status == "succeeded"


async def test_cancelled_aclose_keeps_turn_active_and_result_awaitable(
    native_client: Fabric,
    mock_native: MagicMock,
):
    started = threading.Event()
    release = threading.Event()

    def invoke(plan_json: str, runtime_json: str, request_json: str) -> str:
        started.set()
        assert release.wait(timeout=2)
        return json.dumps(_result(json.loads(request_json), json.loads(runtime_json)))

    mock_native.invoke_runtime.side_effect = invoke
    runtime = await native_client.start_runtime(_config(relay=True))
    stream = runtime.invoke_stream(input="hello")
    assert await _wait_for(started)

    closing = asyncio.create_task(stream.aclose())
    await asyncio.sleep(0)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    with pytest.raises(FabricStateError, match="streaming invocation is active"):
        runtime.invoke_stream(input="too soon")

    release.set()
    await stream.aclose()
    assert (await stream.result()).status == "succeeded"
    await runtime.stop()


async def test_listener_accepts_atof_record_larger_than_default_read_limits():
    listener = await _AtofStreamListener(maxsize=2).start()
    listener.begin_stream()
    record = {"uuid": "large", "payload": "x" * (600 * 1024)}

    await _post_chunked(listener.url, [record])

    assert await listener.records.get() == record
    listener.end_stream()
    await listener.close()
