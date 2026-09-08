# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for Relay-backed SDK streaming."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import urlsplit

import pytest

from nemo_fabric import (
    Fabric,
    FabricCapabilityError,
    FabricConfig,
    FabricConfigError,
    FabricStateError,
    HarnessConfig,
    InvokeStream,
    MetadataConfig,
    RelayAtifConfig,
    RelayAtofConfig,
    RelayAtofFileSinkConfig,
    RelayAtofStreamSinkConfig,
    RelayObservabilityConfig,
    RunRequest,
    RunResult,
)
from nemo_fabric import client as client_mod
from nemo_fabric.streaming import _with_stream_sink


def _config(*, relay: bool = False) -> FabricConfig:
    config = FabricConfig(
        metadata=MetadataConfig(name="demo"),
        harness=HarnessConfig(adapter_id="test.fabric.shim"),
    )
    if relay:
        config.enable_relay(
            observability=RelayObservabilityConfig(
                atof=RelayAtofConfig(
                    enabled=True,
                    sinks=[
                        RelayAtofStreamSinkConfig(
                            name="user-stream",
                            url="https://example.com/events",
                        )
                    ],
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


async def _post_content_length(
    url: str,
    records: list[dict[str, Any]],
) -> None:
    parsed = urlsplit(url)
    assert parsed.hostname is not None
    assert parsed.port is not None
    host_port = parsed.netloc
    reader, writer = await asyncio.open_connection(parsed.hostname, parsed.port)
    payload = b"".join(json.dumps(record).encode() + b"\n" for record in records)
    writer.write(
        f"POST {parsed.path} HTTP/1.1\r\n".encode()
        + f"Host: {host_port}\r\n".encode()
        + f"Content-Length: {len(payload)}\r\n".encode()
        + b"Content-Type: application/x-ndjson\r\n\r\n"
    )
    await writer.drain()
    writer.write(payload)
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


async def test_start_runtime_injects_stream_sink_without_mutating_config(
    native_client: Fabric,
    mock_native: MagicMock,
):
    config = _config(relay=True)

    runtime = await native_client.start_runtime(config, streaming=True)

    planned = json.loads(mock_native.plan_config.call_args.args[0])
    sinks = planned["relay"]["observability"]["atof"]["sinks"]
    assert sinks[0] == {
        "type": "stream",
        "url": "https://example.com/events",
        "transport": "http_post",
        "timeout_millis": 3000,
        "field_name_policy": "preserve",
        "name": "user-stream",
    }
    assert sinks[1]["type"] == "stream"
    assert sinks[1]["name"] == "nemo-fabric-stream"
    assert sinks[1]["transport"] == "ndjson"
    assert sinks[1]["url"].startswith("http://127.0.0.1:")
    assert runtime.supports_streaming is True
    assert len(config.relay.observability.atof.sinks) == 1

    await runtime.stop()


async def test_start_runtime_without_streaming_preserves_disabled_atof(
    native_client: Fabric,
    mock_native: MagicMock,
):
    config = _config(relay=True)
    atof = config.relay.observability.atof
    atof.enabled = False
    atof.sinks = [RelayAtofFileSinkConfig(output_directory="./disabled")]

    runtime = await native_client.start_runtime(config)

    planned = json.loads(mock_native.plan_config.call_args.args[0])
    planned_atof = planned["relay"]["observability"]["atof"]
    assert planned_atof["enabled"] is False
    assert len(planned_atof["sinks"]) == 1
    assert planned_atof["sinks"][0]["type"] == "file"
    assert runtime.supports_streaming is False
    assert config.relay.observability.atof.enabled is False
    assert config.relay.observability.atof.sinks[0].output_directory == "./disabled"

    await runtime.stop()


async def test_start_runtime_without_streaming_does_not_add_atof(
    native_client: Fabric,
    mock_native: MagicMock,
):
    config = _config()
    config.enable_relay(
        observability=RelayObservabilityConfig(
            atif=RelayAtifConfig(enabled=True),
        )
    )

    runtime = await native_client.start_runtime(config)

    planned = json.loads(mock_native.plan_config.call_args.args[0])
    observability = planned["relay"]["observability"]
    assert "atof" not in observability
    assert observability["atif"]["enabled"] is True
    assert runtime.supports_streaming is False

    await runtime.stop()


async def test_start_runtime_streaming_enables_only_reserved_atof_sink(
    native_client: Fabric,
    mock_native: MagicMock,
):
    config = _config(relay=True)
    atof = config.relay.observability.atof
    atof.enabled = False
    atof.sinks = [RelayAtofFileSinkConfig(output_directory="./disabled")]

    runtime = await native_client.start_runtime(config, streaming=True)

    planned = json.loads(mock_native.plan_config.call_args.args[0])
    planned_atof = planned["relay"]["observability"]["atof"]
    assert planned_atof["enabled"] is True
    assert len(planned_atof["sinks"]) == 1
    assert planned_atof["sinks"][0]["type"] == "stream"
    assert planned_atof["sinks"][0]["name"] == "nemo-fabric-stream"
    assert planned_atof["sinks"][0]["url"].startswith("http://127.0.0.1:")
    assert runtime.supports_streaming is True
    assert config.relay.observability.atof.enabled is False
    assert config.relay.observability.atof.sinks[0].output_directory == "./disabled"

    await runtime.stop()


async def test_start_runtime_rejects_streaming_without_relay(
    native_client: Fabric,
):
    with pytest.raises(
        FabricConfigError,
        match="streaming requires Relay telemetry",
    ):
        await native_client.start_runtime(_config(), streaming=True)


def test_with_stream_sink_replaces_reserved_sink_and_preserves_user_sinks():
    config = _config(relay=True)

    first = _with_stream_sink(config, "http://127.0.0.1:4100/atof")
    second = _with_stream_sink(first, "http://127.0.0.1:4200/atof")

    sinks = second.relay.observability.atof.sinks
    assert [sink.name for sink in sinks] == [
        "user-stream",
        "nemo-fabric-stream",
    ]
    assert sinks[-1].url == "http://127.0.0.1:4200/atof"
    assert len(config.relay.observability.atof.sinks) == 1


async def test_invoke_stream_yields_raw_records_and_returns_result_out_of_band(
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
    runtime = await native_client.start_runtime(_config(relay=True), streaming=True)
    endpoint = json.loads(mock_native.plan_config.call_args.args[0])["relay"][
        "observability"
    ]["atof"]["sinks"][-1]["url"]
    request = RunRequest(input="hello", request_id="request-stream")
    records = [
        {
            "kind": "scope",
            "scope_category": "start",
            "uuid": "scope-1",
            "name": "request",
            "metadata": {"nemo_fabric_request_id": request.request_id},
        },
        {"kind": "mark", "uuid": "mark-1", "parent_uuid": "scope-1"},
    ]

    stream = runtime.invoke_stream(request=request)
    assert isinstance(stream, InvokeStream)
    assert await _wait_for(started)
    await _post_content_length(endpoint, records)
    release.set()
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
    started = threading.Event()
    release = threading.Event()

    def invoke(plan_json: str, runtime_json: str, request_json: str) -> str:
        started.set()
        assert release.wait(timeout=2)
        return json.dumps(_result(json.loads(request_json), json.loads(runtime_json)))

    mock_native.invoke_runtime.side_effect = invoke
    runtime = await native_client.start_runtime(_config(relay=True), streaming=True)
    endpoint = json.loads(mock_native.plan_config.call_args.args[0])["relay"][
        "observability"
    ]["atof"]["sinks"][-1]["url"]
    request = RunRequest(input="first", request_id="request-first")
    first = {
        "kind": "scope",
        "scope_category": "start",
        "uuid": "first",
        "metadata": {"nemo_fabric_request_id": request.request_id},
    }
    stream = runtime.invoke_stream(request=request)
    assert await _wait_for(started)
    await _post_content_length(endpoint, [first])

    async for record in stream:
        assert record == first
        break

    with pytest.raises(FabricStateError, match="streaming invocation is active"):
        runtime.invoke_stream(input="second")

    release.set()
    await stream.aclose()
    second = runtime.invoke_stream(input="second")
    assert [record async for record in second] == []
    assert (await second.result()).status == "succeeded"
    await runtime.stop()


async def test_invoke_stream_validates_request_before_returning_stream(
    native_client: Fabric,
):
    runtime = await native_client.start_runtime(_config(relay=True), streaming=True)
    request = RunRequest(input="request")

    with pytest.raises(FabricConfigError, match="mutually exclusive"):
        runtime.invoke_stream(input="input", request=request)

    stream = runtime.invoke_stream(input="valid")
    assert [record async for record in stream] == []
    assert (await stream.result()).status == "succeeded"
    await runtime.stop()


@pytest.mark.parametrize("relay", [False, True])
async def test_invoke_stream_requires_streaming_enabled_at_startup(
    native_client: Fabric,
    relay: bool,
):
    runtime = await native_client.start_runtime(_config(relay=relay))

    assert runtime.supports_streaming is False
    with pytest.raises(
        FabricCapabilityError,
        match=r"requires a configured standalone ATOF collector.*streaming=True",
    ) as caught:
        runtime.invoke_stream(input="hello")

    assert caught.value.code == "streaming_unavailable"
    assert caught.value.details == {"capability": "streaming"}
    await runtime.stop()


async def test_context_manager_finalizes_unconsumed_stream(
    native_client: Fabric,
):
    async with await native_client.start_runtime(
        _config(relay=True), streaming=True
    ) as runtime:
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
    runtime = await native_client.start_runtime(_config(relay=True), streaming=True)
    stream = runtime.invoke_stream(input="hello")
    assert await _wait_for(started)

    closing = asyncio.create_task(stream.aclose())
    await asyncio.sleep(0)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    with pytest.raises(FabricStateError, match="streaming invocation is active"):
        runtime.invoke_stream(input="too soon")
    with pytest.raises(
        FabricStateError,
        match="streaming invocation is active",
    ):
        await runtime.stop()

    release.set()
    await stream.aclose()
    assert (await stream.result()).status == "succeeded"
    await runtime.stop()


async def test_cancelled_anext_does_not_consume_next_record():
    invocation_finished = asyncio.Event()
    records: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def collector_records(_: str):
        while record := await records.get():
            yield record

    async def invoke() -> RunResult:
        await invocation_finished.wait()
        return RunResult.from_mapping(_result({"request_id": "request-1"}, _runtime()))

    registration_ready = asyncio.Event()
    registration_ready.set()
    mock_collector = MagicMock()
    mock_collector.stream.side_effect = collector_records
    stream = InvokeStream(
        invoke(),
        mock_collector,
        request_id="request-1",
        registration_ready=registration_ready,
    )
    pending = asyncio.create_task(stream.__anext__())
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    expected = [{"uuid": "first"}, {"uuid": "second"}]
    for record in expected:
        await records.put(record)

    assert await stream.__anext__() == expected[0]
    assert await stream.__anext__() == expected[1]

    invocation_finished.set()
    await records.put(None)
    await stream.aclose()

async def test_cancelled_anext_retains_record_consumed_during_cancellation():
    invocation_finished = asyncio.Event()
    records: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def collector_records(_: str):
        while record := await records.get():
            yield record

    async def invoke() -> RunResult:
        await invocation_finished.wait()
        return RunResult.from_mapping(_result({"request_id": "request-1"}, _runtime()))

    registration_ready = asyncio.Event()
    registration_ready.set()
    mock_collector = MagicMock()
    mock_collector.stream.side_effect = collector_records
    stream = InvokeStream(
        invoke(),
        mock_collector,
        request_id="request-1",
        registration_ready=registration_ready,
    )
    record = {"uuid": "first"}

    async def cancel_after_getter_completes(
        tasks: set[asyncio.Task[Any]],
        *,
        return_when: str,
    ) -> None:
        assert return_when == asyncio.FIRST_COMPLETED
        getter = next(task for task in tasks if task is not stream._task)
        records.put_nowait(record)
        assert await getter == record
        raise asyncio.CancelledError

    with (
        patch.object(asyncio, "wait", new=cancel_after_getter_completes),
        pytest.raises(asyncio.CancelledError),
    ):
        await stream.__anext__()

    assert await stream.__anext__() == record

    invocation_finished.set()
    await records.put(None)
    await stream.aclose()
