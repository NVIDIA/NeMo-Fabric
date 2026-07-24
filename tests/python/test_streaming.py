# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for Relay-backed SDK streaming."""

from __future__ import annotations

import asyncio
import json
import threading
import warnings
from typing import Any
from unittest.mock import MagicMock

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
    RelayAtofConfig,
    RelayAtofFileSinkConfig,
    RelayAtofStreamSinkConfig,
    RelayObservabilityConfig,
    RunRequest,
    RunResult,
)
from nemo_fabric import client as client_mod
from nemo_fabric.streaming import _AtofStreamListener, _with_stream_sink


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


async def _open_chunked_upload(url: str) -> asyncio.StreamWriter:
    host_port = url.removeprefix("http://").split("/", 1)[0]
    host, port = host_port.split(":", 1)
    _reader, writer = await asyncio.open_connection(host, int(port))
    writer.write(
        b"POST /atof HTTP/1.1\r\n"
        + f"Host: {host_port}\r\n".encode()
        + b"Transfer-Encoding: chunked\r\n"
        + b"Content-Type: application/x-ndjson\r\n\r\n"
    )
    await writer.drain()
    return writer


async def _post_content_length(
    url: str,
    records: list[dict[str, Any]],
    *,
    expect_continue: bool = False,
) -> None:
    host_port = url.removeprefix("http://").split("/", 1)[0]
    host, port = host_port.split(":", 1)
    reader, writer = await asyncio.open_connection(host, int(port))
    payload = b"".join(json.dumps(record).encode() + b"\n" for record in records)
    expect = b"Expect: 100-continue\r\n" if expect_continue else b""
    writer.write(
        b"POST /atof HTTP/1.1\r\n"
        + f"Host: {host_port}\r\n".encode()
        + f"Content-Length: {len(payload)}\r\n".encode()
        + expect
        + b"Content-Type: application/x-ndjson\r\n\r\n"
    )
    await writer.drain()
    if expect_continue:
        assert await reader.readline() == b"HTTP/1.1 100 Continue\r\n"
        assert await reader.readline() == b"\r\n"
    writer.write(payload)
    await writer.drain()
    assert await reader.readline() == b"HTTP/1.1 200 OK\r\n"
    writer.close()
    await writer.wait_closed()


async def _request_status(url: str, request: bytes) -> bytes:
    host_port = url.removeprefix("http://").split("/", 1)[0]
    host, port = host_port.split(":", 1)
    reader, writer = await asyncio.open_connection(host, int(port))
    writer.write(request)
    await writer.drain()
    status = await reader.readline()
    writer.close()
    await writer.wait_closed()
    return status


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

    runtime = await native_client.start_runtime(config)

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


async def test_start_runtime_does_not_enable_disabled_atof_outputs(
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
    assert planned_atof["enabled"] is True
    assert len(planned_atof["sinks"]) == 1
    assert planned_atof["sinks"][0]["type"] == "stream"
    assert planned_atof["sinks"][0]["name"] == "nemo-fabric-stream"
    assert planned_atof["sinks"][0]["url"].startswith("http://127.0.0.1:")
    assert config.relay.observability.atof.enabled is False
    assert config.relay.observability.atof.sinks[0].output_directory == "./disabled"

    await runtime.stop()


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
    runtime = await native_client.start_runtime(_config(relay=True))
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
    await _post_content_length(endpoint, records)
    streamed = [record async for record in stream]
    result = await stream.result()

    assert streamed == records
    assert isinstance(result, RunResult)
    assert result.output["response"] == "done"
    assert all(not isinstance(record, RunResult) for record in streamed)
    await runtime.stop()


async def test_invoke_stream_correlates_relay_gateway_turn_indexes(
    native_client: Fabric,
    mock_native: MagicMock,
):
    runtime = await native_client.start_runtime(_config(relay=True))
    endpoint = json.loads(mock_native.plan_config.call_args.args[0])["relay"][
        "observability"
    ]["atof"]["sinks"][-1]["url"]

    for turn_index in (1, 2):
        records = [
            {
                "kind": "scope",
                "scope_category": "start",
                "uuid": f"turn-{turn_index}",
                "metadata": {
                    "nemo_relay_scope_role": "turn",
                    "turn_index": turn_index,
                },
            },
            {
                "kind": "mark",
                "uuid": f"mark-{turn_index}",
                "parent_uuid": f"turn-{turn_index}",
            },
        ]

        stream = runtime.invoke_stream(input=f"turn {turn_index}")
        await _post_content_length(endpoint, records)

        assert [record async for record in stream] == records
        assert (await stream.result()).status == "succeeded"

    await runtime.stop()


async def test_stream_must_be_finalized_before_another_turn(
    native_client: Fabric,
    mock_native: MagicMock,
):
    runtime = await native_client.start_runtime(_config(relay=True))
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
    await _post_content_length(endpoint, [first])

    async for record in stream:
        assert record == first
        break

    with pytest.raises(FabricStateError, match="streaming invocation is active"):
        runtime.invoke_stream(input="second")

    await stream.aclose()
    second = runtime.invoke_stream(input="second")
    with pytest.warns(RuntimeWarning, match="No Relay ATOF connection"):
        assert [record async for record in second] == []
    assert (await second.result()).status == "succeeded"
    await runtime.stop()


async def test_invoke_stream_validates_request_before_returning_stream(
    native_client: Fabric,
):
    runtime = await native_client.start_runtime(_config(relay=True))
    request = RunRequest(input="request")

    with pytest.raises(FabricConfigError, match="mutually exclusive"):
        runtime.invoke_stream(input="input", request=request)

    stream = runtime.invoke_stream(input="valid")
    with pytest.warns(RuntimeWarning, match="No Relay ATOF connection"):
        assert [record async for record in stream] == []
    assert (await stream.result()).status == "succeeded"
    await runtime.stop()


async def test_invoke_stream_warns_only_once_when_relay_never_connects(
    native_client: Fabric,
):
    runtime = await native_client.start_runtime(_config(relay=True))

    first = runtime.invoke_stream(input="first")
    with pytest.warns(RuntimeWarning, match="same network namespace"):
        assert [record async for record in first] == []
    assert (await first.result()).status == "succeeded"

    second = runtime.invoke_stream(input="second")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert [record async for record in second] == []
    assert caught == []
    assert (await second.result()).status == "succeeded"
    await runtime.stop()


async def test_invoke_stream_does_not_warn_after_relay_connects(
    native_client: Fabric,
    mock_native: MagicMock,
):
    runtime = await native_client.start_runtime(_config(relay=True))
    endpoint = json.loads(mock_native.plan_config.call_args.args[0])["relay"][
        "observability"
    ]["atof"]["sinks"][-1]["url"]

    stream = runtime.invoke_stream(input="empty stream")
    await _post_content_length(endpoint, [])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert [record async for record in stream] == []

    assert caught == []
    assert (await stream.result()).status == "succeeded"
    await runtime.stop()


async def test_invoke_stream_warns_after_long_lived_relay_upload_disconnects(
    native_client: Fabric,
    mock_native: MagicMock,
):
    runtime = await native_client.start_runtime(_config(relay=True))
    endpoint = json.loads(mock_native.plan_config.call_args.args[0])["relay"][
        "observability"
    ]["atof"]["sinks"][-1]["url"]
    writer = await _open_chunked_upload(endpoint)

    first = runtime.invoke_stream(input="connected stream")
    root = {
        "kind": "scope",
        "scope_category": "start",
        "uuid": "turn-1",
        "metadata": {
            "nemo_relay_scope_role": "turn",
            "turn_index": 1,
        },
    }
    payload = json.dumps(root).encode() + b"\n"
    writer.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
    await writer.drain()

    assert [record async for record in first] == [root]
    assert (await first.result()).status == "succeeded"

    writer.close()
    await writer.wait_closed()
    listener = runtime._stream_listener
    while listener._active_atof_connections:
        await asyncio.sleep(0)

    second = runtime.invoke_stream(input="disconnected stream")
    with pytest.warns(RuntimeWarning, match="No Relay ATOF connection"):
        assert [record async for record in second] == []

    assert (await second.result()).status == "succeeded"
    await runtime.stop()


async def test_invoke_stream_warns_when_long_lived_upload_drops_during_turn(
    native_client: Fabric,
    mock_native: MagicMock,
):
    runtime = await native_client.start_runtime(_config(relay=True))
    endpoint = json.loads(mock_native.plan_config.call_args.args[0])["relay"][
        "observability"
    ]["atof"]["sinks"][-1]["url"]
    writer = await _open_chunked_upload(endpoint)
    listener = runtime._stream_listener
    while not listener._active_atof_connections:
        await asyncio.sleep(0)

    stream = runtime.invoke_stream(input="dropped stream")
    writer.close()
    await writer.wait_closed()
    while listener._active_atof_connections:
        await asyncio.sleep(0)

    with pytest.warns(RuntimeWarning, match="No Relay ATOF connection"):
        assert [record async for record in stream] == []

    assert (await stream.result()).status == "succeeded"
    await runtime.stop()


async def test_invoke_stream_warns_when_long_lived_upload_truncates_turn(
    native_client: Fabric,
    mock_native: MagicMock,
):
    runtime = await native_client.start_runtime(_config(relay=True))
    endpoint = json.loads(mock_native.plan_config.call_args.args[0])["relay"][
        "observability"
    ]["atof"]["sinks"][-1]["url"]
    writer = await _open_chunked_upload(endpoint)
    listener = runtime._stream_listener
    while not listener._active_atof_connections:
        await asyncio.sleep(0)

    stream = runtime.invoke_stream(input="truncated stream")
    root = {
        "kind": "scope",
        "scope_category": "start",
        "uuid": "turn-1",
        "metadata": {
            "nemo_relay_scope_role": "turn",
            "turn_index": 1,
        },
    }
    payload = json.dumps(root).encode() + b"\n"
    writer.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
    await writer.drain()
    while listener.records.empty():
        await asyncio.sleep(0)

    writer.close()
    await writer.wait_closed()
    while listener._active_atof_connections:
        await asyncio.sleep(0)

    with pytest.warns(RuntimeWarning, match="streaming may be incomplete"):
        assert [record async for record in stream] == [root]

    assert (await stream.result()).status == "succeeded"
    await runtime.stop()


async def test_invoke_stream_warns_when_long_lived_upload_ends_during_turn(
    native_client: Fabric,
    mock_native: MagicMock,
):
    runtime = await native_client.start_runtime(_config(relay=True))
    endpoint = json.loads(mock_native.plan_config.call_args.args[0])["relay"][
        "observability"
    ]["atof"]["sinks"][-1]["url"]
    root = {
        "kind": "scope",
        "scope_category": "start",
        "uuid": "turn-1",
        "metadata": {
            "nemo_relay_scope_role": "turn",
            "turn_index": 1,
        },
    }

    stream = runtime.invoke_stream(input="ended stream")
    await _post_chunked(endpoint, [root])

    with pytest.warns(RuntimeWarning, match="streaming may be incomplete"):
        assert [record async for record in stream] == [root]

    assert (await stream.result()).status == "succeeded"
    await runtime.stop()


async def test_invoke_stream_warns_when_records_do_not_match_active_turn(
    native_client: Fabric,
    mock_native: MagicMock,
):
    runtime = await native_client.start_runtime(_config(relay=True))
    endpoint = json.loads(mock_native.plan_config.call_args.args[0])["relay"][
        "observability"
    ]["atof"]["sinks"][-1]["url"]
    stream = runtime.invoke_stream(input="unmatched stream")
    await _post_content_length(
        endpoint,
        [
            {
                "kind": "scope",
                "scope_category": "start",
                "uuid": "unexpected-turn",
                "metadata": {
                    "nemo_relay_scope_role": "turn",
                    "turn_index": 99,
                },
            }
        ],
    )

    with pytest.warns(RuntimeWarning, match="no record matched the active Fabric turn"):
        assert [record async for record in stream] == []

    assert (await stream.result()).status == "succeeded"

    second = runtime.invoke_stream(input="another unmatched stream")
    await _post_content_length(
        endpoint,
        [
            {
                "kind": "scope",
                "scope_category": "start",
                "uuid": "another-unexpected-turn",
                "metadata": {
                    "nemo_relay_scope_role": "turn",
                    "turn_index": 99,
                },
            }
        ],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert [record async for record in second] == []

    assert caught == []
    assert (await second.result()).status == "succeeded"
    await runtime.stop()


async def test_stop_finalizes_completed_stream_after_result(
    native_client: Fabric,
):
    runtime = await native_client.start_runtime(_config(relay=True))
    stream = runtime.invoke_stream(input="hello")

    assert (await stream.result()).status == "succeeded"
    assert stream._finalized is False

    await runtime.stop()

    assert stream._finalized is True


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
    with pytest.raises(
        FabricStateError,
        match="streaming invocation is active",
    ):
        await runtime.stop()

    release.set()
    await stream.aclose()
    assert (await stream.result()).status == "succeeded"
    await runtime.stop()


async def test_aclose_drains_backpressure_while_invocation_finishes():
    listener = await _AtofStreamListener(maxsize=1).start()
    producer_finished = asyncio.Event()

    async def invoke() -> RunResult:
        await producer_finished.wait()
        return RunResult.from_mapping(_result({"request_id": "request-1"}, _runtime()))

    stream = InvokeStream(invoke(), listener)
    records = [{"uuid": f"record-{index}"} for index in range(3)]

    async def produce() -> None:
        await _post_chunked(listener.url, records)
        producer_finished.set()

    producer = asyncio.create_task(produce())
    while not listener.records.full():
        await asyncio.sleep(0)
    assert not producer.done()

    await asyncio.wait_for(stream.aclose(), timeout=1)

    assert producer.done()
    assert (await stream.result()).status == "succeeded"
    await listener.close()


@pytest.mark.parametrize(
    "current_metadata",
    [
        {"nemo_fabric_request_id": "request-2"},
        {"nemo_relay_scope_role": "turn", "turn_index": 2},
    ],
)
async def test_listener_correlates_records_to_active_turn(
    current_metadata: dict[str, Any],
):
    listener = await _AtofStreamListener(maxsize=4).start()
    listener.begin_stream(request_id="request-2", turn_index=2)
    current = [
        {
            "kind": "scope",
            "scope_category": "start",
            "uuid": "current",
            "metadata": current_metadata,
        },
        {
            "kind": "scope",
            "scope_category": "start",
            "uuid": "child",
            "parent_uuid": "current",
        },
        {"kind": "mark", "uuid": "mark", "parent_uuid": "child"},
    ]

    await _post_chunked(
        listener.url,
        [
            {
                "kind": "scope",
                "scope_category": "start",
                "uuid": "previous",
                "metadata": {"nemo_fabric_request_id": "request-1"},
            },
            {"kind": "mark", "uuid": "late", "parent_uuid": "previous"},
            *current,
            {"kind": "mark", "uuid": "unrelated", "parent_uuid": "previous"},
        ],
    )

    assert [await listener.records.get() for _ in current] == current
    assert listener.records.empty()
    listener.end_stream()
    await listener.close()


async def test_listener_applies_byte_budget_backpressure():
    record = {"uuid": "record", "payload": "x" * 16}
    record_size = len(json.dumps(record).encode())
    listener = await _AtofStreamListener(
        maxsize=10,
        max_bytes=record_size,
        max_record_bytes=record_size,
    ).start()
    listener.begin_stream()

    producer = asyncio.create_task(_post_chunked(listener.url, [record, record]))
    while listener.records.empty():
        await asyncio.sleep(0)
    assert not producer.done()

    assert await listener.records.get() == record
    assert await listener.records.get() == record
    await producer
    listener.end_stream()
    await listener.close()


async def test_listener_rejects_oversized_record():
    listener = await _AtofStreamListener(max_record_bytes=32).start()
    listener.begin_stream(request_id="request-1")
    payload = json.dumps({"payload": "x" * 64}).encode() + b"\n"
    request = (
        b"POST /atof HTTP/1.1\r\n"
        + f"Content-Length: {len(payload)}\r\n".encode()
        + b"Content-Type: application/x-ndjson\r\n\r\n"
        + payload
    )

    assert await _request_status(listener.url, request) == (
        b"HTTP/1.1 413 Content Too Large\r\n"
    )
    listener.end_stream()
    with pytest.warns(RuntimeWarning, match="no record matched the active Fabric turn"):
        listener.warn_if_unavailable()
    await listener.close()


async def test_listener_accepts_atof_record_larger_than_default_read_limits():
    listener = await _AtofStreamListener(maxsize=2).start()
    listener.begin_stream()
    record = {"uuid": "large", "payload": "x" * (600 * 1024)}

    await _post_chunked(listener.url, [record])

    assert await listener.records.get() == record
    listener.end_stream()
    await listener.close()


async def test_listener_accepts_content_length_and_100_continue():
    listener = await _AtofStreamListener(maxsize=2).start()
    listener.begin_stream()
    records = [{"uuid": "first"}, {"uuid": "second"}]

    await _post_content_length(listener.url, records, expect_continue=True)

    assert [await listener.records.get(), await listener.records.get()] == records
    listener.end_stream()
    await listener.close()


@pytest.mark.parametrize(
    ("raw_request", "expected"),
    [
        (
            b"GET /atof HTTP/1.1\r\nContent-Length: 0\r\n\r\n",
            b"HTTP/1.1 404 Not Found\r\n",
        ),
        (
            b"POST /atof HTTP/1.1\r\n\r\n",
            b"HTTP/1.1 411 Length Required\r\n",
        ),
        (
            b"POST /atof HTTP/1.1\r\nContent-Length: invalid\r\n\r\n",
            b"HTTP/1.1 400 Bad Request\r\n",
        ),
    ],
)
async def test_listener_rejects_invalid_http_requests(
    raw_request: bytes,
    expected: bytes,
):
    listener = await _AtofStreamListener().start()

    assert await _request_status(listener.url, raw_request) == expected

    await listener.close()
