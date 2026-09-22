# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for Relay-backed SDK streaming."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlsplit

import pytest

from nemo_fabric import (
    Fabric,
    FabricCapabilityError,
    FabricConfig,
    FabricConfigError,
    FabricRuntimeError,
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
from nemo_fabric import streaming as streaming_mod
from nemo_fabric._collector_client import _AtofCollectorClient
from nemo_fabric.streaming import _with_stream_sink


def _config(
    *,
    relay: bool = False,
    adapter_id: str = "test.fabric.shim",
) -> FabricConfig:
    config = FabricConfig(
        metadata=MetadataConfig(name="demo"),
        harness=HarnessConfig(adapter_id=adapter_id),
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
    adapter_id = config["harness"]["adapter_id"]
    is_pi = adapter_id == "nvidia.fabric.pi"
    return {
        "agent_name": "demo",
        "base_dir": ".",
        "config": config,
        "adapter_descriptor": {
            "descriptor": {
                "adapter_id": adapter_id,
                "harness": "pi" if is_pi else "hermes",
                "adapter_kind": "typescript" if is_pi else "python",
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


def _pi_turn_records(turn_index: int, turn_seq: int) -> list[dict[str, Any]]:
    turn_uuid = f"pi-turn-{turn_index}"
    model_uuid = f"pi-model-{turn_index}"
    turn = {
        "kind": "scope",
        "uuid": turn_uuid,
        "name": "pi-turn",
        "parent_uuid": "pi-session",
        "metadata": {
            "agent_kind": "pi",
            "hook_event_name": "turn_start",
            "nemo_relay_scope_role": "turn",
            "turn_index": turn_index,
            "turn_seq": turn_seq,
            "turn_source": "turn_start",
        },
    }
    model = {
        "kind": "scope",
        "uuid": model_uuid,
        "name": "openai.responses",
        "parent_uuid": turn_uuid,
    }
    return [
        {**turn, "scope_category": "start"},
        {**model, "scope_category": "start"},
        {
            "kind": "mark",
            "uuid": f"pi-chunk-{turn_index}",
            "parent_uuid": model_uuid,
        },
        {**model, "scope_category": "end"},
        {**turn, "scope_category": "end"},
    ]


def _pi_completion_record(turn_seq: int) -> dict[str, Any]:
    return {
        "kind": "mark",
        "uuid": f"pi-settled-{turn_seq}",
        "parent_uuid": "pi-session",
        "metadata": {
            "agent_kind": "pi",
            "hook_event_name": "agent_settled",
            "turn_seq": turn_seq,
        },
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


async def _collect_records(stream: InvokeStream) -> list[dict[str, Any]]:
    return [record async for record in stream]


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
    assert runtime._collector is not None
    assert len(config.relay.observability.atof.sinks) == 1

    await runtime.stop()


async def test_start_runtime_plumbs_collector_completion_wait_timeout(
    native_client: Fabric,
):
    collector_context = MagicMock()
    collector_context.__aenter__ = AsyncMock(return_value="http://127.0.0.1:4318")
    collector_context.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "nemo_fabric_collector.serve_collector",
        return_value=collector_context,
    ) as mock_serve_collector:
        runtime = await native_client.start_runtime(
            _config(relay=True, adapter_id="nvidia.fabric.pi"),
            streaming=True,
            completion_wait_timeout=5.0,
        )
        assert runtime._collector_client is not None
        assert runtime._collector_client._client.timeout.connect == 6.0
        await runtime.stop()

    mock_serve_collector.assert_called_once_with(
        host="127.0.0.1",
        port=0,
        standalone=True,
        completion_wait_timeout=5.0,
    )


@pytest.mark.parametrize(
    "completion_wait_timeout",
    [0, -1, float("inf"), float("nan"), True, "1"],
)
async def test_start_runtime_rejects_invalid_completion_wait_timeout(
    native_client: Fabric,
    completion_wait_timeout: Any,
):
    with pytest.raises(
        FabricConfigError,
        match="completion_wait_timeout must be a finite number greater than zero",
    ):
        await native_client.start_runtime(
            _config(relay=True, adapter_id="nvidia.fabric.pi"),
            streaming=True,
            completion_wait_timeout=completion_wait_timeout,
        )


async def test_start_runtime_ignores_completion_wait_timeout_without_streaming(
    native_client: Fabric,
):
    runtime = await native_client.start_runtime(
        _config(),
        completion_wait_timeout=0,
    )

    await runtime.stop()


async def test_start_runtime_ignores_completion_wait_timeout_for_non_pi_streaming(
    native_client: Fabric,
):
    runtime = await native_client.start_runtime(
        _config(relay=True),
        streaming=True,
        completion_wait_timeout=0,
    )

    await runtime.stop()


async def test_start_runtime_creates_runtime_owned_collectors(
    native_client: Fabric,
    mock_native: MagicMock,
):
    first = await native_client.start_runtime(_config(relay=True), streaming=True)
    second = await native_client.start_runtime(_config(relay=True), streaming=True)
    try:
        first_plan, second_plan = (
            json.loads(call.args[0]) for call in mock_native.plan_config.call_args_list
        )
        first_sink = first_plan["relay"]["observability"]["atof"]["sinks"][-1]
        second_sink = second_plan["relay"]["observability"]["atof"]["sinks"][-1]

        assert first._collector is not second._collector
        assert first_sink["url"] != second_sink["url"]
    finally:
        await first.stop()
        await second.stop()


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


async def test_pi_streaming_rejects_external_collector(
    native_client: Fabric,
    mock_native: MagicMock,
):
    config = _config(relay=True, adapter_id="nvidia.fabric.pi")
    config.relay.observability.atof.sinks.append(
        RelayAtofStreamSinkConfig(
            name="nemo-fabric-stream",
            url="http://127.0.0.1:4318/v1/atof",
            transport="ndjson",
        )
    )

    with pytest.raises(FabricConfigError, match="requires the embedded collector"):
        await native_client.start_runtime(
            config,
            streaming=True,
            launch_collector=False,
        )

    mock_native.plan_config.assert_not_called()


async def test_collector_client_omits_inactive_pi_control_fields():
    client = _AtofCollectorClient(
        base_url="http://collector.test",
        timeout_seconds=1,
        headers={},
    )
    client._request = AsyncMock()
    try:
        await client.register("request-register-1")
        await client.register(
            "request-register-2",
            correlation_mode="pi_turn_window",
            registration_token="attempt-2",
        )
        await client.deregister("request-1", remove_queue=False)
        await client.deregister(
            "request-2",
            remove_queue=False,
            pi_boundary="wait",
            pi_turn_count=2,
            registration_token="attempt-2",
        )
    finally:
        await client.aclose()

    assert client._request.await_args_list[0].kwargs["json"] == {
        "request_id": "request-register-1"
    }
    assert client._request.await_args_list[1].kwargs["json"] == {
        "request_id": "request-register-2",
        "correlation_mode": "pi_turn_window",
        "registration_token": "attempt-2",
    }
    assert client._request.await_args_list[2].kwargs["params"] == {
        "remove_queue": "false"
    }
    assert client._request.await_args_list[3].kwargs["params"] == {
        "remove_queue": "false",
        "pi_boundary": "wait",
        "pi_turn_count": "2",
        "registration_token": "attempt-2",
    }


async def test_collector_client_preserves_error_detail():
    client = _AtofCollectorClient(
        base_url="http://collector.test",
        timeout_seconds=1,
        headers={},
    )
    response = MagicMock(status_code=409)
    response.json.return_value = {
        "detail": "previous Pi invocation boundary is unresolved"
    }
    client._client.request = AsyncMock(return_value=response)
    try:
        with pytest.raises(
            FabricRuntimeError,
            match="previous Pi invocation boundary is unresolved",
        ):
            await client.register("request-1")
    finally:
        await client.aclose()


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


async def test_pi_like_streaming_captures_sibling_turns_across_invocations(
    native_client: Fabric,
    mock_native: MagicMock,
):
    first_started = threading.Event()
    first_release = threading.Event()
    plain_started = threading.Event()
    plain_release = threading.Event()
    second_started = threading.Event()
    second_release = threading.Event()
    controlled = {
        "request-pi-first": (first_started, first_release),
        "request-pi-plain": (plain_started, plain_release),
        "request-pi-second": (second_started, second_release),
    }

    def invoke(plan_json: str, runtime_json: str, request_json: str) -> str:
        request = json.loads(request_json)
        synchronization = controlled.get(request["request_id"])
        if synchronization is not None:
            started, release = synchronization
            started.set()
            assert release.wait(timeout=2)
        result = _result(request, json.loads(runtime_json))
        result["metadata"] = {
            "adapter": {
                "pi_turn_started": True,
                "pi_turn_count": {
                    "request-pi-first": 2,
                    "request-pi-plain": 3,
                    "request-pi-second": 4,
                }[request["request_id"]],
            }
        }
        return json.dumps(result)

    mock_native.invoke_runtime.side_effect = invoke
    runtime = await native_client.start_runtime(
        _config(relay=True, adapter_id="nvidia.fabric.pi"),
        streaming=True,
    )
    assert runtime.supports_streaming is True
    assert runtime.supports_openai_streaming is False
    endpoint = json.loads(mock_native.plan_config.call_args.args[0])["relay"][
        "observability"
    ]["atof"]["sinks"][-1]["url"]
    first_turns = _pi_turn_records(1, 0) + _pi_turn_records(2, 1)
    first_completion = _pi_completion_record(1)
    first_records = first_turns + [first_completion]

    first = runtime.invoke_stream(
        request=RunRequest(input="use a tool", request_id="request-pi-first")
    )
    assert await _wait_for(first_started)
    first_collection = asyncio.create_task(_collect_records(first))
    await _post_content_length(endpoint, first_turns)
    first_release.set()
    first_result = asyncio.create_task(first.result())
    await asyncio.sleep(0.01)
    assert not first_result.done()
    await _post_content_length(endpoint, [first_completion])

    assert await first_collection == first_records
    assert (await first_result).status == "succeeded"

    ordinary_task = asyncio.create_task(
        runtime.invoke(
            request=RunRequest(
                input="run without streaming",
                request_id="request-pi-plain",
            )
        )
    )
    assert await _wait_for(plain_started)
    plain_release.set()
    await asyncio.sleep(0.01)
    assert not ordinary_task.done()
    plain_records = _pi_turn_records(3, 2) + [_pi_completion_record(2)]
    await _post_content_length(endpoint, plain_records)
    ordinary = await ordinary_task
    assert ordinary.status == "succeeded"

    second_turns = _pi_turn_records(4, 3)
    second_completion = _pi_completion_record(3)
    second_records = second_turns + [second_completion]
    second = runtime.invoke_stream(
        request=RunRequest(input="next turn", request_id="request-pi-second")
    )
    assert await _wait_for(second_started)
    second_collection = asyncio.create_task(_collect_records(second))
    await _post_content_length(endpoint, second_turns)
    second_release.set()
    second_result = asyncio.create_task(second.result())
    await asyncio.sleep(0.01)
    assert not second_result.done()
    await _post_content_length(endpoint, [second_completion])

    assert await second_collection == second_records
    assert (await second_result).status == "succeeded"
    assert mock_native.invoke_runtime.call_count == 3
    await runtime.stop()


async def test_pi_stream_failure_preserves_boundary_until_native_finishes(
    native_client: Fabric,
    mock_native: MagicMock,
):
    first_started = threading.Event()
    first_release = threading.Event()
    first_returned = threading.Event()
    second_started = threading.Event()
    second_release = threading.Event()

    def invoke(plan_json: str, runtime_json: str, request_json: str) -> str:
        request = json.loads(request_json)
        if request["request_id"] == "request-pi-first":
            first_started.set()
            assert first_release.wait(timeout=2)
            first_returned.set()
        elif request["request_id"] == "request-pi-second":
            second_started.set()
            assert second_release.wait(timeout=2)
        result = _result(request, json.loads(runtime_json))
        result["metadata"] = {
            "adapter": {
                "pi_turn_started": True,
                "pi_turn_count": (
                    1 if request["request_id"] == "request-pi-first" else 2
                ),
            }
        }
        return json.dumps(result)

    mock_native.invoke_runtime.side_effect = invoke
    runtime = await native_client.start_runtime(
        _config(relay=True, adapter_id="nvidia.fabric.pi"),
        streaming=True,
    )
    endpoint = json.loads(mock_native.plan_config.call_args.args[0])["relay"][
        "observability"
    ]["atof"]["sinks"][-1]["url"]
    assert runtime._collector_client is not None
    collector_stream = runtime._collector_client.stream

    async def interrupted_records(
        _: str,
        *,
        registration_token: str | None = None,
    ):
        raise FabricRuntimeError(
            "Relay output was interrupted",
            stage="invoke",
            code="collector_request_failed",
        )
        yield {}

    runtime._collector_client.stream = interrupted_records
    first = runtime.invoke_stream(
        request=RunRequest(input="first", request_id="request-pi-first")
    )
    assert await _wait_for(first_started)
    with pytest.raises(FabricRuntimeError, match="Relay output was interrupted"):
        await first.__anext__()
    assert first._finalized

    runtime._collector_client.stream = collector_stream
    first_release.set()
    assert await _wait_for(first_returned)
    async with asyncio.timeout(2):
        while runtime._current_task is not None:
            await asyncio.sleep(0)
    assert not first._task.done()
    with pytest.raises(FabricStateError, match="streaming invocation is active"):
        await runtime.stop()
    with pytest.raises(FabricStateError, match="streaming invocation is active"):
        runtime.invoke_stream(
            request=RunRequest(input="second", request_id="request-pi-second")
        )
    first_result = asyncio.create_task(first.result())
    await asyncio.sleep(0.01)
    assert not first_result.done()
    assert not second_started.is_set()

    old_records = _pi_turn_records(1, 0) + [_pi_completion_record(0)]
    await _post_content_length(endpoint, old_records)
    assert (await first_result).status == "succeeded"
    second = runtime.invoke_stream(
        request=RunRequest(input="second", request_id="request-pi-second")
    )
    assert await _wait_for(second_started)

    second_records = _pi_turn_records(2, 1) + [_pi_completion_record(1)]
    second_collection = asyncio.create_task(_collect_records(second))
    await _post_content_length(endpoint, second_records)
    second_release.set()

    assert await second_collection == second_records
    assert (await second.result()).status == "succeeded"
    await runtime.stop()


@pytest.mark.parametrize(
    "error_code",
    [
        "pi_prompt_rejected",
        "pi_unsupported_input",
        "pi_aborted",
        "pi_model_error",
        "pi_no_assistant_response",
        "pi_extension_shutdown",
    ],
)
async def test_pi_no_agent_run_result_releases_plain_invoke_boundary(
    native_client: Fabric,
    mock_native: MagicMock,
    error_code: str,
):
    def no_agent_run(plan_json: str, runtime_json: str, request_json: str) -> str:
        result = _result(json.loads(request_json), json.loads(runtime_json))
        result.update(
            {
                "status": (
                    "cancelled"
                    if error_code in {"pi_aborted", "pi_extension_shutdown"}
                    else "failed"
                ),
                "output": None,
                "error": {
                    "code": error_code,
                    "message": "Pi stopped before starting an agent run",
                    "retryable": False,
                },
                "metadata": {
                    "adapter": {"pi_turn_started": False, "pi_turn_count": 0}
                },
            }
        )
        return json.dumps(result)

    mock_native.invoke_runtime.side_effect = no_agent_run
    runtime = await native_client.start_runtime(
        _config(relay=True, adapter_id="nvidia.fabric.pi"),
        streaming=True,
    )

    first = await runtime.invoke(input="first")
    second = await runtime.invoke(input="second")

    assert first.error is not None
    assert first.error.code == error_code
    assert second.error is not None
    assert second.error.code == error_code
    await runtime.stop()


async def test_pi_result_without_turn_signal_quarantines_telemetry_only(
    native_client: Fabric,
    mock_native: MagicMock,
):
    def no_agent_run(plan_json: str, runtime_json: str, request_json: str) -> str:
        result = _result(json.loads(request_json), json.loads(runtime_json))
        result.update(
            {
                "status": "failed",
                "output": None,
                "error": {
                    "code": "pi_unsupported_input",
                    "message": "Pi stopped before starting an agent run",
                    "retryable": False,
                },
            }
        )
        return json.dumps(result)

    mock_native.invoke_runtime.side_effect = no_agent_run
    runtime = await native_client.start_runtime(
        _config(relay=True, adapter_id="nvidia.fabric.pi"),
        streaming=True,
        completion_wait_timeout=0.01,
    )

    first = await runtime.invoke(input="first")
    second = await runtime.invoke(input="second")

    assert first.error is not None
    assert first.error.code == "pi_unsupported_input"
    assert second.error is not None
    assert second.error.code == "pi_unsupported_input"
    await runtime.stop()


@pytest.mark.parametrize(
    ("status", "error_code"),
    [
        ("succeeded", None),
        ("cancelled", "pi_aborted"),
        ("failed", "pi_model_error"),
        ("failed", "pi_no_assistant_response"),
        ("cancelled", "pi_extension_shutdown"),
    ],
)
async def test_pi_missing_completion_does_not_block_later_invocations(
    native_client: Fabric,
    mock_native: MagicMock,
    status: str,
    error_code: str | None,
):
    turn_count = 0

    def invoke(plan_json: str, runtime_json: str, request_json: str) -> str:
        nonlocal turn_count
        turn_count += 1
        result = _result(json.loads(request_json), json.loads(runtime_json))
        result["metadata"] = {
            "adapter": {
                "pi_turn_started": True,
                "pi_turn_count": turn_count,
            }
        }
        if error_code is not None:
            result.update(
                {
                    "status": status,
                    "output": None,
                    "error": {
                        "code": error_code,
                        "message": "Pi stopped after starting an agent run",
                        "retryable": False,
                    },
                }
            )
        return json.dumps(result)

    mock_native.invoke_runtime.side_effect = invoke
    runtime = await native_client.start_runtime(
        _config(relay=True, adapter_id="nvidia.fabric.pi"),
        streaming=True,
        completion_wait_timeout=0.01,
    )

    stream = runtime.invoke_stream(input="first")
    assert [record async for record in stream] == []
    assert (await stream.result()).status == status

    for prompt in ("second", "third", "fourth"):
        assert (await runtime.invoke(input=prompt)).status == status

    assert mock_native.invoke_runtime.call_count == 4
    assert runtime.status.value == "active"
    await runtime.stop()


async def test_failed_stream_cleanup_compensates_before_next_invoke(
    native_client: Fabric,
    mock_native: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    def no_agent_run(plan_json: str, runtime_json: str, request_json: str) -> str:
        result = _result(json.loads(request_json), json.loads(runtime_json))
        result.update(
            {
                "status": "failed",
                "output": None,
                "error": {
                    "code": "pi_unsupported_input",
                    "message": "Pi stopped before starting an agent run",
                    "retryable": False,
                },
                "metadata": {
                    "adapter": {"pi_turn_started": False, "pi_turn_count": 0}
                },
            }
        )
        return json.dumps(result)

    mock_native.invoke_runtime.side_effect = no_agent_run
    runtime = await native_client.start_runtime(
        _config(relay=True, adapter_id="nvidia.fabric.pi"),
        streaming=True,
    )
    assert runtime._collector_client is not None
    deregister = runtime._collector_client.deregister
    outcome_failed = False

    async def fail_first_outcome_cleanup(
        request_id: str,
        *,
        remove_queue: bool,
        pi_boundary: str | None = None,
        registration_token: str | None = None,
    ) -> None:
        nonlocal outcome_failed
        if not remove_queue and not outcome_failed:
            outcome_failed = True
            raise FabricRuntimeError(
                "collector unavailable",
                stage="invoke",
                code="collector_request_failed",
            )
        await deregister(
            request_id,
            remove_queue=remove_queue,
            pi_boundary=pi_boundary,
            registration_token=registration_token,
        )

    mock_deregister = AsyncMock(side_effect=fail_first_outcome_cleanup)
    runtime._collector_client.deregister = mock_deregister
    monkeypatch.setattr(streaming_mod, "_FINALIZE_DRAIN_TIMEOUT_SECONDS", 0.01)

    stream = runtime.invoke_stream(input="first")
    async with asyncio.timeout(2):
        assert [record async for record in stream] == []
    with pytest.raises(FabricRuntimeError, match="collector unavailable"):
        await stream.result()

    assert [
        call.kwargs["pi_boundary"] for call in mock_deregister.await_args_list[:3]
    ] == ["release", "preserve", "quarantine"]
    assert runtime._registered_requests == set()
    second = await runtime.invoke(input="second")
    assert second.error is not None
    assert second.error.code == "pi_unsupported_input"
    await runtime.stop()


async def test_invoke_stream_preserves_invocation_failure_and_cleans_up(
    native_client: Fabric,
    mock_native: MagicMock,
):
    mock_native.invoke_runtime.side_effect = RuntimeError("Pi invocation failed")
    runtime = await native_client.start_runtime(
        _config(relay=True, adapter_id="nvidia.fabric.pi"),
        streaming=True,
    )

    stream = runtime.invoke_stream(input="fail")

    assert [record async for record in stream] == []
    with pytest.raises(FabricRuntimeError, match="Pi invocation failed"):
        await stream.result()
    assert stream._finalized
    assert runtime._registered_requests == set()
    assert runtime.status.value == "failed"

    await runtime.stop()
    assert runtime.status.value == "stopped"


async def test_invoke_stream_registration_failure_cleans_up(
    native_client: Fabric,
):
    runtime = await native_client.start_runtime(
        _config(relay=True, adapter_id="nvidia.fabric.pi"),
        streaming=True,
    )
    assert runtime._collector_client is not None
    runtime._collector_client.register = AsyncMock(
        side_effect=FabricRuntimeError(
            "registration failed",
            stage="invoke",
            code="collector_request_failed",
        )
    )

    stream = runtime.invoke_stream(input="hello")

    assert [record async for record in stream] == []
    with pytest.raises(FabricRuntimeError, match="registration failed"):
        await stream.result()
    assert stream._finalized
    assert runtime._registered_requests == set()

    await runtime.stop()


async def test_interrupted_collector_stream_preserves_terminal_result():
    registration_ready = asyncio.Event()
    registration_ready.set()
    mock_collector = MagicMock()
    mock_finalize = AsyncMock()

    async def interrupted_records(
        _: str,
        *,
        registration_token: str | None = None,
    ):
        assert registration_token is None
        raise FabricRuntimeError(
            "Relay output was interrupted",
            stage="invoke",
            code="collector_request_failed",
        )
        yield {}

    async def invoke() -> RunResult:
        return RunResult.from_mapping(_result({"request_id": "request-1"}, _runtime()))

    mock_collector.stream.side_effect = interrupted_records
    stream = InvokeStream(
        invoke(),
        mock_collector,
        request_id="request-1",
        registration_ready=registration_ready,
        on_finalize=mock_finalize,
    )

    with pytest.raises(FabricRuntimeError, match="Relay output was interrupted"):
        await stream.__anext__()

    assert stream._finalized
    assert (await stream.result()).status == "succeeded"
    mock_finalize.assert_awaited_once()


async def test_aclose_waits_for_pending_invocation_after_stream_failure():
    registration_ready = asyncio.Event()
    registration_ready.set()
    finish_invocation = asyncio.Event()
    mock_collector = MagicMock()
    mock_finalize = AsyncMock()

    async def interrupted_records(
        _: str,
        *,
        registration_token: str | None = None,
    ):
        assert registration_token is None
        raise FabricRuntimeError(
            "Relay output was interrupted",
            stage="invoke",
            code="collector_request_failed",
        )
        yield {}

    async def invoke() -> RunResult:
        await finish_invocation.wait()
        return RunResult.from_mapping(_result({"request_id": "request-1"}, _runtime()))

    mock_collector.stream.side_effect = interrupted_records
    stream = InvokeStream(
        invoke(),
        mock_collector,
        request_id="request-1",
        registration_ready=registration_ready,
        on_finalize=mock_finalize,
    )
    with pytest.raises(FabricRuntimeError, match="Relay output was interrupted"):
        await stream.__anext__()
    assert stream._finalized

    closing = asyncio.create_task(stream.aclose())
    await asyncio.sleep(0)
    assert not closing.done()

    finish_invocation.set()
    await closing
    assert (await stream.result()).status == "succeeded"
    mock_finalize.assert_awaited_once()


async def test_cancelled_record_close_still_runs_stream_finalizer():
    registration_ready = asyncio.Event()
    registration_ready.set()
    close_started = asyncio.Event()
    never_close = asyncio.Event()
    mock_collector = MagicMock()
    mock_finalize = AsyncMock()

    async def invoke() -> RunResult:
        return RunResult.from_mapping(_result({"request_id": "request-1"}, _runtime()))

    async def records():
        try:
            yield {"uuid": "record-1"}
        finally:
            close_started.set()
            await never_close.wait()

    stream = InvokeStream(
        invoke(),
        mock_collector,
        request_id="request-1",
        registration_ready=registration_ready,
        on_finalize=mock_finalize,
    )
    stream._records = records()
    assert await anext(stream._records) == {"uuid": "record-1"}

    finishing = asyncio.create_task(stream._finish_stream())
    await close_started.wait()
    finishing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await finishing

    assert stream._finalized
    mock_finalize.assert_awaited_once()
    assert (await stream.result()).status == "succeeded"


async def test_stream_finalizer_is_single_flight():
    registration_ready = asyncio.Event()
    registration_ready.set()
    finalize_started = asyncio.Event()
    finish_finalize = asyncio.Event()
    mock_collector = MagicMock()

    async def invoke() -> RunResult:
        return RunResult.from_mapping(_result({"request_id": "request-1"}, _runtime()))

    async def finalize() -> None:
        finalize_started.set()
        await finish_finalize.wait()

    mock_finalize = AsyncMock(side_effect=finalize)
    stream = InvokeStream(
        invoke(),
        mock_collector,
        request_id="request-1",
        registration_ready=registration_ready,
        on_finalize=mock_finalize,
    )

    first = asyncio.create_task(stream._finish_stream())
    await finalize_started.wait()
    second = asyncio.create_task(stream._finish_stream())
    await asyncio.sleep(0)
    mock_finalize.assert_awaited_once()

    finish_finalize.set()
    await asyncio.gather(first, second)

    assert stream._finalized
    mock_finalize.assert_awaited_once()
    assert (await stream.result()).status == "succeeded"


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


async def test_aclose_bounds_collector_drain_when_stream_never_terminates(
    monkeypatch: pytest.MonkeyPatch,
):
    never = asyncio.Event()
    registration_ready = asyncio.Event()
    registration_ready.set()
    mock_collector = MagicMock()

    async def collector_records(
        _: str,
        *,
        registration_token: str | None = None,
    ):
        assert registration_token is None
        await never.wait()
        yield {"uuid": "unreachable"}

    async def invoke() -> RunResult:
        return RunResult.from_mapping(_result({"request_id": "request-1"}, _runtime()))

    mock_collector.stream.side_effect = collector_records
    monkeypatch.setattr(streaming_mod, "_FINALIZE_DRAIN_TIMEOUT_SECONDS", 0.01)
    stream = InvokeStream(
        invoke(),
        mock_collector,
        request_id="request-1",
        registration_ready=registration_ready,
    )

    await asyncio.wait_for(stream.aclose(), timeout=0.1)

    assert stream._finalized


async def test_aclose_finalizes_when_collector_deregistration_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    never = asyncio.Event()
    registration_ready = asyncio.Event()
    registration_ready.set()
    mock_collector = MagicMock()
    mock_finalize = AsyncMock(side_effect=RuntimeError("deregister failed"))

    async def collector_records(
        _: str,
        *,
        registration_token: str | None = None,
    ):
        assert registration_token is None
        await never.wait()
        yield {"uuid": "unreachable"}

    async def invoke() -> RunResult:
        return RunResult.from_mapping(_result({"request_id": "request-1"}, _runtime()))

    mock_collector.stream.side_effect = collector_records
    monkeypatch.setattr(streaming_mod, "_FINALIZE_DRAIN_TIMEOUT_SECONDS", 0.01)
    stream = InvokeStream(
        invoke(),
        mock_collector,
        request_id="request-1",
        registration_ready=registration_ready,
        on_finalize=mock_finalize,
    )

    with pytest.raises(RuntimeError, match="deregister failed"):
        await asyncio.wait_for(stream.aclose(), timeout=0.1)

    assert stream._finalized
    mock_finalize.assert_awaited_once()


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

    async def collector_records(
        _: str,
        *,
        registration_token: str | None = None,
    ):
        assert registration_token is None
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

    async def collector_records(
        _: str,
        *,
        registration_token: str | None = None,
    ):
        assert registration_token is None
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
    records.put_nowait(record)
    shield = asyncio.shield

    async def cancel_after_record_is_read(task: asyncio.Future[dict[str, Any]]):
        assert await shield(task) == record
        raise asyncio.CancelledError

    with (
        patch.object(asyncio, "shield", new=cancel_after_record_is_read),
        pytest.raises(asyncio.CancelledError),
    ):
        await stream.__anext__()

    assert await stream.__anext__() == record

    invocation_finished.set()
    await records.put(None)
    await stream.aclose()


async def test_failed_invocation_drains_buffered_records_before_finalizing():
    record = {"uuid": "first"}

    async def collector_records(
        _: str,
        *,
        registration_token: str | None = None,
    ):
        assert registration_token is None
        yield record

    async def invoke() -> RunResult:
        raise FabricRuntimeError(
            "outcome cleanup failed",
            stage="invoke",
            code="collector_request_failed",
        )

    registration_ready = asyncio.Event()
    registration_ready.set()
    mock_collector = MagicMock()
    mock_collector.stream.side_effect = collector_records
    mock_finalize = AsyncMock()
    stream = InvokeStream(
        invoke(),
        mock_collector,
        request_id="request-1",
        registration_ready=registration_ready,
        on_finalize=mock_finalize,
    )
    await asyncio.wait({stream._task})

    assert [item async for item in stream] == [record]
    with pytest.raises(FabricRuntimeError, match="outcome cleanup failed"):
        await stream.result()
    mock_finalize.assert_awaited_once_with()


async def test_failed_invocation_reuses_buffer_drain_deadline():
    records: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def collector_records(
        _: str,
        *,
        registration_token: str | None = None,
    ):
        assert registration_token is None
        while record := await records.get():
            yield record

    async def invoke() -> RunResult:
        raise FabricRuntimeError(
            "outcome cleanup failed",
            stage="invoke",
            code="collector_request_failed",
        )

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
    await asyncio.wait({stream._task})
    records.put_nowait({"uuid": "first"})
    records.put_nowait({"uuid": "second"})
    records.put_nowait(None)

    assert await stream.__anext__() == {"uuid": "first"}
    deadline = stream._failed_invocation_drain_deadline
    assert deadline is not None
    assert await stream.__anext__() == {"uuid": "second"}
    assert stream._failed_invocation_drain_deadline == deadline
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()
