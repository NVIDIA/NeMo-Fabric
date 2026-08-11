# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for adapter-native OpenAI streaming."""

from __future__ import annotations

import asyncio
import inspect
import json
import socket
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from nemo_fabric import (
    Fabric,
    FabricCapabilityError,
    FabricRuntimeError,
    FabricStateError,
    RunRequest,
    Runtime,
    RuntimeStatus,
    openai_streaming,
)


def _plan(
    *,
    supports_openai_streaming: bool = True,
    adapter_supports_openai_streaming: bool | None = None,
) -> dict[str, Any]:
    if adapter_supports_openai_streaming is None:
        adapter_supports_openai_streaming = supports_openai_streaming
    return {
        "agent_name": "demo",
        "base_dir": ".",
        "config": {
            "metadata": {"name": "demo"},
            "harness": {"adapter_id": "test.fabric.shim"},
            "runtime": {},
        },
        "adapter_descriptor": {
            "descriptor": {
                "adapter_id": "test.fabric.shim",
                "harness": "shim",
                "adapter_kind": "python",
                "capabilities": {
                    "streaming": adapter_supports_openai_streaming,
                },
            }
        },
        "capabilities": {
            "service": False,
            "streaming": supports_openai_streaming,
            "updates": False,
            "cancellation": False,
        },
    }


def _runtime() -> dict[str, Any]:
    return {
        "runtime_id": "runtime-1",
        "runtime_binding": "fabric-runtime-binding-test",
        "agent_name": "demo",
        "harness": "shim",
        "adapter_kind": "python",
        "adapter_id": "test.fabric.shim",
        "environment": {
            "environment_id": "environment-1",
            "provider": "local",
            "control_location": "external_control",
            "ownership": "caller_owned",
        },
    }


def _result(
    request: dict[str, Any],
    runtime: dict[str, Any],
    *,
    invocation_id: str,
    failed: bool = False,
) -> dict[str, Any]:
    result = {
        "agent_name": "demo",
        "harness": "shim",
        "adapter_kind": "python",
        "adapter_id": "test.fabric.shim",
        "runtime_id": runtime["runtime_id"],
        "invocation_id": invocation_id,
        "request_id": request["request_id"],
        "status": "failed" if failed else "succeeded",
        "output": {"response": None if failed else "hello"},
        "artifacts": {"artifacts": []},
        "events": [],
    }
    if failed:
        result["error"] = {
            "stage": "invoke",
            "code": "shim_failed",
            "message": "shim reported failure",
            "retryable": False,
        }
    return result


def _chunk(identifier: str, content: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "object": "chat.completion.chunk",
        "created": (1 << 64) - 1,
        "model": "test-model",
        "choices": [{"index": 0, "delta": {"content": content}}],
    }


def _record(
    *,
    record_type: str = "chunk",
    sequence: Any = 0,
    runtime_id: str = "runtime-1",
    invocation_id: str = "invocation-1",
    request_id: str = "request-1",
    chunk: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "type": record_type,
        "sequence": sequence,
        "runtime_id": runtime_id,
        "invocation_id": invocation_id,
        "request_id": request_id,
    }
    if record_type == "chunk":
        record["chunk"] = chunk or _chunk("chunk-1", "hello")
    return record


def _read_http_status(stream: Any) -> int:
    status_line = stream.readline().decode("ascii")
    status = int(status_line.split(" ", 2)[1])
    while stream.readline() not in (b"\r\n", b"\n", b""):
        pass
    return status


async def _read_async_http_status(reader: asyncio.StreamReader) -> int:
    status_line = await reader.readline()
    status = int(status_line.decode("ascii").split(" ", 2)[1])
    while await reader.readline() not in (b"\r\n", b"\n", b""):
        pass
    return status


def _stream_request(transport: dict[str, Any], *, token: str | None = None) -> bytes:
    return (
        "POST /openai-stream HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{transport['port']}\r\n"
        f"Authorization: Bearer {token or transport['token']}\r\n"
        "Content-Type: application/x-ndjson\r\n"
        "Transfer-Encoding: chunked\r\n"
        "Expect: 100-continue\r\n"
        "Connection: close\r\n\r\n"
    ).encode("ascii")


def _send_stream(
    transport: dict[str, Any],
    *,
    runtime_id: str,
    invocation_id: str,
    request_id: str,
    chunks: list[dict[str, Any]],
    token: str | None = None,
) -> None:
    with socket.create_connection(("127.0.0.1", transport["port"]), timeout=2) as sock:
        sock.sendall(_stream_request(transport, token=token))
        response = sock.makefile("rb")
        status = _read_http_status(response)
        if status != 100:
            raise RuntimeError(f"listener rejected stream with HTTP {status}")
        records = [
            {
                "type": "chunk",
                "sequence": sequence,
                "runtime_id": runtime_id,
                "invocation_id": invocation_id,
                "request_id": request_id,
                "chunk": chunk,
            }
            for sequence, chunk in enumerate(chunks)
        ]
        records.append(
            {
                "type": "end",
                "sequence": len(chunks),
                "runtime_id": runtime_id,
                "invocation_id": invocation_id,
                "request_id": request_id,
            }
        )
        for record in records:
            encoded = json.dumps(record, separators=(",", ":")).encode() + b"\n"
            sock.sendall(f"{len(encoded):X}\r\n".encode() + encoded + b"\r\n")
        sock.sendall(b"0\r\n\r\n")
        assert _read_http_status(response) == 200


def _send_probe(transport: dict[str, Any], *, token: str) -> int:
    with socket.create_connection(("127.0.0.1", transport["port"]), timeout=2) as sock:
        sock.sendall(_stream_request(transport, token=token))
        return _read_http_status(sock.makefile("rb"))


def _send_records(transport: dict[str, Any], records: list[dict[str, Any]]) -> int:
    with socket.create_connection(("127.0.0.1", transport["port"]), timeout=2) as sock:
        sock.sendall(_stream_request(transport))
        response = sock.makefile("rb")
        assert _read_http_status(response) == 100
        for record in records:
            encoded = json.dumps(record, separators=(",", ":")).encode() + b"\n"
            sock.sendall(f"{len(encoded):X}\r\n".encode() + encoded + b"\r\n")
        sock.sendall(b"0\r\n\r\n")
        return _read_http_status(response)


@pytest.fixture(name="mock_native")
def mock_native_fixture() -> MagicMock:
    mock_native = MagicMock()
    mock_native.requests = []

    def invoke_openai_stream(
        _plan_json: str,
        runtime_json: str,
        request_json: str,
        transport_json: str,
    ) -> str:
        runtime = json.loads(runtime_json)
        request = json.loads(request_json)
        transport = json.loads(transport_json)
        mock_native.requests.append(request)
        invocation_id = f"invocation-{len(mock_native.requests)}"
        _send_stream(
            transport,
            runtime_id=runtime["runtime_id"],
            invocation_id=invocation_id,
            request_id=request["request_id"],
            chunks=[_chunk("chunk-1", "hel"), _chunk("chunk-2", "lo")],
        )
        return json.dumps(_result(request, runtime, invocation_id=invocation_id))

    def invoke(
        _plan_json: str,
        runtime_json: str,
        request_json: str,
    ) -> str:
        runtime = json.loads(runtime_json)
        request = json.loads(request_json)
        mock_native.requests.append(request)
        return json.dumps(
            _result(
                request,
                runtime,
                invocation_id=f"invocation-{len(mock_native.requests)}",
            )
        )

    mock_native.invoke_openai_stream.side_effect = invoke_openai_stream
    mock_native.invoke_runtime.side_effect = invoke
    mock_native.stop_runtime.return_value = "[]"
    return mock_native


def _runtime_wrapper(
    mock_native: MagicMock,
    *,
    supports_openai_streaming: bool = True,
    adapter_supports_openai_streaming: bool | None = None,
    relay_streaming: bool = False,
) -> Runtime:
    client = Fabric()
    client._native_module = lambda: mock_native  # type: ignore[method-assign]
    return Runtime(
        client=client,
        plan=_plan(
            supports_openai_streaming=supports_openai_streaming,
            adapter_supports_openai_streaming=adapter_supports_openai_streaming,
        ),
        runtime=_runtime(),
        stream_listener=MagicMock() if relay_streaming else None,
    )


async def test_invoke_openai_stream_yields_chunks_and_separate_result(mock_native):
    runtime = _runtime_wrapper(mock_native)

    stream = runtime.invoke_openai_stream(input="hello")
    chunks = [chunk async for chunk in stream]
    result = await stream.result()

    assert [chunk["id"] for chunk in chunks] == ["chunk-1", "chunk-2"]
    assert result.output["response"] == "hello"
    assert runtime.invocations == [
        {
            "request_id": result.request_id,
            "runtime_id": "runtime-1",
            "invocation_id": "invocation-1",
        }
    ]
    assert mock_native.invoke_openai_stream.call_count == 1


async def test_invoke_openai_stream_accepts_an_empty_stream(mock_native):
    def invoke_empty(_plan_json, runtime_json, request_json, transport_json):
        runtime = json.loads(runtime_json)
        request = json.loads(request_json)
        _send_stream(
            json.loads(transport_json),
            runtime_id=runtime["runtime_id"],
            invocation_id="invocation-empty",
            request_id=request["request_id"],
            chunks=[],
        )
        return json.dumps(
            _result(request, runtime, invocation_id="invocation-empty")
        )

    mock_native.invoke_openai_stream.side_effect = invoke_empty
    runtime = _runtime_wrapper(mock_native)

    stream = runtime.invoke_openai_stream(input="empty")

    assert [chunk async for chunk in stream] == []
    assert (await stream.result()).status == "succeeded"


async def test_openai_stream_aclose_drains_without_cancelling(mock_native):
    runtime = _runtime_wrapper(mock_native)
    stream = runtime.invoke_openai_stream(input="hello")

    first = await anext(stream)
    await stream.aclose()
    result = await stream.result()

    assert first["id"] == "chunk-1"
    assert result.output["response"] == "hello"
    assert mock_native.invoke_openai_stream.call_count == 1
    assert runtime.status is RuntimeStatus.ACTIVE


async def test_result_drains_a_stream_larger_than_the_bounded_queue(mock_native):
    def invoke_many(_plan_json, runtime_json, request_json, transport_json):
        runtime = json.loads(runtime_json)
        request = json.loads(request_json)
        _send_stream(
            json.loads(transport_json),
            runtime_id=runtime["runtime_id"],
            invocation_id="invocation-many",
            request_id=request["request_id"],
            chunks=[_chunk(f"chunk-{index}", "x") for index in range(1025)],
        )
        return json.dumps(
            _result(request, runtime, invocation_id="invocation-many")
        )

    mock_native.invoke_openai_stream.side_effect = invoke_many
    runtime = _runtime_wrapper(mock_native)

    result = await runtime.invoke_openai_stream(input="many").result()

    assert result.status == "succeeded"
    assert runtime.invocations[0]["invocation_id"] == "invocation-many"


async def test_successful_terminal_result_requires_an_explicit_stream_end(
    mock_native,
    monkeypatch,
):
    def invoke_without_stream(_plan_json, runtime_json, request_json, _transport_json):
        runtime = json.loads(runtime_json)
        request = json.loads(request_json)
        return json.dumps(
            _result(request, runtime, invocation_id="invocation-without-stream")
        )

    mock_native.invoke_openai_stream.side_effect = invoke_without_stream
    monkeypatch.setattr(
        openai_streaming,
        "_OPENAI_STREAM_COMPLETION_TIMEOUT",
        0.01,
    )
    runtime = _runtime_wrapper(mock_native)

    with pytest.raises(FabricRuntimeError, match="did not establish and complete"):
        await runtime.invoke_openai_stream(input="missing stream").result()

    assert runtime.status is RuntimeStatus.FAILED
    assert runtime.invocations == []


@pytest.mark.parametrize("iterate_first", [False, True])
async def test_failed_terminal_result_without_stream_preserves_failure(
    mock_native,
    iterate_first,
):
    def invoke_failed_without_stream(
        _plan_json,
        runtime_json,
        request_json,
        _transport_json,
    ):
        runtime = json.loads(runtime_json)
        request = json.loads(request_json)
        return json.dumps(
            _result(
                request,
                runtime,
                invocation_id="invocation-failed-before-stream",
                failed=True,
            )
        )

    mock_native.invoke_openai_stream.side_effect = invoke_failed_without_stream
    runtime = _runtime_wrapper(mock_native)

    stream = runtime.invoke_openai_stream(input="fail before stream")
    if iterate_first:
        assert [chunk async for chunk in stream] == []
    result = await stream.result()

    assert result.status == "failed"
    assert result.error.code == "shim_failed"
    assert result.error.message == "shim reported failure"
    assert runtime.status is RuntimeStatus.ACTIVE


async def test_unauthenticated_probe_does_not_poison_the_adapter_stream(mock_native):
    def invoke_after_probe(_plan_json, runtime_json, request_json, transport_json):
        runtime = json.loads(runtime_json)
        request = json.loads(request_json)
        transport = json.loads(transport_json)
        assert _send_probe(transport, token="wrong-token") == 401
        _send_stream(
            transport,
            runtime_id=runtime["runtime_id"],
            invocation_id="invocation-after-probe",
            request_id=request["request_id"],
            chunks=[_chunk("chunk-after-probe", "safe")],
        )
        return json.dumps(
            _result(request, runtime, invocation_id="invocation-after-probe")
        )

    mock_native.invoke_openai_stream.side_effect = invoke_after_probe
    runtime = _runtime_wrapper(mock_native)
    stream = runtime.invoke_openai_stream(input="probe")

    assert [chunk["id"] async for chunk in stream] == ["chunk-after-probe"]
    assert (await stream.result()).status == "succeeded"
    assert runtime.status is RuntimeStatus.ACTIVE


async def test_natural_iteration_waits_for_explicit_end_after_terminal_result(
    mock_native,
):
    senders: list[threading.Thread] = []

    def invoke_before_stream(_plan_json, runtime_json, request_json, transport_json):
        runtime = json.loads(runtime_json)
        request = json.loads(request_json)
        transport = json.loads(transport_json)

        def send_later() -> None:
            time.sleep(0.05)
            _send_stream(
                transport,
                runtime_id=runtime["runtime_id"],
                invocation_id="invocation-late-stream",
                request_id=request["request_id"],
                chunks=[_chunk("late-1", "late"), _chunk("late-2", " stream")],
            )

        sender = threading.Thread(target=send_later)
        sender.start()
        senders.append(sender)
        return json.dumps(
            _result(request, runtime, invocation_id="invocation-late-stream")
        )

    mock_native.invoke_openai_stream.side_effect = invoke_before_stream
    runtime = _runtime_wrapper(mock_native)
    stream = runtime.invoke_openai_stream(input="late")

    chunks = [chunk async for chunk in stream]
    result = await stream.result()
    for sender in senders:
        sender.join(timeout=2)

    assert [chunk["id"] for chunk in chunks] == ["late-1", "late-2"]
    assert result.invocation_id == "invocation-late-stream"
    assert runtime.status is RuntimeStatus.ACTIVE


async def test_cancelling_iteration_after_end_preserves_stream_completion(mock_native):
    def invoke_after_end(_plan_json, runtime_json, request_json, transport_json):
        runtime = json.loads(runtime_json)
        request = json.loads(request_json)
        _send_stream(
            json.loads(transport_json),
            runtime_id=runtime["runtime_id"],
            invocation_id="invocation-cancel-after-end",
            request_id=request["request_id"],
            chunks=[],
        )
        time.sleep(0.1)
        return json.dumps(
            _result(
                request,
                runtime,
                invocation_id="invocation-cancel-after-end",
            )
        )

    mock_native.invoke_openai_stream.side_effect = invoke_after_end
    runtime = _runtime_wrapper(mock_native)
    stream = runtime.invoke_openai_stream(input="cancel after end")
    iterator = asyncio.create_task(anext(stream))

    async def wait_for_end() -> None:
        while not stream._end_observed:
            await asyncio.sleep(0.005)

    try:
        await asyncio.wait_for(wait_for_end(), timeout=1)
        iterator.cancel()
        with pytest.raises(asyncio.CancelledError):
            await iterator
    finally:
        if not iterator.done():
            iterator.cancel()
            await asyncio.gather(iterator, return_exceptions=True)

    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert (await stream.result()).status == "succeeded"
    assert runtime.status is RuntimeStatus.ACTIVE


async def test_cancelling_result_during_cleanup_absorbs_terminal_result_once(
    mock_native,
):
    runtime = _runtime_wrapper(mock_native)
    stream = runtime.invoke_openai_stream(input="cancel cleanup")
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    original_close = stream._listener.close

    async def delayed_close() -> None:
        close_started.set()
        await allow_close.wait()
        await original_close()

    stream._listener.close = delayed_close
    first_result = asyncio.create_task(stream.result())
    await close_started.wait()
    first_result.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_result
    assert len(runtime.invocations) == 1

    allow_close.set()
    result = await stream.result()

    assert result.status == "succeeded"
    assert len(runtime.invocations) == 1


async def test_reported_failure_after_a_chunk_keeps_runtime_usable(mock_native):
    def invoke_failed(_plan_json, runtime_json, request_json, transport_json):
        runtime = json.loads(runtime_json)
        request = json.loads(request_json)
        _send_stream(
            json.loads(transport_json),
            runtime_id=runtime["runtime_id"],
            invocation_id="invocation-failed",
            request_id=request["request_id"],
            chunks=[_chunk("chunk-before-failure", "partial")],
        )
        return json.dumps(
            _result(
                request,
                runtime,
                invocation_id="invocation-failed",
                failed=True,
            )
        )

    mock_native.invoke_openai_stream.side_effect = invoke_failed
    runtime = _runtime_wrapper(mock_native)
    stream = runtime.invoke_openai_stream(input="fail")

    assert [chunk["id"] async for chunk in stream] == ["chunk-before-failure"]
    assert (await stream.result()).status == "failed"
    assert (await runtime.invoke(input="next")).status == "succeeded"
    assert runtime.status is RuntimeStatus.ACTIVE


async def test_mismatched_terminal_identity_is_not_absorbed(mock_native):
    def invoke_mismatch(_plan_json, runtime_json, request_json, transport_json):
        runtime = json.loads(runtime_json)
        request = json.loads(request_json)
        _send_stream(
            json.loads(transport_json),
            runtime_id=runtime["runtime_id"],
            invocation_id="invocation-stream",
            request_id=request["request_id"],
            chunks=[],
        )
        return json.dumps(
            _result(request, runtime, invocation_id="invocation-terminal")
        )

    mock_native.invoke_openai_stream.side_effect = invoke_mismatch
    runtime = _runtime_wrapper(mock_native)

    with pytest.raises(FabricRuntimeError, match="does not match its terminal result"):
        await runtime.invoke_openai_stream(input="mismatch").result()

    assert runtime.status is RuntimeStatus.FAILED
    assert runtime.invocations == []


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (_record(sequence=False), "sequence is not monotonic"),
        (_record(sequence=1), "sequence is not monotonic"),
        (_record(runtime_id="runtime-other"), "runtime ID does not match"),
        (_record(request_id="request-other"), "request ID does not match"),
        (
            _record(
                chunk={
                    "id": "missing-model",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "choices": [],
                }
            ),
            "model must be a non-empty string",
        ),
        (
            _record(
                chunk={
                    "id": "   ",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "test-model",
                    "choices": [],
                }
            ),
            "id must be a non-empty string",
        ),
        (
            _record(
                chunk={
                    "id": "blank-model",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "\t",
                    "choices": [],
                }
            ),
            "model must be a non-empty string",
        ),
        (
            _record(
                chunk={
                    "id": "created-overflow",
                    "object": "chat.completion.chunk",
                    "created": 1 << 64,
                    "model": "test-model",
                    "choices": [],
                }
            ),
            "created must be an unsigned 64-bit integer",
        ),
        (
            _record(
                chunk={
                    "id": "index-overflow",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "test-model",
                    "choices": [{"index": 1 << 32, "delta": {}}],
                }
            ),
            "index must be an unsigned 32-bit integer",
        ),
    ],
)
async def test_listener_rejects_invalid_record_invariants(record, message):
    listener = openai_streaming._OpenAIStreamListener(
        runtime_id="runtime-1",
        request_id="request-1",
    )

    with pytest.raises(openai_streaming._ProtocolError, match=message):
        await listener._emit_line(json.dumps(record).encode())


async def test_listener_rejects_malformed_oversized_and_changed_identity():
    listener = openai_streaming._OpenAIStreamListener(
        runtime_id="runtime-1",
        request_id="request-1",
    )

    with pytest.raises(openai_streaming._ProtocolError, match="not valid JSON"):
        await listener._emit_line(b"{")
    nonfinite = _record()
    nonfinite["chunk"]["extension"] = float("nan")
    with pytest.raises(openai_streaming._ProtocolError, match="not valid JSON"):
        await listener._emit_line(json.dumps(nonfinite).encode())

    listener = openai_streaming._OpenAIStreamListener(
        runtime_id="runtime-1",
        request_id="request-1",
        max_record_bytes=64,
    )
    with pytest.raises(openai_streaming._ProtocolError, match="exceeds 64 bytes"):
        await listener._emit_line(b"x" * 65)

    listener = openai_streaming._OpenAIStreamListener(
        runtime_id="runtime-1",
        request_id="request-1",
    )
    await listener._emit_line(json.dumps(_record()).encode())
    with pytest.raises(openai_streaming._ProtocolError, match="invocation ID changed"):
        await listener._emit_line(
            json.dumps(
                _record(
                    sequence=1,
                    invocation_id="invocation-other",
                    chunk=_chunk("chunk-2", "other"),
                )
            ).encode()
        )


async def test_listener_rejects_negative_chunk_size():
    listener = openai_streaming._OpenAIStreamListener(
        runtime_id="runtime-1",
        request_id="request-1",
    )
    reader = asyncio.StreamReader()
    reader.feed_data(b"-1\r\n")
    reader.feed_eof()

    with pytest.raises(
        openai_streaming._ProtocolError,
        match="Invalid OpenAI stream chunk size",
    ):
        await listener._read_chunked(reader, bytearray())


@pytest.mark.parametrize("delimiter", [b"\n", b"\r\n"])
async def test_listener_accepts_record_payload_at_byte_limit(delimiter):
    record = _record()
    encoded = json.dumps(record, separators=(",", ":")).encode()
    listener = openai_streaming._OpenAIStreamListener(
        runtime_id="runtime-1",
        request_id="request-1",
        max_record_bytes=len(encoded),
    )
    buffer = bytearray()

    if delimiter == b"\r\n":
        await listener._feed(buffer, encoded + b"\r")
        assert buffer == bytearray(encoded + b"\r")
        assert listener.records.empty()
        await listener._feed(buffer, b"\n")
    else:
        await listener._feed(buffer, encoded + delimiter)

    assert await listener.records.get() == record["chunk"]
    assert buffer == bytearray()


async def test_missing_end_record_fails_the_invocation(mock_native):
    def invoke_without_end(_plan_json, runtime_json, request_json, transport_json):
        runtime = json.loads(runtime_json)
        request = json.loads(request_json)
        status = _send_records(
            json.loads(transport_json),
            [
                _record(
                    runtime_id=runtime["runtime_id"],
                    invocation_id="invocation-missing-end",
                    request_id=request["request_id"],
                )
            ],
        )
        raise RuntimeError(f"stream listener rejected missing end with HTTP {status}")

    mock_native.invoke_openai_stream.side_effect = invoke_without_end
    runtime = _runtime_wrapper(mock_native)

    with pytest.raises(FabricRuntimeError, match="without an end record"):
        await runtime.invoke_openai_stream(input="missing end").result()

    assert runtime.status is RuntimeStatus.FAILED


async def test_claimed_invalid_chunk_reports_the_stable_protocol_error(mock_native):
    def invoke_invalid_chunk(_plan_json, runtime_json, request_json, transport_json):
        runtime = json.loads(runtime_json)
        request = json.loads(request_json)
        invalid_chunk = _chunk("invalid", "bad")
        invalid_chunk.pop("model")
        status = _send_records(
            json.loads(transport_json),
            [
                _record(
                    runtime_id=runtime["runtime_id"],
                    invocation_id="invocation-invalid-chunk",
                    request_id=request["request_id"],
                    chunk=invalid_chunk,
                )
            ],
        )
        raise RuntimeError(f"native writer failed with HTTP {status}")

    mock_native.invoke_openai_stream.side_effect = invoke_invalid_chunk
    runtime = _runtime_wrapper(mock_native)

    with pytest.raises(FabricRuntimeError) as caught:
        await runtime.invoke_openai_stream(input="invalid chunk").result()

    assert caught.value.code == "openai_stream_protocol_error"
    assert "model must be a non-empty string" in str(caught.value)
    assert runtime.status is RuntimeStatus.FAILED


async def test_listener_accepts_only_one_simultaneous_authenticated_connection():
    listener = openai_streaming._OpenAIStreamListener(
        runtime_id="runtime-1",
        request_id="request-1",
    )
    await listener.start()

    connections: list[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = []

    async def candidate():
        reader, writer = await asyncio.open_connection(
            "127.0.0.1",
            listener.transport["port"],
        )
        connections.append((reader, writer))
        writer.write(_stream_request(listener.transport))
        await writer.drain()
        return await _read_async_http_status(reader), reader, writer

    candidates = [asyncio.create_task(candidate()) for _ in range(2)]
    try:
        results = await asyncio.gather(*candidates)
        assert sorted(status for status, _reader, _writer in results) == [100, 409]
        status, reader, writer = next(item for item in results if item[0] == 100)
        assert status == 100
        encoded = json.dumps(
            _record(record_type="end"), separators=(",", ":")
        ).encode() + b"\n"
        writer.write(f"{len(encoded):X}\r\n".encode() + encoded + b"\r\n0\r\n\r\n")
        await writer.drain()
        assert await _read_async_http_status(reader) == 200
    finally:
        for candidate_task in candidates:
            if not candidate_task.done():
                candidate_task.cancel()
        await asyncio.gather(*candidates, return_exceptions=True)
        for _reader, writer in connections:
            writer.close()
            await writer.wait_closed()
        await listener.close()


async def test_listener_requires_an_exact_chunked_transfer_coding():
    listener = openai_streaming._OpenAIStreamListener(
        runtime_id="runtime-1",
        request_id="request-1",
    )
    await listener.start()
    reader, writer = await asyncio.open_connection(
        "127.0.0.1",
        listener.transport["port"],
    )
    request = _stream_request(listener.transport).replace(
        b"Transfer-Encoding: chunked",
        b"Transfer-Encoding: unchunked",
    )
    try:
        writer.write(request)
        await writer.drain()

        assert await _read_async_http_status(reader) == 411
        assert listener.error is None
    finally:
        writer.close()
        await writer.wait_closed()
        await listener.close()


@pytest.mark.parametrize(
    ("supports_openai_streaming", "supports_relay_streaming"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_native_and_relay_streaming_capabilities_are_independent(
    mock_native,
    supports_openai_streaming: bool,
    supports_relay_streaming: bool,
):
    runtime = _runtime_wrapper(
        mock_native,
        supports_openai_streaming=supports_openai_streaming,
        relay_streaming=supports_relay_streaming,
    )

    assert runtime.supports_openai_streaming is supports_openai_streaming
    assert runtime.supports_streaming is supports_relay_streaming


def test_invoke_openai_stream_rejects_an_unsupported_adapter(mock_native):
    runtime = _runtime_wrapper(mock_native, supports_openai_streaming=False)

    with pytest.raises(FabricCapabilityError) as caught:
        runtime.invoke_openai_stream(input="hello")

    assert caught.value.code == "openai_streaming_unavailable"
    mock_native.invoke_openai_stream.assert_not_called()


def test_openai_streaming_requires_the_resolved_descriptor_claim(mock_native):
    runtime = _runtime_wrapper(
        mock_native,
        supports_openai_streaming=True,
        adapter_supports_openai_streaming=False,
    )

    assert runtime.supports_openai_streaming is False
    with pytest.raises(FabricCapabilityError) as caught:
        runtime.invoke_openai_stream(input="hello")

    assert caught.value.code == "openai_streaming_unavailable"
    mock_native.invoke_openai_stream.assert_not_called()


def test_openai_stream_constructor_closes_run_when_task_creation_fails(monkeypatch):
    captured = []

    def fail_create_task(coroutine):
        captured.append(coroutine)
        raise RuntimeError("no running event loop")

    async def invoke(_transport):
        raise AssertionError("invoke must not run")

    monkeypatch.setattr(asyncio, "create_task", fail_create_task)

    with pytest.raises(RuntimeError, match="no running event loop"):
        openai_streaming.OpenAIInvokeStream(
            invoke,
            runtime_id="runtime-1",
            request_id="request-1",
        )

    assert len(captured) == 1
    assert inspect.getcoroutinestate(captured[0]) == inspect.CORO_CLOSED


async def test_invoke_openai_stream_rejects_concurrent_turns(mock_native):
    runtime = _runtime_wrapper(mock_native)
    stream = runtime.invoke_openai_stream(input="first")

    with pytest.raises(FabricStateError, match="streaming invocation is active"):
        runtime.invoke_openai_stream(input="second")
    with pytest.raises(FabricStateError, match="streaming invocation is active"):
        await runtime.invoke(input="second")

    await stream.aclose()


async def test_openai_stream_protocol_failure_marks_runtime_failed(mock_native):
    def invoke_with_bad_token(_plan_json, runtime_json, request_json, transport_json):
        runtime = json.loads(runtime_json)
        request = json.loads(request_json)
        _send_stream(
            json.loads(transport_json),
            runtime_id=runtime["runtime_id"],
            invocation_id="invocation-bad",
            request_id=request["request_id"],
            chunks=[],
            token="wrong-token",
        )
        raise AssertionError("listener must reject the token")

    mock_native.invoke_openai_stream.side_effect = invoke_with_bad_token
    runtime = _runtime_wrapper(mock_native)
    stream = runtime.invoke_openai_stream(request=RunRequest(input="hello"))

    with pytest.raises(FabricRuntimeError):
        await stream.result()

    assert runtime.status is RuntimeStatus.FAILED
    await stream.aclose()
