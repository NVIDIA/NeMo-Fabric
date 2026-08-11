# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import io
import json
import os
from typing import Any

import pytest
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric.openai_streaming import (
    _END,
    _OpenAIStreamListener,
    _ProtocolError,
    _validate_openai_chunk,
)


def _request(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "operation": operation,
        "payload": payload,
    }


def _streams(requests: list[dict[str, Any]]) -> tuple[io.StringIO, io.StringIO]:
    input_stream = io.StringIO("".join(f"{json.dumps(item)}\n" for item in requests))
    return input_stream, io.StringIO()


class _BackpressuredStreamWriter:
    def __init__(self) -> None:
        self.parts: list[bytes] = []
        self.drain_calls = 0
        self.first_drain_started = asyncio.Event()
        self.release_first_drain = asyncio.Event()
        self.closed = False

    def write(self, data: bytes) -> None:
        self.parts.append(data)

    async def drain(self) -> None:
        self.drain_calls += 1
        if self.drain_calls == 1:
            self.first_drain_started.set()
            await self.release_first_drain.wait()

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass


def _openai_stream_payload(
    listener: _OpenAIStreamListener,
    *,
    runtime_id: str = "runtime-1",
    invocation_id: str = "invocation-1",
    request_id: str = "request-1",
) -> dict[str, Any]:
    return {
        "runtime_context": {
            "runtime_id": runtime_id,
            "invocation_id": invocation_id,
            "request_id": request_id,
        },
        "request": {"request_id": request_id, "input": "hello"},
        "stream": {
            "protocol_version": "fabric.openai_stream/v1alpha1",
            "profile": "openai.chat_completions.chunk/v1",
            "host": "127.0.0.1",
            **listener.transport,
            "runtime_id": runtime_id,
            "invocation_id": invocation_id,
            "request_id": request_id,
        },
    }


async def test_lifecycle_host_streams_openai_chunks_out_of_band():
    listener = _OpenAIStreamListener(runtime_id="runtime-1", request_id="request-1")
    await listener.start()
    stream_payload = _openai_stream_payload(listener)
    input_stream, output_stream = _streams(
        [
            _request("start", {"runtime_context": {"runtime_id": "runtime-1"}}),
            _request("invoke_openai_stream", stream_payload),
            _request("stop", {"runtime_id": "runtime-1"}),
        ]
    )
    received_payloads = []

    class Runtime:
        async def start(self, _payload):
            pass

        async def invoke(self, _payload):
            raise AssertionError("ordinary invoke is not expected")

        async def invoke_openai_stream(self, payload, emit):
            received_payloads.append(payload)
            await emit(
                {
                    "id": "chunk-1",
                    "object": "chat.completion.chunk",
                    "created": (1 << 64) - 1,
                    "model": "test-model",
                    "choices": [{"index": 0, "delta": {"content": "hel"}}],
                }
            )
            await emit(
                {
                    "id": "chunk-2",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "test-model",
                    "choices": [{"index": 0, "delta": {"content": "lo"}}],
                }
            )
            return {"response": "hello"}

        async def stop(self):
            pass

    try:
        await lifecycle._serve(
            Runtime,
            config_loader=None,
            input_stream=input_stream,
            output_stream=output_stream,
        )
        records = [
            await asyncio.wait_for(listener.records.get(), timeout=1),
            await asyncio.wait_for(listener.records.get(), timeout=1),
            await asyncio.wait_for(listener.records.get(), timeout=1),
        ]
    finally:
        await listener.close()

    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert [response["operation"] for response in responses] == [
        "start",
        "invoke_openai_stream",
        "stop",
    ]
    assert responses[1]["outcome"] == {
        "status": "succeeded",
        "output": {"response": "hello"},
    }
    assert [record["id"] for record in records[:2]] == ["chunk-1", "chunk-2"]
    assert records[2] is _END
    assert received_payloads == [
        {
            "runtime_context": stream_payload["runtime_context"],
            "request": stream_payload["request"],
        }
    ]


async def test_openai_stream_writer_serializes_concurrent_emits_under_backpressure():
    transport = _BackpressuredStreamWriter()
    writer = lifecycle._OpenAIStreamWriter(
        asyncio.StreamReader(),
        transport,
        {
            "runtime_id": "runtime-1",
            "invocation_id": "invocation-1",
            "request_id": "request-1",
        },
    )

    def chunk(identifier: str) -> dict[str, Any]:
        return {
            "id": identifier,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": [],
        }

    first = asyncio.create_task(writer.emit(chunk("chunk-1")))
    await asyncio.wait_for(transport.first_drain_started.wait(), timeout=1)
    second = asyncio.create_task(writer.emit(chunk("chunk-2")))
    await asyncio.sleep(0)
    transport.release_first_drain.set()
    await asyncio.gather(first, second)

    records = [json.loads(part) for part in transport.parts if part.startswith(b"{")]
    assert [record["sequence"] for record in records] == [0, 1]
    assert [record["chunk"]["id"] for record in records] == ["chunk-1", "chunk-2"]


async def test_openai_stream_writer_serializes_finish_after_an_inflight_emit():
    reader = asyncio.StreamReader()
    reader.feed_data(b"HTTP/1.1 200 OK\r\n\r\n")
    reader.feed_eof()
    transport = _BackpressuredStreamWriter()
    writer = lifecycle._OpenAIStreamWriter(
        reader,
        transport,
        {
            "runtime_id": "runtime-1",
            "invocation_id": "invocation-1",
            "request_id": "request-1",
        },
    )
    chunk = {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [],
    }

    emit = asyncio.create_task(writer.emit(chunk))
    await asyncio.wait_for(transport.first_drain_started.wait(), timeout=1)
    finish = asyncio.create_task(writer.finish())
    await asyncio.sleep(0)
    transport.release_first_drain.set()
    await asyncio.gather(emit, finish)

    records = [json.loads(part) for part in transport.parts if part.startswith(b"{")]
    assert [(record["type"], record["sequence"]) for record in records] == [
        ("chunk", 0),
        ("end", 1),
    ]
    assert transport.parts[-1] == b"0\r\n\r\n"
    assert transport.closed


async def test_openai_stream_connect_preserves_cancellation_during_cleanup(
    monkeypatch: pytest.MonkeyPatch,
):
    listener = _OpenAIStreamListener(runtime_id="runtime-1", request_id="request-1")
    await listener.start()

    class CancellingWriter:
        def __init__(self) -> None:
            self.closed = False

        def write(self, _data: bytes) -> None:
            pass

        async def drain(self) -> None:
            raise asyncio.CancelledError

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            raise OSError("secondary cleanup failure")

    writer = CancellingWriter()

    async def open_connection(*_args, **_kwargs):
        return asyncio.StreamReader(), writer

    monkeypatch.setattr(asyncio, "open_connection", open_connection)
    try:
        with pytest.raises(asyncio.CancelledError):
            await lifecycle._OpenAIStreamWriter.connect(
                _openai_stream_payload(listener)
            )
    finally:
        await listener.close()

    assert writer.closed


@pytest.mark.parametrize(
    "primary_error",
    [
        lifecycle.LifecycleError("adapter_failure", "adapter failed"),
        asyncio.CancelledError(),
    ],
)
async def test_openai_stream_preserves_adapter_failure_over_finish_failure(
    monkeypatch: pytest.MonkeyPatch,
    primary_error: BaseException,
):
    finish_error = lifecycle.LifecycleError("finish_failure", "finish failed")

    class FailingWriter:
        async def emit(self, _chunk) -> None:
            pass

        async def finish(self) -> None:
            raise finish_error

    async def connect(_payload):
        return FailingWriter(), {
            "runtime_context": {"runtime_id": "runtime-1"},
            "request": {"input": "fail"},
        }

    monkeypatch.setattr(lifecycle._OpenAIStreamWriter, "connect", connect)

    class Runtime:
        async def invoke_openai_stream(self, _payload, _emit):
            raise primary_error

    with pytest.raises(type(primary_error)) as caught:
        await lifecycle._handle_invoke_openai_stream(
            lifecycle._HostState(),
            Runtime(),
            {},
        )

    if isinstance(primary_error, lifecycle.LifecycleError):
        assert caught.value.code == primary_error.code
    assert caught.value.__cause__ is finish_error


def test_lifecycle_host_rejects_unimplemented_openai_stream_without_poisoning_runtime():
    runtime_id = "runtime-1"
    payload = {
        "runtime_context": {
            "runtime_id": runtime_id,
            "invocation_id": "invocation-1",
            "request_id": "request-1",
        },
        "request": {"request_id": "request-1", "input": "hello"},
        "stream": {},
    }
    input_stream, output_stream = _streams(
        [
            _request("start", {"runtime_context": {"runtime_id": runtime_id}}),
            _request("invoke_openai_stream", payload),
            _request(
                "invoke",
                {
                    "runtime_context": {"runtime_id": runtime_id},
                    "request": {"input": "still works"},
                },
            ),
            _request("stop", {"runtime_id": runtime_id}),
        ]
    )

    class Runtime:
        async def start(self, _payload):
            pass

        async def invoke(self, payload):
            return {"input": payload["request"]["input"]}

        async def stop(self):
            pass

    lifecycle.serve(Runtime, input_stream=input_stream, output_stream=output_stream)

    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert responses[1]["outcome"]["error"] == {
        "stage": "invoke",
        "code": "lifecycle_openai_stream_unsupported",
        "message": "Adapter runtime does not implement OpenAI streaming",
        "retryable": False,
    }
    assert responses[2]["outcome"] == {
        "status": "succeeded",
        "output": {"input": "still works"},
    }


def _openai_profile_choice(**overrides: Any) -> dict[str, Any]:
    choice: dict[str, Any] = {"index": 0, "delta": {}}
    choice.update(overrides)
    return choice


def _openai_profile_chunk(**overrides: Any) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [],
    }
    chunk.update(overrides)
    return chunk


@pytest.mark.parametrize(
    "chunk",
    [
        {
            "id": "minimal",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": [],
        },
        {
            "id": "complete",
            "object": "chat.completion.chunk",
            "created": (1 << 64) - 1,
            "model": "test-model",
            "choices": [
                {
                    "index": (1 << 32) - 1,
                    "delta": {
                        "role": "assistant",
                        "content": "hello",
                        "refusal": None,
                        "function_call": {},
                        "tool_calls": [{}],
                    },
                    "finish_reason": "stop",
                    "logprobs": {},
                }
            ],
            "usage": {},
        },
    ],
)
def test_openai_chunk_validators_accept_shared_profile_fixtures(chunk):
    assert lifecycle._validated_openai_chunk(chunk) == chunk
    assert _validate_openai_chunk(chunk) == chunk


@pytest.mark.parametrize(
    "chunk",
    [
        _openai_profile_chunk(object="chat.completion"),
        _openai_profile_chunk(id="   "),
        _openai_profile_chunk(model="\t"),
        _openai_profile_chunk(created=False),
        _openai_profile_chunk(created=-1),
        _openai_profile_chunk(created="0"),
        _openai_profile_chunk(created=1 << 64),
        _openai_profile_chunk(choices={}),
        _openai_profile_chunk(choices=[None]),
        _openai_profile_chunk(choices=[_openai_profile_choice(index=False)]),
        _openai_profile_chunk(choices=[_openai_profile_choice(index=-1)]),
        _openai_profile_chunk(choices=[_openai_profile_choice(index="0")]),
        _openai_profile_chunk(choices=[_openai_profile_choice(index=1 << 32)]),
        _openai_profile_chunk(choices=[{"index": 0}]),
        _openai_profile_chunk(choices=[_openai_profile_choice(delta=[])]),
        _openai_profile_chunk(choices=[_openai_profile_choice(delta={"content": 1})]),
        _openai_profile_chunk(choices=[_openai_profile_choice(delta={"refusal": 1})]),
        _openai_profile_chunk(choices=[_openai_profile_choice(delta={"role": 1})]),
        _openai_profile_chunk(
            choices=[_openai_profile_choice(delta={"function_call": []})]
        ),
        _openai_profile_chunk(
            choices=[_openai_profile_choice(delta={"tool_calls": {}})]
        ),
        _openai_profile_chunk(
            choices=[_openai_profile_choice(delta={"tool_calls": [None]})]
        ),
        _openai_profile_chunk(choices=[_openai_profile_choice(finish_reason=1)]),
        _openai_profile_chunk(choices=[_openai_profile_choice(logprobs=[])]),
        _openai_profile_chunk(usage=[]),
    ],
)
def test_openai_chunk_validators_reject_shared_profile_fixtures(chunk):
    with pytest.raises(lifecycle.LifecycleError) as caught:
        lifecycle._validated_openai_chunk(chunk)

    assert caught.value.code == "lifecycle_invalid_openai_stream_event"
    with pytest.raises(_ProtocolError):
        _validate_openai_chunk(chunk)


def test_malformed_openai_stream_request_uses_the_invoke_error_stage():
    runtime_id = "runtime-1"
    input_stream, output_stream = _streams(
        [
            _request("start", {"runtime_context": {"runtime_id": runtime_id}}),
            _request(
                "invoke_openai_stream",
                {"runtime_context": ["not", "a", "mapping"]},
            ),
            _request("stop", {"runtime_id": runtime_id}),
        ]
    )

    class Runtime:
        async def start(self, _payload):
            pass

        async def invoke(self, _payload):
            pass

        async def stop(self):
            pass

    lifecycle.serve(Runtime, input_stream=input_stream, output_stream=output_stream)

    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert responses[1]["outcome"]["error"] == {
        "stage": "invoke",
        "code": "lifecycle_invalid_request",
        "message": "Invalid lifecycle request",
        "retryable": False,
    }


def test_lifecycle_host_reuses_one_runtime_and_one_event_loop():
    runtime_id = "runtime-1"
    input_stream, output_stream = _streams(
        [
            _request(
                "start",
                {"runtime_context": {"runtime_id": runtime_id}},
            ),
            _request(
                "invoke",
                {
                    "runtime_context": {"runtime_id": runtime_id},
                    "request": {"input": "first"},
                },
            ),
            _request(
                "invoke",
                {
                    "runtime_context": {"runtime_id": runtime_id},
                    "request": {"input": "second"},
                },
            ),
            _request("stop", {"runtime_id": runtime_id}),
        ]
    )
    instances = []

    class Runtime:
        def __init__(self):
            self.loop_ids: list[int] = []
            self.invocations = 0
            instances.append(self)

        async def start(self, _payload):
            self.loop_ids.append(id(asyncio.get_running_loop()))

        async def invoke(self, payload):
            self.loop_ids.append(id(asyncio.get_running_loop()))
            self.invocations += 1
            return {
                "count": self.invocations,
                "input": payload["request"]["input"],
            }

        async def stop(self):
            self.loop_ids.append(id(asyncio.get_running_loop()))

    lifecycle.serve(Runtime, input_stream=input_stream, output_stream=output_stream)

    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert [item["operation"] for item in responses] == [
        "start",
        "invoke",
        "invoke",
        "stop",
    ]
    assert all(item["outcome"]["status"] == "succeeded" for item in responses)
    assert all(set(item) == {"operation", "outcome"} for item in responses)
    assert responses[1]["outcome"]["output"] == {"count": 1, "input": "first"}
    assert responses[2]["outcome"]["output"] == {"count": 2, "input": "second"}
    assert len(instances) == 1
    assert len(set(instances[0].loop_ids)) == 1


def test_lifecycle_host_passes_minimal_invoke_payload_unchanged():
    runtime_id = "runtime-1"
    start_payload = {
        "agent_name": "agent",
        "base_dir": "/workspace",
        "config": {"harness": {"settings": {"mode": "retained"}}},
        "runtime_context": {
            "runtime_id": runtime_id,
            "invocation_id": "runtime-start",
        },
        "capability_plan": {"native": ["tools"]},
    }
    invoke_payload = {
        "runtime_context": {
            "runtime_id": runtime_id,
            "invocation_id": "invocation-1",
        },
        "request": {"input": "hello"},
    }
    input_stream, output_stream = _streams(
        [
            _request("start", start_payload),
            _request("invoke", invoke_payload),
            _request("stop", {"runtime_id": runtime_id}),
        ]
    )
    invocations = []

    class Runtime:
        async def start(self, _payload) -> None:
            pass

        async def invoke(self, payload) -> dict[str, str]:
            invocations.append(payload)
            return {"input": payload["request"]["input"]}

        async def stop(self) -> None:
            pass

    lifecycle.serve(Runtime, input_stream=input_stream, output_stream=output_stream)

    assert invocations == [invoke_payload]


def test_lifecycle_host_validates_opt_in_typed_config_before_adapter_start():
    runtime_id = "runtime-1"
    input_stream, output_stream = _streams(
        [
            _request(
                "start",
                {
                    "config": {"harness": {"settings": {"profile": "typed"}}},
                    "runtime_context": {"runtime_id": runtime_id},
                },
            ),
            _request("stop", {"runtime_id": runtime_id}),
        ]
    )
    starts: list[AgentConfig] = []

    class Runtime:
        async def start(self, payload) -> None:
            starts.append(payload["config"])

        async def invoke(self, _payload):
            raise AssertionError("invoke is not expected")

        async def stop(self) -> None:
            pass

    lifecycle.serve(
        Runtime,
        config_loader=AgentConfig.from_mapping,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    assert len(starts) == 1
    assert isinstance(starts[0], AgentConfig)
    assert starts[0].harness.settings == {"profile": "typed"}


def test_lifecycle_host_rejects_invalid_opt_in_config_before_runtime_creation():
    input_stream, output_stream = _streams(
        [
            _request(
                "start",
                {
                    "config": {"unknown": True},
                    "runtime_context": {"runtime_id": "runtime-1"},
                },
            )
        ]
    )
    created = 0

    class Runtime:
        def __init__(self) -> None:
            nonlocal created
            created += 1

        async def start(self, _payload) -> None:
            pass

        async def invoke(self, _payload):
            raise AssertionError("invoke is not expected")

        async def stop(self) -> None:
            pass

    lifecycle.serve(
        Runtime,
        config_loader=AgentConfig.from_mapping,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    response = json.loads(output_stream.getvalue())
    assert response["outcome"]["error"]["code"] == "lifecycle_invalid_config"
    assert created == 0


def test_lifecycle_host_rejects_runtime_mismatch_without_poisoning_runtime():
    input_stream, output_stream = _streams(
        [
            _request("start", {"runtime_context": {"runtime_id": "runtime-1"}}),
            _request(
                "invoke",
                {
                    "runtime_context": {"runtime_id": "runtime-2"},
                    "request": {"input": "do not run"},
                },
            ),
            _request(
                "invoke",
                {
                    "runtime_context": {"runtime_id": "runtime-1"},
                    "request": {"input": "run"},
                },
            ),
            _request("stop", {"runtime_id": "runtime-1"}),
        ]
    )
    invocations = []

    class Runtime:
        async def start(self, _payload):
            pass

        async def invoke(self, payload):
            invocations.append(payload)
            return {"input": payload["request"]["input"]}

        async def stop(self):
            pass

    lifecycle.serve(Runtime, input_stream=input_stream, output_stream=output_stream)

    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert responses[1]["outcome"]["status"] == "failed"
    assert responses[1]["outcome"]["error"]["code"] == "lifecycle_runtime_mismatch"
    assert responses[2]["outcome"] == {
        "status": "succeeded",
        "output": {"input": "run"},
    }
    assert len(invocations) == 1


def test_lifecycle_host_keeps_adapter_stdout_out_of_protocol(capsys):
    runtime_id = "runtime-1"
    input_stream, output_stream = _streams(
        [
            _request("start", {"runtime_context": {"runtime_id": runtime_id}}),
            _request(
                "invoke",
                {
                    "runtime_context": {"runtime_id": runtime_id},
                    "request": {"input": "hello"},
                },
            ),
            _request("stop", {"runtime_id": runtime_id}),
        ]
    )

    class Runtime:
        async def start(self, _payload):
            pass

        async def invoke(self, _payload):
            print("adapter diagnostic")
            return {"failed": False}

        async def stop(self):
            pass

    lifecycle.serve(Runtime, input_stream=input_stream, output_stream=output_stream)

    assert "adapter diagnostic" not in output_stream.getvalue()
    assert "adapter diagnostic" in capsys.readouterr().err


def test_lifecycle_host_scopes_invocation_telemetry_environment():
    runtime_id = "runtime-1"
    variable = "FABRIC_TEST_LIFECYCLE_ENV"
    os.environ[variable] = "host-value"
    input_stream, output_stream = _streams(
        [
            _request("start", {"runtime_context": {"runtime_id": runtime_id}}),
            _request(
                "invoke",
                {
                    "runtime_context": {
                        "runtime_id": runtime_id,
                        "telemetry": {"env": {variable: "invocation-value"}},
                    },
                    "request": {"input": "hello"},
                },
            ),
            _request("stop", {"runtime_id": runtime_id}),
        ]
    )

    class Runtime:
        async def start(self, _payload):
            pass

        async def invoke(self, _payload):
            return {"value": os.environ[variable]}

        async def stop(self):
            pass

    lifecycle.serve(Runtime, input_stream=input_stream, output_stream=output_stream)

    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert responses[1]["outcome"]["output"] == {"value": "invocation-value"}
    assert os.environ[variable] == "host-value"


def test_lifecycle_host_stops_runtime_when_fabric_closes_stdin():
    input_stream, output_stream = _streams(
        [_request("start", {"runtime_context": {"runtime_id": "runtime-1"}})]
    )
    stopped = []

    class Runtime:
        async def start(self, _payload):
            pass

        async def invoke(self, _payload):
            return None

        async def stop(self):
            stopped.append(True)

    lifecycle.serve(Runtime, input_stream=input_stream, output_stream=output_stream)

    assert stopped == [True]


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        (RuntimeError("adapter failed"), "lifecycle_adapter_invoke_failed"),
        (
            lifecycle.LifecycleError("adapter_known_failure", "Adapter failed"),
            "adapter_known_failure",
        ),
    ],
)
def test_lifecycle_host_rejects_invoke_after_adapter_failure(failure, expected_code):
    runtime_id = "runtime-1"
    invoke_payload = {
        "runtime_context": {"runtime_id": runtime_id},
        "request": {"input": "fail"},
    }
    input_stream, output_stream = _streams(
        [
            _request("start", {"runtime_context": {"runtime_id": runtime_id}}),
            _request("invoke", invoke_payload),
            _request("invoke", invoke_payload),
            _request("stop", {"runtime_id": runtime_id}),
        ]
    )
    invocations = []

    class Runtime:
        async def start(self, _payload):
            pass

        async def invoke(self, payload):
            invocations.append(payload)
            raise failure

        async def stop(self) -> None:
            pass

    lifecycle.serve(Runtime, input_stream=input_stream, output_stream=output_stream)

    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert responses[1]["outcome"]["error"]["code"] == expected_code
    assert responses[2]["outcome"]["error"]["code"] == "lifecycle_runtime_failed"
    assert len(invocations) == 1


def test_lifecycle_host_cleans_up_and_exits_after_start_failure():
    runtime_id = "runtime-1"
    start = _request("start", {"runtime_context": {"runtime_id": runtime_id}})
    input_stream, output_stream = _streams([start, start])
    stopped = []

    class Runtime:
        async def start(self, _payload):
            raise RuntimeError("start failed")

        async def invoke(self, _payload):
            raise AssertionError("failed runtime must not be invoked")

        async def stop(self):
            stopped.append(True)

    lifecycle.serve(Runtime, input_stream=input_stream, output_stream=output_stream)

    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert len(responses) == 1
    assert responses[0]["outcome"]["error"]["code"] == (
        "lifecycle_adapter_start_failed"
    )
    assert stopped == [True]
