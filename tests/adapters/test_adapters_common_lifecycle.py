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


def _request(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "operation": operation,
        "payload": payload,
    }


def _streams(requests: list[dict[str, Any]]) -> tuple[io.StringIO, io.StringIO]:
    input_stream = io.StringIO("".join(f"{json.dumps(item)}\n" for item in requests))
    return input_stream, io.StringIO()


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
        async def start(self, _payload):
            pass

        async def invoke(self, payload):
            invocations.append(payload)
            return {"input": payload["request"]["input"]}

        async def stop(self):
            pass

    lifecycle.serve(Runtime, input_stream=input_stream, output_stream=output_stream)

    assert invocations == [invoke_payload]


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
