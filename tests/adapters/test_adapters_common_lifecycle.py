# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import os
from typing import Any

from nemo_fabric_adapters.common import lifecycle


def _request(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": lifecycle.CONTRACT_VERSION,
        "operation": operation,
        "payload": payload,
    }


def test_lifecycle_host_orders_one_runtime_and_preserves_adapter_results():
    runtime_id = "runtime-1"
    requests = [
        _request("start", {"runtime": {"runtime_id": runtime_id}}),
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
    input_stream = io.StringIO("".join(f"{json.dumps(item)}\n" for item in requests))
    output_stream = io.StringIO()
    seen: list[str] = []

    def run(payload: dict[str, Any]) -> dict[str, Any]:
        prompt = payload["request"]["input"]
        seen.append(prompt)
        return {"failed": prompt == "second", "response": prompt}

    lifecycle.serve(run, input_stream=input_stream, output_stream=output_stream)

    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert [item["operation"] for item in responses] == [
        "start",
        "invoke",
        "invoke",
        "stop",
    ]
    assert all(item["outcome"]["status"] == "succeeded" for item in responses)
    assert responses[2]["outcome"]["output"] == {
        "failed": True,
        "response": "second",
    }
    assert seen == ["first", "second"]


def test_lifecycle_host_rejects_runtime_mismatch_without_invoking_adapter():
    requests = [
        _request("start", {"runtime": {"runtime_id": "runtime-1"}}),
        _request(
            "invoke",
            {
                "runtime_context": {"runtime_id": "runtime-2"},
                "request": {"input": "do not run"},
            },
        ),
        _request("stop", {"runtime_id": "runtime-1"}),
    ]
    input_stream = io.StringIO("".join(f"{json.dumps(item)}\n" for item in requests))
    output_stream = io.StringIO()

    lifecycle.serve(
        lambda payload: (_ for _ in ()).throw(AssertionError(payload)),
        input_stream=input_stream,
        output_stream=output_stream,
    )

    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert responses[1]["outcome"]["status"] == "failed"
    assert responses[1]["outcome"]["error"]["code"] == "lifecycle_runtime_mismatch"


def test_lifecycle_host_keeps_adapter_stdout_out_of_protocol(capsys):
    runtime_id = "runtime-1"
    requests = [
        _request("start", {"runtime": {"runtime_id": runtime_id}}),
        _request(
            "invoke",
            {
                "runtime_context": {"runtime_id": runtime_id},
                "request": {"input": "hello"},
            },
        ),
        _request("stop", {"runtime_id": runtime_id}),
    ]
    input_stream = io.StringIO("".join(f"{json.dumps(item)}\n" for item in requests))
    output_stream = io.StringIO()

    def run(_payload: dict[str, Any]) -> dict[str, Any]:
        print("adapter diagnostic")
        return {"failed": False}

    lifecycle.serve(run, input_stream=input_stream, output_stream=output_stream)

    assert "adapter diagnostic" not in output_stream.getvalue()
    assert "adapter diagnostic" in capsys.readouterr().err


def test_lifecycle_host_scopes_invocation_telemetry_environment(monkeypatch):
    runtime_id = "runtime-1"
    variable = "FABRIC_TEST_LIFECYCLE_ENV"
    monkeypatch.setenv(variable, "host-value")
    requests = [
        _request("start", {"runtime": {"runtime_id": runtime_id}}),
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
    input_stream = io.StringIO("".join(f"{json.dumps(item)}\n" for item in requests))
    output_stream = io.StringIO()

    lifecycle.serve(
        lambda _payload: {"value": os.environ[variable]},
        input_stream=input_stream,
        output_stream=output_stream,
    )

    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert responses[1]["outcome"]["output"] == {"value": "invocation-value"}
    assert os.environ[variable] == "host-value"
