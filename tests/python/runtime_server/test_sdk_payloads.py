# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The command bridge preserves the public SDK's request and result payloads."""

import json
import sys

import pytest
from nemo_fabric import RunRequest, RuntimeHandle
from nemo_fabric.runtime_server import cli, protocol


async def test_start_returns_the_complete_sdk_runtime_handle(
    host, host_request, mock_runtime
):
    mock_runtime.handle = RuntimeHandle.from_mapping(
        {
            "runtime_id": "runtime-1",
            "runtime_binding": "opaque-binding",
            "agent_name": "researcher",
            "harness": "test",
            "adapter_kind": "local_process",
            "adapter_id": "test.host",
            "environment": {"workspace": "workspace"},
        }
    )
    response = await host.handle(host_request("start"))
    assert response["result"] == mock_runtime.handle.to_mapping()


async def test_invoke_forwards_complete_sdk_request_without_translation(
    host, host_request, mock_runtime
):
    await host.handle(host_request("start"))
    request = RunRequest(
        input=["review", 42],
        request_id="caller-request",
        relay_session_root="caller-session",
        context={"repository": "sample"},
        overrides={"models": {"default": {"temperature": 0.2}}},
        caller_extension={"tag": "review"},
    )
    response = await host.handle(host_request("invoke", request=request.to_mapping()))
    assert response["status"] == "succeeded"
    forwarded = mock_runtime.invoke.await_args.kwargs
    assert set(forwarded) == {"request"}
    assert isinstance(forwarded["request"], RunRequest)
    assert forwarded["request"].to_mapping() == request.to_mapping()
    assert (
        response["result"] == mock_runtime.invoke.return_value.to_mapping.return_value
    )


@pytest.mark.parametrize(
    "value", ["review", ["review", 42], 42, False, None, {"stream": True}]
)
async def test_input_convenience_has_sdk_json_semantics(
    host, host_request, mock_runtime, value
):
    await host.handle(host_request("start"))
    response = await host.handle(host_request("invoke", input=value))
    assert response["status"] == "succeeded"
    mock_runtime.invoke.assert_awaited_once_with(input=value)


async def test_stop_returns_the_sdk_null_result(host, host_request, mock_runtime):
    await host.handle(host_request("start"))
    response = await host.handle(host_request("stop"))
    assert response["status"] == "succeeded"
    assert response["result"] is None
    assert (await host.handle(host_request("stop")))["result"] is None
    mock_runtime.stop.assert_awaited_once_with()


@pytest.mark.parametrize("output", [None, {"response": "partial"}])
async def test_sdk_failed_results_do_not_become_transport_errors(
    host, host_request, mock_runtime, output
):
    await host.handle(host_request("start"))
    result = {
        **mock_runtime.invoke.return_value.to_mapping.return_value,
        "status": "failed",
        "output": output,
        "error": {"code": "timeout", "message": "Deadline reached"},
        "artifacts": {
            "artifacts": [{"name": "partial", "path": "partial.txt", "kind": "log"}]
        },
    }
    mock_runtime.invoke.return_value.to_mapping.return_value = result
    response = await host.handle(host_request("invoke", input="review"))
    assert response["status"] == "succeeded"
    assert response["error"] is None
    assert response["result"] == result


@pytest.mark.parametrize("output", [None, {"response": "partial"}])
@pytest.mark.skipif(sys.platform == "win32", reason="Unix socket transport")
async def test_failed_sdk_result_survives_socket_validation(
    host, host_request, mock_runtime, socket_path, output
):
    import asyncio
    from nemo_fabric.runtime_server import transport

    result = {
        **mock_runtime.invoke.return_value.to_mapping.return_value,
        "status": "failed",
        "output": output,
        "error": {"code": "timeout", "message": "Deadline reached"},
    }
    mock_runtime.invoke.return_value.to_mapping.return_value = result
    stop, ready = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(
        transport.serve(
            host, socket_path=socket_path, stop_event=stop, ready_event=ready
        )
    )
    try:
        await asyncio.wait_for(ready.wait(), 1)
        started = await transport.client(host_request("start"), socket_path=socket_path)
        assert started["result"] == mock_runtime.handle.to_mapping()
        invoked = await transport.client(
            host_request("invoke", input="review"), socket_path=socket_path
        )
        assert invoked["status"] == "succeeded"
        assert invoked["error"] is None
        assert invoked["result"] == result
        stopped = await transport.client(host_request("stop"), socket_path=socket_path)
        assert stopped["result"] is None
    finally:
        stop.set()
        await asyncio.wait_for(task, 2)


@pytest.mark.parametrize("value", ["review", [1, 2], 42, False, None])
@pytest.mark.skipif(sys.platform == "win32", reason="Unix command bridge")
def test_cli_input_accepts_all_sdk_json_values(tmp_path, value):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(value))
    request = cli.parse_command(
        ["invoke", "--agent", "researcher", "--runtime-id", "r", "--input", str(path)]
    )
    assert request["input"] == value


@pytest.mark.skipif(sys.platform == "win32", reason="Unix command bridge")
def test_cli_reads_full_run_request(tmp_path):
    payload = RunRequest(
        input="review",
        request_id="caller-id",
        context={"tag": "first"},
        caller_extension=42,
    ).to_mapping()
    path = tmp_path / "request.json"
    path.write_text(json.dumps(payload))
    request = cli.parse_command(
        ["invoke", "--agent", "researcher", "--runtime-id", "r", "--request", str(path)]
    )
    assert request["request"] == payload


async def test_invalid_run_request_rejected_before_sdk_work(
    host, host_request, mock_runtime
):
    await host.handle(host_request("start"))
    response = await host.handle(host_request("invoke", request={"context": []}))
    assert response["error"]["effects"] == "none"
    assert response["changed"] is False
    mock_runtime.invoke.assert_not_awaited()


def test_input_and_request_are_mutually_exclusive(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("{}")
    with pytest.raises(protocol.ProtocolError):
        cli.parse_command(
            [
                "invoke",
                "--agent",
                "researcher",
                "--runtime-id",
                "r",
                "--input",
                str(path),
                "--request",
                str(path),
            ]
        )
