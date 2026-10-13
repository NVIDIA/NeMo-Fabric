# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One sandbox lifecycle and direct delegation to the existing SDK."""

import asyncio

import pytest
from nemo_fabric import FabricConfigError, RuntimeStatus
from nemo_fabric.runtime_server.protocol import ProtocolError


@pytest.mark.parametrize(
    "operation", ["check", "validate", "prepare", "configure", "serve"]
)
async def test_non_runtime_commands_cannot_mutate_runtime(
    host, host_request, mock_fabric, operation
):
    response = await host.handle(host_request(operation))
    assert response["status"] == "failed"
    assert response["error"]["effects"] == "none"
    mock_fabric.start_runtime.assert_not_awaited()


async def test_start_delegates_to_existing_sdk(
    host, host_request, mock_fabric, config, mock_runtime
):
    response = await host.handle(host_request("start"))
    assert response["status"] == "succeeded"
    assert response["result"] == mock_runtime.handle.to_mapping()
    args, kwargs = mock_fabric.start_runtime.call_args
    assert args[0].to_mapping()["metadata"]["name"] == config["metadata"]["name"]
    assert kwargs == {"base_dir": host.directory}
    mock_fabric.start_runtime.assert_awaited_once()


@pytest.mark.parametrize(
    "field", ["expected_descriptor_fingerprint", "expected_generation", "surprise"]
)
async def test_extra_start_fields_have_no_effects(
    host, host_request, mock_fabric, field
):
    rejected = await host.handle(host_request("start", **{field: "PRIVATE"}))
    assert rejected["error"]["code"] == "invalid_request"
    assert rejected["error"]["effects"] == "none"
    assert rejected["changed"] is False
    assert not host.lifecycle_used
    mock_fabric.start_runtime.assert_not_awaited()


@pytest.mark.parametrize("agent", ["researcher", "writer"])
async def test_second_start_preserves_ownership(
    host, host_request, mock_fabric, config, agent
):
    await host.handle(host_request("start"))
    request_config = {**config, "metadata": {"name": agent}}
    response = await host.handle(
        host_request("start", agent=agent, config=request_config)
    )
    assert response["error"]["code"] == "runtime_exists"
    assert response["error"]["effects"] == "none"
    mock_fabric.start_runtime.assert_awaited_once()
    assert list(host.runtimes) == ["runtime-1"]


async def test_competing_starts_call_sdk_once(
    host, host_request, mock_fabric, mock_runtime
):
    entered, release = asyncio.Event(), asyncio.Event()

    async def start(*args, **kwargs):
        entered.set()
        await release.wait()
        return mock_runtime

    mock_fabric.start_runtime.side_effect = start
    first = asyncio.create_task(host.handle(host_request("start")))
    await entered.wait()
    try:
        second = await asyncio.wait_for(host.handle(host_request("start")), 0.5)
        assert second["error"]["code"] == "runtime_exists"
    finally:
        release.set()
        assert (await first)["status"] == "succeeded"
    mock_fabric.start_runtime.assert_awaited_once()


async def test_invalid_configuration_does_not_consume_lifecycle(
    host, host_request, mock_fabric
):
    response = await host.handle(
        host_request(
            "start", config={"metadata": {"name": "researcher"}, "harness": {}}
        )
    )
    assert response["error"]["code"] == "validation_failed"
    assert response["error"]["effects"] == "none"
    assert not host.lifecycle_used
    mock_fabric.start_runtime.assert_not_awaited()


@pytest.mark.parametrize("code", ["invalid_config", "adapter_capability_unverified"])
async def test_sdk_preflight_rejection_does_not_consume_lifecycle(
    host, host_request, mock_fabric, code
):
    mock_fabric.start_runtime.side_effect = FabricConfigError("PRIVATE", code=code)
    response = await host.handle(host_request("start"))
    assert response["error"]["code"] == (
        "validation_unavailable"
        if code == "adapter_capability_unverified"
        else "validation_failed"
    )
    assert response["error"]["effects"] == "none"
    assert not host.lifecycle_used
    assert "PRIVATE" not in str(response)


async def test_uncertain_start_consumes_lifecycle_without_replay(
    host, host_request, mock_fabric
):
    mock_fabric.start_runtime.side_effect = RuntimeError("PRIVATE")
    response = await host.handle(host_request("start"))
    assert response["error"]["code"] == "start_failed"
    assert response["error"]["effects"] == "unknown"
    assert response["changed"] is None
    assert "PRIVATE" not in str(response)
    assert (await host.handle(host_request("start")))["error"][
        "code"
    ] == "runtime_exists"
    mock_fabric.start_runtime.assert_awaited_once()


async def test_stop_keeps_data_and_is_idempotent(host, host_request, mock_runtime):
    retained = host.directory / "retained"
    retained.write_text("KEEP")
    await host.handle(host_request("start"))
    response = await host.handle(host_request("stop"))
    assert response["status"] == "succeeded"
    assert response["changed"] is True
    assert response["result"] is None
    repeated = await host.handle(host_request("stop"))
    assert repeated["status"] == "succeeded"
    assert repeated["changed"] is False
    mock_runtime.stop.assert_awaited_once_with()
    assert retained.read_text() == "KEEP"
    assert (await host.handle(host_request("invoke", input={})))["error"][
        "code"
    ] == "runtime_unavailable"


async def test_failed_stop_keeps_ownership_and_rejects_replacement(
    host, host_request, mock_runtime, mock_fabric
):
    await host.handle(host_request("start"))
    mock_runtime.stop.side_effect = RuntimeError("PRIVATE")
    response = await host.handle(host_request("stop"))
    assert response["error"]["code"] == "stop_failed"
    assert response["error"]["effects"] == "unknown"
    assert response["changed"] is None
    assert "PRIVATE" not in str(response)
    assert host.runtimes["runtime-1"].runtime is mock_runtime
    assert host.runtimes["runtime-1"].state == "unknown"
    assert (await host.handle(host_request("start")))["error"][
        "code"
    ] == "runtime_exists"
    mock_fabric.start_runtime.assert_awaited_once()


async def test_invoke_preserves_normalized_result_without_replay(
    host, host_request, mock_runtime
):
    await host.handle(host_request("start"))
    response = await host.handle(host_request("invoke", input={"prompt": "review"}))
    assert response["status"] == "succeeded"
    assert response["changed"] is None
    assert (
        response["result"] == mock_runtime.invoke.return_value.to_mapping.return_value
    )
    mock_runtime.invoke.assert_awaited_once_with(input={"prompt": "review"})


@pytest.mark.parametrize(
    "fields",
    [
        {"runtime_id": "other"},
        {"agent": "other"},
        {"input": float("nan")},
        {"request": []},
    ],
)
async def test_invalid_invoke_cannot_reach_sdk(
    host, host_request, mock_runtime, fields
):
    await host.handle(host_request("start"))
    response = await host.handle(host_request("invoke", **{"input": {}, **fields}))
    assert response["status"] in ("failed", "unsupported")
    assert response["error"]["effects"] == "none"
    mock_runtime.invoke.assert_not_awaited()


async def test_wrong_stop_identity_cannot_reach_sdk(host, host_request, mock_runtime):
    await host.handle(host_request("start"))
    for fields in ({"runtime_id": "other"}, {"agent": "other"}):
        response = await host.handle(host_request("stop", **fields))
        assert response["error"]["effects"] == "none"
    mock_runtime.stop.assert_not_awaited()


async def test_failed_native_result_retains_error_and_null_output(
    host, host_request, mock_runtime
):
    await host.handle(host_request("start"))
    native = {"status": "failed", "output": None, "error": {"code": "timeout"}}
    mock_runtime.invoke.return_value.to_mapping.return_value = native
    response = await host.handle(host_request("invoke", input={}))
    assert response["error"] is None
    assert response["status"] == "succeeded"
    assert response["result"] == native
    mock_runtime.invoke.assert_awaited_once()


async def test_non_active_returned_runtime_is_owned_for_stop(
    host, host_request, mock_runtime
):
    mock_runtime.status = RuntimeStatus.FAILED
    response = await host.handle(host_request("start"))
    assert response["error"]["code"] == "start_failed"
    assert host.runtimes["runtime-1"].runtime is mock_runtime
    assert (await host.handle(host_request("start")))["error"][
        "code"
    ] == "runtime_exists"
    assert (await host.handle(host_request("stop")))["status"] == "succeeded"
    mock_runtime.stop.assert_awaited_once_with()


async def test_invoke_and_stop_are_serialized(host, host_request, mock_runtime):
    await host.handle(host_request("start"))
    entered, release = asyncio.Event(), asyncio.Event()
    native = mock_runtime.invoke.return_value

    async def invoke(**kwargs):
        entered.set()
        await release.wait()
        return native

    mock_runtime.invoke.side_effect = invoke
    first = asyncio.create_task(host.handle(host_request("invoke", input={})))
    await entered.wait()
    second = asyncio.create_task(host.handle(host_request("stop")))
    await asyncio.sleep(0)
    mock_runtime.stop.assert_not_awaited()
    release.set()
    assert (await first)["status"] == "succeeded"
    assert (await second)["status"] == "succeeded"
    mock_runtime.stop.assert_awaited_once_with()


async def test_busy_shutdown_defers_to_sandbox_supervisor(
    host, host_request, mock_runtime
):
    await host.handle(host_request("start"))
    entered, release = asyncio.Event(), asyncio.Event()
    native = mock_runtime.invoke.return_value

    async def invoke(**kwargs):
        entered.set()
        await release.wait()
        return native

    mock_runtime.invoke.side_effect = invoke
    task = asyncio.create_task(host.handle(host_request("invoke", input={})))
    await entered.wait()
    try:
        with pytest.raises(ProtocolError) as caught:
            await asyncio.wait_for(host.shutdown(), 0.5)
        assert caught.value.code == "cleanup_unconfirmed"
        assert (await host.handle(host_request("stop")))["error"][
            "code"
        ] == "host_stopping"
        mock_runtime.stop.assert_not_awaited()
    finally:
        release.set()
        await task


async def test_failed_shutdown_rejects_work(host, host_request, mock_runtime):
    await host.handle(host_request("start"))
    mock_runtime.stop.side_effect = RuntimeError("PRIVATE")
    with pytest.raises(RuntimeError):
        await host.shutdown()
    assert (await host.handle(host_request("invoke", input={})))["error"][
        "code"
    ] == "host_stopping"
    assert host.lifecycle_used
