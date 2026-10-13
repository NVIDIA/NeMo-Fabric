# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host ownership and routing using the real SDK and a controlled adapter."""

from unittest.mock import MagicMock

import pytest
from nemo_fabric import RunRequest, RuntimeHandle, _native
from nemo_fabric.runtime_server.state import RuntimeServer
from nemo_fabric.runtime_server.protocol import ProtocolError
from _utils.configs import hermes_shim_config
import asyncio
from contextlib import suppress


async def test_real_sdk_busy_shutdown_requires_sandbox_cleanup(
    blocking_adapter_config, tmp_path
):
    host = RuntimeServer(directory=tmp_path)
    config = blocking_adapter_config.to_mapping()
    started = await host.handle(
        {
            "interface_version": 2,
            "operation": "start",
            "agent": "blocking",
            "config": config,
        }
    )
    assert started["status"] == "succeeded"
    runtime_id = started["result"]["runtime_id"]
    owned = host.runtimes[runtime_id].runtime
    invoked = asyncio.create_task(
        host.handle(
            {
                "interface_version": 2,
                "operation": "invoke",
                "agent": "blocking",
                "runtime_id": runtime_id,
                "input": {"message": "slow"},
            }
        )
    )
    try:
        async with asyncio.timeout(5):
            while not (tmp_path / "entered").exists():
                await asyncio.sleep(0.01)
        with pytest.raises(ProtocolError) as caught:
            await asyncio.wait_for(host.shutdown(), 3.5)
        assert caught.value.code == "cleanup_unconfirmed"
        assert host.lifecycle_used
        assert host.runtimes[runtime_id].runtime is owned
        assert not invoked.done()
    finally:
        (tmp_path / "release").write_text("release")
        with suppress(Exception):
            await invoked
        await owned.stop()


async def test_real_sdk_enforces_one_lifecycle_per_sandbox(
    hermes_shim_agent_dir,
):
    config = hermes_shim_config()
    name = config.metadata.name
    host = RuntimeServer(directory=hermes_shim_agent_dir)

    def command(operation, **fields):
        request = {"interface_version": 2, "operation": operation, "agent": name}
        if operation == "start":
            request["config"] = config.to_mapping()
        return {**request, **fields}

    try:
        first = await host.handle(command("start"))
        assert first["status"] == "succeeded", first
        rejected = await host.handle(command("start"))
        assert rejected["error"]["code"] == "runtime_exists"
        assert rejected["error"]["effects"] == "none"
        first_id = first["result"]["runtime_id"]
        assert RuntimeHandle.from_mapping(first["result"]).runtime_id == first_id
        for index, message in enumerate(("first review", "follow-up review")):
            runtime_id = first_id
            payload = {"input": {"message": message}}
            if index == 1:
                payload = {
                    "request": RunRequest(
                        input={"message": message},
                        request_id="caller-follow-up",
                        relay_session_root="00000000-0000-4000-8000-000000000042",
                        context={"review_tag": "cedar-4827"},
                        caller_extension={"turn": 2},
                    ).to_mapping()
                }
            invoked = await host.handle(
                command("invoke", runtime_id=runtime_id, **payload)
            )
            assert invoked["status"] == "succeeded", invoked
            assert invoked["result"]["runtime_id"] == runtime_id
            assert invoked["result"]["output"]["received"] == {"message": message}
            if index == 1:
                assert invoked["result"]["request_id"] == "caller-follow-up"
        stopped = await host.handle(command("stop", runtime_id=first_id))
        assert stopped["status"] == "succeeded", stopped
        assert stopped["result"] is None
        second = await host.handle(command("start"))
        assert second["error"]["code"] == "runtime_exists"
        assert len(host.runtimes) == 1
    finally:
        await host.shutdown()


@pytest.mark.parametrize("invalid_config", ["missing_adapter", "unsupported_setting"])
async def test_real_sdk_rejects_start_before_native_runtime_creation(
    hermes_shim_agent_dir, monkeypatch, invalid_config
):
    config = hermes_shim_config().to_mapping()
    host = RuntimeServer(directory=hermes_shim_agent_dir)
    request = {
        "interface_version": 2,
        "operation": "start",
        "agent": config["metadata"]["name"],
        "config": config,
    }
    if invalid_config == "missing_adapter":
        config["harness"]["adapter_id"] = "test.missing"
    else:
        config["harness"]["settings"] = {"unsupported_setting": True}
    mock_start = MagicMock()
    monkeypatch.setattr(_native, "start_runtime", mock_start)
    try:
        response = await host.handle(request)
        assert response["status"] == "failed", response
        assert response["error"]["code"] == "validation_failed"
        assert response["error"]["effects"] == "none"
        assert response["changed"] is False
        assert host.runtimes == {}
        assert not host.lifecycle_used
        mock_start.assert_not_called()
    finally:
        await host.shutdown()
