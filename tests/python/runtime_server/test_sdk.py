# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host ownership and routing using the real SDK and a controlled adapter."""

from unittest.mock import MagicMock

import pytest
from nemo_fabric import RuntimeHandle, _native
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
        assert caught.value.code == "runtime_busy"
        assert host.has_used_runtime
        assert host.runtimes[runtime_id].runtime is owned
        assert not invoked.done()
    finally:
        (tmp_path / "release").write_text("release")
        with suppress(Exception):
            await invoked
        await owned.stop()


async def test_real_sdk_supports_multiple_independent_runtimes_per_sandbox(
    blocking_adapter_config, tmp_path
):
    config = blocking_adapter_config
    name = config.metadata.name
    (tmp_path / "release").write_text("release")
    host = RuntimeServer(directory=tmp_path)

    def command(operation, **fields):
        request = {"interface_version": 2, "operation": operation, "agent": name}
        if operation == "start":
            request["config"] = config.to_mapping()
        return {**request, **fields}

    try:
        first = await host.handle(command("start"))
        assert first["status"] == "succeeded", first
        first_id = first["result"]["runtime_id"]
        assert RuntimeHandle.from_mapping(first["result"]).runtime_id == first_id
        second = await host.handle(command("start"))
        assert second["status"] == "succeeded", second
        second_id = second["result"]["runtime_id"]
        assert second_id != first_id
        assert set(host.runtimes) == {first_id, second_id}

        for runtime_id in (first_id, second_id):
            invoked = await host.handle(
                command("invoke", runtime_id=runtime_id, input={"message": runtime_id})
            )
            assert invoked["status"] == "succeeded", invoked
            assert invoked["result"]["runtime_id"] == runtime_id
            assert invoked["result"]["output"] == {"response": "released"}

        stopped = await host.handle(command("stop", runtime_id=first_id))
        assert stopped["status"] == "succeeded", stopped
        assert stopped["result"] is None
        still_usable = await host.handle(
            command("invoke", runtime_id=second_id, input={"message": "still alive"})
        )
        assert still_usable["status"] == "succeeded", still_usable
        assert still_usable["result"]["output"] == {"response": "released"}

        third = await host.handle(command("start"))
        assert third["status"] == "succeeded", third
        third_id = third["result"]["runtime_id"]
        assert third_id not in {first_id, second_id}
        assert set(host.runtimes) == {first_id, second_id, third_id}
        assert (await host.handle(command("stop", runtime_id=second_id)))[
            "status"
        ] == "succeeded"
        assert (await host.handle(command("stop", runtime_id=third_id)))[
            "status"
        ] == "succeeded"
    finally:
        if not host.start_uncertain:
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
        assert not host.has_used_runtime
        mock_start.assert_not_called()
    finally:
        await host.shutdown()
