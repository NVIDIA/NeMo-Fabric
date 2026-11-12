# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host ownership and routing using the real SDK and a controlled adapter."""

from unittest.mock import MagicMock

import pytest
import sys
import json
from nemo_fabric import CheckRequest, EnvironmentHandle, Fabric, RuntimeHandle, _native
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


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        ("start", "Fabric could not establish an active runtime."),
        ("invoke", "Fabric did not report a successful invocation."),
        ("stop", "Fabric could not complete the stop operation."),
    ],
)
async def test_real_sdk_failure_names_the_adapter_error_code(
    blocking_adapter_config, tmp_path, operation, message
):
    (tmp_path / "release").write_text("release")
    host = RuntimeServer(directory=tmp_path)
    request = {"interface_version": 2, "operation": operation, "agent": "blocking"}
    if operation == "start":
        request["config"] = blocking_adapter_config.to_mapping()
    else:
        started = await host.handle(
            {
                **request,
                "operation": "start",
                "config": blocking_adapter_config.to_mapping(),
            }
        )
        assert started["status"] == "succeeded", started
        request["runtime_id"] = started["result"]["runtime_id"]
        if operation == "invoke":
            request["input"] = {"message": "hello"}
    (tmp_path / f"fail-{operation}").write_text("test_adapter_failure")
    try:
        response = await host.handle(request)
    finally:
        (tmp_path / f"fail-{operation}").unlink()
        with suppress(Exception):
            await host.shutdown()
    assert response["status"] == "failed", response
    assert response["error"]["code"] == f"{operation}_failed"
    assert response["error"]["message"] == f"{message} Cause: test_adapter_failure."
    assert "PRIVATE" not in str(response)


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


@pytest.mark.skipif(sys.platform == "win32", reason="Unix OpenShell command bridge")
@pytest.mark.parametrize("retained_pipe", ["stdout", "stderr", "both"])
async def test_health_workers_recover_while_provider_descendants_hold_pipes(
    retained_pipe, blocking_adapter_config, server_sdk_bridge, tmp_path, monkeypatch
):
    host, _ = server_sdk_bridge
    config = blocking_adapter_config.to_mapping()
    config["environment"]["workspace"] = str(tmp_path)
    config["discovery"]["local_paths"] = [
        str(tmp_path / "blocking.fabric-adapter.json")
    ]
    config["runtime"]["artifacts"] = str(tmp_path / "artifacts")
    started = await host.handle(
        {
            "interface_version": 2,
            "operation": "start",
            "agent": "blocking",
            "config": config,
        }
    )
    assert started["status"] == "succeeded", json.dumps(started)
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
    holders = tmp_path / "holders"
    holders.mkdir()
    (holders / "mode").write_text(retained_pipe)
    monkeypatch.setenv("FABRIC_TEST_HEALTH_PIPE_DIR", str(holders))
    sentinel = tmp_path / "workspace-sentinel"
    sentinel.write_text("unchanged")
    environment = EnvironmentHandle.from_mapping(
        {
            "environment_id": "check-environment",
            "provider": "openshell",
            "control_location": "in_env_control",
            "ownership": "caller_owned",
            "connection": {
                "gateway": "https://fixture.invalid",
                "sandbox_name": "check-server",
                "sandbox_id": "immutable-check-server",
            },
        }
    )
    try:
        async with asyncio.timeout(5):
            while not (tmp_path / "entered").exists():
                await asyncio.sleep(0.01)
        reports = await asyncio.gather(
            *[
                Fabric().check(
                    environment=environment,
                    request=CheckRequest(scope="server", timeout_seconds=1.5),
                )
                for _ in range(4)
            ]
        )
        assert all(report.reason_code == "probe_timeout" for report in reports)
        assert len(list(holders.glob("*.holding"))) == 4
        (holders / "mode").unlink()
        report = await Fabric().check(
            environment=environment,
            request=CheckRequest(scope="server", timeout_seconds=1),
        )
        assert report.outcome == "passed", report.to_mapping()
        assert not (holders / "release").exists()
        assert host.runtimes[runtime_id].runtime is owned
        assert not invoked.done()
        assert sentinel.read_text() == "unchanged"
        (tmp_path / "release").write_text("release")
        result = await asyncio.wait_for(invoked, 5)
        assert result["status"] == "succeeded", result
        assert result["result"]["output"] == {"response": "released"}
    finally:
        (holders / "release").touch()
        (tmp_path / "release").write_text("release")
        with suppress(Exception):
            await asyncio.wait_for(invoked, 5)
        await owned.stop()


@pytest.mark.skipif(sys.platform == "win32", reason="Unix socket transport")
async def test_result_is_saved_and_retrievable_after_caller_timeout(
    blocking_adapter_config, tmp_path, socket_path
):
    from nemo_fabric import RunRequest
    from nemo_fabric.runtime_server import transport

    host = RuntimeServer(directory=tmp_path)
    command = {"interface_version": 2, "agent": "blocking"}
    started = await host.handle(
        {
            **command,
            "operation": "start",
            "config": blocking_adapter_config.to_mapping(),
        }
    )
    assert started["status"] == "succeeded", started
    command["runtime_id"] = started["result"]["runtime_id"]
    stop, ready = asyncio.Event(), asyncio.Event()
    server = asyncio.create_task(
        transport.serve(
            host, socket_path=socket_path, stop_event=stop, ready_event=ready
        )
    )
    try:
        await asyncio.wait_for(ready.wait(), 3)
        request = RunRequest(input="slow", request_id="recover-this-request")
        timeout = await transport.client(
            {**command, "operation": "invoke", "request": request.to_mapping()},
            socket_path=socket_path,
            timeout=0.1,
        )
        assert timeout["error"]["code"] == "invocation_timeout"
        assert timeout["error"]["effects"] == "unknown"
        assert timeout["result"] == {
            "runtime_id": command["runtime_id"],
            "request_id": request.request_id,
        }
        (tmp_path / "release").touch()
        async with asyncio.timeout(5):
            while True:
                response = await transport.client(
                    {
                        **command,
                        "operation": "result",
                        "request_id": request.request_id,
                    },
                    socket_path=socket_path,
                )
                assert response["status"] == "succeeded", response
                if response["result"] is not None:
                    break
                await asyncio.sleep(0.01)
        result = response["result"]
        assert result["request_id"] == request.request_id
        assert result["output"] == {"response": "released"}
        saved = (
            tmp_path
            / "artifacts"
            / ".fabric"
            / result["runtime_id"]
            / result["invocation_id"]
            / "run-result.json"
        )
        from nemo_fabric import RunResult

        assert (
            RunResult.from_mapping(json.loads(saved.read_text())).to_mapping() == result
        )
        assert saved.stat().st_mode & 0o077 == 0
        sdk_result = await host.fabric.invocation_result(
            agent_name="blocking",
            runtime_id=command["runtime_id"],
            request_id=request.request_id,
            artifacts=tmp_path / "artifacts",
        )
        assert sdk_result.to_mapping() == result
        assert (await host.handle({**command, "operation": "stop"}))[
            "status"
        ] == "succeeded"
        assert (
            await host.handle(
                {**command, "operation": "result", "request_id": request.request_id}
            )
        )["result"] == result
    finally:
        (tmp_path / "release").touch()
        stop.set()
        await asyncio.wait_for(server, 5)


async def test_cached_retry_preserves_later_sdk_history(
    blocking_adapter_config, tmp_path
):
    from nemo_fabric import RunRequest

    (tmp_path / "release").touch()
    script = tmp_path / "adapter.py"
    text = script.read_text().replace("harness=None", "history=[]\nharness=None")
    text = text.replace(
        "output={'status':'succeeded','output':{'response':'released'}}",
        "history.append({'role':'assistant','content':str(len(history)+1)})\n            output={'status':'succeeded','output':{'messages':history}}",
    )
    script.write_text(text)
    runtime = await Fabric().start_runtime(blocking_adapter_config, base_dir=tmp_path)
    try:
        request = RunRequest(input="A", request_id="request-a")
        first = await runtime.invoke(request=request)
        await runtime.invoke(request=RunRequest(input="B", request_id="request-b"))
        history = runtime.messages
        invocations = runtime.invocations
        cached = await runtime.invoke(request=request)
        assert cached.invocation_id == first.invocation_id
        assert len(history) == 2
        assert runtime.messages == history
        assert runtime.invocations == invocations
    finally:
        await runtime.stop()


async def test_lookup_before_first_invocation_returns_no_completed_result(
    blocking_adapter_config, tmp_path
):
    host = RuntimeServer(directory=tmp_path)
    command = {"interface_version": 2, "agent": "blocking"}
    started = await host.handle(
        {
            **command,
            "operation": "start",
            "config": blocking_adapter_config.to_mapping(),
        }
    )
    assert started["status"] == "succeeded", started
    try:
        response = await host.handle(
            {
                **command,
                "operation": "result",
                "runtime_id": started["result"]["runtime_id"],
                "request_id": "not-submitted",
            }
        )
        assert response["status"] == "succeeded", response
        assert response["result"] is None
        assert response["changed"] is False
        assert not (tmp_path / "entered").exists()
    finally:
        await host.shutdown()
