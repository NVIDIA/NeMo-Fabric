# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Hermes health checks during a blocked local model request."""

import asyncio
from functools import partial
import os
from pathlib import Path
import signal
import sys
import threading

import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from _utils.mock_api_server import _stream_chat_completion
from nemo_fabric import (
    CheckRequest,
    EnvironmentConfig,
    Fabric,
    FabricConfig,
    HarnessConfig,
    MetadataConfig,
    ModelConfig,
    ToolsConfig,
)
from nemo_fabric_adapter_contract.models import AdapterCheckResult, AgentConfig
from nemo_fabric_adapters.hermes.adapter import HermesRuntime

pytestmark = pytest.mark.usefixtures("requires_hermes_agent")


@pytest_asyncio.fixture(name="health_model", loop_scope="function")
async def health_model_fixture(unused_tcp_port):
    app = FastAPI()
    entered, release = threading.Event(), threading.Event()
    requests = []

    @app.middleware("http")
    async def record(request: Request, call_next):
        requests.append((request.method, request.url.path))
        return await call_next(request)

    @app.post("/v1/chat/completions")
    async def complete(request: Request):
        payload = await request.json()
        entered.set()
        assert await asyncio.to_thread(release.wait, 30)
        if payload.get("stream"):
            return StreamingResponse(
                _stream_chat_completion(payload, "finished"),
                media_type="text/event-stream",
            )
        return JSONResponse(
            {
                "id": "chatcmpl-local-health",
                "object": "chat.completion",
                "created": 0,
                "model": "local-health-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "finished"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=unused_tcp_port,
            log_config=None,
            access_log=False,
            lifespan="off",
            ws="none",
        )
    )
    serving = threading.Thread(target=server.run, daemon=True)
    serving.start()
    try:
        async with asyncio.timeout(5):
            while not server.started:
                await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{unused_tcp_port}/v1", entered, release, requests
    finally:
        release.set()
        server.should_exit = True
        await asyncio.to_thread(serving.join, 5)
        assert not serving.is_alive()


def _config(endpoint, tmp_path):
    return FabricConfig(
        metadata=MetadataConfig(name="hermes-health"),
        harness=HarnessConfig(adapter_id="nvidia.fabric.hermes"),
        models={
            "default": ModelConfig(
                provider="openai",
                model="local-health-model",
                base_url=endpoint,
                api_key_env="FABRIC_NATIVE_HEALTH_TEST_KEY",
            )
        },
        environment=EnvironmentConfig(
            provider="local",
            workspace=tmp_path,
            env={
                "HOME": str(tmp_path),
                "HERMES_HOME": str(tmp_path / "hermes"),
                "FABRIC_NATIVE_HEALTH_TEST_KEY": "local-test-only",
            },
        ),
        tools=ToolsConfig(enabled=[]),
    )


async def test_hermes_sdk_health_during_invocation(health_model, tmp_path):
    endpoint, entered, release, requests = health_model
    os.environ["ADAPTER_PYTHON"] = sys.executable
    fabric = Fabric()
    config = _config(endpoint, tmp_path)
    environment = await fabric.prepare_environment(config, base_dir=tmp_path)
    runtime = None
    invocation = None
    try:
        runtime = await fabric.start_runtime_in(config, environment, base_dir=tmp_path)
        startup_requests = list(requests)
        assert ("POST", "/v1/chat/completions") not in startup_requests
        for phase in ("before", "during", "after"):
            if phase == "during":
                invocation = asyncio.create_task(
                    runtime.invoke(input="Return finished.")
                )
                assert await asyncio.to_thread(entered.wait, 30)
            elif phase == "after":
                release.set()
                result = await asyncio.wait_for(invocation, 30)
                assert result.status == "succeeded", result.to_mapping()
                assert result.output["response"] == "finished"
            files = [
                path
                for path in tmp_path.rglob("*")
                if path.is_file() and path.suffix in {".yaml", ".toml", ".jsonl"}
            ]
            before = {path: path.read_bytes() for path in files}
            provider_requests_before = list(requests)
            for level in ("live", "active", "ready"):
                report = await fabric.check(
                    environment=environment,
                    request=CheckRequest(
                        scope="runtime",
                        runtime_id=runtime.runtime_id,
                        level=level,
                        timeout_seconds=3,
                    ),
                )
                assert report.outcome == "passed", report.to_mapping()
                assert [item["name"] for item in report.checks] == [
                    "live",
                    "active",
                    "ready",
                ][: ["live", "active", "ready"].index(level) + 1]
                assert report.runtime["activity"] == (
                    "busy" if phase == "during" else "idle"
                )
                if phase == "during":
                    assert not invocation.done()
            assert before == {path: path.read_bytes() for path in files}
            assert requests == provider_requests_before
            assert [item for item in requests if item[1] == "/v1/chat/completions"] == (
                [] if phase == "before" else [("POST", "/v1/chat/completions")]
            )
    finally:
        release.set()
        if invocation is not None:
            await asyncio.gather(invocation, return_exceptions=True)
        if runtime is not None:
            await runtime.stop()
        await fabric.release_environment(environment)


async def test_hermes_native_session_health_and_closure(
    health_model, tmp_path, monkeypatch
):
    from agent import session_activity

    endpoint, _, _, requests = health_model
    os.environ["FABRIC_NATIVE_HEALTH_TEST_KEY"] = "local-test-only"
    os.environ["HOME"] = str(tmp_path)
    os.environ["HERMES_HOME"] = str(tmp_path / "hermes")
    config = _config(endpoint, tmp_path)
    native = HermesRuntime()
    await native.start(
        {
            "agent_name": "native-health",
            "base_dir": str(tmp_path),
            "config": AgentConfig.from_mapping(
                {"models": config.to_mapping()["models"], "tools": {"enabled": []}}
            ),
            "runtime_context": {
                "runtime_id": "native-health",
                "invocation_id": "native-health",
                "request_id": "native-health",
                "environment": {
                    "environment_id": "native-health",
                    "provider": "local",
                    "control_location": "external_control",
                    "ownership": "caller_owned",
                    "workspace": str(tmp_path),
                },
                "artifacts": {},
            },
        }
    )
    try:
        startup_requests = list(requests)
        assert ("POST", "/v1/chat/completions") not in startup_requests
        before = native._agent.get_activity_summary()

        # Advance only the snapshot clock, leaving stored activity untouched.
        monkeypatch.setattr(
            session_activity,
            "build_activity_snapshot",
            partial(
                session_activity.build_activity_snapshot,
                now=before["last_activity_at"] + before["seconds_since_activity"] + 1.0,
            ),
        )
        for level in ("live", "active", "ready"):
            async with asyncio.timeout(3):
                result = await native.check(level)
            AdapterCheckResult.from_mapping(result)
            assert (result["outcome"], result["activity"]) == ("passed", "idle")
        after = native._agent.get_activity_summary()
        assert after.pop("seconds_since_activity") > before.pop(
            "seconds_since_activity"
        )
        assert before == after
        assert native._conversation_history is None
        assert requests == startup_requests
        await asyncio.to_thread(native._agent.close)
        assert isinstance(native._agent.get_activity_summary(), dict)
        for level in ("live", "active", "ready"):
            result = await native.check(level)
            assert (result["outcome"], result["reason_code"]) == (
                "failed",
                "session_closed",
            )
        assert requests == startup_requests
    finally:
        await native.stop()


@pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"},
    reason="requires native process identity reads",
)
async def test_hermes_host_exit_does_not_hide_behind_a_responsive_runtime_server(
    health_model, tmp_path, server_sdk_bridge
):
    from nemo_fabric.runtime_server import transport

    endpoint, _, release, _ = health_model
    release.set()
    os.environ["ADAPTER_PYTHON"] = sys.executable
    host, _ = server_sdk_bridge
    config = _config(endpoint, tmp_path)
    started = await host.handle(
        {
            "interface_version": 2,
            "operation": "start",
            "agent": config.metadata.name,
            "config": config.model_dump(mode="json"),
        }
    )
    assert started["status"] == "succeeded", started
    runtime_id = started["result"]["runtime_id"]
    runtime = host.runtimes[runtime_id].runtime
    target = {"agent": config.metadata.name, "runtime_id": runtime_id}
    socket_path = Path(os.environ["FABRIC_HOST_SOCKET"])
    try:
        # An explicit local invocation supplies the exact owned host PID.
        result = await runtime.invoke(input="Return finished.")
        assert result.status == "succeeded", result.to_mapping()
        os.kill(result.metadata["host_pid"], signal.SIGKILL)
        async with asyncio.timeout(3):
            while True:
                checked = await transport.client(
                    {
                        "interface_version": 2,
                        "operation": "check",
                        "level": "ready",
                        **target,
                    },
                    socket_path=socket_path,
                )
                if checked["result"]["outcome"] == "failed":
                    break
                await asyncio.sleep(0.01)
        report = checked["result"]
        assert checked["status"] == "failed"
        assert report["reason_code"] == "agent_exited"
        assert report["runtime"] == {"state": "failed", "activity": "stopped"}
        assert [check["name"] for check in report["checks"]] == ["live"]
        assert report["checks"][0]["source"] == "fabric.adapter.liveness"
        assert report["checks"][0]["observed_at"] is not None
        server = await transport.client(
            {"interface_version": 2, "operation": "check"}, socket_path=socket_path
        )
        assert server["result"]["outcome"] == "passed"
        assert host.runtimes[runtime_id].runtime is runtime
    finally:
        stopped = await host.handle(
            {"interface_version": 2, "operation": "stop", **target}
        )
        assert stopped["status"] == "succeeded", stopped
