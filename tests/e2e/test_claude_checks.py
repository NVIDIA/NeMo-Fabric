# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude's bundled CLI answers health checks during a blocked local model call."""

import asyncio
import os
import sys

import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from _utils.mock_api_server import _messages_text_events
from nemo_fabric_adapter_contract.models import AgentConfig, AdapterCheckResult
from nemo_fabric_adapters.claude.adapter import ClaudeRuntime
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


@pytest.mark.parametrize("missing_mcp", [False, True])
async def test_claude_native_process_and_status_evidence(tmp_path, missing_mcp):
    environment = {
        name: "" for name in os.environ if name.startswith(("ANTHROPIC_", "CLAUDE_"))
    }
    environment.update(
        {
            "HOME": str(tmp_path),
            "CLAUDE_CONFIG_DIR": str(tmp_path),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
    )
    config = {
        "models": {"default": {"provider": "anthropic", "model": "claude-sonnet-4-5"}}
    }
    if missing_mcp:
        config["mcp"] = {
            "servers": {
                "missing": {
                    "transport": "stdio",
                    "url": str(tmp_path / "nonexistent-mcp"),
                }
            }
        }
    runtime = ClaudeRuntime()
    await runtime.start(
        {
            "agent_name": "claude-native-health",
            "base_dir": str(tmp_path),
            "config": AgentConfig.from_mapping(config),
            "runtime_context": {
                "runtime_id": "claude-native-health",
                "invocation_id": "native-health",
                "request_id": "native-health",
                "environment": {
                    "environment_id": "native-health",
                    "provider": "local",
                    "control_location": "external_control",
                    "ownership": "caller_owned",
                    "workspace": str(tmp_path),
                    "env": environment,
                },
                "artifacts": {},
            },
        }
    )
    try:
        if missing_mcp:
            async with asyncio.timeout(5):
                while True:
                    native = await runtime._client.get_mcp_status()
                    if any(
                        server["name"] == "missing" and server["status"] == "failed"
                        for server in native["mcpServers"]
                    ):
                        break
                    await asyncio.sleep(0.01)
        for level in ("live", "active"):
            observation = await runtime.check(level)
            AdapterCheckResult.from_mapping(observation)
            assert observation["outcome"] == "passed"
            assert observation["activity"] == "idle"
        async with asyncio.timeout(5):
            while True:
                ready = await runtime.check("ready")
                AdapterCheckResult.from_mapping(ready)
                if ready["outcome"] != "unknown":
                    break
                await asyncio.sleep(0.01)
        assert ready["outcome"] == "passed"
        assert ready["reason_code"] is None
        assert runtime._claude_session_id is None
        process = runtime._client._transport._process
        process.terminate()
        await process.wait()
        result = await runtime.check("live")
        assert (result["outcome"], result["reason_code"]) == ("failed", "agent_exited")
    finally:
        await runtime.stop()


@pytest_asyncio.fixture(name="claude_blocking_model", loop_scope="function")
async def claude_blocking_model_fixture(unused_tcp_port):
    app = FastAPI()
    entered, release = asyncio.Event(), asyncio.Event()
    requests = []

    @app.middleware("http")
    async def record_provider_request(request: Request, call_next):
        requests.append((request.method, request.url.path))
        return await call_next(request)

    @app.post("/v1/messages")
    async def complete(request: Request):
        payload = await request.json()
        entered.set()
        await release.wait()
        return StreamingResponse(
            _messages_text_events(payload, "finished"), media_type="text/event-stream"
        )

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=unused_tcp_port,
            log_config=None,
            access_log=False,
            lifespan="off",
        )
    )
    serving = asyncio.create_task(server.serve())
    try:
        async with asyncio.timeout(5):
            while not server.started:
                await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{unused_tcp_port}/v1", entered, release, requests
    finally:
        release.set()
        server.should_exit = True
        await asyncio.wait_for(serving, 5)


async def test_claude_native_health_during_invocation(claude_blocking_model, tmp_path):
    endpoint, entered, release, requests = claude_blocking_model
    os.environ["ADAPTER_PYTHON"] = sys.executable
    environment = {
        name: "" for name in os.environ if name.startswith(("ANTHROPIC_", "CLAUDE_"))
    }
    environment.update(
        {
            "HOME": str(tmp_path),
            "CLAUDE_CONFIG_DIR": str(tmp_path),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "FABRIC_CLAUDE_CHECK_TEST_KEY": "test-only-local-model",
        }
    )
    config = FabricConfig(
        metadata=MetadataConfig(name="claude-health-e2e"),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.claude", settings={"setting_sources": []}
        ),
        environment=EnvironmentConfig(
            provider="local", workspace=tmp_path, env=environment
        ),
        models={
            "default": ModelConfig(
                provider="fabric-test",
                model="claude-sonnet-4-5",
                base_url=endpoint,
                api_key_env="FABRIC_CLAUDE_CHECK_TEST_KEY",
            )
        },
        tools=ToolsConfig(enabled=[]),
    )
    fabric = Fabric()
    prepared_environment = await fabric.prepare_environment(config, base_dir=tmp_path)
    runtime = await fabric.start_runtime_in(
        config, prepared_environment, base_dir=tmp_path
    )
    invocation = None
    try:
        report = await fabric.check(
            environment=prepared_environment,
            request=CheckRequest(
                scope="runtime", runtime_id=runtime.runtime_id, level="ready"
            ),
        )
        assert report.outcome == "passed", report.to_mapping()
        assert report.runtime["activity"] == "idle"
        assert requests == []
        invocation = asyncio.create_task(runtime.invoke(input="Return finished."))
        await asyncio.wait_for(entered.wait(), 30)
        request_count = len(requests)
        invocation_count = runtime.invocations
        for level in ("live", "active", "ready"):
            report = await fabric.check(
                environment=prepared_environment,
                request=CheckRequest(
                    scope="runtime",
                    runtime_id=runtime.runtime_id,
                    level=level,
                    timeout_seconds=5,
                ),
            )
            assert report.outcome == "passed", report.to_mapping()
            assert report.runtime["activity"] == "busy"
            assert not invocation.done()
            assert len(requests) == request_count
            assert runtime.invocations == invocation_count
        release.set()
        result = await asyncio.wait_for(invocation, 30)
        assert result.status == "succeeded", result.to_mapping()
        assert result.output["response"] == "finished"
        assert result.output["num_turns"] == 1
        invocation_count = runtime.invocations
        report = await fabric.check(
            environment=prepared_environment,
            request=CheckRequest(
                scope="runtime", runtime_id=runtime.runtime_id, level="ready"
            ),
        )
        assert report.outcome == "passed"
        assert report.runtime["activity"] == "idle"
        assert len(requests) == request_count
        assert runtime.invocations == invocation_count
    finally:
        release.set()
        try:
            if invocation is not None:
                await asyncio.wait_for(invocation, 30)
        finally:
            try:
                await runtime.stop()
            finally:
                await fabric.release_environment(prepared_environment)
