# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Deep Agents health checks during a blocked local model request."""

import asyncio
import os
import sys

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
from nemo_fabric_adapters.deepagents.adapter import DeepAgentsRuntime


@pytest_asyncio.fixture(name="health_model", loop_scope="function")
async def health_model_fixture(unused_tcp_port):
    app = FastAPI()
    entered, release = asyncio.Event(), asyncio.Event()
    requests = []

    @app.middleware("http")
    async def record(request: Request, call_next):
        requests.append((request.method, request.url.path))
        return await call_next(request)

    @app.post("/v1/chat/completions")
    async def complete(request: Request):
        payload = await request.json()
        entered.set()
        await release.wait()
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


def _config(endpoint, tmp_path):
    return FabricConfig(
        metadata=MetadataConfig(name="deepagents-health"),
        harness=HarnessConfig(adapter_id="nvidia.fabric.langchain.deepagents"),
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


async def test_deepagents_sdk_health_during_invocation(health_model, tmp_path):
    endpoint, entered, release, requests = health_model
    os.environ["ADAPTER_PYTHON"] = sys.executable
    fabric = Fabric()
    config = _config(endpoint, tmp_path)
    environment = await fabric.prepare_environment(config, base_dir=tmp_path)
    runtime = None
    invocation = None
    try:
        runtime = await fabric.start_runtime_in(config, environment, base_dir=tmp_path)
        for phase in ("before", "during", "after"):
            if phase == "during":
                invocation = asyncio.create_task(
                    runtime.invoke(input="Return finished.")
                )
                await asyncio.wait_for(entered.wait(), 30)
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
            assert requests == (
                [] if phase == "before" else [("POST", "/v1/chat/completions")]
            )
    finally:
        release.set()
        if invocation is not None:
            await asyncio.gather(invocation, return_exceptions=True)
        if runtime is not None:
            await runtime.stop()
        await fabric.release_environment(environment)


async def test_deepagents_native_session_health_and_closure(health_model, tmp_path):
    endpoint, _, _, requests = health_model
    os.environ["FABRIC_NATIVE_HEALTH_TEST_KEY"] = "local-test-only"
    os.environ["HOME"] = str(tmp_path)
    os.environ["HERMES_HOME"] = str(tmp_path / "hermes")
    config = _config(endpoint, tmp_path)
    native = DeepAgentsRuntime()
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
        before = await native._checkpointer.conn.execute_fetchall(
            "SELECT * FROM checkpoints"
        )
        for level in ("live", "active", "ready"):
            async with asyncio.timeout(3):
                result = await native.check(level)
            AdapterCheckResult.from_mapping(result)
            assert (result["outcome"], result["activity"]) == ("passed", "idle")
        assert before == await native._checkpointer.conn.execute_fetchall(
            "SELECT * FROM checkpoints"
        )
        assert native._completed_invocations == 0
        assert requests == []
        await native._checkpointer.conn.close()
        for level in ("live", "active", "ready"):
            result = await native.check(level)
            assert (result["outcome"], result["reason_code"]) == (
                "failed",
                "session_closed",
            )
        assert requests == []
    finally:
        await native.stop()
