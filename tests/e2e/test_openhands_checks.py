# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential-free native OpenHands observations during a blocked model request."""

import asyncio
import os
import sys

import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapters.openhands.adapter import OpenHandsRuntime


@pytest_asyncio.fixture(name="openhands_health_model", loop_scope="function")
async def openhands_health_model_fixture(unused_tcp_port):
    pytest.importorskip("openhands.sdk")
    pytest.importorskip("openhands.tools")
    app = FastAPI()
    entered, release = asyncio.Event(), asyncio.Event()
    requests = []

    @app.post("/v1/chat/completions")
    async def complete(request: Request):
        requests.append(await request.json())
        entered.set()
        await release.wait()
        return JSONResponse(
            {
                "id": "chatcmpl-health",
                "object": "chat.completion",
                "created": 0,
                "model": "local-health-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-finish",
                                    "type": "function",
                                    "function": {
                                        "name": "finish",
                                        "arguments": '{"message":"finished"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
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


def _health_config(endpoint, tmp_path):
    return FabricConfig(
        metadata=MetadataConfig(name="openhands-health"),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.openhands", settings={"condenser": "none"}
        ),
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
                "FABRIC_NATIVE_HEALTH_TEST_KEY": "local-test-only",
            },
        ),
        tools=ToolsConfig(enabled=[]),
    )


async def test_openhands_sdk_checks_do_not_wait_for_native_invocation(
    openhands_health_model, tmp_path
):
    endpoint, entered, release, requests = openhands_health_model
    os.environ["ADAPTER_PYTHON"] = sys.executable
    fabric = Fabric()
    config = _health_config(endpoint, tmp_path)
    environment = await fabric.prepare_environment(config, base_dir=tmp_path)
    runtime = invocation = None
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
            for level in ("live", "active"):
                report = await fabric.check(
                    environment=environment,
                    request=CheckRequest(
                        scope="runtime",
                        runtime_id=runtime.runtime_id,
                        level=level,
                        timeout_seconds=1,
                    ),
                )
                expected = (
                    "unknown" if phase == "before" and level == "active" else "passed"
                )
                assert report.outcome == expected, report.to_mapping()
                assert [check["name"] for check in report.checks] == ["live", "active"][
                    : 1 if level == "live" else 2
                ]
                if expected == "passed":
                    assert report.runtime["activity"] == (
                        "busy" if phase == "during" else "idle"
                    )
                if phase == "during":
                    assert not invocation.done()
            assert len(requests) == (0 if phase == "before" else 1)
    finally:
        release.set()
        if invocation is not None:
            await asyncio.gather(invocation, return_exceptions=True)
        if runtime is not None:
            await runtime.stop()
        await fabric.release_environment(environment)


async def test_openhands_native_close_invalidates_liveness(
    openhands_health_model, tmp_path
):
    endpoint, _, _, requests = openhands_health_model
    os.environ["FABRIC_NATIVE_HEALTH_TEST_KEY"] = "local-test-only"
    os.environ["HOME"] = str(tmp_path)
    config = _health_config(endpoint, tmp_path).to_mapping()
    runtime = OpenHandsRuntime()
    await runtime.start(
        {
            "base_dir": str(tmp_path),
            "config": AgentConfig.from_mapping(
                {
                    "models": config["models"],
                    "tools": {"enabled": []},
                    "harness": {"settings": {"condenser": "none"}},
                }
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
        assert (await runtime.check("live"))["outcome"] == "passed"
        # state remains accessible after close; native closure flags must win.
        conversation = runtime._conversation
        before = conversation.state
        conversation.close()
        assert conversation.state is before
        for level in ("live", "active"):
            result = await runtime.check(level)
            assert (result["outcome"], result["reason_code"]) == (
                "failed",
                "session_closed",
            )
        assert requests == []
    finally:
        await runtime.stop()
