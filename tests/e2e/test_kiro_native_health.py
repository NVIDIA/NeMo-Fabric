# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline native CLI qualification; CI installs Kiro 2.34.2 and enables this test.

RUN_FABRIC_KIRO_NATIVE_HEALTH=1 uv run --no-sync pytest tests/e2e/test_kiro_native_health.py
"""

import asyncio
import json
import os
import shutil
import struct
import subprocess
import zlib
from pathlib import Path

import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response

from nemo_fabric import (
    CheckRequest,
    DiscoveryConfig,
    EnvironmentConfig,
    Fabric,
    FabricConfig,
    HarnessConfig,
    MetadataConfig,
    ModelConfig,
    RuntimeConfig,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest_asyncio.fixture(name="kiro_model", loop_scope="function")
async def kiro_model_fixture(unused_tcp_port):
    app = FastAPI()
    entered, release = asyncio.Event(), asyncio.Event()
    requests = []

    @app.post("/v1/chat/completions")
    async def byom_complete(request: Request):
        assert request.headers["authorization"] == "Bearer local-test-only"
        body = await request.json()
        requests.append(body)
        entered.set()
        await release.wait()
        if len(requests) == 1:
            assert any(
                tool["function"]["name"] == "execute_bash" for tool in body["tools"]
            )
            message = {
                "content": None,
                "tool_calls": [
                    {
                        "id": "native-pwd",
                        "type": "function",
                        "function": {
                            "name": "execute_bash",
                            "arguments": '{"command":"pwd"}',
                        },
                    }
                ],
            }
            reason = "tool_calls"
        else:
            tool = next(
                message for message in body["messages"] if message["role"] == "tool"
            )
            assert tool["tool_call_id"] == "native-pwd"
            assert "Exit Code: 0" in tool["content"]
            message, reason = {"content": "finished"}, "stop"
        return {
            "choices": [{"message": message, "finish_reason": reason}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
            },
        }

    @app.get("/List-Available-Models")
    async def models():
        return {
            "models": [
                {
                    "modelId": "test-model",
                    "modelName": "Local health test",
                    "provider": "Test",
                    "tokenLimits": {"maxInputTokens": 100000, "maxOutputTokens": 10000},
                }
            ],
            "defaultModel": {"modelId": "test-model"},
        }

    @app.post("/Get-Profile")
    async def profile():
        return {
            "profile": {
                "profileArn": "arn:aws:codewhisperer:us-east-1:000000000000:profile/test"
            }
        }

    @app.post("/generateAssistantResponse")
    async def complete(request: Request):
        requests.append(await request.json())
        entered.set()
        await release.wait()
        # Kiro uses AWS event-stream frames, including prelude and message CRCs.
        frames = []
        for event_type, body in [
            (
                "assistantResponseEvent",
                {"content": "finished", "modelId": "test-model"},
            ),
            # EOF without this native stop reason triggers truncation recovery.
            ("metadataEvent", {"stopReason": "END_TURN"}),
        ]:
            headers = b""
            for name, value in [
                (":message-type", "event"),
                (":event-type", event_type),
                (":content-type", "application/json"),
            ]:
                name, value = name.encode(), value.encode()
                headers += (
                    bytes([len(name)])
                    + name
                    + b"\x07"
                    + struct.pack(">H", len(value))
                    + value
                )
            payload = json.dumps(body).encode()
            prelude = struct.pack(">II", 16 + len(headers) + len(payload), len(headers))
            message = (
                prelude + struct.pack(">I", zlib.crc32(prelude)) + headers + payload
            )
            frames.append(message + struct.pack(">I", zlib.crc32(message)))
        return Response(
            b"".join(frames),
            media_type="application/vnd.amazon.eventstream",
            headers={
                "x-amzn-requestid": "local-test",
                "x-amzn-kiro-conversation-id": "local-conversation",
            },
        )

    @app.post("/")
    async def control_plane(request: Request):
        operation = request.headers["x-amz-target"].rsplit(".", 1)[-1]
        if operation == "ListAvailableModels":
            return await models()
        if operation == "GetProfile":
            return await profile()
        if operation == "GenerateAssistantResponse":
            return await complete(request)
        return Response(status_code=404)

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=unused_tcp_port, log_level="error")
    )
    serving = asyncio.create_task(server.serve())
    try:
        async with asyncio.timeout(5):
            while not server.started:
                await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{unused_tcp_port}", entered, release, requests
    finally:
        release.set()
        server.should_exit = True
        await asyncio.wait_for(serving, 5)


@pytest.mark.skipif(
    os.environ.get("RUN_FABRIC_KIRO_NATIVE_HEALTH") != "1",
    reason="native CLI qualification runs in the pinned Kiro CI lane",
)
@pytest.mark.parametrize("byom", [False, True])
async def test_kiro_native_cli_health_during_held_model_request(
    kiro_model, tmp_path, byom
):
    executable = shutil.which("kiro-cli")
    assert executable is not None, "Install Kiro CLI 2.34.2 before native qualification"
    endpoint, entered, release, requests = kiro_model
    environment_values = {
        "HOME": str(tmp_path),
        "KIRO_HOME": str(tmp_path),
        "PATH": os.environ["PATH"],
        "KIRO_API_KEY": "local-test-only",
        "KIRO_KAS_ENDPOINT": endpoint,
        "KIRO_KAS_CONTROL_PLANE_ENDPOINT": endpoint,
        "KIRO_KAS_REGION": "us-east-1",
        "KIRO_DISABLE_EXPERIMENT_CONFIG": "true",
        "KIRO_DISABLE_SESSION_TITLE_LLM": "true",
        "KIRO_CONTENT_COLLECTION_ENABLED": "false",
        "OTEL_SDK_DISABLED": "true",
        "CLOUD_CONFIG_ENDPOINT": endpoint,
    }
    version = subprocess.run(
        [executable, "--version"],
        env=environment_values,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert version.stdout.strip() == "kiro-cli 2.34.2"
    config = FabricConfig(
        metadata=MetadataConfig(name="kiro-native-health"),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.kiro",
            settings={"experimental_byom": True, "permission_mode": "allow_once"}
            if byom
            else {},
        ),
        discovery=DiscoveryConfig(
            local_paths=[ROOT / "adapters/typescript/kiro/kiro.fabric-adapter.json"]
        ),
        models={
            "default": ModelConfig(
                provider="openai" if byom else "kiro",
                model="test-model",
                **(
                    {"base_url": endpoint + "/v1", "api_key_env": "KIRO_API_KEY"}
                    if byom
                    else {}
                ),
            )
        },
        runtime=RuntimeConfig(timeout_seconds=30),
        environment=EnvironmentConfig(
            provider="local", workspace=tmp_path, env=environment_values
        ),
    )
    fabric = Fabric()
    environment = await fabric.prepare_environment(config, base_dir=ROOT)
    runtime = invocation = None
    try:
        runtime = await fabric.start_runtime_in(config, environment, base_dir=ROOT)
        request = CheckRequest(
            scope="runtime", runtime_id=runtime.runtime_id, timeout_seconds=2
        )
        assert (
            await fabric.check(environment=environment, request=request)
        ).outcome == "passed"
        assert requests == []
        invocation = asyncio.create_task(
            runtime.invoke(
                input="Run pwd, then return finished." if byom else "Return finished."
            )
        )
        await asyncio.wait_for(entered.wait(), 20)
        for level, expected in [
            ("live", "passed"),
            ("active", "unsupported"),
            ("ready", "unsupported"),
        ]:
            report = await asyncio.wait_for(
                fabric.check(
                    environment=environment,
                    request=CheckRequest(
                        scope="runtime",
                        runtime_id=runtime.runtime_id,
                        level=level,
                        timeout_seconds=2,
                    ),
                ),
                3,
            )
            assert report.outcome == expected, report.to_mapping()
            assert report.runtime["activity"] == "busy"
            assert not invocation.done()
            assert len(requests) == 1
        timed = await fabric.check(
            environment=environment,
            request=CheckRequest(
                scope="runtime", runtime_id=runtime.runtime_id, timeout_seconds=1e-7
            ),
        )
        assert timed.reason_code == "probe_timeout"
        assert not invocation.done()
        assert len(requests) == 1
        release.set()
        result = await asyncio.wait_for(invocation, 15)
        assert result.status == "succeeded", result.to_mapping()
        assert result.output.response == "finished"
        after = await fabric.check(environment=environment, request=request)
        assert after.outcome == "passed"
        assert after.runtime["activity"] == "idle"
        assert len(requests) == (2 if byom else 1)
        if byom:
            assert result.usage.input_tokens == 200
            assert result.usage.output_tokens == 20
    finally:
        release.set()
        try:
            if invocation is not None:
                await asyncio.wait_for(invocation, 30)
        finally:
            try:
                if runtime is not None:
                    await runtime.stop()
            finally:
                await fabric.release_environment(environment)
