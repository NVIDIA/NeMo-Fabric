# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify the bundled Codex app-server against an isolated local model endpoint."""

import asyncio
import os
import sys
from pathlib import Path

import pytest_asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from _utils.mock_api_server import _responses_text_events
from nemo_fabric import (
    CheckRequest,
    EnvironmentConfig,
    Fabric,
    FabricConfig,
    HarnessConfig,
    MetadataConfig,
    ModelConfig,
)
from nemo_fabric_adapter_contract.models import AgentConfig, AdapterCheckResult
from nemo_fabric_adapters.codex.adapter import CodexRuntime


@pytest_asyncio.fixture(name="codex_model", loop_scope="function")
async def codex_model_fixture(unused_tcp_port):
    app = FastAPI()
    entered, release = asyncio.Event(), asyncio.Event()
    requests = []

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    )
    async def endpoint(request: Request, path: str):
        requests.append((request.method, request.url.path))
        if request.method != "POST" or path != "v1/responses":
            return JSONResponse(
                {"error": "unexpected local provider request"}, status_code=400
            )
        payload = await request.json()
        entered.set()
        await release.wait()
        return StreamingResponse(
            _responses_text_events(payload, "finished"), media_type="text/event-stream"
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


def _config(endpoint, tmp_path):
    environment = {
        name: ""
        for name in os.environ
        if name.startswith(("CODEX_", "OPENAI_", "ANTHROPIC_"))
    }
    environment.update(
        HOME=str(tmp_path),
        CODEX_HOME=str(tmp_path / "codex-home"),
        FABRIC_CODEX_CHECK_TEST_KEY="local-test-only",
    )
    return FabricConfig(
        metadata=MetadataConfig(name="codex-health-e2e"),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.codex",
            settings={"config_overrides": {"check_for_update_on_startup": False}},
        ),
        environment=EnvironmentConfig(
            provider="local", workspace=tmp_path, env=environment
        ),
        models={
            "default": ModelConfig(
                provider="fabric-test",
                model="fabric-echo",
                base_url=endpoint,
                api_key_env="FABRIC_CODEX_CHECK_TEST_KEY",
            )
        },
    )


async def test_codex_native_evidence_before_prompt_and_after_process_exit(
    codex_model, tmp_path
):
    endpoint, _, _, requests = codex_model
    config = _config(endpoint, tmp_path)
    runtime = CodexRuntime()
    await asyncio.wait_for(
        runtime.start(
            {
                "agent_name": "codex-native-health",
                "base_dir": str(tmp_path),
                "config": AgentConfig.from_mapping(
                    {
                        "models": {"default": config.models["default"].to_mapping()},
                        "harness": {"settings": config.harness.settings},
                    }
                ),
                "runtime_context": {
                    "runtime_id": "codex-native-health",
                    "invocation_id": "health",
                    "request_id": "health",
                    "environment": {
                        "environment_id": "native-health",
                        "provider": "local",
                        "control_location": "external_control",
                        "ownership": "caller_owned",
                        "workspace": str(tmp_path),
                        "env": config.environment.env,
                    },
                    "artifacts": {},
                },
            }
        ),
        30,
    )
    try:
        thread = runtime._thread
        before = (await thread.read()).thread
        for level in ("live", "active", "ready"):
            async with asyncio.timeout(5):
                result = await runtime.check(level)
            AdapterCheckResult.from_mapping(result)
            assert (result["outcome"], result["activity"]) == ("passed", "idle")
        after = (await thread.read()).thread
        timestamps = {"created_at", "updated_at", "recency_at"}
        assert before.model_dump(exclude=timestamps) == after.model_dump(
            exclude=timestamps
        )
        assert before.turns == []
        assert requests == []
        process = runtime._client._client._sync._proc
        process.terminate()
        await asyncio.to_thread(process.wait, 5)
        for level in ("live", "active", "ready"):
            result = await runtime.check(level)
            assert (result["outcome"], result["reason_code"]) == (
                "failed",
                "agent_exited",
            )
        assert requests == []
    finally:
        await runtime.stop()


async def test_codex_sdk_health_during_invocation(codex_model, tmp_path):
    endpoint, entered, release, requests = codex_model
    os.environ["ADAPTER_PYTHON"] = sys.executable
    config = _config(endpoint, tmp_path)
    fabric = Fabric()
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
            # A transcript can appear after the first turn. Checks must preserve
            # its contents and native configuration files throughout each phase.
            files = [
                path
                for path in Path(tmp_path).rglob("*")
                if path.is_file()
                and (
                    path.suffix == ".jsonl"
                    and "sessions" in path.parts
                    or path.name in {"config.toml", "auth.json", "models_cache.json"}
                )
            ]
            before = {path: path.read_bytes() for path in files}
            for level in ("live", "active", "ready"):
                report = await fabric.check(
                    environment=environment,
                    request=CheckRequest(
                        scope="runtime",
                        runtime_id=runtime.runtime_id,
                        level=level,
                        timeout_seconds=5,
                    ),
                )
                assert report.outcome == "passed", report.to_mapping()
                assert report.runtime["activity"] == (
                    "busy" if phase == "during" else "idle"
                )
                if phase == "during":
                    assert not invocation.done()
            assert before == {path: path.read_bytes() for path in files}
            assert requests == (
                [] if phase == "before" else [("POST", "/v1/responses")]
            )
        assert result.output["thread_id"]
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
