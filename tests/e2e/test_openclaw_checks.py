# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real SDK, native core, Python host, and controlled native Gateway HTTP traffic."""

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from nemo_fabric import (
    CheckRequest,
    DiscoveryConfig,
    EnvironmentConfig,
    Fabric,
    FabricConfig,
    FabricRuntimeError,
    HarnessConfig,
    MetadataConfig,
    ModelConfig,
    ServiceReference,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest_asyncio.fixture(name="native_gateway", loop_scope="function")
async def native_gateway_fixture(unused_tcp_port):
    app = FastAPI()
    state = {"ready": True, "authorized": True, "hold_ready": False}
    paths = []
    ready_entered, ready_release = asyncio.Event(), asyncio.Event()
    invoke_entered, invoke_release = asyncio.Event(), asyncio.Event()

    @app.get("/healthz")
    async def live():
        paths.append("/healthz")
        return {"ok": True, "status": "live"}

    @app.get("/startupz")
    async def startup():
        paths.append("/startupz")
        return {"ok": True, "status": "started", "version": "2026.9.4"}

    @app.get("/v1/models")
    async def models(request: Request):
        paths.append("/v1/models")
        if (
            not state["authorized"]
            or request.headers.get("authorization") != "Bearer TEST_ONLY_GATEWAY_TOKEN"
        ):
            return JSONResponse(
                {"error": "TEST_ONLY_PRIVATE_DIAGNOSTIC"}, status_code=401
            )
        return {
            "object": "list",
            "data": [{"id": "openclaw/default", "object": "model"}],
        }

    @app.get("/readyz")
    async def ready():
        paths.append("/readyz")
        if state["hold_ready"]:
            ready_entered.set()
            await ready_release.wait()
        return JSONResponse(
            {
                "ready": state["ready"],
                "failing": [] if state["ready"] else ["telegram"],
            },
            status_code=200 if state["ready"] else 503,
        )

    @app.post("/v1/chat/completions")
    async def invoke(request: Request):
        paths.append("/v1/chat/completions")
        await request.json()
        invoke_entered.set()
        await invoke_release.wait()
        chunk = {
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {"content": "finished"}, "finish_reason": "stop"}
            ],
        }
        return StreamingResponse(
            iter([f"data: {json.dumps(chunk)}\n\n", "data: [DONE]\n\n"]),
            media_type="text/event-stream",
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
        yield (
            f"http://127.0.0.1:{unused_tcp_port}",
            state,
            paths,
            ready_entered,
            ready_release,
            invoke_entered,
            invoke_release,
        )
    finally:
        ready_release.set()
        invoke_release.set()
        server.should_exit = True
        await asyncio.wait_for(serving, 5)


@pytest_asyncio.fixture(name="gateway_service", loop_scope="function")
async def gateway_service_fixture(native_gateway, tmp_path):
    os.environ["OPENCLAW_HEALTH_TEST_TOKEN"] = "TEST_ONLY_GATEWAY_TOKEN"
    os.environ["ADAPTER_PYTHON"] = sys.executable
    config = FabricConfig(
        metadata=MetadataConfig(name="gateway-check-e2e"),
        harness=HarnessConfig(adapter_id="nvidia.fabric.openclaw"),
        discovery=DiscoveryConfig(
            local_paths=[ROOT / "adapters/python/openclaw/openclaw.fabric-adapter.json"]
        ),
        environment=EnvironmentConfig(provider="local", workspace=tmp_path),
        models={"default": ModelConfig(provider="openai", model="test-model")},
    )
    fabric = Fabric()
    service = await fabric.attach_service(
        config,
        ServiceReference.from_mapping(
            {
                "adapter_id": "nvidia.fabric.openclaw",
                "service_type": "openclaw_gateway",
                "connection": {
                    "gateway_url": native_gateway[0],
                    "gateway_token_env": "OPENCLAW_HEALTH_TEST_TOKEN",
                },
            }
        ),
        base_dir=ROOT,
    )
    try:
        native_gateway[2].clear()
        yield fabric, service, config
    finally:
        await service.release()


async def test_gateway_checks_before_runtime_and_include_native_channel_readiness(
    gateway_service, native_gateway
):
    fabric, service, _ = gateway_service
    url, state, paths, *_ = native_gateway
    request = CheckRequest(
        scope="service", service_id=service.service_id, level="ready"
    )
    report = await fabric.check(service=service.handle, request=request)
    assert report.outcome == "passed", report.to_mapping()
    assert report.service_id == service.service_id
    assert report.runtime_id is None
    assert report.runtime is None
    assert paths == ["/healthz", "/v1/models", "/readyz"]
    state["ready"] = False
    failed = await fabric.check(service=service.handle, request=request)
    assert failed.outcome == "failed"
    assert failed.reason_code == "health_failed"
    assert [check["outcome"] for check in failed.checks] == [
        "passed",
        "passed",
        "failed",
    ]
    state["authorized"] = False
    paths.clear()
    unauthorized = await fabric.check(service=service.handle, request=request)
    assert unauthorized.reason_code == "endpoint_unavailable"
    assert paths == ["/healthz", "/v1/models"]
    assert "TEST_ONLY" not in json.dumps(unauthorized.to_mapping())
    await service.release()
    released = await fabric.check(service=service.handle, request=request)
    assert released.reason_code == "service_unavailable"
    async with httpx.AsyncClient(trust_env=False) as client:
        assert (await client.get(f"{url}/healthz")).status_code == 200
    assert "/v1/chat/completions" not in paths


async def test_gateway_health_deadline_does_not_cancel_a_runtime_invocation(
    gateway_service, native_gateway
):
    fabric, service, config = gateway_service
    _, state, paths, _, ready_release, invoke_entered, invoke_release = native_gateway
    runtime = await fabric.start_runtime(config, service=service, base_dir=ROOT)
    invocation = asyncio.create_task(runtime.invoke(input="TEST_ONLY_PRIVATE_PROMPT"))
    try:
        await asyncio.wait_for(invoke_entered.wait(), 5)
        request = CheckRequest(
            scope="service", service_id=service.service_id, level="ready"
        )
        with pytest.raises(FabricRuntimeError):
            await service.release()
        assert (
            await fabric.check(service=service.handle, request=request)
        ).outcome == "passed"
        state["hold_ready"] = True
        timed_out = await fabric.check(
            service=service.handle,
            request=CheckRequest(
                scope="service",
                service_id=service.service_id,
                level="ready",
                timeout_seconds=0.1,
            ),
        )
        assert timed_out.outcome == "unknown"
        assert timed_out.reason_code == "probe_timeout"
        assert not invocation.done()
        assert paths.count("/v1/chat/completions") == 1
        assert "TEST_ONLY" not in json.dumps(timed_out.to_mapping())
        ready_release.set()
        state["hold_ready"] = False
        assert (
            await fabric.check(service=service.handle, request=request)
        ).outcome == "passed"
        invoke_release.set()
        assert (await asyncio.wait_for(invocation, 5)).status == "succeeded"
    finally:
        ready_release.set()
        invoke_release.set()
        await asyncio.gather(invocation, return_exceptions=True)
        await runtime.stop()
    assert (
        await fabric.check(service=service.handle, request=request)
    ).outcome == "passed"


async def test_gateway_release_invalidates_a_pending_readiness_observation(
    gateway_service, native_gateway
):
    fabric, service, _ = gateway_service
    _, state, _, ready_entered, ready_release, *_ = native_gateway
    state["hold_ready"] = True
    checking = asyncio.create_task(
        fabric.check(
            service=service.handle,
            request=CheckRequest(
                scope="service",
                service_id=service.service_id,
                level="ready",
            ),
        )
    )
    try:
        await asyncio.wait_for(ready_entered.wait(), 5)
        await service.release()
        ready_release.set()
        report = await checking
        assert report.outcome == "unknown"
        assert report.reason_code == "observation_changed"
    finally:
        ready_release.set()
        await asyncio.gather(checking, return_exceptions=True)


@pytest.mark.skipif(sys.platform == "win32", reason="Executable POSIX Gateway fixture")
async def test_fabric_owned_gateway_can_be_checked_and_released_without_runtimes(
    tmp_path,
):
    state_path = tmp_path / "health-state.json"
    state_path.write_text(json.dumps({"ready": True}))
    stopped_path = tmp_path / "stopped.txt"
    request_path = tmp_path / "invocation.json"
    config = FabricConfig(
        metadata=MetadataConfig(name="owned-gateway-check-e2e"),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.openclaw",
            settings={
                "openclaw_command": str(ROOT / "tests/_utils/mock_openclaw.py"),
            },
        ),
        discovery=DiscoveryConfig(
            local_paths=[ROOT / "adapters/python/openclaw/openclaw.fabric-adapter.json"]
        ),
        environment=EnvironmentConfig(
            provider="local",
            workspace=tmp_path,
            env={
                "FAKE_OPENCLAW_CAPTURE": str(tmp_path / "config.json"),
                "FAKE_OPENCLAW_REQUEST": str(request_path),
                "FAKE_OPENCLAW_STOPPED": str(stopped_path),
                "FAKE_OPENCLAW_HEALTH_STATE": str(state_path),
            },
        ),
        models={
            "default": ModelConfig(
                provider="test",
                model="fabric-echo",
                base_url="https://models.example.test/v1",
            )
        },
    )
    os.environ["ADAPTER_PYTHON"] = sys.executable
    fabric = Fabric()
    service = await fabric.prepare_service(config, base_dir=ROOT)
    try:
        request = CheckRequest(
            scope="service", service_id=service.service_id, level="ready"
        )
        report = await fabric.check(service=service.handle, request=request)
        assert report.outcome == "passed", report.to_mapping()
        state_path.write_text(json.dumps({"ready": False}))
        failed = await fabric.check(service=service.handle, request=request)
        assert failed.reason_code == "health_failed"
        assert [check["outcome"] for check in failed.checks] == [
            "passed",
            "passed",
            "failed",
        ]
        operational = await fabric.check(
            service=service.handle,
            request=CheckRequest(
                scope="service",
                service_id=service.service_id,
                level="operational",
            ),
        )
        assert operational.reason_code == "unsupported_level"
        assert operational.checks == []
        assert not request_path.exists()
    finally:
        await service.release()
    assert stopped_path.read_text() == "stopped"
