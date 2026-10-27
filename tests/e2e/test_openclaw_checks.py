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
from pydantic import ValidationError
import pytest_asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from nemo_fabric import (
    CheckRequest,
    DiscoveryConfig,
    EnvironmentConfig,
    EnvironmentHandle,
    EnvironmentReference,
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
    state = {
        "ready": True,
        "authorized": True,
        "hold_ready": False,
        "agent_registered": True,
        "startup": True,
        "chat_enabled": True,
    }
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
        return JSONResponse(
            {
                "ok": state["startup"],
                "status": "started" if state["startup"] else "starting",
                "version": "2026.9.4",
            },
            status_code=200 if state["startup"] else 503,
        )

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
            "data": [{"id": "openclaw/default", "object": "model"}]
            if state["agent_registered"]
            else [],
        }

    @app.get("/v1/chat/completions")
    async def chat_interface():
        paths.append("/v1/chat/completions")
        return JSONResponse(
            {},
            status_code=405 if state["chat_enabled"] else 404,
            headers={"Allow": "POST"} if state["chat_enabled"] else {},
        )

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
            timeout_graceful_shutdown=1,
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


@pytest.fixture(name="owned_gateway_config")
def owned_gateway_config_fixture(tmp_path):
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
    return config, state_path, stopped_path, request_path


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Owned Gateway fixture requires Linux process supervision",
)
async def test_fabric_owned_gateway_can_be_checked_and_released_without_runtimes(
    owned_gateway_config,
):
    config, state_path, stopped_path, request_path = owned_gateway_config
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
        with pytest.raises(ValidationError, match="level"):
            CheckRequest(
                scope="service", service_id=service.service_id, level="operational"
            )
        assert not request_path.exists()
    finally:
        await service.release()
    assert stopped_path.read_text() == "stopped"


async def test_runtime_health_checks_selected_interface_and_survives_a_busy_timeout(
    gateway_service, native_gateway
):
    fabric, service, config = gateway_service
    _, state, paths, ready_entered, ready_release, invoke_entered, invoke_release = (
        native_gateway
    )
    runtime = await fabric.start_runtime(config, service=service, base_dir=ROOT)
    environment = EnvironmentHandle.from_mapping(runtime.handle.environment)
    request = CheckRequest(
        scope="runtime", runtime_id=runtime.runtime_id, level="ready"
    )
    invocation = None
    try:
        paths.clear()
        report = await fabric.check(environment=environment, request=request)
        assert report.outcome == "passed", report.to_mapping()
        assert report.runtime == {"state": "active", "activity": "idle"}
        assert paths == [
            "/healthz",
            "/startupz",
            "/v1/models",
            "/v1/chat/completions",
            "/readyz",
        ]
        for field, expected_paths in [
            ("startup", ["/healthz", "/startupz"]),
            ("agent_registered", ["/healthz", "/startupz", "/v1/models"]),
            (
                "chat_enabled",
                ["/healthz", "/startupz", "/v1/models", "/v1/chat/completions"],
            ),
        ]:
            state[field] = False
            paths.clear()
            failed = await fabric.check(environment=environment, request=request)
            assert failed.reason_code == "endpoint_unavailable"
            assert [check["name"] for check in failed.checks] == ["live", "active"]
            assert paths == expected_paths
            state[field] = True
        invocation = asyncio.create_task(
            runtime.invoke(input="TEST_ONLY_PRIVATE_PROMPT")
        )
        await asyncio.wait_for(invoke_entered.wait(), 5)
        busy = await fabric.check(environment=environment, request=request)
        assert busy.outcome == "passed"
        assert busy.runtime["activity"] == "busy"
        assert runtime.invocations == []
        state["hold_ready"] = True
        timed_out = await fabric.check(
            environment=environment,
            request=CheckRequest(
                scope="runtime",
                runtime_id=runtime.runtime_id,
                level="ready",
                timeout_seconds=0.1,
            ),
        )
        assert timed_out.reason_code == "probe_timeout"
        assert not invocation.done()
        assert "TEST_ONLY" not in json.dumps(timed_out.to_mapping())
        ready_release.set()
        state["hold_ready"] = False
        assert (
            await fabric.check(environment=environment, request=request)
        ).outcome == "passed"
        invoke_release.set()
        assert (await asyncio.wait_for(invocation, 5)).status == "succeeded"
        state["ready"] = False
        assert (
            await fabric.check(environment=environment, request=request)
        ).reason_code == "health_failed"
    finally:
        ready_release.set()
        invoke_release.set()
        if invocation is not None:
            await asyncio.gather(invocation, return_exceptions=True)
        await runtime.stop()
    assert (
        await fabric.check(environment=environment, request=request)
    ).reason_code == "runtime_unavailable"
    assert (
        await fabric.check(
            service=service.handle,
            request=CheckRequest(scope="service", service_id=service.service_id),
        )
    ).outcome == "passed"


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Owned Gateway fixture requires Linux process supervision",
)
@pytest.mark.parametrize("execution", ["local", "openshell"])
async def test_owned_runtime_checks_through_sdk_and_sandbox_command_bridge(
    execution, owned_gateway_config, server_sdk_bridge, tmp_path
):
    config, state_path, stopped_path, request_path = owned_gateway_config
    fabric = Fabric()
    if execution == "local":
        environment = await fabric.prepare_environment(config, base_dir=ROOT)
    else:
        workspace, artifacts = tmp_path / "workspace", tmp_path / "task-artifacts"
        workspace.mkdir()
        artifacts.mkdir()
        config = config.model_copy(
            update={
                "environment": EnvironmentConfig(
                    provider="openshell",
                    ownership="caller_owned",
                    workspace=str(workspace),
                    artifacts=str(artifacts),
                    env=config.environment.env,
                    connection={
                        "gateway": "https://fixture.invalid",
                        "workspace": "default",
                        "sandbox_name": "openclaw-check",
                        "sandbox_id": f"openclaw-check-{tmp_path.name}",
                    },
                    settings={
                        "task_discovery_paths": [
                            str(
                                ROOT
                                / "adapters/python/openclaw/openclaw.fabric-adapter.json"
                            )
                        ]
                    },
                )
            }
        )
        environment = await fabric.attach_environment(
            config,
            EnvironmentReference.from_mapping(
                {
                    "provider": "openshell",
                    "resource": {
                        "sandbox_name": "openclaw-check",
                        "sandbox_id": config.environment.connection["sandbox_id"],
                    },
                }
            ),
            base_dir=ROOT,
        )
    runtime = await fabric.start_runtime_in(config, environment, base_dir=ROOT)
    request = CheckRequest(
        scope="runtime", runtime_id=runtime.runtime_id, level="ready"
    )
    try:
        report = await fabric.check(environment=environment, request=request)
        assert report.outcome == "passed", report.to_mapping()
        assert [check["name"] for check in report.checks] == ["live", "active", "ready"]
        assert [check["source"] for check in report.checks] == [
            "fabric.adapter.liveness",
            "fabric.adapter.interface",
            "fabric.adapter.readiness",
        ]
        assert not request_path.exists()
        state_path.write_text(json.dumps({"ready": False}))
        assert (
            await fabric.check(environment=environment, request=request)
        ).reason_code == "health_failed"
        with pytest.raises(ValidationError, match="level"):
            CheckRequest(
                scope="runtime", runtime_id=runtime.runtime_id, level="operational"
            )
        if execution == "openshell":
            commands = [
                json.loads(line)
                for line in server_sdk_bridge[1].read_text().splitlines()
            ]
            checks = [
                command for command in commands if command["operation"] == "check"
            ]
            assert checks
            assert all(
                command["agent"] == config.metadata.name
                and command["runtime_id"] == runtime.runtime_id
                for command in checks
            )
    finally:
        await runtime.stop()
        await fabric.release_environment(environment)
    assert stopped_path.read_text() == "stopped"


async def test_shared_service_software_retains_original_capture_across_runtimes(
    gateway_service, native_gateway
):
    fabric, service, config = gateway_service
    native_gateway[-1].set()
    async with await fabric.start_runtime(
        config, service=service, base_dir=ROOT
    ) as first_runtime:
        first = await first_runtime.invoke(input="first shared-service observation")
    # A separate SDK instance must not make optional snapshot times invalidate
    # the original service binding, or refresh the service's retained report.
    async with await Fabric().start_runtime(
        config, service=service, base_dir=ROOT
    ) as second_runtime:
        second = await second_runtime.invoke(input="second shared-service observation")
    assert first.status == second.status == "succeeded"
    assert first.runtime_id != second.runtime_id
    assert native_gateway[2].count("/startupz") == 4
    assert native_gateway[2].count("/v1/models") == 2

    def service_entries(result):
        return [
            entry.to_mapping()
            for env in result.execution.software_environments
            if "service" in env.participants
            for entry in env.entries
        ]

    one, two = service_entries(first), service_entries(second)
    assert len(one) == len(two) == 1
    assert one == two
    assert one[0]["version"]["value"] == service.handle.metadata["openclaw_version"]
    assert one[0]["version"]["source"] == "service:startupz"
    assert one[0]["version"]["origin"] == "reported"
    assert not any(
        entry.role == "interpreter"
        for env in second.execution.software_environments
        if "service" in env.participants
        for entry in env.entries
    )


async def test_disabled_runtime_can_use_an_existing_service_without_software_collection(
    gateway_service, native_gateway
):
    fabric, service, config = gateway_service
    native_gateway[-1].set()
    disabled = config.model_copy(deep=True)
    disabled.provenance.enabled = False
    async with await fabric.start_runtime(
        disabled, service=service, base_dir=ROOT
    ) as runtime:
        result = await runtime.invoke(input="metadata disabled for this runtime")
    assert result.status == "succeeded"
    assert result.execution is None
    assert service.handle.metadata["openclaw_version"] == "2026.9.4"
