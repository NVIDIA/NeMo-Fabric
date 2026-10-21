# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real SDK, native core, adapter process, and Pi health-channel checks."""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
import pytest_asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from nemo_fabric import (
    CheckRequest,
    DiscoveryConfig,
    EnvironmentConfig,
    EnvironmentReference,
    Fabric,
    FabricConfig,
    HarnessConfig,
    MetadataConfig,
    ModelConfig,
    RuntimeStatus,
    ToolsConfig,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest_asyncio.fixture(name="blocking_model", loop_scope="function")
async def blocking_model_fixture(unused_tcp_port):
    app = FastAPI()
    entered, release = asyncio.Event(), asyncio.Event()
    requests = []

    @app.post("/v1/chat/completions")
    async def complete(request: Request):
        requests.append(await request.json())
        entered.set()
        await release.wait()
        common = {
            "id": "check-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-4.1-mini",
        }
        chunks = []
        for delta, finish in [
            ({"role": "assistant", "content": "finished"}, None),
            ({}, "stop"),
        ]:
            chunk = {
                **common,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            chunks.append(f"data: {json.dumps(chunk)}\n\n")
        chunks.append("data: [DONE]\n\n")
        return StreamingResponse(iter(chunks), media_type="text/event-stream")

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


@pytest.fixture(name="pi_check_config")
def pi_check_config_fixture(tmp_path, blocking_model):
    endpoint, _, _, _ = blocking_model
    return FabricConfig(
        metadata=MetadataConfig(name="pi-check-e2e"),
        harness=HarnessConfig(adapter_id="nvidia.fabric.pi"),
        discovery=DiscoveryConfig(
            local_paths=[ROOT / "adapters/typescript/pi/pi.fabric-adapter.json"]
        ),
        environment=EnvironmentConfig(
            provider="local",
            workspace=tmp_path,
            env={"CHECK_TEST_KEY": "PRIVATE-test-credential"},
        ),
        models={
            "default": ModelConfig(
                provider="openai",
                model="gpt-4.1-mini",
                api="openai-completions",
                api_key_env="CHECK_TEST_KEY",
                base_url=endpoint,
            )
        },
        tools=ToolsConfig(enabled=[]),
    )


@pytest_asyncio.fixture(
    name="health_execution",
    loop_scope="function",
    params=[
        "local",
        pytest.param(
            "openshell",
            marks=pytest.mark.skipif(
                sys.platform == "win32", reason="Unix command bridge"
            ),
        ),
    ],
)
async def health_execution_fixture(
    request, pi_check_config, server_sdk_bridge, tmp_path
):
    fabric = Fabric()
    config = pi_check_config
    if request.param == "local":
        environment = await fabric.prepare_environment(config, base_dir=ROOT)
    else:
        (tmp_path / "workspace").mkdir()
        (tmp_path / "task-artifacts").mkdir()
        config = config.model_copy(
            update={
                "environment": EnvironmentConfig(
                    provider="openshell",
                    ownership="caller_owned",
                    workspace=str(tmp_path / "workspace"),
                    artifacts=str(tmp_path / "task-artifacts"),
                    env=config.environment.env,
                    connection={
                        "gateway": "https://fixture.invalid",
                        "workspace": "default",
                        "sandbox_name": "pi-check",
                        "sandbox_id": f"pi-check-{tmp_path.name}",
                    },
                    settings={
                        "task_discovery_paths": [
                            str(ROOT / "adapters/typescript/pi/pi.fabric-adapter.json")
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
                        "sandbox_name": "pi-check",
                        "sandbox_id": config.environment.connection["sandbox_id"],
                    },
                }
            ),
            base_dir=ROOT,
        )
    return fabric, config, environment, server_sdk_bridge[1]


async def test_pi_sdk_health_is_independent_of_an_active_invocation(
    health_execution, blocking_model
):
    fabric, pi_check_config, environment, command_log = health_execution
    _, entered, release, requests = blocking_model
    server = await fabric.check(
        environment=environment, request=CheckRequest(scope="server")
    )
    assert server.outcome == (
        "passed" if environment.provider == "openshell" else "unsupported"
    )
    runtime = await fabric.start_runtime_in(pi_check_config, environment, base_dir=ROOT)
    invocation = None
    try:
        report = await Fabric().check(
            environment=environment,
            request=CheckRequest(
                scope="runtime", runtime_id=runtime.runtime_id, level="ready"
            ),
        )
        assert report.outcome == "passed", f"{report.reason_code}: {report.checks}"
        assert [check["name"] for check in report.checks] == ["live", "active", "ready"]
        assert requests == []
        invocation = asyncio.create_task(runtime.invoke(input="PRIVATE prompt"))
        await asyncio.wait_for(entered.wait(), 5)
        before = runtime.invocations
        report = await asyncio.wait_for(
            Fabric().check(
                environment=environment,
                request=CheckRequest(
                    scope="runtime", runtime_id=runtime.runtime_id, level="ready"
                ),
            ),
            4,
        )
        assert report.outcome == "passed"
        assert report.runtime["activity"] == "busy"
        assert not invocation.done()
        assert runtime.invocations == before
        assert len(requests) == 1
        assert "PRIVATE" not in json.dumps(report.to_mapping())
        timeout = await fabric.check(
            environment=environment,
            request=CheckRequest(
                scope="runtime", runtime_id=runtime.runtime_id, timeout_seconds=1e-7
            ),
        )
        assert timeout.outcome == "unknown"
        assert timeout.reason_code == "probe_timeout"
        assert runtime.status is RuntimeStatus.ACTIVE
        assert not invocation.done()
        release.set()
        result = await asyncio.wait_for(invocation, 5)
        assert result.status == "succeeded"
        assert result.output["response"] == "finished"
        if environment.provider == "openshell":
            commands = [
                json.loads(line) for line in command_log.read_text().splitlines()
            ]
            checks = [
                command for command in commands if command["operation"] == "check"
            ]
            assert checks[0].get("runtime_id") is None
            assert all(
                command["runtime_id"] == runtime.runtime_id
                and command["level"] == "ready"
                and command["agent"] == pi_check_config.metadata.name
                for command in checks[1:]
            )
    finally:
        release.set()
        if invocation is not None:
            await asyncio.wait_for(invocation, 5)
        await runtime.stop()
        await fabric.release_environment(environment)


async def test_health_selects_one_runtime_and_release_invalidates_bindings(
    pi_check_config,
):
    fabric = Fabric()
    environment = await fabric.prepare_environment(pi_check_config, base_dir=ROOT)
    server = await fabric.check(
        environment=environment, request=CheckRequest(scope="server")
    )
    assert server.outcome == "unsupported"
    assert server.reason_code == "unsupported_scope"
    assert server.checks == []
    first = await fabric.start_runtime_in(pi_check_config, environment, base_dir=ROOT)
    second = await fabric.start_runtime_in(pi_check_config, environment, base_dir=ROOT)
    try:
        await first.stop()
        missing = await fabric.check(
            environment=environment,
            request=CheckRequest(scope="runtime", runtime_id=first.runtime_id),
        )
        assert missing.reason_code == "runtime_unavailable"
        active = await fabric.check(
            environment=environment,
            request=CheckRequest(scope="runtime", runtime_id=second.runtime_id),
        )
        assert active.outcome == "passed"
        assert active.runtime_id == second.runtime_id
        with pytest.raises(ValidationError, match="level"):
            CheckRequest(
                scope="runtime", runtime_id=second.runtime_id, level="operational"
            )
        await fabric.release_environment(environment)
        released = await fabric.check(
            environment=environment,
            request=CheckRequest(scope="runtime", runtime_id=second.runtime_id),
        )
        assert released.reason_code == "runtime_unavailable"
        assert second.status is RuntimeStatus.ACTIVE
    finally:
        await first.stop()
        await second.stop()
        await fabric.release_environment(environment)
