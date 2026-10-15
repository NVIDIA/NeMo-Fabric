# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the custom LangGraph consumer's normal SDK and OpenShell path.

The tests cover ownership, task-visible configuration, and optional Relay and
MCP wiring without a live gateway or model endpoint.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_fabric import Fabric
from nemo_fabric import FabricRuntimeError

from examples.langgraph_custom_agent.consumer import openshell


def _args(
    tmp_path: Path,
    *,
    ownership: str = "fabric_owned",
    relay: bool = False,
    mcp: bool = False,
):
    policy = tmp_path / "policy.yaml"
    policy.write_text("version: 1\n", encoding="utf-8")
    return SimpleNamespace(
        ownership=ownership,
        gateway="http://127.0.0.1:18080",
        workspace="default",
        image="sha256:" + "a" * 64,
        policy_file=str(policy),
        results_file=str(tmp_path / "report.json"),
        sandbox_name="langgraph-email-analysis",
        sandbox_id="caller-sandbox-id" if ownership == "caller_owned" else None,
        model="nvidia/test-model",
        base_url="https://integrate.api.nvidia.com/v1",
        input="Urgent: verify your password immediately at https://example.invalid.",
        follow_up="Team lunch is at noon.",
        relay=relay,
        mcp=mcp,
    )


@pytest.mark.parametrize(
    "relay,mcp", [(False, False), (True, False), (False, True), (True, True)]
)
def test_config_targets_the_task_image_and_optional_capabilities(relay, mcp, tmp_path):
    args = _args(tmp_path, relay=relay, mcp=mcp)

    config = openshell.build_config(args)

    assert config.harness.adapter_id == "nvidia.fabric.example.langgraph.email-phishing"
    assert config.harness.resolution == "image_provided"
    assert config.discovery.local_paths == [openshell.ADAPTER_ROOT]
    assert config.models["default"].provider == "openai"
    assert config.models["default"].api_key_env == "OPENAI_API_KEY"
    assert config.models["default"].base_url == args.base_url
    assert config.environment.provider == "openshell"
    assert config.environment.workspace == openshell.TASK_WORKSPACE
    assert config.environment.artifacts == openshell.TASK_ARTIFACTS
    if args.ownership == "caller_owned":
        assert config.environment.connection["sandbox_id"] == args.sandbox_id
    else:
        assert "sandbox_id" not in config.environment.connection
    assert config.environment.settings["task_discovery_paths"] == [
        openshell.TASK_ADAPTER_ROOT
    ]
    assert (config.relay is not None) is relay
    if relay:
        assert config.relay.output_dir == f"{openshell.TASK_ARTIFACTS}/relay"
        assert config.relay.observability.atif.output_directory == (
            f"{openshell.TASK_ARTIFACTS}/relay"
        )
    assert (config.mcp is not None) is mcp
    if mcp:
        server = config.mcp.servers["url-inspector"]
        assert server.url == openshell.TASK_PYTHON
        assert server.args == [openshell.TASK_MCP_SERVER]


def test_task_image_plan_resolves_the_custom_adapter_descriptor(tmp_path):
    config = openshell.build_config(_args(tmp_path))
    config.environment.provider = "local"

    plan = Fabric().plan(config, base_dir=Path.cwd())

    assert (
        plan.to_mapping()["adapter_descriptor"]["descriptor"]["adapter_id"]
        == "nvidia.fabric.example.langgraph.email-phishing"
    )


@pytest.mark.parametrize("ownership", ["caller_owned", "fabric_owned"])
def test_consumer_uses_sdk_lifecycle_and_preserves_two_turn_context(
    ownership, tmp_path, monkeypatch
):
    args = _args(tmp_path, ownership=ownership, relay=True, mcp=True)
    events = []
    requests = []

    class FakeRuntime:
        runtime_id = "langgraph-runtime"

        async def invoke(self, *, request):
            requests.append(request)
            events.append("invoke")
            result = {
                "request_id": request.request_id,
                "input": request.input,
                "context": request.context,
            }
            return SimpleNamespace(
                status="succeeded",
                request_id=request.request_id,
                to_mapping=lambda: result,
            )

        async def stop(self):
            events.append("stop")

    class FakeFabric:
        async def attach_environment(self, config, reference):
            events.append("attach")
            assert reference.provider == "openshell"
            assert reference.resource["sandbox_id"] == "caller-sandbox-id"
            return "environment"

        async def prepare_environment(self, config):
            events.append("prepare")
            return "environment"

        async def start_runtime_in(self, config, environment):
            events.append("start")
            assert environment == "environment"
            assert (
                config.harness.adapter_id
                == "nvidia.fabric.example.langgraph.email-phishing"
            )
            return FakeRuntime()

        async def release_environment(self, environment):
            events.append("release")

    monkeypatch.setattr(openshell, "Fabric", FakeFabric)
    report = asyncio.run(openshell.run_demo(args))

    assert report["status"] == "succeeded"
    assert report["runtime_id"] == "langgraph-runtime"
    assert events == [
        "attach" if ownership == "caller_owned" else "prepare",
        "start",
        "invoke",
        "invoke",
        "stop",
        "release",
    ]
    assert [(request.request_id, request.context) for request in requests] == [
        ("langgraph-openshell-turn-1", {"turn": 1}),
        ("langgraph-openshell-turn-2", {"turn": 2}),
    ]
    assert requests[0].relay_session_root == requests[1].relay_session_root
    assert json.loads(Path(args.results_file).read_text(encoding="utf-8")) == report


def test_failed_invocation_still_stops_and_releases_without_exposing_details(
    tmp_path, monkeypatch
):
    args = _args(tmp_path)
    events = []

    class FakeRuntime:
        runtime_id = "failed-langgraph-runtime"

        async def invoke(self, *, request):
            raise FabricRuntimeError(
                "private provider details", code="invoke_failed", stage="invoke"
            )

        async def stop(self):
            events.append("stop")

    class FakeFabric:
        async def prepare_environment(self, config):
            events.append("prepare")
            return "environment"

        async def start_runtime_in(self, config, environment):
            events.append("start")
            return FakeRuntime()

        async def release_environment(self, environment):
            events.append("release")

    monkeypatch.setattr(openshell, "Fabric", FakeFabric)
    report = asyncio.run(openshell.run_demo(args))

    assert report["status"] == "failed"
    assert report["error"] == {"code": "invoke_failed", "stage": "invoke"}
    assert "private provider details" not in json.dumps(report)
    assert events == ["prepare", "start", "stop", "release"]
