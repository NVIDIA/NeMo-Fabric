# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the OO Agents adapter."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import types
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import asynccontextmanager
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import call

import pytest
from nemo_fabric import DiscoveryConfig
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import FabricConfigError
from nemo_fabric import MetadataConfig
from nemo_fabric import WorkflowConfig
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunStatus
from nemo_fabric_adapter_contract.models import RuntimeContext

ROOT = Path(__file__).parents[2]
NOOA_ROOT = ROOT / "adapters" / "python" / "nooa"

if not ((3, 12) <= sys.version_info[:2] < (3, 14)):
    pytest.skip("NOOA supports Python 3.12 and 3.13", allow_module_level=True)

from nemo_fabric_adapters.nooa import adapter
from nemo_fabric_adapters.nooa import model_support
from nemo_fabric_adapters.nooa import telemetry as nooa_telemetry
from nemo_fabric_adapters.nooa.targets import arc_solver


def _workflow(**settings: Any) -> dict[str, Any]:
    return {
        "entrypoint": {
            "kind": "interactive_agent_factory",
            "ref": "fabric_nooa_test_target:create_agent",
        },
        "settings": settings,
    }


def _start_payload(
    tmp_path: Path,
    *,
    runtime_id: str = "runtime-1",
    **settings: Any,
) -> dict[str, Any]:
    return {
        "base_dir": str(tmp_path),
        "config": AgentConfig.from_mapping({"workflow": _workflow(**settings)}),
        "runtime_context": {
            "runtime_id": runtime_id,
            "environment": {
                "workspace": str(tmp_path / "workspace"),
                "artifacts": str(tmp_path / "artifacts"),
            },
        },
    }


def _invocation(
    value: Any = "hello",
    *,
    runtime_id: str = "runtime-1",
    request_id: str = "request-1",
    deadline_millis: int | None = None,
) -> tuple[AgentRunRequest, RuntimeContext]:
    context: dict[str, Any] = {
        "runtime_id": runtime_id,
        "invocation_id": f"invocation-{request_id}",
        "request_id": request_id,
        "environment": {
            "environment_id": "environment-1",
            "provider": "test",
            "control_location": "in_env_control",
            "ownership": "caller_owned",
        },
        "artifacts": {},
    }
    if deadline_millis is not None:
        context["deadline_millis"] = deadline_millis
    return (
        AgentRunRequest.from_mapping({"input": value}),
        RuntimeContext.from_mapping(context),
    )


def _agent_double(*, channel_names: tuple[str, ...] = ("user_messages",)):
    buffers = {name: [] for name in channel_names}
    channels: dict[str, MagicMock] = {}
    for name in channel_names:
        channel = MagicMock(name=f"{name}_channel")
        channel.put.side_effect = buffers[name].append

        def drain(channel_name: str = name) -> list[Any]:
            items = list(buffers[channel_name])
            buffers[channel_name].clear()
            return items

        channel.drain.side_effect = drain
        channels[name] = channel

    queue_manager = MagicMock(name="queue_manager")
    queue_manager.channels.return_value = channels
    queue_manager.get_channel.side_effect = channels.__getitem__
    queue_manager.shutdown = AsyncMock()

    async def race() -> list[tuple[str, Any]]:
        for name in channel_names:
            if buffers[name]:
                return [(name, buffers[name].pop(0))]
        raise AssertionError("test agent raced without a pending channel item")

    queue_manager.race = AsyncMock(side_effect=race)

    handlers: list[Any] = []
    event_manager = MagicMock(name="event_manager")

    def on(event_type: str, handler: Any):
        assert event_type == "AgentMessage"
        handlers.append(handler)

        def unsubscribe() -> None:
            handlers.remove(handler)

        return unsubscribe

    event_manager.on.side_effect = on

    agent = MagicMock(name="interactive_agent")
    agent.queue_manager = queue_manager
    agent.event_manager = event_manager
    agent.handle = AsyncMock()
    agent.close = AsyncMock()
    return agent, channels, handlers


@pytest.fixture(name="install_target")
def install_target_fixture(monkeypatch: pytest.MonkeyPatch):
    def install(factory: Any) -> None:
        module = types.ModuleType("fabric_nooa_test_target")
        module.create_agent = factory
        monkeypatch.setitem(sys.modules, module.__name__, module)

    return install


def _install_deterministic_relay(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    invocations: list[Any] = []

    async def invoke(*, agent: Any, call: Any, **_kwargs: Any):
        invocations.append(agent)
        result = await call()
        return nooa_telemetry.RelayInvocation(
            called=True,
            result=result,
            report=nooa_telemetry.RelayReport(
                enabled=True,
                artifacts=(
                    {"kind": "atof", "path": "/safe/current-turn.jsonl"},
                    {"kind": "atif", "path": "/safe/current-turn.json"},
                ),
            ),
        )

    mock_telemetry = MagicMock(name="relay_telemetry")
    mock_telemetry.invoke = AsyncMock(side_effect=invoke)
    mock_telemetry.close = AsyncMock()
    monkeypatch.setattr(
        adapter,
        "RelayTelemetry",
        MagicMock(name="RelayTelemetry", return_value=mock_telemetry),
    )
    return invocations


def test_descriptor_and_registered_target_declare_the_shared_boundary():
    descriptor = json.loads(
        (NOOA_ROOT / "nooa.fabric-adapter.json").read_text(encoding="utf-8")
    )
    target = json.loads(
        (ROOT / "tests" / "fixtures" / "nooa" / "echo.fabric-target.json").read_text(
            encoding="utf-8"
        )
    )
    coding_target = json.loads(
        (NOOA_ROOT / "targets" / "coding-agent.fabric-target.json").read_text(
            encoding="utf-8"
        )
    )
    arc_target = json.loads(
        (NOOA_ROOT / "targets" / "arc-solver.fabric-target.json").read_text(
            encoding="utf-8"
        )
    )

    assert descriptor["adapter_id"] == "nvidia.fabric.nooa"
    assert descriptor["adapter_kind"] == "python"
    assert descriptor["target_types"] == ["workflow"]
    assert descriptor["runner"] == {"module": "nemo_fabric_adapters.nooa.adapter"}
    assert descriptor["config"]["accepts"] == [
        "models",
        "models.base_url",
        "models.temperature",
        "instructions.system",
        "skills",
        "mcp",
    ]
    assert descriptor["capabilities"] == {
        "cancellation": False,
        "service": False,
        "streaming": False,
        "updates": False,
    }
    assert descriptor["telemetry"] == {
        "providers": {
            "relay": {"outputs": ["atif", "otel", "openinference"]},
        }
    }
    assert target["adapter_id"] == descriptor["adapter_id"]
    assert target["spec"]["entrypoint"] == {
        "kind": "interactive_agent_factory",
        "ref": "fabric_nooa_test_target:create_agent",
    }
    assert coding_target["adapter_id"] == descriptor["adapter_id"]
    assert coding_target["spec"]["entrypoint"] == {
        "kind": "interactive_agent_factory",
        "ref": "nemo_fabric_adapters.nooa.targets.coding_agent:create_agent",
    }
    assert coding_target["spec"]["settings_schema"]["additionalProperties"] is False
    assert arc_target["adapter_id"] == descriptor["adapter_id"]
    assert arc_target["spec"]["entrypoint"] == {
        "kind": "interactive_agent_factory",
        "ref": "nemo_fabric_adapters.nooa.targets.arc_solver:create_agent",
    }
    assert arc_target["spec"]["settings_schema"]["additionalProperties"] is False


def test_registered_arc_target_projects_closed_settings(tmp_path: Path):
    config = FabricConfig(
        metadata=MetadataConfig(name="nooa-arc-test"),
        discovery=DiscoveryConfig(local_paths=[NOOA_ROOT]),
        workflow=WorkflowConfig(
            target_id="nvidia.nooa.arc-solver",
            settings={
                "visual": "off",
                "max_actions_per_turn": 3,
            },
        ),
    )

    plan = Fabric().plan(config, base_dir=tmp_path)

    assert plan.to_mapping()["agent_config"]["workflow"] == {
        "entrypoint": {
            "kind": "interactive_agent_factory",
            "ref": "nemo_fabric_adapters.nooa.targets.arc_solver:create_agent",
        },
        "settings": {
            "visual": "off",
            "max_actions_per_turn": 3,
        },
    }


def test_registered_arc_target_rejects_zero_max_actions(tmp_path: Path):
    config = FabricConfig(
        metadata=MetadataConfig(name="nooa-arc-test"),
        discovery=DiscoveryConfig(local_paths=[NOOA_ROOT]),
        workflow=WorkflowConfig(
            target_id="nvidia.nooa.arc-solver",
            settings={"max_actions_per_turn": 0},
        ),
    )

    with pytest.raises(FabricConfigError, match="max_actions_per_turn"):
        Fabric().plan(config, base_dir=tmp_path)


@pytest.mark.parametrize(
    "settings",
    [
        {"run_dir": "/tmp/unmanaged"},
        {"alias": "real-game-id"},
    ],
)
def test_registered_arc_target_rejects_consumer_identity_and_paths(
    tmp_path: Path,
    settings: dict[str, Any],
):
    config = FabricConfig(
        metadata=MetadataConfig(name="nooa-arc-test"),
        discovery=DiscoveryConfig(local_paths=[NOOA_ROOT]),
        workflow=WorkflowConfig(
            target_id="nvidia.nooa.arc-solver",
            settings=settings,
        ),
    )

    with pytest.raises(FabricConfigError, match="workflow.settings"):
        Fabric().plan(config, base_dir=tmp_path)


def test_registered_target_projects_the_factory_and_settings(tmp_path: Path):
    config = FabricConfig(
        metadata=MetadataConfig(name="nooa-test"),
        discovery=DiscoveryConfig(
            local_paths=[
                NOOA_ROOT,
                ROOT / "tests" / "fixtures" / "nooa",
            ]
        ),
        workflow=WorkflowConfig(
            target_id="nvidia.tests.nooa.echo",
            settings={"prefix": "test"},
        ),
    )

    plan = Fabric().plan(config, base_dir=tmp_path)

    assert (
        plan["adapter_descriptor"]["descriptor"]["adapter_id"] == "nvidia.fabric.nooa"
    )
    assert (
        plan["adapter_target_descriptor"]["descriptor"]["id"]
        == "nvidia.tests.nooa.echo"
    )
    assert plan.to_mapping()["agent_config"]["workflow"] == {
        "entrypoint": {
            "kind": "interactive_agent_factory",
            "ref": "fabric_nooa_test_target:create_agent",
        },
        "settings": {"prefix": "test"},
    }


def test_registered_target_projects_whole_mcp_servers(tmp_path: Path):
    config = FabricConfig(
        metadata=MetadataConfig(name="nooa-mcp-test"),
        discovery=DiscoveryConfig(
            local_paths=[
                NOOA_ROOT,
                ROOT / "tests" / "fixtures" / "nooa",
            ]
        ),
        workflow=WorkflowConfig(
            target_id="nvidia.tests.nooa.echo",
            settings={},
        ),
    )
    config.add_mcp_server(
        "calculator",
        transport="stdio",
        url=sys.executable,
        args=["calculator_server.py"],
        env={"MODE": "test"},
    )

    plan = Fabric().plan(config, base_dir=tmp_path)

    assert plan.to_mapping()["agent_config"]["mcp"] == {
        "servers": {
            "calculator": {
                "transport": "stdio",
                "url": sys.executable,
                "args": ["calculator_server.py"],
                "env": {"MODE": "test"},
            }
        }
    }


def test_registered_target_rejects_mcp_tool_filters(tmp_path: Path):
    config = FabricConfig(
        metadata=MetadataConfig(name="nooa-filtered-mcp-test"),
        discovery=DiscoveryConfig(
            local_paths=[
                NOOA_ROOT,
                ROOT / "tests" / "fixtures" / "nooa",
            ]
        ),
        workflow=WorkflowConfig(
            target_id="nvidia.tests.nooa.echo",
            settings={},
        ),
    )
    config.add_mcp_server(
        "calculator",
        transport="stdio",
        url=sys.executable,
        allowed_tools=["add"],
    )

    with pytest.raises(FabricConfigError, match="allowed_tools"):
        Fabric().plan(config, base_dir=tmp_path)


def test_registered_target_rejects_mcp_authentication(tmp_path: Path):
    config = FabricConfig(
        metadata=MetadataConfig(name="nooa-authenticated-mcp-test"),
        discovery=DiscoveryConfig(
            local_paths=[
                NOOA_ROOT,
                ROOT / "tests" / "fixtures" / "nooa",
            ]
        ),
        workflow=WorkflowConfig(
            target_id="nvidia.tests.nooa.echo",
            settings={},
        ),
    )
    config.add_mcp_server(
        "repository",
        transport="streamable-http",
        url="https://mcp.example.test",
        authentication={
            "type": "service_account",
            "client_id": "fabric",
            "client_secret_env": "MCP_CLIENT_SECRET",
            "token_url": "https://identity.example.test/token",
        },
    )

    with pytest.raises(FabricConfigError, match="authentication"):
        Fabric().plan(config, base_dir=tmp_path)


def test_registered_target_rejects_unknown_settings(tmp_path: Path):
    config = FabricConfig(
        metadata=MetadataConfig(name="nooa-test"),
        discovery=DiscoveryConfig(
            local_paths=[
                NOOA_ROOT,
                ROOT / "tests" / "fixtures" / "nooa",
            ]
        ),
        workflow=WorkflowConfig(
            target_id="nvidia.tests.nooa.echo",
            settings={"unknown": True},
        ),
    )

    with pytest.raises(FabricConfigError, match="workflow.settings"):
        Fabric().plan(config, base_dir=tmp_path)


def test_interactive_runtime_registers_and_invokes_mcp_in_subprocess(
    tmp_path: Path,
):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "tests" / "fixtures" / "nooa" / "src")

    completed = subprocess.run(
        [sys.executable, "-m", "fabric_nooa_mcp_runtime", str(tmp_path)],
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
        cwd=ROOT,
        env=environment,
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "succeeded"
    assert result["output"]["response"] == (
        f"calculator via {sys.executable}: hello MCP"
    )


def test_main_opts_into_typed_agent_config(monkeypatch: pytest.MonkeyPatch):
    serve = MagicMock()
    monkeypatch.setattr(adapter.lifecycle, "serve", serve)

    adapter.main()

    serve.assert_called_once_with(
        adapter.NooaRuntime, config_loader=AgentConfig.from_mapping
    )


async def test_runtime_builds_once_and_preserves_agent_state(
    tmp_path: Path,
    install_target,
):
    agent, _channels, handlers = _agent_double()
    call_count = 0

    async def handle(notification: dict[str, list[Any]]):
        nonlocal call_count
        call_count += 1
        handlers[-1](SimpleNamespace(content=f"reply-{call_count}"))
        return SimpleNamespace(
            kind=SimpleNamespace(value="DONE"),
            explanation="request complete",
        )

    agent.handle.side_effect = handle
    factory = MagicMock(return_value=agent)
    install_target(factory)
    runtime = adapter.NooaRuntime()

    await runtime.start(_start_payload(tmp_path, prefix="test"))
    first = await runtime.invoke(*_invocation("one", request_id="one"))
    second = await runtime.invoke(*_invocation("two", request_id="two"))
    await runtime.stop()

    factory.assert_called_once()
    build_context = factory.call_args.args[0]
    assert build_context.runtime_id == "runtime-1"
    assert build_context.workspace == tmp_path / "workspace"
    assert build_context.artifact_root == tmp_path / "artifacts"
    assert build_context.config.workflow.settings == {"prefix": "test"}
    assert dict(build_context.models) == {}
    assert build_context.system_instruction is None
    assert dict(build_context.settings) == {"prefix": "test"}
    assert build_context.skill_paths == ()
    assert first.status is AgentRunStatus.SUCCEEDED
    assert first.output["response"] == "reply-1"
    assert first.output["messages"] == [{"content": "reply-1"}]
    assert first.output["completed"] is True
    assert second.output["response"] == "reply-2"
    assert call_count == 2
    assert agent.event_manager.on.call_count == 2
    agent.close.assert_awaited_once_with()


async def test_wait_resumes_on_a_background_channel(
    tmp_path: Path,
    install_target,
):
    agent, channels, handlers = _agent_double(channel_names=("user_messages", "jobs"))
    notifications: list[dict[str, list[Any]]] = []

    async def handle(notification: dict[str, list[Any]]):
        notifications.append(notification)
        if len(notifications) == 1:
            channels["jobs"].put("finished")
            return SimpleNamespace(
                kind=SimpleNamespace(value="WAIT"),
                explanation="waiting for the job",
            )
        handlers[-1](SimpleNamespace(content="job complete"))
        return SimpleNamespace(
            kind=SimpleNamespace(value="DONE"),
            explanation="job completed",
        )

    agent.handle.side_effect = handle
    install_target(MagicMock(return_value=agent))
    runtime = adapter.NooaRuntime()

    await runtime.start(_start_payload(tmp_path))
    try:
        result = await runtime.invoke(*_invocation())
    finally:
        await runtime.stop()

    assert notifications == [
        {"user_messages": ["hello"]},
        {"jobs": ["finished"]},
    ]
    assert result.output["response"] == "job complete"
    assert result.output["reason"] == "DONE"


async def test_terminal_explanation_is_response_without_agent_message(
    tmp_path: Path,
    install_target,
):
    agent, _channels, _handlers = _agent_double()
    agent.handle.return_value = SimpleNamespace(
        kind=SimpleNamespace(value="DONE"),
        explanation="Review complete with no correctness issues.",
    )
    install_target(MagicMock(return_value=agent))
    runtime = adapter.NooaRuntime()

    await runtime.start(_start_payload(tmp_path))
    try:
        result = await runtime.invoke(*_invocation())
    finally:
        await runtime.stop()

    assert result.status is AgentRunStatus.SUCCEEDED
    assert result.output["response"] == "Review complete with no correctness issues."
    assert result.output["messages"] == []


@pytest.mark.parametrize("reason", ["NEED_INPUT", "GET_USER_INPUT"])
async def test_human_input_reasons_complete_without_marking_work_done(
    tmp_path: Path,
    install_target,
    reason: str,
):
    agent, _channels, handlers = _agent_double()

    async def handle(_notification: dict[str, list[Any]]):
        handlers[-1](SimpleNamespace(content="Which branch should I use?"))
        return SimpleNamespace(
            kind=SimpleNamespace(value=reason),
            explanation="a branch name is required",
        )

    agent.handle.side_effect = handle
    install_target(MagicMock(return_value=agent))
    runtime = adapter.NooaRuntime()

    await runtime.start(_start_payload(tmp_path))
    try:
        result = await runtime.invoke(*_invocation())
    finally:
        await runtime.stop()

    assert result.status is AgentRunStatus.SUCCEEDED
    assert result.output["reason"] == reason
    assert result.output["completed"] is False


async def test_invalid_input_returns_a_safe_target_failure(
    tmp_path: Path,
    install_target,
):
    agent, _channels, _handlers = _agent_double()
    install_target(MagicMock(return_value=agent))
    runtime = adapter.NooaRuntime()

    await runtime.start(_start_payload(tmp_path))
    try:
        result = await runtime.invoke(*_invocation({"prompt": "hello"}))
    finally:
        await runtime.stop()

    assert result.status is AgentRunStatus.FAILED
    assert result.error.code == "nooa_invalid_request"
    agent.handle.assert_not_awaited()


async def test_relay_report_is_attached_without_replaying_target(
    tmp_path: Path,
    install_target,
    monkeypatch: pytest.MonkeyPatch,
):
    agent, _channels, handlers = _agent_double()

    async def handle(_notification: dict[str, list[Any]]):
        handlers[-1](SimpleNamespace(content="telemetry reply"))
        return SimpleNamespace(kind="DONE", explanation="complete")

    agent.handle.side_effect = handle
    install_target(MagicMock(return_value=agent))

    async def invoke_telemetry(*, call: Any, **_kwargs: Any):
        return nooa_telemetry.RelayInvocation(
            called=True,
            result=await call(),
            report=nooa_telemetry.RelayReport(
                enabled=True,
                artifacts=({"kind": "atif", "path": "/safe/turn.json"},),
            ),
        )

    mock_telemetry = MagicMock(name="relay_telemetry")
    mock_telemetry.invoke = AsyncMock(side_effect=invoke_telemetry)
    mock_telemetry.close = AsyncMock()
    monkeypatch.setattr(
        adapter,
        "RelayTelemetry",
        MagicMock(name="RelayTelemetry", return_value=mock_telemetry),
    )
    runtime = adapter.NooaRuntime()

    await runtime.start(_start_payload(tmp_path))
    try:
        result = await runtime.invoke(*_invocation())
    finally:
        await runtime.stop()

    assert agent.handle.await_count == 1
    assert result.output["response"] == "telemetry reply"
    assert result.output["telemetry"] == {
        "enabled": True,
        "provider": "relay",
        "emitter": "nooa.nemo_relay_middleware",
    }
    assert result.output["relay_artifacts"] == [
        {"kind": "atif", "path": "/safe/turn.json"}
    ]


async def test_relay_setup_failure_does_not_execute_target(
    tmp_path: Path,
    install_target,
    monkeypatch: pytest.MonkeyPatch,
):
    agent, _channels, _handlers = _agent_double()
    install_target(MagicMock(return_value=agent))

    mock_telemetry = MagicMock(name="relay_telemetry")
    mock_telemetry.invoke = AsyncMock(
        return_value=nooa_telemetry.RelayInvocation(
            called=False,
            result=None,
            report=nooa_telemetry.RelayReport(
                enabled=True,
                error="Relay setup failed (ImportError)",
            ),
        )
    )
    mock_telemetry.close = AsyncMock()
    monkeypatch.setattr(
        adapter,
        "RelayTelemetry",
        MagicMock(name="RelayTelemetry", return_value=mock_telemetry),
    )
    runtime = adapter.NooaRuntime()

    await runtime.start(_start_payload(tmp_path))
    try:
        result = await runtime.invoke(*_invocation())
    finally:
        await runtime.stop()

    assert result.status is AgentRunStatus.FAILED
    assert result.error.code == "nooa_telemetry_setup_failed"
    assert result.output["telemetry"]["degraded"] is True
    agent.handle.assert_not_awaited()


async def test_invalid_respond_result_returns_a_safe_target_failure(
    tmp_path: Path,
    install_target,
):
    agent, _channels, _handlers = _agent_double()
    agent.handle.return_value = SimpleNamespace(kind="UNKNOWN", explanation="bad")
    install_target(MagicMock(return_value=agent))
    runtime = adapter.NooaRuntime()

    await runtime.start(_start_payload(tmp_path))
    try:
        result = await runtime.invoke(*_invocation())
    finally:
        await runtime.stop()

    assert result.status is AgentRunStatus.FAILED
    assert result.error.code == "nooa_invalid_respond_result"


async def test_custom_target_cleanup_takes_precedence(
    tmp_path: Path,
    install_target,
):
    agent, _channels, _handlers = _agent_double()
    cleanup = AsyncMock()
    install_target(
        MagicMock(
            return_value=adapter.InteractiveAgentTarget(agent=agent, close=cleanup)
        )
    )
    runtime = adapter.NooaRuntime()

    await runtime.start(_start_payload(tmp_path))
    await runtime.stop()

    cleanup.assert_awaited_once_with()
    agent.close.assert_not_awaited()


async def test_target_continuation_predicate_reuses_generic_queue_dispatch(
    tmp_path: Path,
    install_target,
):
    agent, channels, handlers = _agent_double(
        channel_names=("user_messages", "game_states")
    )
    current_state: dict[str, Any] | None = None
    notifications: list[dict[str, list[Any]]] = []

    async def handle(notification: dict[str, list[Any]]):
        nonlocal current_state
        notifications.append(notification)
        if len(notifications) == 1:
            current_state = {"state": "NOT_FINISHED", "turn": 0}
            channels["game_states"].put("state-0")
            return SimpleNamespace(kind="WAIT", explanation="waiting for state")
        if len(notifications) == 2:
            handlers[-1](SimpleNamespace(content="submitted UP"))
            channels["game_states"].put("state-1")
            return SimpleNamespace(kind="DONE", explanation="turn complete")
        current_state = {"state": "WIN", "turn": 1}
        handlers[-1](SimpleNamespace(content="game solved"))
        return SimpleNamespace(kind="DONE", explanation="game solved")

    def continue_after(_agent: Any, reason: str, _explanation: str) -> bool:
        return (
            reason == "DONE"
            and current_state is not None
            and (current_state["state"] != "WIN")
        )

    agent.handle.side_effect = handle
    install_target(
        MagicMock(
            return_value=adapter.InteractiveAgentTarget(
                agent=agent,
                continue_after=continue_after,
            )
        )
    )
    runtime = adapter.NooaRuntime()

    await runtime.start(_start_payload(tmp_path))
    try:
        result = await runtime.invoke(*_invocation("solve the game"))
    finally:
        await runtime.stop()

    assert notifications == [
        {"user_messages": ["solve the game"]},
        {"game_states": ["state-0"]},
        {"game_states": ["state-1"]},
    ]
    assert result.output["messages"] == [
        {"content": "submitted UP"},
        {"content": "game solved"},
    ]
    assert result.output["response"] == "game solved"
    assert result.output["reason"] == "DONE"


async def test_partial_start_failure_closes_the_factory_result(
    tmp_path: Path,
    install_target,
):
    invalid_agent, _channels, _handlers = _agent_double()
    invalid_agent.handle = None
    cleanup = AsyncMock()
    install_target(
        MagicMock(
            return_value=adapter.InteractiveAgentTarget(
                agent=invalid_agent,
                close=cleanup,
            )
        )
    )
    runtime = adapter.NooaRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        await runtime.start(_start_payload(tmp_path))

    assert error.value.code == "nooa_invalid_interactive_agent"
    cleanup.assert_awaited_once_with()
    await runtime.stop()
    cleanup.assert_awaited_once_with()


async def test_independent_runtimes_do_not_share_agents(
    tmp_path: Path,
    install_target,
):
    first_agent, _first_channels, first_handlers = _agent_double()
    second_agent, _second_channels, second_handlers = _agent_double()

    async def first_handle(_notification: dict[str, list[Any]]):
        first_handlers[-1](SimpleNamespace(content="first runtime"))
        return SimpleNamespace(kind="DONE", explanation="first complete")

    async def second_handle(_notification: dict[str, list[Any]]):
        second_handlers[-1](SimpleNamespace(content="second runtime"))
        return SimpleNamespace(kind="DONE", explanation="second complete")

    first_agent.handle.side_effect = first_handle
    second_agent.handle.side_effect = second_handle
    factory = MagicMock(side_effect=[first_agent, second_agent])
    install_target(factory)
    first_runtime = adapter.NooaRuntime()
    second_runtime = adapter.NooaRuntime()

    await first_runtime.start(_start_payload(tmp_path, runtime_id="runtime-1"))
    await second_runtime.start(_start_payload(tmp_path, runtime_id="runtime-2"))
    try:
        first_result = await first_runtime.invoke(
            *_invocation(runtime_id="runtime-1", request_id="first")
        )
        second_result = await second_runtime.invoke(
            *_invocation(runtime_id="runtime-2", request_id="second")
        )
    finally:
        await first_runtime.stop()
        await second_runtime.stop()

    assert factory.call_count == 2
    assert first_result.output["response"] == "first runtime"
    assert second_result.output["response"] == "second runtime"
    first_agent.close.assert_awaited_once_with()
    second_agent.close.assert_awaited_once_with()


async def test_runtime_mismatch_is_a_lifecycle_failure(
    tmp_path: Path,
    install_target,
):
    agent, _channels, _handlers = _agent_double()
    install_target(MagicMock(return_value=agent))
    runtime = adapter.NooaRuntime()

    await runtime.start(_start_payload(tmp_path))
    try:
        with pytest.raises(adapter.lifecycle.LifecycleError) as error:
            await runtime.invoke(*_invocation(runtime_id="runtime-2"))
    finally:
        await runtime.stop()

    assert error.value.code == "nooa_runtime_mismatch"
    agent.handle.assert_not_awaited()


async def test_factory_failure_is_redacted(
    tmp_path: Path,
    install_target,
):
    install_target(MagicMock(side_effect=RuntimeError("api-key=super-secret")))
    runtime = adapter.NooaRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        await runtime.start(_start_payload(tmp_path))

    assert error.value.code == "nooa_target_start_failed"
    assert "super-secret" not in str(error.value)
    await runtime.stop()


async def test_runtime_registers_and_activates_whole_mcp_servers(
    tmp_path: Path,
    install_target,
    monkeypatch: pytest.MonkeyPatch,
):
    agent, _channels, _handlers = _agent_double()
    install_target(MagicMock(return_value=agent))
    stdio_tool = MagicMock(name="stdio_tool")
    http_tool = MagicMock(name="http_tool")
    manager = MagicMock(name="MCPManager")
    manager.create_stdio_server = AsyncMock(return_value=stdio_tool)
    manager.create_url_server = AsyncMock(return_value=http_tool)
    nooa_module = types.ModuleType("nooa")
    nooa_module.__path__ = []  # type: ignore[attr-defined]
    mcp_module = types.ModuleType("nooa.mcp")
    mcp_module.MCPManager = manager
    monkeypatch.setitem(sys.modules, "nooa", nooa_module)
    monkeypatch.setitem(sys.modules, "nooa.mcp", mcp_module)
    os.environ["NOOA_TEST_MCP_COMMAND"] = sys.executable
    os.environ["MCP_ACCESS_TOKEN"] = "test-token"

    config = AgentConfig.from_mapping(
        {
            "mcp": {
                "servers": {
                    "calculator": {
                        "transport": "stdio",
                        "url": "$NOOA_TEST_MCP_COMMAND",
                        "args": ["calculator_server.py"],
                        "env": {"MODE": "test"},
                    },
                    "repository": {
                        "transport": "streamable-http",
                        "url": "https://mcp.example.test/api",
                        "custom_headers": {
                            "Authorization": "Bearer ${MCP_ACCESS_TOKEN}"
                        },
                    },
                }
            },
            "workflow": _workflow(),
        }
    )
    payload = _start_payload(tmp_path)
    payload["config"] = config
    runtime = adapter.NooaRuntime()

    await runtime.start(payload)
    await runtime.stop()

    manager.create_stdio_server.assert_awaited_once_with(
        "calculator",
        command=sys.executable,
        args=["calculator_server.py"],
        env={"MODE": "test"},
    )
    manager.create_url_server.assert_awaited_once_with(
        "repository",
        "https://mcp.example.test/api",
        headers={"Authorization": "Bearer test-token"},
        transport="streamable-http",
    )
    assert agent.skills.register.call_args_list == [
        call("mcp.calculator", stdio_tool),
        call("mcp.repository", http_tool),
    ]
    agent.skills.activate.assert_called_once_with(["mcp.calculator", "mcp.repository"])
    agent.close.assert_awaited_once_with()


async def test_runtime_rejects_invalid_mcp_header(
    tmp_path: Path,
    install_target,
    monkeypatch: pytest.MonkeyPatch,
):
    agent, _channels, _handlers = _agent_double()
    install_target(MagicMock(return_value=agent))
    manager = MagicMock(name="MCPManager")
    manager.create_url_server = AsyncMock()
    nooa_module = types.ModuleType("nooa")
    nooa_module.__path__ = []  # type: ignore[attr-defined]
    mcp_module = types.ModuleType("nooa.mcp")
    mcp_module.MCPManager = manager
    monkeypatch.setitem(sys.modules, "nooa", nooa_module)
    monkeypatch.setitem(sys.modules, "nooa.mcp", mcp_module)
    config = AgentConfig.from_mapping(
        {
            "mcp": {
                "servers": {
                    "repository": {
                        "transport": "streamable-http",
                        "url": "https://mcp.example.test/api",
                        "custom_headers": {"Invalid Header": "value"},
                    }
                }
            },
            "workflow": _workflow(),
        }
    )
    payload = _start_payload(tmp_path)
    payload["config"] = config
    runtime = adapter.NooaRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        await runtime.start(payload)

    assert error.value.code == "nooa_invalid_mcp_server"
    assert error.value.metadata == {"server": "repository"}
    manager.create_url_server.assert_not_awaited()
    agent.close.assert_awaited_once_with()


async def test_runtime_rejects_mcp_for_target_without_skill_registry(
    tmp_path: Path,
    install_target,
):
    agent, _channels, _handlers = _agent_double()
    agent.skills = None
    install_target(MagicMock(return_value=agent))
    config = AgentConfig.from_mapping(
        {
            "mcp": {
                "servers": {
                    "calculator": {
                        "transport": "stdio",
                        "url": sys.executable,
                    }
                }
            },
            "workflow": _workflow(),
        }
    )
    payload = _start_payload(tmp_path)
    payload["config"] = config
    runtime = adapter.NooaRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        await runtime.start(payload)

    assert error.value.code == "nooa_mcp_unsupported_target"
    agent.close.assert_awaited_once_with()


@pytest.mark.parametrize(
    ("policy", "expected_code"),
    [
        ({"allowed_tools": []}, "nooa_mcp_tool_filters_unsupported"),
        ({"blocked_tools": ["subtract"]}, "nooa_mcp_tool_filters_unsupported"),
        (
            {
                "authentication": {
                    "type": "service_account",
                    "client_id": "client",
                    "client_secret_env": "CLIENT_SECRET",
                    "token_url": "https://identity.example.test/token",
                }
            },
            "nooa_mcp_authentication_unsupported",
        ),
    ],
)
async def test_runtime_rejects_unsupported_mcp_policy(
    tmp_path: Path,
    install_target,
    policy: dict[str, Any],
    expected_code: str,
):
    agent, _channels, _handlers = _agent_double()
    install_target(MagicMock(return_value=agent))
    config = AgentConfig.from_mapping(
        {
            "mcp": {
                "servers": {
                    "calculator": {
                        "transport": "stdio",
                        "url": sys.executable,
                        **policy,
                    }
                }
            },
            "workflow": _workflow(),
        }
    )
    payload = _start_payload(tmp_path)
    payload["config"] = config
    runtime = adapter.NooaRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as error:
        await runtime.start(payload)

    assert error.value.code == expected_code
    agent.close.assert_awaited_once_with()


async def test_model_construction_preserves_original_error_when_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    model = MagicMock(name="model")
    model.aclose = AsyncMock(side_effect=RuntimeError("cleanup failed"))
    get_llm_client = MagicMock(side_effect=[model, ValueError("construction failed")])
    nooa_module = types.ModuleType("nooa")
    nooa_module.__path__ = []  # type: ignore[attr-defined]
    unifiedllm_module = types.ModuleType("nooa.unifiedllm")
    unifiedllm_module.get_llm_client = get_llm_client
    monkeypatch.setitem(sys.modules, "nooa", nooa_module)
    monkeypatch.setitem(sys.modules, "nooa.unifiedllm", unifiedllm_module)
    os.environ["OPENAI_API_KEY"] = "test-key"
    config = AgentConfig.from_mapping(
        {
            "models": {
                "first": {"provider": "openai", "model": "openai/first"},
                "second": {"provider": "openai", "model": "openai/second"},
            },
            "workflow": _workflow(),
        }
    )

    with pytest.raises(ValueError, match="construction failed") as error:
        await model_support.build_models(config)

    assert error.value.__notes__ == [
        "OO Agents model cleanup also failed (RuntimeError)"
    ]
    model.aclose.assert_awaited_once_with()


def test_arc_solver_stops_when_latest_state_is_unavailable():
    agent = MagicMock(name="arc_solver")
    agent.latest_state.return_value = None

    assert arc_solver._continue_arc_session(agent, "DONE", "turn complete") is False


async def test_coding_agent_target_runs_with_normalized_model_and_skills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    relay_invocations = _install_deterministic_relay(monkeypatch)
    agent, _channels, handlers = _agent_double()

    async def handle(_notification: dict[str, list[Any]]):
        handlers[-1](SimpleNamespace(content="No correctness issues found."))
        return SimpleNamespace(kind="DONE", explanation="review complete")

    agent.handle.side_effect = handle
    mock_coding_agent = MagicMock(name="CodingAgent", return_value=agent)
    mock_context = MagicMock(name="Context", side_effect=lambda value, **_kwargs: value)
    code_review_skill = MagicMock(name="code_review_skill")
    code_review_skill.id = "code-review"
    security_skill = MagicMock(name="security_skill")
    security_skill.id = "security-review"
    mock_text_skill = MagicMock(
        name="TextSkill",
        side_effect=[code_review_skill, security_skill],
    )
    mock_llm = MagicMock(name="llm")
    mock_llm.aclose = AsyncMock()
    mock_get_llm_client = MagicMock(name="get_llm_client", return_value=mock_llm)

    modules: dict[str, types.ModuleType] = {}
    for name in (
        "nooa",
        "nooa.skill",
        "nooa.unifiedllm",
        "nooa_cli",
        "nooa_cli.coding",
    ):
        module = types.ModuleType(name)
        if name in {"nooa", "nooa_cli"}:
            module.__path__ = []  # type: ignore[attr-defined]
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    modules["nooa"].Context = mock_context
    modules["nooa.skill"].TextSkill = mock_text_skill
    modules["nooa.unifiedllm"].get_llm_client = mock_get_llm_client
    modules["nooa.unifiedllm"].ResponsesClient = type("ResponsesClient", (), {})
    modules["nooa_cli.coding"].CodingAgent = mock_coding_agent

    skill_path = tmp_path / "skills" / "code-review"
    skill_path.mkdir(parents=True)
    (skill_path / "SKILL.md").write_text(
        "---\nname: code-review\n---\n", encoding="utf-8"
    )
    security_skill_path = tmp_path / "skills" / "security-review"
    security_skill_path.mkdir(parents=True)
    (security_skill_path / "SKILL.md").write_text(
        "---\nname: security-review\n---\n", encoding="utf-8"
    )
    os.environ["NVIDIA_API_KEY"] = "test-key"
    config = AgentConfig.from_mapping(
        {
            "models": {
                "default": {
                    "provider": "nvidia",
                    "model": "nvidia/test-model",
                    "api_key_env": "NVIDIA_API_KEY",
                    "base_url": "https://integrate.api.nvidia.com/v1",
                    "temperature": 0.0,
                }
            },
            "instructions": {
                "system": {"content": "Review the code carefully."},
            },
            "skills": {
                "paths": [
                    str(skill_path),
                    str(skill_path),
                    str(security_skill_path),
                ]
            },
            "workflow": {
                "entrypoint": {
                    "kind": "interactive_agent_factory",
                    "ref": "nemo_fabric_adapters.nooa.targets.coding_agent:create_agent",
                },
                "settings": {},
            },
        }
    )
    payload = _start_payload(tmp_path)
    payload["config"] = config
    runtime = adapter.NooaRuntime()

    await runtime.start(payload)
    try:
        result = await runtime.invoke(*_invocation("Review calculator.py"))
    finally:
        await runtime.stop()

    mock_get_llm_client.assert_called_once_with(
        "nvidia_nim/nvidia/test-model",
        client_type=None,
        api_key="test-key",
        api_base="https://integrate.api.nvidia.com/v1",
        temperature=0.0,
    )
    mock_coding_agent.assert_called_once_with(
        llm=mock_llm,
        cwd=tmp_path / "workspace",
        libs_dir=tmp_path / "artifacts" / "nooa-libs",
    )
    mock_context.assert_called_once_with("Review the code carefully.", prefix=True)
    assert mock_text_skill.call_args_list == [
        call(path=skill_path),
        call(path=security_skill_path),
    ]
    assert agent.skills.register.call_args_list == [
        call("cmd.code-review", code_review_skill),
        call("cmd.security-review", security_skill),
    ]
    agent.skills.activate.assert_called_once_with(
        ["cmd.code-review", "cmd.security-review"]
    )
    assert result.status is AgentRunStatus.SUCCEEDED
    assert result.output["response"] == "No correctness issues found."
    assert relay_invocations == [agent]
    assert {item["kind"] for item in result.output["relay_artifacts"]} == {
        "atof",
        "atif",
    }
    agent.close.assert_awaited_once_with()
    mock_llm.aclose.assert_awaited_once_with()


async def test_arc_solver_target_runs_custom_queue_to_harness_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    relay_invocations = _install_deterministic_relay(monkeypatch)
    agent, channels, handlers = _agent_double(
        channel_names=("user_messages", "game_states")
    )
    current_state: dict[str, Any] | None = None

    def latest_state() -> dict[str, Any] | None:
        return current_state

    agent.latest_state.side_effect = latest_state

    async def handle(notification: dict[str, list[Any]]):
        nonlocal current_state
        if "user_messages" in notification:
            current_state = {"state": "NOT_FINISHED", "turn": 0}
            channels["game_states"].put("initial-state")
            return SimpleNamespace(kind="WAIT", explanation="waiting for the harness")
        if notification == {"game_states": ["initial-state"]}:
            handlers[-1](SimpleNamespace(content="submitted action UP"))
            channels["game_states"].put("winning-state")
            return SimpleNamespace(kind="DONE", explanation="agent turn ended")
        current_state = {"state": "WIN", "turn": 1}
        handlers[-1](SimpleNamespace(content="solved the game"))
        return SimpleNamespace(kind="DONE", explanation="game solved")

    agent.handle.side_effect = handle
    mock_solver = MagicMock(name="MdArcSolverAgent", return_value=agent)
    mock_context = MagicMock(name="Context", side_effect=lambda value, **_kwargs: value)
    mock_llm = MagicMock(name="llm")
    mock_llm.aclose = AsyncMock()
    mock_get_llm_client = MagicMock(name="get_llm_client", return_value=mock_llm)

    modules: dict[str, types.ModuleType] = {}
    for name in (
        "solver_agent",
        "nooa",
        "nooa.unifiedllm",
    ):
        module = types.ModuleType(name)
        if name == "nooa":
            module.__path__ = []  # type: ignore[attr-defined]
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    modules["solver_agent"].MdArcSolverAgent = mock_solver
    modules["nooa"].Context = mock_context
    modules["nooa.unifiedllm"].get_llm_client = mock_get_llm_client
    modules["nooa.unifiedllm"].ResponsesClient = type("ResponsesClient", (), {})

    skill_path = tmp_path / "skills" / "grid-game-solver"
    skill_path.mkdir(parents=True)
    skill_file = skill_path / "SKILL.md"
    skill_file.write_text("---\nname: grid-game-solver\n---\n", encoding="utf-8")
    os.environ["NVIDIA_API_KEY"] = "test-key"
    config = AgentConfig.from_mapping(
        {
            "models": {
                "default": {
                    "provider": "nvidia",
                    "model": "nvidia/test-model",
                }
            },
            "instructions": {
                "system": {"content": "Solve only the anonymous game."},
            },
            "skills": {"paths": [str(skill_path)]},
            "workflow": {
                "entrypoint": {
                    "kind": "interactive_agent_factory",
                    "ref": "nemo_fabric_adapters.nooa.targets.arc_solver:create_agent",
                },
                "settings": {
                    "reflect_every": 4,
                    "visual": "off",
                    "png_scale": 6,
                    "max_actions_per_turn": 3,
                },
            },
        }
    )
    payload = _start_payload(tmp_path)
    payload["config"] = config
    runtime = adapter.NooaRuntime()

    await runtime.start(payload)
    try:
        result = await runtime.invoke(*_invocation("start solving"))
    finally:
        await runtime.stop()

    mock_solver.assert_called_once_with(
        llm=mock_llm,
        run_dir=tmp_path / "artifacts" / "nooa-arc",
        game_id="the game",
        alias="the game",
        reflect_every=4,
        visual="off",
        png_scale=6,
        max_actions_per_turn=3,
        skill_path=skill_file,
    )
    mock_context.assert_called_once_with("Solve only the anonymous game.", prefix=True)
    assert result.status is AgentRunStatus.SUCCEEDED
    assert result.output["messages"] == [
        {"content": "submitted action UP"},
        {"content": "solved the game"},
    ]
    assert result.output["response"] == "solved the game"
    assert relay_invocations == [agent]
    assert {item["kind"] for item in result.output["relay_artifacts"]} == {
        "atof",
        "atif",
    }
    assert agent.handle.await_count == 3
    agent.close.assert_awaited_once_with()
    mock_llm.aclose.assert_awaited_once_with()


class _HeldReply:
    """A model reply that a test releases once the call is in flight."""

    def __init__(self) -> None:
        self.requested = asyncio.Event()
        self.reply: asyncio.Future[Any] = asyncio.get_running_loop().create_future()


@pytest.fixture(name="model_replies")
def model_replies_fixture(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Queue what NOOA's real model clients receive from LiteLLM, in call order.

    An exception in the queue is raised, and a ``_HeldReply`` holds its call in
    flight until the test releases it.
    """
    pytest.importorskip("nooa.unifiedllm")
    import litellm

    replies: list[Any] = []

    def next_reply() -> Any:
        reply = replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    async def async_reply(**_kwargs: Any) -> Any:
        reply = next_reply()
        if isinstance(reply, _HeldReply):
            reply.requested.set()
            return await reply.reply
        return reply

    def sync_reply(**_kwargs: Any) -> Any:
        return next_reply()

    monkeypatch.setattr(litellm, "acompletion", async_reply)
    monkeypatch.setattr(litellm, "aresponses", async_reply)
    monkeypatch.setattr(litellm, "completion", sync_reply)
    monkeypatch.setattr(litellm, "responses", sync_reply)
    os.environ["OPENAI_API_KEY"] = "test-key"
    return replies


def _chat_completion_reply(usage: dict[str, Any] | None, content: str = "done") -> Any:
    """Build a reply the way LiteLLM converts a Chat Completions body."""
    import litellm
    from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
        convert_to_model_response_object,
    )

    body: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "fixture-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
    if usage is not None:
        body["usage"] = usage
    return convert_to_model_response_object(
        response_object=body, model_response_object=litellm.ModelResponse()
    )


def _responses_api_reply(usage: Any) -> Any:
    """Build a Responses API reply; a usage mapping goes through LiteLLM's model."""
    from litellm.types.llms.openai import ResponseAPIUsage
    from litellm.types.llms.openai import ResponsesAPIResponse

    reply = ResponsesAPIResponse(
        id="resp-1",
        created_at=0,
        model="fixture-model",
        object="response",
        status="completed",
        output=[
            {
                "type": "message",
                "id": "msg-1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "done", "annotations": []}],
            }
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )
    if isinstance(usage, dict):
        usage = ResponseAPIUsage(**usage)
    return reply.model_copy(update={"usage": usage})


def _model_reply(client_type: str, **counts: int) -> Any:
    """Build one reply with input, cached, output, and total counts for a client type."""
    if client_type == "responses":
        return _responses_api_reply(
            {
                "input_tokens": counts["input"],
                "input_tokens_details": {"cached_tokens": counts["cached"]},
                "output_tokens": counts["output"],
                "total_tokens": counts["total"],
            }
        )
    return _chat_completion_reply(
        {
            "prompt_tokens": counts["input"],
            "prompt_tokens_details": {"cached_tokens": counts["cached"]},
            "completion_tokens": counts["output"],
            "total_tokens": counts["total"],
        }
    )


_FULL_USAGE = {
    "input_tokens": 100,
    "cached_input_tokens": 40,
    "input_tokens_include_cache": True,
    "output_tokens": 20,
    "total_tokens": 120,
}
_MESSAGES = [{"role": "user", "content": "Fix the failing test."}]


def _model_start_payload(tmp_path: Path, client_type: str) -> dict[str, Any]:
    payload = _start_payload(tmp_path)
    payload["config"] = AgentConfig.from_mapping(
        {
            "models": {
                "default": {
                    "provider": "openai",
                    "model": "openai/fixture-model",
                    "base_url": "https://model.example.test/v1",
                    "settings": {"client_type": client_type},
                }
            },
            "workflow": _workflow(),
        }
    )
    return payload


def _install_model_target(
    install_target: Any,
    turn: Callable[[Any], Awaitable[None]],
) -> None:
    """Install a target whose every turn runs ``turn`` with the adapter-built model."""
    agent, _channels, _handlers = _agent_double()

    def create_agent(context: adapter.InteractiveAgentBuildContext) -> Any:
        model = context.models["default"]

        async def handle(_notification: dict[str, list[Any]]):
            await turn(model)
            return SimpleNamespace(kind="DONE", explanation="turn complete")

        agent.handle.side_effect = handle
        return agent

    install_target(create_agent)


async def _invoke_once(
    tmp_path: Path,
    install_target: Any,
    client_type: str,
    turn: Callable[[Any], Awaitable[None]],
) -> Any:
    _install_model_target(install_target, turn)
    runtime = adapter.NooaRuntime()
    await runtime.start(_model_start_payload(tmp_path, client_type))
    try:
        return await runtime.invoke(*_invocation("Fix the failing test."))
    finally:
        await runtime.stop()


async def _call_until_empty(replies: list[Any], model: Any) -> None:
    while replies:
        await model.acall(_MESSAGES)


@pytest.mark.parametrize("client_type", ["completion", "responses"])
async def test_model_usage_sums_every_call_of_the_invocation(
    tmp_path: Path,
    install_target,
    model_replies: list[Any],
    client_type: str,
):
    model_replies.extend(
        [
            _model_reply(client_type, input=100, cached=40, output=20, total=120),
            _model_reply(client_type, input=150, cached=90, output=30, total=180),
        ]
    )

    async def turn(model: Any) -> None:
        await model.acall(_MESSAGES)
        model.call(_MESSAGES)

    result = await _invoke_once(tmp_path, install_target, client_type, turn)

    assert not model_replies
    assert result.status is AgentRunStatus.SUCCEEDED
    assert result.usage is not None
    assert result.usage.to_mapping() == {
        "input_tokens": 250,
        "cached_input_tokens": 130,
        "input_tokens_include_cache": True,
        "output_tokens": 50,
        "total_tokens": 300,
    }


def _construct_responses_usage(**counts: Any) -> Any:
    from litellm.types.llms.openai import ResponseAPIUsage

    return ResponseAPIUsage.model_construct(**counts)


@pytest.mark.parametrize(
    ("client_type", "make_reply", "expected"),
    [
        pytest.param(
            "completion",
            lambda: _chat_completion_reply(
                {"completion_tokens": 20, "total_tokens": 120}
            ),
            {"output_tokens": 20, "total_tokens": 120},
            id="completion-prompt-absent-zero-filled",
        ),
        pytest.param(
            "completion",
            lambda: _chat_completion_reply(
                {"prompt_tokens": 0, "completion_tokens": 20, "total_tokens": 20}
            ),
            {"output_tokens": 20, "total_tokens": 20},
            id="completion-prompt-zero",
        ),
        pytest.param(
            "completion",
            lambda: _chat_completion_reply(
                {"prompt_tokens": -1, "completion_tokens": 20, "total_tokens": 120}
            ),
            {"output_tokens": 20, "total_tokens": 120},
            id="completion-prompt-negative",
        ),
        pytest.param(
            "completion",
            lambda: _chat_completion_reply({"prompt_tokens": 60}),
            {"input_tokens": 60, "input_tokens_include_cache": True},
            id="completion-output-and-total-zero-filled",
        ),
        pytest.param(
            "completion",
            lambda: _chat_completion_reply(
                {
                    "prompt_tokens": 60,
                    "prompt_tokens_details": {"cached_tokens": 0},
                    "completion_tokens": 5,
                    "total_tokens": 65,
                }
            ),
            {
                "input_tokens": 60,
                "cached_input_tokens": 0,
                "input_tokens_include_cache": True,
                "output_tokens": 5,
                "total_tokens": 65,
            },
            id="completion-explicit-cached-zero",
        ),
        pytest.param(
            "completion",
            lambda: _chat_completion_reply(
                {
                    "prompt_tokens": 60,
                    "prompt_tokens_details": {"cached_tokens": None},
                    "completion_tokens": 5,
                    "total_tokens": 65,
                }
            ),
            {
                "input_tokens": 60,
                "input_tokens_include_cache": True,
                "output_tokens": 5,
                "total_tokens": 65,
            },
            id="completion-cached-null",
        ),
        pytest.param(
            "completion",
            lambda: _chat_completion_reply(
                {"prompt_tokens": 60, "completion_tokens": 5, "total_tokens": 1 << 64}
            ),
            {
                "input_tokens": 60,
                "input_tokens_include_cache": True,
                "output_tokens": 5,
            },
            id="completion-total-past-uint64",
        ),
        pytest.param(
            "responses",
            lambda: _responses_api_reply(
                _construct_responses_usage(output_tokens=5, total_tokens=5)
            ),
            {"output_tokens": 5, "total_tokens": 5},
            id="responses-input-absent-defensive",
        ),
        pytest.param(
            "responses",
            lambda: _responses_api_reply(
                {"input_tokens": 60, "output_tokens": 0, "total_tokens": 60}
            ),
            {
                "input_tokens": 60,
                "input_tokens_include_cache": True,
                "output_tokens": 0,
                "total_tokens": 60,
            },
            id="responses-explicit-output-zero",
        ),
        pytest.param(
            "responses",
            lambda: _responses_api_reply(
                {"input_tokens": -1, "output_tokens": 5, "total_tokens": 5}
            ),
            {"output_tokens": 5, "total_tokens": 5},
            id="responses-input-negative",
        ),
        pytest.param(
            "responses",
            lambda: _responses_api_reply(
                {
                    "input_tokens": 60,
                    "input_tokens_details": {"cached_tokens": 1 << 64},
                    "output_tokens": 5,
                    "total_tokens": (1 << 64) - 1,
                }
            ),
            {
                "input_tokens": 60,
                "input_tokens_include_cache": True,
                "output_tokens": 5,
                "total_tokens": (1 << 64) - 1,
            },
            id="responses-cached-past-uint64-total-at-max",
        ),
    ],
)
async def test_model_usage_keeps_only_reported_counts(
    tmp_path: Path,
    install_target,
    model_replies: list[Any],
    client_type: str,
    make_reply: Callable[[], Any],
    expected: dict[str, Any],
):
    model_replies.append(make_reply())

    result = await _invoke_once(
        tmp_path,
        install_target,
        client_type,
        lambda model: _call_until_empty(model_replies, model),
    )

    assert not model_replies
    assert result.usage is not None
    assert result.usage.to_mapping() == expected


@pytest.mark.parametrize(
    ("client_type", "reply_without_usage"),
    [
        ("completion", lambda: _chat_completion_reply(None)),
        ("responses", lambda: _responses_api_reply(None)),
    ],
)
async def test_model_call_without_usage_leaves_usage_unknown(
    tmp_path: Path,
    install_target,
    model_replies: list[Any],
    client_type: str,
    reply_without_usage: Callable[[], Any],
):
    model_replies.extend(
        [
            _model_reply(client_type, input=100, cached=40, output=20, total=120),
            reply_without_usage(),
        ]
    )

    result = await _invoke_once(
        tmp_path,
        install_target,
        client_type,
        lambda model: _call_until_empty(model_replies, model),
    )

    assert not model_replies
    assert result.status is AgentRunStatus.SUCCEEDED
    assert result.usage is None


@pytest.mark.parametrize("client_type", ["completion", "responses"])
async def test_model_usage_sums_partial_reports_per_count(
    tmp_path: Path,
    install_target,
    model_replies: list[Any],
    client_type: str,
):
    partial = (
        _responses_api_reply(
            _construct_responses_usage(output_tokens=5, total_tokens=5)
        )
        if client_type == "responses"
        else _chat_completion_reply({"completion_tokens": 5, "total_tokens": 5})
    )
    model_replies.extend(
        [_model_reply(client_type, input=100, cached=40, output=20, total=120), partial]
    )

    result = await _invoke_once(
        tmp_path,
        install_target,
        client_type,
        lambda model: _call_until_empty(model_replies, model),
    )

    assert result.usage is not None
    assert result.usage.to_mapping() == {"output_tokens": 25, "total_tokens": 125}


async def test_model_call_that_raises_is_not_counted(
    tmp_path: Path,
    install_target,
    model_replies: list[Any],
):
    model_replies.extend(
        [
            RuntimeError("provider unavailable"),
            _model_reply("completion", input=100, cached=40, output=20, total=120),
        ]
    )

    async def turn(model: Any) -> None:
        with pytest.raises(RuntimeError, match="provider unavailable"):
            await model.acall(_MESSAGES)
        await model.acall(_MESSAGES)

    result = await _invoke_once(tmp_path, install_target, "completion", turn)

    assert result.usage is not None
    assert result.usage.to_mapping() == _FULL_USAGE


def _usage_report(input_tokens: int, output_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _known_usage(input_tokens: int, output_tokens: int) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "input_tokens_include_cache": True,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _answer(input_tokens: int, output_tokens: int, value: str) -> Any:
    """A Chat Completions reply that NOOA's PredictStrategy parses as ``value``."""
    return _chat_completion_reply(
        _usage_report(input_tokens, output_tokens), json.dumps({"value": value})
    )


def _install_predicting_target(
    install_target: Any,
    *,
    summarize: bool,
    after_answer: Callable[[Any], None] | None = None,
) -> None:
    """Install a real NOOA InteractiveAgent whose turns make one PredictStrategy call.

    With ``summarize``, it also installs NOOA's history summarizer. CodingAgent
    installs ``SummarizationConfig()``; the 500-token threshold stands in for a
    long session, so a 1000-token turn starts a summary at its ``AfterTurn``.
    ``after_answer`` runs with the agent after each turn's call.
    """
    pytest.importorskip("nooa.interactive")
    from nooa import strategy
    from nooa.interactive import InteractiveAgent
    from nooa.interactive import RespondReason
    from nooa.interactive import RespondResult
    from nooa.interactive import SummarizationConfig
    from nooa.interactive import install_summarizer
    from nooa.strategies import PredictStrategy

    class PredictingAgent(InteractiveAgent):
        @strategy(PredictStrategy())
        async def answer(self, question: str) -> str:
            """Answer the question in one word."""
            ...

        async def handle(self, notification: dict[str, list[Any]]) -> Any:
            await self.answer(str(notification.get("user_messages", [])))
            if after_answer is not None:
                after_answer(self)
            return RespondResult(kind=RespondReason.DONE, explanation="answered")

    def create_agent(context: adapter.InteractiveAgentBuildContext) -> Any:
        agent = PredictingAgent(llm=context.models["default"])
        agent.queue_manager.queue("background")
        if summarize:
            install_summarizer(
                SummarizationConfig(max_tokens=500, preserve_recent=0), agent
            )
        return agent

    install_target(create_agent)


def _summary_task(runtime: adapter.NooaRuntime) -> asyncio.Task[Any]:
    assert runtime._target is not None
    task = runtime._target.agent._summarizers[0]._pending_task
    assert isinstance(task, asyncio.Task)
    return task


async def test_summary_started_on_last_turn_counts_toward_its_invocation(
    tmp_path: Path,
    install_target,
    model_replies: list[Any],
):
    summary = _HeldReply()
    model_replies.extend([_answer(1000, 10, "one"), summary, _answer(300, 3, "two")])
    _install_predicting_target(install_target, summarize=True)
    runtime = adapter.NooaRuntime()
    await runtime.start(_model_start_payload(tmp_path, "completion"))
    try:
        first = asyncio.create_task(
            runtime.invoke(*_invocation("one", request_id="one"))
        )
        await summary.requested.wait()
        await _drain_ready_callbacks()
        assert not first.done()
        summary.reply.set_result(_answer(7000, 70, "summary of the history"))
        first_result = await first
        second_result = await runtime.invoke(*_invocation("two", request_id="two"))
    finally:
        await runtime.stop()

    assert not model_replies
    assert first_result.usage is not None
    assert first_result.usage.to_mapping() == _known_usage(8000, 80)
    assert second_result.usage is not None
    assert second_result.usage.to_mapping() == _known_usage(300, 3)


@pytest.mark.parametrize("finishes", ["between_invocations", "during_next_invocation"])
async def test_summary_still_running_after_the_wait_leaves_usage_unknown(
    tmp_path: Path,
    install_target,
    model_replies: list[Any],
    monkeypatch: pytest.MonkeyPatch,
    finishes: str,
):
    monkeypatch.setattr(adapter, "_SUMMARIZER_WAIT_SECONDS", 0.0)
    summary = _HeldReply()
    next_answer = _HeldReply()
    model_replies.extend([_answer(1000, 10, "one"), summary, next_answer])
    _install_predicting_target(install_target, summarize=True)
    runtime = adapter.NooaRuntime()
    await runtime.start(_model_start_payload(tmp_path, "completion"))
    try:
        first_result = await runtime.invoke(*_invocation("one", request_id="one"))
        summary_task = _summary_task(runtime)
        assert summary.requested.is_set()
        assert not summary_task.done()
        if finishes == "between_invocations":
            summary.reply.set_result(_answer(7000, 70, "summary of the history"))
            await summary_task
        second = asyncio.create_task(
            runtime.invoke(*_invocation("two", request_id="two"))
        )
        await next_answer.requested.wait()
        if finishes == "during_next_invocation":
            summary.reply.set_result(_answer(7000, 70, "summary of the history"))
            await summary_task
        next_answer.reply.set_result(_answer(300, 3, "two"))
        second_result = await second
    finally:
        await runtime.stop()

    assert first_result.usage is None
    assert second_result.usage is not None
    assert second_result.usage.to_mapping() == _known_usage(300, 3)


async def _drain_ready_callbacks() -> None:
    """Let every ready callback on the loop run, without waiting on a clock."""
    for _ in range(100):
        await asyncio.sleep(0)


async def test_summary_wait_ends_at_the_invocation_deadline(
    tmp_path: Path,
    install_target,
    model_replies: list[Any],
):
    summary = _HeldReply()
    model_replies.extend([_answer(1000, 10, "one"), summary])
    _install_predicting_target(install_target, summarize=True)
    runtime = adapter.NooaRuntime()
    await runtime.start(_model_start_payload(tmp_path, "completion"))
    first = asyncio.create_task(
        runtime.invoke(*_invocation("one", request_id="one", deadline_millis=1))
    )
    try:
        await summary.requested.wait()
        await _drain_ready_callbacks()
        assert first.done()
        first_result = first.result()
        summary_task = _summary_task(runtime)
        assert not summary_task.done()
    finally:
        if not summary.reply.done():
            summary.reply.set_result(_answer(7000, 70, "summary of the history"))
        await first
        await runtime.stop()

    assert first_result.usage is None


async def test_queue_job_still_running_at_return_leaves_usage_unknown(
    tmp_path: Path,
    install_target,
    model_replies: list[Any],
):
    release_job = asyncio.Event()
    job_called = asyncio.Event()
    jobs: list[Any] = []

    async def job(agent: Any) -> None:
        await release_job.wait()
        await agent.llm.acall(_MESSAGES)
        job_called.set()

    def spawn_once(agent: Any) -> None:
        if not jobs:
            jobs.append(agent.queue_manager.spawn(job(agent), channel="background"))

    model_replies.extend(
        [
            _answer(1000, 10, "one"),
            _answer(7000, 70, "job"),
            _answer(300, 3, "two"),
        ]
    )
    _install_predicting_target(install_target, summarize=False, after_answer=spawn_once)
    runtime = adapter.NooaRuntime()
    await runtime.start(_model_start_payload(tmp_path, "completion"))
    try:
        first = await runtime.invoke(*_invocation("one", request_id="one"))
        assert jobs[0].state == "running"
        release_job.set()
        await job_called.wait()
        second = await runtime.invoke(*_invocation("two", request_id="two"))
    finally:
        await runtime.stop()

    assert not model_replies
    assert first.usage is None
    assert second.usage is not None
    assert second.usage.to_mapping() == _known_usage(300, 3)


async def test_cancelled_invocation_closes_its_usage_without_cancelling_the_summary(
    tmp_path: Path,
    install_target,
    model_replies: list[Any],
):
    summary = _HeldReply()
    model_replies.extend([_answer(1000, 10, "one"), summary])
    _install_predicting_target(install_target, summarize=True)
    runtime = adapter.NooaRuntime()
    await runtime.start(_model_start_payload(tmp_path, "completion"))
    try:
        first = asyncio.create_task(
            runtime.invoke(*_invocation("one", request_id="one"))
        )
        await summary.requested.wait()
        await _drain_ready_callbacks()
        assert not first.done()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        summary_task = _summary_task(runtime)
        owner = summary_task.get_context().get(adapter._INVOCATION_USAGE)
        assert owner is not None
        assert owner.closed
        assert not summary_task.done()
        summary.reply.set_result(_answer(7000, 70, "summary of the history"))
        await summary_task
    finally:
        await runtime.stop()

    assert len(owner.calls) == 1


@pytest.mark.parametrize("finishes", ["between_invocations", "during_next_invocation"])
async def test_call_returning_after_its_invocation_is_dropped(
    tmp_path: Path,
    install_target,
    model_replies: list[Any],
    finishes: str,
):
    late = _HeldReply()
    jobs: list[asyncio.Task[Any]] = []

    async def turn(model: Any) -> None:
        if not jobs:
            await model.acall(_MESSAGES)
            jobs.append(asyncio.create_task(model.acall(_MESSAGES)))
            await late.requested.wait()
            return
        if finishes == "during_next_invocation":
            late.reply.set_result(_answer(7000, 70, "late"))
            await jobs[0]
        await model.acall(_MESSAGES)

    model_replies.extend([_answer(1000, 10, "one"), late, _answer(300, 3, "two")])
    _install_model_target(install_target, turn)
    runtime = adapter.NooaRuntime()
    await runtime.start(_model_start_payload(tmp_path, "completion"))
    try:
        first = await runtime.invoke(*_invocation("one", request_id="one"))
        if finishes == "between_invocations":
            late.reply.set_result(_answer(7000, 70, "late"))
            await jobs[0]
        second = await runtime.invoke(*_invocation("two", request_id="two"))
    finally:
        await runtime.stop()

    assert jobs[0].done()
    assert first.usage is None
    assert second.usage is not None
    assert second.usage.to_mapping() == _known_usage(300, 3)


@pytest.mark.parametrize("relay", ["disabled", "enabled"])
async def test_relay_does_not_change_model_usage(
    tmp_path: Path,
    install_target,
    model_replies: list[Any],
    relay: str,
):
    model_replies.extend([_answer(1000, 10, "one"), _answer(2000, 20, "two")])
    _install_predicting_target(install_target, summarize=False)
    if relay == "enabled":
        pytest.importorskip("nemo_relay", reason="nemo-relay extra is required")
        runtime_context = _relay_context(tmp_path)
        config_path = Path(runtime_context.telemetry.config_path)
        config_path.write_text(
            json.dumps(
                {
                    "relay": {
                        "config": {
                            "version": 1,
                            "components": [
                                {
                                    "kind": "observability",
                                    "enabled": True,
                                    "config": {
                                        "version": 4,
                                        "atif": {
                                            "enabled": True,
                                            "output_directory": str(tmp_path / "relay"),
                                            "filename_template": (
                                                "trajectory-{session_id}.atif.json"
                                            ),
                                        },
                                    },
                                }
                            ],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(config_path)
    else:
        runtime_context = _invocation()[1]
    runtime = adapter.NooaRuntime()
    await runtime.start(_model_start_payload(tmp_path, "completion"))
    try:
        results = [
            await runtime.invoke(
                AgentRunRequest.from_mapping({"input": text}), runtime_context
            )
            for text in ("one", "two")
        ]
    finally:
        await runtime.stop()

    if relay == "enabled":
        assert results[0].output["telemetry"]["enabled"] is True
        assert "error" not in results[0].output["telemetry"]
    assert [result.usage.to_mapping() for result in results] == [
        _known_usage(1000, 10),
        _known_usage(2000, 20),
    ]


async def test_failed_invocation_keeps_model_usage_and_the_next_one_starts_fresh(
    tmp_path: Path,
    install_target,
    model_replies: list[Any],
):
    model_replies.extend(
        [
            _model_reply("responses", input=100, cached=40, output=20, total=120),
            _model_reply("responses", input=7, cached=0, output=3, total=10),
        ]
    )
    turns: list[str] = []

    async def turn(model: Any) -> None:
        await model.acall(_MESSAGES)
        turns.append("called")
        if len(turns) == 1:
            raise RuntimeError("tool crashed")

    _install_model_target(install_target, turn)
    runtime = adapter.NooaRuntime()
    await runtime.start(_model_start_payload(tmp_path, "responses"))
    try:
        failed = await runtime.invoke(*_invocation("one", request_id="one"))
        succeeded = await runtime.invoke(*_invocation("two", request_id="two"))
    finally:
        await runtime.stop()

    assert failed.status is AgentRunStatus.FAILED
    assert failed.error is not None
    assert failed.error.code == "nooa_target_invoke_failed"
    assert failed.usage is not None
    assert failed.usage.to_mapping() == _FULL_USAGE
    assert succeeded.status is AgentRunStatus.SUCCEEDED
    assert succeeded.usage is not None
    assert succeeded.usage.to_mapping() == {
        "input_tokens": 7,
        "cached_input_tokens": 0,
        "input_tokens_include_cache": True,
        "output_tokens": 3,
        "total_tokens": 10,
    }


def _relay_context(tmp_path: Path) -> RuntimeContext:
    config_path = tmp_path / "relay.json"
    config_path.write_text("{}", encoding="utf-8")
    return RuntimeContext.from_mapping(
        {
            "runtime_id": "runtime-1",
            "invocation_id": "invocation-relay",
            "request_id": "request-relay",
            "environment": {
                "environment_id": "environment-1",
                "provider": "test",
                "control_location": "in_env_control",
                "ownership": "caller_owned",
            },
            "artifacts": {},
            "telemetry": {
                "relay_enabled": True,
                "config_path": str(config_path),
                "env": {"FABRIC_RELAY_CONFIG_PATH": str(config_path)},
                "metadata": {"telemetry_providers": ["relay"]},
            },
        }
    )


def _install_relay_doubles(
    monkeypatch: pytest.MonkeyPatch,
    *,
    leak_scope: bool = False,
) -> tuple[MagicMock, list[dict[str, Any]]]:
    root_handle = SimpleNamespace(uuid="root")
    child_handle = SimpleNamespace(uuid="child")
    current = [root_handle]
    scope_metadata: list[dict[str, Any]] = []

    @contextmanager
    def enter_scope(_name: str, _kind: Any, *, metadata: dict[str, Any]):
        scope_metadata.append(metadata)
        current[0] = child_handle
        try:
            yield child_handle
        finally:
            if leak_scope:
                raise RuntimeError("scope leaked secret=do-not-report")
            current[0] = root_handle

    @asynccontextmanager
    async def activate(_config: dict[str, Any]):
        yield SimpleNamespace(report={"config": {"diagnostics": []}})

    install = MagicMock(name="install_nemo_relay")
    uninstall = MagicMock(name="uninstall_nemo_relay")
    install.return_value = uninstall
    relay_module = types.ModuleType("nemo_relay")
    relay_module.__path__ = []  # type: ignore[attr-defined]
    relay_module.ScopeType = SimpleNamespace(Agent="agent")
    relay_module.plugin = SimpleNamespace(activate=activate)
    relay_module.scope = MagicMock(name="scope")
    relay_module.scope.get_handle.side_effect = lambda: current[0]
    relay_module.scope.scope.side_effect = enter_scope
    nooa_module = types.ModuleType("nooa")
    nooa_module.__path__ = []  # type: ignore[attr-defined]
    middleware_module = types.ModuleType("nemo_fabric_adapters.nooa.relay_compat")
    middleware_module.install_nemo_relay_compat = install
    monkeypatch.setitem(sys.modules, "nemo_relay", relay_module)
    monkeypatch.setitem(sys.modules, "nooa", nooa_module)
    monkeypatch.setitem(
        sys.modules,
        "nemo_fabric_adapters.nooa.relay_compat",
        middleware_module,
    )
    return install, scope_metadata


@pytest.mark.parametrize(
    ("relay_version", "message"),
    [
        ("0.8.9", "nemo-relay must satisfy >=0.9,<0.11"),
        ("0.11.0", "nemo-relay must satisfy >=0.9,<0.11"),
        ("0.10.0rc1", "the installed nemo-relay version is not a stable release"),
    ],
)
def test_relay_version_rejects_unsupported_release(monkeypatch, relay_version, message):
    monkeypatch.setattr(
        nooa_telemetry.importlib.metadata, "version", lambda _name: relay_version
    )

    with pytest.raises(RuntimeError, match=re.escape(message)):
        nooa_telemetry._relay_version()


async def test_relay_lifecycle_correlates_once_and_collects_current_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime_context = _relay_context(tmp_path)
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(runtime_context.telemetry.config_path)
    monkeypatch.setattr(
        nooa_telemetry.importlib.metadata,
        "version",
        MagicMock(return_value="0.10.0"),
    )
    monkeypatch.setattr(
        nooa_telemetry.common_utils,
        "reject_ambient_relay_plugin_config",
        MagicMock(),
    )
    monkeypatch.setattr(
        nooa_telemetry.common_utils,
        "reject_inherited_relay_plugin_config",
        MagicMock(),
    )
    monkeypatch.setattr(
        nooa_telemetry, "_artifact_snapshot", MagicMock(return_value={})
    )
    monkeypatch.setattr(
        nooa_telemetry,
        "_changed_artifacts",
        MagicMock(return_value=({"kind": "atof", "path": "/safe/current.jsonl"},)),
    )
    monkeypatch.setattr(
        nooa_telemetry.relay_artifacts,
        "snapshot_atif_files",
        MagicMock(return_value={}),
    )
    monkeypatch.setattr(
        nooa_telemetry.relay_artifacts,
        "expects_local_atif",
        MagicMock(return_value=False),
    )
    install, scope_metadata = _install_relay_doubles(monkeypatch)
    config = AgentConfig.from_mapping({"workflow": _workflow()})
    telemetry = nooa_telemetry.RelayTelemetry(
        agent_name="test-agent",
        base_dir=tmp_path,
        config=config,
    )
    monkeypatch.setattr(
        telemetry, "_plugin_config", MagicMock(return_value={"components": []})
    )
    calls = 0

    async def invoke_target() -> str:
        nonlocal calls
        calls += 1
        return "terminal-result"

    agent = SimpleNamespace(event_manager=MagicMock())
    result = await telemetry.invoke(
        agent=agent,
        runtime_context=runtime_context,
        call=invoke_target,
    )

    assert calls == 1
    assert result.called is True
    assert result.result == "terminal-result"
    assert result.report == nooa_telemetry.RelayReport(
        enabled=True,
        artifacts=({"kind": "atof", "path": "/safe/current.jsonl"},),
    )
    install.assert_called_once_with(agent.event_manager)
    install.return_value.assert_called_once_with()
    assert scope_metadata == [
        {
            "nemo_fabric_request_id": "request-relay",
            "nemo_fabric_invocation_id": "invocation-relay",
            "nemo_fabric_runtime_id": "runtime-1",
        }
    ]


async def test_relay_records_none_result_and_collects_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime_context = _relay_context(tmp_path)
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(runtime_context.telemetry.config_path)
    monkeypatch.setattr(
        nooa_telemetry.importlib.metadata,
        "version",
        MagicMock(return_value="0.10.0"),
    )
    monkeypatch.setattr(
        nooa_telemetry.common_utils,
        "reject_ambient_relay_plugin_config",
        MagicMock(),
    )
    monkeypatch.setattr(
        nooa_telemetry.common_utils,
        "reject_inherited_relay_plugin_config",
        MagicMock(),
    )
    monkeypatch.setattr(
        nooa_telemetry, "_artifact_snapshot", MagicMock(return_value={})
    )
    changed_artifacts = MagicMock(
        return_value=({"kind": "atof", "path": "/safe/current.jsonl"},)
    )
    monkeypatch.setattr(nooa_telemetry, "_changed_artifacts", changed_artifacts)
    monkeypatch.setattr(
        nooa_telemetry.relay_artifacts,
        "snapshot_atif_files",
        MagicMock(return_value={}),
    )
    monkeypatch.setattr(
        nooa_telemetry.relay_artifacts,
        "expects_local_atif",
        MagicMock(return_value=False),
    )
    _install_relay_doubles(monkeypatch)
    telemetry = nooa_telemetry.RelayTelemetry(
        agent_name="test-agent",
        base_dir=tmp_path,
        config=AgentConfig.from_mapping({"workflow": _workflow()}),
    )
    monkeypatch.setattr(
        telemetry, "_plugin_config", MagicMock(return_value={"components": []})
    )
    call_target = AsyncMock(return_value=None)

    result = await telemetry.invoke(
        agent=SimpleNamespace(event_manager=MagicMock()),
        runtime_context=runtime_context,
        call=call_target,
    )

    assert result.called is True
    assert result.result is None
    assert result.report == nooa_telemetry.RelayReport(
        enabled=True,
        artifacts=({"kind": "atof", "path": "/safe/current.jsonl"},),
    )
    changed_artifacts.assert_called_once_with({"components": []}, {})
    call_target.assert_awaited_once_with()


@pytest.mark.parametrize(
    ("current", "expected"),
    [(None, True), (SimpleNamespace(uuid="child"), False)],
)
def test_relay_scope_comparison_handles_absent_baseline(
    monkeypatch: pytest.MonkeyPatch,
    current: Any,
    expected: bool,
):
    monkeypatch.setattr(
        nooa_telemetry,
        "_scope_handle",
        MagicMock(return_value=current),
    )

    assert nooa_telemetry._scope_unchanged(None) is expected


@pytest.mark.usefixtures("nemo_relay")
async def test_relay_emits_correlated_atof_and_atif(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime_context = _relay_context(tmp_path)
    config_path = Path(runtime_context.telemetry.config_path)
    config_path.write_text(
        json.dumps(
            {
                "relay": {
                    "config": {
                        "version": 1,
                        "components": [
                            {
                                "kind": "observability",
                                "enabled": True,
                                "config": {
                                    "version": 4,
                                    "atof": {
                                        "enabled": True,
                                        "sinks": [
                                            {
                                                "type": "file",
                                                "output_directory": str(
                                                    tmp_path / "relay"
                                                ),
                                                "filename": "events.atof.jsonl",
                                                "mode": "overwrite",
                                            }
                                        ],
                                    },
                                    "atif": {
                                        "enabled": True,
                                        "output_directory": str(tmp_path / "relay"),
                                        "filename_template": (
                                            "trajectory-{session_id}.atif.json"
                                        ),
                                    },
                                },
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(config_path)

    install = MagicMock(return_value=MagicMock())
    nooa_module = types.ModuleType("nooa")
    nooa_module.__path__ = []  # type: ignore[attr-defined]
    middleware_module = types.ModuleType("nemo_fabric_adapters.nooa.relay_compat")
    middleware_module.install_nemo_relay_compat = install
    monkeypatch.setitem(sys.modules, "nooa", nooa_module)
    monkeypatch.setitem(
        sys.modules,
        "nemo_fabric_adapters.nooa.relay_compat",
        middleware_module,
    )
    telemetry = nooa_telemetry.RelayTelemetry(
        agent_name="test-agent",
        base_dir=tmp_path,
        config=AgentConfig.from_mapping({"workflow": _workflow()}),
    )
    call = AsyncMock(return_value="terminal-result")
    agent = SimpleNamespace(event_manager=MagicMock())

    result = await telemetry.invoke(
        agent=agent,
        runtime_context=runtime_context,
        call=call,
    )

    assert result.result == "terminal-result"
    assert result.report.error is None
    artifacts = {item["kind"]: Path(item["path"]) for item in result.report.artifacts}
    assert set(artifacts) == {"atof", "atif"}
    atof = artifacts["atof"].read_text(encoding="utf-8")
    atif = artifacts["atif"].read_text(encoding="utf-8")
    assert "request-relay" in atof
    assert "invocation-relay" in atof
    assert "runtime-1" in atof
    assert "nooa-interactive-agent-request" in atof
    assert "nooa-interactive-agent-request" in atif
    call.assert_awaited_once_with()
    install.assert_called_once_with(agent.event_manager)
    install.return_value.assert_called_once_with()


async def test_relay_scope_leak_preserves_result_and_quarantines_later_turns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime_context = _relay_context(tmp_path)
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(runtime_context.telemetry.config_path)
    monkeypatch.setattr(
        nooa_telemetry.importlib.metadata,
        "version",
        MagicMock(return_value="0.10.0"),
    )
    monkeypatch.setattr(
        nooa_telemetry.common_utils,
        "reject_ambient_relay_plugin_config",
        MagicMock(),
    )
    monkeypatch.setattr(
        nooa_telemetry.common_utils,
        "reject_inherited_relay_plugin_config",
        MagicMock(),
    )
    monkeypatch.setattr(
        nooa_telemetry, "_artifact_snapshot", MagicMock(return_value={})
    )
    monkeypatch.setattr(
        nooa_telemetry, "_changed_artifacts", MagicMock(return_value=())
    )
    monkeypatch.setattr(
        nooa_telemetry.relay_artifacts,
        "snapshot_atif_files",
        MagicMock(return_value={}),
    )
    monkeypatch.setattr(
        nooa_telemetry.relay_artifacts,
        "expects_local_atif",
        MagicMock(return_value=False),
    )
    _install_relay_doubles(monkeypatch, leak_scope=True)
    telemetry = nooa_telemetry.RelayTelemetry(
        agent_name="test-agent",
        base_dir=tmp_path,
        config=AgentConfig.from_mapping({"workflow": _workflow()}),
    )
    monkeypatch.setattr(
        telemetry, "_plugin_config", MagicMock(return_value={"components": []})
    )
    calls = 0

    async def invoke_target() -> str:
        nonlocal calls
        calls += 1
        return f"result-{calls}"

    agent = SimpleNamespace(event_manager=MagicMock())
    first = await telemetry.invoke(
        agent=agent,
        runtime_context=runtime_context,
        call=invoke_target,
    )
    second = await telemetry.invoke(
        agent=agent,
        runtime_context=runtime_context,
        call=invoke_target,
    )

    assert first.result == "result-1"
    assert first.report.error is not None
    assert "do-not-report" not in first.report.error
    assert second.result == "result-2"
    assert second.report.error == nooa_telemetry._QUARANTINE_NOTE
    assert second.report.quarantine_cause is not None
    assert "do-not-report" not in second.report.quarantine_cause
    assert calls == 2


async def test_incompatible_relay_fails_before_target_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime_context = _relay_context(tmp_path)
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(runtime_context.telemetry.config_path)
    monkeypatch.setattr(
        nooa_telemetry.importlib.metadata,
        "version",
        MagicMock(return_value="0.6.9"),
    )
    telemetry = nooa_telemetry.RelayTelemetry(
        agent_name="test-agent",
        base_dir=tmp_path,
        config=AgentConfig.from_mapping({"workflow": _workflow()}),
    )
    call = AsyncMock(return_value="must-not-run")

    result = await telemetry.invoke(
        agent=SimpleNamespace(event_manager=MagicMock()),
        runtime_context=runtime_context,
        call=call,
    )

    assert result.result is None
    assert result.called is False
    assert result.report.error == "Relay setup failed (RuntimeError)"
    call.assert_not_awaited()
