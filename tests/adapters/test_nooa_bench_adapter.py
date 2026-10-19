# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the OO Agents BenchAgent adapter."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from nemo_fabric import DiscoveryConfig
from nemo_fabric import EnvironmentConfig
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import HarnessConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import ModelConfig
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunStatus
from nemo_fabric_adapter_contract.models import RuntimeContext

ROOT = Path(__file__).parents[2]
NOOA_ROOT = ROOT / "adapters" / "python" / "nooa"

if not ((3, 12) <= sys.version_info[:2] < (3, 14)):
    pytest.skip("NOOA supports Python 3.12 and 3.13", allow_module_level=True)

from nemo_fabric_adapters.nooa import bench_adapter
from nemo_fabric_adapters.nooa import telemetry as nooa_telemetry


def _config() -> AgentConfig:
    return AgentConfig.from_mapping(
        {
            "models": {
                "default": {
                    "provider": "openai",
                    "model": "openai/fixture-model",
                    "temperature": 0.2,
                    "base_url": "https://model.example.test/v1",
                }
            },
            "instructions": {"system": {"content": "Work carefully."}},
        }
    )


def _start_payload(tmp_path: Path) -> dict[str, Any]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return {
        "agent_name": "bench-test",
        "base_dir": str(tmp_path),
        "config": _config(),
        "runtime_context": {
            "runtime_id": "runtime-1",
            "environment": {"workspace": str(workspace)},
        },
    }


def _runtime_context(
    *,
    invocation_id: str = "invocation-1",
    request_id: str = "request-1",
) -> RuntimeContext:
    return RuntimeContext.from_mapping(
        {
            "runtime_id": "runtime-1",
            "invocation_id": invocation_id,
            "request_id": request_id,
            "environment": {
                "environment_id": "environment-1",
                "provider": "test",
                "control_location": "in_env_control",
                "ownership": "caller_owned",
            },
            "artifacts": {},
        }
    )


@pytest.fixture(name="bench_dependencies")
def bench_dependencies_fixture(monkeypatch: pytest.MonkeyPatch):
    mock_model = MagicMock(name="model")
    mock_model.aclose = AsyncMock()
    mock_get_llm_client = MagicMock(return_value=mock_model)
    unifiedllm = types.ModuleType("nooa.unifiedllm")
    unifiedllm.get_llm_client = mock_get_llm_client
    unifiedllm.ResponsesClient = type("ResponsesClient", (), {})
    monkeypatch.setitem(sys.modules, unifiedllm.__name__, unifiedllm)

    def agent_double(index: int) -> tuple[MagicMock, MagicMock, MagicMock]:
        mock_initial_shell = MagicMock(name=f"initial_shell_{index}")
        mock_initial_shell.close = AsyncMock()
        mock_task_shell = MagicMock(name=f"task_shell_{index}")
        mock_task_shell.close = AsyncMock()
        mock_agent = MagicMock(name=f"bench_agent_{index}")
        mock_agent.shell = mock_initial_shell
        mock_agent.event_manager = MagicMock(name=f"event_manager_{index}")

        async def run_evaluation(task_input: dict[str, Any]) -> dict[str, Any]:
            mock_agent.shell = mock_task_shell
            return {
                "response": "pytest -q",
                "success": True,
                "result": {
                    "solution_description": "Fixed the defect.",
                    "evidence": "pytest passed",
                    "command_to_verify": "pytest -q",
                },
            }

        mock_agent._run_evaluation = AsyncMock(side_effect=run_evaluation)
        return mock_agent, mock_initial_shell, mock_task_shell

    agent_doubles = [agent_double(index) for index in range(2)]
    mock_agents = [item[0] for item in agent_doubles]
    mock_initial_shells = [item[1] for item in agent_doubles]
    mock_task_shells = [item[2] for item in agent_doubles]
    mock_bench_agent = MagicMock(side_effect=mock_agents)
    bench_module = types.ModuleType("nooa_bench.bench_agent")
    bench_module.BenchAgent = mock_bench_agent
    bench_package = types.ModuleType("nooa_bench")
    bench_package.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, bench_package.__name__, bench_package)
    monkeypatch.setitem(sys.modules, bench_module.__name__, bench_module)

    mock_start_task_tokens = MagicMock()
    mock_get_task_tokens = MagicMock(
        return_value={"n_input_tokens": 12, "n_output_tokens": 4}
    )
    token_usage = types.ModuleType("nooa.runtime.token_usage")
    token_usage.start_task_tokens = mock_start_task_tokens
    token_usage.get_task_tokens = mock_get_task_tokens
    runtime_package = types.ModuleType("nooa.runtime")
    runtime_package.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, runtime_package.__name__, runtime_package)
    monkeypatch.setitem(sys.modules, token_usage.__name__, token_usage)

    mock_telemetry = MagicMock(name="relay_telemetry")

    async def invoke_telemetry(*, call: Any, **_kwargs: Any):
        return nooa_telemetry.RelayInvocation(
            called=True,
            result=await call(),
            report=None,
        )

    mock_telemetry.invoke = AsyncMock(side_effect=invoke_telemetry)
    mock_telemetry.close = AsyncMock()
    mock_relay_telemetry = MagicMock(return_value=mock_telemetry)
    monkeypatch.setattr(bench_adapter, "RelayTelemetry", mock_relay_telemetry)

    return {
        "agent": mock_agents[0],
        "agents": mock_agents,
        "bench_agent": mock_bench_agent,
        "get_llm_client": mock_get_llm_client,
        "initial_shell": mock_initial_shells[0],
        "initial_shells": mock_initial_shells,
        "model": mock_model,
        "relay_telemetry": mock_relay_telemetry,
        "start_task_tokens": mock_start_task_tokens,
        "task_shell": mock_task_shells[0],
        "task_shells": mock_task_shells,
        "telemetry": mock_telemetry,
    }


@pytest.fixture(name="responses_api_bench")
def responses_api_bench_fixture(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Run a real NOOA ResponsesClient against queued Responses API replies.

    The BenchAgent double calls the real ``ResponsesClient`` directly, once per
    queued reply. NOOA's task token accumulator is left in place, unpatched.
    """
    pytest.importorskip("nooa.unifiedllm")
    import litellm

    replies: list[Any] = []

    async def aresponses(**_kwargs: Any) -> Any:
        return replies.pop(0)

    monkeypatch.setattr(litellm, "aresponses", aresponses)

    class BenchAgentDouble:
        def __init__(self, llm: Any) -> None:
            self.llm = llm
            self.shell = None
            self.event_manager = object()

        async def _run_evaluation(self, task_input: dict[str, Any]) -> dict[str, Any]:
            while replies:
                await self.llm.acall(
                    [{"role": "user", "content": task_input["user_message"]}]
                )
            return {"response": "done", "success": True, "result": None}

    bench_module = types.ModuleType("nooa_bench.bench_agent")
    bench_module.BenchAgent = BenchAgentDouble
    bench_package = types.ModuleType("nooa_bench")
    bench_package.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, bench_package.__name__, bench_package)
    monkeypatch.setitem(sys.modules, bench_module.__name__, bench_module)

    telemetry = MagicMock(name="relay_telemetry")

    async def invoke_telemetry(*, call: Any, **_kwargs: Any):
        return nooa_telemetry.RelayInvocation(
            called=True,
            result=await call(),
            report=None,
        )

    telemetry.invoke = AsyncMock(side_effect=invoke_telemetry)
    telemetry.close = AsyncMock()
    monkeypatch.setattr(
        bench_adapter, "RelayTelemetry", MagicMock(return_value=telemetry)
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return replies


def _responses_api_reply(usage: Any) -> Any:
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
    return reply.model_copy(update={"usage": usage})


def _usage_report(**counters: Any) -> Any:
    from litellm.types.llms.openai import ResponseAPIUsage

    return ResponseAPIUsage(**counters)


async def _invoke_responses_api_bench(tmp_path: Path) -> Any:
    payload = _start_payload(tmp_path)
    payload["config"] = AgentConfig.from_mapping(
        {
            "models": {
                "default": {
                    "provider": "openai",
                    "model": "openai/fixture-model",
                    "base_url": "https://model.example.test/v1",
                    "settings": {"client_type": "responses"},
                }
            }
        }
    )
    runtime = bench_adapter.BenchRuntime()
    await runtime.start(payload)
    try:
        return await runtime.invoke(
            AgentRunRequest.from_mapping({"input": "Fix the failing test."}),
            _runtime_context(),
        )
    finally:
        await runtime.stop()


async def test_bench_responses_api_usage_sums_every_call(
    tmp_path: Path,
    responses_api_bench: list[Any],
):
    responses_api_bench.extend(
        [
            _responses_api_reply(
                _usage_report(
                    input_tokens=100,
                    input_tokens_details={"cached_tokens": 40},
                    output_tokens=20,
                    output_tokens_details={"reasoning_tokens": 5},
                    total_tokens=120,
                )
            ),
            _responses_api_reply(
                _usage_report(
                    input_tokens=150,
                    input_tokens_details={"cached_tokens": 90},
                    output_tokens=30,
                    output_tokens_details={"reasoning_tokens": 0},
                    total_tokens=180,
                )
            ),
        ]
    )

    result = await _invoke_responses_api_bench(tmp_path)

    assert result.status is AgentRunStatus.SUCCEEDED
    assert result.usage is not None
    assert result.usage.to_mapping() == {
        "input_tokens": 250,
        "cached_input_tokens": 130,
        "input_tokens_include_cache": True,
        "output_tokens": 50,
        "total_tokens": 300,
    }


async def test_bench_responses_api_reported_zero_usage_stays_zero(
    tmp_path: Path,
    responses_api_bench: list[Any],
):
    responses_api_bench.append(
        _responses_api_reply(
            _usage_report(
                input_tokens=0,
                input_tokens_details={"cached_tokens": 0},
                output_tokens=0,
                total_tokens=0,
            )
        )
    )

    result = await _invoke_responses_api_bench(tmp_path)

    assert result.usage is not None
    assert result.usage.to_mapping() == {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "input_tokens_include_cache": True,
        "output_tokens": 0,
        "total_tokens": 0,
    }


async def test_bench_responses_api_call_without_usage_leaves_usage_unknown(
    tmp_path: Path,
    responses_api_bench: list[Any],
):
    responses_api_bench.extend(
        [
            _responses_api_reply(
                _usage_report(
                    input_tokens=100,
                    input_tokens_details={"cached_tokens": 40},
                    output_tokens=20,
                    total_tokens=120,
                )
            ),
            _responses_api_reply(None),
        ]
    )

    result = await _invoke_responses_api_bench(tmp_path)

    assert result.status is AgentRunStatus.SUCCEEDED
    assert result.usage is None


async def test_bench_responses_api_partial_usage_keeps_omitted_counters_unknown(
    tmp_path: Path,
    responses_api_bench: list[Any],
):
    from litellm.types.llms.openai import ResponseAPIUsage

    responses_api_bench.extend(
        [
            _responses_api_reply(
                _usage_report(
                    input_tokens=100,
                    input_tokens_details={"cached_tokens": 40},
                    output_tokens=20,
                    total_tokens=120,
                )
            ),
            # No input_tokens_details, so the cached share of this call is unknown.
            _responses_api_reply(
                _usage_report(input_tokens=60, output_tokens=10, total_tokens=70)
            ),
            # Defensive: a usage mapping without output_tokens. litellm 1.100.1
            # cannot hand NOOA 0.0.10 such a reply (NOOA raises first), so this
            # leg covers the summing rule only.
            _responses_api_reply(
                ResponseAPIUsage.model_construct(input_tokens=30, total_tokens=30)
            ),
        ]
    )

    result = await _invoke_responses_api_bench(tmp_path)

    assert result.usage is not None
    assert result.usage.to_mapping() == {
        "input_tokens": 190,
        "input_tokens_include_cache": True,
        "total_tokens": 220,
    }


def test_bench_descriptor_is_a_closed_harness_adapter(tmp_path: Path):
    descriptor = json.loads(
        (NOOA_ROOT / "nooa-bench.fabric-adapter.json").read_text(encoding="utf-8")
    )

    assert descriptor["adapter_id"] == "nvidia.fabric.nooa.bench-agent"
    assert "target_types" not in descriptor
    assert descriptor["runner"] == {"module": "nemo_fabric_adapters.nooa.bench_adapter"}
    assert descriptor["settings_schema"]["additionalProperties"] is False
    assert descriptor["config"]["accepts"] == [
        "models",
        "models.base_url",
        "models.temperature",
        "instructions.system",
    ]
    assert descriptor["telemetry"]["providers"]["relay"]["outputs"] == [
        "atif",
        "otel",
        "openinference",
    ]

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = FabricConfig(
        metadata=MetadataConfig(name="bench-plan"),
        discovery=DiscoveryConfig(local_paths=[NOOA_ROOT]),
        harness=HarnessConfig(adapter_id=descriptor["adapter_id"]),
        models={
            "default": ModelConfig(provider="openai", model="openai/fixture-model")
        },
        environment=EnvironmentConfig(provider="local", workspace=str(workspace)),
    )

    plan = Fabric().plan(config, base_dir=tmp_path).to_mapping()

    assert plan["adapter"]["adapter_id"] == descriptor["adapter_id"]
    assert plan["adapter"]["adapter_kind"] == "python"


async def test_bench_runtime_maps_task_result_usage_and_cleanup(
    tmp_path: Path,
    bench_dependencies: dict[str, MagicMock],
):
    os.environ["OPENAI_API_KEY"] = "test-key"
    runtime = bench_adapter.BenchRuntime()

    await runtime.start(_start_payload(tmp_path))
    result = await runtime.invoke(
        AgentRunRequest.from_mapping({"input": "Fix the failing test."}),
        _runtime_context(),
    )
    await runtime.stop()

    bench_dependencies["get_llm_client"].assert_called_once_with(
        "openai/fixture-model",
        client_type=None,
        api_key="test-key",
        api_base="https://model.example.test/v1",
        temperature=0.2,
    )
    bench_dependencies["bench_agent"].assert_called_once_with(
        llm=bench_dependencies["model"]
    )
    bench_dependencies["agent"]._run_evaluation.assert_awaited_once_with(
        {
            "user_message": "Fix the failing test.",
            "working_dir": str(tmp_path / "workspace"),
            "instructions": "Work carefully.",
        }
    )
    bench_dependencies["initial_shell"].close.assert_awaited_once()
    bench_dependencies["task_shell"].close.assert_awaited_once()
    bench_dependencies["model"].aclose.assert_awaited_once()
    bench_dependencies["start_task_tokens"].assert_called_once_with()
    assert result.status is AgentRunStatus.SUCCEEDED
    assert result.output == {
        "harness": "nooa-bench",
        "adapter": "python",
        "mode": "bench_agent",
        "response": "pytest -q",
        "completed": True,
        "result": {
            "solution_description": "Fixed the defect.",
            "evidence": "pytest passed",
            "command_to_verify": "pytest -q",
        },
    }
    assert result.usage is not None
    assert result.usage.to_mapping() == {
        "input_tokens": 12,
        "output_tokens": 4,
        "total_tokens": 16,
    }
    bench_dependencies["relay_telemetry"].assert_called_once_with(
        agent_name="bench-test",
        base_dir=tmp_path,
        config=_config(),
        scope_name="nooa-bench-agent-request",
    )


async def test_bench_runtime_isolates_agents_between_invocations(
    tmp_path: Path,
    bench_dependencies: dict[str, MagicMock],
):
    os.environ["OPENAI_API_KEY"] = "test-key"
    runtime = bench_adapter.BenchRuntime()
    await runtime.start(_start_payload(tmp_path))

    first = await runtime.invoke(
        AgentRunRequest.from_mapping({"input": "Fix the first task."}),
        _runtime_context(),
    )
    second = await runtime.invoke(
        AgentRunRequest.from_mapping({"input": "Fix the second task."}),
        _runtime_context(invocation_id="invocation-2", request_id="request-2"),
    )
    await runtime.stop()

    agents = bench_dependencies["agents"]
    assert agents[0] is not agents[1]
    assert bench_dependencies["bench_agent"].call_count == 2
    assert [
        item.kwargs["agent"]
        for item in bench_dependencies["telemetry"].invoke.await_args_list
    ] == agents
    agents[0]._run_evaluation.assert_awaited_once_with(
        {
            "user_message": "Fix the first task.",
            "working_dir": str(tmp_path / "workspace"),
            "instructions": "Work carefully.",
        }
    )
    agents[1]._run_evaluation.assert_awaited_once_with(
        {
            "user_message": "Fix the second task.",
            "working_dir": str(tmp_path / "workspace"),
            "instructions": "Work carefully.",
        }
    )
    for shell in (
        *bench_dependencies["initial_shells"],
        *bench_dependencies["task_shells"],
    ):
        shell.close.assert_awaited_once_with()
    assert first.status is second.status is AgentRunStatus.SUCCEEDED
    bench_dependencies["model"].aclose.assert_awaited_once_with()


async def test_bench_runtime_returns_safe_failure(
    tmp_path: Path,
    bench_dependencies: dict[str, MagicMock],
):
    os.environ["OPENAI_API_KEY"] = "test-key"
    bench_dependencies["agent"]._run_evaluation.side_effect = None
    bench_dependencies["agent"]._run_evaluation.return_value = {
        "response": "",
        "success": False,
        "error": "secret-token-value",
    }
    runtime = bench_adapter.BenchRuntime()
    await runtime.start(_start_payload(tmp_path))

    result = await runtime.invoke(
        AgentRunRequest.from_mapping({"input": "Fix it."}),
        _runtime_context(),
    )
    await runtime.stop()

    assert result.status is AgentRunStatus.FAILED
    assert result.error is not None
    assert result.error.code == "nooa_bench_task_failed"
    assert "secret-token-value" not in json.dumps(result.to_mapping())


def test_bench_persistent_host_runs_start_invoke_stop(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fixture_source = ROOT / "tests" / "fixtures" / "nooa_bench" / "src"
    runtime_context = {
        "runtime_id": "runtime-subprocess",
        "invocation_id": "invocation-subprocess",
        "request_id": "request-subprocess",
        "environment": {
            "environment_id": "environment-subprocess",
            "provider": "local",
            "control_location": "in_env_control",
            "ownership": "caller_owned",
            "workspace": str(workspace),
        },
        "artifacts": {},
    }
    requests = [
        {
            "operation": "start",
            "payload": {
                "agent_name": "bench-subprocess",
                "base_dir": str(tmp_path),
                "config": {
                    "models": {
                        "default": {
                            "provider": "openai",
                            "model": "openai/fixture-model",
                        }
                    },
                    "instructions": {
                        "system": {"content": "Return verifiable evidence."}
                    },
                },
                "runtime_context": runtime_context,
            },
        },
        {
            "operation": "invoke",
            "payload": {
                "request": {"input": "Create a task completion artifact."},
                "runtime_context": runtime_context,
            },
        },
        {
            "operation": "stop",
            "payload": {"runtime_id": "runtime-subprocess"},
        },
    ]
    environment = os.environ.copy()
    environment["OPENAI_API_KEY"] = "fixture-key"
    environment["PYTHONPATH"] = str(fixture_source)

    completed = subprocess.run(
        [sys.executable, "-m", "nemo_fabric_adapters.nooa.bench_adapter"],
        input="".join(f"{json.dumps(request)}\n" for request in requests),
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
        cwd=ROOT,
        env=environment,
    )

    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [response["outcome"]["status"] for response in responses] == [
        "succeeded",
        "succeeded",
        "succeeded",
    ]
    invocation = responses[1]["outcome"]["output"]
    assert invocation["output"]["response"] == "test -f bench-agent-result.txt"
    assert invocation["usage"] == {
        "input_tokens": 12,
        "output_tokens": 4,
        "total_tokens": 16,
    }
    assert (workspace / "bench-agent-result.txt").is_file()
