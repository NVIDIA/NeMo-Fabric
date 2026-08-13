# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contract tests for the mini-SWE-agent adapter."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapters.mini_swe_agent import adapter

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(name="fake_mini")
def fake_mini_fixture(monkeypatch):
    model = MagicMock()
    environment = MagicMock()
    agent = MagicMock()
    agent.n_calls = 2
    agent.run.return_value = {"exit_status": "Submitted", "submission": "done"}
    model_factory = MagicMock(return_value=model)
    environment_factory = MagicMock(return_value=environment)
    agent_factory = MagicMock(return_value=agent)
    modules = {
        "minisweagent": types.ModuleType("minisweagent"),
        "minisweagent.agents": types.ModuleType("minisweagent.agents"),
        "minisweagent.agents.default": types.ModuleType("minisweagent.agents.default"),
        "minisweagent.environments": types.ModuleType("minisweagent.environments"),
        "minisweagent.environments.local": types.ModuleType(
            "minisweagent.environments.local"
        ),
        "minisweagent.models": types.ModuleType("minisweagent.models"),
        "minisweagent.models.litellm_model": types.ModuleType(
            "minisweagent.models.litellm_model"
        ),
    }
    modules["minisweagent.agents.default"].DefaultAgent = agent_factory
    modules["minisweagent.environments.local"].LocalEnvironment = environment_factory
    modules["minisweagent.models.litellm_model"].LitellmModel = model_factory
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return {
        "agent": agent,
        "agent_factory": agent_factory,
        "environment_factory": environment_factory,
        "model_factory": model_factory,
    }


@pytest.fixture(name="mini_payload")
def mini_payload_fixture(tmp_path: Path) -> dict:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return {
        "config": {
            "harness": {"settings": {"timeout": 45}},
            "instructions": {
                "system": {"content": "Work with the literal {{template}}."}
            },
            "models": {
                "default": {
                    "provider": "nvidia",
                    "model": "nvidia/test-model",
                    "api_key_env": "TEST_MINI_API_KEY",
                    "base_url": "https://example.test/v1",
                    "temperature": 0.2,
                }
            },
            "runtime": {"max_turns": 3},
        },
        "runtime_context": {
            "runtime_id": "mini-runtime",
            "invocation_id": "mini-invocation",
            "request_id": "mini-request",
            "environment": {
                "environment_id": "mini-environment",
                "provider": "local",
                "control_location": "in_env_control",
                "workspace": str(workspace),
                "env": {"TEST_MINI_API_KEY": "test-key"},
                "ownership": "caller_owned",
            },
            "artifacts": {},
        },
        "request": {"request_id": "mini-request", "input": "Fix the test."},
    }


def test_mini_swe_agent_descriptor_is_narrow_and_versioned():
    descriptor = json.loads(
        (ROOT / "adapters/mini-swe-agent/fabric-adapter.json").read_text(
            encoding="utf-8"
        )
    )

    assert descriptor["contract_version"] == "fabric.adapter/v1alpha2"
    assert descriptor["adapter_id"] == "nvidia.fabric.mini-swe-agent"
    assert descriptor["harness"] == "mini-swe-agent"
    assert descriptor["runner"] == {
        "module": "nemo_fabric_adapters.mini_swe_agent.adapter"
    }
    assert descriptor["config"] == {
        "input": "agent_config",
        "accepts": [
            "models",
            "models.base_url",
            "models.temperature",
            "instructions.system",
            "runtime.max_turns",
        ],
    }
    assert descriptor["capabilities"] == {
        "service": False,
        "streaming": False,
        "updates": False,
        "cancellation": False,
    }


async def test_mini_swe_agent_maps_config_and_returns_normalized_output(
    fake_mini, mini_payload, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("TEST_MINI_API_KEY", "test-key")
    runtime = adapter.MiniSweAgentRuntime()
    start = {**mini_payload, "config": AgentConfig.from_mapping(mini_payload["config"])}
    await runtime.start(start)

    fake_mini["model_factory"].assert_called_once_with(
        model_name="nvidia/test-model",
        model_kwargs={
            "api_key": "test-key",
            "api_base": "https://example.test/v1",
            "temperature": 0.2,
        },
    )
    fake_mini["environment_factory"].assert_called_once_with(timeout=45)

    result = await runtime.invoke(mini_payload)

    assert fake_mini["agent_factory"].call_args.kwargs["system_template"] == (
        "{{system_instruction}}"
    )
    fake_mini["agent"].run.assert_called_once_with(
        "Fix the test.", system_instruction="Work with the literal {{template}}."
    )
    assert result == {
        "failed": False,
        "output": "done",
        "usage": {"api_calls": 2},
        "exit_status": "Submitted",
    }
    await runtime.stop()


async def test_mini_swe_agent_reports_an_incomplete_loop_as_failed(
    fake_mini, mini_payload
):
    fake_mini["agent"].run.return_value = {
        "exit_status": "LimitsExceeded",
        "submission": "",
    }
    runtime = adapter.MiniSweAgentRuntime()
    start = {**mini_payload, "config": AgentConfig.from_mapping(mini_payload["config"])}
    await runtime.start(start)

    result = await runtime.invoke(mini_payload)

    assert result["failed"] is True
    assert result["error"]["code"] == "mini_swe_agent_incomplete"


def test_mini_swe_agent_module_entrypoint_exits_cleanly():
    result = subprocess.run(
        [sys.executable, "-m", "nemo_fabric_adapters.mini_swe_agent.adapter"],
        input="",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
