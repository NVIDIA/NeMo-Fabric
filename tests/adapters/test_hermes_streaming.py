# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free tests for Hermes Relay streaming integration."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import RuntimeContext

if sys.version_info >= (3, 14):
    pytest.skip(
        "Hermes adapter requires Python 3.13 or earlier",
        allow_module_level=True,
    )

from nemo_fabric_adapters.hermes import adapter
from nemo_fabric_adapters.hermes import telemetry


async def test_relay_invocation_passes_fabric_request_id_to_hermes(
    monkeypatch,
    tmp_path: Path,
):
    events: list[str] = []
    task_ids: list[object] = []
    runtime = adapter.HermesRuntime()
    runtime._started = True
    runtime._agent_config = AgentConfig.from_mapping(
        {"models": {"default": {"provider": "nvidia", "model": "test-model"}}}
    )
    runtime._model_config = runtime._agent_config.models["default"]
    runtime._runtime_id = "runtime-1"
    runtime._agent = SimpleNamespace(
        session_id="runtime-1",
        model="test-model",
        platform="fabric",
    )
    runtime._relay_plugin_config = {"components": []}
    runtime._hermes_home = tmp_path
    runtime._hermes_config_path = tmp_path / "config.yaml"
    runtime._enabled_toolsets = []

    def invoke_turn(**_kwargs: object):
        events.append("turn")
        task_ids.append(_kwargs["task_id"])
        return (
            {
                "response": "done",
                "completed": True,
                "failed": False,
                "messages": [],
            },
            "",
        )

    monkeypatch.setattr(adapter, "_invoke_hermes_turn", invoke_turn)
    monkeypatch.setattr(
        telemetry,
        "finalize_hermes_relay_session",
        lambda _session_id: events.append("finalize"),
    )
    monkeypatch.setattr(
        adapter.common_utils,
        "collect_relay_artifacts",
        lambda _config: [],
    )

    await runtime.invoke(
        {
            "runtime_context": RuntimeContext.from_mapping(
                {
                    "runtime_id": "runtime-1",
                    "invocation_id": "invocation-1",
                    "request_id": "request-1",
                    "environment": {
                        "environment_id": "environment-1",
                        "provider": "test",
                        "control_location": "in_env_control",
                        "ownership": "caller_owned",
                    },
                    "artifacts": {},
                }
            ).to_mapping(),
            "request": {"input": "hello"},
        }
    )

    assert task_ids == ["request-1"]
    assert events == ["turn", "finalize"]
