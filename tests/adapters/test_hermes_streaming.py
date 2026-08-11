# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free tests for Hermes Relay streaming integration."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "Hermes adapter requires Python 3.13 or earlier",
        allow_module_level=True,
    )

from nemo_fabric_adapters.hermes import adapter


async def test_relay_invocation_passes_fabric_request_id_to_hermes(
    monkeypatch,
    tmp_path: Path,
):
    events: list[str] = []
    task_ids: list[object] = []
    runtime = adapter.HermesRuntime()
    runtime._started = True
    runtime._start_payload = {}
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
        adapter,
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
            "runtime_context": {"runtime_id": "runtime-1"},
            "request": {"input": "hello", "request_id": "request-1"},
        }
    )

    assert task_ids == ["request-1"]
    assert events == ["turn", "finalize"]
