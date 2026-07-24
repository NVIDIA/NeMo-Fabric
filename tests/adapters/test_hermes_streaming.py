# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free tests for Hermes Relay streaming integration."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from nemo_fabric_adapters.hermes import adapter


async def test_relay_invocation_scope_carries_fabric_request_id(
    monkeypatch,
    tmp_path: Path,
):
    runtime = adapter.HermesRuntime()
    runtime._started = True
    runtime._start_payload = {}
    runtime._runtime_id = "runtime-1"
    runtime._agent = SimpleNamespace(
        session_id="runtime-1",
        model="test-model",
        platform="fabric",
    )
    runtime._invoke_hook = MagicMock()
    runtime._relay_plugin_config = {"components": []}
    runtime._hermes_home = tmp_path
    runtime._hermes_config_path = tmp_path / "config.yaml"
    runtime._enabled_toolsets = []

    monkeypatch.setattr(
        adapter,
        "_invoke_hermes_turn",
        lambda **_kwargs: (
            {
                "response": "done",
                "completed": True,
                "failed": False,
                "messages": [],
            },
            "",
        ),
    )
    monkeypatch.setattr(
        adapter.common_utils,
        "collect_relay_artifacts",
        lambda _config: [],
    )

    from nemo_relay import scope, subscribers

    captured_metadata: list[object] = []

    @contextmanager
    def capture_scope(*_args: object, **kwargs: object):
        captured_metadata.append(kwargs["metadata"])
        yield

    monkeypatch.setattr(scope, "scope", capture_scope)
    monkeypatch.setattr(subscribers, "flush", lambda: None)

    await runtime.invoke(
        {
            "runtime_context": {"runtime_id": "runtime-1"},
            "request": {"input": "hello", "request_id": "request-1"},
        }
    )

    assert captured_metadata == [{"nemo_fabric_request_id": "request-1"}]
