# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free tests for Hermes Relay streaming integration."""

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "Hermes adapter requires Python 3.13 or earlier",
        allow_module_level=True,
    )

from nemo_fabric_adapters.hermes import adapter


@pytest.mark.parametrize(
    ("relay_plugin_config", "expected_metadata"),
    [
        (
            {
                "components": [
                    {
                        "kind": "observability",
                        "config": {
                            "atof": {
                                "enabled": True,
                                "sinks": [
                                    {
                                        "type": "stream",
                                        "name": "nemo-fabric-stream",
                                        "url": "http://127.0.0.1:1234/atof",
                                    }
                                ],
                            }
                        },
                    }
                ]
            },
            [{"nemo_fabric_request_id": "request-1"}],
        ),
        ({"components": []}, []),
    ],
    ids=["streaming", "non-streaming"],
)
async def test_relay_invocation_scope_carries_fabric_request_id(
    monkeypatch,
    tmp_path: Path,
    relay_plugin_config: dict[str, object],
    expected_metadata: list[object],
):
    events: list[str] = []
    runtime = adapter.HermesRuntime()
    runtime._started = True
    runtime._start_payload = {}
    runtime._runtime_id = "runtime-1"
    runtime._agent = SimpleNamespace(
        session_id="runtime-1",
        model="test-model",
        platform="fabric",
    )
    runtime._invoke_hook = lambda *_args, **_kwargs: events.append("finalize")
    runtime._relay_plugin_config = relay_plugin_config
    runtime._hermes_home = tmp_path
    runtime._hermes_config_path = tmp_path / "config.yaml"
    runtime._enabled_toolsets = []

    def invoke_turn(**_kwargs: object):
        events.append("turn")
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
        adapter.common_utils,
        "collect_relay_artifacts",
        lambda _config: [],
    )

    from nemo_relay import scope, subscribers

    captured_metadata: list[object] = []

    @contextmanager
    def capture_scope(*_args: object, **kwargs: object):
        captured_metadata.append(kwargs["metadata"])
        events.append("scope-enter")
        try:
            yield
        finally:
            events.append("scope-exit")

    monkeypatch.setattr(scope, "scope", capture_scope)
    monkeypatch.setattr(subscribers, "flush", lambda: None)

    await runtime.invoke(
        {
            "runtime_context": {"runtime_id": "runtime-1"},
            "request": {"input": "hello", "request_id": "request-1"},
        }
    )

    assert captured_metadata == expected_metadata
    if expected_metadata:
        assert events == ["scope-enter", "turn", "finalize", "scope-exit"]
    else:
        assert events == ["turn", "finalize"]
