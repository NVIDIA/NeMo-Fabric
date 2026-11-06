# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fabric behavior survives creation of additional native Hermes agents."""

import os
from unittest.mock import MagicMock

import pytest
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapters.hermes import configuration
from nemo_fabric_adapters.hermes import native
from nemo_fabric_adapters.hermes.native import configured_kwargs


@pytest.mark.parametrize(
    ("model_limit", "harness_limit", "expected_limit"),
    [(None, None, None), (None, 96, 96), (128, None, 128), (128, 96, 128)],
    ids=["native-default", "harness-only", "model-only", "model-over-harness"],
)
def test_native_agents_apply_behavior_without_losing_routing_or_identity(
    model_limit: int | None, harness_limit: int | None, expected_limit: int | None
):
    model = {
        "provider": "openai",
        "model": "configured",
        "temperature": 0.0,
        "top_p": 0.41,
    }
    if model_limit is not None:
        model["max_tokens"] = model_limit
    config = AgentConfig.from_mapping(
        {
            "models": {"default": model},
            "harness": {
                "settings": {
                    "reasoning_config": {"effort": "none"},
                    **(
                        {"max_tokens": harness_limit}
                        if harness_limit is not None
                        else {}
                    ),
                }
            },
            "runtime": {"max_turns": 2},
            "tools": {"enabled": ["file"], "blocked": ["terminal"]},
            "instructions": {
                "system": {"mode": "replace", "content": "Fabric instruction"}
            },
        }
    )
    native = {
        "model": "configured",
        "provider": "custom",
        "base_url": "http://127.0.0.1:8642/v1",
        "session_id": "native-session",
        "platform": "api_server",
        "enabled_toolsets": ["file", "web"],
        "ephemeral_system_prompt": "native override",
        "request_overrides": {"temperature": 0.9, "extra_body": {"test": True}},
    }

    result = configured_kwargs(config, native, {})

    assert result["enabled_toolsets"] == ["file"]
    assert result["disabled_toolsets"] == ["terminal"]
    assert result["ephemeral_system_prompt"] == "Fabric instruction"
    assert result["max_tokens"] == expected_limit
    assert result["max_iterations"] == 2
    assert result["request_overrides"] == {
        "temperature": 0.0,
        "top_p": 0.41,
        "extra_body": {"test": True},
    }
    assert result["reasoning_config"] == {"effort": "none"}
    for key in ("model", "provider", "base_url", "session_id", "platform"):
        assert result[key] == native[key]
    assert native["enabled_toolsets"] == ["file", "web"]
    assert native["request_overrides"]["temperature"] == 0.9


def test_native_behavior_preserves_native_prompt_when_fabric_has_none():
    config = AgentConfig.from_mapping(
        {
            "models": {"default": {"provider": "openai", "model": "configured"}},
            "tools": {"enabled": []},
        }
    )

    result = configured_kwargs(config, {"ephemeral_system_prompt": "native"}, {})

    assert result["ephemeral_system_prompt"] == "native"
    assert result["enabled_toolsets"] == []
    assert result["request_overrides"] is None
    assert result["max_tokens"] is None
    assert result["max_iterations"] == configuration.DEFAULT_MAX_ITERATIONS


@pytest.mark.parametrize(
    "tools,enabled,blocked",
    [
        (None, ["native"], ["terminal"]),
        ({}, ["native"], ["terminal"]),
        ({"enabled": []}, [], ["terminal"]),
        ({"blocked": ["web"]}, ["native"], ["web"]),
    ],
)
def test_native_tool_defaults_survive_without_explicit_policy(tools, enabled, blocked):
    mapping = {"models": {"default": {"provider": "openai", "model": "test"}}}
    if tools is not None:
        mapping["tools"] = tools
    config = AgentConfig.from_mapping(mapping)
    result = configured_kwargs(
        config,
        {"enabled_toolsets": ["native"], "disabled_toolsets": ["terminal"]},
        {},
    )

    assert result["enabled_toolsets"] == enabled
    assert result["disabled_toolsets"] == blocked


@pytest.mark.parametrize(
    "platform,relay_enabled,expected",
    [("api_server", True, True), ("api_server", False, False), ("cli", True, False)],
)
def test_only_relay_enabled_api_turns_finalize(
    monkeypatch, platform, relay_enabled, expected
):
    if relay_enabled:
        os.environ["HERMES_NEMO_RELAY_PLUGINS_TOML"] = "/test/plugins.toml"
    else:
        os.environ.pop("HERMES_NEMO_RELAY_PLUGINS_TOML", None)
    finalize = MagicMock()
    monkeypatch.setattr(native.telemetry, "finalize_native_relay_segment", finalize)
    agent = MagicMock(platform=platform, session_id="native-session")

    native.finalize_api_turn(agent)

    if expected:
        finalize.assert_called_once_with("native-session")
    else:
        finalize.assert_not_called()


def test_api_relay_finalization_failure_is_reported_without_raising(
    monkeypatch, caplog
):
    os.environ["HERMES_NEMO_RELAY_PLUGINS_TOML"] = "/test/plugins.toml"
    finalize = MagicMock(side_effect=RuntimeError("test export failure"))
    monkeypatch.setattr(native.telemetry, "finalize_native_relay_segment", finalize)

    native.finalize_api_turn(
        MagicMock(platform="api_server", session_id="native-session")
    )

    assert "Hermes API turn Relay finalization failed" in caplog.text
