# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fabric behavior survives creation of additional native Hermes agents."""

import pytest
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapters.hermes import configuration
from nemo_fabric_adapters.hermes.native import configured_kwargs


@pytest.mark.parametrize("model_limit", [None, 128])
def test_native_agents_apply_behavior_without_losing_routing_or_identity(model_limit):
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
                "settings": {"max_tokens": 96, "reasoning_config": {"effort": "none"}}
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
    assert result["max_tokens"] == (model_limit or 96)
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
    assert result["max_tokens"] == 512
    assert result["max_iterations"] == configuration.DEFAULT_MAX_ITERATIONS
