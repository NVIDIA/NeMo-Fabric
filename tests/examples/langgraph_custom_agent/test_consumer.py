# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for independent northbound configuration variants."""

from __future__ import annotations

from examples.langgraph_custom_agent.consumer.config import FRONTIER_DEFAULT_MODEL
from examples.langgraph_custom_agent.consumer.config import PUBLIC_DEFAULT_MODEL
from examples.langgraph_custom_agent.consumer.config import frontier_config
from examples.langgraph_custom_agent.consumer.config import public_config
from examples.langgraph_custom_agent.consumer.config import with_relay
from examples.langgraph_custom_agent.consumer.config import with_system_instruction
from examples.langgraph_custom_agent.consumer.config import with_temperature
from examples.langgraph_custom_agent.consumer.config import with_url_inspector_mcp


def test_public_and_frontier_configs_use_endpoint_specific_defaults(monkeypatch):
    monkeypatch.setenv(
        "NVIDIA_FRONTIER_BASE_URL",
        "https://frontier.example/v1",
    )

    public = public_config()
    frontier = frontier_config()

    assert public.models["default"].model == PUBLIC_DEFAULT_MODEL
    assert frontier.models["default"].model == FRONTIER_DEFAULT_MODEL
    assert public.models["default"].api_key_env == "NVIDIA_API_KEY"
    assert frontier.models["default"].api_key_env == "NVIDIA_FRONTIER_API_KEY"
    assert public.models["default"].base_url == (
        "https://integrate.api.nvidia.com/v1"
    )
    assert frontier.models["default"].base_url == "https://frontier.example/v1"


def test_instruction_and_temperature_variants_do_not_mutate_their_input():
    base = public_config()

    instruction = with_system_instruction(base, "Explain only the strongest signal.")
    temperature = with_temperature(base, 0.4)

    assert base.instructions.system.content != instruction.instructions.system.content
    assert instruction.instructions.system.content == (
        "Explain only the strongest signal."
    )
    assert base.models["default"].temperature == 0.0
    assert temperature.models["default"].temperature == 0.4


def test_relay_variant_is_additive_and_independent():
    base = public_config()

    relay = with_relay(base)

    assert base.telemetry is None
    assert base.relay is None
    assert base.runtime.artifacts is None
    assert "relay" in relay.telemetry.providers
    assert relay.relay.observability.atof.enabled is True
    assert relay.relay.observability.atif.enabled is True
    assert relay.runtime.artifacts == "./artifacts"


def test_stdio_mcp_variant_is_additive_and_bounded():
    base = public_config()

    mcp = with_url_inspector_mcp(base)

    assert base.mcp is None
    server = mcp.mcp.servers["url-inspector"]
    assert server.transport == "stdio"
    assert server.exposure == "harness_native"
    assert server.allowed_tools == ["inspect_url"]
    assert server.blocked_tools == []
