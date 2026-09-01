# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the narrow AgentConfig-to-LangChain mapping."""

from __future__ import annotations

import os
from copy import deepcopy

import pytest
from langchain_openai import ChatOpenAI
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapters.common import lifecycle

from examples.langgraph_custom_agent.adapter.configuration import (
    DEFAULT_SYSTEM_INSTRUCTION,
)
from examples.langgraph_custom_agent.adapter.configuration import (
    MODEL_REQUEST_TIMEOUT_SECONDS,
)
from examples.langgraph_custom_agent.adapter.configuration import (
    resolve_agent_dependencies,
)


def _config_mapping() -> dict:
    return {
        "models": {
            "default": {
                "provider": "nvidia",
                "model": "nvidia/test-model",
                "api_key_env": "TEST_NVIDIA_API_KEY",
                "base_url": "https://example.test/v1",
                "temperature": 0.2,
            }
        },
        "instructions": {
            "system": {
                "content": "Use the extracted signals.",
                "mode": "replace",
            }
        },
    }


def test_resolver_applies_every_advertised_model_and_instruction_field(monkeypatch):
    monkeypatch.setenv("TEST_NVIDIA_API_KEY", "test-key")

    dependencies = resolve_agent_dependencies(
        AgentConfig.from_mapping(_config_mapping())
    )

    assert isinstance(dependencies.model, ChatOpenAI)
    assert dependencies.model.model_name == "nvidia/test-model"
    assert dependencies.model.openai_api_base == "https://example.test/v1"
    assert dependencies.model.temperature == 0.2
    assert dependencies.model.request_timeout == MODEL_REQUEST_TIMEOUT_SECONDS
    assert dependencies.system_instruction == "Use the extracted signals."


def test_resolver_appends_to_the_agent_default_instruction():
    os.environ["TEST_NVIDIA_API_KEY"] = "test-key"
    mapping = _config_mapping()
    mapping["instructions"]["system"]["mode"] = "append"

    dependencies = resolve_agent_dependencies(AgentConfig.from_mapping(mapping))

    assert dependencies.system_instruction == (
        f"{DEFAULT_SYSTEM_INSTRUCTION}\n\nUse the extracted signals."
    )


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (
            lambda value: value["models"].update(
                {"other": deepcopy(value["models"]["default"])}
            ),
            "models",
        ),
        (
            lambda value: value["models"]["default"].update(
                {"settings": {"unsupported": True}}
            ),
            "models.default.settings",
        ),
        (
            lambda value: value["models"]["default"].update(
                {"provider": "anthropic"}
            ),
            "models.default.provider",
        ),
        (
            lambda value: value["models"]["default"].pop("base_url"),
            "models.default.base_url",
        ),
    ],
)
def test_resolver_rejects_accepted_but_unusable_model_configuration(
    monkeypatch,
    mutate,
    field,
):
    monkeypatch.setenv("TEST_NVIDIA_API_KEY", "test-key")
    mapping = _config_mapping()
    mutate(mapping)

    with pytest.raises(lifecycle.LifecycleError) as error:
        resolve_agent_dependencies(AgentConfig.from_mapping(mapping))

    assert error.value.code == "email_phishing_invalid_config"
    assert error.value.metadata == {"field": field}


def test_resolver_reports_the_missing_credential_name(monkeypatch):
    monkeypatch.delenv("TEST_NVIDIA_API_KEY", raising=False)

    with pytest.raises(lifecycle.LifecycleError) as error:
        resolve_agent_dependencies(AgentConfig.from_mapping(_config_mapping()))

    assert error.value.metadata == {"field": "models.default.api_key_env"}
    assert "TEST_NVIDIA_API_KEY" in error.value.message
