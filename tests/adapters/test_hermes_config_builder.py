# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free tests for Hermes configuration construction."""

import builtins
import json
import sys
from pathlib import Path

import pytest
from nemo_fabric_adapter_contract.models import AgentConfig

if sys.version_info >= (3, 14):
    pytest.skip(
        "Hermes adapter requires Python 3.13 or earlier",
        allow_module_level=True,
    )

from nemo_fabric_adapters.hermes import configuration


def test_build_hermes_config_omits_unset_values_without_hermes_agent():
    config = AgentConfig.from_mapping(
        {
            "harness": {"settings": {}},
            "models": {
                "default": {
                    "provider": "nvidia",
                    "model": "nvidia/test-model",
                }
            },
        }
    )

    config = configuration.build_hermes_config(config, workspace=".")

    assert config["model"] == {
        "provider": "nvidia",
        "default": "nvidia/test-model",
    }
    assert config["agent"] == {}


def test_write_hermes_config_round_trips_without_pyyaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    real_import = builtins.__import__

    def import_without_yaml(name: str, *args: object, **kwargs: object) -> object:
        if name == "yaml":
            raise ImportError("No module named yaml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_yaml)
    agent_config = AgentConfig.from_mapping(
        {
            "harness": {"settings": {}},
            "models": {"default": {"provider": "nvidia", "model": "nvidia/test-model"}},
        }
    )

    config_path, config = configuration.write_hermes_config(
        agent_config,
        tmp_path / "hermes-home",
        workspace=".",
    )

    assert json.loads(config_path.read_text(encoding="utf-8")) == config



@pytest.mark.parametrize(
    ("api", "api_mode"),
    [
        ("openai-completions", "chat_completions"),
        ("openai-responses", "codex_responses"),
        ("anthropic-messages", "anthropic_messages"),
        (None, None),
    ],
)
def test_model_api_selects_the_hermes_api_mode(api, api_mode):
    model = {"provider": "openai", "model": "test-model"}
    if api is not None:
        model["api"] = api
    config = AgentConfig.from_mapping({"models": {"default": model}})
    native = configuration.build_hermes_config(config, workspace=".")
    assert native["model"].get("api_mode") == api_mode


def test_native_config_adds_sections_but_cannot_replace_owned_ones():
    def build(native_config):
        config = AgentConfig.from_mapping(
            {
                "harness": {"settings": {"native_config": native_config}},
                "models": {"default": {"provider": "openai", "model": "test"}},
            }
        )
        return configuration.build_hermes_config(config, workspace="/workspace")

    assert build({"web": {"backend": "tavily"}})["web"] == {"backend": "tavily"}
    with pytest.raises(ValueError, match="NeMo Fabric-owned"):
        build({"model": {"default": "other"}})


def test_api_server_mode_registers_the_endpoint_and_tool_allowlist():
    def build(mode):
        config = AgentConfig.from_mapping(
            {
                "harness": {"settings": {"mode": mode}},
                "models": {
                    "default": {
                        "provider": "openai",
                        "model": "test",
                        "base_url": "https://models.invalid/v1",
                        "api_key_env": "MODEL_KEY",
                    }
                },
                "tools": {"enabled": ["terminal"]},
            }
        )
        return configuration.build_hermes_config(config, workspace=".")

    native = build("api_server")
    assert native["platform_toolsets"]["api_server"] == ["terminal"]
    assert native["model"]["provider"] == "custom:fabric"
    assert native["providers"]["fabric"] == {
        "base_url": "https://models.invalid/v1",
        "key_env": "MODEL_KEY",
    }
    sdk = build("sdk")
    assert sdk["platform_toolsets"] == {"cli": ["terminal"]}
    assert "providers" not in sdk
