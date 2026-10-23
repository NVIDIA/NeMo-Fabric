# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free tests for Hermes configuration construction."""

import builtins
import json
import os
import sys
import tomllib
from pathlib import Path

import pytest
from nemo_fabric_adapter_contract.models import AgentConfig

if sys.version_info >= (3, 15):
    pytest.skip(
        "Hermes adapter requires Python 3.14 or earlier",
        allow_module_level=True,
    )

from nemo_fabric_adapters.hermes import configuration
from nemo_fabric_adapters.hermes import telemetry


@pytest.mark.parametrize("with_observability", [False, True])
def test_dynamic_relay_plugins_reach_hermes_toml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    with_observability: bool,
):
    from nemo_fabric import FabricConfig, _native

    os.environ["XDG_CONFIG_HOME"] = str(tmp_path / "xdg")
    monkeypatch.chdir(tmp_path)
    plugins = {
        "dynamic": [
            {
                "manifest": "/opt/switchyard/relay-plugin.toml",
                "enabled": True,
                "config": {"route": "canary"},
            }
        ]
    }
    relay = {"plugins": plugins}
    if with_observability:
        relay["observability"] = {"atif": {"enabled": True}}
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "dynamic-plugin-test"},
            "harness": {"adapter_id": "nvidia.fabric.hermes"},
            "models": {"default": {"provider": "nvidia", "model": "test-model"}},
            "telemetry": {"providers": {"relay": {}}},
            "relay": relay,
        }
    )
    plan = json.loads(
        _native.plan_config(json.dumps(config.to_mapping()), str(tmp_path))
    )
    resolved = plan["telemetry_plan"]["relay_config"]
    assert resolved["plugins"] == plugins

    wrapper_path = tmp_path / "relay.json"
    wrapper_path.write_text(json.dumps({"relay": {"config": resolved}}))
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(wrapper_path)
    config_path, staged = telemetry.write_hermes_relay_plugin_config(
        {
            "agent_name": "dynamic-plugin-test",
            "base_dir": str(tmp_path),
            "runtime_context": {"runtime_id": "runtime-dynamic"},
        }
    )

    with config_path.open("rb") as stream:
        saved = tomllib.load(stream)
    assert staged["plugins"] == saved["plugins"] == plugins
    assert len(saved["components"]) == int(with_observability)


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
