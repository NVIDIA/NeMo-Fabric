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


def _native_controls_config(settings: dict[str, object]) -> AgentConfig:
    return AgentConfig.from_mapping(
        {
            "harness": {"settings": settings},
            "models": {"default": {"provider": "nvidia", "model": "nvidia/test-model"}},
            "tools": {"enabled": ["file"], "blocked": ["terminal"]},
        }
    )


@pytest.mark.parametrize("relay_enabled", [False, True])
@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"auxiliary": {}, "tools": {}},
        {"auxiliary": {"title_generation": {}, "background_review": {}}},
        {"tools": {"tool_search": {}}},
    ],
    ids=["absent", "empty-sections", "empty-tasks", "empty-tool-search"],
)
def test_omitted_native_controls_keep_the_existing_config(
    settings: dict[str, object], relay_enabled: bool
):
    config = configuration.build_hermes_config(
        _native_controls_config(settings),
        workspace=".",
        relay_enabled=relay_enabled,
    )

    # The configuration NeMo Fabric wrote before native controls existed.
    expected = {
        "model": {"provider": "nvidia", "default": "nvidia/test-model"},
        "agent": {"disabled_toolsets": ["terminal"]},
        "terminal": {"backend": "local", "cwd": ".", "timeout": 60},
        "platform_toolsets": {"cli": ["file"]},
    }
    if relay_enabled:
        expected["auxiliary"] = {"title_generation": {"enabled": False}}
    assert config == expected


@pytest.mark.parametrize("enabled", [True, False])
def test_authored_auxiliary_controls_reach_native_config(enabled: bool):
    config = configuration.build_hermes_config(
        _native_controls_config(
            {
                "auxiliary": {
                    "title_generation": {"enabled": enabled},
                    "background_review": {"enabled": enabled},
                }
            }
        ),
        workspace=".",
    )

    assert config["auxiliary"] == {
        "title_generation": {"enabled": enabled},
        "background_review": {"enabled": enabled},
    }
    assert "tools" not in config


@pytest.mark.parametrize("value", [True, False, "auto", "on", "off"])
def test_tool_search_value_passes_through_as_authored(value: object):
    config = configuration.build_hermes_config(
        _native_controls_config({"tools": {"tool_search": {"enabled": value}}}),
        workspace=".",
    )

    assert config["tools"] == {"tool_search": {"enabled": value}}
    # Normalized tool selection stays in its own native sections.
    assert config["platform_toolsets"] == {"cli": ["file"]}
    assert config["agent"]["disabled_toolsets"] == ["terminal"]
    assert "auxiliary" not in config


@pytest.mark.parametrize(
    ("title_generation", "relay_enabled", "expected"),
    [
        pytest.param(True, True, {"enabled": True}, id="explicit-true-relay"),
        pytest.param(False, True, {"enabled": False}, id="explicit-false-relay"),
        pytest.param(True, False, {"enabled": True}, id="explicit-true-no-relay"),
        pytest.param(False, False, {"enabled": False}, id="explicit-false-no-relay"),
        pytest.param(None, True, {"enabled": False}, id="omitted-relay"),
        pytest.param(None, False, None, id="omitted-no-relay"),
    ],
)
def test_title_generation_precedence_over_relay(
    title_generation: bool | None,
    relay_enabled: bool,
    expected: dict[str, bool] | None,
):
    auxiliary: dict[str, object] = {"background_review": {"enabled": False}}
    if title_generation is not None:
        auxiliary["title_generation"] = {"enabled": title_generation}

    config = configuration.build_hermes_config(
        _native_controls_config({"auxiliary": auxiliary}),
        workspace=".",
        relay_enabled=relay_enabled,
    )

    # The Relay default must not replace other authored auxiliary controls.
    assert config["auxiliary"]["background_review"] == {"enabled": False}
    assert config["auxiliary"].get("title_generation") == expected
