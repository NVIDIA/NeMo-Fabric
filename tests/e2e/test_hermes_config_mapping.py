# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for Fabric-to-Hermes native config mapping."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml
from nemo_fabric_adapter_contract.models import AgentConfig

pytestmark = pytest.mark.usefixtures("requires_hermes_agent")

if importlib.util.find_spec("run_agent") is not None:
    from nemo_fabric_adapters.hermes import configuration


def test_hermes_config_mapping(tmp_path: Path):
    hermes_home = tmp_path / "home"
    skill = tmp_path / "skills" / "code-review"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Code review\n", encoding="utf-8")
    config_path, config = configuration.write_hermes_config(
        agent_config(skill),
        hermes_home,
        workspace=str(tmp_path / "workspace"),
        base_dir=tmp_path,
        relay_enabled=True,
    )

    assert config_path == hermes_home / "config.yaml"
    assert config_path.is_file()
    saved = yaml.safe_load(config_path.read_text())
    assert saved == config

    assert config["model"] == {
        "provider": "nvidia",
        "default": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "base_url": "https://integrate.api.nvidia.com/v1",
    }
    assert config["terminal"]["backend"] == "local"
    assert Path(config["terminal"]["cwd"]) == tmp_path / "workspace"
    assert config["terminal"]["timeout"] == 30
    assert config["skills"]["external_dirs"] == [str(skill.resolve())]
    assert config["mcp_servers"]["github"] == {
        "enabled": True,
        "url": "https://mcp.github.example/mcp",
        "transport": "streamable-http",
    }
    assert config["platform_toolsets"]["cli"] == []
    assert config["plugins"]["enabled"] == ["observability/nemo_relay"]


def agent_config(skill: Path) -> AgentConfig:
    return AgentConfig.from_mapping(
        {
            "harness": {"settings": {"terminal_timeout": 30}},
            "models": {
                "default": {
                    "provider": "nvidia",
                    "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
                    "api_key_env": "NVIDIA_API_KEY",
                    "base_url": "https://integrate.api.nvidia.com/v1",
                }
            },
            "tools": {"enabled": []},
            "skills": {"paths": [str(skill)]},
            "mcp": {
                "servers": {
                    "github": {
                        "transport": "streamable-http",
                        "url": "https://mcp.github.example/mcp",
                    }
                }
            },
        }
    )
