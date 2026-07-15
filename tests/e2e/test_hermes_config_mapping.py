# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for Fabric-to-Hermes native config mapping."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.usefixtures("requires_hermes_agent")

if importlib.util.find_spec("run_agent") is not None:
    from nemo_fabric_adapters.hermes import adapter as hermes_adapter


def test_hermes_config_mapping(tmp_path: Path):
    hermes_home = tmp_path / "home"
    config_path, config = hermes_adapter.write_hermes_config(
        payload(str(tmp_path)),
        hermes_home,
        relay_enabled=True,
    )

    assert config_path == hermes_home / "config.yaml"
    assert config_path.is_file()
    saved = yaml.safe_load(config_path.read_text())
    assert saved == config

    assert config["model"] == {
        "provider": "nvidia",
        "default": "nvidia/nemotron-3-nano-30b-a3b",
        "base_url": "https://integrate.api.nvidia.com/v1",
    }
    assert config["terminal"]["backend"] == "local"
    assert config["terminal"]["cwd"].endswith("/workspace")
    assert config["terminal"]["timeout"] == 30
    assert config["skills"]["external_dirs"] == ["/tmp/fabric-skills/code-review"]
    assert config["mcp_servers"]["github"] == {
        "enabled": True,
        "url": "https://mcp.github.example/mcp",
        "transport": "streamable-http",
    }
    assert config["platform_toolsets"]["cli"] == []
    assert config["plugins"]["enabled"] == ["observability/nemo_relay"]


def payload(tmpdir: str) -> dict:
    return {
        "agent_name": "code-review-agent",
        "config_root": tmpdir,
        "environment": {
            "workspace": f"{tmpdir}/workspace",
        },
        "models": {
            "default": {
                "provider": "nvidia",
                "model": "nvidia/nemotron-3-nano-30b-a3b",
                "api_key_env": "NVIDIA_API_KEY",
            }
        },
        "settings": {
            "enabled_toolsets": [],
            "terminal_backend": "local",
            "terminal_timeout": 30,
        },
        "capabilities": {
            "native": {
                "skill_paths": ["/tmp/fabric-skills/code-review"],
                "mcp_servers": {
                    "github": {
                        "transport": "streamable-http",
                        "url": "https://mcp.github.example/mcp",
                        "exposure": "harness_native",
                    }
                },
            }
        },
    }
