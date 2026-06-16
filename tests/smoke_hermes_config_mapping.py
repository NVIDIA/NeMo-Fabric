# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for Fabric-to-Hermes native config mapping."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "adapters" / "hermes-sdk" / "src" / "nemo_fabric_adapters" / "hermes_sdk" / "adapter.py"


def main() -> None:
    adapter = load_adapter()
    with tempfile.TemporaryDirectory(prefix="fabric-hermes-config-") as tmpdir:
        hermes_home = Path(tmpdir) / "home"
        config_path, config = adapter.write_hermes_config(
            payload(tmpdir),
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


def load_adapter():
    spec = importlib.util.spec_from_file_location("fabric_hermes_adapter", ADAPTER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
                        "expose_as": "native",
                    }
                },
            }
        },
    }


if __name__ == "__main__":
    main()
