# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Hermes adapter's Fabric runtime mapping."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import nemo_fabric_adapters.common.utils as common_utils

ROOT = Path(__file__).resolve().parents[2]
HERMES_SRC = ROOT / "adapters" / "hermes" / "src"
if str(HERMES_SRC) not in sys.path:
    sys.path.insert(0, str(HERMES_SRC))

from nemo_fabric_adapters.hermes import adapter  # noqa: E402


@pytest.mark.parametrize("providers", [None, {"relay": {}}])
def test_validate_hermes_telemetry_provider_accepts_relay(
    providers: dict[str, object] | None,
):
    payload = {}
    if providers is not None:
        payload["telemetry_plan"] = {"providers": list(providers)}

    adapter.validate_hermes_telemetry_provider(payload)


def test_validate_hermes_telemetry_provider_rejects_native():
    payload = {"telemetry_plan": {"providers": ["native"], "relay_enabled": False}}

    with pytest.raises(ValueError, match="only relay telemetry is supported for Hermes"):
        hermes_common.validate_hermes_telemetry_provider(payload)


def test_validate_hermes_telemetry_provider_rejects_mixed_native_and_relay():
    payload = {"telemetry_plan": {"providers": ["relay", "native"], "relay_enabled": True}}

    with pytest.raises(ValueError, match="only relay telemetry is supported for Hermes"):
        adapter.validate_hermes_telemetry_provider(payload)


def test_build_hermes_config_maps_fabric_config_to_hermes_config():
    os.environ["MCP_URL"] = "http://localhost:9000/mcp"
    payload = {
        "runtime_context": {"environment": {"workspace": "/workspace/repo"}},
        "capability_plan": {
            "native": {
                "skill_paths": ["skills/review"],
                "mcp_servers": {
                    "github": {"transport": "stdio", "url": "github-mcp --stdio"},
                    "memory": {"transport": "sse", "url": "${MCP_URL}"},
                },
            }
        },
        "effective_config": {
            "config": {
                "harness": {
                    "settings": {
                        "model": "review",
                        "max_iterations": 4,
                        "disabled_toolsets": ["browser"],
                        "terminal_backend": "local",
                        "terminal_timeout": 90,
                        "enabled_toolsets": "git",
                        "toolset_platform": "cli",
                        "plugins_enabled": ["custom/plugin"],
                    }
                },
                "models": {
                    "review": {
                        "provider": "nvidia",
                        "model": "nvidia/review-model",
                        "settings": {"base_url": "https://model.example/v1"},
                    }
                },
            }
        },
    }

    config = adapter.build_hermes_config(payload, relay_enabled=True)

    assert config == {
        "model": {
            "provider": "nvidia",
            "default": "nvidia/review-model",
            "base_url": "https://model.example/v1",
        },
        "agent": {
            "max_turns": 4,
            "disabled_toolsets": ["browser"],
        },
        "terminal": {
            "backend": "local",
            "cwd": "/workspace/repo",
            "timeout": 90,
        },
        "skills": {"external_dirs": ["skills/review"]},
        "mcp_servers": {
            "github": {
                "enabled": True,
                "command": "github-mcp --stdio",
            },
            "memory": {
                "enabled": True,
                "url": "http://localhost:9000/mcp",
                "transport": "sse",
            },
        },
        "platform_toolsets": {"cli": ["git"]},
        "plugins": {"enabled": ["custom/plugin", "observability/nemo_relay"]},
    }


def test_hermes_config_variation_matrix_surfaces_supported_capabilities(
    tmp_path: Path,
):
    relay_config = tmp_path / "relay.json"
    relay_config.write_text(
        json.dumps(
            {
                "relay": {
                    "config": {
                        "atof": {"enabled": True, "output_directory": "relay/atof"},
                        "atif": {"enabled": True, "output_directory": "relay/atif"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(relay_config)
    payload = {
        "runtime_context": {
            "runtime_id": "runtime-matrix",
            "environment": {
                "workspace": str(tmp_path / "workspace"),
                "artifacts": str(tmp_path / "artifacts"),
            },
            "telemetry": {"relay_enabled": True},
        },
        "capability_plan": {
            "native": {
                "skill_paths": [tmp_path / "skills" / "review"],
                "mcp_servers": {
                    "github": {
                        "transport": "stdio",
                        "url": "github-mcp --stdio",
                        "exposure": "harness_native",
                    },
                    "memory": {
                        "transport": "streamable-http",
                        "url": "https://mcp.example/memory",
                        "exposure": "harness_native",
                    },
                },
            }
        },
        "effective_config": {
            "agent_name": "matrix-agent",
            "config_root": str(tmp_path),
            "config": {
                "harness": {
                    "settings": {
                        "model": "review",
                        "enabled_toolsets": ["git", "shell"],
                        "toolset_platform": "cli",
                        "terminal_backend": "local",
                    }
                },
                "models": {
                    "review": {
                        "provider": "nvidia",
                        "model": "nvidia/review-model",
                    }
                },
            },
        },
    }

    config = adapter.build_hermes_config(payload, relay_enabled=True)
    plugin_config = common_utils.load_relay_plugin_config(payload)
    observability = plugin_config["components"][0]["config"]

    assert config["model"] == {
        "provider": "nvidia",
        "default": "nvidia/review-model",
        "base_url": "https://integrate.api.nvidia.com/v1",
    }
    assert config["terminal"]["cwd"] == str(tmp_path / "workspace")
    assert config["skills"]["external_dirs"] == [str(tmp_path / "skills" / "review")]
    assert config["mcp_servers"] == {
        "github": {"enabled": True, "command": "github-mcp --stdio"},
        "memory": {
            "enabled": True,
            "url": "https://mcp.example/memory",
            "transport": "streamable-http",
        },
    }
    assert config["platform_toolsets"] == {"cli": ["git", "shell"]}
    assert config["plugins"]["enabled"] == ["observability/nemo_relay"]
    assert observability["atof"]["output_directory"] == str(tmp_path / "relay" / "atof" / "runtime-matrix")
    assert observability["atif"]["output_directory"] == str(tmp_path / "relay" / "atif" / "runtime-matrix")
    assert observability["atif"]["agent_name"] == "matrix-agent"
    assert observability["atif"]["model_name"] == "nvidia/review-model"


def test_write_hermes_config_writes_file(tmp_path: Path):
    payload = {
        "effective_config": {
            "config": {
                "harness": {"settings": {}},
                "models": {"default": {"provider": "nvidia", "model": "nvidia/test-model"}},
            }
        }
    }

    config_path, config = adapter.write_hermes_config(payload, tmp_path / "hermes-home")

    assert config_path == tmp_path / "hermes-home" / "config.yaml"
    assert config_path.exists()
    assert config["model"]["default"] == "nvidia/test-model"
    assert "nvidia/test-model" in config_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("server", "expected"),
    [
        (
            {"transport": "stdio", "url": "server --stdio"},
            {"enabled": True, "command": "server --stdio"},
        ),
        (
            {"transport": "stdio", "command": "server --stdio"},
            {"enabled": True, "command": "server --stdio"},
        ),
        (
            {"transport": "command", "url": "server --command"},
            {"enabled": True, "command": "server --command"},
        ),
        (
            {"transport": "process", "command": "server --process"},
            {"enabled": True, "command": "server --process"},
        ),
        (
            {"transport": "sse", "url": "http://localhost:9000/sse"},
            {"enabled": True, "url": "http://localhost:9000/sse", "transport": "sse"},
        ),
        (
            {"url": "http://localhost:9000/default"},
            {"enabled": True, "url": "http://localhost:9000/default"},
        ),
        (
            {"transport": "websocket", "url": "ws://localhost:9000"},
            {"enabled": True, "url": "ws://localhost:9000", "transport": "websocket"},
        ),
    ],
)
def test_hermes_mcp_server_config(
    server: dict[str, str],
    expected: dict[str, object],
):
    assert adapter.hermes_mcp_server_config(server) == expected


@pytest.mark.parametrize(
    "server",
    [
        {"transport": "stdio"},
        {"transport": "stdio", "url": "   "},
    ],
)
def test_hermes_mcp_server_config_rejects_unsupported_mappings(
    server: dict[str, str],
):
    with pytest.raises(ValueError, match="requires url or command"):
        adapter.hermes_mcp_server_config(server)


def test_summarize_hermes_config():
    assert adapter.summarize_hermes_config(
        {
            "model": {"default": "demo"},
            "terminal": {"backend": "local"},
            "skills": {"external_dirs": ["skills"]},
            "mcp_servers": {"z": {}, "a": {}},
            "plugins": {"enabled": ["observability/nemo_relay"]},
            "platform_toolsets": {"cli": ["git"]},
        }
    ) == {
        "model": {"default": "demo"},
        "terminal": {"backend": "local"},
        "skill_dirs": ["skills"],
        "mcp_servers": ["a", "z"],
        "plugins": ["observability/nemo_relay"],
        "platform_toolsets": {"cli": ["git"]},
    }


def test_configure_hermes_relay_sets_hermes_plugin_environment(
    tmp_path: Path,
):
    config_path = tmp_path / "relay.json"
    config_path.write_text(
        json.dumps(
            {
                "relay": {
                    "config": {
                        "version": 1,
                        "components": [
                            {
                                "kind": "observability",
                                "enabled": True,
                                "config": {
                                    "atof": {
                                        "enabled": True,
                                        "output_directory": "atof",
                                        "filename": "custom.atof.jsonl",
                                        "mode": "append",
                                    },
                                    "atif": {
                                        "enabled": True,
                                        "output_directory": "atif",
                                        "filename_template": "trace-{session_id}.atif.json",
                                        "agent_version": "1.2.3",
                                    },
                                },
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(config_path)
    payload = {
        "telemetry_plan": {"providers": ["relay"], "relay_enabled": True},
        "runtime_context": {"runtime_id": "runtime-relay"},
        "effective_config": {
            "agent_name": "review-agent",
            "config_root": str(tmp_path),
            "config": {
                "harness": {"settings": {"model": "review"}},
                "models": {"review": {"model": "nvidia/review-model"}},
            },
        },
    }

    with pytest.raises(ValueError, match="only relay telemetry is supported for Hermes"):
        await adapter.run_hermes(payload)

    assert plugin_config is not None
    assert os.environ["HERMES_NEMO_RELAY_ATOF_ENABLED"] == "1"
    assert os.environ["HERMES_NEMO_RELAY_ATOF_OUTPUT_DIRECTORY"] == str(tmp_path / "atof" / "runtime-relay")
    assert os.environ["HERMES_NEMO_RELAY_ATOF_FILENAME"] == "custom.atof.jsonl"
    assert os.environ["HERMES_NEMO_RELAY_ATOF_MODE"] == "append"
    assert os.environ["HERMES_NEMO_RELAY_ATIF_ENABLED"] == "1"
    assert os.environ["HERMES_NEMO_RELAY_ATIF_OUTPUT_DIRECTORY"] == str(tmp_path / "atif" / "runtime-relay")
    assert os.environ["HERMES_NEMO_RELAY_ATIF_FILENAME_TEMPLATE"] == "trace-{session_id}.atif.json"
    assert os.environ["HERMES_NEMO_RELAY_ATIF_AGENT_NAME"] == "review-agent"
    assert os.environ["HERMES_NEMO_RELAY_ATIF_AGENT_VERSION"] == "1.2.3"
    assert os.environ["HERMES_NEMO_RELAY_ATIF_MODEL_NAME"] == "nvidia/review-model"


def test_configure_hermes_relay_returns_none_when_disabled():
    assert hermes_common.configure_hermes_relay({}) is None
