# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import types
from pathlib import Path

import pytest


def test_request_payload(hermes_common: types.ModuleType):
    assert hermes_common.request_payload({"request": {"input": "hello"}}) == {"input": "hello"}
    assert hermes_common.request_payload({}) == {}


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("nvidia", "https://integrate.api.nvidia.com/v1"),
        ("openai", None),
        (None, None),
    ],
)
def test_default_base_url(
    hermes_common: types.ModuleType,
    provider: str | None,
    expected: str | None,
) -> None:
    assert hermes_common.default_base_url(provider) == expected


@pytest.mark.parametrize(
    ("settings", "model_config", "expected"),
    [
        (
            {"base_url": "https://settings.example/v1"},
            {"provider": "nvidia", "settings": {"base_url": "https://model.example/v1"}},
            "https://settings.example/v1",
        ),
        (
            {},
            {"provider": "openai", "settings": {"base_url": "https://model.example/v1"}},
            "https://model.example/v1",
        ),
        ({}, {"provider": "nvidia"}, "https://integrate.api.nvidia.com/v1"),
        ({}, {"provider": "other"}, None),
    ],
)
def test_get_base_url(
    hermes_common: types.ModuleType,
    settings: dict[str, object],
    model_config: dict[str, object],
    expected: str | None,
) -> None:
    assert hermes_common.get_base_url(settings, model_config) == expected


@pytest.mark.parametrize(
    ("selected_model", "models", "expected"),
    [
        (
            "fast",
            {"fast": {"provider": "nvidia", "model": "fast-model"}},
            {"provider": "nvidia", "model": "fast-model"},
        ),
        (
            None,
            {"default": {"provider": "nvidia", "model": "default-model"}},
            {"provider": "nvidia", "model": "default-model"},
        ),
        ("bad", {"bad": "not-a-model-config"}, {}),
    ],
)
def test_selected_model_config(
    hermes_common: types.ModuleType,
    selected_model: str | None,
    models: dict[str, object],
    expected: dict[str, object],
) -> None:
    settings = {}
    if selected_model is not None:
        settings["model"] = selected_model
    payload = {
        "effective_config": {
            "config": {
                "harness": {"settings": settings},
                "models": models,
            }
        }
    }

    assert hermes_common.selected_model_config(payload) == expected


@pytest.mark.parametrize("provider", [None, "relay"])
def test_validate_hermes_telemetry_provider_accepts_relay(
    hermes_common: types.ModuleType,
    provider: str | None,
) -> None:
    telemetry = {"enabled": True}
    if provider is not None:
        telemetry["provider"] = provider
    payload = {"effective_config": {"config": {"telemetry": telemetry}}}

    hermes_common.validate_hermes_telemetry_provider(payload)


def test_validate_hermes_telemetry_provider_rejects_native(
    hermes_common: types.ModuleType,
) -> None:
    payload = {
        "effective_config": {
            "config": {"telemetry": {"enabled": True, "provider": "native"}}
        }
    }

    with pytest.raises(ValueError, match="only relay telemetry is supported for Hermes"):
        hermes_common.validate_hermes_telemetry_provider(payload)


def test_build_hermes_config_maps_fabric_config_to_hermes_config(
    hermes_common: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_URL", "http://localhost:9000/mcp")
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

    config = hermes_common.build_hermes_config(payload, relay_enabled=True)

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
    hermes_common: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import nemo_fabric_adapters.common.utils as common_utils

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
    monkeypatch.setenv("FABRIC_RELAY_CONFIG_PATH", str(relay_config))
    payload = {
        "runtime_context": {
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

    config = hermes_common.build_hermes_config(payload, relay_enabled=True)
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
    assert observability["atof"]["output_directory"] == str(tmp_path / "relay" / "atof")
    assert observability["atif"]["output_directory"] == str(tmp_path / "relay" / "atif")
    assert observability["atif"]["agent_name"] == "matrix-agent"
    assert observability["atif"]["model_name"] == "nvidia/review-model"


def test_write_hermes_config_writes_file(hermes_common: types.ModuleType, tmp_path: Path) -> None:
    payload = {
        "effective_config": {
            "config": {
                "harness": {"settings": {}},
                "models": {"default": {"provider": "nvidia", "model": "nvidia/test-model"}},
            }
        }
    }

    config_path, config = hermes_common.write_hermes_config(payload, tmp_path / "hermes-home")

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
    hermes_common: types.ModuleType,
    server: dict[str, str],
    expected: dict[str, object],
) -> None:
    assert hermes_common.hermes_mcp_server_config(server) == expected


@pytest.mark.parametrize(
    "server",
    [
        {"transport": "stdio"},
        {"transport": "stdio", "url": "   "},
    ],
)
def test_hermes_mcp_server_config_rejects_unsupported_mappings(
    hermes_common: types.ModuleType,
    server: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="requires url or command"):
        hermes_common.hermes_mcp_server_config(server)


def test_without_none(hermes_common: types.ModuleType) -> None:
    assert hermes_common.without_none({"a": 1, "b": None, "c": False}) == {"a": 1, "c": False}


def test_summarize_hermes_config(hermes_common: types.ModuleType) -> None:
    assert hermes_common.summarize_hermes_config(
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
    hermes_common: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
    monkeypatch.setenv("FABRIC_RELAY_ENABLED", "true")
    monkeypatch.setenv("FABRIC_RELAY_CONFIG_PATH", str(config_path))
    payload = {
        "effective_config": {
            "agent_name": "review-agent",
            "config_root": str(tmp_path),
            "config": {
                "harness": {"settings": {"model": "review"}},
                "models": {"review": {"model": "nvidia/review-model"}},
            },
        }
    }

    plugin_config = hermes_common.configure_hermes_relay(payload)

    assert plugin_config is not None
    assert os.environ["HERMES_NEMO_RELAY_ATOF_ENABLED"] == "1"
    assert os.environ["HERMES_NEMO_RELAY_ATOF_OUTPUT_DIRECTORY"] == str(tmp_path / "atof")
    assert os.environ["HERMES_NEMO_RELAY_ATOF_FILENAME"] == "custom.atof.jsonl"
    assert os.environ["HERMES_NEMO_RELAY_ATOF_MODE"] == "append"
    assert os.environ["HERMES_NEMO_RELAY_ATIF_ENABLED"] == "1"
    assert os.environ["HERMES_NEMO_RELAY_ATIF_OUTPUT_DIRECTORY"] == str(tmp_path / "atif")
    assert os.environ["HERMES_NEMO_RELAY_ATIF_FILENAME_TEMPLATE"] == "trace-{session_id}.atif.json"
    assert os.environ["HERMES_NEMO_RELAY_ATIF_AGENT_NAME"] == "review-agent"
    assert os.environ["HERMES_NEMO_RELAY_ATIF_AGENT_VERSION"] == "1.2.3"
    assert os.environ["HERMES_NEMO_RELAY_ATIF_MODEL_NAME"] == "nvidia/review-model"


def test_configure_hermes_relay_returns_none_when_disabled(
    hermes_common: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FABRIC_RELAY_ENABLED", raising=False)

    assert hermes_common.configure_hermes_relay({}) is None
