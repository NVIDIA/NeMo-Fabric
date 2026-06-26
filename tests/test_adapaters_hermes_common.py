# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins
import json
import os
import types
from pathlib import Path

import pytest


def test_payload_accessors_prefer_effective_config(hermes_common: types.ModuleType) -> None:
    payload = {
        "agent_name": "outer-agent",
        "config_root": "/outer",
        "request": {"input": "hello"},
        "environment": {"workspace": "/outer-workspace"},
        "settings": {"outer": True},
        "models": {"outer": {"model": "outer-model"}},
        "capabilities": {"outer": True},
        "runtime_context": {
            "environment": {"workspace": "/runtime-workspace"},
        },
        "effective_config": {
            "agent_name": "effective-agent",
            "config_root": "/effective",
            "config": {
                "harness": {"settings": {"inner": True}},
                "models": {"inner": {"model": "inner-model"}},
            },
        },
        "capability_plan": {"native": {"skill_paths": ["skills"]}},
    }

    assert hermes_common.effective_config(payload) == payload["effective_config"]
    assert hermes_common.fabric_config(payload) == payload["effective_config"]["config"]
    assert hermes_common.agent_name(payload) == "effective-agent"
    assert hermes_common.config_root(payload) == "/effective"
    assert hermes_common.runtime_context(payload) == payload["runtime_context"]
    assert hermes_common.request_payload(payload) == {"input": "hello"}
    assert hermes_common.environment_payload(payload) == {"workspace": "/runtime-workspace"}
    assert hermes_common.settings_payload(payload) == {"inner": True}
    assert hermes_common.models_payload(payload) == {"inner": {"model": "inner-model"}}
    assert hermes_common.capability_plan(payload) == {"native": {"skill_paths": ["skills"]}}


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


def test_dump_yaml_falls_back_to_json_when_yaml_is_unavailable(
    hermes_common: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "yaml":
            raise ImportError("No module named yaml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert hermes_common.dump_yaml({"model": {"default": "demo"}}) == json.dumps(
        {"model": {"default": "demo"}},
        indent=2,
        sort_keys=False,
    ) + "\n"


@pytest.mark.parametrize(
    ("server", "expected"),
    [
        (
            {"transport": "stdio", "url": "server --stdio"},
            {"enabled": True, "command": "server --stdio"},
        ),
        (
            {"transport": "command", "url": "server --command"},
            {"enabled": True, "command": "server --command"},
        ),
        (
            {"transport": "sse", "url": "http://localhost:9000/sse"},
            {"enabled": True, "url": "http://localhost:9000/sse", "transport": "sse"},
        ),
        (
            {"url": "http://localhost:9000/default"},
            {"enabled": True, "url": "http://localhost:9000/default"},
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
    ("value", "expected"),
    [
        (None, []),
        ("git", ["git"]),
        (["git", 7, ""], ["git", "7"]),
        (42, ["42"]),
    ],
)
def test_normalize_list(hermes_common: types.ModuleType, value: object, expected: list[str]) -> None:
    assert hermes_common.normalize_list(value) == expected


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


def test_load_relay_plugin_config_wraps_and_normalizes_bare_observability_config(
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
                        "atof": {
                            "enabled": True,
                            "output_directory": "custom-relay",
                        },
                        "atif": {"enabled": True},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
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

    plugin_config = hermes_common.load_relay_plugin_config(payload)
    observability = plugin_config["components"][0]["config"]

    assert plugin_config["version"] == 1
    assert plugin_config["components"][0]["kind"] == "observability"
    assert observability["atof"]["output_directory"] == str(tmp_path / "custom-relay")
    assert observability["atof"]["filename"] == "events.atof.jsonl"
    assert observability["atof"]["mode"] == "overwrite"
    assert observability["atif"]["output_directory"] == str(tmp_path / "artifacts" / "relay")
    assert observability["atif"]["filename_template"] == "trajectory-{session_id}.atif.json"
    assert observability["atif"]["agent_name"] == "review-agent"
    assert observability["atif"]["model_name"] == "nvidia/review-model"


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


def test_collect_relay_artifacts(hermes_common: types.ModuleType, tmp_path: Path) -> None:
    atof_dir = tmp_path / "atof"
    atif_dir = tmp_path / "atif"
    atof_dir.mkdir()
    atif_dir.mkdir()
    atof_file = atof_dir / "events.atof.jsonl"
    atif_file = atif_dir / "trajectory-1.atif.json"
    ignored_file = atif_dir / "ignored.txt"
    atof_file.write_text("{}", encoding="utf-8")
    atif_file.write_text("{}", encoding="utf-8")
    ignored_file.write_text("ignored", encoding="utf-8")
    plugin_config = {
        "components": [
            {
                "kind": "observability",
                "config": {
                    "atof": {"enabled": True, "output_directory": str(atof_dir)},
                    "atif": {"enabled": True, "output_directory": str(atif_dir)},
                },
            }
        ]
    }

    assert hermes_common.collect_relay_artifacts(plugin_config) == [
        {"kind": "atof", "path": str(atof_file)},
        {"kind": "atif", "path": str(atif_file)},
    ]
