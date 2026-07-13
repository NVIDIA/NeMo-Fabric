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
        adapter.validate_hermes_telemetry_provider(payload)


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


async def test_hermes_rejects_native_telemetry():
    payload = {"telemetry_plan": {"providers": ["native"], "relay_enabled": False}}

    with pytest.raises(ValueError, match="only relay telemetry is supported for Hermes"):
        await adapter.run_hermes(payload)


async def test_fabric_runtime_id_drives_hermes_session_id_and_db_history(
    monkeypatch,
    tmp_path: Path,
):
    captured: dict[str, Any] = {}
    db_history = [{"role": "user", "content": "from hermes db"}]

    class FakeSessionDB:
        def get_session(self, session_id: str) -> dict[str, str] | None:
            captured.setdefault("db_get_session", []).append(session_id)
            if session_id == "runtime-resolved-456":
                return {"id": session_id}
            return None

        def resolve_resume_session_id(self, session_id: str) -> str:
            captured["db_resolve_session"] = session_id
            return "runtime-resolved-456"

        def get_messages_as_conversation(self, session_id: str) -> list[dict[str, str]]:
            captured["db_get_messages"] = session_id
            return list(db_history)

    class FakeAIAgent:
        def __init__(
            self,
            *,
            base_url: str | None = None,
            api_key: str | None = None,
            provider: str | None = None,
            model: str = "",
            max_iterations: int = 1,
            enabled_toolsets: list[str] | None = None,
            quiet_mode: bool = True,
            skip_context_files: bool = True,
            skip_memory: bool = True,
            save_trajectories: bool = False,
            max_tokens: int = 512,
            temperature: float = 0.0,
            reasoning_config: dict[str, Any] | None = None,
            insert_reasoning: bool = False,
            platform: str | None = None,
            session_id: str | None = None,
            session_db: Any | None = None,
        ):
            captured["init"] = {
                "session_id": session_id,
                "session_db": session_db,
                "platform": platform,
                "model": model,
                "provider": provider,
            }
            self.session_id = session_id or "generated-session"
            self.model = model
            self.platform = platform

        def run_conversation(
            self,
            user_message: str,
            *,
            system_message: str | None = None,
            conversation_history: list[dict[str, str]] | None = None,
            sync_honcho: bool = False,
            dont_review: bool = True,
        ) -> dict[str, Any]:
            captured["conversation"] = {
                "user_message": user_message,
                "system_message": system_message,
                "conversation_history": conversation_history,
                "sync_honcho": sync_honcho,
                "dont_review": dont_review,
            }
            return {
                "response": "ok",
                "completed": True,
                "failed": False,
                "messages": [{"role": "assistant", "content": "ok"}],
            }

    hermes_cli = ModuleType("hermes_cli")
    hermes_config = ModuleType("hermes_cli.config")
    hermes_config.load_config = lambda: {}  # type: ignore[attr-defined]
    hermes_plugins = ModuleType("hermes_cli.plugins")
    hermes_plugins.discover_plugins = lambda force=False: None  # type: ignore[attr-defined]
    hermes_plugins.invoke_hook = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    hermes_state = ModuleType("hermes_state")
    hermes_state.SessionDB = FakeSessionDB  # type: ignore[attr-defined]
    run_agent = ModuleType("run_agent")
    run_agent.AIAgent = FakeAIAgent  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", hermes_config)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", hermes_plugins)
    monkeypatch.setitem(sys.modules, "hermes_state", hermes_state)
    monkeypatch.setitem(sys.modules, "run_agent", run_agent)
    os.environ["TEST_API_KEY"] = "secret"

    payload = {
        "effective_config": {
            "agent_name": "demo",
            "config_root": str(tmp_path),
            "config": {
                "harness": {
                    "settings": {
                        "hermes_home": "./hermes-home",
                        "enabled_toolsets": [],
                        "system_prompt": "system",
                    }
                },
                "models": {
                    "default": {
                        "provider": "test-provider",
                        "model": "test-model",
                        "api_key_env": "TEST_API_KEY",
                    }
                },
            },
        },
        "runtime_context": {
            "runtime_id": "runtime-fabric-123",
            "environment": {"workspace": str(tmp_path)},
        },
        "request": {
            "input": "hello",
            "context": {"history": [{"role": "user", "content": "stale"}]},
        },
        "capability_plan": {"native": {}},
    }

    output = await adapter.run_hermes(payload)

    assert captured["db_resolve_session"] == "runtime-fabric-123"
    assert captured["db_get_session"] == ["runtime-resolved-456"]
    assert captured["db_get_messages"] == "runtime-resolved-456"
    assert captured["init"]["session_id"] == "runtime-fabric-123"
    assert isinstance(captured["init"]["session_db"], FakeSessionDB)
    assert captured["init"]["platform"] == "fabric"
    assert captured["conversation"]["conversation_history"] == db_history
    assert "session_id" not in output
    assert Path(output["hermes_home"]) == (
        tmp_path / "hermes-home" / "runtimes" / "runtime-fabric-123"
    )
