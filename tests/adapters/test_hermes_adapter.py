# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Hermes adapter's Fabric runtime mapping."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.usefixtures("requires_hermes_agent")

if importlib.util.find_spec("run_agent") is not None:
    from hermes_state import SessionDB
    from run_agent import AIAgent

    import nemo_fabric_adapters.common.utils as common_utils

    from nemo_fabric_adapters.hermes import adapter


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

    with pytest.raises(
        ValueError, match="only relay telemetry is supported for Hermes"
    ):
        adapter.validate_hermes_telemetry_provider(payload)


def test_validate_hermes_telemetry_provider_rejects_mixed_native_and_relay():
    payload = {
        "telemetry_plan": {"providers": ["relay", "native"], "relay_enabled": True}
    }

    with pytest.raises(
        ValueError, match="only relay telemetry is supported for Hermes"
    ):
        adapter.validate_hermes_telemetry_provider(payload)


def test_finalize_relay_session_flushes_before_artifact_collection(monkeypatch):
    calls: list[str] = []
    invoke_hook = MagicMock(side_effect=lambda *args, **kwargs: calls.append("hook"))
    runtime = adapter.HermesRuntime()
    runtime._relay_plugin_config = {"components": []}
    runtime._agent = SimpleNamespace(
        session_id="runtime-1",
        model="test-model",
        platform="fabric",
    )
    runtime._invoke_hook = invoke_hook
    runtime._relay_session_pending = True

    from nemo_relay import subscribers

    monkeypatch.setattr(subscribers, "flush", lambda: calls.append("flush"))

    runtime._finalize_relay_session()

    assert calls == ["hook", "flush"]
    invoke_hook.assert_called_once_with(
        "on_session_finalize",
        session_id="runtime-1",
        model="test-model",
        platform="fabric",
    )


async def test_stop_does_not_refinalize_completed_relay_turn(monkeypatch):
    invoke_hook = MagicMock()
    runtime = adapter.HermesRuntime()
    runtime._started = True
    runtime._relay_plugin_config = {"components": []}
    agent = MagicMock(
        session_id="runtime-1",
        model="test-model",
        platform="fabric",
    )
    session_db = MagicMock()
    runtime._agent = agent
    runtime._session_db = session_db
    runtime._invoke_hook = invoke_hook
    runtime._relay_session_pending = True

    from nemo_relay import subscribers

    flush = MagicMock()
    monkeypatch.setattr(subscribers, "flush", flush)

    runtime._finalize_relay_session()
    await runtime.stop()

    invoke_hook.assert_called_once_with(
        "on_session_finalize",
        session_id="runtime-1",
        model="test-model",
        platform="fabric",
    )
    flush.assert_called_once_with()
    agent.close.assert_called_once_with()
    session_db.close.assert_called_once_with()
    assert runtime._started is False
    assert runtime._relay_session_pending is False
    assert runtime._relay_finalize_hook_invoked is False


async def test_stop_retries_failed_relay_flush_without_refinalizing(monkeypatch):
    invoke_hook = MagicMock()
    runtime = adapter.HermesRuntime()
    runtime._started = True
    runtime._relay_plugin_config = {"components": []}
    runtime._agent = MagicMock(
        session_id="runtime-1",
        model="test-model",
        platform="fabric",
    )
    runtime._session_db = MagicMock()
    runtime._invoke_hook = invoke_hook
    runtime._relay_session_pending = True

    from nemo_relay import subscribers

    flush = MagicMock(side_effect=[RuntimeError("flush failed"), None])
    monkeypatch.setattr(subscribers, "flush", flush)

    with pytest.raises(RuntimeError, match="flush failed"):
        runtime._finalize_relay_session()

    assert runtime._relay_session_pending is True
    assert runtime._relay_finalize_hook_invoked is True
    await runtime.stop()

    invoke_hook.assert_called_once_with(
        "on_session_finalize",
        session_id="runtime-1",
        model="test-model",
        platform="fabric",
    )
    assert flush.call_count == 2
    assert runtime._relay_session_pending is False
    assert runtime._relay_finalize_hook_invoked is False


def test_build_hermes_config_maps_fabric_config_to_hermes_config():
    os.environ["MCP_URL"] = "http://localhost:9000/mcp"
    payload = {
        "runtime_context": {"environment": {"workspace": "/workspace/repo"}},
        "capability_plan": {
            "native": {
                "skill_paths": ["skills/review"],
                "mcp_servers": {
                    "github": {
                        "transport": "stdio",
                        "url": "github-mcp",
                        "args": ["--stdio"],
                    },
                    "memory": {"transport": "sse", "url": "${MCP_URL}"},
                },
            }
        },
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
            "tools": {"blocked": ["shell", "browser"]},
            "models": {
                "review": {
                    "provider": "nvidia",
                    "model": "nvidia/review-model",
                    "settings": {"base_url": "https://model.example/v1"},
                }
            },
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
            "disabled_toolsets": ["shell", "browser"],
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
                "command": "github-mcp",
                "args": ["--stdio"],
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


def test_default_max_iterations_matches_hermes_library_default():
    # Regression guard for FABRIC-85: the adapter must not override Hermes' own
    # sane loop budget with a starving value like 1, which silently truncates
    # multi-step tasks while the trial still reports success.
    assert adapter.DEFAULT_MAX_ITERATIONS > 1

    hermes_default = (
        inspect.signature(AIAgent.__init__).parameters["max_iterations"].default
    )
    assert adapter.DEFAULT_MAX_ITERATIONS == hermes_default


def test_build_hermes_config_omits_max_turns_when_max_iterations_unset():
    # When max_iterations is unset the config layer must leave agent.max_turns
    # absent so Hermes applies its own default rather than a starving override.
    payload = {
        "config": {
            "harness": {"settings": {}},
            "models": {"default": {"provider": "nvidia", "model": "nvidia/test-model"}},
        }
    }

    config = adapter.build_hermes_config(payload)

    assert "max_turns" not in config["agent"]


def test_build_hermes_config_omits_max_turns_when_max_iterations_null():
    # An explicit null max_iterations is treated like unset: agent.max_turns is
    # omitted so Hermes applies its own default instead of a starving override.
    payload = {
        "config": {
            "harness": {"settings": {"max_iterations": None}},
            "models": {"default": {"provider": "nvidia", "model": "nvidia/test-model"}},
        }
    }

    config = adapter.build_hermes_config(payload)

    assert "max_turns" not in config["agent"]


def test_hermes_config_variation_matrix_surfaces_supported_capabilities(
    tmp_path: Path,
):
    relay_config = tmp_path / "relay.json"
    relay_config.write_text(
        json.dumps(
            {
                "relay": {
                    "config": {
                        "atof": {
                            "enabled": True,
                            "sinks": [
                                {
                                    "type": "file",
                                    "output_directory": "relay/atof",
                                }
                            ],
                        },
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
                        "url": "github-mcp",
                        "args": ["--stdio"],
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
        "agent_name": "matrix-agent",
        "base_dir": str(tmp_path),
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
        "github": {
            "enabled": True,
            "command": "github-mcp",
            "args": ["--stdio"],
        },
        "memory": {
            "enabled": True,
            "url": "https://mcp.example/memory",
            "transport": "streamable-http",
        },
    }
    assert config["platform_toolsets"] == {"cli": ["git", "shell"]}
    assert config["plugins"]["enabled"] == ["observability/nemo_relay"]
    assert observability["atof"]["sinks"][0]["output_directory"] == str(
        tmp_path / "relay" / "atof" / "runtime-matrix"
    )
    assert observability["atif"]["output_directory"] == str(
        tmp_path / "relay" / "atif" / "runtime-matrix"
    )
    assert observability["atif"]["agent_name"] == "matrix-agent"
    assert observability["atif"]["model_name"] == "nvidia/review-model"


def test_write_hermes_config_writes_file(tmp_path: Path):
    payload = {
        "config": {
            "harness": {"settings": {}},
            "models": {"default": {"provider": "nvidia", "model": "nvidia/test-model"}},
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
            {"transport": "stdio", "url": "python3", "args": ["server.py"]},
            {"enabled": True, "command": "python3", "args": ["server.py"]},
        ),
        (
            {"transport": "sse", "url": "http://localhost:9000/sse"},
            {"enabled": True, "url": "http://localhost:9000/sse", "transport": "sse"},
        ),
        (
            {"transport": "websocket", "url": "ws://localhost:9000"},
            {"enabled": True, "url": "ws://localhost:9000", "transport": "websocket"},
        ),
    ],
)
def test_hermes_mcp_server_config(
    server: dict[str, object],
    expected: dict[str, object],
):
    assert adapter.hermes_mcp_server_config(server) == expected


@pytest.mark.parametrize(
    "server",
    [
        {"transport": "stdio"},
        {"transport": "stdio", "command": "server --stdio"},
        {"transport": "stdio", "url": "   "},
    ],
)
def test_hermes_mcp_server_config_rejects_unsupported_mappings(
    server: dict[str, str],
):
    with pytest.raises(ValueError, match="requires a URL"):
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
        "disabled_toolsets": [],
    }


async def test_runtime_start_rejects_native_telemetry():
    payload = {"telemetry_plan": {"providers": ["native"], "relay_enabled": False}}

    with pytest.raises(
        ValueError, match="only relay telemetry is supported for Hermes"
    ):
        await adapter.HermesRuntime().start(payload)


async def test_persistent_runtime_reuses_hermes_agent_session_and_history(
    monkeypatch,
    tmp_path: Path,
):
    mock_session_db = MagicMock(spec=SessionDB)
    mock_session_db_type = MagicMock(spec=SessionDB, return_value=mock_session_db)

    mock_ai_agent = MagicMock(spec=AIAgent)
    mock_ai_agent.session_id = "runtime-fabric-123"
    mock_ai_agent.model = "test-model"
    mock_ai_agent.platform = "fabric"
    mock_ai_agent.run_conversation.__signature__ = inspect.signature(
        AIAgent.run_conversation
    )
    first_messages = [
        {"role": "assistant", "content": "first response"},
    ]
    second_messages = [
        *first_messages,
        {"role": "assistant", "content": "second response"},
    ]
    mock_ai_agent.run_conversation.side_effect = [
        {
            "response": "first response",
            "completed": True,
            "failed": False,
            "messages": first_messages,
        },
        {
            "response": "second response",
            "completed": True,
            "failed": False,
            "messages": second_messages,
        },
    ]
    mock_ai_agent_type = MagicMock(spec=AIAgent, return_value=mock_ai_agent)
    monkeypatch.setattr(
        mock_ai_agent_type.__init__.__func__,
        "__signature__",
        inspect.signature(AIAgent.__init__),
        raising=False,
    )

    hermes_cli = ModuleType("hermes_cli")
    hermes_config = ModuleType("hermes_cli.config")
    hermes_config.load_config = lambda: {}  # type: ignore[attr-defined]
    hermes_plugins = ModuleType("hermes_cli.plugins")
    hermes_plugins.discover_plugins = lambda force=False: None  # type: ignore[attr-defined]
    hermes_plugins.invoke_hook = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    hermes_state = ModuleType("hermes_state")
    hermes_state.SessionDB = mock_session_db_type  # type: ignore[attr-defined]
    run_agent = ModuleType("run_agent")
    run_agent.AIAgent = mock_ai_agent_type  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", hermes_config)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", hermes_plugins)
    monkeypatch.setitem(sys.modules, "hermes_state", hermes_state)
    monkeypatch.setitem(sys.modules, "run_agent", run_agent)
    os.environ["TEST_API_KEY"] = "secret"

    payload = {
        "agent_name": "demo",
        "base_dir": str(tmp_path),
        "config": {
            "harness": {
                "settings": {
                    "hermes_home": "./hermes-home",
                    "enabled_toolsets": [],
                    "system_prompt": "system",
                    # Explicit null must resolve to DEFAULT_MAX_ITERATIONS (not int(None)).
                    "max_iterations": None,
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

    start_payload = {key: value for key, value in payload.items() if key != "request"}
    runtime = adapter.HermesRuntime()

    await runtime.start(start_payload)
    first = await runtime.invoke(
        {
            "runtime_context": payload["runtime_context"],
            "request": payload["request"],
        }
    )
    payload["runtime_context"]["invocation_id"] = "invocation-2"
    payload["request"]["input"] = "continue"
    second = await runtime.invoke(
        {
            "runtime_context": payload["runtime_context"],
            "request": payload["request"],
        }
    )
    await runtime.stop()

    mock_session_db_type.assert_called_once_with()
    mock_ai_agent_type.assert_called_once_with(
        base_url=None,
        api_key="secret",
        provider="test-provider",
        model="test-model",
        max_iterations=90,
        enabled_toolsets=[],
        disabled_toolsets=None,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        max_tokens=512,
        reasoning_config={"effort": "none"},
        platform="fabric",
        session_id="runtime-fabric-123",
        session_db=mock_session_db,
    )
    assert mock_ai_agent.run_conversation.call_count == 2
    first_call, second_call = mock_ai_agent.run_conversation.call_args_list
    assert first_call.args == ("hello",)
    assert first_call.kwargs == {
        "system_message": "system",
        "conversation_history": None,
    }
    assert second_call.args == ("continue",)
    assert second_call.kwargs == {
        "system_message": "system",
        "conversation_history": first_messages,
    }
    mock_ai_agent.close.assert_called_once_with()
    mock_session_db.close.assert_called_once_with()
    assert runtime._agent is None
    assert runtime._session_db is None
    assert runtime._start_payload is None
    assert runtime._conversation_history is None
    assert first["response"] == "first response"
    assert second["response"] == "second response"
    assert "session_id" not in second
    assert Path(second["hermes_home"]) == (
        tmp_path / "hermes-home" / "runtimes" / "runtime-fabric-123"
    )


def test_main_serves_persistent_runtime(monkeypatch):
    serve = MagicMock()
    monkeypatch.setattr(adapter.lifecycle, "serve", serve)

    adapter.main()

    serve.assert_called_once_with(adapter.HermesRuntime)
