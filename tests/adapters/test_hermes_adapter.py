# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Hermes adapter's Fabric runtime mapping."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import os
import sys
import threading
import tomllib
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentMcpServerConfig
from nemo_fabric_adapter_contract.models import RuntimeContext

pytestmark = pytest.mark.usefixtures("requires_hermes_agent")

if importlib.util.find_spec("run_agent") is not None:
    from hermes_state import SessionDB
    from run_agent import AIAgent

    import nemo_fabric_adapters.common.utils as common_utils

    from nemo_fabric_adapters.hermes import adapter
    from nemo_fabric_adapters.hermes import configuration
    from nemo_fabric_adapters.hermes import telemetry


def _agent_config(value: dict[str, object]) -> AgentConfig:
    return AgentConfig.from_mapping(value)


def _runtime_context(
    *,
    runtime_id: str = "runtime-1",
    workspace: str | None = None,
    artifact_root: str | None = None,
    providers: list[str] | None = None,
) -> RuntimeContext:
    environment: dict[str, object] = {
        "environment_id": "environment-1",
        "provider": "test",
        "control_location": "in_env_control",
        "ownership": "caller_owned",
    }
    if workspace is not None:
        environment["workspace"] = workspace
    telemetry = None
    if providers is not None:
        telemetry = {
            "relay_enabled": providers == ["relay"],
            "metadata": {"telemetry_providers": providers},
        }
    return RuntimeContext.from_mapping(
        {
            "runtime_id": runtime_id,
            "invocation_id": "invocation-1",
            "request_id": "request-1",
            "environment": environment,
            "artifacts": {"root": artifact_root} if artifact_root else {},
            "telemetry": telemetry,
        }
    )


@pytest.mark.parametrize("providers", [None, ["relay"]])
def test_validate_hermes_telemetry_provider_accepts_relay(
    providers: list[str] | None,
):
    telemetry.validate_hermes_telemetry_provider(_runtime_context(providers=providers))


def test_validate_hermes_telemetry_provider_rejects_native():
    with pytest.raises(
        ValueError, match="only relay telemetry is supported for Hermes"
    ):
        telemetry.validate_hermes_telemetry_provider(
            _runtime_context(providers=["native"])
        )


def test_validate_hermes_telemetry_provider_rejects_mixed_native_and_relay():
    with pytest.raises(
        ValueError, match="only relay telemetry is supported for Hermes"
    ):
        telemetry.validate_hermes_telemetry_provider(
            _runtime_context(providers=["relay", "native"])
        )


def test_descriptor_uses_the_typed_agent_config_contract():
    descriptor_path = (
        Path(__file__).parents[2] / "adapters" / "hermes" / "fabric-adapter.json"
    )
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))

    assert descriptor["contract_version"] == "fabric.adapter/v1alpha2"
    assert descriptor["config"]["input"] == "agent_config"
    assert descriptor["config"]["accepts"] == [
        "models",
        "models.base_url",
        "models.temperature",
        "instructions.system",
        "runtime.max_turns",
        "tools.enabled",
        "tools.blocked",
        "mcp",
        "skills",
    ]


def test_write_hermes_relay_plugin_config_uses_upstream_toml(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(telemetry, "distribution_version", lambda _name: "0.6.0")
    relay_config_path = tmp_path / "relay.json"
    relay_config_path.write_text(
        json.dumps(
            {
                "relay": {
                    "config": {
                        "atof": {"enabled": True, "sinks": [{"type": "file"}]},
                        "atif": {"enabled": True},
                        "opentelemetry": {
                            "enabled": True,
                            "endpoint": "https://otel.example/v1/traces",
                            "service_name": "fabric",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FABRIC_RELAY_CONFIG_PATH", str(relay_config_path))
    payload = {
        "agent_name": "hermes-test-agent",
        "base_dir": str(tmp_path),
        "config": {
            "models": {"default": {"provider": "nvidia", "model": "nvidia/test-model"}}
        },
        "runtime_context": {"runtime_id": "runtime-hermes-relay"},
    }

    plugin_config_path, plugin_config = telemetry.write_hermes_relay_plugin_config(
        payload
    )

    assert plugin_config_path == tmp_path / "relay-config" / "plugins.toml"
    with plugin_config_path.open("rb") as stream:
        staged_plugin_config = tomllib.load(stream)
    staged_observability = staged_plugin_config["components"][0]["config"]
    assert staged_observability["version"] == 2
    assert staged_observability["atif"]["enabled"] is True
    assert staged_observability["atof"]["sinks"][0]["mode"] == "append"
    assert staged_observability["opentelemetry"] == {
        "enabled": True,
        "endpoint": "https://otel.example/v1/traces",
        "service_name": "fabric",
    }
    assert plugin_config["components"][0]["config"]["atof"]["sinks"][0][
        "output_directory"
    ] == str(tmp_path / "artifacts" / "relay" / "runtime-hermes-relay")
    assert (
        plugin_config["components"][0]["config"]["atof"]["sinks"][0]["mode"]
        == "overwrite"
    )


def test_write_hermes_relay_plugin_config_migrates_otlp_exporters_to_relay_v3(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(telemetry, "distribution_version", lambda _name: "0.7.2")
    relay_config_path = tmp_path / "relay.json"
    relay_config_path.write_text(
        json.dumps(
            {
                "relay": {
                    "config": {
                        "opentelemetry": {
                            "enabled": True,
                            "endpoint": "https://otel.example/v1/traces",
                            "service_name": "fabric",
                        },
                        "openinference": {
                            "enabled": True,
                            "endpoint": "https://openinference.example/v1/traces",
                            "service_name": "fabric",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FABRIC_RELAY_CONFIG_PATH", str(relay_config_path))
    payload = {
        "agent_name": "hermes-test-agent",
        "base_dir": str(tmp_path),
        "config": {
            "models": {"default": {"provider": "nvidia", "model": "nvidia/test-model"}}
        },
        "runtime_context": {"runtime_id": "runtime-hermes-relay"},
    }

    plugin_config_path, _ = telemetry.write_hermes_relay_plugin_config(payload)

    with plugin_config_path.open("rb") as stream:
        staged_observability = tomllib.load(stream)["components"][0]["config"]
    assert staged_observability["version"] == 3
    assert staged_observability["opentelemetry"] == {
        "enabled": True,
        "endpoints": [
            {
                "type": "full",
                "endpoint": "https://otel.example/v1/traces",
                "service_name": "fabric",
            },
            {
                "type": "openinference",
                "endpoint": "https://openinference.example/v1/traces",
                "service_name": "fabric",
            },
        ],
    }


def test_finalize_hermes_relay_session_uses_legacy_plugin_hook(monkeypatch):
    hermes_cli = ModuleType("hermes_cli")
    hermes_plugins = ModuleType("hermes_cli.plugins")
    mock_invoke_hook = MagicMock()
    hermes_plugins.invoke_hook = mock_invoke_hook  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", hermes_plugins)
    monkeypatch.delitem(sys.modules, "hermes_cli.lifecycle", raising=False)

    telemetry.finalize_hermes_relay_session("session-legacy")

    mock_invoke_hook.assert_called_once_with(
        "on_session_finalize", session_id="session-legacy", platform="fabric"
    )


async def test_runtime_start_stages_upstream_relay_plugin_configuration(
    monkeypatch,
    tmp_path: Path,
):
    plugin_config_path = tmp_path / "relay-config" / "plugins.toml"
    monkeypatch.setattr(
        telemetry,
        "write_hermes_relay_plugin_config",
        lambda _payload: (plugin_config_path, {"version": 1}),
    )

    def stop_after_staging(*_args, **kwargs):
        assert kwargs["relay_enabled"] is True
        assert os.environ["HERMES_NEMO_RELAY_PLUGINS_TOML"] == str(plugin_config_path)
        assert all(
            name not in os.environ
            for name in telemetry.HERMES_RELAY_ENV_NAMES
            if name != "HERMES_NEMO_RELAY_PLUGINS_TOML"
        )
        raise RuntimeError("stop after Relay plugin staging")

    monkeypatch.setattr(configuration, "write_hermes_config", stop_after_staging)
    for name in telemetry.HERMES_RELAY_ENV_NAMES:
        monkeypatch.setenv(name, "before")
    payload = {
        "base_dir": str(tmp_path),
        "config": _agent_config(
            {
                "harness": {"settings": {}},
                "models": {
                    "default": {"provider": "nvidia", "model": "test-model"}
                },
            }
        ),
        "runtime_context": _runtime_context(
            runtime_id="runtime-relay-plugin",
            workspace=str(tmp_path),
            artifact_root=str(tmp_path / "artifacts"),
            providers=["relay"],
        ).to_mapping(),
    }

    with pytest.raises(RuntimeError, match="stop after Relay plugin staging"):
        await adapter.HermesRuntime().start(payload)

    assert all(name not in os.environ for name in telemetry.HERMES_RELAY_ENV_NAMES)


def test_build_hermes_config_maps_fabric_config_to_hermes_config():
    os.environ["MCP_URL"] = "http://localhost:9000/mcp"
    agent_config = _agent_config(
        {
            "harness": {
                "settings": {
                    "terminal_timeout": 90,
                    "plugins_enabled": ["custom/plugin"],
                }
            },
            "runtime": {"max_turns": 4},
            "tools": {
                "enabled": ["git"],
                "blocked": ["browser"],
            },
            "models": {
                "review": {
                    "provider": "nvidia",
                    "model": "nvidia/review-model",
                    "base_url": "https://model.example/v1",
                }
            },
            "skills": {"paths": ["skills/review"]},
            "mcp": {
                "servers": {
                    "github": {
                        "transport": "stdio",
                        "url": "github-mcp",
                        "args": ["--stdio"],
                    },
                    "memory": {"transport": "sse", "url": "${MCP_URL}"},
                }
            },
        }
    )
    config = configuration.build_hermes_config(
        agent_config,
        workspace="/workspace/repo",
        relay_enabled=True,
    )

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


def test_build_hermes_config_omits_max_turns_when_fabric_limit_unset():
    # When max_turns is unset the config layer must leave agent.max_turns
    # absent so Hermes applies its own default rather than a starving override.
    agent_config = _agent_config(
        {
            "harness": {"settings": {}},
            "models": {"default": {"provider": "nvidia", "model": "nvidia/test-model"}},
        }
    )

    config = configuration.build_hermes_config(agent_config, workspace=".")

    assert "max_turns" not in config["agent"]


def test_build_hermes_config_omits_max_turns_when_fabric_limit_null():
    # An explicit null max_turns is treated like unset: agent.max_turns is
    # omitted so Hermes applies its own default instead of a starving override.
    agent_config = _agent_config(
        {
            "harness": {"settings": {}},
            "runtime": {"max_turns": None},
            "models": {"default": {"provider": "nvidia", "model": "nvidia/test-model"}},
        }
    )

    config = configuration.build_hermes_config(agent_config, workspace=".")

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
        "agent_name": "matrix-agent",
        "base_dir": str(tmp_path),
    }
    agent_config = _agent_config(
        {
            "harness": {"settings": {}},
            "tools": {"enabled": ["git", "shell"]},
            "models": {
                "review": {
                    "provider": "nvidia",
                    "model": "nvidia/review-model",
                }
            },
            "skills": {"paths": [tmp_path / "skills" / "review"]},
            "mcp": {
                "servers": {
                    "github": {
                        "transport": "stdio",
                        "url": "github-mcp",
                        "args": ["--stdio"],
                    },
                    "memory": {
                        "transport": "streamable-http",
                        "url": "https://mcp.example/memory",
                    },
                }
            },
        }
    )
    payload["config"] = agent_config.to_mapping()

    config = configuration.build_hermes_config(
        agent_config,
        workspace=str(tmp_path / "workspace"),
        relay_enabled=True,
    )
    plugin_config = common_utils.load_relay_plugin_config(payload)
    observability = plugin_config["components"][0]["config"]

    assert config["model"] == {
        "provider": "nvidia",
        "default": "nvidia/review-model",
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


def test_build_hermes_config_maps_stdio_mcp_args_and_env_from_agent_config(
    tmp_path: Path,
):
    agent_config = _agent_config(
        {
            "harness": {"settings": {}},
            "models": {
                "default": {
                    "provider": "nvidia",
                    "model": "nvidia/test-model",
                }
            },
            "mcp": {
                "servers": {
                    "analyzer": {
                        "transport": "stdio",
                        "url": str(tmp_path / "analyzer-mcp"),
                        "args": ["--stdio"],
                        "env": {"NVIDIA_API_KEY": "${NVIDIA_API_KEY}"},
                    }
                }
            },
        }
    )

    config = configuration.build_hermes_config(
        agent_config,
        workspace=str(tmp_path / "workspace"),
    )

    assert config["mcp_servers"] == {
        "analyzer": {
            "enabled": True,
            "command": str(tmp_path / "analyzer-mcp"),
            "args": ["--stdio"],
            "env": {"NVIDIA_API_KEY": "${NVIDIA_API_KEY}"},
        }
    }


async def test_runtime_start_discovers_mcp_tools_when_configured(
    monkeypatch,
    tmp_path: Path,
):
    mock_session_db = MagicMock(spec=SessionDB)
    mock_session_db_type = MagicMock(spec=SessionDB, return_value=mock_session_db)
    mock_ai_agent = MagicMock(spec=AIAgent)
    mock_ai_agent_type = MagicMock(spec=AIAgent, return_value=mock_ai_agent)
    monkeypatch.setattr(
        mock_ai_agent_type.__init__.__func__,
        "__signature__",
        inspect.signature(AIAgent.__init__),
        raising=False,
    )

    discover_calls: list[str] = []
    shutdown_calls: list[str] = []

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
    tools_pkg = ModuleType("tools")
    tools_mcp = ModuleType("tools.mcp_tool")
    tools_mcp.discover_mcp_tools = lambda: discover_calls.append("discover") or []  # type: ignore[attr-defined]

    def shutdown_mcp_servers() -> None:
        shutdown_calls.append("shutdown")
        raise RuntimeError("mcp shutdown failed")

    tools_mcp.shutdown_mcp_servers = shutdown_mcp_servers  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", hermes_config)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", hermes_plugins)
    monkeypatch.setitem(sys.modules, "hermes_state", hermes_state)
    monkeypatch.setitem(sys.modules, "run_agent", run_agent)
    monkeypatch.setitem(sys.modules, "tools", tools_pkg)
    monkeypatch.setitem(sys.modules, "tools.mcp_tool", tools_mcp)
    monkeypatch.setenv("TEST_API_KEY", "secret")

    payload = {
        "agent_name": "mcp-discover",
        "base_dir": str(tmp_path),
        "runtime_context": _runtime_context(
            runtime_id="runtime-mcp-discover",
            workspace=str(tmp_path),
            artifact_root=str(tmp_path / "artifacts"),
        ).to_mapping(),
    }
    payload["config"] = _agent_config(
        {
            "harness": {"settings": {}},
            "models": {
                "default": {
                    "provider": "test-provider",
                    "model": "test-model",
                    "api_key_env": "TEST_API_KEY",
                }
            },
            "tools": {"enabled": ["mcp-analyzer"]},
            "mcp": {
                "servers": {
                    "analyzer": {
                        "transport": "stdio",
                        "url": str(tmp_path / "analyzer-mcp"),
                        "env": {"NVIDIA_API_KEY": "${NVIDIA_API_KEY}"},
                    }
                }
            },
        }
    )

    runtime = adapter.HermesRuntime()
    await runtime.start(payload)
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime.stop()

    assert discover_calls == ["discover"]
    assert shutdown_calls == ["shutdown"]
    mock_ai_agent_type.assert_called_once()
    mock_ai_agent.close.assert_called_once_with()
    mock_session_db.close.assert_called_once_with()
    assert caught.value.code == "hermes_runtime_stop_failed"


def test_write_hermes_config_writes_file(tmp_path: Path):
    agent_config = _agent_config(
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

    assert config_path == tmp_path / "hermes-home" / "config.yaml"
    assert config_path.exists()
    assert config["model"]["default"] == "nvidia/test-model"
    assert "nvidia/test-model" in config_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("server", "expected"),
    [
        (
            AgentMcpServerConfig(
                transport="stdio",
                url="python3",
                args=["server.py"],
            ),
            {"enabled": True, "command": "python3", "args": ["server.py"]},
        ),
        (
            AgentMcpServerConfig(
                transport="sse",
                url="http://localhost:9000/sse",
            ),
            {"enabled": True, "url": "http://localhost:9000/sse", "transport": "sse"},
        ),
        (
            AgentMcpServerConfig(
                transport="websocket",
                url="ws://localhost:9000",
            ),
            {"enabled": True, "url": "ws://localhost:9000", "transport": "websocket"},
        ),
    ],
)
def test_hermes_mcp_server_config(
    server: AgentMcpServerConfig,
    expected: dict[str, object],
):
    assert configuration.hermes_mcp_server_config(server) == expected


def test_summarize_hermes_config():
    assert configuration.summarize_hermes_config(
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
    payload = {
        "config": _agent_config({}),
        "runtime_context": _runtime_context(providers=["native"]).to_mapping(),
    }

    with pytest.raises(
        ValueError, match="only relay telemetry is supported for Hermes"
    ):
        await adapter.HermesRuntime().start(payload)


async def test_runtime_start_requires_typed_agent_config():
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await adapter.HermesRuntime().start(
            {
                "config": {
                    "models": {"default": {"provider": "nvidia", "model": "test-model"}}
                },
                "runtime_context": _runtime_context(
                    runtime_id="runtime-untyped"
                ).to_mapping(),
            }
        )

    assert caught.value.code == "hermes_invalid_config"


async def test_runtime_start_overrides_inherited_terminal_environment(
    monkeypatch,
    tmp_path: Path,
):
    os.environ["TERMINAL_ENV"] = "docker"

    def stop_after_environment_setup(*_args, **_kwargs):
        assert os.environ["TERMINAL_ENV"] == "local"
        raise RuntimeError("stop after environment setup")

    monkeypatch.setattr(
        configuration, "write_hermes_config", stop_after_environment_setup
    )
    payload = {
        "base_dir": str(tmp_path),
        "runtime_context": _runtime_context(
            runtime_id="runtime-terminal-env",
            workspace=str(tmp_path),
            artifact_root=str(tmp_path / "artifacts"),
        ).to_mapping(),
    }
    payload["config"] = _agent_config(
        {
            "harness": {"settings": {}},
            "models": {"default": {"provider": "nvidia", "model": "test-model"}},
        }
    )

    with pytest.raises(RuntimeError, match="stop after environment setup"):
        await adapter.HermesRuntime().start(payload)


def test_artifact_root_resolves_relative_to_base_dir(tmp_path: Path):
    assert adapter._artifact_root(
        _runtime_context(artifact_root="run-artifacts"),
        str(tmp_path),
    ) == (tmp_path / "run-artifacts").resolve()


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
    monkeypatch.setenv("TEST_API_KEY", "secret")

    payload = {
        "agent_name": "demo",
        "base_dir": str(tmp_path),
        "runtime_context": _runtime_context(
            runtime_id="runtime-fabric-123",
            workspace=str(tmp_path),
        ).to_mapping(),
        "request": {
            "input": "hello",
            "request_id": "request-1",
            "context": {"history": [{"role": "user", "content": "stale"}]},
        },
    }
    payload["config"] = _agent_config(
        {
            "harness": {"settings": {}},
            "instructions": {"system": {"content": "system", "mode": "replace"}},
            "runtime": {"max_turns": None},
            "tools": {"enabled": []},
            "models": {
                "default": {
                    "provider": "test-provider",
                    "model": "test-model",
                    "api_key_env": "TEST_API_KEY",
                    "temperature": 0.2,
                }
            },
        }
    )

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
    payload["runtime_context"]["request_id"] = "request-2"
    payload["request"]["input"] = "continue"
    payload["request"]["request_id"] = "request-2"
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
        request_overrides={"temperature": 0.2},
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
        "task_id": "request-1",
    }
    assert second_call.args == ("continue",)
    assert second_call.kwargs == {
        "system_message": "system",
        "conversation_history": first_messages,
        "task_id": "request-2",
    }
    mock_ai_agent.close.assert_called_once_with()
    mock_session_db.close.assert_called_once_with()
    assert runtime._agent is None
    assert runtime._session_db is None
    assert runtime._agent_config is None
    assert runtime._conversation_history is None
    assert first["response"] == "first response"
    assert second["response"] == "second response"
    assert "session_id" not in second
    assert Path(second["hermes_home"]) == (
        tmp_path
        / "artifacts"
        / ".fabric"
        / "hermes"
        / "runtimes"
        / "runtime-fabric-123"
    )


async def test_runtime_stop_waits_for_cancelled_invoke_worker(monkeypatch):
    worker_started = threading.Event()
    worker_release = threading.Event()
    mock_agent = MagicMock()
    mock_session_db = MagicMock()

    def run_turn(**_kwargs):
        worker_started.set()
        assert worker_release.wait(timeout=1)
        return (
            {
                "response": "completed after cancellation",
                "completed": True,
                "failed": False,
                "messages": [],
            },
            "",
        )

    monkeypatch.setattr(adapter, "_invoke_hermes_turn", run_turn)
    runtime = adapter.HermesRuntime()
    runtime._started = True
    runtime._runtime_id = "runtime-cancelled-invoke"
    runtime._agent_config = _agent_config(
        {"models": {"default": {"provider": "test", "model": "test-model"}}}
    )
    runtime._model_config = runtime._agent_config.models["default"]
    runtime._agent = mock_agent
    runtime._session_db = mock_session_db
    invocation = {
        "runtime_context": _runtime_context(
            runtime_id=runtime._runtime_id
        ).to_mapping(),
        "request": {"input": "wait"},
    }

    invoke_task = asyncio.create_task(runtime.invoke(invocation))
    assert await asyncio.to_thread(worker_started.wait, 1)

    invoke_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invoke_task

    stop_task = asyncio.create_task(runtime.stop())
    await asyncio.sleep(0)
    mock_agent.close.assert_not_called()
    mock_session_db.close.assert_not_called()

    worker_release.set()
    await stop_task

    mock_agent.close.assert_called_once_with()
    mock_session_db.close.assert_called_once_with()


async def test_runtime_allows_invoke_after_cancelled_worker_finishes(monkeypatch):
    worker_started = threading.Event()
    worker_release = threading.Event()
    mock_agent = MagicMock()
    mock_session_db = MagicMock()
    calls = 0

    def run_turn(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            worker_started.set()
            assert worker_release.wait(timeout=1)
        return (
            {
                "response": f"turn-{calls}",
                "completed": True,
                "failed": False,
                "messages": [],
            },
            "",
        )

    monkeypatch.setattr(adapter, "_invoke_hermes_turn", run_turn)
    runtime = adapter.HermesRuntime()
    runtime._started = True
    runtime._runtime_id = "runtime-cancelled-invoke"
    runtime._agent_config = _agent_config(
        {"models": {"default": {"provider": "test", "model": "test-model"}}}
    )
    runtime._model_config = runtime._agent_config.models["default"]
    runtime._agent = mock_agent
    runtime._session_db = mock_session_db
    invocation = {
        "runtime_context": _runtime_context(
            runtime_id=runtime._runtime_id
        ).to_mapping(),
        "request": {"input": "wait"},
    }

    cancelled_invoke = asyncio.create_task(runtime.invoke(invocation))
    assert await asyncio.to_thread(worker_started.wait, 1)
    active_invoke_task = runtime._active_invoke_task
    assert active_invoke_task is not None

    cancelled_invoke.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_invoke

    worker_release.set()
    await asyncio.wait_for(asyncio.shield(active_invoke_task), timeout=1)

    result = await runtime.invoke(invocation)

    assert result["response"] == "turn-2"
    await runtime.stop()


def test_main_serves_persistent_runtime(monkeypatch):
    serve = MagicMock()
    monkeypatch.setattr(adapter.lifecycle, "serve", serve)

    adapter.main()

    serve.assert_called_once_with(
        adapter.HermesRuntime,
        config_loader=AgentConfig.from_mapping,
    )
