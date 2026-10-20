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
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentMcpServerConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunStatus
from nemo_fabric_adapter_contract.models import RuntimeContext

pytestmark = pytest.mark.usefixtures("requires_hermes_agent")

if importlib.util.find_spec("run_agent") is not None:
    # Load the Hermes module the adapter's usage observer wraps before tests
    # replace the tools package with stubs.
    import agent.chat_completion_helpers  # noqa: F401
    from hermes_state import SessionDB
    from run_agent import AIAgent

    import nemo_fabric_adapters.common.utils as common_utils

    from nemo_fabric_adapters.hermes import adapter
    from nemo_fabric_adapters.hermes import configuration
    from nemo_fabric_adapters.hermes import telemetry


def _agent_config(value: dict[str, object]) -> AgentConfig:
    return AgentConfig.from_mapping(value)


def _invocation(
    payload: dict[str, object],
) -> tuple[AgentRunRequest, RuntimeContext]:
    request_mapping = payload["request"]
    assert isinstance(request_mapping, dict)
    request = {
        key: value for key, value in request_mapping.items() if key != "request_id"
    }
    context = payload["runtime_context"]
    if not isinstance(context, RuntimeContext):
        assert isinstance(context, dict)
        context = RuntimeContext.from_mapping(context)
    return AgentRunRequest.from_mapping(request), context


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


@pytest.mark.parametrize("relay_version", ["0.9.0", "0.10.0", "0.10.0+build.1"])
def test_validate_hermes_relay_compatibility_accepts_supported_checkout(
    monkeypatch, relay_version
):
    monkeypatch.setattr(telemetry, "version", lambda _name: relay_version)
    telemetry.validate_hermes_relay_compatibility()


@pytest.mark.parametrize(
    "relay_version",
    ["0.8.4", "0.09.0", "0.11.0", "0.9.0rc1", "0.10.0rc1", "0.9.0.dev1"],
)
def test_validate_hermes_relay_compatibility_rejects_relay_outside_0_9_and_0_10(
    monkeypatch, relay_version
):
    monkeypatch.setattr(telemetry, "version", lambda _name: relay_version)

    with pytest.raises(
        RuntimeError,
        match=f"requires nemo-relay 0.9 or 0.10; found {relay_version}",
    ):
        telemetry.validate_hermes_relay_compatibility()


async def test_runtime_start_rejects_published_hermes_relay_before_setup(
    monkeypatch, tmp_path: Path
):
    from agent import relay_runtime

    monkeypatch.delattr(relay_runtime, "resolve_plugin_sources")
    payload = {
        "base_dir": str(tmp_path),
        "config": _agent_config(
            {"models": {"default": {"provider": "nvidia", "model": "test-model"}}}
        ),
        "runtime_context": _runtime_context(
            artifact_root=str(tmp_path / "artifacts"), providers=["relay"]
        ).to_mapping(),
    }

    with pytest.raises(
        RuntimeError, match="does not support Relay 0.9 or 0.10 telemetry"
    ):
        await adapter.HermesRuntime().start(payload)

    assert not (tmp_path / "artifacts" / ".fabric").exists()


def test_descriptor_uses_the_typed_agent_config_contract():
    """The Hermes descriptor declares its typed normalized config."""
    descriptor_path = (
        Path(__file__).parents[2]
        / "adapters"
        / "python"
        / "hermes"
        / "hermes.fabric-adapter.json"
    )
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))

    assert descriptor["contract_version"] == "fabric.adapter/v1alpha2"
    assert descriptor["config"]["accepts"] == [
        "models",
        "models.base_url",
        "models.temperature",
        "models.top_p",
        "models.max_tokens",
        "instructions.system",
        "runtime.max_turns",
        "tools.enabled",
        "tools.blocked",
        "mcp",
        "mcp.auth.oauth2",
        "skills",
    ]
    assert "model" not in descriptor["extension_schemas"]
    assert descriptor["config"]["system_instruction_modes"] == ["replace"]


async def test_runtime_start_rejects_append_system_instruction(tmp_path: Path):
    payload = {
        "base_dir": str(tmp_path),
        "config": _agent_config(
            {
                "instructions": {
                    "system": {
                        "content": "Follow repository policy.",
                        "mode": "append",
                    }
                }
            }
        ),
        "runtime_context": {},
    }

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await adapter.HermesRuntime().start(payload)

    assert caught.value.code == "unsupported_system_instruction_mode"
    assert caught.value.metadata["field"] == "instructions.system.mode"


def test_write_hermes_relay_plugin_config_passes_through_v4(
    monkeypatch,
    tmp_path: Path,
):
    ambient_guard = MagicMock()
    monkeypatch.setattr(
        telemetry.common_utils,
        "reject_ambient_relay_plugin_config",
        ambient_guard,
    )
    relay_config_path = tmp_path / "relay.json"
    relay_config_path.write_text(
        json.dumps(
            {
                "relay": {
                    "config": {
                        "version": 4,
                        "atof": {"enabled": True, "sinks": [{"type": "file"}]},
                        "atif": {"enabled": True},
                        "opentelemetry": {
                            "enabled": True,
                            "endpoints": [
                                {
                                    "type": "full",
                                    "endpoint": "https://otel.example/v1/traces",
                                    "service_name": "fabric",
                                    "instrumentation_scope": "nemo-relay-otel",
                                }
                            ],
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
    assert staged_observability["version"] == 4
    assert staged_observability["atif"]["enabled"] is True
    assert staged_observability["atof"]["sinks"][0]["mode"] == "append"
    assert staged_observability["opentelemetry"] == {
        "enabled": True,
        "endpoints": [
            {
                "type": "full",
                "endpoint": "https://otel.example/v1/traces",
                "service_name": "fabric",
                "instrumentation_scope": "nemo-relay-otel",
            }
        ],
    }
    assert plugin_config["components"][0]["config"]["atof"]["sinks"][0][
        "output_directory"
    ] == str(tmp_path / "artifacts" / "relay" / "runtime-hermes-relay")
    assert (
        plugin_config["components"][0]["config"]["atof"]["sinks"][0]["mode"] == "append"
    )
    ambient_guard.assert_called_once_with()


def test_write_hermes_relay_plugin_config_preserves_typed_v4_otlp_endpoints(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        telemetry.common_utils,
        "reject_ambient_relay_plugin_config",
        MagicMock(),
    )
    relay_config_path = tmp_path / "relay.json"
    relay_config_path.write_text(
        json.dumps(
            {
                "relay": {
                    "config": {
                        "version": 4,
                        "opentelemetry": {
                            "enabled": True,
                            "endpoints": [
                                {
                                    "type": "full",
                                    "endpoint": "https://otel.example/v1/traces",
                                    "service_name": "fabric",
                                    "instrumentation_scope": "nemo-relay-otel",
                                },
                                {
                                    "type": "openinference",
                                    "endpoint": (
                                        "https://openinference.example/v1/traces"
                                    ),
                                    "service_name": "fabric",
                                    "instrumentation_scope": (
                                        "nemo-relay-openinference"
                                    ),
                                },
                            ],
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
    assert staged_observability["version"] == 4
    assert staged_observability["opentelemetry"] == {
        "enabled": True,
        "endpoints": [
            {
                "type": "full",
                "endpoint": "https://otel.example/v1/traces",
                "service_name": "fabric",
                "instrumentation_scope": "nemo-relay-otel",
            },
            {
                "type": "openinference",
                "endpoint": "https://openinference.example/v1/traces",
                "service_name": "fabric",
                "instrumentation_scope": "nemo-relay-openinference",
            },
        ],
    }


def test_write_hermes_relay_plugin_config_rejects_v2(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        telemetry.common_utils,
        "reject_ambient_relay_plugin_config",
        MagicMock(),
    )
    relay_config_path = tmp_path / "relay.json"
    relay_config_path.write_text(
        '{"relay":{"config":{"version":2}}}',
        encoding="utf-8",
    )
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(relay_config_path)
    payload = {
        "agent_name": "hermes-test-agent",
        "base_dir": str(tmp_path),
        "config": {
            "models": {"default": {"provider": "nvidia", "model": "nvidia/test-model"}}
        },
        "runtime_context": {"runtime_id": "runtime-hermes-relay"},
    }

    with pytest.raises(
        ValueError,
        match="unsupported NeMo Relay observability config version 2",
    ):
        telemetry.write_hermes_relay_plugin_config(payload)


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


def test_finalize_hermes_relay_session_flushes_relay_after_the_hook(monkeypatch):
    nemo_relay = pytest.importorskip("nemo_relay")
    order: list[str] = []
    hermes_cli = ModuleType("hermes_cli")
    hermes_lifecycle = ModuleType("hermes_cli.lifecycle")
    hermes_lifecycle.finalize_session = lambda **_: order.append("finalize")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.lifecycle", hermes_lifecycle)
    monkeypatch.setattr(nemo_relay.subscribers, "flush", lambda: order.append("flush"))

    telemetry.finalize_hermes_relay_session("session-1")

    assert order == ["finalize", "flush"]


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
                "models": {"default": {"provider": "nvidia", "model": "test-model"}},
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


async def test_runtime_rechecks_ambient_relay_config_before_each_turn(
    monkeypatch,
):
    mock_turn = MagicMock()
    mock_finalize = MagicMock()
    monkeypatch.setattr(adapter, "_invoke_hermes_turn", mock_turn)
    monkeypatch.setattr(telemetry, "finalize_hermes_relay_session", mock_finalize)
    monkeypatch.setattr(
        common_utils,
        "reject_ambient_relay_plugin_config",
        MagicMock(side_effect=RuntimeError("ambient Relay config")),
    )
    runtime = adapter.HermesRuntime()
    runtime._started = True
    runtime._runtime_id = "runtime-relay-guard"
    runtime._agent_config = _agent_config(
        {"models": {"default": {"provider": "test", "model": "test-model"}}}
    )
    runtime._model_config = runtime._agent_config.models["default"]
    runtime._agent = MagicMock()
    runtime._relay_plugin_config = {"version": 1, "components": []}

    with pytest.raises(RuntimeError, match="ambient Relay config"):
        await runtime.invoke(
            *_invocation(
                {
                    "runtime_context": _runtime_context(
                        runtime_id=runtime._runtime_id
                    ).to_mapping(),
                    "request": {"input": "hello"},
                }
            )
        )

    mock_turn.assert_not_called()
    mock_finalize.assert_not_called()


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
        "auxiliary": {"title_generation": {"model_upgrade_enabled": False}},
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
        "plugins": {"enabled": ["custom/plugin"]},
    }


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
                        "version": 4,
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
    assert "plugins" not in config
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


def test_hermes_maps_http_mcp_headers_and_oauth():
    os.environ["FABRIC_MCP_CLIENT_SECRET"] = "oauth-secret"

    config = adapter.hermes_mcp_server_config(
        AgentMcpServerConfig(
            transport="sse",
            url="https://mcp.example.test/sse",
            custom_headers={"X-Tenant": "${FABRIC_MCP_HEADER}"},
            authentication={
                "type": "oauth2",
                "client_id": "fabric-client",
                "client_secret_env": "FABRIC_MCP_CLIENT_SECRET",
                "scopes": ["read", "write"],
                "redirect_uri": "http://127.0.0.1:8765/callback",
            },
        )
    )

    assert config == {
        "enabled": True,
        "url": "https://mcp.example.test/sse",
        "transport": "sse",
        "headers": {"X-Tenant": "${FABRIC_MCP_HEADER}"},
        "auth": "oauth",
        "oauth": {
            "client_id": "fabric-client",
            "client_secret": "${FABRIC_MCP_CLIENT_SECRET}",
            "scope": "read write",
            "redirect_uri": "http://127.0.0.1:8765/callback",
        },
    }


def test_hermes_rejects_unset_mcp_oauth_client_secret():
    with pytest.raises(ValueError, match="unset environment variable"):
        adapter.hermes_mcp_server_config(
            AgentMcpServerConfig(
                transport="sse",
                url="https://mcp.example.test/sse",
                authentication={
                    "type": "oauth2",
                    "client_id": "fabric-client",
                    "client_secret_env": "FABRIC_MCP_CLIENT_SECRET",
                },
            )
        )


def test_hermes_rejects_mcp_service_account_authentication():
    with pytest.raises(ValueError, match="service_account"):
        adapter.hermes_mcp_server_config(
            AgentMcpServerConfig(
                transport="streamable-http",
                url="https://mcp.example.test/mcp",
                authentication={
                    "type": "service_account",
                    "client_id": "fabric-client",
                    "client_secret_env": "FABRIC_MCP_CLIENT_SECRET",
                    "token_url": "https://auth.example.test/token",
                },
            )
        )


def test_hermes_rejects_configured_mcp_oauth_timeout():
    with pytest.raises(
        ValueError,
        match="authorization_timeout_seconds is not supported by Hermes Agent",
    ):
        adapter.hermes_mcp_server_config(
            AgentMcpServerConfig(
                transport="streamable-http",
                url="https://mcp.example.test/mcp",
                authentication={
                    "type": "oauth2",
                    "authorization_timeout_seconds": 30,
                },
            )
        )


def test_hermes_accepts_default_mcp_oauth_timeout():
    config = adapter.hermes_mcp_server_config(
        AgentMcpServerConfig(
            transport="streamable-http",
            url="https://mcp.example.test/mcp",
            authentication={
                "type": "oauth2",
                "authorization_timeout_seconds": 300,
            },
        )
    )

    assert config == {
        "enabled": True,
        "url": "https://mcp.example.test/mcp",
        "transport": "streamable-http",
        "auth": "oauth",
        "oauth": {},
    }


async def test_runtime_start_discovers_mcp_tools_when_configured(
    monkeypatch,
    tmp_path: Path,
):
    mock_session_db = MagicMock(spec=SessionDB)
    mock_session_db_type = MagicMock(spec=SessionDB, return_value=mock_session_db)
    mock_ai_agent = MagicMock(spec=AIAgent)
    mock_ai_agent.context_compressor = MagicMock()
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


async def test_runtime_authenticates_oauth_mcp_servers_before_invoke(monkeypatch):
    thread_calls = []

    async def to_thread(function, *args, **kwargs):
        thread_calls.append((function, args, kwargs))
        return function(*args, **kwargs)

    monkeypatch.setattr(adapter.asyncio, "to_thread", to_thread)
    force_interactive_oauth = MagicMock()
    oauth_stdin_reads: list[str] = []
    discover_mcp_tools = MagicMock(
        side_effect=lambda: oauth_stdin_reads.append(sys.stdin.readline())
    )
    get_mcp_status = MagicMock(
        side_effect=[
            [
                {
                    "name": "confluence",
                    "connected": False,
                    "status": "failed",
                }
            ],
            [
                {
                    "name": "confluence",
                    "connected": True,
                    "status": "connected",
                }
            ],
        ]
    )
    refresh_agent_mcp_tools = MagicMock()

    tools_oauth = ModuleType("tools.mcp_oauth")
    tools_oauth.force_interactive_oauth = (  # type: ignore[attr-defined]
        force_interactive_oauth
    )
    tools_mcp = ModuleType("tools.mcp_tool")
    tools_mcp.discover_mcp_tools = discover_mcp_tools  # type: ignore[attr-defined]
    tools_mcp.get_mcp_status = get_mcp_status  # type: ignore[attr-defined]
    tools_mcp.refresh_agent_mcp_tools = (  # type: ignore[attr-defined]
        refresh_agent_mcp_tools
    )
    monkeypatch.setitem(sys.modules, "tools.mcp_oauth", tools_oauth)
    monkeypatch.setitem(sys.modules, "tools.mcp_tool", tools_mcp)

    runtime = adapter.HermesRuntime()
    runtime._agent = MagicMock()
    runtime._hermes_config = {
        "mcp_servers": {
            "confluence": {"auth": "oauth"},
            "time": {"transport": "stdio"},
        }
    }
    lifecycle_stdin = StringIO('{"operation":"stop"}\n')
    monkeypatch.setattr(sys, "stdin", lifecycle_stdin)

    await runtime._authenticate_mcp_servers()
    await runtime._authenticate_mcp_servers()

    force_interactive_oauth.assert_called_once_with()
    discover_mcp_tools.assert_called_once_with()
    assert get_mcp_status.call_count == 2
    refresh_agent_mcp_tools.assert_called_once_with(
        runtime._agent,
        quiet_mode=True,
    )
    assert oauth_stdin_reads == [""]
    assert lifecycle_stdin.readline() == '{"operation":"stop"}\n'
    assert sys.stdin is lifecycle_stdin
    assert runtime._mcp_authentication_checked is True
    assert len(thread_calls) == 4
    assert thread_calls[0] == (get_mcp_status, (), {})
    assert thread_calls[1][0].__name__ == "authenticate"
    assert thread_calls[2] == (get_mcp_status, (), {})
    assert thread_calls[3] == (
        refresh_agent_mcp_tools,
        (runtime._agent,),
        {"quiet_mode": True},
    )


async def test_runtime_reports_failed_oauth_mcp_authentication(monkeypatch):
    tools_oauth = ModuleType("tools.mcp_oauth")
    tools_oauth.force_interactive_oauth = MagicMock()  # type: ignore[attr-defined]
    tools_mcp = ModuleType("tools.mcp_tool")
    tools_mcp.discover_mcp_tools = MagicMock()  # type: ignore[attr-defined]
    tools_mcp.get_mcp_status = MagicMock(  # type: ignore[attr-defined]
        return_value=[
            {
                "name": "confluence",
                "connected": False,
                "status": "failed",
            }
        ]
    )
    tools_mcp.refresh_agent_mcp_tools = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tools.mcp_oauth", tools_oauth)
    monkeypatch.setitem(sys.modules, "tools.mcp_tool", tools_mcp)

    runtime = adapter.HermesRuntime()
    runtime._agent = MagicMock()
    runtime._hermes_config = {"mcp_servers": {"confluence": {"auth": "oauth"}}}

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime._authenticate_mcp_servers()

    assert caught.value.code == "hermes_mcp_authentication_failed"
    assert caught.value.metadata == {"servers": ["confluence"]}
    tools_mcp.refresh_agent_mcp_tools.assert_not_called()  # type: ignore[attr-defined]
    assert runtime._mcp_authentication_checked is False


def test_build_hermes_config_disables_model_titles_and_relay_titles():
    agent_config = _agent_config(
        {
            "harness": {"settings": {}},
            "models": {"default": {"provider": "nvidia", "model": "nvidia/test-model"}},
        }
    )

    ordinary_config = configuration.build_hermes_config(agent_config, workspace=".")
    relay_config = configuration.build_hermes_config(
        agent_config, workspace=".", relay_enabled=True
    )

    # The model title call's tokens bypass the counters the adapter reports,
    # so Hermes keeps its locally derived title; Relay runs have no titles.
    assert ordinary_config["auxiliary"] == {
        "title_generation": {"model_upgrade_enabled": False}
    }
    assert relay_config["auxiliary"] == {
        "title_generation": {"enabled": False, "model_upgrade_enabled": False}
    }


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
    assert (
        adapter._artifact_root(
            _runtime_context(artifact_root="run-artifacts"),
            str(tmp_path),
        )
        == (tmp_path / "run-artifacts").resolve()
    )


@pytest.fixture(name="started_hermes_runtime")
def started_hermes_runtime_fixture(
    monkeypatch,
) -> tuple[adapter.HermesRuntime, MagicMock]:
    mock_turn = MagicMock()
    monkeypatch.setattr(adapter, "_invoke_hermes_turn", mock_turn)
    agent_config = _agent_config(
        {"models": {"default": {"provider": "test", "model": "test-model"}}}
    )
    runtime = adapter.HermesRuntime()
    runtime._started = True
    runtime._agent_config = agent_config
    runtime._model_config = agent_config.models["default"]
    runtime._runtime_id = "runtime-incomplete"
    runtime._agent = MagicMock()
    return runtime, mock_turn


@pytest.mark.parametrize(
    ("result_update", "expected_failed", "expected_error"),
    [
        pytest.param(
            {"failed": True},
            True,
            {
                "code": "hermes_reported_failure",
                "message": "Hermes did not complete the invocation",
                "retryable": False,
            },
            id="failed",
        ),
        pytest.param(
            {"partial": True},
            True,
            {
                "code": "hermes_reported_failure",
                "message": "Hermes did not complete the invocation",
                "retryable": False,
            },
            id="partial",
        ),
        pytest.param(
            {"error": "Response remained truncated after 4 continuation attempts"},
            True,
            {
                "code": "hermes_reported_failure",
                "message": "Response remained truncated after 4 continuation attempts",
                "retryable": False,
            },
            id="string-error",
        ),
        pytest.param(
            {
                "error": {
                    "code": "hermes_rate_limited",
                    "message": "provider rate limited the request",
                    "retryable": True,
                    "metadata": {"provider": "test"},
                }
            },
            True,
            {
                "code": "hermes_rate_limited",
                "message": "provider rate limited the request",
                "retryable": True,
                "extensions": {"provider": "test"},
            },
            id="structured-error",
        ),
        pytest.param({"error": ""}, False, None, id="empty-error"),
        pytest.param({}, False, None, id="completed-false-only"),
    ],
)
async def test_hermes_failure_signals_are_normalized(
    started_hermes_runtime,
    result_update: dict[str, object],
    *,
    expected_failed: bool,
    expected_error: object | None,
):
    runtime, mock_turn = started_hermes_runtime
    mock_turn.return_value = (
        {
            "final_response": "done",
            "completed": False,
            "messages": [],
            **result_update,
        },
        "",
    )

    output = await runtime.invoke(
        *_invocation(
            {
                "runtime_context": _runtime_context(
                    runtime_id="runtime-incomplete"
                ).to_mapping(),
                "request": {"input": "complete the task"},
            }
        )
    )

    assert output.output["completed"] is False
    assert (output.status is AgentRunStatus.FAILED) is expected_failed
    assert (
        None if output.error is None else output.error.to_mapping()
    ) == expected_error


async def test_persistent_runtime_reuses_hermes_agent_session_and_history(
    monkeypatch,
    tmp_path: Path,
):
    """A persistent Hermes runtime applies model extensions and reuses state."""
    mock_session_db = MagicMock(spec=SessionDB)
    mock_session_db_type = MagicMock(spec=SessionDB, return_value=mock_session_db)

    mock_ai_agent = MagicMock(spec=AIAgent)
    mock_ai_agent.context_compressor = MagicMock()
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
    turns = iter(
        [
            {
                "response": "first response",
                "completed": True,
                "failed": False,
                "messages": first_messages,
                "input_tokens": 100,
                "cache_read_tokens": 0,
                "cache_write_tokens": 50,
                "output_tokens": 20,
            },
            {
                "response": "second response",
                "completed": True,
                "failed": False,
                "messages": second_messages,
                "input_tokens": 130,
                "cache_read_tokens": 50,
                "cache_write_tokens": 50,
                "output_tokens": 35,
            },
        ]
    )
    session_counters = (
        "input_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "output_tokens",
    )
    for counter in session_counters:
        setattr(mock_ai_agent, f"session_{counter}", 0)

    def run_conversation(*args, **kwargs):
        # Hermes stamps its session counters into a finished turn's result.
        result = next(turns)
        for counter in session_counters:
            setattr(mock_ai_agent, f"session_{counter}", result[counter])
        return result

    mock_ai_agent.run_conversation.side_effect = run_conversation
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
            "harness": {"settings": {"max_tokens": 1024}},
            "instructions": {"system": {"content": "system", "mode": "replace"}},
            "runtime": {"max_turns": None},
            "tools": {"enabled": []},
            "models": {
                "default": {
                    "provider": "test-provider",
                    "model": "test-model",
                    "api_key_env": "TEST_API_KEY",
                    "temperature": 0.2,
                    "top_p": 0.85,
                    "max_tokens": 768,
                }
            },
        }
    )

    start_payload = {key: value for key, value in payload.items() if key != "request"}
    runtime = adapter.HermesRuntime()

    await runtime.start(start_payload)
    first = await runtime.invoke(
        *_invocation(
            {
                "runtime_context": payload["runtime_context"],
                "request": payload["request"],
            }
        )
    )
    payload["runtime_context"]["invocation_id"] = "invocation-2"
    payload["runtime_context"]["request_id"] = "request-2"
    payload["request"]["input"] = "continue"
    payload["request"]["request_id"] = "request-2"
    second = await runtime.invoke(
        *_invocation(
            {
                "runtime_context": payload["runtime_context"],
                "request": payload["request"],
            }
        )
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
        skip_background_review=True,
        save_trajectories=False,
        max_tokens=768,
        request_overrides={"temperature": 0.2, "top_p": 0.85},
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
    assert first.output["response"] == "first response"
    assert second.output["response"] == "second response"
    assert "session_id" not in second.output
    # Hermes reports session totals; each invocation reports its own change,
    # with input counting its cache reads and writes.
    assert first.usage is not None and second.usage is not None
    assert (first.usage.input_tokens, first.usage.cached_input_tokens) == (150, 0)
    assert first.usage.output_tokens == 20
    assert (second.usage.input_tokens, second.usage.cached_input_tokens) == (80, 50)
    assert second.usage.input_tokens_include_cache is True
    assert (second.usage.output_tokens, second.usage.total_tokens) == (15, 95)
    assert Path(second.output["hermes_home"]) == (
        tmp_path
        / "artifacts"
        / ".fabric"
        / "hermes"
        / "runtimes"
        / "runtime-fabric-123"
    )


def test_invocation_usage_with_lower_or_missing_counters_leaves_them_unknown():
    counters = {
        "input_tokens": 10,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "output_tokens": 4,
    }
    usage = adapter._invocation_usage(
        counters, {**counters, "input_tokens": 50, "output_tokens": 1}
    )
    missing = adapter._invocation_usage({}, counters)
    no_baseline = adapter._invocation_usage(counters, {})

    assert usage is not None
    assert usage.input_tokens is None
    assert usage.output_tokens == 3
    assert missing is None
    assert no_baseline is None


def test_invocation_usage_keeps_cached_input_when_input_is_unknown():
    before = dict.fromkeys(adapter._SESSION_TOKEN_KEYS, 0)
    after = {**before, "input_tokens": 10, "cache_read_tokens": 2, "output_tokens": 3}
    without_cache_writes = {
        key: value for key, value in after.items() if key != "cache_write_tokens"
    }

    unreported = adapter._invocation_usage(
        after, before, unreported=frozenset({"input"})
    )
    no_write_counter = adapter._invocation_usage(without_cache_writes, before)
    cache_only = adapter._invocation_usage(
        after, before, unreported=frozenset({"input", "output"})
    )

    for usage in (unreported, no_write_counter):
        assert usage is not None
        assert (usage.input_tokens, usage.total_tokens) == (None, None)
        assert (usage.cached_input_tokens, usage.output_tokens) == (2, 3)
    assert cache_only is not None
    assert cache_only.cached_input_tokens == 2
    assert (cache_only.input_tokens, cache_only.output_tokens) == (None, None)


def test_invocation_usage_above_the_contract_bound_is_unknown():
    zero = dict.fromkeys(adapter._SESSION_TOKEN_KEYS, 0)
    near_max = {**zero, "input_tokens": (1 << 64) - 2, "output_tokens": 3}

    usage = adapter._invocation_usage(
        near_max, zero, skipped={"input_tokens": 5, "output_tokens": 1}
    )

    assert usage is not None
    assert usage.input_tokens is None
    assert usage.total_tokens is None
    assert usage.output_tokens == 4


def test_session_counters_keep_only_valid_counts():
    agent = SimpleNamespace(
        session_input_tokens=7,
        session_cache_read_tokens=True,
        session_cache_write_tokens=-1,
        session_output_tokens="3",
    )

    assert adapter._session_counters(agent) == {"input_tokens": 7}


# A turn step that ends the turn with a failure result carrying no session
# counters, like Hermes' billing, content-policy and partial-turn results.
_COUNTERLESS_FAILURE = "counterless failure"


def _record_response(agent: Any, response: Any) -> None:
    """Fold a provider response the way Hermes' turn loop does."""
    from agent import turn_response_check

    turn_response_check.record_response_usage(
        agent,
        response,
        messages=[],
        api_call_count=1,
        api_duration=0.0,
        compression_attempts=0,
        max_compression_attempts=3,
    )


class _UsageRecordingAgent:
    """Hermes agent stand-in whose turns run scripted provider responses
    through Hermes' own usage accounting and context compressor, called the
    way Hermes' turn loop calls it. A callable step runs against the agent and
    may raise; ``_COUNTERLESS_FAILURE`` ends the turn with a failure result."""

    def __init__(
        self,
        turns: list[list[dict[str, object] | Callable[[Any], None] | str | None]],
        api_mode: str = "chat_completions",
    ) -> None:
        from agent.context_compressor import ContextCompressor

        self._turns = turns
        self.context_compressor = ContextCompressor(
            model="test-model", quiet_mode=True
        )
        self.provider = "custom"
        self.api_mode = api_mode
        self.model = "test-model"
        self.base_url = "http://127.0.0.1:1/v1"
        self.api_key = ""
        self.client = None
        self.session_id = "runtime-usage"
        self._session_db = None
        self.quiet_mode = True
        self.verbose_logging = False
        for counter in (
            "api_calls",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "estimated_cost_usd",
        ):
            setattr(self, f"session_{counter}", 0)

    def close(self) -> None:
        pass

    def run_conversation(self, user_message: str) -> dict[str, object]:
        messages = [{"role": "user", "content": user_message}]
        for usage in self._turns.pop(0):
            if usage == _COUNTERLESS_FAILURE:
                return {"final_response": "", "failed": True, "messages": messages}
            if callable(usage):
                usage(self)
                continue
            _record_response(self, SimpleNamespace(usage=usage))
        return {
            "final_response": "done",
            "completed": True,
            "messages": messages,
            **{
                key: getattr(self, f"session_{key}")
                for key in (
                    "input_tokens",
                    "cache_read_tokens",
                    "cache_write_tokens",
                    "output_tokens",
                )
            },
        }


def _usage_recording_runtime(agent: _UsageRecordingAgent) -> adapter.HermesRuntime:
    agent_config = _agent_config(
        {"models": {"default": {"provider": "test", "model": "test-model"}}}
    )
    runtime = adapter.HermesRuntime()
    runtime._started = True
    runtime._agent_config = agent_config
    runtime._model_config = agent_config.models["default"]
    runtime._runtime_id = "runtime-usage"
    runtime._agent = agent
    runtime._observe_usage_reports(agent)
    return runtime


def _usage_invocation() -> tuple[AgentRunRequest, RuntimeContext]:
    return _invocation(
        {
            "runtime_context": _runtime_context(
                runtime_id="runtime-usage"
            ).to_mapping(),
            "request": {"input": "next"},
        }
    )


async def test_hermes_usage_from_provider_responses_keeps_unreported_tokens_unknown():
    def reported(prompt: int, completion: int) -> dict[str, int]:
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    agent = _UsageRecordingAgent(
        [
            [reported(120, 7), reported(200, 3)],
            [reported(0, 0)],
            [None],
            [reported(50, 5), None],
            [reported(400, 40)],
        ]
    )
    runtime = _usage_recording_runtime(agent)
    invocation = _usage_invocation()

    two_calls = await runtime.invoke(*invocation)
    reported_zero = await runtime.invoke(*invocation)
    omitted = await runtime.invoke(*invocation)
    mixed = await runtime.invoke(*invocation)
    after_mixed = await runtime.invoke(*invocation)

    assert agent.session_api_calls == 7
    assert two_calls.usage is not None
    assert (two_calls.usage.input_tokens, two_calls.usage.output_tokens) == (320, 10)
    assert reported_zero.usage is not None
    assert reported_zero.usage.input_tokens == 0
    assert reported_zero.usage.output_tokens == 0
    assert reported_zero.usage.total_tokens == 0
    # Hermes counts a response without usage as an API call but adds no
    # tokens, so the invocation's tokens are unknown rather than 0 or partial.
    assert omitted.usage is None
    assert mixed.usage is None
    assert after_mixed.usage is not None
    assert after_mixed.usage.input_tokens == 400
    assert after_mixed.usage.output_tokens == 40


async def test_hermes_usage_with_partial_provider_report_leaves_omitted_count_unknown():
    agent = _UsageRecordingAgent(
        [
            [{"prompt_tokens": 100}],
            [{"completion_tokens": 5}],
            [{"prompt_tokens": 0, "completion_tokens": 0}],
            [{"prompt_tokens": 60, "completion_tokens": 6}, {"prompt_tokens": 70}],
            [{"prompt_tokens": 400, "completion_tokens": 40}],
        ]
    )
    runtime = _usage_recording_runtime(agent)
    invocation = _usage_invocation()

    input_only = await runtime.invoke(*invocation)
    output_only = await runtime.invoke(*invocation)
    reported_zero = await runtime.invoke(*invocation)
    one_partial = await runtime.invoke(*invocation)
    after_partial = await runtime.invoke(*invocation)

    # Hermes normalizes an omitted count to 0; the provider's report decides.
    assert input_only.usage is not None
    assert input_only.usage.input_tokens == 100
    # The provider sent no cache details, so the cached share is unknown.
    assert input_only.usage.cached_input_tokens is None
    assert input_only.usage.output_tokens is None
    assert input_only.usage.total_tokens is None
    assert output_only.usage is not None
    assert output_only.usage.input_tokens is None
    assert output_only.usage.cached_input_tokens is None
    assert output_only.usage.output_tokens == 5
    assert output_only.usage.total_tokens is None
    assert reported_zero.usage is not None
    assert reported_zero.usage.input_tokens == 0
    assert reported_zero.usage.output_tokens == 0
    assert reported_zero.usage.total_tokens == 0
    # One response without an output count leaves the invocation's output unknown.
    assert one_partial.usage is not None
    assert one_partial.usage.input_tokens == 130
    assert one_partial.usage.output_tokens is None
    assert after_partial.usage is not None
    assert after_partial.usage.input_tokens == 400
    assert after_partial.usage.output_tokens == 40
    assert after_partial.usage.total_tokens == 440


async def test_hermes_usage_with_empty_provider_usage_is_unknown():
    # Hermes treats an empty usage object like a missing one; the adapter also
    # counts it as a report with every count left out.
    agent = _UsageRecordingAgent(
        [[{}], [{"prompt_tokens": 400, "completion_tokens": 40}]]
    )
    runtime = _usage_recording_runtime(agent)
    invocation = _usage_invocation()

    empty = await runtime.invoke(*invocation)
    after_empty = await runtime.invoke(*invocation)

    assert agent.session_api_calls == 2
    assert empty.usage is None
    assert after_empty.usage is not None
    assert (after_empty.usage.input_tokens, after_empty.usage.output_tokens) == (
        400,
        40,
    )


async def test_hermes_usage_reports_cache_counts_only_when_the_provider_sent_them():
    chat = _UsageRecordingAgent(
        [
            [
                {
                    "prompt_tokens": 50,
                    "completion_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 20},
                }
            ],
        ]
    )
    # Anthropic's input_tokens excludes the cache, so input needs both cache fields.
    anthropic = _UsageRecordingAgent(
        [
            [{"input_tokens": 100, "output_tokens": 5}],
            [
                {
                    "input_tokens": 100,
                    "cache_read_input_tokens": 30,
                    "cache_creation_input_tokens": 0,
                    "output_tokens": 5,
                }
            ],
        ],
        api_mode="anthropic_messages",
    )
    chat_runtime = _usage_recording_runtime(chat)
    anthropic_runtime = _usage_recording_runtime(anthropic)
    invocation = _usage_invocation()

    cached = await chat_runtime.invoke(*invocation)
    without_cache_fields = await anthropic_runtime.invoke(*invocation)
    with_cache_fields = await anthropic_runtime.invoke(*invocation)

    assert cached.usage is not None
    assert (cached.usage.input_tokens, cached.usage.cached_input_tokens) == (50, 20)
    assert without_cache_fields.usage is not None
    assert without_cache_fields.usage.input_tokens is None
    assert without_cache_fields.usage.cached_input_tokens is None
    assert without_cache_fields.usage.output_tokens == 5
    assert without_cache_fields.usage.total_tokens is None
    assert with_cache_fields.usage is not None
    assert with_cache_fields.usage.input_tokens == 130
    assert with_cache_fields.usage.cached_input_tokens == 30
    assert with_cache_fields.usage.total_tokens == 135


def _reported(prompt: int, completion: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


async def test_hermes_usage_after_failed_or_counterless_turn_is_each_turns_own():
    def fail_turn(agent: Any) -> None:
        raise RuntimeError("turn failed")

    agent = _UsageRecordingAgent(
        [
            [_reported(100, 10)],
            # Hermes' early failure results carry no session counters.
            [_reported(40, 4), _COUNTERLESS_FAILURE],
            [_reported(50, 5)],
            [_reported(70, 7), fail_turn],
            [_reported(60, 6)],
        ]
    )
    runtime = _usage_recording_runtime(agent)
    invocation = _usage_invocation()

    first = await runtime.invoke(*invocation)
    counterless = await runtime.invoke(*invocation)
    after_counterless = await runtime.invoke(*invocation)
    with pytest.raises(RuntimeError):
        await runtime.invoke(*invocation)
    after_exception = await runtime.invoke(*invocation)

    usages = [
        (turn.usage.input_tokens, turn.usage.output_tokens, turn.usage.total_tokens)
        for turn in (first, counterless, after_counterless, after_exception)
        if turn.usage is not None
    ]
    # Each invocation reports its own tokens; the turn that raised never
    # reaches the next invocation's usage.
    assert usages == [(100, 10, 110), (40, 4, 44), (50, 5, 55), (60, 6, 66)]
    assert counterless.status == AgentRunStatus.FAILED


async def test_hermes_usage_after_session_counter_reset_is_unknown_for_that_turn():
    def reset_session(agent: Any) -> None:
        for counter in ("input_tokens", "cache_read_tokens", "output_tokens"):
            setattr(agent, f"session_{counter}", 0)

    agent = _UsageRecordingAgent(
        [
            [_reported(100, 10)],
            [reset_session, _reported(30, 3)],
            [_reported(20, 2)],
        ]
    )
    runtime = _usage_recording_runtime(agent)
    invocation = _usage_invocation()

    await runtime.invoke(*invocation)
    reset = await runtime.invoke(*invocation)
    after_reset = await runtime.invoke(*invocation)

    assert reset.usage is None
    assert after_reset.usage is not None
    assert (after_reset.usage.input_tokens, after_reset.usage.output_tokens) == (20, 2)


@pytest.mark.parametrize(
    ("handler", "action"),
    [
        ("handle_content_policy_refusal", "break"),
        ("recover_from_truncation", "continue"),
        ("recover_from_truncation", "fallthrough"),
        ("retry_invalid_response", "continue"),
        ("retry_invalid_response", "fallthrough"),
    ],
)
async def test_hermes_skipped_response_usage_is_counted_once(handler, action):
    # check_api_response returns after these handlers before recording the
    # response's usage, unless the handler falls through to
    # record_response_usage, which then counts it.
    def positional_handler(agent: Any, response: Any, *args: Any) -> Any:
        return SimpleNamespace(action=action)

    def keyword_handler(agent: Any, *, response: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(action=action)

    # Hermes passes retry_invalid_response the response by keyword.
    observed = adapter._observed_response_handler(
        handler,
        keyword_handler if handler == "retry_invalid_response" else positional_handler,
    )

    def skipped(agent: Any) -> None:
        response = SimpleNamespace(usage=_reported(40, 1))
        if handler == "retry_invalid_response":
            observed(agent, response=response, error_details=[])
        else:
            observed(agent, response, None)
        if action == "fallthrough":
            _record_response(agent, response)

    agent = _UsageRecordingAgent([[skipped, _reported(50, 5)]])
    runtime = _usage_recording_runtime(agent)

    turn = await runtime.invoke(*_usage_invocation())

    assert turn.usage is not None
    assert (turn.usage.input_tokens, turn.usage.output_tokens) == (90, 6)


async def test_hermes_skipped_response_uses_the_route_that_served_it():
    # A refusal handler can switch the agent to a fallback provider before it
    # returns; the refused response still has the first route's usage shape.
    def refuse_and_fall_back(agent: Any, response: Any, *args: Any) -> Any:
        agent.provider = "custom"
        agent.api_mode = "chat_completions"
        return SimpleNamespace(action="break")

    observed = adapter._observed_response_handler(
        "handle_content_policy_refusal", refuse_and_fall_back
    )

    def refused(agent: Any) -> None:
        anthropic_usage = {
            "input_tokens": 100,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 0,
            "output_tokens": 0,
        }
        observed(agent, SimpleNamespace(usage=anthropic_usage), None)

    fallback_usage = {
        **_reported(50, 5),
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    agent = _UsageRecordingAgent(
        [[refused, fallback_usage]], api_mode="anthropic_messages"
    )
    runtime = _usage_recording_runtime(agent)

    turn = await runtime.invoke(*_usage_invocation())

    assert turn.usage is not None
    assert (turn.usage.input_tokens, turn.usage.cached_input_tokens) == (180, 30)
    assert turn.usage.output_tokens == 5


async def test_hermes_stop_unregisters_the_agent_usage_observer():
    agent = _UsageRecordingAgent([])
    runtime = _usage_recording_runtime(agent)
    assert agent in adapter._USAGE_OBSERVERS
    assert agent in adapter._SUMMARY_ATTEMPT_OBSERVERS

    await runtime.stop()

    assert agent not in adapter._USAGE_OBSERVERS
    assert agent not in adapter._SUMMARY_ATTEMPT_OBSERVERS


@pytest.mark.parametrize("drift", ["replaced", "not_called"])
def test_hermes_usage_observer_rejects_unknown_usage_call_site(monkeypatch, drift):
    from agent import turn_response_check
    from agent import turn_usage

    monkeypatch.setattr(
        turn_response_check, "record_response_usage", turn_usage.record_response_usage
    )
    if drift == "replaced":
        monkeypatch.setattr(
            turn_response_check, "record_response_usage", lambda *args, **kwargs: None
        )
    else:
        monkeypatch.setattr(
            turn_response_check, "check_api_response", lambda *args, **kwargs: None
        )

    with pytest.raises(adapter.lifecycle.LifecycleError) as raised:
        adapter._install_usage_observer()

    assert raised.value.code == "hermes_usage_accounting_unsupported"


@pytest.mark.parametrize(
    ("request_input", "expected_message"),
    [
        (0, "0"),
        (False, "false"),
        ([], "[]"),
        ({}, "{}"),
    ],
)
def test_user_message_preserves_falsy_request_input(
    request_input: object,
    expected_message: str,
):
    request = AgentRunRequest(input=request_input)

    assert common_utils.normalize_user_input(request.input) == expected_message


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

    invoke_task = asyncio.create_task(runtime.invoke(*_invocation(invocation)))
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
    mock_agent.session_input_tokens = 0
    mock_agent.session_cache_read_tokens = 0
    mock_agent.session_cache_write_tokens = 0
    mock_agent.session_output_tokens = 0
    mock_session_db = MagicMock()
    calls = 0

    def run_turn(**_kwargs):
        nonlocal calls
        calls += 1
        # Each turn bills calls * 100 input and calls * 10 output tokens.
        mock_agent.session_input_tokens += calls * 100
        mock_agent.session_output_tokens += calls * 10
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

    cancelled_invoke = asyncio.create_task(runtime.invoke(*_invocation(invocation)))
    assert await asyncio.to_thread(worker_started.wait, 1)
    active_invoke_task = runtime._active_invoke_task
    assert active_invoke_task is not None

    cancelled_invoke.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_invoke

    worker_release.set()
    await asyncio.wait_for(asyncio.shield(active_invoke_task), timeout=1)

    result = await runtime.invoke(*_invocation(invocation))

    assert result.output["response"] == "turn-2"
    # The cancelled turn's tokens are in the counters before this turn starts,
    # so this result reports only its own.
    assert result.usage is not None
    assert (result.usage.input_tokens, result.usage.output_tokens) == (200, 20)
    await runtime.stop()


def test_main_serves_persistent_runtime(monkeypatch):
    serve = MagicMock()
    monkeypatch.setattr(adapter.lifecycle, "serve", serve)

    adapter.main()

    serve.assert_called_once_with(
        adapter.HermesRuntime,
        config_loader=AgentConfig.from_mapping,
    )


def _chat_response(
    content: str,
    usage: dict[str, int] | None,
    tool_calls: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    response: dict[str, object] = {
        "id": "scripted",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
    }
    if usage is not None:
        response["usage"] = usage
    return response


def _stream_events(response: dict[str, object]) -> bytes:
    """Return a chat completion as the server-sent events of a stream."""
    choice = response["choices"][0]  # type: ignore[index]
    message = choice["message"]
    delta = {"role": "assistant", "content": message["content"]}
    if "tool_calls" in message:
        delta["tool_calls"] = [
            {**call, "index": index} for index, call in enumerate(message["tool_calls"])
        ]
    header = {"id": "scripted", "object": "chat.completion.chunk", "created": 0}
    chunks = [
        {**header, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {
            **header,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}
            ],
        },
    ]
    if "usage" in response:
        chunks.append({**header, "choices": [], "usage": response["usage"]})
    events = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return f"{events}data: [DONE]\n\n".encode()


def _usage(prompt: int, completion: int | None = None) -> dict[str, int]:
    usage = {"prompt_tokens": prompt}
    if completion is not None:
        usage["completion_tokens"] = completion
        usage["total_tokens"] = prompt + completion
    return usage


class _ScriptedProvider:
    """An OpenAI-compatible endpoint that answers Hermes' requests from scripts.

    Hermes' main turn loop sends its tools, a session title call asks for a
    JSON response format, and the compression summary call sends neither.
    When ``failed_summary`` holds a status and body, every summary call gets
    that answer instead.
    """

    def __init__(self) -> None:
        from http.server import BaseHTTPRequestHandler
        from http.server import ThreadingHTTPServer

        self.main: list[dict[str, object]] = []
        self.summaries: list[dict[str, object]] = []
        self.failed_summary: tuple[int, bytes] | None = None
        self.served: list[str] = []
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                self.send_response(404)
                self.end_headers()

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if not self.path.endswith("/chat/completions"):
                    self.send_response(404)
                    self.end_headers()
                    return
                if "response_format" in body:
                    kind = "title"
                    response = _chat_response('{"title": "Scripted"}', _usage(30, 5))
                elif body.get("tools"):
                    kind = "main"
                    response = provider.main.pop(0)
                elif provider.failed_summary is not None:
                    status, payload = provider.failed_summary
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    provider.served.append("failed summary")
                    return
                else:
                    kind = "summary"
                    response = provider.summaries.pop(0)
                if body.get("stream"):
                    payload = _stream_events(response)
                    content_type = "text/event-stream"
                else:
                    payload = json.dumps(response).encode()
                    content_type = "application/json"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                provider.served.append(kind)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture(name="scripted_provider")
def scripted_provider_fixture():
    provider = _ScriptedProvider()
    yield provider
    provider.close()


async def _start_scripted_runtime(
    provider: _ScriptedProvider, tmp_path: Path
) -> tuple[adapter.HermesRuntime, RuntimeContext]:
    os.environ["NVIDIA_API_KEY"] = "scripted-test-key"
    runtime_context = _runtime_context(
        runtime_id="runtime-aux",
        workspace=str(tmp_path),
        artifact_root=str(tmp_path / "artifacts"),
    )
    runtime = adapter.HermesRuntime()
    await runtime.start(
        {
            "config": _agent_config(
                {
                    "models": {
                        "default": {
                            "provider": "nvidia",
                            "model": "test-model",
                            "base_url": provider.base_url,
                        }
                    }
                }
            ),
            "runtime_context": runtime_context.to_mapping(),
            "base_dir": str(tmp_path),
        }
    )
    return runtime, runtime_context


def _tool_call_over_compression_threshold(runtime: adapter.HermesRuntime) -> dict:
    """A main response whose prompt count makes Hermes compress before the next."""
    prompt = runtime._agent.context_compressor.threshold_tokens + 10
    return _chat_response(
        "",
        _usage(prompt, 7),
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "todo", "arguments": "{}"},
            }
        ],
    )


# Enough earlier exchanges for Hermes' compressor to summarize some of them.
_LONG_HISTORY = [
    message
    for step in range(30)
    for message in (
        {"role": "user", "content": f"step {step}: " + "detail " * 400},
        {"role": "assistant", "content": f"did step {step}: " + "result " * 400},
    )
]


async def _invoke(
    runtime: adapter.HermesRuntime, runtime_context: RuntimeContext, text: str
) -> Any:
    return await runtime.invoke(AgentRunRequest(input=text), runtime_context)


async def test_hermes_usage_counts_compression_summary_calls(
    scripted_provider: _ScriptedProvider, tmp_path: Path
):
    runtime, runtime_context = await _start_scripted_runtime(
        scripted_provider, tmp_path
    )
    try:
        threshold_call = _tool_call_over_compression_threshold(runtime)
        prompt = threshold_call["usage"]["prompt_tokens"]  # type: ignore[index]
        scripted_provider.main.extend(
            [threshold_call, _chat_response("done", _usage(500, 9))]
        )
        scripted_provider.summaries.append(
            _chat_response("summary of earlier steps", _usage(1000, 100))
        )
        runtime._conversation_history = list(_LONG_HISTORY)

        compressed = await _invoke(runtime, runtime_context, "Fix the parser bug")

        scripted_provider.main.append(_chat_response("again", _usage(40, 4)))
        plain = await _invoke(runtime, runtime_context, "Thanks")
    finally:
        await runtime.stop()

    assert scripted_provider.served == ["main", "summary", "main", "main"]
    assert compressed.status == AgentRunStatus.SUCCEEDED
    assert compressed.usage is not None
    # The main calls and the compression summary call.
    assert compressed.usage.input_tokens == prompt + 500 + 1000
    assert compressed.usage.output_tokens == 7 + 9 + 100
    assert compressed.usage.total_tokens == prompt + 500 + 1000 + 116
    # A turn without auxiliary calls reports its main calls only.
    assert plain.usage is not None
    assert (plain.usage.input_tokens, plain.usage.output_tokens) == (40, 4)
    assert plain.usage.total_tokens == 44


async def test_hermes_usage_leaves_counts_omitted_by_compression_unknown(
    scripted_provider: _ScriptedProvider, tmp_path: Path
):
    runtime, runtime_context = await _start_scripted_runtime(
        scripted_provider, tmp_path
    )
    try:
        threshold_call = _tool_call_over_compression_threshold(runtime)
        prompt = threshold_call["usage"]["prompt_tokens"]  # type: ignore[index]
        scripted_provider.main.extend(
            [threshold_call, _chat_response("done", _usage(500, 9))]
        )
        # The summary call reports its input count and leaves out its output.
        scripted_provider.summaries.append(
            _chat_response("summary of earlier steps", _usage(1000))
        )
        runtime._conversation_history = list(_LONG_HISTORY)

        turn = await _invoke(runtime, runtime_context, "Fix the parser bug")
    finally:
        await runtime.stop()

    assert scripted_provider.served == ["main", "summary", "main"]
    assert turn.usage is not None
    assert turn.usage.input_tokens == prompt + 500 + 1000
    assert turn.usage.output_tokens is None
    assert turn.usage.total_tokens is None


@pytest.mark.parametrize(
    "failed_summary",
    [
        # The provider answered, but the client cannot parse the body.
        (200, b'{"id": "scripted", "choices": [ {"message": '),
        # The request raised.
        (400, b'{"error": {"message": "upstream reset", "type": "bad_request"}}'),
    ],
    ids=["unparseable", "rejected"],
)
async def test_hermes_usage_is_unknown_when_a_compression_summary_request_fails(
    scripted_provider: _ScriptedProvider, tmp_path: Path, failed_summary
):
    runtime, runtime_context = await _start_scripted_runtime(
        scripted_provider, tmp_path
    )
    try:
        scripted_provider.main.extend(
            [
                _tool_call_over_compression_threshold(runtime),
                _chat_response("done", _usage(500, 9)),
            ]
        )
        scripted_provider.failed_summary = failed_summary
        runtime._conversation_history = list(_LONG_HISTORY)

        turn = await _invoke(runtime, runtime_context, "Fix the parser bug")
    finally:
        await runtime.stop()

    # A failed summary request may have been billed, so the counts the
    # invocation reports would be an undercount.
    assert "failed summary" in scripted_provider.served
    assert turn.status == AgentRunStatus.SUCCEEDED
    assert turn.usage is None


def _write_file_call(step: int) -> dict[str, object]:
    return _chat_response(
        "",
        _usage(100, 1),
        tool_calls=[
            {
                "id": f"call-{step}",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps(
                        {"path": f"out-{step}.txt", "content": f"line {step}"}
                    ),
                },
            }
        ],
    )


async def test_hermes_runs_no_background_skill_review_after_a_turn(
    scripted_provider: _ScriptedProvider, tmp_path: Path
):
    runtime, runtime_context = await _start_scripted_runtime(
        scripted_provider, tmp_path
    )
    agent = runtime._agent
    reviews: list[dict[str, object]] = []
    spawn_review = agent._spawn_background_review

    def record_review(**kwargs: object) -> None:
        reviews.append(kwargs)
        spawn_review(**kwargs)

    agent._spawn_background_review = record_review
    try:
        # Ten tool iterations reach Hermes' default skill-review interval.
        scripted_provider.main.extend(
            [_write_file_call(step) for step in range(10)]
            + [_chat_response("done", _usage(100, 1))]
        )
        turn = await _invoke(runtime, runtime_context, "Write the files")
    finally:
        await runtime.stop()

    # The review fork would run after the turn returned, with its tokens
    # outside the auxiliary client.
    assert agent.skip_background_review is True
    assert reviews == []
    assert scripted_provider.served == ["main"] * 11
    assert turn.usage is not None
    assert (turn.usage.input_tokens, turn.usage.output_tokens) == (1100, 11)


def _install_auxiliary_observer_fresh(monkeypatch) -> tuple[Any, list[tuple]]:
    """Install the auxiliary observer over a recording stand-in for Hermes'
    recorder, and return the stand-in's calls."""
    from agent import aux_accounting

    recorded: list[tuple] = []

    def record_aux_usage(response, task, *, provider=None, base_url=None):
        recorded.append((response, task, provider))

    monkeypatch.setattr(aux_accounting, "record_aux_usage", record_aux_usage)
    adapter._install_auxiliary_usage_observer()
    return aux_accounting, recorded


def test_auxiliary_usage_counts_only_the_runtime_session_and_not_moa(monkeypatch):
    aux_accounting, recorded = _install_auxiliary_observer_fresh(monkeypatch)
    session_db = object()
    calls = adapter._AuxiliaryCalls(session_db)
    response = SimpleNamespace(usage=_usage(10, 2))

    def record(task: str, db: object) -> None:
        owner = adapter._AUXILIARY_CALLS.set(calls)
        session = aux_accounting._accounting.set((db, "session"))
        try:
            aux_accounting.record_aux_usage(response, task, provider="nvidia")
        finally:
            aux_accounting._accounting.reset(session)
            adapter._AUXILIARY_CALLS.reset(owner)

    record("compression", session_db)
    # MoA usage is already in the session counters, and a delegated
    # subagent bills its own session database handle.
    record("moa_reference", session_db)
    record("compression", object())
    # Outside an invocation the call belongs to no invocation.
    aux_accounting.record_aux_usage(response, "compression", provider="nvidia")
    tokens, unreported = calls.close()

    assert len(recorded) == 4
    assert tokens == {
        "input_tokens": 10,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "output_tokens": 2,
    }
    # The chat usage sent no cache details.
    assert unreported == frozenset({"cached"})


@pytest.mark.parametrize(
    ("usage", "unknown"),
    [
        (None, {"input", "cached", "output"}),
        ({}, {"input", "cached", "output"}),
        ({"prompt_tokens": "10", "completion_tokens": 2}, {"input", "cached"}),
        (
            {"prompt_tokens": -1, "completion_tokens": True},
            {"input", "cached", "output"},
        ),
        (
            {
                "prompt_tokens": 10,
                "completion_tokens": 0,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
            set(),
        ),
        # Hermes' converting clients build this shape and fill omitted
        # counts with 0.
        (
            SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            {"input", "cached", "output"},
        ),
    ],
)
def test_auxiliary_calls_leave_unreported_counts_unknown(usage, unknown):
    calls = adapter._AuxiliaryCalls(object())

    calls.add_response(SimpleNamespace(usage=usage), "nvidia")

    assert calls.close()[1] == frozenset(unknown)


def test_auxiliary_calls_drop_usage_that_arrives_after_the_invocation():
    calls = adapter._AuxiliaryCalls(object())

    tokens, unreported = calls.close()
    calls.add_response(SimpleNamespace(usage=_usage(10, 2)), "nvidia")
    calls.mark_unaccounted()

    assert tokens == dict.fromkeys(adapter._SESSION_TOKEN_KEYS, 0)
    assert unreported == frozenset()
    assert calls.close() == (tokens, unreported)


@pytest.mark.parametrize("drift", ["not_called", "no_excluded_tasks"])
def test_auxiliary_usage_observer_rejects_unknown_call_site(monkeypatch, drift):
    from agent import aux_accounting
    from agent import auxiliary_client

    monkeypatch.setattr(
        aux_accounting, "record_aux_usage", lambda *args, **kwargs: None
    )
    if drift == "not_called":
        monkeypatch.setattr(
            auxiliary_client,
            "_validate_llm_response",
            lambda response, task=None: response,
        )
    else:
        monkeypatch.setattr(aux_accounting, "_EXCLUDED_TASKS", None)

    with pytest.raises(adapter.lifecycle.LifecycleError) as raised:
        adapter._install_auxiliary_usage_observer()

    assert raised.value.code == "hermes_usage_accounting_unsupported"


def _anthropic_aux_client(usage: Any) -> Any:
    """Hermes' Anthropic auxiliary client over a scripted Anthropic SDK client."""
    from agent import auxiliary_client

    message = SimpleNamespace(
        id="msg-scripted",
        type="message",
        role="assistant",
        model="claude-test",
        content=[SimpleNamespace(type="text", text="summary")],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=usage,
    )
    sdk = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: message))
    return auxiliary_client.AnthropicAuxiliaryClient(
        sdk, "claude-test", api_key="test", base_url="https://api.anthropic.com"
    )


_ANTHROPIC_REQUEST = {
    "model": "claude-test",
    "messages": [{"role": "user", "content": "summarize"}],
    "max_tokens": 64,
}


@pytest.mark.parametrize("relay_route", [False, True])
@pytest.mark.parametrize(
    ("usage", "tokens", "unknown"),
    [
        (
            SimpleNamespace(
                input_tokens=100,
                output_tokens=20,
                cache_read_input_tokens=30,
                cache_creation_input_tokens=5,
            ),
            (100, 30, 5, 20),
            set(),
        ),
        (
            SimpleNamespace(
                input_tokens=100, output_tokens=0, cache_read_input_tokens=30
            ),
            (100, 30, 0, 0),
            {"input"},
        ),
    ],
)
def test_anthropic_auxiliary_usage_comes_from_the_raw_anthropic_message(
    monkeypatch, relay_route, usage, tokens, unknown
):
    from agent import auxiliary_client

    aux_accounting, _ = _install_auxiliary_observer_fresh(monkeypatch)
    session_db = object()
    calls = adapter._AuxiliaryCalls(session_db)
    owner = adapter._AUXILIARY_CALLS.set(calls)
    session = aux_accounting._accounting.set((session_db, "session"))
    # With a Relay auxiliary route, Hermes runs the call through
    # run_with_aux_hooks and relay_llm.execute.
    route = auxiliary_client._RELAY_AUX_CALL_CONTEXT.set(
        {
            "task": "compression",
            "request_id": "request-1",
            "provider": "anthropic",
            "model": "claude-test",
        }
        if relay_route
        else None
    )
    try:
        response = auxiliary_client._relay_sync_completion(
            _anthropic_aux_client(usage),
            dict(_ANTHROPIC_REQUEST),
            provider="anthropic",
            api_mode="anthropic_messages",
        )
        auxiliary_client._validate_llm_response(
            response, "compression", provider="anthropic"
        )
    finally:
        auxiliary_client._RELAY_AUX_CALL_CONTEXT.reset(route)
        aux_accounting._accounting.reset(session)
        adapter._AUXILIARY_CALLS.reset(owner)

    counted, unreported = calls.close()

    # Hermes' converted usage reports 100 input tokens with no cache counts.
    assert response.usage.prompt_tokens == 100
    assert tuple(counted[key] for key in adapter._SESSION_TOKEN_KEYS) == tokens
    assert unreported == frozenset(unknown)


@pytest.mark.parametrize("provider", ["anthropic", "minimax", None])
def test_anthropic_auxiliary_response_seen_as_another_object_is_unknown(
    monkeypatch, provider
):
    from agent import anthropic_adapter
    from agent import auxiliary_client

    aux_accounting, _ = _install_auxiliary_observer_fresh(monkeypatch)
    session_db = object()
    calls = adapter._AuxiliaryCalls(session_db)
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=20,
        cache_read_input_tokens=30,
        cache_creation_input_tokens=5,
    )
    client = _anthropic_aux_client(usage)
    owner = adapter._AUXILIARY_CALLS.set(calls)
    session = aux_accounting._accounting.set((session_db, "session"))
    try:
        response = client.chat.completions.create(**_ANTHROPIC_REQUEST)
        # A managed Relay attempt that rewrote the response returns a copy.
        auxiliary_client._validate_llm_response(
            SimpleNamespace(**vars(response)), "compression", provider=provider
        )
        # A main-loop Anthropic call is not an auxiliary call.
        anthropic_adapter.create_anthropic_message(
            client._real_client, dict(_ANTHROPIC_REQUEST)
        )
    finally:
        aux_accounting._accounting.reset(session)
        adapter._AUXILIARY_CALLS.reset(owner)

    counted, unreported = calls.close()

    assert counted == dict.fromkeys(adapter._SESSION_TOKEN_KEYS, 0)
    assert unreported == frozenset({"input", "cached", "output"})


def _owned_context(aux_accounting: Any, calls: Any, session_db: object) -> Callable:
    """Run a function as an auxiliary call of ``calls``' invocation."""

    def run(function: Callable[[], Any]) -> Any:
        owner = adapter._AUXILIARY_CALLS.set(calls)
        session = aux_accounting._accounting.set((session_db, "session"))
        try:
            return function()
        finally:
            aux_accounting._accounting.reset(session)
            adapter._AUXILIARY_CALLS.reset(owner)

    return run


def test_auxiliary_request_that_raised_leaves_counts_unknown(monkeypatch):
    from agent import auxiliary_client

    aux_accounting, _ = _install_auxiliary_observer_fresh(monkeypatch)
    session_db = object()
    calls = adapter._AuxiliaryCalls(session_db)

    def refuse(**kwargs):
        raise RuntimeError("connection reset after the provider ran")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=refuse))
    )
    with pytest.raises(RuntimeError):
        _owned_context(aux_accounting, calls, session_db)(
            lambda: auxiliary_client._relay_sync_completion(
                client, dict(_ANTHROPIC_REQUEST), provider="nvidia"
            )
        )

    assert calls.close()[1] == frozenset({"input", "cached", "output"})


def test_async_auxiliary_request_that_raised_leaves_counts_unknown(monkeypatch):
    aux_accounting, _ = _install_auxiliary_observer_fresh(monkeypatch)
    session_db = object()
    calls = adapter._AuxiliaryCalls(session_db)

    async def refuse(*args, **kwargs):
        raise TimeoutError("no response")

    request = adapter._observed_async_relay_completion(refuse)
    with pytest.raises(TimeoutError):
        _owned_context(aux_accounting, calls, session_db)(
            lambda: asyncio.run(request())
        )

    assert calls.close()[1] == frozenset({"input", "cached", "output"})


def test_auxiliary_request_in_flight_when_the_invocation_returns_is_unknown(
    monkeypatch,
):
    aux_accounting, _ = _install_auxiliary_observer_fresh(monkeypatch)
    session_db = object()
    calls = adapter._AuxiliaryCalls(session_db)
    sent = threading.Event()
    release = threading.Event()

    def slow_request(*args, **kwargs):
        sent.set()
        release.wait(timeout=10)
        return SimpleNamespace(usage=_usage(10, 2))

    request = adapter._observed_relay_completion(slow_request)
    worker = threading.Thread(
        target=_owned_context(aux_accounting, calls, session_db), args=(request,)
    )
    worker.start()
    try:
        assert sent.wait(timeout=10)
        tokens, unreported = calls.close()
    finally:
        release.set()
        worker.join(timeout=10)

    assert tokens == dict.fromkeys(adapter._SESSION_TOKEN_KEYS, 0)
    assert unreported == frozenset({"input", "cached", "output"})


@pytest.mark.parametrize("call", ["mixture_of_agents", "subagent"])
def test_anthropic_capture_of_an_uncounted_call_is_forgotten(monkeypatch, call):
    from agent import auxiliary_client

    aux_accounting, _ = _install_auxiliary_observer_fresh(monkeypatch)
    session_db = object()
    calls = adapter._AuxiliaryCalls(session_db)
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=20,
        cache_read_input_tokens=30,
        cache_creation_input_tokens=5,
    )
    client = _anthropic_aux_client(usage)
    # A subagent bills its own session database handle.
    billed_db = session_db if call == "mixture_of_agents" else object()
    task = "moa_reference" if call == "mixture_of_agents" else "compression"

    def make_call() -> None:
        response = client.chat.completions.create(**_ANTHROPIC_REQUEST)
        auxiliary_client._validate_llm_response(response, task, provider="anthropic")

    _owned_context(aux_accounting, calls, billed_db)(make_call)

    assert calls.close() == (dict.fromkeys(adapter._SESSION_TOKEN_KEYS, 0), frozenset())


@pytest.mark.parametrize("site", ["anthropic_create", "request_caller"])
def test_auxiliary_usage_observer_rejects_moved_request_paths(monkeypatch, site):
    from agent import aux_accounting
    from agent import auxiliary_client

    monkeypatch.setattr(
        aux_accounting, "record_aux_usage", lambda *args, **kwargs: None
    )
    if site == "anthropic_create":
        monkeypatch.setattr(
            auxiliary_client._AnthropicCompletionsAdapter,
            "create",
            lambda self, **kwargs: None,
        )
    else:
        monkeypatch.setattr(
            auxiliary_client, "_retry_same_provider_sync", lambda **kwargs: None
        )

    with pytest.raises(adapter.lifecycle.LifecycleError) as raised:
        adapter._install_auxiliary_usage_observer()

    assert raised.value.code == "hermes_usage_accounting_unsupported"
