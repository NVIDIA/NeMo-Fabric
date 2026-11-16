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
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from _hermes_skills import loaded_skill
from _hermes_skills import other_skill
from _hermes_skills import skill_copy
from _hermes_skills import skipped_skill
from _hermes_skills import write_skill
from _utils.utils import POSIX_PERMISSIONS
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentMcpServerConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunStatus
from nemo_fabric_adapter_contract.models import RuntimeContext

pytestmark = pytest.mark.usefixtures("requires_hermes_agent")

if importlib.util.find_spec("run_agent") is not None:
    from hermes_state import SessionDB
    from run_agent import AIAgent

    import nemo_fabric_adapters.common.utils as common_utils

    from nemo_fabric_adapters.hermes import adapter
    from nemo_fabric_adapters.hermes import configuration
    from nemo_fabric_adapters.hermes import skills as hermes_skills
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


def test_native_relay_segment_does_not_fire_session_end_hooks(monkeypatch):
    from agent import relay_runtime
    from hermes_cli import lifecycle as hermes_lifecycle
    from nemo_relay import subscribers

    coordinator = MagicMock()
    finalize_session = MagicMock()
    flush = MagicMock()
    monkeypatch.setattr(relay_runtime, "SESSION_COORDINATOR", coordinator)
    monkeypatch.setattr(relay_runtime, "current_profile_key", lambda: "test-profile")
    monkeypatch.setattr(hermes_lifecycle, "finalize_session", finalize_session)
    monkeypatch.setattr(subscribers, "flush", flush)

    telemetry.finalize_native_relay_segment("native-session")

    coordinator.finalize_conversation.assert_called_once_with(
        profile_key="test-profile", session_id="native-session"
    )
    flush.assert_called_once_with()
    finalize_session.assert_not_called()


def test_validate_hermes_telemetry_provider_rejects_mixed_native_and_relay():
    with pytest.raises(
        ValueError, match="only relay telemetry is supported for Hermes"
    ):
        telemetry.validate_hermes_telemetry_provider(
            _runtime_context(providers=["relay", "native"])
        )


@pytest.mark.parametrize(
    "relay_version",
    ["0.10.0", "0.10.7", "0.10.0+build.1", "0.11.0", "0.11.7", "0.11.0+build.1"],
)
def test_validate_hermes_relay_compatibility_accepts_supported_checkout(
    monkeypatch, relay_version
):
    monkeypatch.setattr(telemetry, "version", lambda _name: relay_version)
    telemetry.validate_hermes_relay_compatibility()


@pytest.mark.parametrize(
    "relay_version",
    [
        "0.8.4",
        "0.010.0",
        "0.011.0",
        "0.9.0",
        "0.12.0",
        "1.0.0",
        "0.10.0rc1",
        "0.10.0.dev1",
        "0.11.0rc1",
        "0.11.0.dev1",
    ],
)
def test_validate_hermes_relay_compatibility_rejects_unsupported_relay(
    monkeypatch, relay_version
):
    monkeypatch.setattr(telemetry, "version", lambda _name: relay_version)

    with pytest.raises(
        RuntimeError,
        match=f"requires nemo-relay 0.10 or 0.11; found {relay_version}",
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

    with pytest.raises(RuntimeError, match="does not support Relay telemetry"):
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
        "runtime.keep_state",
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
    """Runtime start discovers MCP tools when configured."""
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
    tools_mcp.get_mcp_status = lambda configured: [  # type: ignore[attr-defined]
        {"name": name, "status": "connected"} for name in configured
    ]

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


def _partial_mcp_config() -> AgentConfig:
    return _agent_config(
        {
            "models": {"default": {"provider": "test", "model": "test-model"}},
            "mcp": {
                "servers": {
                    "time": {"transport": "stdio", "url": "time-mcp"},
                    "offline": {
                        "transport": "streamable-http",
                        "url": "http://127.0.0.1:9/mcp",
                    },
                    "missing": {"transport": "stdio", "url": "missing-mcp"},
                    "pending": {"transport": "stdio", "url": "pending-mcp"},
                }
            },
        }
    )


def _install_mcp_status(monkeypatch, *snapshots: list[dict[str, object]]) -> MagicMock:
    """Serve Hermes's MCP status, one snapshot per read."""

    get_mcp_status = MagicMock(side_effect=list(snapshots))
    discovery = ModuleType("tools.mcp_tool_discovery")
    discovery.get_mcp_status = get_mcp_status  # type: ignore[attr-defined]
    discovery.discover_mcp_tools = MagicMock(return_value=[])  # type: ignore[attr-defined]
    lifecycle = ModuleType("tools.mcp_tool_lifecycle")
    lifecycle.shutdown_mcp_servers = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tools", ModuleType("tools"))
    monkeypatch.setitem(sys.modules, "tools.mcp_tool_discovery", discovery)
    monkeypatch.setitem(sys.modules, "tools.mcp_tool_lifecycle", lifecycle)
    return get_mcp_status


async def test_runtime_reports_current_unavailable_mcp_servers_on_each_result(
    monkeypatch, capsys
):
    """Runtime reports current unavailable MCP servers on each result."""
    mock_turn = MagicMock(
        return_value=({"final_response": "done", "completed": True}, "")
    )
    monkeypatch.setattr(adapter, "_invoke_hermes_turn", mock_turn)
    agent_config = _partial_mcp_config()
    runtime = adapter.HermesRuntime()
    runtime._started = True
    runtime._agent_config = agent_config
    runtime._model_config = agent_config.models["default"]
    runtime._runtime_id = "runtime-partial-mcp"
    runtime._agent = MagicMock()
    runtime._hermes_config = configuration.build_hermes_config(
        agent_config, workspace="."
    )
    get_mcp_status = _install_mcp_status(
        monkeypatch,
        [
            {"name": "time", "status": "connected"},
            {"name": "offline", "status": "failed", "error": "secret-token"},
            {"name": "missing", "status": "failed", "error": "missing executable"},
            {"name": "pending", "status": "connecting"},
        ],
        [
            {"name": "time", "status": "connected"},
            {"name": "offline", "status": "connected"},
            {"name": "missing", "status": "failed", "error": "missing executable"},
            {"name": "pending", "status": "configured"},
        ],
        [
            {"name": "time", "status": "connected"},
            {"name": "offline", "status": "connected"},
            {"name": "missing", "status": "lazy"},
            {"name": "pending", "status": "connected"},
        ],
    )
    request, context = _invocation(
        {
            "runtime_context": _runtime_context(
                runtime_id="runtime-partial-mcp"
            ).to_mapping(),
            "request": {"input": "what time is it"},
        }
    )

    first = await runtime.invoke(request, context)
    first_warnings = capsys.readouterr().err
    second = await runtime.invoke(request, context)
    second_warnings = capsys.readouterr().err
    third = await runtime.invoke(request, context)

    assert first.status is AgentRunStatus.SUCCEEDED
    assert first.output["mcp"] == {
        "unavailable_servers": [
            {"name": "missing", "transport": "stdio", "reason": "failed"},
            {"name": "offline", "transport": "streamable-http", "reason": "failed"},
        ]
    }
    # A server Hermes reconnected is no longer reported.
    assert second.output["mcp"] == {
        "unavailable_servers": [
            {"name": "missing", "transport": "stdio", "reason": "failed"},
        ]
    }
    assert "mcp" not in third.output
    assert "secret-token" not in json.dumps(first.output)
    assert first_warnings.count("is unavailable") == 2
    assert "'offline' (streamable-http) is unavailable: failed" in first_warnings
    assert second_warnings == ""
    get_mcp_status.assert_called_with(runtime._hermes_config["mcp_servers"])


async def test_runtime_omits_mcp_report_when_hermes_status_fails(monkeypatch):
    """A failing status read leaves the completed turn intact without a report."""
    mock_turn = MagicMock(
        return_value=({"final_response": "done", "completed": True}, "")
    )
    monkeypatch.setattr(adapter, "_invoke_hermes_turn", mock_turn)
    agent_config = _partial_mcp_config()
    runtime = adapter.HermesRuntime()
    runtime._started = True
    runtime._agent_config = agent_config
    runtime._model_config = agent_config.models["default"]
    runtime._runtime_id = "runtime-partial-mcp"
    runtime._agent = MagicMock()
    runtime._hermes_config = configuration.build_hermes_config(
        agent_config, workspace="."
    )
    get_mcp_status = _install_mcp_status(monkeypatch)
    get_mcp_status.side_effect = RuntimeError("status secret-token")
    request, context = _invocation(
        {
            "runtime_context": _runtime_context(
                runtime_id="runtime-partial-mcp"
            ).to_mapping(),
            "request": {"input": "what time is it"},
        }
    )

    result = await runtime.invoke(request, context)

    assert result.status is AgentRunStatus.SUCCEEDED
    assert "mcp" not in result.output
    assert "secret-token" not in json.dumps(result.output)


async def test_runtime_starts_when_mcp_servers_are_unavailable(
    monkeypatch, tmp_path: Path, capsys
):
    """Runtime starts when MCP servers are unavailable."""
    mock_ai_agent = MagicMock(spec=AIAgent)
    mock_ai_agent_type = MagicMock(spec=AIAgent, return_value=mock_ai_agent)
    monkeypatch.setattr(
        mock_ai_agent_type.__init__.__func__,
        "__signature__",
        inspect.signature(AIAgent.__init__),
        raising=False,
    )
    hermes_config = ModuleType("hermes_cli.config")
    hermes_config.load_config = lambda: {}  # type: ignore[attr-defined]
    hermes_plugins = ModuleType("hermes_cli.plugins")
    hermes_plugins.discover_plugins = lambda force=False: None  # type: ignore[attr-defined]
    hermes_state = ModuleType("hermes_state")
    hermes_state.SessionDB = MagicMock(spec=SessionDB)  # type: ignore[attr-defined]
    run_agent = ModuleType("run_agent")
    run_agent.AIAgent = mock_ai_agent_type  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_cli", ModuleType("hermes_cli"))
    monkeypatch.setitem(sys.modules, "hermes_cli.config", hermes_config)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", hermes_plugins)
    monkeypatch.setitem(sys.modules, "hermes_state", hermes_state)
    monkeypatch.setitem(sys.modules, "run_agent", run_agent)
    get_mcp_status = _install_mcp_status(
        monkeypatch,
        [
            {"name": "time", "status": "connected"},
            {"name": "offline", "status": "failed"},
            {"name": "missing", "status": "failed"},
            {"name": "pending", "status": "failed"},
        ],
    )
    discovery = sys.modules["tools.mcp_tool_discovery"]
    mcp_lifecycle = sys.modules["tools.mcp_tool_lifecycle"]
    monkeypatch.setenv("TEST_API_KEY", "secret")
    agent_config = _agent_config(
        {
            **_partial_mcp_config().to_mapping(),
            "tools": {"enabled": ["mcp-time"]},
            "models": {
                "default": {
                    "provider": "test",
                    "model": "test-model",
                    "api_key_env": "TEST_API_KEY",
                }
            },
        }
    )

    runtime = adapter.HermesRuntime()
    await runtime.start(
        {
            "base_dir": str(tmp_path),
            "config": agent_config,
            "runtime_context": _runtime_context(
                runtime_id="runtime-partial-start",
                workspace=str(tmp_path),
                artifact_root=str(tmp_path / "artifacts"),
            ).to_mapping(),
        }
    )
    try:
        assert runtime._started is True
        discovery.discover_mcp_tools.assert_called_once_with()  # type: ignore[attr-defined]
        get_mcp_status.assert_called_once()
        warnings = capsys.readouterr().err
        assert warnings.count("is unavailable") == 3
        assert [entry["name"] for entry in runtime._mcp_unavailable] == [
            "offline",
            "missing",
            "pending",
        ]
    finally:
        await runtime.stop()
    assert runtime._mcp_unavailable == []
    mcp_lifecycle.shutdown_mcp_servers.assert_called_once_with()  # type: ignore[attr-defined]


def test_build_hermes_config_disables_session_title_generation_only_for_relay():
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

    assert "auxiliary" not in ordinary_config
    assert relay_config["auxiliary"] == {"title_generation": {"enabled": False}}


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


@pytest.mark.parametrize("kept", [False, True])
def test_hermes_home_is_the_agents_state_folder_when_fabric_keeps_state(
    tmp_path: Path, kept: bool
):
    """Select the shared state folder or the current runtime's home."""
    state_dir = tmp_path / "artifacts/.fabric/state/nvidia.fabric.hermes/assistant"
    context = _runtime_context(artifact_root="artifacts").to_mapping()
    if kept:
        context["state_dir"] = str(state_dir)

    home = adapter._hermes_home(RuntimeContext.from_mapping(context), str(tmp_path))

    assert home == (
        state_dir if kept else tmp_path / "artifacts/.fabric/hermes/runtimes/runtime-1"
    )


@pytest.mark.parametrize("kept", [False, True])
async def test_hermes_sessions_survive_runtime_replacement_only_with_kept_state(
    tmp_path: Path, api_server: str, kept: bool
):
    """Preserve native session records when runtimes share a state folder."""
    os.environ["TEST_API_KEY"] = "fabric-test-model-key"
    config = _agent_config(
        {
            "models": {
                "default": {
                    "provider": "openai",
                    "model": "fabric-echo",
                    "base_url": f"{api_server}/v1",
                    "api_key_env": "TEST_API_KEY",
                }
            },
            "tools": {"enabled": []},
        }
    )
    contexts = []
    for runtime_id in ("runtime-first", "runtime-second"):
        context = _runtime_context(
            runtime_id=runtime_id,
            workspace=str(tmp_path),
            artifact_root=str(tmp_path / "artifacts"),
        ).to_mapping()
        if kept:
            context["state_dir"] = str(tmp_path / "kept-state")
        contexts.append(RuntimeContext.from_mapping(context))
    first = adapter.HermesRuntime()
    try:
        await first.start(
            {
                "config": config,
                "runtime_context": contexts[0].to_mapping(),
                "base_dir": str(tmp_path),
            }
        )
        result = await first.invoke(
            AgentRunRequest.from_mapping({"input": "retained session marker"}),
            contexts[0],
        )
        assert result.status is AgentRunStatus.SUCCEEDED
        session_id = first._agent.session_id
        assert first._session_db.get_session(session_id) is not None
        saved_messages = first._session_db.get_messages(session_id)
        assert any(
            message["content"] == "retained session marker"
            for message in saved_messages
        )
    finally:
        await first.stop()

    second = adapter.HermesRuntime()
    try:
        await second.start(
            {
                "config": config,
                "runtime_context": contexts[1].to_mapping(),
                "base_dir": str(tmp_path),
            }
        )
        if kept:
            assert second._session_db.get_session(session_id) is not None
            assert second._session_db.get_messages(session_id) == saved_messages
        else:
            assert second._session_db.get_session(session_id) is None
        assert second._agent.session_id != session_id
        if not kept:
            assert second._agent.session_id == "runtime-second"
        assert second._conversation_history is None
    finally:
        await second.stop()


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


@pytest.mark.parametrize(
    ("model_max_tokens", "harness_max_tokens", "expected_max_tokens"),
    [
        (None, None, None),
        (None, 1024, 1024),
        (768, None, 768),
        (768, 1024, 768),
    ],
    ids=["native-default", "harness-only", "model-only", "model-over-harness"],
)
async def test_persistent_runtime_reuses_hermes_agent_session_and_history(
    monkeypatch,
    tmp_path: Path,
    model_max_tokens: int | None,
    harness_max_tokens: int | None,
    expected_max_tokens: int | None,
):
    """A persistent Hermes runtime applies model extensions and reuses state."""
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
            "harness": {
                "settings": {"max_tokens": harness_max_tokens}
                if harness_max_tokens is not None
                else {}
            },
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
                    **(
                        {"max_tokens": model_max_tokens}
                        if model_max_tokens is not None
                        else {}
                    ),
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
        save_trajectories=False,
        max_tokens=expected_max_tokens,
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
    assert Path(second.output["hermes_home"]) == (
        tmp_path
        / "artifacts"
        / ".fabric"
        / "hermes"
        / "runtimes"
        / "runtime-fabric-123"
    )


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
    await runtime.stop()


def test_main_serves_persistent_runtime(monkeypatch):
    serve = MagicMock()
    monkeypatch.setattr(adapter.lifecycle, "serve", serve)

    adapter.main()

    serve.assert_called_once_with(
        adapter.HermesRuntime,
        config_loader=AgentConfig.from_mapping,
    )


@pytest.fixture(name="mock_hermes_agent")
def mock_hermes_agent_fixture(monkeypatch) -> MagicMock:
    """Replace the embedded agent and session store; Hermes's skill code stays real."""
    import hermes_state
    import run_agent

    mock_ai_agent_type = MagicMock(spec=AIAgent, return_value=MagicMock(spec=AIAgent))
    monkeypatch.setattr(
        mock_ai_agent_type.__init__.__func__,
        "__signature__",
        inspect.signature(AIAgent.__init__),
        raising=False,
    )
    monkeypatch.setattr(run_agent, "AIAgent", mock_ai_agent_type)
    monkeypatch.setattr(hermes_state, "SessionDB", MagicMock(spec=SessionDB))
    monkeypatch.setattr(
        adapter,
        "_invoke_hermes_turn",
        MagicMock(return_value=({"final_response": "done", "completed": True}, "")),
    )
    os.environ["TEST_API_KEY"] = "secret"
    return mock_ai_agent_type


def _skill_payload(
    tmp_path: Path,
    skills: dict[str, object],
    *,
    runtime_id: str = "runtime-skills",
    kept: bool = False,
) -> dict[str, object]:
    """A start payload with ``skills``. With ``kept``, the runtime keeps the
    agent's state folder, as ``runtime.keep_state`` does, so the Hermes home
    outlives the runtime."""
    context = _runtime_context(
        runtime_id=runtime_id,
        workspace=str(tmp_path),
        artifact_root=str(tmp_path / "artifacts"),
    ).to_mapping()
    if kept:
        context["state_dir"] = str(
            tmp_path
            / "artifacts"
            / ".fabric"
            / "state"
            / "nvidia.fabric.hermes"
            / "agent"
        )
    return {
        "base_dir": str(tmp_path),
        "config": _agent_config(
            {
                "models": {
                    "default": {
                        "provider": "test",
                        "model": "test-model",
                        "api_key_env": "TEST_API_KEY",
                    }
                },
                "skills": skills,
            }
        ),
        "runtime_context": context,
    }


def _skills_home(tmp_path: Path) -> Path:
    """The Hermes home of a runtime started from :func:`_skill_payload`."""
    return tmp_path / "artifacts" / ".fabric" / "hermes" / "runtimes" / "runtime-skills"


async def _start_with_skills(
    tmp_path: Path, skills: dict[str, object]
) -> tuple[dict[str, object] | None, Path, dict[str, object]]:
    """Start a runtime, then return its skill report, home, and native config."""
    runtime = adapter.HermesRuntime()
    await runtime.start(_skill_payload(tmp_path, skills))
    try:
        assert runtime._hermes_home is not None
        return (
            runtime._skills_report,
            runtime._hermes_home,
            configuration.summarize_hermes_config(runtime._hermes_config),
        )
    finally:
        await runtime.stop()


async def _restarted_skills_report(payload: dict[str, object]) -> dict[str, object]:
    """Start a new runtime with the same config, invoke it once, and return
    the result's ``output.skills``. A new runtime ID gives it a new Hermes home."""
    context = _runtime_context(
        runtime_id="runtime-restarted",
        workspace=str(payload["base_dir"]),
        artifact_root=str(Path(str(payload["base_dir"])) / "artifacts"),
    )
    runtime = adapter.HermesRuntime()
    await runtime.start({**payload, "runtime_context": context.to_mapping()})
    try:
        result = await runtime.invoke(AgentRunRequest(input="hello"), context)
    finally:
        await runtime.stop()
    return result.output["skills"]


def _first_bundled_name() -> str:
    inventory = hermes_skills.bundled_inventory()
    return sorted(set(inventory) - hermes_skills.ESSENTIAL_SKILLS)[0]


def _home_skill_names(home: Path) -> set[str]:
    """Names of the skills in the Hermes home, as Hermes's sync reads them."""
    return {
        hermes_skills.read_skill_name(skill_md, skill_md.parent.name)
        for skill_md in (home / "skills").rglob("SKILL.md")
    }


def _bundled_place(name: str) -> Path:
    """Where Hermes's sync copies a bundled skill, below the home's skills folder."""
    source = hermes_skills.bundled_inventory()[name]
    return source.relative_to(hermes_skills.bundled_dir())


def _write_hermes_config(
    home: Path, external_dirs: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point the Hermes home ``home`` at ``external_dirs`` and make it current."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        common_utils.dump_yaml({"skills": {"external_dirs": external_dirs}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_loads_skill_directories_and_reports_on_every_result(
    tmp_path: Path, capsys
):
    team = tmp_path / "team"
    write_skill(team / "debugging")
    write_skill(team / "review")
    opt_review = write_skill(tmp_path / "opt" / "review")
    inventory = hermes_skills.bundled_inventory()
    excluded, kept = sorted(set(inventory) - hermes_skills.ESSENTIAL_SKILLS)[:2]
    runtime = adapter.HermesRuntime()

    await runtime.start(
        _skill_payload(
            tmp_path,
            {
                "directories": [{"folder": str(team)}, {"skill": str(opt_review)}],
                "bundled": {"exclude": [excluded]},
            },
        )
    )
    try:
        start_warnings = capsys.readouterr().err
        context = _runtime_context(runtime_id="runtime-skills")
        first = await runtime.invoke(AgentRunRequest(input="hello"), context)
        second = await runtime.invoke(AgentRunRequest(input="hello"), context)
        home = runtime._hermes_home
        native = (home / "config.yaml").read_text(encoding="utf-8")
    finally:
        await runtime.stop()

    assert first.output["skills"] == second.output["skills"]
    report = first.output["skills"]
    team = team.resolve()
    external = [item for item in report["loaded"] if item["origin"] == "external"]
    assert [skill["name"] for skill in external] == ["debugging"]
    # Every other skill Hermes loads follows: here, the bundled copies in
    # the home, each with its folder.
    others = report["loaded"][len(external) :]
    assert {item["origin"] for item in others} == {"bundled"}
    assert sorted(item["name"] for item in others) == sorted(
        report["bundled"]["selected"]
    )
    for item in others:
        assert item["path"] == str(
            (home / "skills" / _bundled_place(item["name"])).resolve()
        )
    # Hermes refuses two different same-tier skills named review.
    assert report["duplicates"][0] == {
        "name": "review",
        "outcome": "ambiguous",
        "copies": [
            skill_copy("external", team / "review", 0),
            skill_copy("external", opt_review, 1),
        ],
    }
    assert excluded not in report["bundled"]["selected"]
    # A kept skill loads, or is reported unavailable when Hermes hides it on
    # this platform (for example the macOS-only skills on Linux).
    unavailable = {item["name"] for item in report["bundled"].get("unavailable", [])}
    assert kept in report["bundled"]["selected"] or kept in unavailable
    assert "hermes-agent" in report["bundled"]["selected"]
    assert report["bundled"]["essential"] == ["hermes-agent"]
    # Only the selected bundled skills are copied into the home, so nothing
    # needs Hermes's skills.disabled.
    assert kept in _home_skill_names(home)
    assert excluded not in _home_skill_names(home)
    assert f"- {excluded}" not in native
    assert "skills_disabled" not in first.output["hermes_native_config"]
    assert first.output["hermes_native_config"]["skill_dirs"] == [
        str(team),
        str(opt_review),
    ]
    # Warnings are written once, at start.
    assert "skill 'review' has 2 copies, ambiguous" in start_warnings
    assert "review" not in capsys.readouterr().err


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_reports_a_home_skill_that_shadows_a_directory_skill(
    tmp_path: Path,
):
    """A skill in the Hermes home outranks skills.external_dirs."""
    team = tmp_path / "team"
    external = write_skill(team / "notes")
    local = write_skill(_skills_home(tmp_path) / "skills" / "notes")

    report, _home, _native = await _start_with_skills(
        tmp_path, {"directories": [{"folder": str(team)}], "duplicates": "reject"}
    )

    assert report == {
        "loaded": [other_skill("notes", "local", local)],
        "duplicates": [
            {
                "name": "notes",
                "outcome": "shadowed",
                "selected": str(local),
                "copies": [
                    skill_copy("local", local),
                    skill_copy("external", external, 0),
                ],
            }
        ],
    }


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_reports_a_bundled_copy_in_the_home_by_its_origin(
    tmp_path: Path,
):
    """A home that already holds Hermes's bundled copies, as a kept home or a
    native interface's own copy leaves it, outranks a directory skill. The
    report names the home's copy as bundled even without skills.bundled."""
    name = _first_bundled_name()
    # A first start copies the bundled skills into the runtime's home.
    await _start_with_skills(tmp_path, {"bundled": {}})
    team = tmp_path / "team"
    external = write_skill(team / name)
    bundled_copy = (_skills_home(tmp_path) / "skills" / _bundled_place(name)).resolve()

    report, _home, _native = await _start_with_skills(
        tmp_path, {"directories": [{"folder": str(team)}]}
    )

    assert report is not None
    [record] = [item for item in report["duplicates"] if item["name"] == name]
    assert record["copies"] == [
        skill_copy("bundled", bundled_copy),
        skill_copy("external", external, 0),
    ]
    assert record["outcome"] == "shadowed"


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_keeps_skill_paths_without_a_report(tmp_path: Path):
    skill = write_skill(tmp_path / "skills" / "review")
    runtime = adapter.HermesRuntime()

    await runtime.start(_skill_payload(tmp_path, {"paths": [str(skill)]}))
    try:
        result = await runtime.invoke(
            AgentRunRequest(input="hello"),
            _runtime_context(runtime_id="runtime-skills"),
        )
        home = runtime._hermes_home
    finally:
        await runtime.stop()

    assert "skills" not in result.output
    assert result.output["hermes_native_config"]["skill_dirs"] == [str(skill)]
    # skills.paths never copies bundled skills into the home.
    assert not (home / "skills").exists() or not any(
        (home / "skills").rglob("SKILL.md")
    )


@pytest.mark.parametrize(
    ("skills", "code", "field"),
    [
        (
            {"directories": [{"folder": "missing"}]},
            "hermes_skill_not_found",
            "skills.directories[0].folder",
        ),
        (
            {"bundled": {"include": ["no-such-bundled-skill"]}},
            "hermes_skill_invalid",
            "skills.bundled.include",
        ),
    ],
)
@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_start_fails_before_writing_config(
    tmp_path: Path, skills: dict[str, object], code: str, field: str
):
    runtime = adapter.HermesRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime.start(_skill_payload(tmp_path, skills))

    assert caught.value.code == code
    assert caught.value.metadata["field"] == field
    assert runtime._started is False
    assert not (_skills_home(tmp_path) / "config.yaml").exists()


@POSIX_PERMISSIONS
@pytest.mark.parametrize("kind", ["skill", "folder"])
@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_start_fails_on_an_entry_it_cannot_read(
    tmp_path: Path, kind: str
):
    """A ``skill`` entry whose SKILL.md can't be read, or a ``folder`` entry
    that can't be listed, fails start as invalid, naming the entry."""
    team = tmp_path / "team"
    review = write_skill(team / "review")
    # Unreadable SKILL.md; a folder Hermes can search but not list.
    locked, mode = (review / "SKILL.md", 0o000) if kind == "skill" else (team, 0o300)
    configured = review if kind == "skill" else team
    runtime = adapter.HermesRuntime()
    locked.chmod(mode)
    try:
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            await runtime.start(
                _skill_payload(tmp_path, {"directories": [{kind: str(configured)}]})
            )
    finally:
        locked.chmod(0o644 if kind == "skill" else 0o755)

    assert caught.value.code == "hermes_skill_invalid"
    assert caught.value.metadata == {
        "field": f"skills.directories[0].{kind}",
        "reason": "invalid",
        "path": str(locked),
    }
    assert runtime._started is False
    assert not (_skills_home(tmp_path) / "config.yaml").exists()


@pytest.mark.parametrize("bundled", [False, True])
async def test_bundled_copy_failure_fails_start_only_with_bundled_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bundled: bool
):
    """With an interface on and ``skills.bundled`` omitted, the adapter copies
    the bundled skills only for the report, so a failure is logged, as the
    interfaces' own copy logs it."""
    from nemo_fabric_adapter_contract.models import BundledSkills
    from nemo_fabric_adapter_contract.skills import DirectoryLoad

    def fail() -> None:
        raise OSError("disk full")

    monkeypatch.setattr(hermes_skills, "sync_bundled", fail)
    monkeypatch.setattr(hermes_skills, "load_inventory", lambda plan: None)
    monkeypatch.setattr(hermes_skills, "native_catalog", lambda: [])
    plan = hermes_skills.SkillPlan(
        load=DirectoryLoad(),
        bundled=BundledSkills.from_mapping({}) if bundled else None,
        sync=True,
    )

    if not bundled:
        assert await adapter._prepare_skills(plan, tmp_path) == {"loaded": []}
        return
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await adapter._prepare_skills(plan, tmp_path)
    assert caught.value.metadata == {"field": "skills.bundled", "reason": "sync_failed"}


def test_hermes_discovery_rules_match_the_pinned_loader():
    from agent import skill_utils

    assert hermes_skills.HERMES_EXCLUDED_DIRS == skill_utils.EXCLUDED_SKILL_DIRS
    assert hermes_skills.HERMES_SUPPORT_DIRS == skill_utils.SKILL_SUPPORT_DIRS
    assert hermes_skills.ESSENTIAL_SKILLS == skill_utils.ESSENTIAL_SKILLS
    assert (hermes_skills.TIER_LOCAL, hermes_skills.TIER_EXTERNAL) == (
        skill_utils.TIER_LOCAL,
        skill_utils.TIER_EXTERNAL,
    )


def test_reader_reads_skills_as_hermes_catalog_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The reader names, keeps, and refuses each skill as Hermes's own catalog
    does, for front matter only a YAML parser reads and names Hermes drops."""
    team = tmp_path.resolve() / "team"
    folders = {
        key: write_skill(team / key, text=f"---\n{front_matter}\n---\nBody\n")
        for key, front_matter in {
            "folded": "name: >-\n  notes",
            "alias": "label: &label aliased\nname: *label",
            "number": "name: 123",
            "empty": 'name: ""',
        }.items()
    }
    # Hermes drops every skill below a folder such as site-packages.
    vendored = tmp_path.resolve() / "site-packages"
    folders["excluded"] = write_skill(vendored / "tool")
    _write_hermes_config(tmp_path / "home", [str(team), str(vendored)], monkeypatch)

    infos = hermes_skills.hermes_reader()(list(folders.values()))
    catalog = {
        Path(row["path"]).parent: row["load_name"]
        for row in hermes_skills.native_catalog()
    }

    # The name each skill loads under, or None when Hermes does not load it.
    read = {
        key: (infos[folder].name or folder.name) if infos[folder].valid else None
        for key, folder in folders.items()
    }
    assert read == {key: catalog.get(folder) or None for key, folder in folders.items()}
    assert read == {
        "folded": "notes",
        "alias": "aliased",
        "number": None,
        "empty": None,
        "excluded": None,
    }
    # Hermes lists an empty name but cannot load it.
    assert catalog[folders["empty"]] == ""
    assert {key: infos[folders[key]].reason for key in read if not read[key]} == {
        "number": "not_loaded_by_hermes",
        "empty": "no_name",
        "excluded": "excluded_path",
    }


def test_descriptor_advertises_implemented_skill_support():
    descriptor_path = (
        Path(__file__).parents[2]
        / "adapters"
        / "python"
        / "hermes"
        / "hermes.fabric-adapter.json"
    )
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))

    skills = descriptor["skills"]
    assert skills["directory_loading"] == {
        "supported_entries": ["skill", "folder"],
        "skill_file": "SKILL.md",
        "discovery": {
            "traversal": "recursive",
            "nested_skills": "include",
            "depth_limit": {"kind": "no_fixed_limit"},
        },
        "parsing": {
            "required_frontmatter_fields": [],
            "name_fallback": "directory_name",
            "frontmatter_errors": "recover_key_value",
        },
    }
    # Hermes always loads hermes-agent, so the adapter can neither disable
    # every bundled skill nor honor an include list (not even an empty one).
    assert skills["bundled"] == {
        "default": "disabled",
        "supports_enable": True,
        "supports_disable": False,
        "supports_include": False,
        "supports_exclude": True,
    }
    assert skills["duplicates"]["supported_policies"] == [
        "reject",
        "first_wins",
        "last_wins",
    ]
    assert skills["duplicates"]["name_source"] == "frontmatter"


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_seeds_the_essential_skill_into_a_fresh_home(tmp_path: Path):
    """Hermes always loads hermes-agent; its sync puts it in the home, even
    when every other bundled skill is left out."""
    others = sorted(
        set(hermes_skills.bundled_inventory()) - hermes_skills.ESSENTIAL_SKILLS
    )
    report, home, native = await _start_with_skills(
        tmp_path, {"bundled": {"exclude": others}}
    )

    essential = (home / "skills" / _bundled_place("hermes-agent")).resolve()
    assert report == {
        "loaded": [other_skill("hermes-agent", "bundled", essential)],
        "bundled": {"selected": ["hermes-agent"], "essential": ["hermes-agent"]},
    }
    assert _home_skill_names(home) == {"hermes-agent"}
    assert "skills_disabled" not in native


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_reports_exactly_the_bundled_skills_left_in(tmp_path: Path):
    other = _first_bundled_name()
    excluded = sorted(
        set(hermes_skills.bundled_inventory())
        - hermes_skills.ESSENTIAL_SKILLS
        - {other}
    )

    report, home, native = await _start_with_skills(
        tmp_path, {"bundled": {"exclude": excluded}}
    )

    assert report is not None
    loaded = report["bundled"]["selected"]
    unavailable = [item["name"] for item in report["bundled"].get("unavailable", [])]
    # A selected skill loads, or is unavailable when Hermes hides it here.
    assert sorted([*loaded, *unavailable]) == sorted(["hermes-agent", other])
    assert "hermes-agent" in loaded
    assert [item["name"] for item in report["loaded"]] == loaded
    # Hermes's sync copies exactly the skills left in into the home.
    assert _home_skill_names(home) == {"hermes-agent", other}
    assert "skills_disabled" not in native


@pytest.mark.parametrize(
    ("bundled", "field"),
    [
        ({"include": []}, "skills.bundled.include"),
        ({"include": ["hermes-agent"]}, "skills.bundled.include"),
        ({"exclude": ["hermes-agent"]}, "skills.bundled.exclude"),
        ({"enabled": False}, "skills.bundled.enabled"),
    ],
    ids=["include-none", "include-essential", "exclude-essential", "disabled"],
)
@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_refuses_a_selection_without_the_essential_skill(
    tmp_path: Path, bundled: dict[str, object], field: str
):
    """Hermes always loads hermes-agent, so a selection must keep it, and no
    include list can mean what it says. Planning refuses include and
    enabled: false first; the adapter refuses them when planning is skipped."""
    runtime = adapter.HermesRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime.start(_skill_payload(tmp_path, {"bundled": bundled}))

    assert caught.value.code == "hermes_skill_invalid"
    assert caught.value.metadata == {
        "field": field,
        "reason": "essential_bundled_skill",
        "name": "hermes-agent",
    }
    assert runtime._started is False


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_keeps_bundled_names_of_skill_paths_enabled(tmp_path: Path):
    """skills.disabled would hide the skills.paths skill too."""
    from tools import skills_tool

    name = _first_bundled_name()
    mine = tmp_path / "mine"
    own = write_skill(mine / name)
    runtime = adapter.HermesRuntime()

    await runtime.start(
        _skill_payload(
            tmp_path,
            {"paths": [str(mine)], "bundled": {"exclude": [name]}},
        )
    )
    try:
        native = runtime._hermes_config["skills"]
        skills_tool.clear_skills_cache()
        loadable = {
            skill["name"]: skill
            for skill in skills_tool._skill_catalog()
            if skill["load_name"]
        }
    finally:
        await runtime.stop()

    assert "disabled" not in native
    assert Path(loadable[name]["path"]).parent.resolve() == own


@pytest.mark.parametrize("in_place", [True, False], ids=["bundled-place", "elsewhere"])
@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_loads_the_agents_own_home_skill_of_an_excluded_name(
    tmp_path: Path, capsys, in_place: bool
):
    """A left-out bundled skill never reaches the home, so the agent's own
    skill of that name loads wherever it is, and nothing is ambiguous."""
    from tools import skills_tool

    excluded = _first_bundled_name()
    place = _bundled_place(excluded)
    own_place = place if in_place else Path("mine") / excluded
    own = write_skill(_skills_home(tmp_path) / "skills" / own_place)
    runtime = adapter.HermesRuntime()

    await runtime.start(_skill_payload(tmp_path, {"bundled": {"exclude": [excluded]}}))
    try:
        report = runtime._skills_report
        home = runtime._hermes_home
        native = configuration.summarize_hermes_config(runtime._hermes_config)
        skills_tool.clear_skills_cache()
        rows = [
            row
            for row in skills_tool._skill_catalog(include_hidden=True)
            if row["name"] == excluded
        ]
    finally:
        await runtime.stop()

    assert report is not None
    assert "skills_disabled" not in native
    assert excluded not in report["bundled"]["selected"]
    found = (home / "skills").rglob(f"{excluded}/SKILL.md")
    assert [path.parent.resolve() for path in found] == [own]
    # Hermes resolves the name to the agent's skill alone and loads it.
    assert [(Path(row["path"]).parent.resolve(), row["status"]) for row in rows] == [
        (own, "unique")
    ]
    assert "duplicates" not in report
    warnings = capsys.readouterr().err
    assert excluded not in warnings


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_stages_only_the_selected_bundled_skills(tmp_path: Path):
    """Hermes's sync, and each interface's, reads a staged bundled folder."""
    from tools import skills_sync

    excluded = _first_bundled_name()
    real_dir = hermes_skills.bundled_dir()
    before = os.environ.get("HERMES_BUNDLED_SKILLS")
    runtime = adapter.HermesRuntime()

    await runtime.start(_skill_payload(tmp_path, {"bundled": {"exclude": [excluded]}}))
    try:
        stage = Path(os.environ["HERMES_BUNDLED_SKILLS"])
        # What Hermes's sync reads while the runtime runs.
        staged = dict(
            skills_sync._discover_bundled_skills(skills_sync._get_bundled_dir())
        )
    finally:
        await runtime.stop()

    assert stage != real_dir
    assert set(staged) == set(hermes_skills.bundled_inventory()) - {excluded}
    # The same layout as the installation's, so the home gets the same copies.
    for name, folder in staged.items():
        assert folder.relative_to(stage) == _bundled_place(name)
    # Stop removes the folder and restores the variable.
    assert not stage.exists()
    assert os.environ.get("HERMES_BUNDLED_SKILLS") == before
    assert hermes_skills.bundled_dir() == real_dir


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_passes_the_staged_folder_to_native_interfaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The API, dashboard, and terminal processes run Hermes's sync themselves."""
    from nemo_fabric_adapters.hermes import interfaces

    if not interfaces.POSIX:
        pytest.skip("Hermes interfaces require a POSIX host")
    excluded = _first_bundled_name()
    start = AsyncMock()
    monkeypatch.setattr(interfaces.InterfaceChildren, "start", start)
    payload = _skill_payload(tmp_path, {"bundled": {"exclude": [excluded]}})
    config = payload["config"].to_mapping()
    config["harness"] = {
        "settings": {"sdk": {"enabled": False}, "interfaces": {"api": {"port": 1}}}
    }
    payload["config"] = _agent_config(config)
    runtime = adapter.HermesRuntime()

    await runtime.start(payload)
    try:
        stage = os.environ["HERMES_BUNDLED_SKILLS"]
    finally:
        await runtime.stop()

    environment = start.call_args.kwargs["environment"]
    assert environment["HERMES_BUNDLED_SKILLS"] == stage
    child = interfaces._child_command(
        "api", 1, token="token", web_dist=None, environment=environment
    )[1]
    assert child["HERMES_BUNDLED_SKILLS"] == stage


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_removes_an_unchanged_excluded_copy_from_a_kept_home(
    tmp_path: Path,
):
    """A home's earlier, unchanged copy of a left-out skill is removed, stays
    out, and comes back when a later runtime selects the skill again."""
    excluded = _first_bundled_name()
    _report, home, _native = await _start_with_skills(tmp_path, {"bundled": {}})
    copy_path = home / "skills" / _bundled_place(excluded)
    assert copy_path.is_dir()
    manifest = home / "skills" / ".bundled_manifest"
    assert f"{excluded}:" in manifest.read_text(encoding="utf-8")

    for _ in range(2):
        report, home, _native = await _start_with_skills(
            tmp_path, {"bundled": {"exclude": [excluded]}}
        )
        assert report is not None
        assert not copy_path.exists()
        assert excluded not in _home_skill_names(home)
        assert "hermes-agent" in report["bundled"]["selected"]
        # Hermes's sync would read a kept entry as a deletion by the user.
        assert f"{excluded}:" not in manifest.read_text(encoding="utf-8")

    # Hermes's sync copies it again, rather than treat it as deleted.
    report, home, _native = await _start_with_skills(tmp_path, {"bundled": {}})
    assert copy_path.is_dir()
    assert f"{excluded}:" in manifest.read_text(encoding="utf-8")


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_refuses_a_changed_copy_of_an_excluded_skill(tmp_path: Path):
    """A changed copy is the user's or agent's work: never removed, and it
    would still load, so start fails."""
    excluded = _first_bundled_name()
    bundled = {"exclude": [excluded]}
    _report, home, _native = await _start_with_skills(tmp_path, {"bundled": {}})
    copy_path = home / "skills" / _bundled_place(excluded)
    skill_md = copy_path / "SKILL.md"
    edited = skill_md.read_text(encoding="utf-8") + "\nAn edit.\n"
    skill_md.write_text(edited, encoding="utf-8")
    runtime = adapter.HermesRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime.start(_skill_payload(tmp_path, {"bundled": bundled}))

    assert caught.value.code == "hermes_skill_invalid"
    assert caught.value.metadata == {
        "field": "skills.bundled.exclude",
        "reason": "excluded_bundled_skill_modified",
        "name": excluded,
        "path": str(copy_path),
    }
    assert str(copy_path) in str(caught.value)
    assert runtime._started is False
    assert skill_md.read_text(encoding="utf-8") == edited
    assert "HERMES_BUNDLED_SKILLS" not in os.environ


def test_remove_copies_drops_only_their_bundled_manifest_entries(tmp_path: Path):
    """A removed copy leaves the manifest, so a later sync copies it again."""
    root = tmp_path / "bundled"
    source = write_skill(root / "research" / "arxiv")
    home = tmp_path / "home"
    skills_dir = home / "skills"
    copy_path = write_skill(skills_dir / "research" / "arxiv")
    kept = write_skill(skills_dir / "devops" / "github")
    manifest = skills_dir / ".bundled_manifest"
    manifest.write_text("arxiv:aaa\ngithub:bbb\nhermes-agent:ccc\n", "utf-8")
    plan = hermes_skills.SkillPlan(
        inventory={"arxiv": source},
        bundled_root=root,
        stale_copies=[("arxiv", copy_path)],
    )

    hermes_skills.remove_copies(plan, home)

    assert not copy_path.exists()
    assert kept.is_dir()
    assert manifest.read_text("utf-8") == "github:bbb\nhermes-agent:ccc\n"


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_keeps_an_excluded_copy_changed_after_the_first_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A copy changed between the start check and its removal is kept."""
    excluded = _first_bundled_name()
    _report, home, _native = await _start_with_skills(tmp_path, {"bundled": {}})
    copy_path = home / "skills" / _bundled_place(excluded)
    skill_md = copy_path / "SKILL.md"
    edited = skill_md.read_text(encoding="utf-8") + "\nAn edit.\n"
    real_stage = hermes_skills.stage_bundled

    def stage_then_edit(plan: hermes_skills.SkillPlan, folder: Path) -> None:
        # Runs after excluded_copies and before remove_copies.
        real_stage(plan, folder)
        skill_md.write_text(edited, encoding="utf-8")

    monkeypatch.setattr(hermes_skills, "stage_bundled", stage_then_edit)
    runtime = adapter.HermesRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime.start(
            _skill_payload(tmp_path, {"bundled": {"exclude": [excluded]}})
        )

    assert caught.value.metadata["reason"] == "excluded_bundled_skill_modified"
    assert caught.value.metadata["path"] == str(copy_path)
    assert skill_md.read_text(encoding="utf-8") == edited


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_recovers_from_a_failure_after_removing_an_excluded_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A copy removed before a later removal fails leaves the manifest, so a
    later runtime that selects it again gets it back."""
    first, second = sorted(
        set(hermes_skills.bundled_inventory()) - hermes_skills.ESSENTIAL_SKILLS
    )[:2]
    _report, home, _native = await _start_with_skills(tmp_path, {"bundled": {}})
    first_copy = home / "skills" / _bundled_place(first)
    second_copy = home / "skills" / _bundled_place(second)
    manifest = home / "skills" / ".bundled_manifest"
    real_rmtree = hermes_skills.shutil.rmtree
    failing = [True]

    def fail_second(path: str | os.PathLike[str], *args: object, **kwargs: object):
        if failing[0] and Path(path) == second_copy.resolve():
            raise PermissionError(path)
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(hermes_skills.shutil, "rmtree", fail_second)
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await adapter.HermesRuntime().start(
            _skill_payload(tmp_path, {"bundled": {"exclude": [first, second]}})
        )
    failing[0] = False

    assert caught.value.metadata["reason"] == "excluded_bundled_skill_not_removed"
    assert not first_copy.exists()
    assert second_copy.is_dir()
    entries = manifest.read_text(encoding="utf-8")
    assert f"{first}:" not in entries
    assert f"{second}:" in entries

    await _start_with_skills(tmp_path, {"bundled": {"exclude": [first, second]}})
    assert not second_copy.exists()
    await _start_with_skills(tmp_path, {"bundled": {}})
    assert first_copy.is_dir()
    assert second_copy.is_dir()


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_copies_back_a_selected_skill_nested_in_a_removed_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Removing a left-out skill's folder takes a selected skill nested in it.
    Its manifest entry goes too, so Hermes's sync copies it back at once."""
    from tools import skills_sync

    bundled = tmp_path / "bundled"
    write_skill(bundled / "tools" / "parent", "parent")
    write_skill(bundled / "tools" / "parent" / "child", "child")
    monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(bundled))
    _report, home, _native = await _start_with_skills(tmp_path, {"bundled": {}})
    parent = home / "skills" / "tools" / "parent"
    child = parent / "child"
    manifest = home / "skills" / ".bundled_manifest"
    assert {"parent", "child"} <= set(hermes_skills.manifest_names(home / "skills"))
    assert (parent / "SKILL.md").is_file()

    report, home, _native = await _start_with_skills(
        tmp_path, {"bundled": {"exclude": ["parent"]}}
    )

    assert report is not None
    assert not (parent / "SKILL.md").exists()
    assert (child / "SKILL.md").is_file()
    assert skills_sync.dir_hash(child) == skills_sync.dir_hash(
        bundled / "tools/parent/child"
    )
    assert _home_skill_names(home) == {"child"}
    entries = manifest.read_text(encoding="utf-8")
    assert "child:" in entries
    assert "parent:" not in entries
    assert not (home / "skills" / hermes_skills.PENDING_MANIFEST).exists()


def _pending_record(home: Path) -> dict[str, list[str]]:
    """Each recorded name with its folders. Every folder has its files."""
    record = home / "skills" / hermes_skills.PENDING_MANIFEST
    data = json.loads(record.read_text(encoding="utf-8"))
    assert data["version"] == hermes_skills.PENDING_VERSION
    folders = {folder for paths in data["skills"].values() for folder in paths}
    assert set(data["files"]) == folders
    return data["skills"]


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_completes_a_manifest_update_that_failed_after_a_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The removal is recorded first, so the next start drops the entry that
    a failed start left, and a runtime that selects the skill gets it back."""
    from tools import skills_sync
    from tools import skills_tool

    excluded = _first_bundled_name()
    _report, home, _native = await _start_with_skills(tmp_path, {"bundled": {}})
    copy_path = home / "skills" / _bundled_place(excluded)
    manifest = home / "skills" / ".bundled_manifest"
    record = home / "skills" / hermes_skills.PENDING_MANIFEST
    real_drop = hermes_skills._drop_manifest_entries
    failing = [True]

    def fail_drop(skills_dir: Path, names: set[str]) -> None:
        if failing[0] and names:
            raise PermissionError(skills_dir / ".bundled_manifest")
        real_drop(skills_dir, names)

    monkeypatch.setattr(hermes_skills, "_drop_manifest_entries", fail_drop)
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await adapter.HermesRuntime().start(
            _skill_payload(tmp_path, {"bundled": {"exclude": [excluded]}})
        )
    failing[0] = False

    assert caught.value.metadata["reason"] == "excluded_bundled_skill_not_removed"
    assert not copy_path.exists()
    assert f"{excluded}:" in manifest.read_text(encoding="utf-8")
    assert _pending_record(home) == {excluded: [_bundled_place(excluded).as_posix()]}
    # Hermes doesn't read the record as a skill, or touch it.
    monkeypatch.setenv("HERMES_HOME", str(home))
    before = record.read_bytes()
    assert all(
        hermes_skills.PENDING_MANIFEST not in str(skill_md)
        for skill_md in skills_sync._iter_active_skill_mds()
    )
    with redirect_stdout(StringIO()):
        skills_tool.clear_skills_cache()
        rows = skills_tool._skill_catalog(skip_disabled=True, include_hidden=True)
        skills_tool.clear_skills_cache()
    assert all(hermes_skills.PENDING_MANIFEST not in str(row["path"]) for row in rows)
    assert record.read_bytes() == before

    await _start_with_skills(tmp_path, {"bundled": {}})

    assert copy_path.is_dir()
    assert f"{excluded}:" in manifest.read_text(encoding="utf-8")
    assert not record.exists()


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_completes_a_removal_that_failed_part_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A removal that fails after taking the copy's SKILL.md leaves the rest of
    the folder; the next start removes it, drops the entry, and copies the
    skill back when it is selected."""
    from tools import skills_sync

    excluded = _first_bundled_name()
    _report, home, _native = await _start_with_skills(tmp_path, {"bundled": {}})
    copy_path = home / "skills" / _bundled_place(excluded)
    manifest = home / "skills" / ".bundled_manifest"
    record = home / "skills" / hermes_skills.PENDING_MANIFEST
    real_rmtree = hermes_skills.shutil.rmtree
    failing = [True]

    def fail_part_way(path: str | os.PathLike[str], *args: object, **kwargs: object):
        if failing[0] and Path(path) == copy_path.resolve():
            (Path(path) / "SKILL.md").unlink()
            raise PermissionError(path)
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(hermes_skills.shutil, "rmtree", fail_part_way)
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await adapter.HermesRuntime().start(
            _skill_payload(tmp_path, {"bundled": {"exclude": [excluded]}})
        )
    failing[0] = False

    assert caught.value.metadata["reason"] == "excluded_bundled_skill_not_removed"
    assert copy_path.is_dir()
    assert not (copy_path / "SKILL.md").exists()
    assert f"{excluded}:" in manifest.read_text(encoding="utf-8")
    assert _pending_record(home) == {excluded: [_bundled_place(excluded).as_posix()]}

    await _start_with_skills(tmp_path, {"bundled": {}})

    source = hermes_skills.bundled_inventory()[excluded]
    assert skills_sync.dir_hash(copy_path) == skills_sync.dir_hash(source)
    assert f"{excluded}:" in manifest.read_text(encoding="utf-8")
    assert not record.exists()


async def _fail_a_removal_after_a_support_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Copy a bundled skill with support files into the home, then fail the
    start that removes it after it took a support file but not ``SKILL.md``.
    Return the copy."""
    bundled = tmp_path / "bundled"
    source = write_skill(bundled / "notes" / "notes", "notes")
    (source / "references").mkdir()
    (source / "references" / "guide.md").write_text("Guide\n", encoding="utf-8")
    (source / "scripts").mkdir()
    (source / "scripts" / "run.sh").write_text("echo run\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(bundled))
    _report, home, _native = await _start_with_skills(tmp_path, {"bundled": {}})
    copy_path = home / "skills" / "notes" / "notes"
    real_rmtree = hermes_skills.shutil.rmtree

    def fail_part_way(path: str | os.PathLike[str], *args: object, **kwargs: object):
        if Path(path) == copy_path.resolve():
            (Path(path) / "scripts" / "run.sh").unlink()
            raise PermissionError(path)
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(hermes_skills.shutil, "rmtree", fail_part_way)
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await adapter.HermesRuntime().start(
            _skill_payload(tmp_path, {"bundled": {"exclude": ["notes"]}})
        )
    monkeypatch.setattr(hermes_skills.shutil, "rmtree", real_rmtree)

    assert caught.value.metadata["reason"] == "excluded_bundled_skill_not_removed"
    assert (copy_path / "SKILL.md").is_file()
    assert not (copy_path / "scripts" / "run.sh").exists()
    manifest = home / "skills" / ".bundled_manifest"
    assert "notes:" in manifest.read_text(encoding="utf-8")
    assert _pending_record(home) == {"notes": ["notes/notes"]}
    return copy_path


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_completes_a_removal_that_kept_skill_md(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A removal that fails after taking a support file but not SKILL.md
    leaves only files it found unchanged; the next start removes them and
    drops the entry, so a runtime that selects the skill gets it back."""
    from tools import skills_sync

    copy_path = await _fail_a_removal_after_a_support_file(tmp_path, monkeypatch)
    home = _skills_home(tmp_path)
    manifest = home / "skills" / ".bundled_manifest"

    await _start_with_skills(tmp_path, {"bundled": {"exclude": ["notes"]}})

    assert not copy_path.exists()
    assert "notes:" not in manifest.read_text(encoding="utf-8")
    assert not (home / "skills" / hermes_skills.PENDING_MANIFEST).exists()

    await _start_with_skills(tmp_path, {"bundled": {}})

    assert skills_sync.dir_hash(copy_path) == skills_sync.dir_hash(
        tmp_path / "bundled" / "notes" / "notes"
    )


@pytest.mark.parametrize("edit", ["changed", "added"])
@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_keeps_a_part_removed_copy_edited_since(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edit: str
):
    """A file changed or added in a part-removed copy is the user's work: the
    next start keeps the copy and fails, as for a changed copy."""
    copy_path = await _fail_a_removal_after_a_support_file(tmp_path, monkeypatch)
    home = _skills_home(tmp_path)
    if edit == "changed":
        # The same size, so only the content hash tells.
        edited = copy_path / "references" / "guide.md"
        edited.write_text("GUIDE\n", encoding="utf-8")
    else:
        edited = copy_path / "scripts" / "new.sh"
        edited.write_text("echo new\n", encoding="utf-8")
    content = edited.read_bytes()
    runtime = adapter.HermesRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime.start(
            _skill_payload(tmp_path, {"bundled": {"exclude": ["notes"]}})
        )

    assert caught.value.code == "hermes_skill_invalid"
    assert caught.value.metadata == {
        "field": "skills.bundled.exclude",
        "reason": "excluded_bundled_skill_modified",
        "name": "notes",
        "path": str(copy_path),
    }
    assert str(copy_path) in str(caught.value)
    assert runtime._started is False
    assert (copy_path / "SKILL.md").is_file()
    assert edited.read_bytes() == content
    assert "notes:" in (home / "skills" / ".bundled_manifest").read_text("utf-8")
    assert _pending_record(home) == {"notes": ["notes/notes"]}


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_completes_a_part_removed_copy_with_runtime_cache_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Running a skill's scripts writes cache files that Hermes's dir_hash
    skips. They are left out of the record and skipped when a later start
    checks a part-removed copy, so they never block the removal."""
    from tools import skills_sync

    bundled = tmp_path / "bundled"
    source = write_skill(bundled / "notes" / "notes", "notes")
    (source / "scripts").mkdir()
    (source / "scripts" / "tool.py").write_text("print('tool')\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(bundled))
    _report, home, _native = await _start_with_skills(tmp_path, {"bundled": {}})
    copy_path = home / "skills" / "notes" / "notes"
    # Bytecode beside its source, there when the removal records the copy.
    (copy_path / "scripts" / "tool.pyc").write_bytes(b"bytecode")
    assert skills_sync.dir_hash(copy_path) == skills_sync.dir_hash(source)
    real_rmtree = hermes_skills.shutil.rmtree

    def fail_part_way(path: str | os.PathLike[str], *args: object, **kwargs: object):
        if Path(path) == copy_path.resolve():
            # The source goes first and leaves its bytecode behind.
            (Path(path) / "scripts" / "tool.py").unlink()
            raise PermissionError(path)
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(hermes_skills.shutil, "rmtree", fail_part_way)
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await adapter.HermesRuntime().start(
            _skill_payload(tmp_path, {"bundled": {"exclude": ["notes"]}})
        )
    monkeypatch.setattr(hermes_skills.shutil, "rmtree", real_rmtree)

    assert caught.value.metadata["reason"] == "excluded_bundled_skill_not_removed"
    assert _pending_record(home) == {"notes": ["notes/notes"]}
    data = json.loads(
        (home / "skills" / hermes_skills.PENDING_MANIFEST).read_text("utf-8")
    )
    assert sorted(data["files"]["notes/notes"]) == ["SKILL.md", "scripts/tool.py"]
    # Caches written after the failed start.
    for cache in ("__pycache__", "scripts/__pycache__", ".pytest_cache/v"):
        (copy_path / cache).mkdir(parents=True)
        (copy_path / cache / "cached.pyc").write_bytes(b"cache")

    await _start_with_skills(tmp_path, {"bundled": {"exclude": ["notes"]}})

    assert not copy_path.exists()
    manifest = home / "skills" / ".bundled_manifest"
    assert "notes:" not in manifest.read_text(encoding="utf-8")
    assert not (home / "skills" / hermes_skills.PENDING_MANIFEST).exists()


def test_finish_pending_removals_checks_files_hermes_hashes(tmp_path: Path):
    """A file that Hermes's dir_hash hashes still blocks the removal when it
    is new, even beside cache files, and a source-less .pyc is such a file."""
    skills_dir = tmp_path / "home" / "skills"
    partial = skills_dir / "notes"
    (partial / "__pycache__").mkdir(parents=True)
    (partial / "__pycache__" / "tool.pyc").write_bytes(b"cache")
    (partial / "shipped.pyc").write_bytes(b"not a cache")
    (skills_dir / ".bundled_manifest").write_text("notes:aaa\n", "utf-8")
    record = skills_dir / hermes_skills.PENDING_MANIFEST
    record.write_text(
        json.dumps(
            {
                "version": hermes_skills.PENDING_VERSION,
                "field": "skills.bundled.exclude",
                "skills": {"notes": ["notes"]},
                "files": {"notes": {}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(hermes_skills.lifecycle.LifecycleError) as caught:
        hermes_skills.finish_pending_removals(tmp_path / "home")

    assert caught.value.metadata["reason"] == "excluded_bundled_skill_modified"
    assert (partial / "shipped.pyc").is_file()

    (partial / "shipped.pyc").unlink()
    hermes_skills.finish_pending_removals(tmp_path / "home")

    assert not partial.exists()
    assert (skills_dir / ".bundled_manifest").read_text("utf-8") == ""
    assert not record.exists()


def test_finish_pending_removals_reads_a_record_without_files(tmp_path: Path):
    """A record from before files were recorded keeps its rule: only a folder
    without SKILL.md is removed, and a copy still there stays recorded."""
    skills_dir = tmp_path / "home" / "skills"
    partial = skills_dir / "research" / "arxiv"
    (partial / "references").mkdir(parents=True)
    (partial / "references" / "notes.md").write_text("Edited\n", encoding="utf-8")
    whole = write_skill(skills_dir / "devops" / "github")
    (skills_dir / ".bundled_manifest").write_text("arxiv:aaa\ngithub:bbb\n", "utf-8")
    record = skills_dir / hermes_skills.PENDING_MANIFEST
    record.write_text(
        json.dumps({"arxiv": ["research/arxiv"], "github": ["devops/github"]}),
        encoding="utf-8",
    )

    hermes_skills.finish_pending_removals(tmp_path / "home")

    assert not partial.exists()
    assert (whole / "SKILL.md").is_file()
    assert (skills_dir / ".bundled_manifest").read_text("utf-8") == "github:bbb\n"
    assert json.loads(record.read_text(encoding="utf-8"))["skills"] == {
        "github": ["devops/github"]
    }


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_cancelled_start_waits_for_the_bundled_staging_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Stop's removal of the staged folder is final: no copy lands after it."""
    excluded = _first_bundled_name()
    real_stage = hermes_skills.stage_bundled
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    stages: list[Path] = []

    def slow_stage(plan: hermes_skills.SkillPlan, folder: Path) -> None:
        stages.append(folder)
        entered.set()
        release.wait(5)
        try:
            real_stage(plan, folder)
        finally:
            finished.set()

    monkeypatch.setattr(hermes_skills, "stage_bundled", slow_stage)
    runtime = adapter.HermesRuntime()
    start = asyncio.create_task(
        runtime.start(_skill_payload(tmp_path, {"bundled": {"exclude": [excluded]}}))
    )
    assert await asyncio.to_thread(entered.wait, 5)

    start.cancel()
    asyncio.get_running_loop().call_later(0.2, release.set)
    with pytest.raises(asyncio.CancelledError):
        await start
    assert await asyncio.to_thread(finished.wait, 5)

    assert runtime._started is False
    assert stages and not stages[0].exists()
    assert "HERMES_BUNDLED_SKILLS" not in os.environ


@pytest.mark.parametrize(
    "skills", [{"bundled": {}}, {"bundled": {"exclude": []}}, {}], ids=str
)
@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_keeps_stock_bundled_behavior_without_exclusions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, skills: dict[str, object]
):
    """Without a left-out skill, nothing is staged or removed."""
    name = _first_bundled_name()
    _report, home, _native = await _start_with_skills(tmp_path, {"bundled": {}})
    copy_path = home / "skills" / _bundled_place(name)
    stage = MagicMock(wraps=hermes_skills.stage_bundled)
    monkeypatch.setattr(hermes_skills, "stage_bundled", stage)
    if not skills:
        team = tmp_path / "team"
        write_skill(team / "notes")
        skills = {"directories": [{"folder": str(team)}]}
    runtime = adapter.HermesRuntime()

    await runtime.start(_skill_payload(tmp_path, skills))
    try:
        assert "HERMES_BUNDLED_SKILLS" not in os.environ
    finally:
        await runtime.stop()

    stage.assert_not_called()
    assert copy_path.is_dir()
    assert set(hermes_skills.bundled_inventory()) <= _home_skill_names(home)


@pytest.mark.parametrize(
    ("entries", "debug_entry"),
    [
        ([{"folder": "team"}, {"skill": "team/review"}], 0),
        ([{"skill": "team/review"}, {"folder": "team"}], 1),
    ],
    ids=["skill-after", "skill-before"],
)
@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_passes_overlapping_entries_once(
    tmp_path: Path, entries: list[dict[str, str]], debug_entry: int
):
    """A skill Hermes reaches through two roots would be refused as ambiguous."""
    review = write_skill(tmp_path / "team" / "review")
    debug = write_skill(tmp_path / "team" / "debug")
    skills = {
        "directories": [
            {kind: str(tmp_path / path) for kind, path in entry.items()}
            for entry in entries
        ]
    }

    report, _home, native = await _start_with_skills(tmp_path, skills)

    assert native["skill_dirs"] == [str(tmp_path.resolve() / "team")]
    loaded = [
        loaded_skill("debug", debug, debug_entry),
        loaded_skill("review", review, 0),
    ]
    assert report == {"loaded": loaded if debug_entry == 0 else loaded[::-1]}


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_reports_skills_hermes_skips_or_refuses(tmp_path: Path):
    """The checks skip what Hermes drops; Hermes's catalog decides ambiguity."""
    team = tmp_path / "team"
    alpha = write_skill(team / "foo", name="alpha")
    # Hermes refuses a name that is also another skill's folder name.
    clash = write_skill(team / "bar", name="foo")
    # Hermes drops a non-string name, and cannot load an empty one.
    number = write_skill(team / "number", text="---\nname: 123\n---\nBody\n")
    empty = write_skill(team / "quoted", text='---\nname: ""\n---\nBody\n')

    report, _home, _native = await _start_with_skills(
        tmp_path, {"directories": [{"folder": str(team)}]}
    )

    assert report == {
        "loaded": [loaded_skill("alpha", alpha)],
        "skipped": [
            skipped_skill(number, "not_loaded_by_hermes"),
            skipped_skill(empty, "no_name"),
            skipped_skill(clash, "ambiguous"),
        ],
    }


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_ignores_skills_in_support_folders(tmp_path: Path):
    """Hermes never scans a skill's scripts folder, so its copy is no duplicate."""
    team = tmp_path / "team"
    first = write_skill(team / "a", name="dup")
    second = write_skill(team / "b")
    write_skill(team / "b" / "scripts" / "x", name="dup")

    report, _home, native = await _start_with_skills(
        tmp_path,
        {"directories": [{"folder": str(team)}], "duplicates": "first_wins"},
    )

    assert native["skill_dirs"] == [str(first), str(second)]
    assert report == {"loaded": [loaded_skill("dup", first), loaded_skill("b", second)]}


@POSIX_PERMISSIONS
@pytest.mark.parametrize("kind", ["folder", "skill"], ids=["folder", "chosen-skills"])
@pytest.mark.usefixtures("mock_hermes_agent")
async def test_read_only_skills_load_and_cannot_be_changed(
    tmp_path: Path, make_read_only, kind: str
):
    """A read-only folder, or read-only chosen skills, load through Hermes as they are."""
    team = tmp_path / "team"
    debugging = write_skill(team / "debugging")
    review = write_skill(team / "review")
    make_read_only(team)
    entries = (
        [{"folder": str(team)}]
        if kind == "folder"
        else [{"skill": str(debugging)}, {"skill": str(review)}]
    )

    report, _home, native = await _start_with_skills(tmp_path, {"directories": entries})

    assert report is not None
    assert [(skill["name"], skill["path"]) for skill in report["loaded"]] == [
        ("debugging", str(debugging)),
        ("review", str(review)),
    ]
    assert native["skill_dirs"] == [entry[kind] for entry in entries]
    with pytest.raises(PermissionError):
        (review / "SKILL.md").write_text("changed", encoding="utf-8")
    with pytest.raises(PermissionError):
        write_skill(team / "new-skill")


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_skill_written_into_a_writable_folder_survives_a_restart(
    tmp_path: Path,
):
    agent_skills = tmp_path / "agent-skills"
    agent_skills.mkdir()
    payload = _skill_payload(tmp_path, {"directories": [{"folder": str(agent_skills)}]})
    runtime = adapter.HermesRuntime()

    await runtime.start(payload)
    try:
        first = runtime._skills_report
        # The agent saves a new skill into its read-write folder.
        written = write_skill(agent_skills / "triage")
    finally:
        await runtime.stop()
    restarted = await _restarted_skills_report(payload)

    # Hermes's catalog was read, so the report says that no skill loads.
    assert first == {"loaded": [], "empty_entries": ["skills.directories[0]"]}
    assert restarted == {"loaded": [loaded_skill("triage", written)]}


@POSIX_PERMISSIONS
@pytest.mark.usefixtures("mock_hermes_agent")
async def test_agent_adds_skills_beside_read_only_skills_it_cannot_change(
    tmp_path: Path, make_read_only
):
    team = tmp_path / "team"
    protected = write_skill(team / "review")
    original = (protected / "SKILL.md").read_text(encoding="utf-8")
    make_read_only(team)
    agent_skills = tmp_path / "agent-skills"
    agent_skills.mkdir()
    payload = _skill_payload(
        tmp_path,
        {"directories": [{"folder": str(team)}, {"folder": str(agent_skills)}]},
    )
    runtime = adapter.HermesRuntime()

    await runtime.start(payload)
    try:
        with pytest.raises(PermissionError):
            (protected / "SKILL.md").write_text("changed", encoding="utf-8")
        with pytest.raises(PermissionError):
            write_skill(team / "triage")
        written = write_skill(agent_skills / "triage")
    finally:
        await runtime.stop()
    restarted = await _restarted_skills_report(payload)

    assert (protected / "SKILL.md").read_text(encoding="utf-8") == original
    assert restarted == {
        "loaded": [
            loaded_skill("review", protected, 0),
            loaded_skill("triage", written, 1),
        ]
    }


_AGENT_SKILL = (
    "---\nname: {name}\ndescription: Use when triaging a failed build. Read the "
    "logs first.\n---\n\n# Triage\n\nRead the newest log before changing code.\n"
)


def _skill_manage(**arguments: object) -> dict[str, object]:
    """Call Hermes's own skill_manage tool through Hermes's tool registry, as
    the agent's tool call reaches it, against the current Hermes home."""
    from agent import skill_utils
    from tools import skill_manager_tool  # noqa: F401  # registers skill_manage
    from tools import skills_tool
    from tools.registry import registry

    skill_utils._external_dirs_cache_clear()
    skills_tool.clear_skills_cache()
    return json.loads(registry.dispatch("skill_manage", dict(arguments)))


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_skill_created_by_hermes_survives_a_restart_with_kept_state(
    tmp_path: Path,
):
    """Hermes's skill_manage tool saves a new skill in the Hermes home. With
    kept state, a new runtime for the same agent loads it, and the report
    lists it when it shadows a configured skill of the same name."""
    from tools import skills_tool

    first = adapter.HermesRuntime()
    await first.start(
        _skill_payload(
            tmp_path, {"directories": []}, runtime_id="runtime-first", kept=True
        )
    )
    try:
        home = first._hermes_home
        assert home is not None
        created = _skill_manage(
            action="create", name="triage", content=_AGENT_SKILL.format(name="triage")
        )
    finally:
        await first.stop()
    assert created["success"] is True, created
    written = (home / "skills" / "triage").resolve()
    assert Path(str(created["skill_md"])).resolve() == written / "SKILL.md"

    # A configured skill of the same name appears before the next start.
    team = tmp_path / "team"
    configured = write_skill(team / "triage")
    second = adapter.HermesRuntime()
    await second.start(
        _skill_payload(
            tmp_path,
            {"directories": [{"folder": str(team)}]},
            runtime_id="runtime-second",
            kept=True,
        )
    )
    try:
        assert second._hermes_home == home
        report = second._skills_report
        skills_tool.clear_skills_cache()
        loadable = {
            skill["name"]: Path(skill["path"]).parent.resolve()
            for skill in skills_tool._skill_catalog()
            if skill["load_name"]
        }
        viewed = json.loads(skills_tool.skill_view("triage"))
    finally:
        await second.stop()

    assert loadable["triage"] == written
    assert viewed["success"] is True, viewed
    assert "Read the newest log" in viewed["content"]
    assert report == {
        "loaded": [other_skill("triage", "local", written)],
        "duplicates": [
            {
                "name": "triage",
                "outcome": "shadowed",
                "selected": str(written),
                "copies": [
                    skill_copy("local", written),
                    skill_copy("external", configured, 0),
                ],
            }
        ],
    }


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_report_lists_the_agents_own_skills_after_a_kept_state_restart(
    tmp_path: Path,
):
    """With skills.directories: [] the report still lists what Hermes loads:
    a skill the agent created with Hermes's skill_manage tool appears as
    local, with its folder, once a new runtime for the same agent starts.
    With skills.bundled the bundled copies are listed beside it."""
    first = adapter.HermesRuntime()
    await first.start(
        _skill_payload(
            tmp_path, {"directories": []}, runtime_id="runtime-first", kept=True
        )
    )
    try:
        home = first._hermes_home
        assert home is not None
        # Hermes loads nothing yet, and the report says so.
        assert first._skills_report == {"loaded": []}
        created = _skill_manage(
            action="create", name="triage", content=_AGENT_SKILL.format(name="triage")
        )
    finally:
        await first.stop()
    assert created["success"] is True, created
    written = (home / "skills" / "triage").resolve()

    second = adapter.HermesRuntime()
    await second.start(
        _skill_payload(
            tmp_path, {"directories": []}, runtime_id="runtime-second", kept=True
        )
    )
    try:
        assert second._hermes_home == home
        report = second._skills_report
        output = second._skills_output()
    finally:
        await second.stop()

    assert report == {"loaded": [other_skill("triage", "local", written)]}
    # Each run result carries the report.
    assert output == {"skills": report}

    others = sorted(
        set(hermes_skills.bundled_inventory()) - hermes_skills.ESSENTIAL_SKILLS
    )
    third = adapter.HermesRuntime()
    await third.start(
        _skill_payload(
            tmp_path,
            {"bundled": {"exclude": others}},
            runtime_id="runtime-third",
            kept=True,
        )
    )
    try:
        report = third._skills_report
    finally:
        await third.stop()

    essential = (home / "skills" / _bundled_place("hermes-agent")).resolve()
    assert report is not None
    assert report["loaded"] == [
        other_skill("hermes-agent", "bundled", essential),
        other_skill("triage", "local", written),
    ]


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_hermes_creates_skills_in_its_home_not_in_a_writable_folder(
    tmp_path: Path,
):
    """Hermes edits a skill in a writable folder entry in place, but saves a
    new skill in the Hermes home: the adapter never sets skills.create_dir."""
    agent_skills = tmp_path / "agent-skills"
    review = write_skill(agent_skills / "review", name="review")
    payload = _skill_payload(
        tmp_path,
        {"directories": [{"folder": str(agent_skills)}]},
        runtime_id="runtime-first",
        kept=True,
    )
    runtime = adapter.HermesRuntime()
    await runtime.start(payload)
    try:
        home = runtime._hermes_home
        assert home is not None
        patched = _skill_manage(
            action="patch", name="review", old_string="Body", new_string="Patched body"
        )
        created = _skill_manage(
            action="create", name="triage", content=_AGENT_SKILL.format(name="triage")
        )
    finally:
        await runtime.stop()

    assert patched["success"] is True, patched
    assert "Patched body" in (review / "SKILL.md").read_text(encoding="utf-8")
    assert created["success"] is True, created
    assert (home / "skills" / "triage" / "SKILL.md").is_file()
    assert not (agent_skills / "triage").exists()
    restarted = await _restarted_skills_report(payload)
    assert restarted == {"loaded": [loaded_skill("review", review)]}


@POSIX_PERMISSIONS
@pytest.mark.parametrize(
    "arguments",
    [
        {"action": "patch", "old_string": "Body", "new_string": "Changed"},
        {"action": "edit", "content": _AGENT_SKILL.format(name="review")},
        {
            "action": "write_file",
            "file_path": "references/notes.md",
            "file_content": "notes",
        },
    ],
    ids=["patch", "edit", "write-file"],
)
@pytest.mark.usefixtures("mock_hermes_agent")
async def test_hermes_cannot_change_a_read_only_skill_entry(
    tmp_path: Path, make_read_only, arguments: dict[str, str]
):
    """Hermes's skill_manage tool fails on a read-only mounted skill and leaves
    it as it was."""
    review = write_skill(tmp_path / "mounted" / "review", name="review")
    original = (review / "SKILL.md").read_text(encoding="utf-8")
    make_read_only(review)
    runtime = adapter.HermesRuntime()
    await runtime.start(
        _skill_payload(
            tmp_path,
            {"directories": [{"skill": str(review)}]},
            runtime_id="runtime",
            kept=True,
        )
    )
    try:
        result = _skill_manage(name="review", **arguments)
        report = runtime._skills_report
    finally:
        await runtime.stop()

    # The tool reports the file system's refusal to the agent as its result.
    assert result.get("success") is not True, result
    assert "Permission denied" in str(result["error"]), result
    assert (review / "SKILL.md").read_text(encoding="utf-8") == original
    assert sorted(path.name for path in review.iterdir()) == ["SKILL.md"]
    assert report == {"loaded": [loaded_skill("review", review)]}


def test_external_skill_names_match_hermes_sync_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import skills_sync

    home = tmp_path / "home"
    write_skill(tmp_path / "mine" / "github")
    write_skill(tmp_path / "mine" / "renamed", name="'pdf'")
    write_skill(tmp_path / "mine" / "github" / "scripts" / "hidden")
    write_skill(tmp_path / "mine" / "node_modules" / "vendored")
    write_skill(tmp_path / "mine" / "scripts" / "tool")
    write_skill(home / "relative" / "nested")
    roots = [str(tmp_path / "mine"), "relative", str(tmp_path / "missing")]
    _write_hermes_config(home, roots, monkeypatch)

    expected = skills_sync._build_external_skill_index()

    assert expected == {"github", "renamed", "pdf", "tool", "nested"}
    assert hermes_skills.external_skill_names(roots, home) == expected


async def test_sdk_disabled_result_reports_skills():
    runtime = adapter.HermesRuntime()
    runtime._started = True
    runtime._sdk_enabled = False
    runtime._skills_report = {"loaded": [{"name": "review"}]}
    request, context = _invocation(
        {
            "runtime_context": _runtime_context().to_mapping(),
            "request": {"input": "hello"},
        }
    )

    first = await runtime.invoke(request, context)
    first.output["skills"]["loaded"].clear()
    second = await runtime.invoke(request, context)

    assert first.error is not None
    assert first.error.code == "hermes_sdk_disabled"
    assert second.output == {"skills": {"loaded": [{"name": "review"}]}}


@pytest.fixture(name="retained_hermes_runtime")
def retained_hermes_runtime_fixture(started_hermes_runtime, tmp_path):
    from nemo_fabric_adapters.hermes.sessions import NativeSession

    runtime, turn = started_hermes_runtime
    database = MagicMock()
    database.get_messages.return_value = [{"id": 1, "role": "assistant"}]
    native = NativeSession("original", "original", database, tmp_path / "turn.json")
    runtime._native_session = native
    runtime._agent.session_id = "original"
    return runtime, turn, native


@pytest.mark.parametrize("failure", ["partial", "unfinished", "exception"])
async def test_retained_runtime_requires_recovery_after_an_uncertain_turn(
    retained_hermes_runtime, failure
):
    runtime, turn, native = retained_hermes_runtime
    context = _runtime_context(runtime_id=runtime._runtime_id)
    turn.return_value = (
        {"final_response": "unfinished", "partial": failure == "partial"},
        "",
    )
    if failure == "unfinished":
        native.database.get_messages.return_value = [{"id": 1, "role": "user"}]
    if failure == "exception":
        turn.side_effect = RuntimeError("native turn failed")
        with pytest.raises(RuntimeError):
            await runtime.invoke(AgentRunRequest(input="Create a ticket."), context)
    else:
        await runtime.invoke(AgentRunRequest(input="Create a ticket."), context)
    assert native.recovery_required
    marker = native.marker.read_bytes()
    turn.reset_mock()
    blocked = await runtime.invoke(AgentRunRequest(input="Do more work."), context)
    assert blocked.error.code == "session_recovery_required"
    turn.assert_not_called()
    assert native.marker.read_bytes() == marker

    turn.side_effect = None
    turn.return_value = ({"final_response": "completed"}, "")
    native.database.get_messages.return_value = [{"id": 2, "role": "assistant"}]
    result = await runtime.invoke(
        AgentRunRequest(input="I inspected the state.", acknowledge_recovery=True),
        context,
    )
    assert result.status == AgentRunStatus.SUCCEEDED
    assert not native.recovery_required


async def test_cancelled_turn_requires_recovery_only_after_worker_finishes(
    retained_hermes_runtime,
):
    runtime, turn, native = retained_hermes_runtime
    entered, release = threading.Event(), threading.Event()

    def partial_turn(**kwargs):
        entered.set()
        assert release.wait(5)
        return {"final_response": "unfinished", "partial": True}, ""

    turn.side_effect = partial_turn
    context = _runtime_context(runtime_id=runtime._runtime_id)
    invocation = asyncio.create_task(
        runtime.invoke(AgentRunRequest(input="Create a ticket."), context)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        worker = runtime._active_invoke_task
        invocation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await invocation
        assert not native.recovery_required
    finally:
        release.set()
    await asyncio.wait_for(asyncio.shield(worker), 2)
    assert native.recovery_required
