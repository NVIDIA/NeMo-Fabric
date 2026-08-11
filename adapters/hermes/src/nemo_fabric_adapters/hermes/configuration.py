# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate Fabric's typed configuration into Hermes-native settings."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentMcpServerConfig
from nemo_fabric_adapter_contract.models import AgentModelConfig
from nemo_fabric_adapters.common import mcp_auth
import nemo_fabric_adapters.common.utils as common_utils


PROVIDER_DEFAULT_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def _settings(config: AgentConfig) -> dict[str, Any]:
    return config.harness.settings if config.harness is not None else {}


def _selected_model(config: AgentConfig) -> AgentModelConfig:
    model = config.models.get("default")
    if model is None and len(config.models) == 1:
        model = next(iter(config.models.values()))
    if model is None:
        raise ValueError("Hermes requires a default model or exactly one model")
    return model


def _max_turns(config: AgentConfig) -> int | None:
    return config.runtime.max_turns if config.runtime is not None else None


def _api_key_env(model_config: AgentModelConfig) -> str:
    explicit = model_config.api_key_env
    if isinstance(explicit, str) and explicit:
        return explicit
    provider = str(model_config.provider or "").lower()
    default = PROVIDER_DEFAULT_API_KEY_ENV.get(provider)
    if default is None:
        raise ValueError(
            f"selected model api_key_env is required for provider {provider!r}"
        )
    return default


def disabled_toolsets(config: AgentConfig) -> list[str]:
    return config.tools.blocked if config.tools is not None else []


def build_hermes_config(
    agent_config: AgentConfig,
    *,
    workspace: str,
    relay_enabled: bool = False,
) -> dict[str, Any]:
    settings = _settings(agent_config)
    model_config = _selected_model(agent_config)
    blocked_toolsets = disabled_toolsets(agent_config)
    enabled_toolsets = (
        agent_config.tools.enabled if agent_config.tools is not None else None
    )

    config: dict[str, Any] = {
        "model": common_utils.without_none(
            {
                "provider": model_config.provider,
                "default": model_config.model,
                "base_url": model_config.base_url,
            }
        ),
        "agent": common_utils.without_none(
            {
                "max_turns": _max_turns(agent_config),
                "disabled_toolsets": blocked_toolsets or None,
            }
        ),
        "terminal": common_utils.without_none(
            {
                "backend": "local",
                "cwd": workspace,
                "timeout": settings.get("terminal_timeout", 60),
            }
        ),
    }

    skill_dirs = (
        [str(path) for path in agent_config.skills.paths]
        if agent_config.skills is not None
        else []
    )
    if skill_dirs:
        config["skills"] = {"external_dirs": skill_dirs}

    mcp_servers = agent_config.mcp.servers if agent_config.mcp is not None else {}
    if mcp_servers:
        config["mcp_servers"] = {
            name: hermes_mcp_server_config(server, name=name)
            for name, server in sorted(mcp_servers.items())
        }

    if enabled_toolsets is not None:
        config["platform_toolsets"] = {"cli": enabled_toolsets}

    plugins = common_utils.normalize_list(settings.get("plugins_enabled"))
    if relay_enabled and "observability/nemo_relay" not in plugins:
        plugins.append("observability/nemo_relay")
    if plugins:
        config["plugins"] = {"enabled": plugins}

    return config


def write_hermes_config(
    agent_config: AgentConfig,
    hermes_home: Path,
    *,
    workspace: str,
    relay_enabled: bool = False,
) -> tuple[Path, dict[str, Any]]:
    hermes_home.mkdir(parents=True, exist_ok=True)
    config = build_hermes_config(
        agent_config,
        workspace=workspace,
        relay_enabled=relay_enabled,
    )
    config_path = hermes_home / "config.yaml"
    config_path.write_text(common_utils.dump_yaml(config), encoding="utf-8")
    return config_path, config


def hermes_mcp_server_config(
    server: AgentMcpServerConfig, *, name: str = "configured"
) -> dict[str, Any]:
    transport = server.transport.strip().lower()
    target = os.path.expandvars(server.url).strip()

    if transport == "stdio":
        if server.authentication:
            raise ValueError("MCP authentication is not supported for stdio transport")
        if server.custom_headers:
            raise ValueError("MCP custom_headers are not supported for stdio transport")
        return common_utils.without_none(
            {
                "enabled": True,
                "command": target,
                "args": server.args or None,
                "env": server.env or None,
            }
        )

    result: dict[str, Any] = {
        "enabled": True,
        "url": target,
        "transport": transport,
    }
    if headers := server.custom_headers:
        try:
            result["headers"] = mcp_auth.normalize_custom_headers(name, headers)
        except mcp_auth.McpAuthConfigError as error:
            raise ValueError(str(error)) from error
    if authentication := server.authentication:
        raw_authentication = authentication
        try:
            if (
                isinstance(authentication, Mapping)
                and authentication.get("type") == "service_account"
            ):
                raise mcp_auth.McpAuthConfigError(
                    f"MCP server {name!r} service_account authentication is not supported by Hermes"
                )
            authentication = mcp_auth.parse_oauth2_config(name, authentication)
        except mcp_auth.McpAuthConfigError as error:
            raise ValueError(str(error)) from error
        if authentication.client_name:
            raise ValueError(
                f"MCP server {name!r} authentication.client_name is not supported by Hermes"
            )
        if authentication.token_endpoint_auth_method:
            raise ValueError(
                f"MCP server {name!r} authentication.token_endpoint_auth_method is not supported by Hermes"
            )
        if "authorization_timeout_seconds" in raw_authentication:
            raise ValueError(
                f"MCP server {name!r} authentication.authorization_timeout_seconds is not supported by Hermes"
            )
        oauth = common_utils.without_none(
            {
                "client_id": authentication.client_id,
                "scope": authentication.scope,
                "redirect_uri": authentication.redirect_uri,
            }
        )
        if secret_env := authentication.client_secret_env:
            try:
                mcp_auth.resolve_client_secret(name, authentication)
            except mcp_auth.McpAuthConfigError as error:
                raise ValueError(str(error)) from error
            oauth["client_secret"] = f"${{{secret_env}}}"
        result["auth"] = "oauth"
        result["oauth"] = oauth
    return result


def summarize_hermes_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": config.get("model", {}),
        "terminal": config.get("terminal", {}),
        "skill_dirs": (config.get("skills") or {}).get("external_dirs", []),
        "mcp_servers": sorted((config.get("mcp_servers") or {}).keys()),
        "plugins": (config.get("plugins") or {}).get("enabled", []),
        "platform_toolsets": config.get("platform_toolsets", {}),
        "disabled_toolsets": (config.get("agent") or {}).get("disabled_toolsets", []),
    }


def resolve_hermes_toolsets(
    agent_config: AgentConfig, config: dict[str, Any]
) -> list[str] | None:
    enabled = agent_config.tools.enabled if agent_config.tools is not None else None
    if enabled is not None:
        return enabled

    from hermes_cli.tools_config import _get_platform_tools

    return sorted(_get_platform_tools(config, "cli"))
