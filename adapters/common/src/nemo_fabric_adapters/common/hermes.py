# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Hermes adapter helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import nemo_fabric_adapters.common.utils as common_utils


def request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("request") or {}


def default_base_url(provider: str | None) -> str | None:
    if provider == "nvidia":
        return "https://integrate.api.nvidia.com/v1"
    return None


def get_base_url(settings: dict[str, Any], model_config: dict[str, Any]) -> str | None:
    return (
        settings.get("base_url")
        or (model_config.get("settings") or {}).get("base_url")
        or default_base_url(model_config.get("provider"))
    )


def selected_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    settings = common_utils.settings_payload(payload)
    models = common_utils.models_payload(payload)
    model_config = models.get(settings.get("model", "default"), {})
    if not isinstance(model_config, dict):
        return {}
    return model_config


def build_hermes_config(payload: dict[str, Any], *, relay_enabled: bool = False) -> dict[str, Any]:
    settings = common_utils.settings_payload(payload)
    model_config = selected_model_config(payload)
    native = common_utils.capability_plan(payload).get("native") or {}
    environment = common_utils.environment_payload(payload)

    model_name = settings.get("model_name") or model_config.get("model", "")
    provider = settings.get("provider") or model_config.get("provider")
    base_url = get_base_url(settings, model_config)

    config: dict[str, Any] = {
        "model": without_none(
            {
                "provider": provider,
                "default": model_name,
                "base_url": base_url,
            }
        ),
        "agent": without_none(
            {
                "max_turns": settings.get("max_iterations"),
                "disabled_toolsets": settings.get("disabled_toolsets"),
            }
        ),
        "terminal": without_none(
            {
                "backend": settings.get("terminal_backend", "local"),
                "cwd": str(environment.get("workspace") or settings.get("workspace") or "."),
                "timeout": settings.get("terminal_timeout", 60),
            }
        ),
    }

    skill_dirs = [str(path) for path in native.get("skill_paths", [])]
    if skill_dirs:
        config["skills"] = {"external_dirs": skill_dirs}

    mcp_servers = native.get("mcp_servers") or {}
    if mcp_servers:
        config["mcp_servers"] = {
            name: hermes_mcp_server_config(server)
            for name, server in sorted(mcp_servers.items())
        }

    if "enabled_toolsets" in settings:
        config["platform_toolsets"] = {
            settings.get("toolset_platform", "cli"): common_utils.normalize_list(settings.get("enabled_toolsets"))
        }

    plugins = common_utils.normalize_list(settings.get("plugins_enabled"))
    if relay_enabled and "observability/nemo_relay" not in plugins:
        plugins.append("observability/nemo_relay")
    if plugins:
        config["plugins"] = {"enabled": plugins}

    return config


def write_hermes_config(
    payload: dict[str, Any],
    hermes_home: Path,
    *,
    relay_enabled: bool = False,
) -> tuple[Path, dict[str, Any]]:
    hermes_home.mkdir(parents=True, exist_ok=True)
    config = build_hermes_config(payload, relay_enabled=relay_enabled)
    config_path = hermes_home / "config.yaml"
    config_path.write_text(common_utils.dump_yaml(config), encoding="utf-8")
    return config_path, config


def hermes_mcp_server_config(server: dict[str, Any]) -> dict[str, Any]:
    transport = str(server.get("transport") or "").strip().lower()
    raw_target = server.get("url")
    if not raw_target and transport in {"stdio", "command", "process"}:
        raw_target = server.get("command")
    target = os.path.expandvars(str(raw_target or "")).strip()
    if not target:
        raise ValueError("MCP server mapping requires url or command")

    config: dict[str, Any] = {"enabled": True}
    if transport in {"stdio", "command", "process"}:
        config["command"] = target
    else:
        config["url"] = target
        if transport:
            config["transport"] = transport
    return config


def without_none(mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if value is not None}


def summarize_hermes_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": config.get("model", {}),
        "terminal": config.get("terminal", {}),
        "skill_dirs": (config.get("skills") or {}).get("external_dirs", []),
        "mcp_servers": sorted((config.get("mcp_servers") or {}).keys()),
        "plugins": (config.get("plugins") or {}).get("enabled", []),
        "platform_toolsets": config.get("platform_toolsets", {}),
    }


def configure_hermes_relay(payload: dict[str, Any]) -> dict[str, Any] | None:
    if os.environ.get("FABRIC_RELAY_ENABLED") != "true":
        return None

    relay_plugin_config = common_utils.load_relay_plugin_config(payload)
    (_, relay_plugins_toml_path) = common_utils.write_relay_configs(plugin_config=relay_plugin_config)
    if relay_plugins_toml_path is not None:
        os.environ["HERMES_NEMO_RELAY_PLUGINS_TOML"] = str(relay_plugins_toml_path)

    observability = next(
        (
            component.get("config") or {}
            for component in relay_plugin_config.get("components", [])
            if component.get("kind") == "observability" and component.get("enabled", True)
        ),
        {},
    )
    atof = observability.get("atof") if isinstance(observability, dict) else None
    atif = observability.get("atif") if isinstance(observability, dict) else None

    if isinstance(atof, dict) and atof.get("enabled"):
        os.environ["HERMES_NEMO_RELAY_ATOF_ENABLED"] = "1"
        os.environ["HERMES_NEMO_RELAY_ATOF_OUTPUT_DIRECTORY"] = str(atof["output_directory"])
        os.environ["HERMES_NEMO_RELAY_ATOF_FILENAME"] = str(atof.get("filename", "events.atof.jsonl"))
        os.environ["HERMES_NEMO_RELAY_ATOF_MODE"] = str(atof.get("mode", "overwrite"))

    if isinstance(atif, dict) and atif.get("enabled"):
        os.environ["HERMES_NEMO_RELAY_ATIF_ENABLED"] = "1"
        os.environ["HERMES_NEMO_RELAY_ATIF_OUTPUT_DIRECTORY"] = str(atif["output_directory"])
        os.environ["HERMES_NEMO_RELAY_ATIF_FILENAME_TEMPLATE"] = str(
            atif.get("filename_template", "trajectory-{session_id}.atif.json")
        )
        os.environ["HERMES_NEMO_RELAY_ATIF_AGENT_NAME"] = str(
            atif.get("agent_name") or common_utils.agent_name(payload)
        )
        os.environ["HERMES_NEMO_RELAY_ATIF_AGENT_VERSION"] = str(atif.get("agent_version", "fabric-poc"))
        os.environ["HERMES_NEMO_RELAY_ATIF_MODEL_NAME"] = str(atif.get("model_name") or relay_model_name(payload))

    return relay_plugin_config


def relay_model_name(payload: dict[str, Any]) -> str:
    settings = common_utils.settings_payload(payload)
    models = common_utils.models_payload(payload)
    model_config = models.get(settings.get("model", "default"), {})
    return settings.get("model_name") or model_config.get("model") or "unknown"


def ensure_hermes_session(
    fabric_session_id: str,
    model_name: str,
    model_config: dict[str, Any],
    hermes_home: Path,
) -> dict[str, Any]:
    """
    Ensure that Hermes has a session mapped from Fabric's session key.

    Fabric chooses this key from runtime_context.session_id when the caller
    supplies one, otherwise from runtime_context.runtime_id. The adapter maps
    that Fabric-owned key onto Hermes' session id/title.

    If the session does not exist, it will be created.

    When creating a new session, Hermes allows us to provide our own session_id (as long as it's unique), which for
    convenience will be set to the Fabric session key.

    However when Hermes compresses a session, it will return a new session_id, so we can't depend on the
    Fabric session key being the same as the session_id after a session has been compressed.

    However looking up a session by title will always return the most recent session, so after creating the session
    we will set the title to the Fabric session key, and then we can always look up the session by title.
    """
    from hermes_state import SessionDB

    session_db = SessionDB(db_path=hermes_home / "state.db")
    session = session_db.get_session_by_title(fabric_session_id)
    if session is None:
        session_db.ensure_session(
            fabric_session_id,
            source="fabric",
            model=model_name,
            model_config=model_config,
        )
        session_db.set_session_title(session_id=fabric_session_id, title=fabric_session_id)
        session = session_db.get_session_by_title(fabric_session_id)

    return session
