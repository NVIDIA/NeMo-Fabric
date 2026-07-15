#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermes adapter for Fabric.

This adapter maps Fabric's normalized config into Hermes' native Python SDK
surface and invokes the installed Hermes runtime.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import nemo_fabric_adapters.common.utils as common_utils

# Default agent loop budget when harness.settings.max_iterations is unset.
# Mirrors Hermes' own AIAgent default (agent/agent_init.py); a lower value such
# as 1 silently starves multi-step tasks (they run out of budget before
# answering while the trial still reports success). See FABRIC-85.
DEFAULT_MAX_ITERATIONS: int = 90


def validate_hermes_telemetry_provider(payload: dict[str, Any]) -> None:
    providers = common_utils.telemetry_providers(payload)
    if any(provider != "relay" for provider in providers):
        raise ValueError("only relay telemetry is supported for Hermes")


def build_hermes_config(payload: dict[str, Any], *, relay_enabled: bool = False) -> dict[str, Any]:
    settings = common_utils.settings_payload(payload)
    model_config = common_utils.selected_model_config(payload)
    native = common_utils.capability_plan(payload).get("native") or {}
    environment = common_utils.environment_payload(payload)

    model_name = settings.get("model_name") or model_config.get("model", "")
    provider = settings.get("provider") or model_config.get("provider")
    base_url = common_utils.get_base_url(settings, model_config)

    config: dict[str, Any] = {
        "model": common_utils.without_none(
            {
                "provider": provider,
                "default": model_name,
                "base_url": base_url,
            }
        ),
        "agent": common_utils.without_none(
            {
                "max_turns": settings.get("max_iterations"),
                "disabled_toolsets": settings.get("disabled_toolsets"),
            }
        ),
        "terminal": common_utils.without_none(
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
    target = os.path.expandvars(str(raw_target or "")).strip()
    if not target:
        raise ValueError("MCP server mapping requires a URL")

    return {"enabled": True, "url": target, "transport": transport}


def summarize_hermes_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": config.get("model", {}),
        "terminal": config.get("terminal", {}),
        "skill_dirs": (config.get("skills") or {}).get("external_dirs", []),
        "mcp_servers": sorted((config.get("mcp_servers") or {}).keys()),
        "plugins": (config.get("plugins") or {}).get("enabled", []),
        "platform_toolsets": config.get("platform_toolsets", {}),
    }


def main() -> None:
    payload = json.load(sys.stdin)
    output = run(payload)
    print(json.dumps(output, sort_keys=True))
    if output.get("failed"):
        raise SystemExit(2)


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """Fabric adapter entrypoint used by script and native SDK runtime calls."""

    return asyncio.run(run_hermes(payload))


def resolve_hermes_toolsets(settings: dict[str, Any], config: dict[str, Any]) -> list[str] | None:
    if "enabled_toolsets" in settings:
        return common_utils.normalize_list(settings.get("enabled_toolsets"))

    from hermes_cli.tools_config import _get_platform_tools

    platform = settings.get("toolset_platform", "cli")
    return sorted(_get_platform_tools(config, platform))


def load_runtime_history(session_db: Any, session_id: str | None) -> list[dict[str, Any]] | None:
    if not session_id:
        return None

    resolved_id = session_id
    resolve_session = getattr(session_db, "resolve_resume_session_id", None)
    if resolve_session is not None:
        resolved_id = resolve_session(session_id) or session_id
    if session_db.get_session(resolved_id) is None:
        return None

    messages = session_db.get_messages_as_conversation(resolved_id)
    messages = [message for message in messages if message.get("role") != "session_meta"]
    return messages or None


async def run_hermes(payload: dict[str, Any]) -> dict[str, Any]:
    validate_hermes_telemetry_provider(payload)
    settings = common_utils.settings_payload(payload)
    request = common_utils.request_payload(payload)
    model_config = common_utils.selected_model_config(payload)
    hermes_home_base = Path(common_utils.config_root(payload)).joinpath(
        settings.get("hermes_home", "./artifacts/hermes-home")
    )
    hermes_home = common_utils.runtime_state_directory(hermes_home_base, payload)
    hermes_home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(hermes_home)
    os.environ["HERMES_HOME"] = str(hermes_home)
    os.environ.setdefault("HERMES_YOLO_MODE", "1")
    os.environ.setdefault("HERMES_ACCEPT_HOOKS", "1")
    os.environ["HERMES_SESSION_SOURCE"] = "fabric"
    os.environ.setdefault("TERMINAL_ENV", settings.get("terminal_backend", "local"))
    os.environ.setdefault("TERMINAL_TIMEOUT", str(settings.get("terminal_timeout", 60)))
    relay_enabled = common_utils.relay_enabled(payload)

    relay_plugin_config = None
    if relay_enabled:
        relay_plugin_config = common_utils.load_relay_plugin_config(payload)

    hermes_config_path, hermes_config = write_hermes_config(
        payload,
        hermes_home,
        relay_enabled=relay_enabled,
    )

    api_key_env = settings.get("api_key_env") or model_config.get("api_key_env") or "NVIDIA_API_KEY"
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is required for Hermes mode")

    base_url = common_utils.get_base_url(settings, model_config)
    user_message = request.get("input") or ""
    if not isinstance(user_message, str):
        user_message = json.dumps(user_message, sort_keys=True)

    hermes_kwargs = {
        "payload": payload,
        "settings": settings,
        "model_config": model_config,
        "base_url": base_url,
        "api_key": api_key,
        "user_message": user_message,
        "relay_plugin_config": relay_plugin_config,
    }

    if relay_enabled:
        relay_api_config = common_utils.relay_api_plugin_config(relay_plugin_config or {})
        from nemo_relay import plugin

        async with plugin.plugin(relay_api_config):
            (result, enabled_toolsets, relay_artifacts, adapter_stdout) = _invoke_hermes(**hermes_kwargs)
    else:
        (result, enabled_toolsets, relay_artifacts, adapter_stdout) = _invoke_hermes(**hermes_kwargs)

    if relay_plugin_config is not None:
        relay_artifacts = common_utils.collect_relay_artifacts(relay_plugin_config)

    response = result.get("response") or result.get("final_response")
    messages = result.get("messages") or []
    output = {
        "harness": "hermes",
        "adapter": "python",
        "mode": "hermes",
        "model": model_config.get("model"),
        "base_url": base_url,
        "response": response,
        "completed": bool(result.get("completed")),
        "failed": bool(result.get("failed")),
        "api_calls": result.get("api_calls"),
        "messages": messages,
        "message_count": len(messages),
        "error": result.get("error"),
        "adapter_stdout": adapter_stdout,
        "hermes_home": str(hermes_home),
        "hermes_config_path": str(hermes_config_path),
        "hermes_native_config": summarize_hermes_config(hermes_config),
        "enabled_toolsets": enabled_toolsets,
    }
    if relay_plugin_config is not None:
        output["relay_runtime"] = {
            "enabled": True,
            "config_path": os.environ.get("FABRIC_RELAY_CONFIG_PATH"),
            "emitter": "hermes.observability/nemo_relay",
        }
        output["relay_artifacts"] = relay_artifacts
    return output


def _invoke_hermes(
    *,
    payload: dict[str, Any],
    settings: dict[str, Any],
    model_config: dict[str, Any],
    base_url: str | None,
    api_key: str,
    user_message: str,
    relay_plugin_config: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str] | None, list[dict[str, str]], str]:
    from hermes_cli.config import load_config
    from hermes_cli.plugins import discover_plugins
    from hermes_cli.plugins import invoke_hook
    from hermes_state import SessionDB
    from run_agent import AIAgent

    relay_artifacts: list[dict[str, str]] = []
    hermes_stdout = StringIO()
    with redirect_stdout(hermes_stdout):
        discover_plugins(force=True)
        loaded_hermes_config = load_config()
        enabled_toolsets = resolve_hermes_toolsets(settings, loaded_hermes_config)
        session_id = common_utils.runtime_id(payload)
        session_db = SessionDB()
        conversation_history = load_runtime_history(session_db, session_id)
        # Treat an explicit null max_iterations like an unset one (avoid int(None)).
        max_iterations = settings.get("max_iterations")
        if max_iterations is None:
            max_iterations = DEFAULT_MAX_ITERATIONS
        agent = None
        agent = AIAgent(
            **filter_supported_kwargs(
                AIAgent,
                base_url=base_url,
                api_key=api_key,
                provider=settings.get("provider") or model_config.get("provider"),
                model=settings.get("model_name") or model_config.get("model", ""),
                max_iterations=int(max_iterations),
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=settings.get("disabled_toolsets"),
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                save_trajectories=bool(settings.get("save_trajectories", False)),
                max_tokens=settings.get("max_tokens", 512),
                temperature=settings.get("temperature", model_config.get("temperature", 0.0)),
                reasoning_config=settings.get("reasoning_config", {"effort": "none"}),
                insert_reasoning=bool(settings.get("insert_reasoning", False)),
                platform="fabric",
                session_id=session_id,
                session_db=session_db,
            )
        )
        try:
            conversation_kwargs = filter_supported_call_kwargs(
                agent.run_conversation,
                system_message=settings.get("system_prompt"),
                conversation_history=conversation_history,
                sync_honcho=False,
                dont_review=True,
            )
            result = agent.run_conversation(
                user_message,
                **conversation_kwargs,
            )
        finally:
            if relay_plugin_config is not None and agent is not None:
                invoke_hook(
                    "on_session_finalize",
                    session_id=getattr(agent, "session_id", ""),
                    model=getattr(agent, "model", None) or common_utils.relay_model_name(payload),
                    platform=getattr(agent, "platform", None) or "fabric",
                )

    return result, enabled_toolsets, relay_artifacts, hermes_stdout.getvalue()


def filter_supported_kwargs(callable_obj: Any, **kwargs: Any) -> dict[str, Any]:
    signature = inspect.signature(callable_obj.__init__)
    supported = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in supported}


def filter_supported_call_kwargs(func: Any, **kwargs: Any) -> dict[str, Any]:
    signature = inspect.signature(func)
    supported = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in supported}


if __name__ == "__main__":
    main()
