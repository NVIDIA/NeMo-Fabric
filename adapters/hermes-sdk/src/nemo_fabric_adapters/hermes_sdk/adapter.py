#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermes SDK adapter for Fabric.

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

CUR_DIR = Path(__file__).parent
ADAPTERS_DIR = CUR_DIR.parent.parent.parent.parent
COMMON_DIR = (ADAPTERS_DIR / "common/src").resolve().as_posix()
if COMMON_DIR not in sys.path:
    sys.path.append(COMMON_DIR)

import nemo_fabric_adapters.common.hermes as hermes_common  # noqa: E402
import nemo_fabric_adapters.common.utils as common_utils  # noqa: E402


def main() -> None:
    payload = json.load(sys.stdin)
    output = run(payload)
    print(json.dumps(output, sort_keys=True))
    if output.get("failed"):
        raise SystemExit(2)


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """Fabric adapter entrypoint used by script and native SDK runtime calls."""

    return asyncio.run(run_hermes_sdk(payload))


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


async def run_hermes_sdk(payload: dict[str, Any]) -> dict[str, Any]:
    hermes_common.validate_hermes_telemetry_provider(payload)
    settings = common_utils.settings_payload(payload)
    request = hermes_common.request_payload(payload)
    model_config = hermes_common.selected_model_config(payload)
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
    relay_enabled = os.environ.get("FABRIC_RELAY_ENABLED", "").strip().lower() == "true"

    relay_plugin_config = None
    if relay_enabled:
        relay_plugin_config = common_utils.load_relay_plugin_config(payload)

    hermes_config_path, hermes_config = hermes_common.write_hermes_config(
        payload,
        hermes_home,
        relay_enabled=relay_enabled,
    )

    api_key_env = settings.get("api_key_env") or model_config.get("api_key_env") or "NVIDIA_API_KEY"
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is required for Hermes SDK mode")

    base_url = hermes_common.get_base_url(settings, model_config)
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
            (result, enabled_toolsets, relay_artifacts, adapter_stdout) = _invoke_hermes(
                **hermes_kwargs
            )
    else:
        (result, enabled_toolsets, relay_artifacts, adapter_stdout) = _invoke_hermes(**hermes_kwargs)

    if relay_plugin_config is not None:
        relay_artifacts = common_utils.collect_relay_artifacts(relay_plugin_config)

    response = result.get("response") or result.get("final_response")
    messages = result.get("messages") or []
    output = {
        "harness": "hermes",
        "adapter": "python",
        "mode": "hermes_sdk",
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
        "hermes_native_config": hermes_common.summarize_hermes_config(hermes_config),
        "enabled_toolsets": enabled_toolsets,
    }
    if relay_plugin_config is not None:
        output["relay_runtime"] = {
            "enabled": True,
            "mode": os.environ.get("FABRIC_RELAY_MODE"),
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
    from hermes_cli.plugins import discover_plugins, invoke_hook
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
        agent = None
        agent = AIAgent(
            **filter_supported_kwargs(
                AIAgent,
                base_url=base_url,
                api_key=api_key,
                provider=settings.get("provider") or model_config.get("provider"),
                model=settings.get("model_name") or model_config.get("model", ""),
                max_iterations=int(settings.get("max_iterations", 1)),
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
                    model=getattr(agent, "model", None) or hermes_common.relay_model_name(payload),
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
