#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermes SDK adapter for Fabric.

This adapter maps Fabric's normalized config into Hermes' native Python SDK
surface and invokes the installed Hermes runtime.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any


def main() -> None:
    payload = json.load(sys.stdin)
    output = run_hermes_sdk(payload)
    print(json.dumps(output, sort_keys=True))
    if output.get("failed"):
        raise SystemExit(2)


def load_relay_plugin_config(payload: dict[str, Any]) -> dict[str, Any]:
    config_path = os.environ.get("FABRIC_RELAY_CONFIG_PATH")
    if not config_path:
        raise RuntimeError("FABRIC_RELAY_CONFIG_PATH is required when Relay is enabled")

    with Path(config_path).open() as stream:
        wrapper = json.load(stream)

    relay = wrapper.get("relay", {})
    plugin_config = relay.get("config") or {}
    if "components" not in plugin_config:
        plugin_config = {
            "version": 1,
            "components": [
                {
                    "kind": "observability",
                    "enabled": True,
                    "config": plugin_config or {"version": 1},
                }
            ],
        }
    plugin_config.setdefault("version", 1)
    plugin_config.setdefault("components", [])
    normalize_relay_output_dirs(plugin_config, payload)
    return plugin_config


def normalize_relay_output_dirs(plugin_config: dict[str, Any], payload: dict[str, Any]) -> None:
    base = Path(payload.get("config_root") or ".").resolve()
    for component in plugin_config.get("components", []):
        if component.get("kind") != "observability":
            continue
        config = component.setdefault("config", {})
        config.setdefault("version", 1)
        for section_name in ("atof", "atif"):
            section = config.get(section_name)
            if not isinstance(section, dict) or not section.get("enabled"):
                continue
            output_directory = section.get("output_directory")
            if output_directory:
                path = Path(output_directory)
                if not path.is_absolute():
                    section["output_directory"] = str(base / path)
            else:
                section["output_directory"] = str(base / "artifacts" / "relay")
            if section_name == "atof":
                section.setdefault("filename", "events.atof.jsonl")
                section.setdefault("mode", "overwrite")
            if section_name == "atif":
                section.setdefault("filename_template", "trajectory-{session_id}.atif.json")
                section.setdefault("agent_name", payload.get("agent_name") or "fabric-agent")
                section.setdefault("model_name", relay_model_name(payload))


def relay_model_name(payload: dict[str, Any]) -> str:
    settings = payload.get("settings", {})
    models = payload.get("models", {})
    model_config = models.get(settings.get("model", "default"), {})
    return settings.get("model_name") or model_config.get("model") or "unknown"


def collect_relay_artifacts(plugin_config: dict[str, Any]) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    for component in plugin_config.get("components", []):
        if component.get("kind") != "observability":
            continue
        config = component.get("config") or {}
        for section_name, pattern in (
            ("atof", "*.jsonl"),
            ("atif", "*.json"),
        ):
            section = config.get(section_name)
            if not isinstance(section, dict) or not section.get("enabled"):
                continue
            directory = Path(section.get("output_directory") or ".")
            if not directory.exists():
                continue
            for path in sorted(directory.glob(pattern)):
                artifacts.append({"kind": section_name, "path": str(path)})
    return artifacts


def configure_hermes_relay(payload: dict[str, Any]) -> dict[str, Any] | None:
    if os.environ.get("FABRIC_RELAY_ENABLED") != "true":
        return None

    relay_plugin_config = load_relay_plugin_config(payload)
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
            atif.get("agent_name") or payload.get("agent_name") or "fabric-agent"
        )
        os.environ["HERMES_NEMO_RELAY_ATIF_AGENT_VERSION"] = str(atif.get("agent_version", "fabric-poc"))
        os.environ["HERMES_NEMO_RELAY_ATIF_MODEL_NAME"] = str(atif.get("model_name") or relay_model_name(payload))

    return relay_plugin_config


def write_hermes_config(
    payload: dict[str, Any],
    hermes_home: Path,
    *,
    relay_enabled: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Materialize Fabric config into Hermes' native config.yaml shape."""

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Hermes SDK mode requires PyYAML to write Hermes config") from exc

    hermes_home.mkdir(parents=True, exist_ok=True)
    config_path = hermes_home / "config.yaml"
    config = build_hermes_config(payload, relay_enabled=relay_enabled)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, config


def build_hermes_config(payload: dict[str, Any], *, relay_enabled: bool = False) -> dict[str, Any]:
    settings = payload.get("settings", {})
    model_config = selected_model_config(payload)
    native = (payload.get("capabilities") or {}).get("native") or {}
    environment = payload.get("environment", {})

    model_name = settings.get("model_name") or model_config.get("model", "")
    provider = settings.get("provider") or model_config.get("provider")
    base_url = (
        settings.get("base_url")
        or (model_config.get("settings") or {}).get("base_url")
        or default_base_url(model_config.get("provider"))
    )

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
                "max_turns": settings.get("max_turns"),
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
            settings.get("toolset_platform", "cli"): normalize_list(settings.get("enabled_toolsets"))
        }

    plugins = normalize_list(settings.get("plugins_enabled"))
    if relay_enabled and "observability/nemo_relay" not in plugins:
        plugins.append("observability/nemo_relay")
    if plugins:
        config["plugins"] = {"enabled": plugins}

    return config


def hermes_mcp_server_config(server: dict[str, Any]) -> dict[str, Any]:
    transport = str(server.get("transport") or "").strip().lower()
    target = os.path.expandvars(str(server.get("url") or ""))
    config: dict[str, Any] = {
        "enabled": True,
    }
    if transport in {"stdio", "command", "process"}:
        config["command"] = target
    else:
        config["url"] = target
        if transport:
            config["transport"] = transport
    return config


def selected_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    settings = payload.get("settings", {})
    models = payload.get("models", {})
    key = settings.get("model", "default")
    model_config = models.get(key, {})
    if not isinstance(model_config, dict):
        return {}
    return model_config


def normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        value = [value]
    return [str(item) for item in value if str(item)]


def without_none(mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if value is not None}


def resolve_hermes_toolsets(settings: dict[str, Any], config: dict[str, Any]) -> list[str] | None:
    if "enabled_toolsets" in settings:
        return normalize_list(settings.get("enabled_toolsets"))

    from hermes_cli.tools_config import _get_platform_tools

    platform = settings.get("toolset_platform", "cli")
    return sorted(_get_platform_tools(config, platform))


def run_hermes_sdk(payload: dict[str, Any]) -> dict[str, Any]:
    settings = payload.get("settings", {})
    request = payload.get("request", {})
    model_config = selected_model_config(payload)
    hermes_home = Path(payload.get("config_root", ".")).joinpath(
        settings.get("hermes_home", "./artifacts/hermes-home")
    )
    hermes_home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(hermes_home)
    os.environ["HERMES_HOME"] = str(hermes_home)
    os.environ.setdefault("HERMES_YOLO_MODE", "1")
    os.environ.setdefault("HERMES_ACCEPT_HOOKS", "1")
    os.environ.setdefault("TERMINAL_ENV", settings.get("terminal_backend", "local"))
    os.environ.setdefault("TERMINAL_TIMEOUT", str(settings.get("terminal_timeout", 60)))
    relay_plugin_config = configure_hermes_relay(payload)
    hermes_config_path, hermes_config = write_hermes_config(
        payload,
        hermes_home,
        relay_enabled=relay_plugin_config is not None,
    )

    api_key_env = settings.get("api_key_env") or model_config.get("api_key_env") or "NVIDIA_API_KEY"
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is required for Hermes SDK mode")

    base_url = (
        settings.get("base_url")
        or (model_config.get("settings") or {}).get("base_url")
        or default_base_url(model_config.get("provider"))
    )
    user_message = request.get("input") or ""
    if not isinstance(user_message, str):
        user_message = json.dumps(user_message, sort_keys=True)

    hermes_stdout = StringIO()
    relay_artifacts: list[dict[str, str]] = []
    with redirect_stdout(hermes_stdout):
        from hermes_cli.config import load_config
        from hermes_cli.plugins import discover_plugins, invoke_hook
        from run_agent import AIAgent

        discover_plugins(force=True)
        loaded_hermes_config = load_config()
        enabled_toolsets = resolve_hermes_toolsets(settings, loaded_hermes_config)
        agent = None
        agent = AIAgent(**filter_supported_kwargs(
            AIAgent,
            base_url=base_url,
            api_key=api_key,
            provider=settings.get("provider") or model_config.get("provider"),
            model=settings.get("model_name") or model_config.get("model", ""),
            max_iterations=int(settings.get("max_turns", 1)),
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
        ))
        try:
            conversation_kwargs = filter_supported_call_kwargs(
                agent.run_conversation,
                system_message=settings.get("system_prompt"),
                conversation_history=settings.get("history"),
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
                    model=getattr(agent, "model", None) or relay_model_name(payload),
                    platform=getattr(agent, "platform", None) or "fabric",
                )
                relay_artifacts = collect_relay_artifacts(relay_plugin_config)
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
        "adapter_stdout": hermes_stdout.getvalue(),
        "hermes_home": str(hermes_home),
        "hermes_config_path": str(hermes_config_path),
        "hermes_native_config": summarize_hermes_config(hermes_config),
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


def summarize_hermes_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": config.get("model", {}),
        "terminal": config.get("terminal", {}),
        "skill_dirs": (config.get("skills") or {}).get("external_dirs", []),
        "mcp_servers": sorted((config.get("mcp_servers") or {}).keys()),
        "plugins": (config.get("plugins") or {}).get("enabled", []),
        "platform_toolsets": config.get("platform_toolsets", {}),
    }


def filter_supported_kwargs(callable_obj: Any, **kwargs: Any) -> dict[str, Any]:
    signature = inspect.signature(callable_obj.__init__)
    supported = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in supported}


def filter_supported_call_kwargs(func: Any, **kwargs: Any) -> dict[str, Any]:
    signature = inspect.signature(func)
    supported = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in supported}


def default_base_url(provider: str | None) -> str | None:
    if provider == "nvidia":
        return "https://integrate.api.nvidia.com/v1"
    return None


if __name__ == "__main__":
    main()
