# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Hermes adapter helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def effective_config(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("effective_config") or {}


def fabric_config(payload: dict[str, Any]) -> dict[str, Any]:
    return effective_config(payload).get("config") or {}


def config_root(payload: dict[str, Any]) -> str:
    return effective_config(payload).get("config_root") or payload.get("config_root") or "."


def agent_name(payload: dict[str, Any]) -> str:
    return effective_config(payload).get("agent_name") or payload.get("agent_name") or "fabric-agent"


def runtime_context(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("runtime_context") or {}


def request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("request") or {}


def environment_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return runtime_context(payload).get("environment") or payload.get("environment") or {}


def settings_payload(payload: dict[str, Any]) -> dict[str, Any]:
    harness = fabric_config(payload).get("harness") or {}
    return harness.get("settings") or payload.get("settings") or {}


def models_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return fabric_config(payload).get("models") or payload.get("models") or {}


def capability_plan(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("capability_plan") or payload.get("capabilities") or {}


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
    settings = settings_payload(payload)
    models = models_payload(payload)
    model_config = models.get(settings.get("model", "default"), {})
    if not isinstance(model_config, dict):
        return {}
    return model_config


def build_hermes_config(payload: dict[str, Any], *, relay_enabled: bool = False) -> dict[str, Any]:
    settings = settings_payload(payload)
    model_config = selected_model_config(payload)
    native = capability_plan(payload).get("native") or {}
    environment = environment_payload(payload)

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
            settings.get("toolset_platform", "cli"): normalize_list(settings.get("enabled_toolsets"))
        }

    plugins = normalize_list(settings.get("plugins_enabled"))
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
    require_yaml: bool = False,
    missing_yaml_message: str = "PyYAML is required to write Hermes config",
) -> tuple[Path, dict[str, Any]]:
    hermes_home.mkdir(parents=True, exist_ok=True)
    config = build_hermes_config(payload, relay_enabled=relay_enabled)
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        dump_yaml(
            config,
            require_yaml=require_yaml,
            missing_yaml_message=missing_yaml_message,
        ),
        encoding="utf-8",
    )
    return config_path, config


def dump_yaml(
    value: dict[str, Any],
    *,
    require_yaml: bool = False,
    missing_yaml_message: str = "PyYAML is required to write Hermes config",
) -> str:
    try:
        import yaml
    except ImportError as exc:
        if require_yaml:
            raise RuntimeError(missing_yaml_message) from exc
        return json.dumps(value, indent=2, sort_keys=False) + "\n"
    return yaml.safe_dump(value, sort_keys=False)


def hermes_mcp_server_config(server: dict[str, Any]) -> dict[str, Any]:
    transport = str(server.get("transport") or "").strip().lower()
    target = os.path.expandvars(str(server.get("url") or ""))
    config: dict[str, Any] = {"enabled": True}
    if transport in {"stdio", "command", "process"}:
        config["command"] = target
    else:
        config["url"] = target
        if transport:
            config["transport"] = transport
    return config


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
            atif.get("agent_name") or agent_name(payload)
        )
        os.environ["HERMES_NEMO_RELAY_ATIF_AGENT_VERSION"] = str(atif.get("agent_version", "fabric-poc"))
        os.environ["HERMES_NEMO_RELAY_ATIF_MODEL_NAME"] = str(atif.get("model_name") or relay_model_name(payload))

    return relay_plugin_config


def load_relay_plugin_config(payload: dict[str, Any]) -> dict[str, Any]:
    config_path = os.environ.get("FABRIC_RELAY_CONFIG_PATH")
    if not config_path:
        raise RuntimeError("FABRIC_RELAY_CONFIG_PATH is required when Relay is enabled")

    with Path(config_path).open(encoding="utf-8") as stream:
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
    base = Path(config_root(payload)).resolve()
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
                section.setdefault("agent_name", agent_name(payload))
                section.setdefault("model_name", relay_model_name(payload))


def relay_model_name(payload: dict[str, Any]) -> str:
    settings = settings_payload(payload)
    models = models_payload(payload)
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
