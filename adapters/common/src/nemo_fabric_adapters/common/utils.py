# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared adapter utility helpers."""

from __future__ import annotations

import json
import os
import sys
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


def environment_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return runtime_context(payload).get("environment") or payload.get("environment") or {}


def settings_payload(payload: dict[str, Any]) -> dict[str, Any]:
    harness = fabric_config(payload).get("harness") or {}
    return harness.get("settings") or payload.get("settings") or {}


def models_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return fabric_config(payload).get("models") or payload.get("models") or {}


def capability_plan(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("capability_plan") or payload.get("capabilities") or {}


def normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        value = [value]
    return [str(item) for item in value if str(item)]


def dump_yaml(value: dict[str, Any]) -> str:
    try:
        import yaml
        return yaml.safe_dump(value, sort_keys=False)
    except ImportError:
        # Since yaml is a super-set of json, we can always dump to json to a yaml file
        return json.dumps(value, indent=2, sort_keys=False) + "\n"


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
                section.setdefault("model_name", _relay_model_name(payload))


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


def write_relay_plugins_toml(plugin_config: dict[str, Any]) -> Path | None:
    try:
        import tomli_w

        config_path = os.environ.get("FABRIC_RELAY_CONFIG_PATH")
        if not config_path:
            raise RuntimeError("FABRIC_RELAY_CONFIG_PATH is required when Relay is enabled")

        path = Path(config_path).with_name("relay-plugins.toml")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tomli_w.dumps(plugin_config), encoding="utf-8")
        return path
    except ImportError:
        print("tomli_w is not installed, skipping writing relay plugins TOML", file=sys.stderr)
        return None


def _relay_model_name(payload: dict[str, Any]) -> str:
    settings = settings_payload(payload)
    models = models_payload(payload)
    model_config = models.get(settings.get("model", "default"), {})
    return settings.get("model_name") or model_config.get("model") or "unknown"
