#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermes CLI adapter for Fabric.

This adapter maps Fabric's normalized config into Hermes' native config files,
then invokes the installed `hermes` CLI in one-shot mode.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    payload = load_payload()
    output = run_hermes_cli(payload)
    print(json.dumps(output, sort_keys=True))
    if output.get("failed"):
        raise SystemExit(2)


def load_payload() -> dict[str, Any]:
    invocation_path = os.environ.get("FABRIC_INVOCATION")
    if invocation_path:
        path = Path(invocation_path)
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return json.load(sys.stdin)


def effective_config(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("effective_config") or {}


def fabric_config(payload: dict[str, Any]) -> dict[str, Any]:
    return effective_config(payload).get("config") or {}


def config_root_payload(payload: dict[str, Any]) -> str:
    return effective_config(payload).get("config_root") or payload.get("config_root") or "."


def runtime_context(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("runtime_context") or {}


def request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("request") or {}


def environment_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return runtime_context(payload).get("environment") or payload.get("environment") or {}


def settings_payload(payload: dict[str, Any]) -> dict[str, Any]:
    harness = (fabric_config(payload).get("harness") or {})
    return harness.get("settings") or payload.get("settings") or {}


def models_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return fabric_config(payload).get("models") or payload.get("models") or {}


def capability_plan(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("capability_plan") or payload.get("capabilities") or {}


def _api_key_preflight_check(settings: dict[str, Any], model_config: dict[str, Any]) -> None:
    api_key_env = settings.get("api_key_env") or model_config.get("api_key_env")
    if api_key_env:
        try:
            os.environ[api_key_env]
        except KeyError as exc:
            raise RuntimeError(f"api_key_env={api_key_env} is defined in the configuration but is not set in the "
                               "environment. Please set it to your API key.") from exc

def run_hermes_cli(payload: dict[str, Any]) -> dict[str, Any]:
    settings = settings_payload(payload)
    request = request_payload(payload)
    config_root = Path(config_root_payload(payload)).resolve()
    environment = environment_payload(payload)
    model_config = selected_model_config(payload)
    _api_key_preflight_check(settings, model_config)

    model_name = settings.get("model_name") or model_config.get("model")

    hermes_home = resolve_path(
        config_root,
        settings.get("hermes_home", "./artifacts/hermes-cli/home"),
    )
    hermes_home.mkdir(parents=True, exist_ok=True)
    hermes_config_path, hermes_config = write_hermes_config(
        payload, hermes_home)

    prompt = request_to_prompt(request)
    toolsets = normalize_list(settings.get("enabled_toolsets"))
    command = build_command(settings,
                            config_root,
                            model_config,
                            model_name,
                            prompt,
                            toolsets=toolsets)
    cwd = resolve_path(
        config_root,
        settings.get("cwd") or environment.get("workspace") or ".")
    env = build_env(settings, hermes_home)

    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    response = completed.stdout.strip()
    output = {
        "harness": "hermes",
        "adapter": "cli",
        "base_url": get_base_url(settings, model_config),
        "mode": "hermes_cli_oneshot",
        "command": redact_command(command),
        "cwd": str(cwd),
        "enabled_toolsets": toolsets,
        "error": completed.stderr or None,
        "fabric_home": os.environ.get("FABRIC_HOME"),
        "fabric_invocation": os.environ.get("FABRIC_INVOCATION"),
        "model": model_name,
        "returncode": completed.returncode,
        "response": response,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "failed": completed.returncode != 0,
        "hermes_home": str(hermes_home),
        "hermes_config_path": str(hermes_config_path),
        "hermes_native_config": summarize_hermes_config(hermes_config),
    }
    return output


def build_command(
    settings: dict[str, Any],
    config_root: Path,
    model_config: dict[str, Any],
    model_name: str | None,
    prompt: str,
    toolsets: list[str] | None = None
) -> list[str]:
    command = resolve_command(
        config_root,
        settings.get("hermes_command") or settings.get("command", "hermes"),
    )
    command_args = normalize_list(settings.get("hermes_args") or settings.get("command_args"))
    provider = settings.get("provider") or model_config.get("provider")

    args = [command, *command_args, "-z", prompt]
    if model_name:
        args.extend(["--model", str(model_name)])
    if provider and settings.get("pass_provider_flag", True):
        args.extend(["--provider", str(provider)])
    if toolsets:
        args.extend(["--toolsets", ",".join(toolsets)])
    return args


def build_env(settings: dict[str, Any], hermes_home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in (settings.get("env") or {}).items()})
    env["HOME"] = str(hermes_home)
    env["HERMES_HOME"] = str(hermes_home)
    env.setdefault("HERMES_YOLO_MODE", "1")
    env.setdefault("HERMES_ACCEPT_HOOKS", "1")
    return env


def write_hermes_config(payload: dict[str, Any], hermes_home: Path) -> tuple[Path, dict[str, Any]]:
    hermes_home.mkdir(parents=True, exist_ok=True)
    config = build_hermes_config(payload)
    config_path = hermes_home / "config.yaml"
    config_path.write_text(dump_yaml(config), encoding="utf-8")
    return config_path, config

def get_base_url(settings: dict[str, Any], model_config: dict[str, Any]) -> str | None:
    return (
        settings.get("base_url")
        or (model_config.get("settings") or {}).get("base_url")
        or default_base_url(model_config.get("provider"))
    )

def build_hermes_config(payload: dict[str, Any]) -> dict[str, Any]:
    settings = settings_payload(payload)
    model_config = selected_model_config(payload)
    native = (capability_plan(payload)).get("native") or {}
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
            settings.get("toolset_platform", "cli"): normalize_list(
                settings.get("enabled_toolsets")
            )
        }

    plugins = normalize_list(settings.get("plugins_enabled"))
    if plugins:
        config["plugins"] = {"enabled": plugins}

    return config


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


def selected_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    settings = settings_payload(payload)
    models = models_payload(payload)
    model_config = models.get(settings.get("model", "default"), {})
    if not isinstance(model_config, dict):
        return {}
    return model_config


def request_to_prompt(request: dict[str, Any]) -> str:
    value = request.get("input", "")
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return root / path


def resolve_command(root: Path, value: Any) -> str:
    command = str(value)
    path = Path(command)
    if path.is_absolute() or len(path.parts) > 1:
        return str(resolve_path(root, path))
    return command


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


def dump_yaml(value: dict[str, Any]) -> str:
    try:
        import yaml
    except ImportError:
        return json.dumps(value, indent=2, sort_keys=False) + "\n"
    return yaml.safe_dump(value, sort_keys=False)


def redact_command(command: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for arg in command:
        if redact_next:
            redacted.append("<prompt>")
            redact_next = False
            continue
        if any(secret in arg.upper() for secret in ("API_KEY", "TOKEN", "SECRET")):
            redacted.append("<redacted>")
        else:
            redacted.append(arg)
        if arg in {"-z", "--oneshot"}:
            redact_next = True
    return redacted


def default_base_url(provider: str | None) -> str | None:
    if provider == "nvidia":
        return "https://integrate.api.nvidia.com/v1"
    return None


if __name__ == "__main__":
    main()
