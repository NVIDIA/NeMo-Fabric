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
from pathlib import Path
from typing import Any

import nemo_fabric_adapters.common.hermes as hermes_common
import nemo_fabric_adapters.common.utils as common_utils


def main() -> None:
    payload = common_utils.load_payload()
    output = run_hermes_cli(payload)
    print(json.dumps(output, sort_keys=True))
    if output.get("failed"):
        raise SystemExit(2)


def _api_key_preflight_check(settings: dict[str, Any], model_config: dict[str, Any]) -> None:
    api_key_env = settings.get("api_key_env") or model_config.get("api_key_env")
    if api_key_env:
        try:
            os.environ[api_key_env]
        except KeyError as exc:
            raise RuntimeError(
                f"api_key_env={api_key_env} is defined in the configuration but is not set in the "
                "environment. Please set it to your API key."
            ) from exc


def run_hermes_cli(payload: dict[str, Any]) -> dict[str, Any]:
    hermes_common.validate_hermes_telemetry_provider(payload)
    settings = common_utils.settings_payload(payload)
    request = hermes_common.request_payload(payload)
    config_root = Path(common_utils.config_root(payload)).resolve()
    environment = common_utils.environment_payload(payload)
    model_config = hermes_common.selected_model_config(payload)
    model_name = settings.get("model_name") or model_config.get("model")
    fabric_runtime_id = common_utils.runtime_id(payload)

    relay_plugin_config = hermes_common.configure_hermes_relay(payload)

    _api_key_preflight_check(settings, model_config)

    hermes_home_base = resolve_path(
        config_root,
        settings.get("hermes_home", "./artifacts/hermes-cli/home"),
    )
    hermes_home = common_utils.runtime_state_directory(hermes_home_base, payload)
    hermes_home.mkdir(parents=True, exist_ok=True)
    hermes_config_path, hermes_config = hermes_common.write_hermes_config(
        payload,
        hermes_home,
        relay_enabled=relay_plugin_config is not None,
    )

    if settings.get("prepare_runtime_state", True):
        hermes_common.ensure_hermes_runtime_session(
            fabric_runtime_id,
            model_name,
            model_config,
            hermes_home,
        )

    prompt = request_to_prompt(request)
    toolsets = common_utils.normalize_list(settings.get("enabled_toolsets"))

    command = build_command(
        settings,
        config_root,
        model_config,
        model_name,
        prompt,
        toolsets=toolsets,
        use_native_session=True,
        fabric_runtime_id=fabric_runtime_id,
    )
    cwd = resolve_path(
        config_root,
        settings.get("cwd") or environment.get("workspace") or ".",
    )
    env = build_env(settings, hermes_home)

    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    # Hermes prints its native session id to stderr when --continue is used.
    response = completed.stdout.strip()
    stderr_output = completed.stderr.strip()
    return_code = completed.returncode
    if return_code != 0:
        error_message = stderr_output or f"hermes CLI exited with return code {return_code}"
    else:
        error_message = None

    output = {
        "harness": "hermes",
        "adapter": "cli",
        "base_url": hermes_common.get_base_url(settings, model_config),
        "mode": "hermes_cli_runtime",
        "command": redact_command(command),
        "cwd": str(cwd),
        "enabled_toolsets": toolsets,
        "error": error_message,
        "fabric_home": os.environ.get("FABRIC_HOME"),
        "fabric_invocation": os.environ.get("FABRIC_INVOCATION"),
        "model": model_name,
        "returncode": return_code,
        "response": response,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "failed": return_code != 0,
        "hermes_home": str(hermes_home),
        "hermes_config_path": str(hermes_config_path),
        "hermes_native_config": hermes_common.summarize_hermes_config(hermes_config),
    }

    if relay_plugin_config is not None:
        relay_artifacts = common_utils.collect_relay_artifacts(relay_plugin_config)
        output["relay_runtime"] = {
            "enabled": True,
            "mode": os.environ.get("FABRIC_RELAY_MODE"),
            "config_path": os.environ.get("FABRIC_RELAY_CONFIG_PATH"),
            "emitter": "hermes.observability/nemo_relay",
        }
        output["relay_artifacts"] = relay_artifacts
    return output


def build_command(
    settings: dict[str, Any],
    config_root: Path,
    model_config: dict[str, Any],
    model_name: str | None,
    prompt: str,
    toolsets: list[str] | None = None,
    use_native_session: bool = False,
    fabric_runtime_id: str | None = None,
) -> list[str]:
    command = resolve_command(
        config_root,
        settings.get("hermes_command") or settings.get("command", "hermes"),
    )
    command_args = common_utils.normalize_list(settings.get("hermes_args") or settings.get("command_args"))
    provider = settings.get("provider") or model_config.get("provider")

    args = [command, *command_args, "chat", "--quiet", "--query", prompt]
    if use_native_session:
        if not fabric_runtime_id:
            raise RuntimeError("Hermes native session mode requires a Fabric runtime_id")
        # The Fabric runtime id is mapped onto Hermes' native session id/title.
        # On the first invocation, this resumes an empty session created up front.
        args.extend(["--continue", fabric_runtime_id])

    if model_name:
        args.extend(["--model", str(model_name)])
    if provider and settings.get("pass_provider_flag", True):
        args.extend(["--provider", str(provider)])
    if toolsets:
        args.extend(["--toolsets", ",".join(toolsets)])
    return args


def build_env(settings: dict[str, Any], hermes_home: Path) -> dict[str, str]:
    env = os.environ.copy()

    # If we are in a virtual env those values take precedence over the current environment variables
    virtual_env_vars = common_utils.virtualenv_subprocess_env()
    if len(virtual_env_vars) > 0:
        env.pop("PYTHONHOME", None)
        env.update(virtual_env_vars)

    env.update({
        str(key): str(value)
        for key, value in (settings.get("env") or {}).items()
    })
    env["HOME"] = str(hermes_home)
    env["HERMES_HOME"] = str(hermes_home)
    env.setdefault("HERMES_YOLO_MODE", "1")
    env.setdefault("HERMES_ACCEPT_HOOKS", "1")

    return env


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
        if arg in {"-z", "--oneshot", "--query"}:
            redact_next = True
    return redacted


if __name__ == "__main__":
    main()
