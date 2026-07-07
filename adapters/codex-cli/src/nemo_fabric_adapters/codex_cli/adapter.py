#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map Fabric one-shot and session invocations onto ``codex exec``."""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import nemo_fabric_adapters.common.utils as common_utils
import tomli_w

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

SANDBOXES = {"read-only", "workspace-write", "danger-full-access"}
DEFAULT_TIMEOUT_SECONDS = 1800
RELAY_HEALTH_TIMEOUT_SECONDS = 30
RELAY_HOOK_EVENTS = (
    "Notification",
    "PermissionRequest",
    "PostCompact",
    "PostToolUse",
    "PostToolUseFailure",
    "PreCompact",
    "PreToolUse",
    "SessionEnd",
    "SessionStart",
    "Stop",
    "SubagentStart",
    "SubagentStop",
    "UserPromptSubmit",
)
RELAY_HOOK_MATCHER_EVENTS = {
    "PermissionRequest",
    "PostToolUse",
    "PostToolUseFailure",
    "PreToolUse",
}
INHERITED_ENV_NAMES = {
    "APPDATA",
    "CODEX_HOME",
    "COMSPEC",
    "HOME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOCALAPPDATA",
    "NO_PROXY",
    "PATH",
    "PATHEXT",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USERPROFILE",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}


class CodexSettings(NamedTuple):
    telemetry_provider: str
    relay_enabled: bool
    codex_profile_name: str | None
    codex_profile_path: Path | None
    relay_gateway_host: str | None
    relay_gateway_url: str | None
    relay_gateway_port: int | None
    relay_config_path: Path | None
    relay_plugin_config: dict[str, Any] | None


def runtime_mode(payload: dict[str, Any]) -> str:
    runtime = common_utils.fabric_config(payload).get("runtime") or {}
    mode = str(runtime.get("mode") or "oneshot")
    if mode not in {"oneshot", "session"}:
        raise ValueError("Codex CLI adapter supports only oneshot and session modes")
    return mode


def state_dir(payload: dict[str, Any]) -> Path:
    settings = common_utils.settings_payload(payload)
    config_root = Path(common_utils.config_root(payload)).resolve()
    configured = settings.get("codex_state_dir")
    if configured:
        path = Path(str(configured))
        return path if path.is_absolute() else config_root / path
    artifacts = common_utils.runtime_context(payload).get("artifacts") or {}
    root = artifacts.get("root") or os.environ.get("FABRIC_ARTIFACTS")
    if root:
        return Path(str(root)).resolve() / ".fabric" / "codex-cli"
    return config_root / "artifacts" / "codex-cli" / ".fabric"


def session_state_path(payload: dict[str, Any], session_id: str) -> Path:
    key = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    return state_dir(payload) / "sessions" / f"{key}.json"


def load_thread_id(payload: dict[str, Any], session_id: str) -> str | None:
    path = session_state_path(payload, session_id)
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("session_id") != session_id or not value.get("thread_id"):
        raise RuntimeError(f"invalid Codex session state in {path}")
    return str(value["thread_id"])


def save_thread_id(payload: dict[str, Any], session_id: str, thread_id: str) -> None:
    path = session_state_path(payload, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    invocation_id = common_utils.runtime_context(payload).get("invocation_id") or "pending"
    temporary = path.with_suffix(f".{invocation_id}.tmp")
    temporary.write_text(
        json.dumps({"session_id": session_id, "thread_id": thread_id}, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def selected_model(payload: dict[str, Any]) -> str | None:
    settings = common_utils.settings_payload(payload)
    models = common_utils.models_payload(payload)
    model_config = models.get(settings.get("model", "default"), {})
    value = settings.get("model_name")
    if not value and isinstance(model_config, dict):
        value = model_config.get("model")
    if not value:
        return None
    model = str(value)
    return model.removeprefix("openai/")


def build_command(
    payload: dict[str, Any],
    *,
    thread_id: str | None = None,
    codex_settings: CodexSettings,
) -> list[str]:
    settings = common_utils.settings_payload(payload)
    command = resolve_command(payload, settings.get("codex_command") or "codex")
    mode = runtime_mode(payload)
    sandbox = str(settings.get("sandbox") or "read-only")
    if sandbox not in SANDBOXES:
        raise ValueError(
            f"unsupported Codex sandbox {sandbox!r}; expected one of {sorted(SANDBOXES)}"
        )

    args = [command, "exec", "--json"]

    if mode == "oneshot":
        args.append("--ephemeral")
    args.extend(["--sandbox", sandbox])

    if codex_settings.codex_profile_name is not None:
        args.extend(("--profile", codex_settings.codex_profile_name))

        # relay_enabled will only ever be true when codex_profile_name is not None
        if codex_settings.relay_enabled:
            # By default Codex will not enable hooks for profiles that are not trusted until the user explicitly
            # enables them. This is a problem for Fabric, because we want to be able to use hooks in a non-interactive
            # way.So we add the --dangerously-bypass-hook-trust flag to bypass this check.
            args.append("--dangerously-bypass-hook-trust")

    model = selected_model(payload)
    if model:
        args.extend(["--model", model])
    if settings.get("skip_git_repo_check", False):
        args.append("--skip-git-repo-check")
    args.extend(common_utils.normalize_list(settings.get("codex_args")))

    if thread_id:
        args.extend(["resume", thread_id, "-"])
    else:
        args.append("-")

    return args


def config_overrides(settings: dict[str, Any]) -> Mapping[str, Any]:
    overrides = settings.get("config_overrides")
    if overrides is None:
        return {}
    if not isinstance(overrides, Mapping):
        raise ValueError("config_overrides must be a mapping")
    return overrides


def native_codex_telemetry_config(payload: dict[str, Any]) -> dict[str, Any]:
    telemetry = common_utils.telemetry_payload(payload)
    if not telemetry.get("enabled") or common_utils.telemetry_provider(payload) != "native":
        return {}

    telemetry_config = telemetry.get("config") or {}
    components = telemetry_config.get("components") or []
    for component in components:
        if (
            not isinstance(component, dict)
            or component.get("kind") != "observability"
            or not component.get("enabled", True)
        ):
            continue
        component_config = component.get("config") or {}
        opentelemetry = component_config.get("opentelemetry") or {}
        if not isinstance(opentelemetry, dict) or not opentelemetry.get("enabled"):
            continue

        otel: dict[str, Any] = {}
        resource_attributes = opentelemetry.get("resource_attributes") or {}
        environment = resource_attributes.get("deployment.environment")
        if environment is not None:
            otel["environment"] = environment

        endpoint = opentelemetry.get("endpoint")
        if endpoint:
            transport = opentelemetry.get("transport", "http_binary")
            if transport == "http_binary":
                exporter = "otlp-http"
                protocol = "binary"
            elif transport == "http_json":
                exporter = "otlp-http"
                protocol = "json"
            else:
                raise ValueError(
                    f"unsupported Codex native OpenTelemetry transport {transport!r}"
                )
            otel["trace_exporter"] = {
                exporter: {
                    "endpoint": endpoint,
                    "protocol": protocol,
                }
            }
        return {"otel": otel}
    return {}


def apply_config_overrides(
    config: dict[str, Any],
    overrides: Mapping[str, Any],
) -> None:
    for dotted_key, value in sorted(overrides.items()):
        toml_value(value)
        parts = str(dotted_key).split(".")
        if any(not part for part in parts):
            raise ValueError(f"invalid Codex config override key {dotted_key!r}")
        target = config
        for part in parts[:-1]:
            existing = target.setdefault(part, {})
            if not isinstance(existing, dict):
                raise ValueError(
                    f"Codex config override {dotted_key!r} conflicts with {part!r}"
                )
            target = existing
        target[parts[-1]] = value


def merge_config(config: dict[str, Any], layer: Mapping[str, Any]) -> None:
    for key, value in layer.items():
        existing = config.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merge_config(existing, value)
        else:
            config[key] = value


def codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".codex"


def load_codex_profile(settings: dict[str, Any]) -> dict[str, Any]:
    profile = settings.get("codex_profile")
    if not profile:
        return {}
    path = codex_home() / f"{profile}.toml"
    with path.open("rb") as profile_file:
        return tomllib.load(profile_file)


def write_config_files(payload: dict[str, Any]) -> CodexSettings:
    settings = common_utils.settings_payload(payload)
    telemetry_provider = common_utils.telemetry_provider(payload)
    relay_enabled = (
        telemetry_provider == "relay"
        and os.environ.get("FABRIC_RELAY_ENABLED") == "true"
    )
    overrides = config_overrides(settings)
    config = load_codex_profile(settings)
    if telemetry_provider == "native":
        merge_config(config, native_codex_telemetry_config(payload))

    codex_profile_name = None
    codex_profile_path = None
    relay_gateway_host = None
    relay_gateway_url = None
    relay_gateway_port = None
    relay_config_path = None
    relay_plugin_config = None
    if relay_enabled or bool(config) or bool(overrides):
        codex_profile_name, codex_profile_path = get_codex_profile_path(payload)

        if relay_enabled:
            relay_gateway_port = find_available_tcp_port()
            relay_gateway_host = f"127.0.0.1:{relay_gateway_port}"
            relay_gateway_url = f"http://{relay_gateway_host}"

            # nemo-relay infers the plugin config location from the relay config.
            relay_plugin_config = common_utils.load_relay_plugin_config(payload)
            relay_config_path, _ = common_utils.write_relay_configs(
                relay_config={"agents": {"codex": {"command": "codex"}}},
                plugin_config=relay_plugin_config,
            )
            if relay_config_path is None:
                raise RuntimeError(
                    "NeMo Relay configuration did not produce a gateway config"
                )

            relay_command = resolve_command(
                payload,
                settings.get("nemo_relay_command") or "nemo-relay",
            )
            hook_command = f"{relay_command} hook-forward codex"
            hooks = {}
            for event in RELAY_HOOK_EVENTS:
                hook_group: dict[str, Any] = {
                    "hooks": [
                        {
                            "type": "command",
                            "command": hook_command,
                            "timeout": 30,
                        }
                    ]
                }
                if event in RELAY_HOOK_MATCHER_EVENTS:
                    hook_group["matcher"] = "*"
                hooks[event] = [hook_group]

            merge_config(
                config,
                {
                    "model_provider": "nemo-relay-openai",
                    "model_providers": {
                        "nemo-relay-openai": {
                            "name": "NeMo Relay OpenAI",
                            "base_url": relay_gateway_url,
                            "wire_api": "responses",
                            "requires_openai_auth": True,
                            "supports_websockets": False,
                        }
                    },
                    "features": {"hooks": True},
                    "hooks": hooks,
                }
            )

        apply_config_overrides(config, overrides)
        codex_profile_path.parent.mkdir(parents=True, exist_ok=True)
        codex_profile_path.write_text(
            tomli_w.dumps(config),
            encoding="utf-8",
        )

    return CodexSettings(
        telemetry_provider=telemetry_provider,
        relay_enabled=relay_enabled,
        codex_profile_name=codex_profile_name,
        codex_profile_path=codex_profile_path,
        relay_gateway_host=relay_gateway_host,
        relay_gateway_url=relay_gateway_url,
        relay_gateway_port=relay_gateway_port,
        relay_config_path=relay_config_path,
        relay_plugin_config=relay_plugin_config,
    )


def get_codex_profile_path(payload: dict[str, Any]) -> tuple[str, Path]:
    runtime_id = common_utils.runtime_context(payload).get("runtime_id")
    if not runtime_id:
        raise RuntimeError(
            "runtime_context.runtime_id is required for generated Codex profiles"
        )

    name = f"fabric-{runtime_id}"
    return name, codex_home() / f"{name}.config.toml"


def resolve_command(payload: dict[str, Any], value: Any) -> str:
    command = Path(str(value))
    if command.is_absolute() or len(command.parts) == 1:
        return str(command)
    config_root = Path(common_utils.config_root(payload)).resolve()
    return str((config_root / command).resolve())


def toml_value(value: Any) -> str:
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Codex config overrides require finite numbers")
        if isinstance(item, Mapping):
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
    try:
        document = tomli_w.dumps({"value": value})
    except TypeError as error:
        raise ValueError(
            "Codex config override values must be a TOML scalar or array"
        ) from error
    prefix = "value = "
    if not document.startswith(prefix):
        raise ValueError("Codex config override values must be a TOML scalar or array")
    return document.removeprefix(prefix).rstrip()


def parse_events(contents: str) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    thread_id = None
    response = None
    usage = None
    error = None
    for line in contents.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        events.append(event)
        event_type = event.get("type")
        if event_type == "thread.started":
            thread_id = event.get("thread_id")
        elif event_type == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message":
                response = item.get("text")
        elif event_type == "turn.completed":
            usage = event.get("usage")
        elif event_type in {"turn.failed", "error"}:
            failure = event.get("error") or event.get("message") or event
            error = failure.get("message") if isinstance(failure, dict) else str(failure)
    return {
        "events": events,
        "thread_id": str(thread_id) if thread_id else None,
        "response": response,
        "usage": usage,
        "error": error,
    }


def request_to_prompt(payload: dict[str, Any]) -> str:
    value = (payload.get("request") or {}).get("input", "")
    if not isinstance(value, str):
        raise ValueError("Codex CLI adapter requires text input")
    return value


def resolve_cwd(payload: dict[str, Any]) -> Path:
    settings = common_utils.settings_payload(payload)
    environment = common_utils.environment_payload(payload)
    config_root = Path(common_utils.config_root(payload)).resolve()
    path = Path(str(settings.get("cwd") or environment.get("workspace") or "."))
    return path.resolve() if path.is_absolute() else (config_root / path).resolve()


def build_env(
    payload: dict[str, Any],
    *,
    relay_gateway_url: str | None = None,
) -> dict[str, str]:
    env = {name: os.environ[name] for name in INHERITED_ENV_NAMES if name in os.environ}
    configured = common_utils.settings_payload(payload).get("env")
    if configured is None:
        configured = {}
    if not isinstance(configured, Mapping):
        raise ValueError("env must be a mapping of variable names to values")
    env.update({str(key): str(value) for key, value in configured.items()})
    if relay_gateway_url is not None:
        env["NEMO_RELAY_GATEWAY_URL"] = relay_gateway_url
    return env


def find_available_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def wait_for_relay_gateway(
    process: subprocess.Popen[Any],
    health_url: str,
    *,
    timeout: float = RELAY_HEALTH_TIMEOUT_SECONDS,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"NeMo Relay gateway exited with status {returncode} before becoming ready"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=1) as response:
                if 200 <= response.status < 300:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.1)
    raise RuntimeError(f"NeMo Relay gateway did not become ready at {health_url}")


def stop_relay_gateway(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def start_relay_gateway(
    payload: dict[str, Any],
    cwd: Path,
    codex_settings: CodexSettings,
) -> subprocess.Popen:
    settings = common_utils.settings_payload(payload)
    relay_command = resolve_command(
        payload,
        settings.get("nemo_relay_command") or "nemo-relay",
    )

    process = subprocess.Popen(
        [
            relay_command,
            "--config",
            str(codex_settings.relay_config_path),
            "--bind",
            codex_settings.relay_gateway_host,
        ],
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        wait_for_relay_gateway(process, f"{codex_settings.relay_gateway_url}/healthz")
    except Exception as e:
        stop_relay_gateway(process)
        raise RuntimeError("NeMo Relay gateway failed to start") from e

    return process


def process_timeout(payload: dict[str, Any]) -> float:
    value = common_utils.settings_payload(payload).get(
        "timeout_seconds", DEFAULT_TIMEOUT_SECONDS
    )
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("timeout_seconds must be a positive finite number")
    return float(value)


def exception_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def run_codex(payload: dict[str, Any]) -> dict[str, Any]:
    mode = runtime_mode(payload)
    session_id = common_utils.runtime_session_id(payload) if mode == "session" else None
    if mode == "session" and not session_id:
        raise RuntimeError("runtime.mode=session requires a session_id or runtime_id")
    prior_thread_id = load_thread_id(payload, session_id) if session_id else None
    cwd = resolve_cwd(payload)
    codex_settings = write_config_files(payload)
    relay_gateway_process = None

    try:
        if codex_settings.relay_enabled:
            if codex_settings.relay_config_path is None or codex_settings.relay_gateway_port is None:
                raise RuntimeError("NeMo Relay configuration files were not generated")

            relay_gateway_process = start_relay_gateway(
                payload,
                cwd,
                codex_settings,
            )

        command = build_command(
            payload,
            thread_id=prior_thread_id,
            codex_settings=codex_settings,
        )

        timeout = process_timeout(payload)
        launch_error = None
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=build_env(
                    payload,
                    relay_gateway_url=codex_settings.relay_gateway_url,
                ),
                input=request_to_prompt(payload),
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            completed = subprocess.CompletedProcess(
                command,
                124,
                exception_output(error.stdout),
                exception_output(error.stderr),
            )
            launch_error = f"Codex CLI timed out after {timeout:g} seconds"
        except OSError as error:
            completed = subprocess.CompletedProcess(command, 127, "", str(error))
            launch_error = f"Codex CLI could not start: {error}"
    finally:
        if codex_settings.codex_profile_path is not None:
            codex_settings.codex_profile_path.unlink(missing_ok=True)

        if relay_gateway_process is not None:
            stop_relay_gateway(relay_gateway_process)

    parsed = parse_events(completed.stdout)
    thread_id = parsed["thread_id"] or prior_thread_id
    error = launch_error or parsed["error"]
    if completed.returncode != 0:
        error = error or completed.stderr.strip() or "Codex CLI exited with a non-zero status"
    if parsed["response"] is None:
        error = error or "Codex invocation did not return a final agent message"
    if session_id and not thread_id:
        error = error or "Codex session invocation did not return a thread identity"
    if session_id and prior_thread_id and thread_id != prior_thread_id:
        error = (
            f"Codex resumed thread {thread_id}, expected persisted thread {prior_thread_id}"
        )
    if session_id and thread_id and not error:
        save_thread_id(payload, session_id, thread_id)

    output = {
        "harness": "codex",
        "adapter": "cli",
        "mode": f"codex_cli_{mode}",
        "command": redact_command(command),
        "cwd": str(cwd),
        "model": selected_model(payload),
        "session_id": session_id,
        "thread_id": thread_id,
        "response": parsed["response"],
        "usage": parsed["usage"],
        "returncode": completed.returncode,
        "error": error,
        "failed": error is not None,
        "state_dir": str(state_dir(payload)),
    }

    if codex_settings.relay_plugin_config is not None:
        relay_artifacts = common_utils.collect_relay_artifacts(
            codex_settings.relay_plugin_config
        )
        output["relay_runtime"] = {
            "enabled": True,
            "mode": os.environ.get("FABRIC_RELAY_MODE"),
            "config_path": os.environ.get("FABRIC_RELAY_CONFIG_PATH"),
            "emitter": "nemo-relay",
        }
        output["relay_artifacts"] = relay_artifacts

    return output


def redact_command(command: list[str]) -> list[str]:
    redacted = list(command)
    for index, value in enumerate(redacted[:-1]):
        if value == "--config" and any(
            marker in redacted[index + 1].lower()
            for marker in ("key", "token", "secret", "password")
        ):
            redacted[index + 1] = "<redacted>"
    return redacted


def main() -> None:
    output = run_codex(common_utils.load_payload())
    print(json.dumps(output, sort_keys=True))
    if output["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
