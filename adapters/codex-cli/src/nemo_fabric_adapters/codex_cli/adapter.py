#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map Fabric one-shot and session invocations onto ``codex exec``."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import tomli_w

CUR_DIR = Path(__file__).parent
ADAPTERS_DIR = CUR_DIR.parent.parent.parent.parent
COMMON_DIR = (ADAPTERS_DIR / "common/src").resolve().as_posix()
if COMMON_DIR not in sys.path:
    sys.path.append(COMMON_DIR)

import nemo_fabric_adapters.common.utils as common_utils  # noqa: E402

SANDBOXES = {"read-only", "workspace-write", "danger-full-access"}
DEFAULT_TIMEOUT_SECONDS = 1800
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


def load_payload() -> dict[str, Any]:
    invocation_path = os.environ.get("FABRIC_INVOCATION")
    if invocation_path and Path(invocation_path).is_file():
        return json.loads(Path(invocation_path).read_text(encoding="utf-8"))
    return json.load(sys.stdin)


def runtime_mode(payload: dict[str, Any]) -> str:
    runtime = common_utils.fabric_config(payload).get("runtime") or {}
    mode = str(runtime.get("mode") or "oneshot")
    if mode not in {"oneshot", "session"}:
        raise ValueError("Codex CLI adapter supports only oneshot and session modes")
    return mode


def fabric_session_id(payload: dict[str, Any]) -> str | None:
    context = common_utils.runtime_context(payload)
    value = context.get("session_id") or context.get("runtime_id")
    return str(value) if value else None


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
    payload: dict[str, Any], *, thread_id: str | None = None
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

    codex_profile = settings.get("codex_profile")
    if codex_profile:
        args.extend(["--profile", str(codex_profile)])
    for key, value in sorted((settings.get("config_overrides") or {}).items()):
        args.extend(["--config", f"{key}={toml_value(value)}"])
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


def resolve_command(payload: dict[str, Any], value: Any) -> str:
    command = Path(str(value))
    if command.is_absolute() or len(command.parts) == 1:
        return str(command)
    config_root = Path(common_utils.config_root(payload)).resolve()
    return str((config_root / command).resolve())


def toml_value(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Codex config overrides require finite numbers")
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


def build_env(payload: dict[str, Any]) -> dict[str, str]:
    env = {name: os.environ[name] for name in INHERITED_ENV_NAMES if name in os.environ}
    configured = common_utils.settings_payload(payload).get("env") or {}
    env.update({str(key): str(value) for key, value in configured.items()})
    return env


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
    session_id = fabric_session_id(payload) if mode == "session" else None
    if mode == "session" and not session_id:
        raise RuntimeError("runtime.mode=session requires a session_id or runtime_id")
    prior_thread_id = load_thread_id(payload, session_id) if session_id else None
    command = build_command(payload, thread_id=prior_thread_id)
    cwd = resolve_cwd(payload)
    timeout = process_timeout(payload)
    launch_error = None
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=build_env(payload),
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

    return {
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
    output = run_codex(load_payload())
    print(json.dumps(output, sort_keys=True))
    if output["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
