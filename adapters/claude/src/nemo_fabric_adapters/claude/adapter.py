# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Claude Agent SDK through the Fabric adapter process contract."""

from __future__ import annotations

import asyncio
import json
import math
import os
import shlex
import shutil
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ClaudeAgentOptions,
    ClaudeSDKError,
    Message,
    ProcessError,
    ResultMessage,
    query,
)
from claude_agent_sdk._errors import MessageParseError

import nemo_fabric_adapters.common.utils as common_utils


PERMISSION_MODES = {"default", "acceptEdits", "bypassPermissions", "plan", "dontAsk", "auto"}
SETTING_SOURCES = {"user", "project", "local"}
NORMALIZED_SETTING_FIELDS = {
    "model_name": "FabricConfig.models",
    "cwd": "FabricConfig.environment.workspace",
    "tools": "FabricConfig.tools",
    "mcp_servers": "FabricConfig.mcp",
    "skills": "FabricConfig.skills",
}
INHERITED_ENV_NAMES = {
    "ANTHROPIC_API_KEY",
    "APPDATA",
    "CLAUDE_CONFIG_DIR",
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


class ClaudeAdapterError(Exception):
    """Expected adapter error with a stable public code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class AdapterInputError(ClaudeAdapterError):
    """Invalid Fabric invocation input."""


class AdapterConfigError(ClaudeAdapterError):
    """Invalid Claude adapter configuration."""


class AdapterStateError(ClaudeAdapterError):
    """Invalid persisted runtime state."""


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise AdapterConfigError("claude_invalid_configuration", f"{name} must be a mapping")
    return value


def _string_list(value: Any, *, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise AdapterConfigError(
            "claude_invalid_configuration", f"{name} must be a list of non-empty strings"
        )
    return list(value)


def _positive_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterConfigError("claude_invalid_configuration", f"{name} must be positive")
    number = float(value)
    if number <= 0 or not math.isfinite(number):
        raise AdapterConfigError("claude_invalid_configuration", f"{name} must be positive")
    return number


def runtime_id(payload: dict[str, Any]) -> str:
    value = common_utils.runtime_context(payload).get("runtime_id")
    if not isinstance(value, str) or not value:
        raise AdapterInputError("claude_invalid_request", "Fabric runtime ID is required")
    return value


def request_prompt(payload: dict[str, Any]) -> str:
    request = payload.get("request") or {}
    value = request.get("input")
    if not isinstance(value, str):
        raise AdapterInputError("claude_invalid_request", "Claude input must be text")
    return value


def _settings(payload: dict[str, Any]) -> dict[str, Any]:
    return _mapping(common_utils.settings_payload(payload), name="harness.settings")


def _validate_settings_boundary(settings: dict[str, Any]) -> None:
    for name, normalized_field in NORMALIZED_SETTING_FIELDS.items():
        if name in settings:
            raise AdapterConfigError(
                "claude_invalid_configuration",
                f"harness.settings.{name} is not supported; use {normalized_field}",
            )


def _models(payload: dict[str, Any]) -> dict[str, Any]:
    return _mapping(common_utils.models_payload(payload), name="models")


def _selected_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    models = _models(payload)
    if not models:
        return {}
    selected = models.get("default") or next(iter(models.values()))
    return _mapping(selected, name="selected model")


def _resolve_path(payload: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(common_utils.config_root(payload)) / path
    return path


def resolve_cwd(payload: dict[str, Any]) -> Path:
    environment = common_utils.environment_payload(payload)
    workspace = environment.get("workspace")
    return _resolve_path(payload, workspace or common_utils.config_root(payload))


def selected_model(payload: dict[str, Any]) -> str | None:
    model_config = _selected_model_config(payload)
    value = model_config.get("model")
    if value is None:
        return None
    if model_config.get("provider") != "anthropic":
        raise AdapterConfigError(
            "claude_invalid_configuration",
            "models.default.provider must be anthropic for the Claude adapter",
        )
    if not isinstance(value, str) or not value:
        raise AdapterConfigError("claude_invalid_configuration", "model must be a non-empty string")
    return value.removeprefix("anthropic/")


def _mcp_servers(payload: dict[str, Any]) -> dict[str, Any]:
    native = _mapping(
        common_utils.capability_plan(payload), name="capability_plan"
    ).get("native") or {}
    servers = _mapping(native, name="capability_plan.native").get("mcp_servers") or {}
    result: dict[str, Any] = {}
    for name, raw in sorted(_mapping(servers, name="native MCP servers").items()):
        server = _mapping(raw, name=f"MCP server {name}")
        transport = server.get("transport")
        url = server.get("url")
        if not isinstance(url, str) or not url:
            raise AdapterConfigError("claude_invalid_configuration", "MCP server URL is required")
        if transport == "stdio":
            command = shlex.split(url)
            if not command:
                raise AdapterConfigError("claude_invalid_configuration", "MCP command is required")
            result[name] = {"type": "stdio", "command": command[0], "args": command[1:]}
        elif transport in {"http", "streamable-http"}:
            result[name] = {"type": "http", "url": url}
        elif transport == "sse":
            result[name] = {"type": "sse", "url": url}
        else:
            raise AdapterConfigError(
                "claude_invalid_configuration", f"unsupported MCP transport: {transport}"
            )
    return result


def _normalized_tools(
    payload: dict[str, Any], *, include_skills: bool
) -> list[str] | dict[str, Any] | None:
    native = _mapping(
        common_utils.capability_plan(payload), name="capability_plan"
    ).get("native") or {}
    if not _mapping(native, name="capability_plan.native").get("tools_configured"):
        return None
    tools = common_utils.fabric_config(payload).get("tools")
    if tools is not None and not isinstance(tools, (list, dict)):
        raise AdapterConfigError("claude_invalid_configuration", "tools is invalid")
    if isinstance(tools, list):
        normalized = _string_list(tools, name="tools")
        if include_skills and "Skill" not in normalized:
            normalized.append("Skill")
        return normalized
    if isinstance(tools, dict) and tools != {"type": "preset", "preset": "claude_code"}:
        raise AdapterConfigError(
            "claude_invalid_configuration",
            "tools preset must be {'type': 'preset', 'preset': 'claude_code'}",
        )
    return tools


def _native_skill_paths(payload: dict[str, Any]) -> list[Path]:
    native = _mapping(
        common_utils.capability_plan(payload), name="capability_plan"
    ).get("native") or {}
    values = _mapping(native, name="capability_plan.native").get("skill_paths") or []
    if not isinstance(values, list) or any(
        not isinstance(value, (str, Path)) for value in values
    ):
        raise AdapterConfigError(
            "claude_invalid_configuration", "native skill_paths must be a list of paths"
        )
    return [_resolve_path(payload, value) for value in values]


def _stage_skill_plugin(payload: dict[str, Any]) -> list[dict[str, str]]:
    skill_paths = _native_skill_paths(payload)
    if not skill_paths:
        return []

    skills: list[tuple[str, Path]] = []
    names: set[str] = set()
    for skill_path in skill_paths:
        if not skill_path.is_dir() or not (skill_path / "SKILL.md").is_file():
            raise AdapterConfigError(
                "claude_invalid_configuration",
                f"Fabric skill path must be a directory containing SKILL.md: {skill_path}",
            )
        name = skill_path.name
        if name in names:
            raise AdapterConfigError(
                "claude_invalid_configuration", f"Fabric skill names must be unique: {name}"
            )
        names.add(name)
        skills.append((name, skill_path))

    plugin_key = sha256(runtime_id(payload).encode()).hexdigest()
    plugin_root = _artifact_root(payload) / ".fabric" / "claude" / "plugins" / plugin_key
    if plugin_root.exists():
        shutil.rmtree(plugin_root)
    (plugin_root / ".claude-plugin").mkdir(parents=True)
    (plugin_root / "skills").mkdir()
    (plugin_root / ".claude-plugin" / "plugin.json").write_text(
        json.dumps(
            {
                "name": "nemo-fabric-skills",
                "description": "Skills provided by NeMo Fabric",
                "version": "1.0.0",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for name, skill_path in skills:
        shutil.copytree(skill_path, plugin_root / "skills" / name)
    return [{"type": "local", "path": str(plugin_root)}]


def discard_stderr(_: str) -> None:
    """Consume Claude Code stderr without exposing it through Fabric artifacts."""


def build_options(payload: dict[str, Any], *, resume: str | None) -> ClaudeAgentOptions:
    settings = _settings(payload)
    _validate_settings_boundary(settings)
    permission_mode = settings.get("permission_mode")
    if permission_mode is not None and permission_mode not in PERMISSION_MODES:
        raise AdapterConfigError("claude_invalid_configuration", "permission_mode is invalid")
    max_turns = settings.get("max_turns")
    if max_turns is not None and (
        isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns <= 0
    ):
        raise AdapterConfigError("claude_invalid_configuration", "max_turns must be positive")
    max_budget = settings.get("max_budget_usd")
    if max_budget is not None:
        max_budget = _positive_number(max_budget, name="max_budget_usd")
    sources = settings.get("setting_sources", [])
    sources = _string_list(sources, name="setting_sources")
    if any(source not in SETTING_SOURCES for source in sources):
        raise AdapterConfigError("claude_invalid_configuration", "setting_sources is invalid")
    cli_path = settings.get("cli_path")
    if cli_path is not None and not isinstance(cli_path, (str, Path)):
        raise AdapterConfigError("claude_invalid_configuration", "cli_path must be a path")

    system_prompt = settings.get("system_prompt")
    if system_prompt is not None and not isinstance(system_prompt, (str, dict)):
        raise AdapterConfigError("claude_invalid_configuration", "system_prompt is invalid")
    plugins = _stage_skill_plugin(payload)

    return ClaudeAgentOptions(
        resume=resume,
        cwd=resolve_cwd(payload),
        model=selected_model(payload),
        system_prompt=system_prompt,
        tools=_normalized_tools(payload, include_skills=bool(plugins)),
        allowed_tools=_string_list(settings.get("allowed_tools"), name="allowed_tools"),
        disallowed_tools=_string_list(settings.get("disallowed_tools"), name="disallowed_tools"),
        permission_mode=permission_mode,
        max_turns=max_turns,
        max_budget_usd=max_budget,
        setting_sources=sources,
        cli_path=_resolve_path(payload, cli_path) if cli_path is not None else None,
        mcp_servers=_mcp_servers(payload),
        strict_mcp_config=True,
        skills="all" if plugins else None,
        plugins=plugins,
        env=child_environment(payload),
        stderr=discard_stderr,
    )


def timeout_seconds(payload: dict[str, Any]) -> float:
    value = _settings(payload).get("timeout_seconds", 1800)
    return _positive_number(value, name="timeout_seconds")


def _artifact_root(payload: dict[str, Any]) -> Path:
    artifacts = common_utils.runtime_context(payload).get("artifacts") or {}
    root = artifacts.get("root") if isinstance(artifacts, dict) else None
    if root:
        return Path(root)
    return Path(common_utils.config_root(payload)) / "artifacts" / "claude"


def runtime_state_path(payload: dict[str, Any], fabric_runtime_id: str) -> Path:
    digest = sha256(fabric_runtime_id.encode("utf-8")).hexdigest()
    return _artifact_root(payload) / ".fabric" / "claude" / "runtimes" / f"{digest}.json"


def load_claude_session_id(payload: dict[str, Any], fabric_runtime_id: str) -> str | None:
    path = runtime_state_path(payload, fabric_runtime_id)
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError("state must be an object")
        if state.get("runtime_id") != fabric_runtime_id:
            raise ValueError("runtime mismatch")
        session_id = state.get("claude_session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("missing Claude session")
        return session_id
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise AdapterStateError(
            "claude_invalid_runtime_state", "Claude runtime state is invalid"
        ) from error


def save_claude_session_id(
    payload: dict[str, Any], fabric_runtime_id: str, claude_session_id: str
) -> None:
    if not claude_session_id:
        raise AdapterStateError("claude_invalid_runtime_state", "Claude session ID is missing")
    path = runtime_state_path(payload, fabric_runtime_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    invocation_id = common_utils.runtime_context(payload).get("invocation_id") or "invocation"
    temporary = path.with_suffix(f".{invocation_id}.tmp")
    temporary.write_text(
        json.dumps(
            {"runtime_id": fabric_runtime_id, "claude_session_id": claude_session_id},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise AdapterConfigError("claude_invalid_configuration", "Claude message is not JSON-safe")


def normalize_message(message: Message) -> dict[str, Any]:
    return {"type": type(message).__name__, "message": _json_safe(message)}


def normalize_result(
    payload: dict[str, Any], messages: list[Message], result: ResultMessage
) -> dict[str, Any]:
    del payload
    failed = bool(result.is_error) or (
        isinstance(result.subtype, str) and result.subtype.startswith("error_")
    )
    error = None
    if failed:
        error = {
            "code": "claude_result_failed",
            "message": "Claude returned an error result",
            "retryable": False,
            "metadata": {"subtype": result.subtype},
        }
    return {
        "harness": "claude",
        "adapter": "sdk",
        "response": result.result,
        "session_id": result.session_id,
        "usage": _json_safe(result.usage or {}),
        "model_usage": _json_safe(result.model_usage or {}),
        "cost_usd": result.total_cost_usd,
        "duration_ms": result.duration_ms,
        "duration_api_ms": result.duration_api_ms,
        "num_turns": result.num_turns,
        "stop_reason": result.stop_reason,
        "subtype": result.subtype,
        "completed": not failed,
        "failed": failed,
        "error": error,
        "events": [normalize_message(message) for message in messages],
    }


def _failure(code: str, message: str, **metadata: Any) -> dict[str, Any]:
    error = {"code": code, "message": message, "retryable": False}
    if metadata:
        error["metadata"] = metadata
    return {
        "harness": "claude",
        "adapter": "sdk",
        "response": None,
        "completed": False,
        "failed": True,
        "error": error,
        "events": [],
    }


def adapter_failure(error: ClaudeAdapterError) -> dict[str, Any]:
    return _failure(error.code, error.message)


def sdk_failure(error: BaseException) -> dict[str, Any]:
    if isinstance(error, TimeoutError):
        return _failure("claude_timed_out", "Claude invocation timed out")
    if isinstance(error, CLINotFoundError):
        return _failure("claude_cli_not_found", "Claude Code executable was not found")
    if isinstance(error, CLIConnectionError):
        return _failure("claude_connection_failed", "Claude Code connection failed")
    if isinstance(error, ProcessError):
        return _failure(
            "claude_process_failed",
            "Claude Code process failed",
            exit_code=error.exit_code,
        )
    if isinstance(error, CLIJSONDecodeError):
        return _failure("claude_invalid_json", "Claude Code returned invalid JSON")
    if isinstance(error, MessageParseError):
        return _failure("claude_message_parse_failed", "Claude message parsing failed")
    return _failure("claude_failed", "Claude invocation failed")


def child_environment(payload: dict[str, Any]) -> dict[str, str]:
    values = {name: "" for name in os.environ}
    values.update(
        {
            name: value
            for name in INHERITED_ENV_NAMES
            if (value := os.environ.get(name))
        }
    )
    model = _selected_model_config(payload)
    api_key_env = model.get("api_key_env")
    if isinstance(api_key_env, str) and api_key_env in os.environ:
        values[api_key_env] = os.environ[api_key_env]
    configured = _mapping(_settings(payload).get("env"), name="harness.settings.env")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in configured.items()):
        raise AdapterConfigError(
            "claude_invalid_configuration", "harness.settings.env must contain strings"
        )
    values.update(configured)
    return values


async def run_claude(payload: dict[str, Any]) -> dict[str, Any]:
    fabric_runtime_id = runtime_id(payload)
    prior_session_id = load_claude_session_id(payload, fabric_runtime_id)
    options = build_options(payload, resume=prior_session_id)
    messages: list[Message] = []
    result: ResultMessage | None = None
    try:
        async with asyncio.timeout(timeout_seconds(payload)):
            async for message in query(prompt=request_prompt(payload), options=options):
                if isinstance(message, ResultMessage):
                    result = message
                else:
                    messages.append(message)
    except (TimeoutError, ClaudeSDKError) as error:
        return sdk_failure(error)

    if result is None:
        return _failure("claude_missing_result", "Claude returned no terminal result")
    output = normalize_result(payload, messages, result)
    if output["failed"]:
        return output
    if prior_session_id is not None and result.session_id != prior_session_id:
        return _failure("claude_session_mismatch", "Claude session identity changed during resume")
    save_claude_session_id(payload, fabric_runtime_id, result.session_id)
    return output


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one Fabric invocation."""

    try:
        return asyncio.run(run_claude(payload))
    except ClaudeAdapterError as error:
        return adapter_failure(error)
    except Exception:  # Adapter boundary must always return normalized JSON.
        return _failure(
            "claude_adapter_internal_error", "Claude adapter failed unexpectedly"
        )


def main() -> None:
    try:
        payload = common_utils.load_payload()
    except Exception:  # Malformed invocation input must still satisfy the process contract.
        output = _failure(
            "claude_adapter_internal_error", "Claude adapter failed unexpectedly"
        )
    else:
        output = run(payload)
    print(json.dumps(output, sort_keys=True))
    if output.get("failed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
