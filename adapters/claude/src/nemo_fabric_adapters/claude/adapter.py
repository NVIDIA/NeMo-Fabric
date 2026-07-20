# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Claude Agent SDK through the Fabric adapter process contract."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shlex
import shutil
import subprocess
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import is_dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk import ClaudeSDKError
from claude_agent_sdk import CLIConnectionError
from claude_agent_sdk import CLIJSONDecodeError
from claude_agent_sdk import CLINotFoundError
from claude_agent_sdk import Message
from claude_agent_sdk import ProcessError
from claude_agent_sdk import ResultMessage
from claude_agent_sdk import query
from claude_agent_sdk._errors import MessageParseError
from nemo_fabric_adapters.common import relay_gateway
from nemo_fabric_adapters.common import relay_hooks
from nemo_fabric_adapters.common import utils as common_utils

LOGGER = logging.getLogger(__name__)

PERMISSION_MODES = {
    "default",
    "acceptEdits",
    "bypassPermissions",
    "plan",
    "dontAsk",
    "auto",
}
SETTING_SOURCES = {"user", "project", "local"}
NORMALIZED_SETTING_FIELDS = {
    "model_name": "FabricConfig.models",
    "cwd": "FabricConfig.environment.workspace",
    "tools": "FabricConfig.tools",
    "disallowed_tools": "FabricConfig.tools.blocked",
    "mcp_servers": "FabricConfig.mcp",
    "skills": "FabricConfig.skills",
}
INHERITED_ENV_NAMES = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_CONFIG_DIR",
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_IDENTITY_TOKEN",
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
    "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_PROFILE",
    "ANTHROPIC_SERVICE_ACCOUNT_ID",
    "ANTHROPIC_WORKSPACE_ID",
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
    "USER",
    "USERPROFILE",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}


@dataclass(frozen=True)
class ClaudeRelaySettings:
    """Invocation-scoped Relay gateway and Claude plugin settings."""

    gateway: relay_gateway.RelayGatewayLaunch
    plugin_config: dict[str, Any]
    plugin_path: Path


class ClaudeAdapterError(Exception):
    """Expected adapter error with a stable public code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.metadata = metadata or {}


class AdapterInputError(ClaudeAdapterError):
    """Invalid Fabric invocation input."""


class AdapterConfigError(ClaudeAdapterError):
    """Invalid Claude adapter configuration."""


class AdapterStateError(ClaudeAdapterError):
    """Invalid persisted runtime state."""


class AdapterRelayError(ClaudeAdapterError):
    """NeMo Relay setup or lifecycle failure."""


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise AdapterConfigError(
            "claude_invalid_configuration", f"{name} must be a mapping"
        )
    return value


def _string_list(value: Any, *, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise AdapterConfigError(
            "claude_invalid_configuration",
            f"{name} must be a list of non-empty strings",
        )
    return list(value)


def _positive_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterConfigError(
            "claude_invalid_configuration", f"{name} must be positive"
        )
    number = float(value)
    if number <= 0 or not math.isfinite(number):
        raise AdapterConfigError(
            "claude_invalid_configuration", f"{name} must be positive"
        )
    return number


def runtime_id(payload: dict[str, Any]) -> str:
    value = common_utils.runtime_context(payload).get("runtime_id")
    if not isinstance(value, str) or not value:
        raise AdapterInputError(
            "claude_invalid_request", "Fabric runtime ID is required"
        )
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
        path = Path(common_utils.base_dir(payload)) / path
    return path


def resolve_cwd(payload: dict[str, Any]) -> Path:
    environment = common_utils.environment_payload(payload)
    workspace = environment.get("workspace")
    return _resolve_path(payload, workspace or common_utils.base_dir(payload))


def selected_model(payload: dict[str, Any]) -> str | None:
    model_config = _selected_model_config(payload)
    value = model_config.get("model")
    if value is None:
        return None
    provider = model_config.get("provider")
    if provider not in {"anthropic", "nvidia"}:
        raise AdapterConfigError(
            "claude_invalid_configuration",
            "models.default.provider must be anthropic or nvidia for the Claude adapter",
        )
    if not isinstance(value, str) or not value:
        raise AdapterConfigError(
            "claude_invalid_configuration", "model must be a non-empty string"
        )
    return value.removeprefix("anthropic/") if provider == "anthropic" else value


def _nvidia_environment(payload: dict[str, Any]) -> dict[str, str]:
    model = _selected_model_config(payload)
    if model.get("provider") != "nvidia":
        return {}
    api_key_env = model.get("api_key_env") or "NVIDIA_API_KEY"
    if not isinstance(api_key_env, str) or not api_key_env:
        raise AdapterConfigError(
            "claude_invalid_configuration",
            "models.default.api_key_env must be a non-empty string",
        )
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise AdapterConfigError(
            "claude_invalid_configuration",
            f"{api_key_env} is required for the NVIDIA model provider",
        )
    settings = _settings(payload)
    model_settings = _mapping(model.get("settings"), name="models.default.settings")
    base_url = (
        settings.get("base_url")
        or model_settings.get("base_url")
        or os.environ.get("NVIDIA_FRONTIER_BASE_URL")
    )
    if not isinstance(base_url, str) or not base_url:
        raise AdapterConfigError(
            "claude_invalid_configuration",
            "models.default.settings.base_url or NVIDIA_FRONTIER_BASE_URL is required "
            "for the NVIDIA model provider",
        )
    # Claude Code appends the Anthropic API version path itself, while Fabric's
    # shared NVIDIA endpoint includes it for OpenAI-compatible clients.
    claude_base_url = base_url.rstrip("/").removesuffix("/v1")
    return {
        "ANTHROPIC_API_KEY": api_key,
        "ANTHROPIC_AUTH_TOKEN": "",
        "ANTHROPIC_BASE_URL": claude_base_url,
    }


def _mcp_servers(payload: dict[str, Any]) -> dict[str, Any]:
    native = (
        _mapping(common_utils.capability_plan(payload), name="capability_plan").get(
            "native"
        )
        or {}
    )
    servers = _mapping(native, name="capability_plan.native").get("mcp_servers") or {}
    result: dict[str, Any] = {}
    for name, raw in sorted(_mapping(servers, name="native MCP servers").items()):
        server = _mapping(raw, name=f"MCP server {name}")
        transport = server.get("transport")
        url = server.get("url")
        if not isinstance(url, str) or not url:
            raise AdapterConfigError(
                "claude_invalid_configuration", "MCP server URL is required"
            )
        if transport == "stdio":
            command = shlex.split(url)
            if not command:
                raise AdapterConfigError(
                    "claude_invalid_configuration", "MCP command is required"
                )
            result[name] = {"type": "stdio", "command": command[0], "args": command[1:]}
        elif transport in {"http", "streamable-http"}:
            result[name] = {"type": "http", "url": url}
        elif transport == "sse":
            result[name] = {"type": "sse", "url": url}
        else:
            raise AdapterConfigError(
                "claude_invalid_configuration",
                f"unsupported MCP transport: {transport}",
            )
    return result


def _native_skill_paths(payload: dict[str, Any]) -> list[Path]:
    native = (
        _mapping(common_utils.capability_plan(payload), name="capability_plan").get(
            "native"
        )
        or {}
    )
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
                "claude_invalid_configuration",
                f"Fabric skill names must be unique: {name}",
            )
        names.add(name)
        skills.append((name, skill_path))

    plugin_key = sha256(runtime_id(payload).encode()).hexdigest()
    plugin_root = (
        _artifact_root(payload) / ".fabric" / "claude" / "plugins" / plugin_key
    )
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


def _stage_relay_plugin(plugin_path: Path, executable: Path) -> None:
    if plugin_path.exists():
        shutil.rmtree(plugin_path)
    (plugin_path / ".claude-plugin").mkdir(parents=True)
    (plugin_path / "hooks").mkdir()
    (plugin_path / ".claude-plugin" / "plugin.json").write_text(
        json.dumps(
            {
                "name": "nemo-fabric-relay",
                "description": "NeMo Relay hooks managed by NeMo Fabric",
                "version": "1.0.0",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (plugin_path / "hooks" / "hooks.json").write_text(
        json.dumps(
            relay_hooks.render_relay_hooks("claude", executable),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def prepare_claude_relay(payload: dict[str, Any]) -> ClaudeRelaySettings | None:
    """Generate invocation-scoped Relay and Claude hook configuration."""

    if not common_utils.relay_enabled(payload):
        return None
    settings = _settings(payload)
    command = settings.get("nemo_relay_command") or "nemo-relay"
    if not isinstance(command, (str, Path)):
        raise AdapterConfigError(
            "claude_invalid_configuration",
            "nemo_relay_command must be a path",
        )
    try:
        executable = relay_gateway.resolve_relay_command(
            Path(common_utils.base_dir(payload)).resolve(),
            command,
        )
    except FileNotFoundError as error:
        raise AdapterRelayError(
            "claude_relay_unavailable",
            "NeMo Relay CLI executable was not found",
        ) from error

    try:
        relay_contract = relay_gateway.relay_cli_contract(executable)
        plugin_config = common_utils.load_relay_plugin_config(payload)
        config_path, plugin_config_path = common_utils.write_relay_configs(
            relay_config={"agents": {"claude": {"command": "claude"}}},
            plugin_config=plugin_config,
            observability_version=relay_contract.observability_version,
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise AdapterRelayError(
            "claude_relay_configuration_failed",
            "NeMo Relay runtime configuration is unavailable",
        ) from error
    if config_path is None or plugin_config_path is None:
        raise AdapterRelayError(
            "claude_relay_configuration_failed",
            "NeMo Relay runtime configuration is unavailable",
        )

    port = relay_gateway.find_available_tcp_port()
    gateway_bind = f"127.0.0.1:{port}"
    gateway = relay_gateway.RelayGatewayLaunch(
        executable=executable,
        config_path=config_path,
        bind=gateway_bind,
        url=f"http://{gateway_bind}",
        log_path=config_path.parent / "gateway.log",
    )
    plugin_path = config_path.parent / "claude-plugin"
    try:
        _stage_relay_plugin(plugin_path, executable)
    except OSError as error:
        shutil.rmtree(plugin_path, ignore_errors=True)
        raise AdapterRelayError(
            "claude_relay_configuration_failed",
            "Claude Relay hook configuration could not be generated",
        ) from error
    return ClaudeRelaySettings(
        gateway=gateway,
        plugin_config=plugin_config,
        plugin_path=plugin_path,
    )


def discard_stderr(_: str) -> None:
    """Consume Claude Code stderr without exposing it through Fabric artifacts."""


def build_options(
    payload: dict[str, Any],
    *,
    resume: str | None,
    relay: ClaudeRelaySettings | None = None,
) -> ClaudeAgentOptions:
    settings = _settings(payload)
    _validate_settings_boundary(settings)
    permission_mode = settings.get("permission_mode")
    if permission_mode is not None and permission_mode not in PERMISSION_MODES:
        raise AdapterConfigError(
            "claude_invalid_configuration", "permission_mode is invalid"
        )
    max_turns = settings.get("max_turns")
    if max_turns is not None and (
        isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns <= 0
    ):
        raise AdapterConfigError(
            "claude_invalid_configuration", "max_turns must be positive"
        )
    max_budget = settings.get("max_budget_usd")
    if max_budget is not None:
        max_budget = _positive_number(max_budget, name="max_budget_usd")
    sources = settings.get("setting_sources", [])
    sources = _string_list(sources, name="setting_sources")
    if any(source not in SETTING_SOURCES for source in sources):
        raise AdapterConfigError(
            "claude_invalid_configuration", "setting_sources is invalid"
        )
    cli_path = settings.get("cli_path")
    if cli_path is not None and not isinstance(cli_path, (str, Path)):
        raise AdapterConfigError(
            "claude_invalid_configuration", "cli_path must be a path"
        )

    system_prompt = settings.get("system_prompt")
    if system_prompt is not None and not isinstance(system_prompt, (str, dict)):
        raise AdapterConfigError(
            "claude_invalid_configuration", "system_prompt is invalid"
        )
    plugins = _stage_skill_plugin(payload)
    has_skill_plugin = bool(plugins)
    if relay is not None:
        plugins.append({"type": "local", "path": str(relay.plugin_path)})

    return ClaudeAgentOptions(
        resume=resume,
        cwd=resolve_cwd(payload),
        model=selected_model(payload),
        system_prompt=system_prompt,
        tools=None,
        allowed_tools=_string_list(settings.get("allowed_tools"), name="allowed_tools"),
        disallowed_tools=common_utils.blocked_tools(payload),
        permission_mode=permission_mode,
        max_turns=max_turns,
        max_budget_usd=max_budget,
        setting_sources=sources,
        cli_path=_resolve_path(payload, cli_path) if cli_path is not None else None,
        mcp_servers=_mcp_servers(payload),
        strict_mcp_config=True,
        skills="all" if has_skill_plugin else None,
        plugins=plugins,
        env=child_environment(
            payload,
            relay_gateway_url=relay.gateway.url if relay is not None else None,
        ),
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
    return Path(common_utils.base_dir(payload)) / "artifacts" / "claude"


def runtime_state_path(payload: dict[str, Any], fabric_runtime_id: str) -> Path:
    digest = sha256(fabric_runtime_id.encode("utf-8")).hexdigest()
    return (
        _artifact_root(payload) / ".fabric" / "claude" / "runtimes" / f"{digest}.json"
    )


def load_claude_session_id(
    payload: dict[str, Any], fabric_runtime_id: str
) -> str | None:
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
        raise AdapterStateError(
            "claude_invalid_runtime_state", "Claude session ID is missing"
        )
    path = runtime_state_path(payload, fabric_runtime_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    invocation_id = (
        common_utils.runtime_context(payload).get("invocation_id") or "invocation"
    )
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
    raise AdapterConfigError(
        "claude_invalid_configuration", "Claude message is not JSON-safe"
    )


def normalize_message(message: Message) -> dict[str, Any]:
    return {"type": type(message).__name__, "message": _json_safe(message)}


def _result_failed(result: ResultMessage) -> bool:
    return bool(result.is_error) or (
        isinstance(result.subtype, str) and result.subtype.startswith("error_")
    )


def normalize_result(
    payload: dict[str, Any], messages: list[Message], result: ResultMessage
) -> dict[str, Any]:
    del payload
    failed = _result_failed(result)
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
    return _failure(error.code, error.message, **error.metadata)


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


def child_environment(
    payload: dict[str, Any],
    *,
    relay_gateway_url: str | None = None,
) -> dict[str, str]:
    values = {name: "" for name in os.environ}
    values.update(
        {name: os.environ[name] for name in INHERITED_ENV_NAMES if name in os.environ}
    )
    model = _selected_model_config(payload)
    api_key_env = model.get("api_key_env")
    if isinstance(api_key_env, str) and api_key_env in os.environ:
        values[api_key_env] = os.environ[api_key_env]
    configured = _mapping(_settings(payload).get("env"), name="harness.settings.env")
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in configured.items()
    ):
        raise AdapterConfigError(
            "claude_invalid_configuration", "harness.settings.env must contain strings"
        )
    values.update(configured)
    values.update(_nvidia_environment(payload))
    if relay_gateway_url is not None:
        values["NEMO_RELAY_GATEWAY_URL"] = relay_gateway_url
        values["ANTHROPIC_BASE_URL"] = relay_gateway_url
    return values


def _relay_output(
    output: dict[str, Any],
    relay: ClaudeRelaySettings,
) -> dict[str, Any]:
    output["relay_runtime"] = {
        "enabled": True,
        "emitter": "claude-agent-sdk/nemo-relay",
        "config_path": os.environ.get("FABRIC_RELAY_CONFIG_PATH"),
        "gateway_config_path": str(relay.gateway.config_path),
        "gateway_url": relay.gateway.url,
        "gateway_log_path": str(relay.gateway.log_path),
    }
    output["relay_artifacts"] = common_utils.collect_relay_artifacts(
        relay.plugin_config
    )
    return output


def _start_relay_gateway(
    payload: dict[str, Any],
    relay: ClaudeRelaySettings | None,
) -> subprocess.Popen[Any] | None:
    if relay is None:
        return None
    try:
        return relay_gateway.start_relay_gateway(
            launch=relay.gateway,
            cwd=resolve_cwd(payload),
        )
    except relay_gateway.RelayGatewayError as error:
        raise AdapterRelayError(
            "claude_relay_start_failed",
            "NeMo Relay gateway failed to start",
            metadata={"gateway_log_path": str(relay.gateway.log_path)},
        ) from error


def _cleanup_relay(
    relay: ClaudeRelaySettings | None,
    gateway_process: subprocess.Popen[Any] | None,
) -> AdapterRelayError | None:
    cleanup_error: AdapterRelayError | None = None
    if gateway_process is not None:
        try:
            relay_gateway.stop_relay_gateway(gateway_process)
        except relay_gateway.RelayGatewayError:
            cleanup_error = AdapterRelayError(
                "claude_relay_stop_failed",
                "NeMo Relay gateway failed to stop",
                metadata={
                    "gateway_log_path": str(relay.gateway.log_path)
                    if relay is not None
                    else ""
                },
            )
    if relay is not None and relay.plugin_path.exists():
        try:
            shutil.rmtree(relay.plugin_path)
        except OSError:
            if cleanup_error is None:
                cleanup_error = AdapterRelayError(
                    "claude_relay_cleanup_failed",
                    "Claude Relay hook configuration could not be removed",
                )
    return cleanup_error


def _merge_relay_output(
    output: dict[str, Any],
    relay: ClaudeRelaySettings | None,
    cleanup_error: AdapterRelayError | None,
) -> dict[str, Any]:
    if relay is None:
        return output
    output = _relay_output(output, relay)
    if cleanup_error is None:
        return output
    cleanup: dict[str, Any] = {
        "code": cleanup_error.code,
        "message": cleanup_error.message,
        "retryable": False,
    }
    if cleanup_error.metadata:
        cleanup["metadata"] = cleanup_error.metadata
    output["relay_runtime"]["cleanup_error"] = cleanup
    if not output["failed"]:
        output["completed"] = False
        output["failed"] = True
        output["error"] = cleanup
    return output


def _persist_result_session(
    payload: dict[str, Any],
    fabric_runtime_id: str,
    prior_session_id: str | None,
    result: ResultMessage,
) -> dict[str, Any] | None:
    if prior_session_id is not None and result.session_id != prior_session_id:
        return _failure(
            "claude_session_mismatch",
            "Claude session identity changed during resume",
        )
    save_claude_session_id(payload, fabric_runtime_id, result.session_id)
    return None


async def run_claude(payload: dict[str, Any]) -> dict[str, Any]:
    fabric_runtime_id = runtime_id(payload)
    prior_session_id = load_claude_session_id(payload, fabric_runtime_id)
    relay = prepare_claude_relay(payload)
    gateway_process = None
    cleanup_error: AdapterRelayError | None = None
    messages: list[Message] = []
    result: ResultMessage | None = None
    try:
        gateway_process = _start_relay_gateway(payload, relay)
        options = build_options(payload, resume=prior_session_id, relay=relay)
        try:
            async with asyncio.timeout(timeout_seconds(payload)):
                async for message in query(
                    prompt=request_prompt(payload), options=options
                ):
                    if isinstance(message, ResultMessage):
                        result = message
                    else:
                        messages.append(message)
        except (TimeoutError, ClaudeSDKError) as error:
            output = sdk_failure(error)
        except Exception:
            # Claude Agent SDK 0.2.120 can yield an error ResultMessage and then
            # raise a plain Exception while closing the query stream. Preserve
            # the typed terminal result, but do not hide unrelated exceptions.
            if result is None or not _result_failed(result):
                raise
            LOGGER.exception("Claude SDK stream raised after a failed terminal result")
            output = normalize_result(payload, messages, result)
        else:
            if result is None:
                output = _failure(
                    "claude_missing_result", "Claude returned no terminal result"
                )
            else:
                output = normalize_result(payload, messages, result)
                if not output["failed"]:
                    output = (
                        _persist_result_session(
                            payload,
                            fabric_runtime_id,
                            prior_session_id,
                            result,
                        )
                        or output
                    )
    finally:
        cleanup_error = _cleanup_relay(relay, gateway_process)

    return _merge_relay_output(output, relay, cleanup_error)


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
    except (
        Exception
    ):  # Malformed invocation input must still satisfy the process contract.
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
