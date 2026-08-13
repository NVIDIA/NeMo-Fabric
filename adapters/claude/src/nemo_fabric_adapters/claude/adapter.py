# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Claude Agent SDK through the Fabric adapter contract."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import shutil
import subprocess
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import is_dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk import ClaudeSDKClient
from claude_agent_sdk import ClaudeSDKError
from claude_agent_sdk import CLIConnectionError
from claude_agent_sdk import CLIJSONDecodeError
from claude_agent_sdk import CLINotFoundError
from claude_agent_sdk import Message
from claude_agent_sdk import ProcessError
from claude_agent_sdk import ResultMessage
from claude_agent_sdk import HookMatcher
from claude_agent_sdk._errors import MessageParseError
from nemo_fabric_adapter_contract.codec import ContractValidationError
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentModelConfig
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common import relay_artifacts
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
MCP_HEADER_ENVIRONMENT_VARIABLE = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-[^}]*)?\}"
    r"|\$([A-Za-z_][A-Za-z0-9_]*)"
    r"|%([A-Za-z_][A-Za-z0-9_]*)%"
)


@dataclass(frozen=True)
class ClaudeRelaySettings:
    """Relay gateway and Claude plugin settings owned by one adapter run."""

    gateway: relay_gateway.RelayGatewayLaunch
    plugin_config: dict[str, Any]
    plugin_path: Path


@dataclass(frozen=True)
class ClaudeMcpSettings:
    """Staged MCP configuration and its process-only environment values."""

    config_path: Path
    environment: dict[str, str]


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


class AdapterRelayError(ClaudeAdapterError):
    """NeMo Relay setup or lifecycle failure."""


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


def _runtime_context(payload: dict[str, Any]) -> RuntimeContext:
    try:
        return RuntimeContext.from_mapping(payload.get("runtime_context"))
    except ContractValidationError as error:
        raise lifecycle.LifecycleError(
            "claude_invalid_runtime_context",
            "Claude runtime context is invalid",
        ) from error


def request_prompt(payload: dict[str, Any]) -> str:
    request = payload.get("request") or {}
    value = request.get("input")
    if not isinstance(value, str):
        raise AdapterInputError("claude_invalid_request", "Claude input must be text")
    return value


def _settings(config: AgentConfig) -> dict[str, Any]:
    return config.harness.settings if config.harness is not None else {}


def _selected_model_config(config: AgentConfig) -> AgentModelConfig:
    model = config.models.get("default")
    if model is None and len(config.models) == 1:
        model = next(iter(config.models.values()))
    if model is None:
        raise AdapterConfigError(
            "claude_invalid_configuration",
            "Claude requires a default model or exactly one model",
        )
    return model


def _resolve_path(base_dir: str, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path


def resolve_cwd(runtime_context: RuntimeContext, base_dir: str) -> Path:
    return _resolve_path(base_dir, runtime_context.environment.workspace or base_dir)


def selected_model(model: AgentModelConfig) -> str:
    return (
        model.model.removeprefix("anthropic/")
        if model.provider == "anthropic"
        else model.model
    )


def _anthropic_base_url(model: AgentModelConfig) -> str | None:
    if model.base_url is None:
        return None
    base_url = model.base_url.rstrip("/")
    return base_url.removesuffix("/v1") if model.provider != "anthropic" else base_url


def _model_environment(
    model: AgentModelConfig, environment: dict[str, str]
) -> dict[str, str]:
    api_key_env = model.api_key_env
    api_key = (
        environment.get(api_key_env) or os.environ.get(api_key_env)
        if api_key_env
        else None
    )
    if model.provider != "anthropic" and api_key_env is None:
        raise AdapterConfigError(
            "claude_invalid_configuration",
            "selected model api_key_env is required for a custom "
            "Anthropic Messages-compatible provider",
        )
    if api_key_env is not None and not api_key:
        raise AdapterConfigError(
            "claude_invalid_configuration",
            f"{api_key_env} is required for the selected model provider",
        )
    base_url = _anthropic_base_url(model)
    if model.provider != "anthropic" and not base_url:
        raise AdapterConfigError(
            "claude_invalid_configuration",
            "selected model base_url is required for a custom "
            "Anthropic Messages-compatible provider",
        )
    values: dict[str, str] = {}
    if api_key:
        values["ANTHROPIC_API_KEY"] = api_key
    if base_url:
        values["ANTHROPIC_BASE_URL"] = base_url
    if model.provider != "anthropic":
        values["ANTHROPIC_AUTH_TOKEN"] = ""
    return values


def _mcp_servers(config: AgentConfig) -> dict[str, Any]:
    servers = config.mcp.servers if config.mcp is not None else {}
    result: dict[str, Any] = {}
    for name, server in sorted(servers.items()):
        transport = server.transport
        url = server.url
        if transport == "stdio":
            result[name] = {
                "type": "stdio",
                "command": url,
                "args": server.args,
            }
            if env := server.env:
                result[name]["env"] = env
        elif transport in {"http", "streamable-http"}:
            result[name] = {"type": "http", "url": url}
        elif transport == "sse":
            result[name] = {"type": "sse", "url": url}
        else:
            raise AdapterConfigError(
                "claude_invalid_configuration",
                f"unsupported MCP transport: {transport}",
            )
        if headers := server.custom_headers:
            try:
                common_utils.validate_http_headers(name, headers)
                result[name]["headers"] = headers
            except ValueError as error:
                raise AdapterConfigError(
                    "claude_invalid_configuration", str(error)
                ) from error

        auth = server.authentication
        if auth is not None:
            raise AdapterConfigError(
                "claude_invalid_configuration",
                f"MCP server {name!r} {auth.type!r} authentication is not supported by Claude",
            )
    return result


def _stage_mcp_config(
    config: AgentConfig, runtime_context: RuntimeContext, base_dir: str
) -> ClaudeMcpSettings | None:
    # Dictionary-valued ClaudeAgentOptions.mcp_servers are JSON-serialized by
    # claude-agent-sdk into the literal `--mcp-config` command-line argument,
    # where MCP credentials can be observed by process-inspection tools. Passing
    # a Path puts only the filename in argv. We also replace each credential with
    # a generated ${VAR} reference and provide its value through the deliberately
    # scoped child environment. This extra mapping keeps credentials out of the
    # staged file if the process is terminated before runtime cleanup can remove
    # it. The file remains owner-only as defense in depth and is retained until
    # the SDK client disconnects normally.
    servers = _mcp_servers(config)
    if not servers:
        return None
    fabric_runtime_id = runtime_context.runtime_id
    environment: dict[str, str] = {}

    def project_environment_value(server_name: str, value_name: str, value: str) -> str:
        projection_key = (
            sha256(f"{fabric_runtime_id}\0{server_name}\0{value_name}".encode())
            .hexdigest()
            .upper()
        )
        projected_name = f"NEMO_FABRIC_CLAUDE_MCP_{projection_key}"
        environment[projected_name] = value
        return f"${{{projected_name}}}"

    for server_name, server in servers.items():
        raw_environment = server.get("env")
        if raw_environment is not None:
            projected_environment: dict[str, str] = {}
            for variable_name, value in sorted(raw_environment.items()):
                projected_environment[variable_name] = project_environment_value(
                    server_name, variable_name, value
                )
            server["env"] = projected_environment

        if headers := server.get("headers"):
            projected_headers: dict[str, str] = {}
            for header_name, header_value in headers.items():

                def project_header_reference(match: re.Match[str]) -> str:
                    variable_name = next(
                        group for group in match.groups() if group is not None
                    )
                    if variable_name in os.environ:
                        value = os.environ[variable_name]
                    else:
                        return match.group(0)
                    return project_environment_value(
                        server_name, f"header:{header_name}:{variable_name}", value
                    )

                projected_headers[header_name] = MCP_HEADER_ENVIRONMENT_VARIABLE.sub(
                    project_header_reference, header_value
                )
            server["headers"] = projected_headers

    config_root = (
        _artifact_root(runtime_context, base_dir)
        / ".fabric"
        / "claude"
        / "mcp"
        / sha256(fabric_runtime_id.encode()).hexdigest()
    )
    if config_root.exists():
        shutil.rmtree(config_root)
    config_root.mkdir(parents=True, mode=0o700)
    config_root.chmod(0o700)
    config_path = config_root / "mcp.json"

    try:
        descriptor = os.open(
            config_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"mcpServers": servers}, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except BaseException:
        shutil.rmtree(config_root, ignore_errors=True)
        raise
    return ClaudeMcpSettings(config_path=config_path, environment=environment)


def _cleanup_mcp_config(config_path: Path | None) -> None:
    if config_path is None:
        return
    try:
        shutil.rmtree(config_path.parent)
    except OSError:
        LOGGER.exception("Claude MCP runtime configuration could not be removed")


def _native_skill_paths(config: AgentConfig, base_dir: str) -> list[Path]:
    values = config.skills.paths if config.skills is not None else []
    return [_resolve_path(base_dir, value) for value in values]


def _stage_skill_plugin(
    config: AgentConfig, runtime_context: RuntimeContext, base_dir: str
) -> list[dict[str, str]]:
    skill_paths = _native_skill_paths(config, base_dir)
    if not skill_paths:
        return []

    skills: list[tuple[str, Path]] = []
    names: set[str] = set()
    for skill_path in skill_paths:
        if not skill_path.is_dir() or not (skill_path / "SKILL.md").is_file():
            raise AdapterConfigError(
                "claude_invalid_configuration",
                f"NeMo Fabric skill path must be a directory containing SKILL.md: {skill_path}",
            )
        name = skill_path.name
        if name in names:
            raise AdapterConfigError(
                "claude_invalid_configuration",
                f"NeMo Fabric skill names must be unique: {name}",
            )
        names.add(name)
        skills.append((name, skill_path))

    plugin_key = sha256(runtime_context.runtime_id.encode()).hexdigest()
    plugin_root = (
        _artifact_root(runtime_context, base_dir)
        / ".fabric"
        / "claude"
        / "plugins"
        / plugin_key
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


def prepare_claude_relay(
    agent_name: str,
    model: AgentModelConfig,
    runtime_context: RuntimeContext,
    base_dir: str,
) -> ClaudeRelaySettings | None:
    """Generate Relay gateway and Claude hook configuration."""

    if runtime_context.telemetry is None or not runtime_context.telemetry.relay_enabled:
        return None
    command = os.environ.get("FABRIC_TEST_NEMO_RELAY_COMMAND", "nemo-relay")
    try:
        executable = relay_gateway.resolve_relay_command(
            Path(base_dir).resolve(),
            command,
        )
    except FileNotFoundError as error:
        raise AdapterRelayError(
            "claude_relay_unavailable",
            "NeMo Relay CLI executable was not found",
        ) from error

    try:
        relay_contract = relay_gateway.relay_cli_contract(executable)
        plugin_config = common_utils.load_relay_plugin_config(
            {
                "agent_name": agent_name,
                "base_dir": base_dir,
                "config": {"models": {"default": model.to_mapping()}},
                "runtime_context": runtime_context.to_mapping(),
            }
        )
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
        anthropic_base_url=_anthropic_base_url(model),
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


def tool_policy_hooks(config: AgentConfig) -> dict[str, list[HookMatcher]] | None:
    """Enforce the normalized tool policy across built-in, MCP, and plugin tools."""

    enabled = config.tools.enabled if config.tools is not None else None
    blocked = set(config.tools.blocked if config.tools is not None else [])
    if enabled is None and not blocked:
        return None
    enabled_set = None if enabled is None else set(enabled)

    async def enforce_policy(
        hook_input: dict[str, Any],
        _tool_use_id: str | None,
        _context: dict[str, Any],
    ) -> dict[str, Any]:
        tool_name = str(hook_input.get("tool_name") or "")
        is_blocked = tool_name in blocked or (
            enabled_set is not None and tool_name not in enabled_set
        )
        if not is_blocked:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"Tool '{tool_name}' is blocked by the configured tools policy."
                ),
            }
        }

    return {"PreToolUse": [HookMatcher(hooks=[enforce_policy])]}


def build_options(
    config: AgentConfig,
    runtime_context: RuntimeContext,
    base_dir: str,
    *,
    relay: ClaudeRelaySettings | None = None,
) -> ClaudeAgentOptions:
    settings = _settings(config)
    model = _selected_model_config(config)
    permission_mode = settings.get("permission_mode")
    if permission_mode is not None and permission_mode not in PERMISSION_MODES:
        raise AdapterConfigError(
            "claude_invalid_configuration", "permission_mode is invalid"
        )
    max_turns = config.runtime.max_turns if config.runtime is not None else None
    max_budget = settings.get("max_budget_usd")
    if max_budget is not None:
        max_budget = _positive_number(max_budget, name="max_budget_usd")
    sources = settings.get("setting_sources", [])
    sources = _string_list(sources, name="setting_sources")
    if any(source not in SETTING_SOURCES for source in sources):
        raise AdapterConfigError(
            "claude_invalid_configuration", "setting_sources is invalid"
        )
    cli_path = os.environ.get("FABRIC_TEST_CLAUDE_CLI_PATH")

    instructions = config.instructions
    system_prompt = (
        instructions.system.content if instructions and instructions.system else None
    )
    enabled_tools = config.tools.enabled if config.tools is not None else None
    allowed_tools = (
        enabled_tools
        if permission_mode == "dontAsk" and enabled_tools is not None
        else []
    )
    plugins = _stage_skill_plugin(config, runtime_context, base_dir)
    has_skill_plugin = bool(plugins)
    if relay is not None:
        plugins.append({"type": "local", "path": str(relay.plugin_path)})

    environment = child_environment(
        model,
        runtime_context,
        relay_gateway_url=relay.gateway.url if relay is not None else None,
    )
    mcp = _stage_mcp_config(config, runtime_context, base_dir)
    if mcp is not None:
        environment.update(mcp.environment)
    try:
        return ClaudeAgentOptions(
            cwd=resolve_cwd(runtime_context, base_dir),
            model=selected_model(model),
            system_prompt=system_prompt,
            tools=enabled_tools,
            allowed_tools=allowed_tools,
            disallowed_tools=config.tools.blocked if config.tools is not None else [],
            hooks=tool_policy_hooks(config),
            permission_mode=permission_mode,
            max_turns=max_turns,
            max_budget_usd=max_budget,
            setting_sources=sources,
            cli_path=_resolve_path(base_dir, cli_path) if cli_path else None,
            mcp_servers=mcp.config_path if mcp is not None else {},
            strict_mcp_config=True,
            skills="all" if has_skill_plugin else None,
            plugins=plugins,
            env=environment,
            stderr=discard_stderr,
        )
    except BaseException:
        _cleanup_mcp_config(mcp.config_path if mcp is not None else None)
        raise


def timeout_seconds() -> float:
    return 1800.0


def _remaining_timeout(deadline: float) -> float:
    return max(0.0, deadline - asyncio.get_running_loop().time())


def _artifact_root(runtime_context: RuntimeContext, base_dir: str) -> Path:
    root = runtime_context.artifacts.root
    if root:
        return Path(root)
    return Path(base_dir) / "artifacts" / "claude"


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


def normalize_result(messages: list[Message], result: ResultMessage) -> dict[str, Any]:
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
    model: AgentModelConfig,
    runtime_context: RuntimeContext,
    *,
    relay_gateway_url: str | None = None,
) -> dict[str, str]:
    values = {name: "" for name in os.environ}
    values.update(
        {name: os.environ[name] for name in INHERITED_ENV_NAMES if name in os.environ}
    )
    api_key_env = model.api_key_env
    if api_key_env is not None and api_key_env in os.environ:
        values[api_key_env] = os.environ[api_key_env]
    configured = runtime_context.environment.env
    values.update(configured)
    model_environment = _model_environment(model, values)
    conflicts = sorted(
        name
        for name, value in model_environment.items()
        if name in configured and configured[name] != value
    )
    if conflicts:
        fields = ", ".join(f"environment.env.{name}" for name in conflicts)
        raise AdapterConfigError(
            "claude_invalid_configuration",
            f"{fields} conflicts with the selected model configuration; "
            "configure model credentials and endpoints through models.<role>, "
            "or remove the duplicate environment.env values",
        )
    values.update(model_environment)
    if relay_gateway_url is not None:
        values["NEMO_RELAY_GATEWAY_URL"] = relay_gateway_url
        values["ANTHROPIC_BASE_URL"] = relay_gateway_url
    return values


def _relay_output(
    output: dict[str, Any],
    relay: ClaudeRelaySettings,
    *,
    artifacts: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    output["relay_runtime"] = {
        "enabled": True,
        "emitter": "claude-agent-sdk/nemo-relay",
        "config_path": os.environ.get("FABRIC_RELAY_CONFIG_PATH"),
        "gateway_config_path": str(relay.gateway.config_path),
        "gateway_url": relay.gateway.url,
        "gateway_log_path": str(relay.gateway.log_path),
    }
    output["relay_artifacts"] = (
        common_utils.collect_relay_artifacts(relay.plugin_config)
        if artifacts is None
        else artifacts
    )
    return output


def _start_relay_gateway(
    runtime_context: RuntimeContext,
    base_dir: str,
    relay: ClaudeRelaySettings | None,
) -> subprocess.Popen[Any] | None:
    if relay is None:
        return None
    try:
        return relay_gateway.start_relay_gateway(
            launch=relay.gateway,
            cwd=resolve_cwd(runtime_context, base_dir),
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


def _validate_result_session(
    current_session_id: str | None, result: ResultMessage
) -> dict[str, Any] | None:
    if current_session_id is not None and result.session_id != current_session_id:
        return _failure(
            "claude_session_mismatch",
            "Claude session identity changed during the runtime",
        )
    return None


def _as_lifecycle_error(error: ClaudeAdapterError) -> lifecycle.LifecycleError:
    return lifecycle.LifecycleError(
        error.code,
        error.message,
        metadata=error.metadata,
    )


def _sdk_lifecycle_error(error: BaseException) -> lifecycle.LifecycleError:
    output = sdk_failure(error)
    reported = output["error"]
    return lifecycle.LifecycleError(
        reported["code"],
        reported["message"],
        retryable=reported["retryable"],
        metadata=reported.get("metadata"),
    )


class ClaudeRuntime:
    """One connected Claude SDK client owned by a Fabric runtime."""

    def __init__(self) -> None:
        self._agent_config: AgentConfig | None = None
        self._fabric_runtime_id: str | None = None
        self._claude_session_id: str | None = None
        self._client: ClaudeSDKClient | None = None
        self._relay: ClaudeRelaySettings | None = None
        self._gateway_process: subprocess.Popen[Any] | None = None
        self._mcp_config_path: Path | None = None
        self._unusable = False

    async def start(self, payload: dict[str, Any]) -> None:
        if self._client is not None:
            raise lifecycle.LifecycleError(
                "claude_runtime_already_started",
                "Claude runtime is already started",
            )
        try:
            agent_config = payload["config"]
            runtime_context = _runtime_context(payload)
            base_dir = common_utils.base_dir(payload)
            model = _selected_model_config(agent_config)
            fabric_runtime_id = runtime_context.runtime_id
            relay = prepare_claude_relay(
                common_utils.agent_name(payload), model, runtime_context, base_dir
            )
            self._relay = relay
            self._gateway_process = _start_relay_gateway(
                runtime_context, base_dir, relay
            )
            options = build_options(
                agent_config, runtime_context, base_dir, relay=relay
            )
            if isinstance(options.mcp_servers, Path):
                self._mcp_config_path = options.mcp_servers

            client = ClaudeSDKClient(options)
            await client.connect()
        except ClaudeAdapterError as error:
            self._cleanup_failed_start()
            raise _as_lifecycle_error(error) from error
        except ClaudeSDKError as error:
            self._cleanup_failed_start()
            raise _sdk_lifecycle_error(error) from error
        except BaseException:
            self._cleanup_failed_start()
            raise

        self._agent_config = agent_config
        self._fabric_runtime_id = fabric_runtime_id
        self._client = client

    async def invoke(self, invocation: dict[str, Any]) -> dict[str, Any]:
        client = self._client
        agent_config = self._agent_config
        fabric_runtime_id = self._fabric_runtime_id
        if client is None or agent_config is None or fabric_runtime_id is None:
            raise lifecycle.LifecycleError(
                "claude_runtime_not_started",
                "Claude runtime is not started",
            )
        runtime_context = _runtime_context(invocation)
        if runtime_context.runtime_id != fabric_runtime_id:
            raise lifecycle.LifecycleError(
                "claude_runtime_mismatch",
                "Claude invocation does not match the connected runtime",
            )
        if self._unusable:
            return _failure(
                "claude_runtime_unavailable",
                "Claude runtime cannot accept another invocation after a runtime failure",
            )

        invocation_deadline = asyncio.get_running_loop().time() + timeout_seconds()
        try:
            prompt = request_prompt(invocation)
        except ClaudeAdapterError as error:
            output = adapter_failure(error)
        except ClaudeSDKError as error:
            output = sdk_failure(error)
        else:
            relay = self._relay
            atif_before = (
                relay_artifacts.snapshot_atif_files(relay.plugin_config)
                if relay is not None
                and relay_artifacts.expects_local_atif(relay.plugin_config)
                else None
            )
            output = await self._run_query(
                client,
                prompt,
                _remaining_timeout(invocation_deadline),
            )
            if (
                output.get("completed")
                and relay is not None
                and atif_before is not None
            ):
                finalized = await relay_artifacts.wait_for_finalized_atif(
                    relay.plugin_config, atif_before
                )
                if finalized is None:
                    self._unusable = True
                    return _relay_output(
                        adapter_failure(
                            AdapterRelayError(
                                "claude_relay_atif_timeout",
                                "NeMo Relay did not finalize an ATIF artifact before the deadline",
                                metadata={
                                    "timeout_seconds": relay_artifacts.ATIF_FINALIZATION_TIMEOUT_SECONDS,
                                },
                            )
                        ),
                        relay,
                        artifacts=[],
                    )

        if self._relay is not None:
            output = _relay_output(output, self._relay)
        return output

    async def _run_query(
        self,
        client: ClaudeSDKClient,
        prompt: str,
        remaining_timeout: float,
    ) -> dict[str, Any]:
        """Run one SDK query and normalize its terminal result."""

        messages: list[Message] = []
        result: ResultMessage | None = None
        try:
            async with asyncio.timeout(remaining_timeout):
                await client.query(prompt)
                async for message in client.receive_response():
                    if isinstance(message, ResultMessage):
                        result = message
                    else:
                        messages.append(message)
        except (TimeoutError, ClaudeSDKError) as error:
            self._unusable = True
            await self._interrupt_failed_invocation()
            output = sdk_failure(error)
        except Exception:
            # Claude Agent SDK 0.2.120 can yield an error ResultMessage and then
            # raise a plain Exception while closing the response stream.
            self._unusable = True
            if result is None or not _result_failed(result):
                raise
            LOGGER.exception("Claude SDK stream raised after a failed terminal result")
            output = self._normalize_invocation(messages, result)
        else:
            if result is None:
                self._unusable = True
                output = _failure(
                    "claude_missing_result", "Claude returned no terminal result"
                )
            else:
                output = self._normalize_invocation(messages, result)
        return output

    def _normalize_invocation(
        self,
        messages: list[Message],
        result: ResultMessage,
    ) -> dict[str, Any]:
        try:
            output = normalize_result(messages, result)
            if not output["failed"]:
                invalid_session = _validate_result_session(
                    self._claude_session_id, result
                )
                if invalid_session is not None:
                    self._unusable = True
                    return invalid_session
                self._claude_session_id = result.session_id
            return output
        except ClaudeAdapterError as error:
            self._unusable = True
            return adapter_failure(error)

    async def stop(self) -> None:
        client = self._client
        self._client = None
        self._agent_config = None
        self._fabric_runtime_id = None
        self._claude_session_id = None
        self._unusable = True

        disconnect_error: BaseException | None = None
        try:
            if client is not None:
                await client.disconnect()
        except BaseException as error:
            if not isinstance(error, asyncio.CancelledError):
                LOGGER.exception("Claude SDK client failed to disconnect")
            disconnect_error = error
        finally:
            cleanup_error = _cleanup_relay(self._relay, self._gateway_process)
            self._relay = None
            self._gateway_process = None
            _cleanup_mcp_config(self._mcp_config_path)
            self._mcp_config_path = None
        if isinstance(disconnect_error, asyncio.CancelledError):
            raise disconnect_error
        if disconnect_error is not None:
            if cleanup_error is not None:
                LOGGER.error(
                    "Claude Relay cleanup also failed during disconnect: %s",
                    cleanup_error.code,
                )
            raise lifecycle.LifecycleError(
                "claude_disconnect_failed",
                "Claude SDK client failed to disconnect",
            ) from disconnect_error
        if cleanup_error is not None:
            raise _as_lifecycle_error(cleanup_error)

    def _cleanup_failed_start(self) -> None:
        cleanup_error = _cleanup_relay(self._relay, self._gateway_process)
        self._relay = None
        self._gateway_process = None
        _cleanup_mcp_config(self._mcp_config_path)
        self._mcp_config_path = None
        if cleanup_error is not None:
            LOGGER.error(
                "Claude runtime cleanup after start failure also failed: %s",
                cleanup_error.code,
            )

    async def _interrupt_failed_invocation(self) -> None:
        if self._client is None:
            return
        try:
            async with asyncio.timeout(5):
                await self._client.interrupt()
        except Exception:
            LOGGER.exception("Claude SDK invocation could not be interrupted")


def main() -> None:
    """Serve the persistent local-host lifecycle protocol."""

    lifecycle.serve(ClaudeRuntime, config_loader=AgentConfig.from_mapping)


if __name__ == "__main__":
    main()
