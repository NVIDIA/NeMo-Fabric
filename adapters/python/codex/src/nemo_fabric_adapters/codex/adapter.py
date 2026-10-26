#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Codex through its native Python SDK and the Fabric adapter contract."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from openai_codex import (
    ApprovalMode,
    AsyncCodex,
    AsyncThread,
    CodexConfig,
    CodexError,
    JsonRpcError,
    Sandbox,
    TransportClosedError,
    TurnResult,
    is_retryable_error,
)
from openai_codex.generated.v2_all import (
    ErrorNotification,
    ItemStartedNotification,
    ItemCompletedNotification,
    ListMcpServerStatusResponse,
    RawResponseCompletedNotification,
    McpAuthStatus,
    McpServerOauthLoginCompletedNotification,
    McpServerOauthLoginResponse,
    SkillsExtraRootsSetResponse,
    ThreadTokenUsage,
    ThreadStartParams,
    ThreadStartResponse,
    ThreadTokenUsageUpdatedNotification,
    TurnCompletedNotification,
)

# Private: the collector AsyncTurnHandle.run uses; recheck it on every openai-codex bump.
from openai_codex._run import _collect_async_turn_result
from openai_codex.models import Notification, UnknownNotification
from openai_codex.types import Personality, ReasoningEffort, TurnStatus

from nemo_fabric_adapter_contract.codec import ContractValidationError
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentModelConfig
from nemo_fabric_adapter_contract.models import AgentRunError
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunResult
from nemo_fabric_adapter_contract.models import AgentRunStatus
from nemo_fabric_adapter_contract.models import McpAuthenticationConfig
from nemo_fabric_adapter_contract.models import McpOAuth2Config
from nemo_fabric_adapter_contract.models import McpServiceAccountConfig
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapter_contract.models import (
    GapReason,
    ReportedUsage,
)
from nemo_fabric_adapter_contract.usage import UsageObservation, usage_from_observations
from nemo_fabric_adapters.common import instructions as common_instructions
import nemo_fabric_adapters.common.relay_gateway as relay_gateway
import nemo_fabric_adapters.common.relay_hooks as relay_hooks
import nemo_fabric_adapters.common.relay_artifacts as relay_artifacts
import nemo_fabric_adapters.common.utils as common_utils
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common import provenance

INTERRUPT_TIMEOUT_SECONDS = 5.0
_PACKAGE_DISTRIBUTION = "nemo-fabric-adapters-codex"
_HARNESS_NAME = "codex"
_METHOD_HARNESS_API = "harness_api"
CLOCK_RESOLUTION_SECONDS = time.get_clock_info("monotonic").resolution
SANDBOXES = {
    "read-only": Sandbox.read_only,
    "workspace-write": Sandbox.workspace_write,
    "danger-full-access": Sandbox.full_access,
}
APPROVAL_MODES = {
    "auto_review": ApprovalMode.auto_review,
    "deny_all": ApprovalMode.deny_all,
}


async def _open_authorization_url(authorization_url: str) -> bool:
    """Open a Codex OAuth authorization URL in a process that cancellation can kill.

    ``webbrowser.open`` can block, and a worker thread cannot be cancelled.
    """
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-c",
        "import sys, webbrowser; sys.exit(0 if webbrowser.open(sys.argv[1]) else 1)",
        authorization_url,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,  # POSIX only: lets cancellation kill launcher children.
    )
    try:
        return await process.wait() == 0
    finally:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            await process.wait()


INHERITED_ENV_NAMES = {
    "APPDATA",
    "CODEX_HOME",
    "CODEX_SQLITE_HOME",
    "COMSPEC",
    "DBUS_SESSION_BUS_ADDRESS",
    "HOME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOCALAPPDATA",
    "NO_PROXY",
    "OPENAI_API_KEY",
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
    "XDG_RUNTIME_DIR",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CodexRelaySettings:
    """Runtime-scoped Relay state consumed by the Codex SDK adapter."""

    gateway: relay_gateway.RelayGatewayLaunch
    plugin_config: dict[str, Any]


class CodexAdapterError(Exception):
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


class AdapterInputError(CodexAdapterError):
    """Invalid Fabric invocation input."""


class AdapterConfigError(CodexAdapterError):
    """Invalid Codex adapter configuration."""


class AdapterRelayError(CodexAdapterError):
    """NeMo Relay setup or lifecycle failure."""


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise AdapterConfigError(
            "codex_invalid_configuration", f"{name} must be a mapping"
        )
    return value


def _settings(config: AgentConfig) -> dict[str, Any]:
    return config.harness.settings if config.harness else {}


def request_prompt(request: AgentRunRequest) -> str:
    value = request.input
    if not isinstance(value, str):
        raise AdapterInputError("codex_invalid_request", "Codex input must be text")
    return value


def _native_mcp_servers(config: AgentConfig) -> dict[str, dict[str, Any]]:
    servers = config.mcp.servers if config.mcp else {}
    result: dict[str, dict[str, Any]] = {}
    for name, server in sorted(servers.items()):
        transport = server.transport
        target = os.path.expandvars(server.url).strip()
        if not target:
            raise AdapterConfigError(
                "codex_invalid_configuration",
                f"MCP server {name} URL is required",
            )
        normalized_transport = transport.strip().lower().replace("_", "-")
        if normalized_transport == "stdio":
            result[name] = {
                "command": target,
                "args": server.args,
            }
            if env := server.env:
                result[name]["env"] = env
        elif normalized_transport in {"http", "streamable-http"}:
            result[name] = {"url": target}
        else:
            raise AdapterConfigError(
                "codex_invalid_configuration",
                f"unsupported Codex MCP transport: {server.transport}",
            )
        if headers := server.custom_headers:
            try:
                headers = common_utils.expand_http_headers(name, headers)
            except ValueError as error:
                raise AdapterConfigError(
                    "codex_invalid_configuration", str(error)
                ) from error
            result[name]["http_headers"] = headers
        if authentication := server.authentication:
            oauth = _mcp_oauth_config(name, authentication)
            if oauth.client_secret_env:
                raise AdapterConfigError(
                    "codex_invalid_configuration",
                    f"MCP server {name} authentication.client_secret_env is not supported by Codex",
                )
            if oauth.client_id:
                raise AdapterConfigError(
                    "codex_invalid_configuration",
                    f"MCP server {name} authentication.client_id is not supported by Codex",
                )
            if oauth.client_name:
                raise AdapterConfigError(
                    "codex_invalid_configuration",
                    f"MCP server {name} authentication.client_name is not supported by Codex",
                )
            if oauth.token_endpoint_auth_method:
                raise AdapterConfigError(
                    "codex_invalid_configuration",
                    f"MCP server {name} authentication.token_endpoint_auth_method is not supported by Codex",
                )
            result[name]["auth"] = "oauth"
            if oauth.scopes:
                result[name]["scopes"] = list(oauth.scopes)
    return result


def _mcp_oauth_config(name: str, value: McpAuthenticationConfig) -> McpOAuth2Config:
    if isinstance(value, McpServiceAccountConfig):
        raise AdapterConfigError(
            "codex_invalid_configuration",
            f"MCP server {name!r} service_account authentication is not supported by Codex",
        )
    return value


def _mcp_oauth_callback_url(config: AgentConfig) -> str | None:
    values = {
        oauth.redirect_uri
        for name, server in (config.mcp.servers if config.mcp else {}).items()
        if (authentication := server.authentication)
        and (oauth := _mcp_oauth_config(name, authentication)).redirect_uri
    }
    if len(values) > 1:
        raise AdapterConfigError(
            "codex_invalid_configuration",
            "Codex supports only one MCP OAuth callback URL per adapter process",
        )
    return next(iter(values)) if values else None


def _native_skill_paths(config: AgentConfig, base_dir: str) -> list[Path]:
    values = config.skills.paths if config.skills else []

    paths: list[Path] = []
    names: set[str] = set()
    config_root = Path(base_dir)
    for value in values:
        skill_path = Path(value)
        if not skill_path.is_absolute():
            skill_path = config_root / skill_path
        skill_path = skill_path.resolve()
        skill_file = skill_path / "SKILL.md"
        if not skill_path.is_dir() or not skill_file.is_file():
            raise AdapterConfigError(
                "codex_invalid_configuration",
                "NeMo Fabric skill path must be a directory containing SKILL.md: "
                f"{skill_path}",
            )
        name = skill_path.name
        if not name or name in names:
            raise AdapterConfigError(
                "codex_invalid_configuration",
                f"NeMo Fabric skill names must be unique: {name}",
            )
        names.add(name)
        paths.append(skill_path)
    return paths


async def _register_skill_roots(codex: AsyncCodex, skill_paths: list[Path]) -> None:
    if not skill_paths:
        return

    # The pinned SDK does not yet wrap the app-server's process-scoped
    # skills/extraRoots/set request. Keep the pinned-SDK compatibility seam
    # here so arbitrary Fabric skill paths become discoverable without
    # modifying the consumer workspace.
    await codex.models()
    client = getattr(codex, "_client", None)
    request = getattr(client, "request", None)
    if not callable(request):
        raise AdapterConfigError(
            "codex_invalid_configuration",
            "Codex SDK does not expose the required skill registration request",
        )
    await request(
        "skills/extraRoots/set",
        {"extraRoots": [str(path) for path in skill_paths]},
        response_model=SkillsExtraRootsSetResponse,
    )


def _mcp_oauth_servers(
    config: AgentConfig,
) -> dict[str, McpOAuth2Config]:
    result: dict[str, McpOAuth2Config] = {}
    for name, server in (config.mcp.servers if config.mcp else {}).items():
        authentication = server.authentication
        if not authentication:
            continue
        result[name] = _mcp_oauth_config(name, authentication)
    return result


def _codex_protocol_client(codex: AsyncCodex) -> Any:
    client = getattr(codex, "_client", None)
    if not callable(getattr(client, "request", None)) or not callable(
        getattr(client, "next_notification", None)
    ):
        raise AdapterConfigError(
            "codex_invalid_configuration",
            "Codex SDK does not expose the required app-server requests",
        )
    return client


async def _mcp_auth_statuses(
    client: Any, *, thread_id: str, timeout: float
) -> dict[str, McpAuthStatus]:
    statuses: dict[str, McpAuthStatus] = {}
    seen_cursors: set[str] = set()
    cursor: str | None = None
    try:
        async with asyncio.timeout(timeout):
            while True:
                params: dict[str, Any] = {
                    "detail": "toolsAndAuthOnly",
                    "threadId": thread_id,
                }
                if cursor is not None:
                    params["cursor"] = cursor
                try:
                    response = await client.request(
                        "mcpServerStatus/list",
                        params,
                        response_model=ListMcpServerStatusResponse,
                    )
                except TimeoutError as error:
                    raise AdapterConfigError(
                        "codex_mcp_authentication_failed",
                        "Codex MCP status listing request timed out",
                    ) from error
                statuses.update(
                    {server.name: server.auth_status for server in response.data}
                )
                cursor = response.next_cursor
                if cursor is None:
                    return statuses
                if cursor in seen_cursors:
                    raise AdapterConfigError(
                        "codex_mcp_authentication_failed",
                        "Codex MCP status listing returned a repeated cursor",
                    )
                seen_cursors.add(cursor)
    except TimeoutError as error:
        raise AdapterConfigError(
            "codex_mcp_authentication_failed",
            "Codex MCP status listing timed out",
        ) from error


async def _login_mcp_server(
    client: Any,
    *,
    name: str,
    scopes: list[str] | None,
    thread_id: str,
    timeout: float,
) -> None:
    params: dict[str, Any] = {
        "name": name,
        "threadId": thread_id,
        "timeoutSecs": math.ceil(timeout),  # Cast to an int
    }
    if scopes:
        params["scopes"] = scopes
    try:
        async with asyncio.timeout(timeout):
            response = await client.request(
                "mcpServer/oauth/login",
                params,
                response_model=McpServerOauthLoginResponse,
            )
            opened = await _open_authorization_url(response.authorization_url)
            if not opened:
                raise AdapterConfigError(
                    "codex_mcp_authentication_failed",
                    f"Codex could not open a browser to authenticate MCP server {name!r}",
                )

            while True:
                notification = await client.next_notification()
                completed = notification.payload
                if (
                    notification.method == "mcpServer/oauthLogin/completed"
                    and isinstance(completed, McpServerOauthLoginCompletedNotification)
                    and completed.name == name
                    and completed.thread_id in {None, thread_id}
                ):
                    if completed.success:
                        return
                    raise AdapterConfigError(
                        "codex_mcp_authentication_failed",
                        f"Codex MCP OAuth login failed for server {name!r}",
                    )
    except TimeoutError as error:
        raise AdapterConfigError(
            "codex_mcp_authentication_failed",
            f"Codex MCP OAuth login timed out for server {name!r}",
        ) from error


async def _authenticate_mcp_servers(
    codex: AsyncCodex,
    thread: Any,
    config: AgentConfig,
    invocation_deadline: float,
) -> None:
    oauth_servers = _mcp_oauth_servers(config)
    if not oauth_servers:
        return

    client = _codex_protocol_client(codex)
    thread_id = str(thread.id)
    try:
        statuses = await _mcp_auth_statuses(
            client,
            thread_id=thread_id,
            timeout=_remaining_timeout(invocation_deadline),
        )
        for name, oauth in oauth_servers.items():
            status = statuses.get(name)
            if status in {McpAuthStatus.o_auth, McpAuthStatus.bearer_token}:
                continue
            if status is None:
                raise AdapterConfigError(
                    "codex_mcp_authentication_failed",
                    f"Codex did not report a status for MCP server {name!r}",
                )
            if status != McpAuthStatus.not_logged_in:
                raise AdapterConfigError(
                    "codex_mcp_authentication_failed",
                    f"Codex MCP server {name!r} does not support the configured OAuth login",
                )
            await _login_mcp_server(
                client,
                name=name,
                scopes=list(oauth.scopes) or None,
                thread_id=thread_id,
                timeout=min(
                    _remaining_timeout(invocation_deadline),
                    oauth.authorization_timeout_seconds,
                ),
            )
    except CodexAdapterError as error:
        # asyncio fires timers up to one clock tick early, so allow that tolerance.
        if _remaining_timeout(invocation_deadline) <= CLOCK_RESOLUTION_SECONDS:
            # Fabric's deadline, not the authentication step, ended the invocation.
            raise CodexAdapterError("timeout", "Codex invocation timed out") from error
        raise
    except (CodexError, RuntimeError, OSError) as error:
        raise AdapterConfigError(
            "codex_mcp_authentication_failed",
            "Codex MCP OAuth login could not be completed",
        ) from error


def _remaining_timeout(deadline: float) -> float:
    return max(0.0, deadline - asyncio.get_running_loop().time())


def resolve_cwd(context: RuntimeContext, base_dir: str) -> Path:
    path = Path(context.environment.workspace or base_dir)
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve()


def _selected_model_config(config: AgentConfig) -> AgentModelConfig:
    model = config.models.get("default")
    if model is None and len(config.models) == 1:
        model = next(iter(config.models.values()))
    if model is None:
        raise AdapterConfigError(
            "codex_invalid_configuration",
            "Codex requires a default model or exactly one model",
        )
    return model


def selected_model(config: AgentConfig) -> str:
    model = _selected_model_config(config)
    return (
        model.model.removeprefix("openai/")
        if model.provider == "openai"
        else model.model
    )


def custom_model_provider_config(
    config: AgentConfig, context: RuntimeContext
) -> dict[str, Any]:
    model_config = _selected_model_config(config)
    provider = model_config.provider
    if provider == "openai":
        return {}
    api_key_env = model_config.api_key_env
    if api_key_env is None:
        raise AdapterConfigError(
            "codex_invalid_configuration",
            "selected model api_key_env is required for a custom "
            "Responses-compatible provider",
        )
    if not (context.environment.env.get(api_key_env) or os.environ.get(api_key_env)):
        raise AdapterConfigError(
            "codex_invalid_configuration",
            f"{api_key_env} is required for the selected model provider",
        )
    base_url = model_config.base_url
    if not base_url:
        raise AdapterConfigError(
            "codex_invalid_configuration",
            "selected model base_url is required for a custom "
            "Responses-compatible provider",
        )
    return {
        "model_providers": {
            provider: {
                "name": provider,
                "base_url": base_url.rstrip("/"),
                "env_key": api_key_env,
                "wire_api": "responses",
            }
        }
    }


def openai_model_provider_config(config: AgentConfig) -> dict[str, Any]:
    model_config = _selected_model_config(config)
    if model_config.provider != "openai":
        return {}
    base_url = model_config.base_url
    return {"openai_base_url": base_url.rstrip("/")} if base_url else {}


def sandbox(config: AgentConfig) -> Sandbox:
    value = _settings(config).get("sandbox", "read-only")
    try:
        return SANDBOXES[value]
    except (KeyError, TypeError) as error:
        raise AdapterConfigError(
            "codex_invalid_configuration",
            f"sandbox must be one of: {', '.join(sorted(SANDBOXES))}",
        ) from error


def approval_mode(config: AgentConfig) -> ApprovalMode:
    value = _settings(config).get("approval_mode", "auto_review")
    try:
        return APPROVAL_MODES[value]
    except (KeyError, TypeError) as error:
        raise AdapterConfigError(
            "codex_invalid_configuration",
            f"approval_mode must be one of: {', '.join(sorted(APPROVAL_MODES))}",
        ) from error


def _optional_string(settings: dict[str, Any], name: str) -> str | None:
    value = settings.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise AdapterConfigError(
            "codex_invalid_configuration",
            f"harness.settings.{name} must be a non-empty string",
        )
    return value


def child_environment(
    config: AgentConfig,
    context: RuntimeContext,
    base_dir: str,
    *,
    relay_gateway_url: str | None = None,
) -> dict[str, str]:
    values = dict.fromkeys(os.environ, "")
    values.update(
        {name: os.environ[name] for name in INHERITED_ENV_NAMES if name in os.environ}
    )
    telemetry_env = context.telemetry.env if context.telemetry else {}
    values.update(telemetry_env)
    model_config = _selected_model_config(config)
    api_key_env = model_config.api_key_env
    if api_key_env is not None and api_key_env in os.environ:
        values[api_key_env] = os.environ[api_key_env]
    configured = context.environment.env
    values.update(configured)
    if model_config.provider == "openai" and api_key_env is not None:
        if api_key_env in values:
            values["OPENAI_API_KEY"] = values[api_key_env]
    if model_config.provider != "openai":
        codex_home = state_dir(context, base_dir) / "custom-provider-home"
        values["CODEX_HOME"] = str(codex_home)
    # The SDK overlays this mapping on the parent environment. An empty
    # originator is still treated as an override by Codex and produces invalid
    # initialize metadata ("/<version>"). Use the official SDK client identity
    # without inheriting the identity of a parent Codex process.
    values["CODEX_INTERNAL_ORIGINATOR_OVERRIDE"] = "codex_python_sdk"
    if relay_gateway_url is not None:
        values["NEMO_RELAY_GATEWAY_URL"] = relay_gateway_url
    return values


def _artifact_root(context: RuntimeContext, base_dir: str) -> Path:
    root = context.artifacts.root
    if root:
        return Path(str(root))
    return Path(base_dir) / "artifacts" / "codex"


def state_dir(context: RuntimeContext, base_dir: str) -> Path:
    return _artifact_root(context, base_dir) / ".fabric" / "codex"


def _merge_config(target: dict[str, Any], layer: dict[str, Any]) -> None:
    for key, value in layer.items():
        existing = target.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            _merge_config(existing, value)
        else:
            target[key] = value


def _json_value(value: Any, *, name: str) -> Any:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise AdapterConfigError(
            "codex_invalid_configuration", f"{name} must be JSON-compatible"
        ) from error
    return value


def _apply_config_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> None:
    for dotted_key, value in sorted(overrides.items()):
        if not isinstance(dotted_key, str):
            raise AdapterConfigError(
                "codex_invalid_configuration",
                "config_overrides keys must be strings",
            )
        parts = dotted_key.split(".")
        if any(not part for part in parts):
            raise AdapterConfigError(
                "codex_invalid_configuration",
                f"invalid Codex config override key {dotted_key!r}",
            )
        target = config
        for part in parts[:-1]:
            existing = target.setdefault(part, {})
            if not isinstance(existing, dict):
                raise AdapterConfigError(
                    "codex_invalid_configuration",
                    f"Codex config override {dotted_key!r} conflicts with {part!r}",
                )
            target = existing
        target[parts[-1]] = _json_value(value, name=f"config_overrides.{dotted_key}")


def native_codex_telemetry_config(context: RuntimeContext) -> dict[str, Any]:
    telemetry = context.telemetry
    if telemetry is None or "native" not in telemetry.metadata.get(
        "telemetry_providers", []
    ):
        return {}

    telemetry_config = telemetry.metadata.get("native_config", {})
    if not isinstance(telemetry_config, dict):
        raise AdapterConfigError(
            "codex_invalid_configuration",
            "runtime_context.telemetry.metadata.native_config must be a mapping",
        )
    for component in telemetry_config.get("components") or []:
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
            exporters = {
                "http_binary": ("otlp-http", "binary"),
                "grpc": ("otlp-grpc", "grpc"),
                "http_json": ("otlp-http", "json"),
            }
            try:
                exporter, protocol = exporters[transport]
            except (KeyError, TypeError) as error:
                raise AdapterConfigError(
                    "codex_invalid_configuration",
                    f"unsupported Codex native OpenTelemetry transport {transport!r}",
                ) from error
            otel["trace_exporter"] = {
                exporter: {"endpoint": endpoint, "protocol": protocol}
            }
        return {"otel": otel}
    return {}


def prepare_codex_relay(
    agent_name: str,
    config: AgentConfig,
    context: RuntimeContext,
    base_dir: str,
) -> CodexRelaySettings | None:
    """Generate invocation-scoped Relay gateway configuration."""

    if context.telemetry is None or not context.telemetry.relay_enabled:
        return None
    command = os.environ.get("FABRIC_TEST_NEMO_RELAY_COMMAND", "nemo-relay")
    try:
        executable = relay_gateway.resolve_relay_command(
            Path(base_dir).resolve(), command
        )
    except FileNotFoundError as error:
        raise AdapterRelayError(
            "codex_relay_unavailable", "NeMo Relay CLI executable was not found"
        ) from error

    try:
        relay_gateway.relay_cli_contract(executable)
        plugin_config = common_utils.load_relay_plugin_config(
            {
                "agent_name": agent_name,
                "base_dir": base_dir,
                "config": config.to_mapping(),
                "runtime_context": context.to_mapping(),
            }
        )
        config_path, plugin_config_path = common_utils.write_relay_configs(
            # Codex execution remains SDK-owned; Relay runs only as a gateway.
            relay_config={},
            plugin_config=plugin_config,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        raise AdapterRelayError(
            "codex_relay_configuration_failed",
            "NeMo Relay runtime configuration is unavailable",
        ) from error
    if config_path is None or plugin_config_path is None:
        raise AdapterRelayError(
            "codex_relay_configuration_failed",
            "NeMo Relay runtime configuration is unavailable",
        )

    base_url = _selected_model_config(config).base_url
    port = relay_gateway.find_available_tcp_port()
    bind = f"127.0.0.1:{port}"
    return CodexRelaySettings(
        gateway=relay_gateway.RelayGatewayLaunch(
            executable=executable,
            config_path=config_path,
            bind=bind,
            url=f"http://{bind}",
            log_path=config_path.parent / "gateway.log",
            openai_base_url=base_url.rstrip("/") if base_url else None,
        ),
        plugin_config=plugin_config,
    )


def thread_config(
    config: AgentConfig,
    context: RuntimeContext,
    relay: CodexRelaySettings | None,
) -> dict[str, Any]:
    """Build request-scoped Codex config without writing a user profile."""

    result = native_codex_telemetry_config(context)
    _merge_config(result, custom_model_provider_config(config, context))
    _merge_config(result, openai_model_provider_config(config))
    mcp_servers = _native_mcp_servers(config)
    if mcp_servers:
        result["mcp_servers"] = mcp_servers
    # Keep the pre-0.154 behavior of waiting for configured MCP servers.
    result["mcp_optional_startup_grace_ms"] = 0
    if callback_url := _mcp_oauth_callback_url(config):
        result["mcp_oauth_callback_url"] = callback_url
    overrides = _mapping(
        _settings(config).get("config_overrides"),
        name="harness.settings.config_overrides",
    )
    _apply_config_overrides(result, overrides)
    if relay is not None:
        provider = _selected_model_config(config).provider
        transport_config = (
            {"openai_base_url": relay.gateway.url}
            if provider == "openai"
            else {
                "model_providers": {
                    provider: {
                        "base_url": relay.gateway.url,
                    }
                }
            }
        )
        _merge_config(
            result,
            {
                **transport_config,
                "features": {
                    "hooks": True,
                    # Relay disables delegated multi-agent execution because
                    # Codex encrypts delegated task content before it reaches
                    # the gateway, making those spans opaque.
                    "multi_agent_v2": {"enabled": False},
                },
                "hooks": relay_hooks.render_relay_hooks(
                    "codex", relay.gateway.executable
                )["hooks"],
                # This runtime-only request override is the SDK-native equivalent
                # of the former non-interactive CLI flag. Fabric generated and
                # vetted every hook command above.
                "bypass_hook_trust": True,
            },
        )
    return result


def sdk_config(
    config: AgentConfig,
    context: RuntimeContext,
    base_dir: str,
    relay: CodexRelaySettings | None,
) -> CodexConfig:
    codex_bin = os.environ.get("FABRIC_TEST_CODEX_BIN")
    if codex_bin:
        path = Path(codex_bin)
        if not path.is_absolute():
            path = (Path(base_dir) / path).resolve()
        codex_bin = str(path)
    return CodexConfig(
        codex_bin=codex_bin,
        cwd=str(resolve_cwd(context, base_dir)),
        env=child_environment(
            config,
            context,
            base_dir,
            relay_gateway_url=relay.gateway.url if relay is not None else None,
        ),
    )


def _personality(config: AgentConfig) -> Personality | None:
    value = _optional_string(_settings(config), "personality")
    if value is None:
        return None
    try:
        return Personality(value)
    except ValueError as error:
        raise AdapterConfigError(
            "codex_invalid_configuration", "personality is invalid"
        ) from error


def _reasoning_effort(config: AgentConfig) -> ReasoningEffort | None:
    value = _optional_string(_settings(config), "reasoning_effort")
    if value is None:
        return None
    try:
        return ReasoningEffort(value)
    except ValueError as error:
        raise AdapterConfigError(
            "codex_invalid_configuration", "reasoning_effort is invalid"
        ) from error


def _output_schema(config: AgentConfig) -> dict[str, Any] | None:
    value = _settings(config).get("output_schema")
    if value is None:
        return None
    return _mapping(_json_value(value, name="output_schema"), name="output_schema")


def validate_runtime_payload(
    config: AgentConfig, context: RuntimeContext, base_dir: str
) -> str:
    """Validate runtime-owned configuration before starting SDK or Relay processes."""

    settings = _settings(config)
    _native_skill_paths(config, base_dir)
    fabric_runtime_id = context.runtime_id
    resolve_cwd(context, base_dir)
    selected_model(config)
    sandbox(config)
    approval_mode(config)
    for name in (
        "developer_instructions",
        "service_tier",
    ):
        _optional_string(settings, name)
    _personality(config)
    _reasoning_effort(config)
    _output_schema(config)
    child_environment(config, context, base_dir)
    thread_config(config, context, None)
    return fabric_runtime_id


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json", by_alias=True))
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise AdapterConfigError(
        "codex_invalid_configuration", "Codex SDK result is not JSON-safe"
    )


def _failure(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    **metadata: Any,
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "retryable": retryable,
    }
    if metadata:
        error["metadata"] = metadata
    return {
        "harness": "codex",
        "adapter": "sdk",
        "mode": "codex_sdk_runtime",
        "response": None,
        "completed": False,
        "failed": True,
        "error": error,
        "events": [],
    }


def adapter_failure(error: CodexAdapterError) -> dict[str, Any]:
    return _failure(error.code, error.message, **error.metadata)


def sdk_failure(error: BaseException) -> dict[str, Any]:
    if isinstance(error, TimeoutError):
        # Only Fabric's invocation deadline is reported as `timeout`.
        return _failure("codex_timed_out", "Codex SDK request timed out")
    if isinstance(error, TransportClosedError):
        return _failure(
            "codex_connection_failed", "Codex SDK runtime connection closed"
        )
    if isinstance(error, CodexError):
        metadata = {"sdk_error": type(error).__name__}
        if isinstance(error, JsonRpcError):
            metadata["jsonrpc_code"] = error.code
        return _failure(
            "codex_sdk_failed",
            "Codex SDK request failed",
            retryable=is_retryable_error(error),
            **metadata,
        )
    if isinstance(error, OSError):
        return _failure(
            "codex_runtime_unavailable", "Codex SDK runtime could not start"
        )
    return _failure(
        "codex_turn_failed",
        str(error) or "Codex turn failed",
    )


def _response_usage(value: Any) -> ReportedUsage:
    """Read the pinned app-server's original Responses usage before zero filling."""
    if not isinstance(value, dict):
        return ReportedUsage()
    paths = {
        "input_tokens": ("input_tokens",),
        "cached_input_tokens": ("input_tokens_details", "cached_tokens"),
        "cache_write_input_tokens": ("input_tokens_details", "cache_write_tokens"),
        "output_tokens": ("output_tokens",),
        "reasoning_tokens": ("output_tokens_details", "reasoning_tokens"),
        "total_tokens": ("total_tokens",),
    }
    counts = {}
    for name, path in paths.items():
        count = value
        for key in path:
            count = count.get(key) if isinstance(count, dict) else None
        if type(count) is int and 0 <= count <= (1 << 64) - 1:
            counts[name] = count
    return ReportedUsage(
        **counts,
        input_tokens_include_cache=True if "input_tokens" in counts else None,
        output_tokens_include_reasoning=True if "output_tokens" in counts else None,
    )


async def read_cli_version(
    thread: Any, timeout: float = provenance.PROBE_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """The version of the Codex CLI that created ``thread``, from its thread data.

    One bounded ``thread/read`` request: a timeout reads ``collection_failed``
    with reason ``timeout``, and a failed request reads reason ``no_result``.
    """

    try:
        async with asyncio.timeout(timeout):
            response = await thread.read()
    except TimeoutError:
        return provenance.missing_fact(
            "collection_failed", "timeout", _METHOD_HARNESS_API
        )
    except Exception:
        # Provenance failures must not prevent execution; cancellation still propagates.
        return provenance.missing_fact(
            "collection_failed", "no_result", _METHOD_HARNESS_API
        )
    version = getattr(getattr(response, "thread", None), "cli_version", None)
    if not isinstance(version, str) or not version:
        return provenance.missing_fact(
            "unavailable", "empty_response", _METHOD_HARNESS_API
        )
    return provenance.known_fact(version, _METHOD_HARNESS_API)


def _agent_run_result(output: dict[str, Any]) -> AgentRunResult:
    normalized = dict(output)
    failed = bool(normalized.pop("failed", False))
    usage_complete = normalized.pop("usage_complete", True)
    source_gaps = normalized.pop("usage_gaps", [])
    observations = normalized.pop("usage_observations", [])
    reported_error = normalized.pop("error", None)
    error = None
    if failed:
        reported = reported_error if isinstance(reported_error, dict) else {}
        metadata = reported.get("metadata")
        error = AgentRunError(
            code=str(reported.get("code") or "codex_turn_failed"),
            message=str(reported.get("message") or "Codex invocation failed"),
            retryable=bool(reported.get("retryable", False)),
            extensions=metadata if isinstance(metadata, dict) else {},
        )
    # Frozen per-response records are authoritative. Thread totals can conceal
    # omitted reports and are retained only as native diagnostics.
    observations = list(observations)
    if not usage_complete:
        source_gaps = [*source_gaps, GapReason.UNFINISHED_WORK.value]
    observations.extend(
        UsageObservation(reported=ReportedUsage(), gap=GapReason(reason))
        for reason in source_gaps
    )
    agent_usage = (
        usage_from_observations(observations)
        if observations or "turn_id" in normalized or "usage" in normalized
        else None
    )
    return AgentRunResult(
        status=AgentRunStatus.FAILED if failed else AgentRunStatus.SUCCEEDED,
        output=normalized,
        error=error,
        usage=agent_usage,
    )


def normalize_result(
    config: AgentConfig,
    context: RuntimeContext,
    base_dir: str,
    *,
    thread_id: str,
    result: Any,
) -> dict[str, Any]:
    status = _json_safe(result.status)
    completed = (
        result.status == TurnStatus.completed and result.final_response is not None
    )
    error = None
    if not completed:
        message = (
            result.error.message
            if result.error is not None
            else "Codex invocation did not return a final response"
        )
        error = {
            "code": "codex_turn_incomplete",
            "message": message,
            "retryable": False,
            "metadata": {"status": status},
        }
    return {
        "harness": "codex",
        "adapter": "sdk",
        "mode": "codex_sdk_runtime",
        "cwd": str(resolve_cwd(context, base_dir)),
        "model": selected_model(config),
        "thread_id": thread_id,
        "turn_id": result.id,
        "turn_status": status,
        "response": result.final_response,
        "usage": _json_safe(result.usage),
        "started_at": result.started_at,
        "completed_at": result.completed_at,
        "duration_ms": result.duration_ms,
        "completed": completed,
        "failed": not completed,
        "error": error,
        "events": [_json_safe(item) for item in result.items],
        "state_dir": str(state_dir(context, base_dir)),
    }


@dataclass
class _TurnUsage:
    """Invocation-owned per-response records and the native thread diagnostic."""

    snapshot: ThreadTokenUsage | None = None
    # Codex ends every response of a turn before it sends turn/completed.
    turn_completed: bool = False
    gaps: set[str] = field(default_factory=set)
    responses: dict[str, UsageObservation] = field(default_factory=dict)

    def annotate(self, output: dict[str, Any]) -> dict[str, Any]:
        output["usage_observations"] = tuple(self.responses.values())
        if not self.responses:
            self.gaps.add(GapReason.SCOPE_UNVERIFIED.value)
        if self.gaps:
            output["usage_gaps"] = sorted(self.gaps)
        return output


def _fills_context_window(usage: ThreadTokenUsage) -> bool:
    """Detect Codex's context-window marker, which replaces the counted thread totals.

    On a context-window overflow, Codex sets the thread total to the window size with
    every other counter at zero and then fails the turn.
    """
    total = usage.total
    return (
        total.total_tokens == usage.model_context_window
        and total.input_tokens == 0
        and total.output_tokens == 0
    )


async def _turn_notifications(
    handle: Any, usage: _TurnUsage
) -> AsyncIterator[Notification]:
    stream = handle.stream()
    try:
        async for event in stream:
            payload = event.payload
            if (
                isinstance(payload, ThreadTokenUsageUpdatedNotification)
                and payload.turn_id == handle.id
                and not _fills_context_window(payload.token_usage)
            ):
                usage.snapshot = payload.token_usage
            elif (
                isinstance(payload, TurnCompletedNotification)
                and payload.turn.id == handle.id
            ):
                usage.turn_completed = True
            elif (
                event.method == "rawResponse/completed"
                and isinstance(payload, UnknownNotification)
                and payload.params.get("turnId") == handle.id
            ):
                # The pinned SDK routes this internal app-server event as unknown.
                # Unlike thread totals, it exposes a completed response with no usage.
                response = RawResponseCompletedNotification.model_validate(
                    payload.params
                )
                metadata = response.usage_metadata
                raw_usage = metadata.metadata if metadata is not None else None
                record = UsageObservation(
                    reported=_response_usage(raw_usage),
                    gap=GapReason.MISSING_USAGE
                    if not isinstance(raw_usage, dict)
                    else None,
                    tokens_aligned=True,
                    is_model_call=True,
                )
                previous = usage.responses.get(response.response_id)
                if previous is None:
                    usage.responses[response.response_id] = record
                elif previous != record:
                    usage.gaps.add(GapReason.SCOPE_UNVERIFIED.value)
            elif (
                isinstance(payload, ErrorNotification) and payload.turn_id == handle.id
            ):
                # A model error can be retried inside a successful turn. Terminal
                # success does not establish usage for the unsuccessful attempt.
                info = _json_safe(payload.error.codex_error_info)
                if payload.will_retry or info not in (
                    "sandboxError",
                    "threadRollbackFailed",
                ):
                    usage.gaps.add(GapReason.MISSING_USAGE.value)
            elif (
                isinstance(
                    payload, (ItemStartedNotification, ItemCompletedNotification)
                )
                and payload.turn_id == handle.id
            ):
                item = getattr(payload.item, "root", payload.item)
                if (
                    getattr(item, "type", None) == "collabAgentToolCall"
                    and item.receiver_thread_ids
                    and _json_safe(item.tool)
                    in {
                        "spawnAgent",
                        "sendInput",
                        "resumeAgent",
                        "sendMessage",
                        "followupTask",
                    }
                ):
                    # This parent-turn stream does not establish coverage of
                    # model calls in the dispatched child's separate thread.
                    usage.gaps.add(GapReason.SCOPE_UNVERIFIED.value)
            yield event
    finally:
        await stream.aclose()


async def _run_turn(handle: Any, usage: _TurnUsage) -> TurnResult:
    """Collect the turn as ``AsyncTurnHandle.run`` does, keeping usage if the turn fails."""
    events = _turn_notifications(handle, usage)
    try:
        return await _collect_async_turn_result(events, turn_id=handle.id)
    finally:
        await events.aclose()


async def _interrupt_turn(handle: Any) -> None:
    if handle is None:
        return
    try:
        async with asyncio.timeout(INTERRUPT_TIMEOUT_SECONDS):
            await handle.interrupt()
    except (TimeoutError, CodexError, RuntimeError, OSError):
        # The SDK process is closed immediately afterwards, which is the final
        # cancellation boundary if the runtime cannot acknowledge interrupt.
        pass


def _thread_options(
    config: AgentConfig,
    context: RuntimeContext,
    base_dir: str,
    relay: CodexRelaySettings | None,
) -> dict[str, Any]:
    settings = _settings(config)
    instruction = common_instructions.system_instruction(
        config,
        adapter="Codex",
        supported_modes={"replace"},
    )
    return {
        "approval_mode": approval_mode(config),
        "base_instructions": instruction.content if instruction else None,
        "config": thread_config(config, context, relay) or None,
        "cwd": str(resolve_cwd(context, base_dir)),
        "developer_instructions": _optional_string(settings, "developer_instructions"),
        "model": selected_model(config),
        "model_provider": _selected_model_config(config).provider,
        "personality": _personality(config),
        "sandbox": sandbox(config),
        "service_tier": _optional_string(settings, "service_tier"),
    }


def _thread_start_params(
    config: AgentConfig,
    context: RuntimeContext,
    base_dir: str,
    relay: CodexRelaySettings | None,
) -> dict[str, Any]:
    """Use generated fields, plus the pinned observational raw-event flag."""
    options = _thread_options(config, context, base_dir, relay)
    mode = options.pop("approval_mode")
    sandbox_mode = options.pop("sandbox")
    params = ThreadStartParams.model_validate(
        {
            **options,
            "approval_policy": "never"
            if mode is ApprovalMode.deny_all
            else "on-request",
            "approvals_reviewer": None
            if mode is ApprovalMode.deny_all
            else "auto_review",
            "sandbox": "danger-full-access"
            if sandbox_mode is Sandbox.full_access
            else sandbox_mode.value,
        }
    ).model_dump(mode="json", by_alias=True, exclude_none=True)
    # SDK0.154 omits this app-server field. It changes notifications only.
    params["experimentalRawEvents"] = True
    return params


async def _open_thread(
    codex: AsyncCodex,
    config: AgentConfig,
    context: RuntimeContext,
    base_dir: str,
    *,
    relay: CodexRelaySettings | None,
) -> AsyncThread:
    params = _thread_start_params(config, context, base_dir, relay)
    started = await _codex_protocol_client(codex).request(
        "thread/start", params, response_model=ThreadStartResponse
    )
    return AsyncThread(codex, started.thread.id)


async def _invoke_thread(
    config: AgentConfig,
    context: RuntimeContext,
    base_dir: str,
    request: AgentRunRequest,
    thread: Any,
    invocation_deadline: float,
) -> tuple[dict[str, Any], bool]:
    """Run one turn and report whether the connected SDK transport remains usable."""

    handle = None
    usage = _TurnUsage()
    deadline = asyncio.timeout_at(invocation_deadline)
    try:
        async with deadline:
            handle = await thread.turn(
                request_prompt(request),
                effort=_reasoning_effort(config),
                output_schema=_output_schema(config),
            )
            result = await _run_turn(handle, usage)
            return (
                usage.annotate(
                    normalize_result(
                        config, context, base_dir, thread_id=thread.id, result=result
                    )
                ),
                True,
            )
    except TimeoutError as error:
        await _interrupt_turn(handle)
        if deadline.expired():
            failure = _failure("timeout", "Codex invocation timed out")
        else:
            failure = sdk_failure(error)
    except CodexAdapterError:
        raise
    except (CodexError, RuntimeError, OSError) as error:
        failure = sdk_failure(error)
    if usage.snapshot is not None:
        failure["usage"] = _json_safe(usage.snapshot)
        # Codex sends a response's usage after its tool calls finish. Without
        # turn/completed, a response may still have been running, so the last
        # snapshot is not the invocation's complete total.
    failure["usage_complete"] = usage.turn_completed
    return usage.annotate(failure), False


def _relay_output(
    output: dict[str, Any],
    relay: CodexRelaySettings,
    *,
    artifacts: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    output["relay_runtime"] = {
        "enabled": True,
        "emitter": "codex-sdk/nemo-relay",
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
    context: RuntimeContext,
    base_dir: str,
    relay: CodexRelaySettings | None,
) -> subprocess.Popen[Any] | None:
    if relay is None:
        return None
    try:
        return relay_gateway.start_relay_gateway(
            launch=relay.gateway, cwd=resolve_cwd(context, base_dir)
        )
    except relay_gateway.RelayGatewayError as error:
        raise AdapterRelayError(
            "codex_relay_start_failed",
            "NeMo Relay gateway failed to start",
            metadata={"gateway_log_path": str(relay.gateway.log_path)},
        ) from error


def _cleanup_relay(
    relay: CodexRelaySettings | None,
    process: subprocess.Popen[Any] | None,
) -> AdapterRelayError | None:
    if process is None:
        return None
    try:
        relay_gateway.stop_relay_gateway(process)
    except relay_gateway.RelayGatewayError:
        return AdapterRelayError(
            "codex_relay_stop_failed",
            "NeMo Relay gateway failed to stop",
            metadata={
                "gateway_log_path": str(relay.gateway.log_path)
                if relay is not None
                else ""
            },
        )
    return None


def _as_lifecycle_error(error: CodexAdapterError) -> lifecycle.LifecycleError:
    return lifecycle.LifecycleError(
        error.code,
        error.message,
        metadata=error.metadata,
    )


def _runtime_context(payload: dict[str, Any]) -> RuntimeContext:
    try:
        return RuntimeContext.from_mapping(payload.get("runtime_context"))
    except ContractValidationError as error:
        raise lifecycle.LifecycleError(
            "codex_invalid_runtime_context",
            "Codex runtime context is invalid",
        ) from error


class CodexRuntime:
    """One Codex app-server client and thread owned by a Fabric runtime."""

    def __init__(self) -> None:
        self._config: AgentConfig | None = None
        self._context: RuntimeContext | None = None
        self._base_dir: str | None = None
        self._fabric_runtime_id: str | None = None
        self._client: AsyncCodex | None = None
        self._thread: Any = None
        self._relay: CodexRelaySettings | None = None
        self._gateway_process: subprocess.Popen[Any] | None = None
        self._api_key_home: tempfile.TemporaryDirectory | None = None
        self._mcp_authentication_checked = False
        self._unusable = False
        self._harness_version = provenance.missing_fact(
            "unavailable", "not_collected", _METHOD_HARNESS_API
        )
        self._package_version = provenance.package_version_fact(_PACKAGE_DISTRIBUTION)

    async def start(self, payload: dict[str, Any]) -> None:
        if self._client is not None:
            raise lifecycle.LifecycleError(
                "codex_runtime_already_started",
                "Codex runtime is already started",
            )

        try:
            agent_config = payload["config"]
            context = _runtime_context(payload)
            base_dir = common_utils.base_dir(payload)
            fabric_runtime_id = validate_runtime_payload(
                agent_config, context, base_dir
            )
            relay = prepare_codex_relay(
                common_utils.agent_name(payload), agent_config, context, base_dir
            )
            self._relay = relay
            self._gateway_process = _start_relay_gateway(context, base_dir, relay)
            client_config = sdk_config(agent_config, context, base_dir, relay)
            model_config = _selected_model_config(agent_config)
            api_key_env = model_config.api_key_env
            api_key = None
            if model_config.provider == "openai" and api_key_env is not None:
                api_key = client_config.env.get(api_key_env)
                if not api_key:
                    raise AdapterConfigError(
                        "codex_invalid_configuration",
                        f"{api_key_env} is required for Codex API-key authentication",
                    )
                self._api_key_home = tempfile.TemporaryDirectory(
                    prefix="nemo-fabric-codex-"
                )
                private_home = Path(self._api_key_home.name).resolve()
                if private_home.is_relative_to(
                    _artifact_root(context, base_dir).resolve()
                ):
                    raise AdapterConfigError(
                        "codex_invalid_configuration",
                        "Temporary Codex credential directory must be outside artifacts",
                    )
                client_config.env["CODEX_HOME"] = str(private_home)
            elif model_config.provider != "openai":
                await asyncio.to_thread(
                    Path(client_config.env["CODEX_HOME"]).mkdir,
                    parents=True,
                    exist_ok=True,
                )
            client = AsyncCodex(config=client_config)
            self._client = client
            await client.__aenter__()
            if api_key is not None:
                await client.login_api_key(api_key)
            await _register_skill_roots(
                client, _native_skill_paths(agent_config, base_dir)
            )
            thread = await _open_thread(
                client,
                agent_config,
                context,
                base_dir,
                relay=relay,
            )
            harness_version = await read_cli_version(thread)
        except CodexAdapterError as error:
            await self._cleanup_failed_start()
            raise _as_lifecycle_error(error) from error
        except (CodexError, RuntimeError, OSError) as error:
            await self._cleanup_failed_start()
            reported = sdk_failure(error)["error"]
            raise lifecycle.LifecycleError(
                reported["code"],
                reported["message"],
                retryable=reported["retryable"],
                metadata=reported.get("metadata"),
            ) from error
        except BaseException:
            await self._cleanup_failed_start()
            raise

        self._config = agent_config
        self._context = context
        self._base_dir = base_dir
        self._fabric_runtime_id = fabric_runtime_id
        self._thread = thread
        self._harness_version = harness_version

    def _versions_extensions(self) -> dict[str, Any]:
        return {
            provenance.VERSIONS_REPORT_KEY: provenance.versions_report(
                package_version=self._package_version,
                harness_name=provenance.known_fact(
                    "codex", provenance.METHOD_ADAPTER_REPORT
                ),
                harness_version=self._harness_version,
                served_models={
                    role: provenance.no_source_fact()
                    for role in (
                        self._config.models if self._config is not None else ()
                    )
                },
            )
        }

    async def invoke(
        self,
        request: AgentRunRequest,
        runtime_context: RuntimeContext,
    ) -> AgentRunResult:
        result = await self._invoke(request, runtime_context)
        result.extensions.update(self._versions_extensions())
        return result

    async def _invoke(
        self,
        request: AgentRunRequest,
        runtime_context: RuntimeContext,
    ) -> AgentRunResult:
        if (
            self._config is None
            or self._context is None
            or self._base_dir is None
            or self._client is None
            or self._thread is None
            or self._fabric_runtime_id is None
        ):
            raise lifecycle.LifecycleError(
                "codex_runtime_not_started",
                "Codex runtime is not started",
            )
        config = self._config
        context = self._context
        base_dir = self._base_dir
        if runtime_context.runtime_id != self._fabric_runtime_id:
            raise lifecycle.LifecycleError(
                "codex_runtime_mismatch",
                "Codex invocation does not match the connected runtime",
            )
        if self._unusable:
            return _agent_run_result(
                _failure(
                    "codex_runtime_unavailable",
                    "Codex runtime cannot accept another invocation after a runtime failure",
                ),
            )

        invocation_deadline = lifecycle.invocation_deadline(runtime_context)
        turn_requested = False
        try:
            request_prompt(request)
            _reasoning_effort(config)
            _output_schema(config)
            if not self._mcp_authentication_checked:
                await _authenticate_mcp_servers(
                    self._client,
                    self._thread,
                    config,
                    invocation_deadline,
                )
                self._mcp_authentication_checked = True
            relay = self._relay
            atif_before = (
                relay_artifacts.snapshot_atif_files(relay.plugin_config)
                if relay is not None
                and relay_artifacts.expects_local_atif(relay.plugin_config)
                else None
            )
            turn_requested = True
            # Only a fully handled turn result can restore reuse. Unexpected
            # SDK exceptions and cancellation must leave this runtime quarantined.
            self._unusable = True
            output, usable = await _invoke_thread(
                config,
                runtime_context,
                base_dir,
                request,
                self._thread,
                invocation_deadline,
            )
            if (
                output.get("completed")
                and relay is not None
                and atif_before is not None
            ):
                atif_timeout = min(
                    relay_artifacts.ATIF_FINALIZATION_TIMEOUT_SECONDS,
                    _remaining_timeout(invocation_deadline),
                )
                finalized = await relay_artifacts.wait_for_finalized_atif(
                    relay.plugin_config, atif_before, timeout_seconds=atif_timeout
                )
                if finalized is None:
                    self._unusable = True
                    failure = adapter_failure(
                        AdapterRelayError(
                            "codex_relay_atif_timeout",
                            "NeMo Relay did not finalize an ATIF artifact before the deadline",
                            metadata={"timeout_seconds": atif_timeout},
                        )
                    )
                    failure["usage"] = output.get("usage")
                    for key in (
                        "usage_complete",
                        "usage_gaps",
                        "usage_observations",
                    ):
                        if key in output:
                            failure[key] = output[key]
                    return _agent_run_result(
                        _relay_output(failure, relay, artifacts=[])
                    )
        except AdapterRelayError as error:
            output = adapter_failure(error)
            usable = False
        except CodexAdapterError as error:
            output = adapter_failure(error)
            usable = not turn_requested

        self._unusable = not usable
        if self._relay is not None:
            output = _relay_output(output, self._relay)
        return _agent_run_result(output)

    async def stop(self) -> None:
        client = self._client
        self._client = None
        self._config = None
        self._context = None
        self._base_dir = None
        self._thread = None
        self._fabric_runtime_id = None
        self._mcp_authentication_checked = False
        self._unusable = True

        close_error: BaseException | None = None
        try:
            if client is not None:
                await client.close()
        except BaseException as error:
            if not isinstance(error, asyncio.CancelledError):
                LOGGER.exception("Codex SDK client failed to close")
            close_error = error
        finally:
            cleanup_error = _cleanup_relay(self._relay, self._gateway_process)
            self._relay = None
            self._gateway_process = None
            if self._api_key_home is not None:
                self._api_key_home.cleanup()
                self._api_key_home = None

        if isinstance(close_error, asyncio.CancelledError):
            raise close_error
        if close_error is not None:
            if cleanup_error is not None:
                LOGGER.error(
                    "Codex Relay cleanup also failed during close: %s",
                    cleanup_error.code,
                )
            raise lifecycle.LifecycleError(
                "codex_sdk_stop_failed",
                "Codex SDK runtime failed to stop",
            ) from close_error
        if cleanup_error is not None:
            raise _as_lifecycle_error(cleanup_error)

    async def _cleanup_failed_start(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            try:
                await client.close()
            except Exception:
                LOGGER.exception("Codex SDK cleanup after start failure also failed")
        cleanup_error = _cleanup_relay(self._relay, self._gateway_process)
        self._relay = None
        self._gateway_process = None
        if self._api_key_home is not None:
            self._api_key_home.cleanup()
            self._api_key_home = None
        if cleanup_error is not None:
            LOGGER.error(
                "Codex Relay cleanup after start failure also failed: %s",
                cleanup_error.code,
            )


def main() -> None:
    """Serve the persistent local-host lifecycle protocol."""

    lifecycle.serve(CodexRuntime, config_loader=AgentConfig.from_mapping)


if __name__ == "__main__":
    main()
