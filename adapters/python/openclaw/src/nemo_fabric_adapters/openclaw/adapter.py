#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch and invoke an isolated OpenClaw Gateway for one NeMo Fabric runtime."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import secrets
import shutil
import signal
import socket
import sys
import tempfile
import threading
from collections import deque
from pathlib import Path
from types import FrameType
from typing import Any

import httpx
from nemo_fabric_adapter_contract import models as contract
from nemo_fabric_adapters.common import instructions as common_instructions
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common import openai_chat
from nemo_fabric_adapters.common import utils as common_utils
from nemo_fabric_adapters.openclaw import _windows_job


DEFAULT_STARTUP_TIMEOUT_SECONDS = 30.0
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 10.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_READ_TIMEOUT_SECONDS = 600.0
OPENCLAW_CHAT_MODEL = "openclaw/default"
PLUGIN_ID = "nemo-relay"
RANDOM_PORT_MIN = 20_000
RANDOM_PORT_MAX = 64_000
MAX_PORT_ATTEMPTS = 20
SUPERVISOR_MODULE = "nemo_fabric_adapters.openclaw._supervisor"
WINDOWS_START_BYTE = b"\x01"


logger = logging.getLogger(__name__)


def _selected_model(config: contract.AgentConfig) -> contract.AgentModelConfig:
    model = config.models.get("default")
    if model is None and len(config.models) == 1:
        model = next(iter(config.models.values()))
    if model is None:
        raise lifecycle.LifecycleError(
            "openclaw_missing_model",
            "OpenClaw requires a default model or exactly one model",
        )
    return model


def _positive_setting(settings: dict[str, Any], name: str, default: float) -> float:
    value = settings.get(name, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise lifecycle.LifecycleError(
            "openclaw_invalid_configuration",
            f"OpenClaw {name} must be a positive finite number",
            metadata={"field": f"harness.settings.{name}"},
        )
    return float(value)


def _resolve_command(settings: dict[str, Any], base_dir: Path) -> Path:
    configured = settings.get("openclaw_command", "openclaw")
    if not isinstance(configured, str) or not configured.strip():
        raise lifecycle.LifecycleError(
            "openclaw_invalid_configuration",
            "OpenClaw openclaw_command must be a non-empty string",
            metadata={"field": "harness.settings.openclaw_command"},
        )
    value = configured.strip()
    candidate: Path | None
    if Path(value).is_absolute():
        candidate = Path(value)
    elif len(Path(value).parts) > 1:
        candidate = base_dir / value
    else:
        found = shutil.which(value)
        candidate = Path(found) if found else None
    if (
        candidate is None
        or not candidate.is_file()
        or not os.access(candidate, os.X_OK)
    ):
        raise lifecycle.LifecycleError(
            "openclaw_command_not_found",
            f"OpenClaw command {configured!r} was not found or is not executable",
            metadata={"field": "harness.settings.openclaw_command"},
        )
    return candidate.resolve()


def _port_available(port: int) -> bool:
    sockets: list[socket.socket] = []
    try:
        for candidate in (port, port + 2):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(sock)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", candidate))
        for candidate in range(port + 11, port + 111):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", candidate))
            except OSError:
                sock.close()
                continue
            sockets.append(sock)
            return True
        return False
    except OSError:
        return False
    finally:
        for sock in sockets:
            sock.close()


def _select_port(settings: dict[str, Any]) -> int:
    configured = settings.get("port")
    if configured is not None:
        if isinstance(configured, bool) or not isinstance(configured, int):
            raise lifecycle.LifecycleError(
                "openclaw_invalid_configuration",
                "OpenClaw port must be an integer",
                metadata={"field": "harness.settings.port"},
            )
        if not _port_available(configured):
            raise lifecycle.LifecycleError(
                "openclaw_port_unavailable",
                f"OpenClaw port range derived from {configured} is unavailable",
                metadata={"port": configured},
            )
        return configured
    for _ in range(MAX_PORT_ATTEMPTS):
        candidate = RANDOM_PORT_MIN + secrets.randbelow(
            RANDOM_PORT_MAX - RANDOM_PORT_MIN + 1
        )
        if _port_available(candidate):
            return candidate
    raise lifecycle.LifecycleError(
        "openclaw_port_unavailable",
        "OpenClaw could not find an available local port range",
        retryable=True,
    )


def _mcp_oauth_config(
    name: str, authentication: contract.McpAuthenticationConfig
) -> dict[str, str]:
    if isinstance(authentication, contract.McpServiceAccountConfig):
        raise lifecycle.LifecycleError(
            "openclaw_unsupported_mcp_authentication",
            f"OpenClaw MCP server {name!r} does not support service_account authentication",
            metadata={"field": f"mcp.servers.{name}.authentication.type"},
        )

    unsupported = {
        "enable_dynamic_registration": not authentication.enable_dynamic_registration,
        "client_secret_env": authentication.client_secret_env is not None,
        "client_id": authentication.client_id is not None,
        "client_name": authentication.client_name is not None,
        "token_endpoint_auth_method": authentication.token_endpoint_auth_method
        is not None,
        "authorization_timeout_seconds": authentication.authorization_timeout_seconds
        != 300,
    }
    for field, configured in unsupported.items():
        if configured:
            raise lifecycle.LifecycleError(
                "openclaw_unsupported_mcp_authentication",
                f"OpenClaw MCP server {name!r} authentication.{field} is not supported",
                metadata={"field": f"mcp.servers.{name}.authentication.{field}"},
            )

    return {
        key: value
        for key, value in {
            "scope": authentication.scope,
            "redirectUrl": authentication.redirect_uri,
        }.items()
        if value is not None
    }


def _mcp_config(config: contract.AgentConfig) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, server in (config.mcp.servers if config.mcp else {}).items():
        transport = server.transport.strip().lower().replace("_", "-")
        if transport in {"command", "process"}:
            transport = "stdio"
        if transport == "http":
            transport = "streamable-http"
        if transport not in {"stdio", "sse", "streamable-http"}:
            raise lifecycle.LifecycleError(
                "openclaw_unsupported_mcp_transport",
                f"OpenClaw MCP server {name!r} uses unsupported transport {server.transport!r}",
                metadata={"field": f"mcp.servers.{name}.transport"},
            )
        item: dict[str, Any] = {"enabled": True, "transport": transport}
        if transport == "stdio":
            item.update(command=server.url, args=server.args)
            if server.env:
                item["env"] = server.env
        else:
            item["url"] = server.url
            if server.custom_headers:
                common_utils.validate_http_headers(name, server.custom_headers)
                item["headers"] = server.custom_headers
        if server.authentication is not None:
            if transport == "stdio":
                raise lifecycle.LifecycleError(
                    "openclaw_unsupported_mcp_authentication",
                    f"OpenClaw MCP server {name!r} cannot use authentication with stdio transport",
                    metadata={"field": f"mcp.servers.{name}.authentication"},
                )
            item["auth"] = "oauth"
            item["oauth"] = _mcp_oauth_config(name, server.authentication)
        tool_filter: dict[str, list[str]] = {}
        if server.allowed_tools is not None:
            tool_filter["include"] = server.allowed_tools
        if server.blocked_tools:
            tool_filter["exclude"] = server.blocked_tools
        if tool_filter:
            item["toolFilter"] = tool_filter
        result[name] = item
    return result


def _relay_plugin_config(plugin_config: dict[str, Any]) -> dict[str, Any]:
    for component in plugin_config.get("components", []):
        if not isinstance(component, dict) or not component.get("enabled", True):
            continue
        if component.get("kind") != "observability":
            continue
        observability = component.get("config") or {}
        atof = observability.get("atof")
        if isinstance(atof, dict) and atof.get("enabled"):
            raise lifecycle.LifecycleError(
                "openclaw_relay_streaming_unsupported",
                "OpenClaw Relay streaming is not supported by this adapter version",
            )
        otel = observability.get("opentelemetry")
        if isinstance(otel, dict) and otel.get("enabled"):
            raise lifecycle.LifecycleError(
                "openclaw_relay_otel_unsupported",
                "OpenClaw Relay OpenTelemetry is not supported by this adapter version",
            )
    return {
        "enabled": True,
        "backend": "hooks",
        "plugins": plugin_config,
    }


def _openclaw_config(
    config: contract.AgentConfig,
    context: contract.RuntimeContext,
    *,
    base_dir: Path,
    port: int,
    token_env: str,
    relay_root: str | None,
    relay_config: dict[str, Any] | None,
) -> dict[str, Any]:
    model = _selected_model(config)
    model_ref = f"{model.provider}/{model.model}"
    params = {
        key: value
        for key, value in {
            "temperature": model.temperature,
            "topP": model.top_p,
            "maxTokens": model.max_tokens,
        }.items()
        if value is not None
    }
    workspace = context.environment.workspace or "."
    result: dict[str, Any] = {
        "gateway": {
            "mode": "local",
            "port": port,
            "bind": "loopback",
            "auth": {
                "mode": "token",
                "token": {"source": "env", "provider": "default", "id": token_env},
            },
            "controlUi": {"enabled": False},
            "http": {"endpoints": {"chatCompletions": {"enabled": True}}},
        },
        "agents": {
            "defaults": {
                "workspace": str(Path(workspace).resolve()),
                "model": {"primary": model_ref},
                "models": {model_ref: {"params": params}},
            }
        },
        "telemetry": {"enabled": False},
        "update": {"checkOnStart": False},
        "discovery": {"mdns": {"mode": "off"}},
    }
    if config.skills and config.skills.paths:
        result["skills"] = {
            "load": {
                "extraDirs": [
                    str((base_dir / path).resolve()) for path in config.skills.paths
                ]
            }
        }
    if config.tools is not None:
        tools: dict[str, list[str]] = {}
        if config.tools.enabled is not None:
            tools["allow"] = config.tools.enabled
        if config.tools.blocked:
            tools["deny"] = config.tools.blocked
        if tools:
            result["tools"] = tools
    mcp = _mcp_config(config)
    if mcp:
        result["mcp"] = {"servers": mcp}
    provider: dict[str, Any] = {}
    if model.base_url is not None:
        provider.update(
            {
                "baseUrl": model.base_url,
                "api": "openai-completions",
                "models": [{"id": model.model, "name": model.model}],
            }
        )
        if model.max_tokens is not None:
            provider["models"][0]["maxTokens"] = model.max_tokens
    if model.api_key_env is not None:
        provider["apiKey"] = {
            "source": "env",
            "provider": "default",
            "id": model.api_key_env,
        }
    if provider:
        result["models"] = {"providers": {model.provider: provider}}
    if relay_root is not None and relay_config is not None:
        result["plugins"] = {
            "allow": [PLUGIN_ID],
            "load": {"paths": [relay_root]},
            "entries": {
                PLUGIN_ID: {
                    "enabled": True,
                    "hooks": {"allowConversationAccess": True},
                    "config": relay_config,
                }
            },
        }
    return result


async def _command_output(
    command: Path,
    *args: str,
    env: dict[str, str],
    timeout: float,
) -> str:
    process = await asyncio.create_subprocess_exec(
        str(command),
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError as error:
        await _kill_and_reap(process)
        command_context = " ".join(args)
        raise lifecycle.LifecycleError(
            "openclaw_command_timeout",
            f"OpenClaw command timed out after {timeout:g} seconds: {command_context}",
            retryable=True,
            metadata={"command": command_context, "timeout_seconds": timeout},
        ) from error
    except asyncio.CancelledError:
        await _kill_and_reap(process)
        raise
    if process.returncode != 0:
        detail = stderr.decode(errors="replace").strip()
        raise lifecycle.LifecycleError(
            "openclaw_command_failed",
            f"OpenClaw command failed: {' '.join(args)}",
            metadata={"detail": detail[-2000:]},
        )
    return stdout.decode(errors="replace").strip()


async def _kill_and_reap(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    await process.communicate()


async def _relay_plugin_root(
    command: Path, env: dict[str, str], *, timeout: float
) -> str:
    try:
        value = json.loads(
            await _command_output(
                command,
                "plugins",
                "list",
                "--json",
                env=env,
                timeout=timeout,
            )
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise lifecycle.LifecycleError(
            "openclaw_relay_plugin_check_failed",
            "OpenClaw returned an invalid plugin inventory",
        ) from error
    if not isinstance(value, dict) or not isinstance(value.get("plugins"), list):
        raise lifecycle.LifecycleError(
            "openclaw_relay_plugin_check_failed",
            "OpenClaw returned an invalid plugin inventory",
        )
    for plugin in value["plugins"]:
        if not isinstance(plugin, dict):
            continue
        if plugin.get("id") != PLUGIN_ID:
            continue
        root = plugin.get("rootDir")
        if plugin.get("status") == "error" or not isinstance(root, str) or not root:
            break
        return str(Path(root).resolve())
    raise lifecycle.LifecycleError(
        "openclaw_relay_plugin_missing",
        "Relay was requested, but the nemo-relay-openclaw plugin is not installed",
    )


async def _capture_stream(
    stream: asyncio.StreamReader | None, tail: deque[str]
) -> None:
    if stream is None:
        return
    while line := await stream.readline():
        tail.append(line.decode(errors="replace").rstrip())


def _gateway_command(
    command: Path,
    *,
    port: int,
    parent_pid: int,
    shutdown_timeout: float,
) -> list[str]:
    gateway = [
        str(command),
        "gateway",
        "run",
        "--port",
        str(port),
        "--bind",
        "loopback",
        "--auth",
        "token",
    ]
    if sys.platform != "linux" and os.name != "nt":
        return gateway
    return [
        sys.executable,
        "-m",
        SUPERVISOR_MODULE,
        "--parent-pid",
        str(parent_pid),
        "--shutdown-timeout",
        str(shutdown_timeout),
        "--",
        *gateway,
    ]


class OpenClawRuntime:
    """One isolated OpenClaw Gateway owned by a NeMo Fabric runtime."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._config: contract.AgentConfig | None = None
        self._context: contract.RuntimeContext | None = None
        self._command: Path | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._log_tasks: list[asyncio.Task[None]] = []
        self._stderr_tail: deque[str] = deque(maxlen=50)
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._port: int | None = None
        self._token: str | None = None
        self._relay_config: dict[str, Any] | None = None
        self._base_dir: Path | None = None
        self._shutdown_timeout = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS
        self._previous_signal_handlers: dict[int, Any] = {}
        self._signal_handler: Any = None
        self._windows_job: int | None = None

    async def start(self, payload: dict[str, Any]) -> None:
        config: contract.AgentConfig = payload["config"]
        context = contract.RuntimeContext.from_mapping(payload["runtime_context"])
        settings = config.harness.settings if config.harness else {}
        base_dir = Path(common_utils.base_dir(payload)).resolve()
        common_instructions.system_instruction(
            config, adapter="OpenClaw", supported_modes={"replace"}
        )
        model = _selected_model(config)
        child_env = common_utils.virtualenv_subprocess_env()
        child_env.update(context.environment.env)
        if context.telemetry is not None:
            child_env.update(context.telemetry.env)
        if model.api_key_env is not None and not child_env.get(model.api_key_env):
            raise lifecycle.LifecycleError(
                "openclaw_missing_api_key",
                f"OpenClaw API key environment variable {model.api_key_env} is not set",
            )
        command = _resolve_command(settings, base_dir)
        startup_timeout = _positive_setting(
            settings, "startup_timeout_seconds", DEFAULT_STARTUP_TIMEOUT_SECONDS
        )
        version = await _command_output(
            command, "--version", env=child_env, timeout=startup_timeout
        )
        if "openclaw" not in version.lower():
            raise lifecycle.LifecycleError(
                "openclaw_version_check_failed",
                "Configured command did not identify itself as OpenClaw",
            )

        plugin_config: dict[str, Any] | None = None
        relay_config = None
        relay_root = None
        if context.telemetry is not None and context.telemetry.relay_enabled:
            try:
                plugin_config = common_utils.load_relay_plugin_config(payload)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                raise lifecycle.LifecycleError(
                    "openclaw_invalid_relay_configuration",
                    "OpenClaw could not load the Relay configuration",
                ) from error
            relay_config = _relay_plugin_config(plugin_config)
            relay_root = await _relay_plugin_root(
                command, child_env, timeout=startup_timeout
            )

        port = _select_port(settings)
        token = secrets.token_urlsafe(48) # Generate a one-time use token for the OpenClaw gateway
        token_env = "OPENCLAW_GATEWAY_TOKEN"
        temp_dir = tempfile.TemporaryDirectory(prefix="nemo-fabric-openclaw-")
        state_dir = Path(temp_dir.name) / "state"
        state_dir.mkdir(mode=0o700)
        config_path = Path(temp_dir.name) / "openclaw.json"
        generated = _openclaw_config(
            config,
            context,
            base_dir=base_dir,
            port=port,
            token_env=token_env,
            relay_root=relay_root,
            relay_config=relay_config,
        )
        config_path.write_text(json.dumps(generated, indent=2) + "\n", encoding="utf-8")
        config_path.chmod(0o600)
        child_env.update(
            {
                "OPENCLAW_CONFIG_PATH": str(config_path),
                "OPENCLAW_STATE_DIR": str(state_dir),
                "OPENCLAW_CONFIG_READONLY": "1", # Tells OpenClaw to treat the configuration as read-only
                token_env: token,
                "DO_NOT_TRACK": "1", # opt out of tracking
            }
        )
        self._temp_dir = temp_dir
        try:
            self._shutdown_timeout = _positive_setting(
                settings, "shutdown_timeout_seconds", DEFAULT_SHUTDOWN_TIMEOUT_SECONDS
            )
            await _command_output(
                command,
                "config",
                "validate",
                "--json",
                env=child_env,
                timeout=startup_timeout,
            )
            if os.name == "nt":
                try:
                    self._windows_job = _windows_job.create_kill_on_close_job()
                except OSError as error:
                    raise lifecycle.LifecycleError(
                        "openclaw_process_supervision_failed",
                        "OpenClaw could not create a Windows Job Object",
                    ) from error
            gateway_command = _gateway_command(
                command,
                port=port,
                parent_pid=os.getpid(),
                shutdown_timeout=self._shutdown_timeout,
            )
            self._process = await asyncio.create_subprocess_exec(
                *gateway_command,
                stdin=(
                    asyncio.subprocess.PIPE
                    if os.name == "nt"
                    else asyncio.subprocess.DEVNULL
                ),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=child_env,
                start_new_session=os.name != "nt",
            )
            if self._windows_job is not None:
                try:
                    _windows_job.assign_process(self._windows_job, self._process.pid)
                    if self._process.stdin is None:
                        raise RuntimeError("Windows supervisor stdin was unavailable")
                    self._process.stdin.write(WINDOWS_START_BYTE)
                    await self._process.stdin.drain()
                    self._process.stdin.close()
                except (OSError, RuntimeError) as error:
                    raise lifecycle.LifecycleError(
                        "openclaw_process_supervision_failed",
                        "OpenClaw could not join the Windows Job Object",
                    ) from error
            self._log_tasks = [
                asyncio.create_task(
                    _capture_stream(self._process.stdout, deque(maxlen=10))
                ),
                asyncio.create_task(
                    _capture_stream(self._process.stderr, self._stderr_tail)
                ),
            ]
            self._install_signal_handlers()
            await self._wait_ready(port, token, startup_timeout)
            if relay_config is not None:
                await _command_output(
                    command,
                    "gateway",
                    "call",
                    "nemoRelay.status",
                    "--port",
                    str(port),
                    "--json",
                    env=child_env,
                    timeout=startup_timeout,
                )
            self._client = httpx.AsyncClient(
                headers={"authorization": f"Bearer {token}"},
                http2=True,
                trust_env=False,
                timeout=httpx.Timeout(
                    connect=_positive_setting(
                        settings,
                        "connect_timeout_seconds",
                        DEFAULT_CONNECT_TIMEOUT_SECONDS,
                    ),
                    read=_positive_setting(
                        settings, "read_timeout_seconds", DEFAULT_READ_TIMEOUT_SECONDS
                    ),
                    write=DEFAULT_READ_TIMEOUT_SECONDS,
                    pool=DEFAULT_CONNECT_TIMEOUT_SECONDS,
                ),
            )
        except BaseException:
            await self.stop()
            raise

        self._config = config
        self._context = context
        self._command = command
        self._port = port
        self._token = token
        self._relay_config = plugin_config
        self._base_dir = base_dir

    def _install_signal_handlers(self) -> None:
        if os.name == "nt":
            return
        if threading.current_thread() is not threading.main_thread():
            raise lifecycle.LifecycleError(
                "openclaw_process_supervision_failed",
                "OpenClaw signal supervision requires the adapter main thread",
            )

        def handle_signal(signum: int, _frame: FrameType | None) -> None:
            self._forward_gateway_signal(signum)
            raise SystemExit(128 + signum)

        self._signal_handler = handle_signal
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous_signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle_signal)

    def _restore_signal_handlers(self) -> None:
        if self._signal_handler is None:
            return
        for signum, previous in self._previous_signal_handlers.items():
            if signal.getsignal(signum) is self._signal_handler:
                signal.signal(signum, previous)
        self._previous_signal_handlers = {}
        self._signal_handler = None

    def _forward_gateway_signal(self, signum: int) -> None:
        process = self._process
        if process is None or process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    async def _wait_ready(self, port: int, token: str, timeout: float) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        async with httpx.AsyncClient(
            headers={"authorization": f"Bearer {token}"}, timeout=1.0, trust_env=False
        ) as client:
            while asyncio.get_running_loop().time() < deadline:
                process = self._process
                if process is None or process.returncode is not None:
                    detail = "\n".join(self._stderr_tail)
                    raise lifecycle.LifecycleError(
                        "openclaw_gateway_start_failed",
                        "OpenClaw Gateway exited before becoming ready",
                        metadata={"detail": detail[-2000:]},
                    )
                try:
                    response = await client.get(f"http://127.0.0.1:{port}/readyz")
                    if response.status_code == 200:
                        return
                except httpx.RequestError:
                    pass
                await asyncio.sleep(0.05)
        raise lifecycle.LifecycleError(
            "openclaw_gateway_start_timeout",
            "OpenClaw Gateway did not become ready before the startup timeout",
            retryable=True,
        )

    async def invoke(
        self,
        request: contract.AgentRunRequest,
        context: contract.RuntimeContext,
    ) -> contract.AgentRunResult:
        if (
            self._client is None
            or self._config is None
            or self._port is None
            or self._process is None
        ):
            raise lifecycle.LifecycleError(
                "openclaw_not_started", "OpenClaw runtime is not started"
            )
        if self._process.returncode is not None:
            raise lifecycle.LifecycleError(
                "openclaw_gateway_exited",
                "OpenClaw Gateway exited unexpectedly "
                f"with exit status {self._process.returncode}",
                metadata={"exit_code": self._process.returncode},
            )
        if self._context is None or context.runtime_id != self._context.runtime_id:
            raise lifecycle.LifecycleError(
                "openclaw_runtime_mismatch",
                "OpenClaw invocation does not match the active runtime",
            )
        messages: list[dict[str, str]] = []
        instruction = (
            self._config.instructions.system if self._config.instructions else None
        )
        if instruction is not None:
            messages.append({"role": "system", "content": instruction.content})
        messages.append(
            {
                "role": "user",
                "content": common_utils.normalize_user_input(request.input),
            }
        )
        model = _selected_model(self._config)
        try:
            text, usage = await openai_chat.invoke(
                self._client,
                openai_chat.endpoint(f"http://127.0.0.1:{self._port}/v1"),
                model=OPENCLAW_CHAT_MODEL,
                messages=messages,
                temperature=model.temperature,
                top_p=model.top_p,
                max_tokens=model.max_tokens,
                user=context.runtime_id,
            )
        except httpx.HTTPStatusError as error:
            return contract.AgentRunResult(
                status=contract.AgentRunStatus.FAILED,
                output={},
                error=contract.AgentRunError(
                    code="openclaw_http_error",
                    message=f"OpenClaw Gateway returned HTTP status {error.response.status_code}",
                    retryable=error.response.status_code >= 500
                    or error.response.status_code == 429,
                ),
            )
        except httpx.RequestError as error:
            raise lifecycle.LifecycleError(
                "openclaw_transport_failed",
                "OpenClaw Gateway request could not be completed",
                retryable=True,
            ) from error
        except Exception as error:
            raise lifecycle.LifecycleError(
                "openclaw_invalid_response",
                "OpenClaw Gateway returned an invalid response",
            ) from error
        return contract.AgentRunResult(
            status=contract.AgentRunStatus.SUCCEEDED,
            output={"response": text},
            usage=usage,
            artifacts=self._relay_artifacts(),
        )

    def _relay_artifacts(self) -> list[contract.AgentArtifact]:
        if self._relay_config is None or self._base_dir is None:
            return []
        artifacts: list[contract.AgentArtifact] = []
        for index, item in enumerate(
            common_utils.collect_relay_artifacts(self._relay_config), start=1
        ):
            try:
                relative = Path(item["path"]).resolve().relative_to(self._base_dir)
            except (KeyError, OSError, RuntimeError, ValueError):
                continue
            artifacts.append(
                contract.AgentArtifact(
                    name=f"openclaw-{item['kind']}-{index}",
                    kind=item["kind"],
                    path=relative,
                    media_type="application/json",
                )
            )
        return artifacts

    async def stop(self) -> None:
        try:
            self._restore_signal_handlers()
        except Exception:
            logger.error(
                "OpenClaw could not restore signal handlers during cleanup",
                exc_info=True,
            )
        client, self._client = self._client, None
        try:
            if client is not None:
                await client.aclose()
        except Exception:
            logger.error("OpenClaw could not close its HTTP client", exc_info=True)
        process, self._process = self._process, None
        try:
            if process is not None and process.returncode is None:
                if os.name == "nt":
                    process.terminate()
                else:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                try:
                    await asyncio.wait_for(
                        process.wait(), timeout=self._shutdown_timeout
                    )
                except TimeoutError:
                    if os.name == "nt":
                        process.kill()
                    else:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    await process.wait()
        except Exception:
            logger.error("OpenClaw Gateway process cleanup failed", exc_info=True)
        windows_job, self._windows_job = self._windows_job, None
        try:
            if windows_job is not None:
                _windows_job.close_job(windows_job)
        except Exception:
            logger.error("OpenClaw could not close its Windows Job Object", exc_info=True)
        log_tasks, self._log_tasks = self._log_tasks, []
        try:
            for task in log_tasks:
                if not task.done():
                    task.cancel()
            if log_tasks:
                await asyncio.gather(*log_tasks, return_exceptions=True)
        except Exception:
            logger.error("OpenClaw could not stop log-capture tasks", exc_info=True)
        temp_dir, self._temp_dir = self._temp_dir, None
        try:
            if temp_dir is not None:
                temp_dir.cleanup()
        except Exception:
            logger.error(
                "OpenClaw could not remove its temporary Gateway configuration",
                exc_info=True,
            )
        self._config = None
        self._context = None
        self._command = None
        self._port = None
        self._token = None
        self._relay_config = None
        self._base_dir = None


def main() -> None:
    lifecycle.serve(OpenClawRuntime, config_loader=contract.AgentConfig.from_mapping)


if __name__ == "__main__":
    main()
