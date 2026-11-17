#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain Deep Agents adapter for Fabric.

Maps Fabric's normalized invocation onto the ``deepagents`` SDK and returns a
normalized Fabric result. A started runtime retains one compiled graph and
LangGraph checkpointer across ordered invocations.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import math
import os
import sys
import uuid
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Iterable
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any
from typing import NamedTuple

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import StateSnapshot
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentMcpServerConfig
from nemo_fabric_adapter_contract.models import AgentModelConfig
from nemo_fabric_adapter_contract.models import AgentRunError
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunResult
from nemo_fabric_adapter_contract.models import AgentRunStatus
from nemo_fabric_adapter_contract.models import AgentUsage
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common import instructions as common_instructions
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common import mcp_availability
from nemo_fabric_adapters.common import mcp_oauth
from nemo_fabric_adapters.common import mcp_service_account
import nemo_fabric_adapters.common.utils as common_utils
from nemo_fabric_adapters.deepagents import skills as skill_mounts
from pydantic import ValidationError

HARNESS = "deepagents"
# Providers we serve through the OpenAI-compatible ``ChatOpenAI`` client.
OPENAI_COMPATIBLE_PROVIDERS = {"nvidia", "openai", "openai-compatible"}
# MCP transports langchain-mcp-adapters accepts (after normalization).
VALID_MCP_TRANSPORTS = {"stdio", "sse", "streamable_http", "websocket"}
# Upper bound on listing one MCP server's tools at startup. Unreachable servers
# usually fail fast; this bounds a server that accepts the connection but never
# answers, so it is reported as unavailable instead of stalling startup.
MCP_DISCOVERY_TIMEOUT_SECONDS = 60.0
# How long a timed-out discovery may spend cleaning up its connection before
# startup moves on. Transport cleanup (for example a streamable HTTP session
# DELETE) can otherwise hold startup well past the discovery timeout.
MCP_DISCOVERY_CANCEL_GRACE_SECONDS = 5.0
# create_deep_agent arguments Fabric derives from normalized config; the
# harness.settings.deepagents passthrough must not override them (doing so would
# bypass the normalized model config, MCP tool resolution, workspace confinement,
# or tool gating). The declarative backend selector is validated separately and
# never forwarded as a create_deep_agent argument.
FABRIC_OWNED_AGENT_KEYS = frozenset(
    {
        "model",
        "tools",
        "backend",
        "skills",
        "system_prompt",
        "middleware",
        "checkpointer",
    }
)
# Documented, JSON-serializable create_deep_agent options callers may forward
# through harness.settings.deepagents. Executable objects (AgentMiddleware, BaseTool,
# Python callables) cannot cross the SDK->JSON->payload boundary and are excluded.
DEEPAGENTS_PASSTHROUGH_KEYS = frozenset({"subagents", "interrupt_on"})
DEEPAGENTS_ADAPTER_SETTING_KEYS = frozenset({"backend"})
# Appended to the fault that poisoned Relay's scope stack, and then reported on every
# later turn of the same runtime so none of them can look telemetry-clean. Deliberately
# does not claim later turns are untraced: the Relay middleware is attached to the
# compiled agent at start and keeps emitting, so what is actually lost is trustworthy
# nesting, not all telemetry.
_QUARANTINE_NOTE = (
    "telemetry unreliable for the rest of this runtime: an earlier turn left the Relay "
    "scope stack dirty, so this turn is not wrapped in a request scope and any events "
    "the agent middleware still emits are nested under a stale scope"
)
# Sentinel for "this handle carries no identity", kept distinct from a real ``None``
# attribute value so an unreadable handle can never compare equal to another one.
_UNREADABLE = object()


class AdapterConfigError(RuntimeError):
    """Raised for invalid Deep Agents adapter configuration (normalized to a failure)."""


class ToolGateMiddleware(AgentMiddleware):  # type: ignore[misc]
    """Block tool calls selected by an adapter-owned policy."""

    def __init__(
        self,
        is_blocked: Callable[[Any], bool],
        message: Callable[[Any], str],
    ):
        self._is_blocked = is_blocked
        self._message = message

    def _blocked(self, request: Any) -> ToolMessage:
        name = request.tool_call.get("name")
        return ToolMessage(
            content=self._message(name),
            tool_call_id=request.tool_call.get("id", ""),
            status="error",
        )

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        if self._is_blocked(request.tool_call.get("name")):
            return self._blocked(request)
        return await handler(request)

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        if self._is_blocked(request.tool_call.get("name")):
            return self._blocked(request)
        return handler(request)


def resolve_api_key_env(model_config: AgentModelConfig) -> str:
    """Resolve the credential env var.

    OpenAI retains its conventional environment variable. Other providers must
    name the credential explicitly so a key is never sent to the wrong endpoint.
    """

    explicit = model_config.api_key_env
    if explicit is not None:
        return explicit
    provider = model_config.provider
    if provider == "openai":
        return "OPENAI_API_KEY"
    raise AdapterConfigError(
        f"models.default.api_key_env is required for provider '{provider}'."
    )


def main() -> None:
    """Serve the persistent local-host lifecycle protocol."""

    lifecycle.serve(DeepAgentsRuntime, config_loader=AgentConfig.from_mapping)


def preflight_check(model_config: AgentModelConfig) -> None:
    """Validate invocation-time prerequisites and fail fast with clear errors.

    These checks run during adapter startup. Core descriptor diagnostics do not
    invoke adapter-specific preflight hooks, so they cannot verify these
    prerequisites. At invocation time the ``deepagents`` package must be
    importable and the configured model-provider credential must be present in
    the environment.
    """

    import importlib.util

    if importlib.util.find_spec("deepagents") is None:
        raise RuntimeError(
            "the 'deepagents' package is required for the Deep Agents adapter; install "
            "a compatible Deep Agents harness in the adapter environment."
        )

    api_key_env = resolve_api_key_env(model_config)
    if not os.environ.get(api_key_env):
        raise RuntimeError(
            f"the model-provider credential env var '{api_key_env}' is not set in the "
            "environment. Set it to your API key, or set models.default.api_key_env to the "
            "variable that holds it."
        )


def selected_model_config(config: AgentConfig) -> AgentModelConfig:
    if model_config := config.models.get("default"):
        return model_config
    if len(config.models) == 1:
        return next(iter(config.models.values()))
    raise AdapterConfigError(
        "Deep Agents requires a default model or exactly one model."
    )


def _max_turns(config: AgentConfig) -> int | None:
    return config.runtime.max_turns if config.runtime is not None else None


def selected_model_name(model_config: AgentModelConfig) -> str:
    """The model id sent to the endpoint.

    Fabric model slugs are ``provider/model``; for the ``openai`` provider the
    prefix is Fabric's, not part of OpenAI's id, so it is dropped (the same rule
    the Codex adapter applies). Other providers' ids are sent as written, since
    namespaces such as ``nvidia/`` belong to the id itself.
    """

    if model_config.provider == "openai":
        return model_config.model.removeprefix("openai/")
    return model_config.model


def build_chat_model(
    model_config: AgentModelConfig, settings: dict[str, Any]
) -> tuple[Any, str, str | None]:
    """Build a LangChain chat model from Fabric model config.

    Known OpenAI-compatible providers use ``ChatOpenAI``. Other providers are
    delegated to ``langchain.chat_models.init_chat_model``.
    """

    model_name = selected_model_name(model_config)
    api_key_env = resolve_api_key_env(model_config)
    api_key = os.environ[api_key_env]

    provider = model_config.provider
    base_url = model_config.base_url
    temperature = model_config.temperature
    top_p = model_config.top_p
    max_tokens = model_config.max_tokens

    if "reasoning_effort" in settings:
        if settings["reasoning_effort"] not in (
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
        ):
            raise AdapterConfigError("harness.settings.reasoning_effort is invalid.")
        if provider not in OPENAI_COMPATIBLE_PROVIDERS:
            raise AdapterConfigError(
                "harness.settings.reasoning_effort requires an OpenAI-compatible provider."
            )

    if provider in OPENAI_COMPATIBLE_PROVIDERS - {"openai"} and not base_url:
        raise AdapterConfigError(
            f"models.default.base_url is required for provider '{provider}'."
        )

    if provider not in OPENAI_COMPATIBLE_PROVIDERS:
        # Generic provider hook: honor an explicit non-OpenAI-compatible provider.
        from langchain.chat_models import init_chat_model

        kwargs = {"model": model_name, "model_provider": provider, "api_key": api_key}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if base_url:
            kwargs["base_url"] = base_url
        return (
            init_chat_model(**_supported_kwargs(init_chat_model, kwargs)),
            model_name,
            base_url,
        )

    from langchain_openai import ChatOpenAI

    kwargs = {"model": model_name, "api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = max_tokens
    if "reasoning_effort" in settings:
        kwargs["reasoning_effort"] = settings["reasoning_effort"]
    return ChatOpenAI(**_supported_kwargs(ChatOpenAI, kwargs)), model_name, base_url


def _local_shell_backend(**kwargs: Any) -> Any:
    from deepagents.backends import LocalShellBackend

    workspace = Path(kwargs["root_dir"]).absolute()

    class WorkspaceLocalShellBackend(LocalShellBackend):
        """Accept the shell's workspace paths without weakening virtual file checks."""

        def _to_virtual_path(self, path: Path) -> str:
            # Listings and searches must return paths the resolver reads back
            # unambiguously, even if a nested path repeats the workspace prefix.
            virtual = super()._to_virtual_path(path)
            # Deep Agents tools reject drive-prefixed paths. Windows virtual
            # paths cannot collide with the drive-qualified physical aliases.
            if self.cwd.drive:
                return virtual
            return str(self.cwd / virtual.lstrip("/"))

        def _resolve_path(self, key: str) -> Path:
            path = Path(key)
            if path.is_absolute():
                for root in (workspace, self.cwd):
                    try:
                        relative = path.relative_to(root)
                    except ValueError:
                        continue
                    # Native resolution still rejects traversal and symlink escapes.
                    return super()._resolve_path("/" + relative.as_posix())
            return super()._resolve_path(key)

    return WorkspaceLocalShellBackend(**kwargs)


def _workspace_root(runtime_context: RuntimeContext, base_dir: str) -> Path | None:
    workspace = runtime_context.environment.workspace
    if not workspace:
        return None
    root = Path(str(workspace))
    return root if root.is_absolute() else Path(base_dir) / root


def resolve_backend(
    runtime_context: RuntimeContext,
    base_dir: str,
    backend_config: dict[str, Any] | None = None,
    withheld_env: Iterable[str] = (),
) -> Any:
    """Resolve the adapter-owned backend selection and root directory.

    Variables named in ``withheld_env``, such as MCP client secrets, are kept
    out of the local shell's environment so agent commands cannot read them.
    """

    workspace = runtime_context.environment.workspace
    if backend_config is not None and not workspace:
        raise AdapterConfigError(
            "harness.settings.deepagents.backend.type='local_shell' requires "
            "environment.workspace so host shell execution is never rooted at the "
            "NeMo Fabric base directory."
        )
    root = _workspace_root(runtime_context, base_dir)
    if root is None:
        return None

    if backend_config is not None:
        withheld = set(withheld_env)
        return _local_shell_backend(
            root_dir=str(root),
            virtual_mode=True,
            env={
                key: value
                for key, value in runtime_context.environment.env.items()
                if key not in withheld
            },
            inherit_env=False,
        )

    from deepagents.backends import FilesystemBackend

    # virtual_mode=True confines the agent to root_dir; absolute paths and ``..``
    # cannot escape it (and it silences the deepagents 0.6 default-flip warning).
    return FilesystemBackend(root_dir=str(root), virtual_mode=True)


async def resolve_tools(
    config: AgentConfig,
    unavailable_mcp_servers: list[dict[str, str]] | None = None,
    oauth_token_dir: Path | None = None,
    mcp_sign_ins: dict[str, McpSignInServer] | None = None,
) -> list[Any] | None:
    """Resolve Fabric MCP servers into Deep Agents tools.

    Servers skipped as unavailable are appended to ``unavailable_mcp_servers``
    when a list is supplied, and servers waiting for a browser sign-in are added
    to ``mcp_sign_ins`` when a dictionary is supplied.
    """

    resolution = await resolve_mcp_tools(config, oauth_token_dir)
    if unavailable_mcp_servers is not None:
        unavailable_mcp_servers.extend(resolution.unavailable)
    if mcp_sign_ins is not None:
        mcp_sign_ins.update(resolution.sign_ins)
    return resolution.tools or None


def _blocked_tool_names(config: AgentConfig) -> set[str]:
    return set(config.tools.blocked if config.tools is not None else [])


def _enabled_tool_names(config: AgentConfig) -> set[str] | None:
    enabled = config.tools.enabled if config.tools is not None else None
    return None if enabled is None else set(enabled)


def _tool_gate_middleware(
    is_blocked: Callable[[Any], bool], message: Callable[[Any], str]
) -> ToolGateMiddleware:
    return ToolGateMiddleware(is_blocked, message)


def tool_policy_middleware(enabled: set[str] | None, blocked: set[str]) -> Any:
    """Enforce tool selection and blocking across the full tool surface."""

    return _tool_gate_middleware(
        lambda name: name in blocked or (enabled is not None and name not in enabled),
        lambda name: f"Tool '{name}' is blocked by the configured tools policy.",
    )


def _interrupt_is_enabled(policy: Any) -> bool:
    return policy is True or isinstance(policy, dict)


def resolve_skills(
    config: AgentConfig,
    runtime_context: RuntimeContext,
    base_dir: str,
    backend: Any,
    skills_report: list[dict[str, Any]] | None = None,
) -> tuple[list[str] | None, Any]:
    """Make the configured skills readable by Deep Agents.

    Returns the Deep Agents ``skills`` sources and the backend to use. The
    skill folders are resolved in the task environment, which the backend
    (confined to the workspace) cannot read directly, so each entry's skills
    are routed into the backend at their own paths. The ``output.skills``
    report is appended to ``skills_report`` when a list is supplied.
    """

    mounts = skill_mounts.plan_skills(
        config.skills, _workspace_root(runtime_context, base_dir)
    )
    mounts.warn()
    if skills_report is not None and (report := mounts.report()):
        skills_report.append(report)
    return mounts.sources or None, skill_mounts.mounted_backend(backend, mounts)


def _mcp_validation_feedback(error: ValidationError) -> str:
    """Report field paths and validation reasons, not argument payloads."""
    details = []
    for detail in error.errors(
        include_input=False, include_context=False, include_url=False
    ):
        path = ".".join(str(part) for part in detail["loc"]) or "arguments"
        details.append(f"{path}: {detail['msg']}")
    return "Tool input validation error: " + "; ".join(details)


def _mcp_tool_coroutine(
    coroutine: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Keep MCP response validation failures out of input-error feedback."""

    @wraps(coroutine)
    async def call_tool(*args: Any, **kwargs: Any) -> Any:
        """Run once, preserving fatal failures after local argument parsing."""
        try:
            return await coroutine(*args, **kwargs)
        except ValidationError as error:
            # Local tool input validation runs before this coroutine is entered.
            raise RuntimeError("MCP tool response validation failed.") from error

    return call_tool


def _mcp_secret_env_names(config: AgentConfig) -> set[str]:
    """Return the client-secret variables of authenticated MCP servers."""
    servers = config.mcp.servers if config.mcp is not None else {}
    return {
        auth.client_secret_env
        for spec in servers.values()
        if (auth := spec.authentication) is not None and auth.client_secret_env
    }


class McpSignInServer(NamedTuple):
    """An ``oauth2`` MCP server whose tools wait for a browser sign-in."""

    client: Any
    sign_in: mcp_oauth.BrowserSignIn
    transport: str


class McpToolResolution(NamedTuple):
    """MCP tools loaded at startup and the servers that could not be reached.

    ``sign_ins`` holds the ``oauth2`` servers skipped because they need a
    browser sign-in, which the runtime runs on its first invocation.
    """

    tools: list[Any]
    unavailable: list[dict[str, str]]
    sign_ins: dict[str, McpSignInServer] = {}


async def _discover_server_tools(
    client: Any, name: str, token_client: Any | None
) -> list[Any]:
    """Fetch the server's token, if it has one, then list its tools."""
    if token_client is not None:
        # Fetch the token before connecting so a rejected client is reported as
        # needing authentication instead of as a generic connection failure.
        await token_client.get_access_token()
    return await client.get_tools(server_name=name)


def _unavailable_reason(error: BaseException) -> str:
    """Classify a discovery failure; credential problems need authentication."""
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, mcp_service_account.ServiceAccountTokenError):
            if current.rejected:
                return mcp_availability.AUTHENTICATION_REQUIRED
        elif mcp_oauth.is_sign_in_error(current):
            return mcp_availability.AUTHENTICATION_REQUIRED
        response = getattr(current, "response", None)
        if getattr(response, "status_code", None) == 401:
            return mcp_availability.AUTHENTICATION_REQUIRED
        pending.extend(getattr(current, "exceptions", ()))
        for linked in (current.__cause__, current.__context__):
            if linked is not None:
                pending.append(linked)
    return mcp_availability.FAILED


async def _load_server_tools(
    client: Any, name: str, transport: str, token_client: Any | None = None
) -> tuple[list[Any], dict[str, str] | None]:
    """List one server's tools, reporting an unreachable server instead of raising."""
    task, timed_out = await mcp_availability.wait_bounded(
        _discover_server_tools(client, name, token_client),
        timeout=MCP_DISCOVERY_TIMEOUT_SECONDS,
        grace=MCP_DISCOVERY_CANCEL_GRACE_SECONDS,
    )
    if timed_out:
        if not task.done():
            print(
                f"NeMo Fabric Deep Agents: MCP server {name!r} is still closing its "
                "connection after the discovery timeout; continuing startup",
                file=sys.stderr,
            )
        return [], mcp_availability.unavailable_server(
            name, transport, mcp_availability.TIMED_OUT, "TimeoutError"
        )
    if task.cancelled():
        # Caller cancellation is re-raised by wait_bounded, so this cancellation
        # came from inside the MCP client; treat the server as failed.
        return [], mcp_availability.unavailable_server(
            name, transport, mcp_availability.FAILED, "CancelledError"
        )
    try:
        tools = task.result()
    except Exception as error:
        return [], mcp_availability.unavailable_server(
            name,
            transport,
            _unavailable_reason(error),
            mcp_availability.exception_type(error),
        )
    return list(tools), None


def _prepare_mcp_tools(tools: list[Any]) -> list[Any]:
    """Return MCP tool failures to the model as feedback."""
    for tool in tools:
        # MCP isError results become ToolException, not graph-fatal failures.
        # Return them as tool feedback; never replay a call automatically.
        tool.coroutine = _mcp_tool_coroutine(tool.coroutine)
        tool.handle_tool_error = True
        tool.handle_validation_error = _mcp_validation_feedback
    return tools


async def resolve_mcp_tools(
    config: AgentConfig, oauth_token_dir: Path | None = None
) -> McpToolResolution:
    """Load tools from every reachable MCP server.

    Invalid server configuration still fails startup. A server that cannot be
    reached or does not list its tools in time is skipped and reported, so the
    runtime starts with the tools of the remaining servers. Discovery runs again
    on every runtime start; a skipped server is not remembered. An ``oauth2``
    server uses its stored tokens here and never opens a browser; one that
    needs a sign-in is returned in ``sign_ins``.
    """
    servers = config.mcp.servers if config.mcp is not None else {}
    connections = {
        name: _mcp_connection(name, spec, oauth_token_dir)
        for name, spec in servers.items()
    }
    if not connections:
        return McpToolResolution([], [])
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(connections)
    loaded = await asyncio.gather(
        *(
            _load_server_tools(
                client,
                name,
                connection["transport"],
                getattr(connection.get("auth"), "token_client", None),
            )
            for name, connection in connections.items()
        )
    )
    tools: list[Any] = []
    unavailable: list[dict[str, str]] = []
    sign_ins: dict[str, McpSignInServer] = {}
    for server_tools, failure in loaded:
        if failure is None:
            tools.extend(server_tools)
            continue
        unavailable.append(failure)
        name = failure["name"]
        sign_in = getattr(connections[name].get("auth"), "sign_in", None)
        if (
            sign_in is not None
            and failure["reason"] == mcp_availability.AUTHENTICATION_REQUIRED
        ):
            sign_ins[name] = McpSignInServer(client, sign_in, failure["transport"])
    mcp_availability.warn("Deep Agents", unavailable)
    return McpToolResolution(_prepare_mcp_tools(tools), unavailable, sign_ins)


def _mcp_connection(
    name: str, spec: AgentMcpServerConfig, oauth_token_dir: Path | None = None
) -> dict[str, Any]:
    """Translate one Fabric MCP server into a langchain-mcp-adapters connection."""
    # A misconfigured server must fail loudly, not be silently dropped.
    transport = spec.transport.strip().lower().replace("-", "_")
    # AgentMcpServerConfig carries the command in ``url`` and stdio extensions.
    target = os.path.expandvars(spec.url).strip()
    if not target:
        raise AdapterConfigError(
            f"MCP server '{name}' requires a url (or command in url)."
        )
    if transport in ("stdio", "command", "process"):
        if spec.authentication is not None:
            # A local process has no HTTP requests to carry a token.
            raise AdapterConfigError(
                f"MCP server {name!r} does not support {spec.authentication.type} "
                "authentication with the stdio transport."
            )
        connection: dict[str, Any] = {
            "transport": "stdio",
            "command": target,
            "args": spec.args,
        }
        if env := spec.env:
            connection["env"] = env
        return connection
    if transport in ("", "http", "streamable_http", "streamablehttp"):
        transport = "streamable_http"
    if transport not in VALID_MCP_TRANSPORTS:
        raise AdapterConfigError(
            f"MCP server '{name}' has unsupported transport '{transport}'."
        )
    if transport == "websocket" and spec.custom_headers:
        raise AdapterConfigError(
            f"MCP server '{name}' does not support custom headers with the "
            "websocket transport."
        )
    connection = {"transport": transport, "url": target}
    if headers := spec.custom_headers:
        try:
            headers = common_utils.expand_http_headers(name, headers)
        except Exception as error:
            raise AdapterConfigError(f"{error}.") from error
        connection["headers"] = headers

    auth = spec.authentication
    if auth is None:
        return connection
    if transport == "websocket":
        raise AdapterConfigError(
            f"MCP server {name!r} does not support {auth.type} authentication "
            "with the websocket transport."
        )
    if any(header.lower() == "authorization" for header in spec.custom_headers):
        raise AdapterConfigError(
            f"MCP server {name!r} sets both {auth.type} authentication and an "
            "Authorization custom header; remove the header."
        )
    try:
        # Bearer tokens must not cross the network in clear text.
        mcp_service_account.validate_secure_url(name, "url", target)
    except ValueError as error:
        raise AdapterConfigError(f"{error}.") from None
    if auth.type == "oauth2":
        connection["auth"] = _browser_sign_in_auth(name, target, auth, oauth_token_dir)
        connection["httpx_client_factory"] = mcp_oauth.secure_http_client_factory(name)
        return connection
    try:
        token_url = mcp_service_account.validate_token_url(name, auth.token_url)
        client_secret = mcp_service_account.resolve_client_secret(
            name, auth.client_secret_env
        )
    except ValueError as error:
        raise AdapterConfigError(f"{error}.") from None
    token_client = mcp_service_account.ServiceAccountTokenClient(
        client_id=auth.client_id,
        client_secret=client_secret,
        token_url=token_url,
        scopes=auth.scopes,
        token_endpoint_auth_method=auth.token_endpoint_auth_method,
        token_cache_buffer_seconds=auth.token_cache_buffer_seconds,
    )
    connection["auth"] = mcp_service_account.service_account_auth(token_client)
    return connection


def _browser_sign_in_auth(
    name: str, url: str, auth: Any, token_dir: Path | None
) -> Any:
    """Build the browser sign-in auth for one ``oauth2`` MCP server."""
    if token_dir is None:
        raise AdapterConfigError(
            f"MCP server {name!r} oauth2 authentication requires a token directory."
        )
    try:
        client_secret = (
            mcp_service_account.resolve_client_secret(name, auth.client_secret_env)
            if auth.client_secret_env
            else None
        )
        if auth.redirect_uri:
            mcp_oauth.validate_redirect_uri(name, auth.redirect_uri)
    except ValueError as error:
        raise AdapterConfigError(f"{error}.") from None
    method = auth.token_endpoint_auth_method
    return mcp_oauth.browser_sign_in_auth(
        name,
        url,
        token_dir,
        client_id=auth.client_id,
        client_secret=client_secret,
        client_name=auth.client_name,
        scopes=list(auth.scopes),
        redirect_uri=auth.redirect_uri,
        token_endpoint_auth_method=str(method) if method is not None else None,
        authorization_timeout_seconds=auth.authorization_timeout_seconds,
    )


# --- runtime state ---------------------------------------------------------


def state_dir(runtime_context: RuntimeContext, base_dir: str) -> Path:
    """Return the adapter's state folder.

    With ``runtime.keep_state`` this is the agent's state folder from Fabric,
    shared by every runtime of the agent.
    """
    if runtime_context.state_dir:
        return Path(str(runtime_context.state_dir))
    base_dir = Path(base_dir).resolve()
    root = runtime_context.artifacts.root
    if root:
        return Path(str(root)).resolve() / ".fabric" / "deepagents"
    return base_dir / "artifacts" / "deepagents" / ".fabric"


def checkpointer_path(context: RuntimeContext, base_dir: str, runtime_id: str) -> Path:
    key = hashlib.sha256(runtime_id.encode("utf-8")).hexdigest()
    base = state_dir(context, base_dir) / "runtimes"
    return base / f"{key}.sqlite"


def mcp_token_dir(context: RuntimeContext, base_dir: str, runtime_id: str) -> Path:
    """Return where ``oauth2`` MCP tokens are stored.

    With ``runtime.keep_state`` later runtimes of the agent reuse them;
    otherwise each runtime keeps its own.
    """
    if context.state_dir:
        return state_dir(context, base_dir) / mcp_oauth.TOKEN_DIRECTORY_NAME
    key = hashlib.sha256(runtime_id.encode("utf-8")).hexdigest()
    return (
        state_dir(context, base_dir) / "runtimes" / key / mcp_oauth.TOKEN_DIRECTORY_NAME
    )


async def open_checkpointer(state_sqlite: Path) -> Any:
    """Open the async LangGraph checkpointer owned by a started runtime.

    The agent is driven with ``astream``, so the checkpointer must be async: the
    synchronous ``SqliteSaver`` raises ``NotImplementedError`` from its async methods.
    """

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    state_sqlite.parent.mkdir(parents=True, exist_ok=True)
    saver_cm = AsyncSqliteSaver.from_conn_string(str(state_sqlite))
    saver = await saver_cm.__aenter__()
    # Keep the context manager alive for the duration of the runtime.
    saver._fabric_cm = saver_cm  # type: ignore[attr-defined]
    return saver


async def close_checkpointer(checkpointer: Any) -> None:
    saver_cm = getattr(checkpointer, "_fabric_cm", None)
    if saver_cm is not None:
        await saver_cm.__aexit__(None, None, None)


# --- invocation ------------------------------------------------------------


async def build_agent_kwargs(
    config: AgentConfig,
    runtime_context: RuntimeContext,
    base_dir: str,
    model: Any,
    settings: dict[str, Any],
    unavailable_mcp_servers: list[dict[str, str]] | None = None,
    skills_report: list[dict[str, Any]] | None = None,
    oauth_token_dir: Path | None = None,
    mcp_sign_ins: dict[str, McpSignInServer] | None = None,
) -> dict[str, Any]:
    """Build the ``create_deep_agent`` arguments from the normalized config."""
    instruction = common_instructions.system_instruction(
        config,
        adapter="Deep Agents",
        supported_modes={"replace"},
    )
    extra = _validated_deepagents_settings(settings.get("deepagents"))
    enabled = _enabled_tool_names(config)
    blocked = _blocked_tool_names(config)
    if extra.get("backend") is not None:
        if enabled is None and "execute" not in blocked:
            raise AdapterConfigError(
                "harness.settings.deepagents.backend.type='local_shell' requires an "
                "explicit tools.enabled allowlist or 'execute' in tools.blocked so "
                "the policy is enforced for the main agent and declarative subagents."
            )
        interrupt_on = extra.get("interrupt_on")
        if isinstance(interrupt_on, dict) and _interrupt_is_enabled(
            interrupt_on.get("execute")
        ):
            raise AdapterConfigError(
                "harness.settings.deepagents.interrupt_on.execute is not supported "
                "with the local_shell backend because NeMo Fabric cannot resume the "
                "LangGraph interrupt; use tools.enabled or tools.blocked instead."
            )
        for index, subagent in enumerate(extra.get("subagents") or []):
            if not isinstance(subagent, dict):
                continue
            subagent_interrupt_on = subagent.get("interrupt_on")
            if isinstance(subagent_interrupt_on, dict) and _interrupt_is_enabled(
                subagent_interrupt_on.get("execute")
            ):
                raise AdapterConfigError(
                    "harness.settings.deepagents.subagents"
                    f"[{index}].interrupt_on.execute is not supported with the "
                    "local_shell backend because NeMo Fabric cannot resume the "
                    "LangGraph interrupt; use tools.enabled or tools.blocked instead."
                )
    kwargs: dict[str, Any] = {
        "model": model,
        "tools": await resolve_tools(
            config, unavailable_mcp_servers, oauth_token_dir, mcp_sign_ins
        ),
        # deepagents 0.5.x/0.6.x take the system prompt as ``system_prompt``.
        "system_prompt": instruction.content if instruction else None,
    }
    kwargs["skills"], kwargs["backend"] = resolve_skills(
        config,
        runtime_context,
        base_dir,
        resolve_backend(
            runtime_context,
            base_dir,
            extra.get("backend"),
            withheld_env=_mcp_secret_env_names(config),
        ),
        skills_report,
    )
    # Deep Agents-specific settings (e.g. subagents, interrupt_on) pass through,
    # after validation against the documented JSON-serializable allow-list.
    kwargs.update(
        {key: extra[key] for key in DEEPAGENTS_PASSTHROUGH_KEYS if key in extra}
    )
    if enabled is not None or blocked:
        middleware = list(kwargs.get("middleware") or [])
        middleware.append(tool_policy_middleware(enabled, blocked))
        kwargs["middleware"] = middleware
        kwargs["subagents"] = _gated_subagents(
            kwargs.get("subagents"), enabled, blocked
        )
    return {key: value for key, value in kwargs.items() if value is not None}


def _validated_deepagents_settings(extra: Any) -> dict[str, Any]:
    """Validate adapter-owned and passthrough Deep Agents settings.

    Only documented, JSON-serializable create_deep_agent options are forwarded.
    Fabric-owned keys cannot be overridden (that would bypass the normalized model
    config, MCP tool resolution, workspace confinement, and tool gating), and unknown
    keys fail clearly instead of being silently dropped.
    """

    if extra is None:
        return {}
    if not isinstance(extra, dict):
        raise AdapterConfigError(
            f"harness.settings.deepagents must be a mapping of JSON-serializable options, not {type(extra).__name__}."
        )
    reserved = sorted(
        FABRIC_OWNED_AGENT_KEYS.intersection(extra) - DEEPAGENTS_ADAPTER_SETTING_KEYS
    )
    if reserved:
        raise AdapterConfigError(
            f"harness.settings.deepagents cannot override NeMo Fabric-owned keys {reserved}; "
            "they are derived from the normalized NeMo Fabric config."
        )
    supported = DEEPAGENTS_PASSTHROUGH_KEYS | DEEPAGENTS_ADAPTER_SETTING_KEYS
    unknown = sorted(set(extra) - supported)
    if unknown:
        raise AdapterConfigError(
            f"harness.settings.deepagents has unsupported option(s) {unknown}; supported "
            f"settings are {sorted(supported)}."
        )

    backend_config = extra.get("backend")
    if backend_config is not None:
        if not isinstance(backend_config, dict):
            raise AdapterConfigError(
                "harness.settings.deepagents.backend must be a mapping with "
                "type='local_shell'."
            )
        unknown_backend = sorted(set(backend_config) - {"type"})
        if unknown_backend:
            raise AdapterConfigError(
                "harness.settings.deepagents.backend has unsupported field(s) "
                f"{unknown_backend}; only 'type' is supported."
            )
        if backend_config.get("type") != "local_shell":
            raise AdapterConfigError(
                "harness.settings.deepagents.backend.type must be 'local_shell'."
            )
    return dict(extra)


def _gate_subagent(
    subagent: dict[str, Any], enabled: set[str] | None, blocked: set[str]
) -> dict[str, Any]:
    gated = dict(subagent)
    gated["middleware"] = [
        *(gated.get("middleware") or []),
        tool_policy_middleware(enabled, blocked),
    ]
    return gated


def _gated_subagents(
    subagents: Any, enabled: set[str] | None, blocked: set[str]
) -> list[dict[str, Any]]:
    if subagents is None:
        configured: list[Any] = []
    elif isinstance(subagents, list):
        configured = subagents
    else:
        raise AdapterConfigError(
            "harness.settings.deepagents.subagents must be a list when a tools policy is configured."
        )

    gated: list[dict[str, Any]] = []
    for subagent in configured:
        if not isinstance(subagent, dict):
            raise AdapterConfigError(
                "Deep Agents subagents must be mappings when a tools policy is configured."
            )
        name = str(subagent.get("name") or "<unnamed>")
        if "graph_id" in subagent:
            raise AdapterConfigError(
                f"the tools policy cannot be enforced for remote Deep Agents subagent '{name}'."
            )
        if "runnable" in subagent:
            raise AdapterConfigError(
                f"the tools policy cannot be enforced for precompiled Deep Agents subagent '{name}'."
            )
        gated.append(_gate_subagent(subagent, enabled, blocked))

    if not any(subagent.get("name") == "general-purpose" for subagent in gated):
        from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT

        gated.insert(
            0, _gate_subagent(dict(GENERAL_PURPOSE_SUBAGENT), enabled, blocked)
        )
    return gated


class DeepAgentsRuntime:
    """One compiled Deep Agents graph and checkpointer owned by a Fabric runtime."""

    def __init__(self) -> None:
        """Create an unstarted runtime with no graph or checkpointer."""
        self._started = False
        self._stopping = False
        self._active_invoke_task: asyncio.Task[Any] | None = None
        self._health_lock = asyncio.Lock()
        self._health_task: asyncio.Task[Any] | None = None
        self._start_payload: dict[str, Any] | None = None
        self._runtime_id: str | None = None
        self._model_name: str | None = None
        self._base_url: str | None = None
        self._thread_id: str | None = None
        self._completed_invocations = 0
        self._mcp_unavailable: list[dict[str, str]] = []
        self._mcp_sign_ins: dict[str, McpSignInServer] = {}
        self._agent_kwargs: dict[str, Any] | None = None
        self._max_turns: int | None = None
        self._skills_report: dict[str, Any] | None = None
        self._checkpointer: Any = None
        self._agent: Any = None
        self._observability: Observability | None = None
        self._telemetry_provider = ""
        self._relay_plugin: Any = None
        self._relay_scope: Any = None
        self._relay_scope_type: Any = None
        self._relay_plugin_config: dict[str, Any] | None = None
        self._callback_handler_type: Any = None
        self._telemetry_quarantine: str | None = None
        self._telemetry_quarantine_cause: str | None = None

    async def start(self, payload: dict[str, Any]) -> None:
        """Build the graph, open the checkpointer and load MCP tools."""
        if self._started:
            raise lifecycle.LifecycleError(
                "deepagents_runtime_already_started",
                "Deep Agents runtime is already started",
            )

        self._stopping = False
        try:
            agent_config = payload["config"]
            runtime_context = RuntimeContext.from_mapping(
                payload.get("runtime_context")
            )
            model_config = selected_model_config(agent_config)
            preflight_check(model_config)
            settings = agent_config.harness.settings if agent_config.harness else {}
            base_dir = common_utils.base_dir(payload)
            runtime_id = runtime_context.runtime_id
            model, self._model_name, self._base_url = build_chat_model(
                model_config, settings
            )
            self._runtime_id = runtime_id
            self._thread_id = uuid.uuid4().hex

            telemetry = runtime_context.telemetry
            telemetry_providers = (
                telemetry.metadata.get("telemetry_providers", []) if telemetry else []
            )
            relay_enabled = bool(telemetry and telemetry.relay_enabled)
            self._telemetry_provider = (
                "relay"
                if relay_enabled
                else "native"
                if "native" in telemetry_providers
                else ""
            )
            self._observability = resolve_observability(
                runtime_context,
                base_dir,
                common_utils.agent_name(payload),
                model_config.model,
                self._telemetry_provider,
                relay_enabled,
            )

            mcp_unavailable: list[dict[str, str]] = []
            mcp_sign_ins: dict[str, McpSignInServer] = {}
            skills_report: list[dict[str, Any]] = []
            agent_kwargs = await build_agent_kwargs(
                agent_config,
                runtime_context,
                base_dir,
                model,
                settings,
                mcp_unavailable,
                skills_report,
                oauth_token_dir=mcp_token_dir(runtime_context, base_dir, runtime_id),
                mcp_sign_ins=mcp_sign_ins,
            )
            self._mcp_unavailable = mcp_unavailable
            self._mcp_sign_ins = mcp_sign_ins
            self._skills_report = skills_report[0] if skills_report else None
            self._checkpointer = await open_checkpointer(
                checkpointer_path(runtime_context, base_dir, runtime_id)
            )
            await self._checkpointer.setup()
            agent_kwargs["checkpointer"] = self._checkpointer

            if self._observability is not None:
                agent_kwargs = self._configure_observability(agent_kwargs)

            self._agent_kwargs = agent_kwargs
            self._max_turns = _max_turns(agent_config)
            self._agent = self._compile_agent()
            self._start_payload = payload
            self._started = True
        except BaseException:
            await self.stop()
            raise

    def _compile_agent(self) -> Any:
        from deepagents import create_deep_agent

        assert self._agent_kwargs is not None
        agent = create_deep_agent(
            **_supported_kwargs(create_deep_agent, self._agent_kwargs)
        )
        if self._max_turns is not None:
            agent = agent.with_config({"recursion_limit": self._max_turns})
        return agent

    async def _sign_in_mcp_servers(self, runtime_context: RuntimeContext) -> list[str]:
        """Sign in to ``oauth2`` MCP servers that need it, then add their tools.

        Runs on the first invocation, not at start, so a runtime starts without
        waiting for a browser. Each server's sign-in is bounded by its
        ``authorization_timeout_seconds`` and the invocation deadline. Returns
        the servers whose sign-in did not complete; the caller fails the
        invocation, and the next invocation tries them again.
        """
        pending = self._mcp_sign_ins
        if not pending:
            return []
        loop = asyncio.get_running_loop()
        deadline = lifecycle.invocation_deadline(runtime_context)
        signed_in: dict[str, list[Any]] = {}
        failed: list[str] = []
        for name, server in sorted(pending.items()):
            timeout = min(
                server.sign_in.timeout + MCP_DISCOVERY_TIMEOUT_SECONDS,
                max(0.0, deadline - loop.time()),
            )
            server.sign_in.interactive = True
            try:
                task, timed_out = await mcp_availability.wait_bounded(
                    server.client.get_tools(server_name=name),
                    timeout=timeout,
                    grace=MCP_DISCOVERY_CANCEL_GRACE_SECONDS,
                )
            finally:
                server.sign_in.interactive = False
            if not timed_out and not task.cancelled() and task.exception() is None:
                signed_in[name] = _prepare_mcp_tools(list(task.result()))
                continue
            error_type = (
                "TimeoutError"
                if timed_out
                else "CancelledError"
                if task.cancelled()
                else mcp_availability.exception_type(task.exception())
            )
            print(
                f"NeMo Fabric Deep Agents: sign-in to MCP server {name!r} did not "
                f"complete ({error_type})",
                file=sys.stderr,
            )
            failed.append(name)

        if signed_in:
            assert self._agent_kwargs is not None
            tools = list(self._agent_kwargs.get("tools") or [])
            for server_tools in signed_in.values():
                tools.extend(server_tools)
            self._agent_kwargs["tools"] = tools
            self._agent = self._compile_agent()
            self._mcp_unavailable = [
                entry
                for entry in self._mcp_unavailable
                if entry["name"] not in signed_in
            ]
        self._mcp_sign_ins = {name: pending[name] for name in failed}
        return failed

    def _sign_in_failed_result(self, servers: list[str]) -> AgentRunResult:
        """Return a failed result for an incomplete sign-in.

        A result, not an exception: an adapter exception ends the lifecycle
        runtime, and the next invocation must be able to sign in again.
        """
        message = (
            "Deep Agents could not sign in to MCP servers: " + ", ".join(servers)
        )
        output = normalize_output(
            model_name=self._model_name,
            base_url=self._base_url,
            runtime_id=self._runtime_id,
            thread_id=self._thread_id,
            resumed=self._completed_invocations > 0,
            result_state=None,
            events=[],
            turn_messages=[],
            error=message,
            telemetry_runtime=None,
            relay_artifacts=None,
            unavailable_mcp_servers=self._mcp_unavailable,
            skills_report=self._skills_report,
        )
        output.pop("failed", None)
        output.pop("error", None)
        return AgentRunResult(
            status=AgentRunStatus.FAILED,
            output=output,
            error=AgentRunError(
                code="deepagents_mcp_authentication_failed",
                message=message,
                retryable=True,
            ),
        )

    def _configure_observability(self, agent_kwargs: dict[str, Any]) -> dict[str, Any]:
        import importlib.util

        if importlib.util.find_spec("nemo_relay") is None:
            raise _relay_dependency_error()
        common_utils.reject_ambient_relay_plugin_config()
        try:
            from nemo_relay import ScopeType, plugin, scope
            from nemo_relay.integrations.deepagents import (
                NemoRelayDeepAgentsCallbackHandler,
                add_nemo_relay_integration,
            )
        except (ImportError, AttributeError) as exc:
            raise _relay_dependency_error() from exc

        assert self._observability is not None
        self._relay_plugin_config = self._observability.plugin_config
        self._relay_plugin = plugin
        self._relay_scope = scope
        self._relay_scope_type = ScopeType
        self._callback_handler_type = NemoRelayDeepAgentsCallbackHandler
        return add_nemo_relay_integration(agent_kwargs)

    async def invoke(
        self,
        request: AgentRunRequest,
        runtime_context: RuntimeContext,
    ) -> AgentRunResult:
        task = asyncio.current_task()
        self._active_invoke_task = task
        try:
            return await self._invoke(request, runtime_context)
        finally:
            if self._active_invoke_task is task:
                self._active_invoke_task = None

    async def _invoke(
        self,
        request: AgentRunRequest,
        runtime_context: RuntimeContext,
    ) -> AgentRunResult:
        """Run one turn on the started graph and normalize its result."""
        start_payload = self._start_payload
        if not self._started or self._agent is None or start_payload is None:
            raise lifecycle.LifecycleError(
                "deepagents_runtime_not_started",
                "Deep Agents runtime is not started",
            )
        runtime_id = runtime_context.runtime_id
        if runtime_id != self._runtime_id:
            raise lifecycle.LifecycleError(
                "deepagents_runtime_mismatch",
                "Deep Agents invocation does not match the active runtime",
            )

        user_message = common_utils.normalize_user_input(request.input)
        if failed_sign_ins := await self._sign_in_mcp_servers(runtime_context):
            return self._sign_in_failed_result(failed_sign_ins)

        resumed = self._completed_invocations > 0
        inherited_quarantine = self._telemetry_quarantine is not None
        if self._observability is None:
            outcome = await self._invoke_agent(user_message)
        else:
            outcome = await self._invoke_with_telemetry(
                user_message,
                runtime_context,
                request.relay_session_root,
            )

        if outcome.error is None:
            self._completed_invocations += 1

        telemetry_runtime, relay_artifacts, collect_error = self._telemetry_output(
            inherited_quarantine=inherited_quarantine
        )
        output = normalize_output(
            model_name=self._model_name,
            base_url=self._base_url,
            runtime_id=self._runtime_id,
            thread_id=self._thread_id,
            resumed=resumed,
            result_state=outcome.result_state,
            events=outcome.events or [],
            turn_messages=outcome.turn_messages or [],
            error=outcome.error,
            telemetry_runtime=telemetry_runtime,
            relay_artifacts=relay_artifacts,
            telemetry_error=_join_faults(outcome.telemetry_error, collect_error),
            telemetry_quarantine_cause=(
                self._telemetry_quarantine_cause if inherited_quarantine else None
            ),
            unavailable_mcp_servers=self._mcp_unavailable,
            skills_report=self._skills_report,
        )
        failed = bool(output.pop("failed", False))
        reported_error = output.pop("error", None)
        raw_usage = output.get("usage")
        usage = raw_usage if isinstance(raw_usage, dict) else {}
        token_fields = {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
        tokens = {
            name: value
            for name, value in token_fields.items()
            if isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value <= (1 << 64) - 1
        }
        cost = usage.get("cost")
        if (
            not isinstance(cost, (int, float))
            or isinstance(cost, bool)
            or not math.isfinite(cost)
            or cost < 0
        ):
            cost = None
        return AgentRunResult(
            status=AgentRunStatus.FAILED if failed else AgentRunStatus.SUCCEEDED,
            output=output,
            error=(
                AgentRunError(
                    code=outcome.error_code or "deepagents_invocation_failed",
                    message=str(reported_error or "Deep Agents invocation failed"),
                )
                if failed
                else None
            ),
            usage=(
                # LangChain UsageMetadata input_tokens includes all input types,
                # including cache. Preserve that convention for consumers such
                # as Harbor without inventing a missing cache breakdown.
                AgentUsage(
                    **tokens,
                    input_tokens_include_cache=(
                        True if "input_tokens" in tokens else None
                    ),
                    cost_usd=cost,
                )
                if tokens or cost is not None
                else None
            ),
        )

    async def _invoke_with_telemetry(
        self,
        user_message: str,
        runtime_context: RuntimeContext,
        session_root: str | None,
    ) -> TurnOutcome:
        """Run one turn inside the Relay plugin/scope, isolating telemetry faults.

        ``_invoke_agent`` has already absorbed any invocation failure, so an exception
        caught here can only have come from telemetry setup or teardown.
        """

        if self._telemetry_quarantine is not None:
            # Relay's scope stack is still dirty from an earlier turn. Skip the request
            # scope: pushing onto that stack would nest this turn under a stale scope and
            # invite another failed pop.
            outcome = await self._invoke_agent(user_message)
            return outcome._replace(telemetry_error=self._telemetry_quarantine)

        baseline = _current_scope_handle()
        outcome: TurnOutcome | None = None
        scope_error: str | None = None
        try:
            common_utils.reject_ambient_relay_plugin_config()
            callback_handler = self._callback_handler_type()
            async with self._relay_plugin.activate(
                self._relay_plugin_config
            ) as activation:
                common_utils.reject_inherited_relay_plugin_config(activation.report)
                # Caught here rather than left to propagate: an exception crossing the
                # plugin's ``__aexit__`` is replaced by any fault the plugin raises in
                # turn, which would lose one of the two.
                try:
                    request_context, metadata = common_utils.relay_request_context(
                        runtime_context.request_id, session_root
                    )
                    metadata.update(
                        common_utils.relay_correlation_metadata(runtime_context)
                    )
                    with request_context:
                        with self._relay_scope.scope(
                            "deepagents-request",
                            self._relay_scope_type.Agent,
                            metadata=metadata,
                        ):
                            outcome = await self._invoke_agent(
                                user_message,
                                callbacks=[callback_handler],
                            )
                except Exception as exc:
                    scope_error = _error_text(exc)
        except Exception as exc:  # telemetry lifecycle fault
            telemetry_error = _join_faults(scope_error, _error_text(exc))
        else:
            telemetry_error = scope_error

        if telemetry_error is not None and not _scope_top_unchanged(baseline):
            self._telemetry_quarantine = _QUARANTINE_NOTE
            self._telemetry_quarantine_cause = telemetry_error
            telemetry_error = _join_faults(telemetry_error, _QUARANTINE_NOTE)

        if outcome is None:
            # No outcome means the agent never ran, so there is nothing to preserve.
            return TurnOutcome(error=telemetry_error, telemetry_error=telemetry_error)
        return outcome._replace(telemetry_error=telemetry_error)

    async def _invoke_agent(
        self,
        user_message: str,
        callbacks: list[Any] | None = None,
    ) -> TurnOutcome:
        """Run one agent turn, normalizing an invocation failure into an error string."""

        try:
            result_state, events, turn_messages = await invoke_compiled_agent(
                self._agent,
                user_message,
                self._thread_id,
                callbacks=callbacks,
            )
        except GraphRecursionError:
            return TurnOutcome(
                error="Deep Agents reached its graph recursion limit.",
                error_code="deepagents_recursion_limit_reached",
            )
        except Exception as exc:  # normalized adapter failure
            return TurnOutcome(error=_error_text(exc))
        return TurnOutcome(
            result_state=result_state,
            events=events,
            turn_messages=turn_messages,
        )

    def _telemetry_output(
        self,
        *,
        inherited_quarantine: bool,
    ) -> tuple[dict[str, Any] | None, list[dict[str, str]] | None, str | None]:
        """Return the telemetry block, artifact references, and any collection fault.

        Collecting references walks the filesystem, so it is returned as a fault rather
        than raised: raising here would discard an already-completed turn.

        ``inherited_quarantine`` is the state from *before* this turn: the turn that
        poisoned the runtime opened a scope and produced its own partial artifacts, so it
        still publishes them; only turns that inherit the quarantine have none of their
        own to publish.
        """

        if self._observability is None:
            return None, None, None
        telemetry_runtime = {
            "enabled": True,
            "provider": self._telemetry_provider,
            "emitter": self._observability.emitter,
        }
        if not self._observability.collect_artifacts:
            return telemetry_runtime, None, None
        if inherited_quarantine:
            return telemetry_runtime, None, None
        try:
            relay_artifacts = common_utils.collect_relay_artifacts(
                self._observability.plugin_config
            )
        except Exception as exc:
            return telemetry_runtime, None, _error_text(exc)
        return telemetry_runtime, relay_artifacts, None

    async def check(self, level: str) -> dict[str, Any]:
        """Observe the owned graph's checkpoint connection and request interface."""
        result: dict[str, Any] = {
            "outcome": "unknown",
            "reason_code": "probe_error",
            "activity": "unknown",
            "source": "fabric.adapter",
            "observed_at": None,
        }
        sources = {
            "live": "fabric.adapter.liveness",
            "active": "fabric.adapter.interface",
            "ready": "fabric.adapter.readiness",
        }
        if level not in sources:
            return {
                **result,
                "outcome": "unsupported",
                "reason_code": "unsupported_level",
            }
        if self._stopping:
            return {**result, "reason_code": "observation_changed"}
        agent, saver = self._agent, self._checkpointer
        runtime_id, thread_id = self._runtime_id, self._thread_id
        payload = self._start_payload
        if (
            not self._started
            or agent is None
            or saver is None
            or not runtime_id
            or not thread_id
            or payload is None
        ):
            return {**result, "reason_code": "runtime_unavailable"}
        connection = saver.conn

        def unchanged() -> bool:
            return (
                self._started
                and not self._stopping
                and self._agent is agent
                and self._checkpointer is saver
                and saver.conn is connection
                and agent.checkpointer is saver
                and self._runtime_id == runtime_id
                and self._thread_id == thread_id
                and self._start_payload is payload
            )

        def observed(outcome: str, reason: str | None) -> dict[str, Any]:
            if not unchanged():
                return {**result, "reason_code": "observation_changed"}
            if not connection._running:
                outcome, reason = "failed", "session_closed"
            return {
                **result,
                "outcome": outcome,
                "reason_code": reason,
                "activity": "stopped"
                if reason == "session_closed"
                else (
                    "busy"
                    if self._active_invoke_task is not None
                    and not self._active_invoke_task.done()
                    else "idle"
                ),
                "source": sources[level],
                "observed_at": datetime.now(timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z"),
            }

        if not unchanged():
            return {**result, "reason_code": "observation_changed"}
        if not connection._running:
            return observed("failed", "session_closed")

        async def probe() -> Any:
            if level == "live":
                return await connection.execute_fetchall("SELECT 1")
            if level == "active":
                return await asyncio.to_thread(agent.get_input_jsonschema)
            return await agent.aget_state({"configurable": {"thread_id": thread_id}})

        try:
            async with self._health_lock:
                pending = self._health_task
                if pending is not None and not pending.done():
                    await asyncio.wait({pending})
                # Keep one native read in flight when a caller's deadline expires.
                task = asyncio.create_task(probe())
                task.add_done_callback(
                    lambda done: None if done.cancelled() else done.exception()
                )
                self._health_task = task
                await asyncio.wait({task})
                evidence = task.result()
                self._health_task = None
            if not unchanged():
                return {**result, "reason_code": "observation_changed"}
            if not connection._running:
                return observed("failed", "session_closed")
            if level == "live":
                if evidence != [(1,)]:
                    return result
            elif level == "active":
                if (
                    not callable(agent.astream)
                    or not isinstance(evidence, dict)
                    or "messages" not in evidence.get("properties", {})
                ):
                    return observed("failed", "endpoint_unavailable")
            else:
                if not isinstance(evidence, StateSnapshot):
                    return result
                native_config = evidence.config
                if not isinstance(native_config, dict) or not isinstance(
                    native_config.get("configurable"), dict
                ):
                    return result
                if native_config["configurable"].get("thread_id") != thread_id:
                    return {**result, "reason_code": "observation_changed"}
        except Exception:
            if not unchanged():
                return {**result, "reason_code": "observation_changed"}
            if not connection._running:
                return observed("failed", "session_closed")
            return result
        return observed("passed", None)

    async def stop(self) -> None:
        """Release the graph and checkpointer; safe to call more than once."""
        self._stopping = True
        checkpointer = self._checkpointer
        self._start_payload = None
        self._runtime_id = None
        self._model_name = None
        self._base_url = None
        self._thread_id = None
        self._completed_invocations = 0
        self._mcp_unavailable = []
        self._mcp_sign_ins = {}
        self._agent_kwargs = None
        self._max_turns = None
        self._skills_report = None
        self._checkpointer = None
        self._agent = None
        self._observability = None
        self._telemetry_provider = ""
        self._relay_plugin = None
        self._relay_scope = None
        self._relay_scope_type = None
        self._relay_plugin_config = None
        self._callback_handler_type = None
        self._started = False
        if checkpointer is not None:
            await close_checkpointer(checkpointer)
        self._stopping = False


def _apply_callbacks(
    config: dict[str, Any], callbacks: list[Any] | None
) -> dict[str, Any]:
    """Append ``callbacks`` to a LangGraph run config, preserving any already set.

    The Relay callback handler is added after any consumer-provided callbacks so
    it never replaces them; the existing callbacks keep their order and position
    ahead of the appended ones.
    """

    if callbacks:
        config["callbacks"] = [*(config.get("callbacks") or []), *callbacks]
    return config


async def invoke_compiled_agent(
    agent: Any,
    user_message: str,
    thread_id: str | None,
    callbacks: list[Any] | None = None,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    """Run a compiled agent and return state, events, and current-turn messages.

    On a later runtime invocation the final ``values`` state also replays
    prior-turn messages, so usage/cost must be aggregated from the messages
    emitted *this* turn — the ``updates`` deltas — rather than the full final
    state.

    ``callbacks`` (e.g. the NeMo Relay callback handler) are appended to the
    LangGraph run config; any callbacks already present are preserved rather than
    dropped.
    """

    inputs = {"messages": [{"role": "user", "content": user_message}]}
    config: dict[str, Any] = {}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
    _apply_callbacks(config, callbacks)
    config = config or None

    events: list[dict[str, Any]] = []
    turn_messages: list[Any] = []
    final: Any = None
    # subgraphs=True surfaces subagent (delegated ``task``) steps as 3-tuples whose
    # namespace identifies the subgraph. Folding their message deltas into this turn
    # keeps usage/cost accurate for delegating runs; the final ``values`` state is
    # taken from the main graph only (empty namespace).
    stream = (
        agent.astream(inputs, config, stream_mode=["updates", "values"], subgraphs=True)
        if config
        else agent.astream(inputs, stream_mode=["updates", "values"], subgraphs=True)
    )
    async for namespace, mode, chunk in stream:
        if mode == "updates" and isinstance(chunk, dict):
            event: dict[str, Any] = {"nodes": [str(node) for node in chunk]}
            if namespace:
                event["subgraph"] = "/".join(str(part) for part in namespace)
            events.append(event)
            for node_output in chunk.values():
                if isinstance(node_output, dict):
                    new = node_output.get("messages")
                    if isinstance(new, list):
                        turn_messages.extend(new)
                    elif new is not None:
                        turn_messages.append(new)
        elif mode == "values" and not namespace:
            final = chunk
    return final, events, _dedup_messages(_messages_to_dicts(turn_messages))


class Observability(NamedTuple):
    plugin_config: dict[str, Any]
    emitter: str
    collect_artifacts: bool


class TurnOutcome(NamedTuple):
    """One agent turn, with its two failure domains kept apart: ``error`` means the
    agent failed, ``telemetry_error`` means recording it did.
    """

    result_state: Any = None
    events: list[dict[str, Any]] | None = None
    turn_messages: list[dict[str, Any]] | None = None
    error: str | None = None
    error_code: str | None = None
    telemetry_error: str | None = None


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _current_scope_handle() -> Any:
    """Return Relay's current scope handle, or ``None`` when it cannot be read."""

    try:
        import nemo_relay

        return nemo_relay.scope.get_handle()
    except Exception:
        return None


def _scope_top_unchanged(baseline: Any) -> bool:
    """Report whether the scope current now is the one current before the turn.

    This checks the top of the stack, not the whole stack: Relay exposes no depth, so a
    fault that left the stack deeper while restoring the top would read as unchanged.
    The observed failure strands a child scope on top, which this does catch. Anything
    unreadable — a missing handle, or a handle without the identity attribute — counts
    as changed, so a Relay rename cannot silently turn the check off.
    """

    if baseline is None:
        return False
    current = _current_scope_handle()
    if current is None:
        return False
    baseline_uuid = getattr(baseline, "uuid", _UNREADABLE)
    current_uuid = getattr(current, "uuid", _UNREADABLE)
    if baseline_uuid is _UNREADABLE or current_uuid is _UNREADABLE:
        return False
    return bool(current_uuid == baseline_uuid)


def _join_faults(*faults: str | None) -> str | None:
    """Combine telemetry faults into one message; teardown and artifact collection can
    both fail in the same turn.
    """

    present = [fault for fault in faults if fault]
    if not present:
        return None
    return "; ".join(present)


def _relay_dependency_error() -> RuntimeError:
    return RuntimeError(
        "telemetry is enabled but a compatible 'nemo-relay' package is not installed; "
        "install the Relay extra (pip install 'nemo-fabric-adapters-deepagents[relay]')."
    )


def resolve_observability(
    runtime_context: RuntimeContext,
    base_dir: str,
    agent_name: str,
    model_name: str,
    telemetry_provider: str,
    relay_enabled: bool,
) -> Observability | None:
    """Resolve the nemo_relay observability plugin config for relay or native telemetry.

    Relay telemetry loads its plugin config from ``FABRIC_RELAY_CONFIG_PATH`` and
    collects ATOF/ATIF/OTLP artifacts. Native telemetry reads
    ``RuntimeContext.telemetry.metadata.native_config`` (e.g. an
    OpenTelemetry/OpenInference exporter) and exports spans directly to the
    configured collector without writing relay artifacts.
    """

    if relay_enabled and telemetry_provider == "relay":
        plugin_config = common_utils.load_relay_plugin_config(
            {
                "agent_name": agent_name,
                "base_dir": base_dir,
                "config": {"models": {"default": {"model": model_name}}},
                "runtime_context": runtime_context.to_mapping(),
            }
        )
        return Observability(
            plugin_config,
            "deepagents.observability/nemo_relay",
            True,
        )
    if telemetry_provider == "native":
        native_config = runtime_context.telemetry.metadata.get("native_config", {})
        if not isinstance(native_config, dict):
            raise AdapterConfigError(
                "runtime_context.telemetry.metadata.native_config must be a mapping."
            )
        if native_config.get("components"):
            return Observability(
                native_config, "deepagents.observability/native", False
            )
    return None


# --- normalization ---------------------------------------------------------


def normalize_output(
    *,
    model_name: str | None,
    base_url: str | None,
    runtime_id: str | None,
    thread_id: str | None,
    resumed: bool,
    result_state: Any,
    events: list[dict[str, Any]],
    turn_messages: list[dict[str, Any]],
    error: str | None,
    telemetry_runtime: dict[str, Any] | None,
    relay_artifacts: list[dict[str, str]] | None,
    telemetry_error: str | None = None,
    telemetry_quarantine_cause: str | None = None,
    unavailable_mcp_servers: list[dict[str, str]] | None = None,
    skills_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the adapter output for one turn from the final graph state."""
    messages = _extract_messages(result_state)
    response = _final_response(messages)
    # Aggregate usage/cost from this turn's messages only; the replayed final state
    # replays prior-turn messages that must not be re-counted.
    usage = _aggregate_usage(turn_messages)

    output: dict[str, Any] = {
        "harness": HARNESS,
        "adapter": "python",
        "mode": "deepagents",
        "model": model_name,
        "base_url": base_url,
        "response": response,
        "messages": messages,
        "message_count": len(messages),
        "events": events,
        "event_count": len(events),
        "usage": usage,
        "runtime_id": runtime_id,
        "thread_id": thread_id,
        "resumed": resumed,
        "completed": error is None,
        "failed": error is not None,
        "error": error,
    }
    if telemetry_runtime is not None or telemetry_error is not None:
        # ``degraded`` marks the referenced artifacts as possibly truncated; both keys
        # are absent on a clean run, so the telemetry block keeps its existing shape.
        telemetry: dict[str, Any] = dict(telemetry_runtime or {})
        if telemetry_error is not None:
            telemetry["degraded"] = True
            telemetry["error"] = telemetry_error
        if telemetry_quarantine_cause is not None:
            telemetry["quarantine_cause"] = telemetry_quarantine_cause
        output["telemetry"] = telemetry
    if relay_artifacts is not None:
        output["relay_artifacts"] = relay_artifacts
    # Absent when every configured server loaded, so clean runs keep their shape.
    if mcp := mcp_availability.output_block(unavailable_mcp_servers or []):
        output["mcp"] = mcp
    # Present when skills are configured; each result gets its own copy.
    if skills_report:
        output["skills"] = copy.deepcopy(skills_report)
    return output


def _extract_messages(result_state: Any) -> list[dict[str, Any]]:
    if not isinstance(result_state, dict):
        return []
    return _messages_to_dicts(result_state.get("messages") or [])


def _messages_to_dicts(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [item if isinstance(item, dict) else _message_to_dict(item) for item in raw]


def _dedup_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate messages by id, preserving first-seen order.

    With ``subgraphs=True`` the same message can surface in both a subgraph's
    updates and the parent stream; de-duplicating by id keeps usage aggregation
    from counting a message twice. Messages without an id are always kept.
    """

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for message in messages:
        message_id = message.get("id")
        if isinstance(message_id, str):
            if message_id in seen:
                continue
            seen.add(message_id)
        unique.append(message)
    return unique


def _message_to_dict(message: Any) -> dict[str, Any]:
    role = (
        getattr(message, "type", None) or getattr(message, "role", None) or "assistant"
    )
    content = getattr(message, "content", "")
    entry: dict[str, Any] = {"role": role, "content": content}
    message_id = getattr(message, "id", None)
    if message_id is not None:
        entry["id"] = message_id
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        entry["tool_calls"] = tool_calls
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        entry["usage"] = usage
    metadata = getattr(message, "response_metadata", None)
    if isinstance(metadata, dict) and metadata:
        entry["response_metadata"] = metadata
    return entry


def _final_response(messages: list[dict[str, Any]]) -> Any:
    for message in reversed(messages):
        role = str(message.get("role", ""))
        if role in ("ai", "assistant"):
            return message.get("content")
    return messages[-1].get("content") if messages else None


def _aggregate_usage(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    totals: dict[str, Any] = {}
    unknown: set[str] = set()
    cost = 0.0
    has_cost = False
    for message in messages:
        usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
        for source, target in (
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            value = usage.get(source)
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value <= (1 << 64) - 1
            ):
                totals[target] = int(totals.get(target, 0)) + value
            elif message.get("role") in ("ai", "assistant"):
                # One missing model response makes this dimension incomplete;
                # do not promote the other responses' partial sum to a total.
                unknown.add(target)
        # Cost is not part of LangChain's UsageMetadata; surface it only when a
        # model/provider reports it on the usage or response metadata.
        candidate = (
            usage.get("total_cost") or usage.get("cost") or _metadata_cost(message)
        )
        if (
            isinstance(candidate, (int, float))
            and not isinstance(candidate, bool)
            and math.isfinite(candidate)
            and candidate >= 0
        ):
            cost += float(candidate)
            has_cost = True
    if has_cost:
        totals["cost"] = cost
    for target in unknown:
        totals.pop(target, None)
    for target in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if totals.get(target, 0) > (1 << 64) - 1:
            totals.pop(target)
    return totals or None


def _metadata_cost(message: dict[str, Any]) -> float | None:
    metadata = message.get("response_metadata")
    if isinstance(metadata, dict):
        value = metadata.get("cost")
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        ):
            return float(value)
    return None


def _supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)
    parameters = signature.parameters.values()
    if any(param.kind == param.VAR_KEYWORD for param in parameters):
        return dict(kwargs)
    supported = {param.name for param in parameters}
    return {key: value for key, value in kwargs.items() if key in supported}



# Selected software reported by the shared lifecycle host; no dependency scan.
PROVENANCE_PACKAGES = (
    (
        "adapter",
        "nemo-fabric-adapters-deepagents",
        "nemo_fabric_adapters.deepagents.adapter",
    ),
    ("harness", "deepagents", "deepagents"),
    ("framework", "langgraph", "langgraph.graph"),
    ("framework", "langchain-core", "langchain_core"),
)

if __name__ == "__main__":
    main()
