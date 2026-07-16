#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain Deep Agents adapter for Fabric.

Maps Fabric's normalized invocation onto the ``deepagents`` SDK and returns a
normalized Fabric result. Supports one-shot, multi-turn, and resumed execution
via a persistent LangGraph checkpointer keyed by the Fabric session id, so
conversation state survives across separate adapter invocations.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shlex
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
from typing import NamedTuple

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
import nemo_fabric_adapters.common.utils as common_utils

HARNESS = "deepagents"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
# Providers we serve through the OpenAI-compatible ``ChatOpenAI`` client.
OPENAI_COMPATIBLE_PROVIDERS = {"", "nvidia", "openai", "openai-compatible"}
# Providers whose default endpoint is NVIDIA's OpenAI-compatible gateway.
NVIDIA_DEFAULT_PROVIDERS = {"", "nvidia"}
# Conventional credential env var per provider; others must set api_key_env.
PROVIDER_DEFAULT_API_KEY_ENV = {
    "": "NVIDIA_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "openai": "OPENAI_API_KEY",
}
# MCP transports langchain-mcp-adapters accepts (after normalization).
VALID_MCP_TRANSPORTS = {"stdio", "sse", "streamable_http", "websocket"}
# create_deep_agent arguments Fabric derives from normalized config; the
# harness.settings.deepagents passthrough must not override them (doing so would
# bypass the normalized model config, MCP tool resolution, workspace confinement,
# or tool gating).
FABRIC_OWNED_AGENT_KEYS = frozenset(
    {"model", "tools", "backend", "skills", "system_prompt", "middleware", "checkpointer"}
)
# Documented, JSON-serializable create_deep_agent options callers may forward
# through harness.settings.deepagents. Executable objects (AgentMiddleware, BaseTool,
# Python callables) cannot cross the SDK->JSON->payload boundary and are excluded.
DEEPAGENTS_PASSTHROUGH_KEYS = frozenset({"subagents", "interrupt_on"})


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


def resolve_api_key_env(settings: dict[str, Any], model_config: dict[str, Any]) -> str:
    """Resolve the credential env var, defaulting per provider.

    An explicit ``api_key_env`` always wins. Otherwise nvidia/unspecified default
    to ``NVIDIA_API_KEY`` and openai to ``OPENAI_API_KEY``; any other provider must
    set ``api_key_env`` explicitly so a key is never sent to the wrong endpoint.
    """

    explicit = settings.get("api_key_env") or model_config.get("api_key_env")
    if explicit:
        return str(explicit)
    provider = (settings.get("provider") or model_config.get("provider") or "").lower()
    default = PROVIDER_DEFAULT_API_KEY_ENV.get(provider)
    if default is None:
        raise AdapterConfigError(f"models.default.api_key_env is required for provider '{provider}'.")
    return default


def main() -> None:
    """Subprocess entrypoint used by the ``python -m`` process path."""

    output = run(common_utils.load_payload())
    print(json.dumps(output, sort_keys=True))
    if output.get("failed"):
        raise SystemExit(2)


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """Fabric adapter entrypoint used by the ``python -m`` and native SDK paths."""

    import asyncio

    return asyncio.run(run_deepagents(payload))


def preflight_check(payload: dict[str, Any]) -> None:
    """Validate invocation-time prerequisites and fail fast with clear errors.

    These are runtime preflight checks, not ``fabric doctor`` checks: Fabric core
    has no adapter-doctor hook, so doctor cannot verify these. At invocation time
    the ``deepagents`` package must be importable and the configured
    model-provider credential must be present in the environment.
    """

    import importlib.util

    if importlib.util.find_spec("deepagents") is None:
        raise RuntimeError(
            "the 'deepagents' package is required for the Deep Agents adapter; install "
            "it with the 'deepagents' extra (pip install nemo-fabric-adapters-deepagents)."
        )

    settings = common_utils.settings_payload(payload)
    model_config = selected_model_config(payload)
    api_key_env = resolve_api_key_env(settings, model_config)
    if api_key_env not in os.environ:
        raise RuntimeError(
            f"the model-provider credential env var '{api_key_env}' is not set in the "
            "environment. Set it to your API key, or set models.default.api_key_env to the "
            "variable that holds it."
        )


def selected_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    settings = common_utils.settings_payload(payload)
    models = common_utils.models_payload(payload)
    return models.get(settings.get("model", "default"), {}) or {}


def resolve_base_url(settings: dict[str, Any], model_config: dict[str, Any]) -> str | None:
    base_url = (
        settings.get("base_url") or (model_config.get("settings") or {}).get("base_url") or model_config.get("base_url")
    )
    if base_url:
        return base_url
    provider = (settings.get("provider") or model_config.get("provider") or "").lower()
    # Only NVIDIA (or an unspecified provider) defaults to NVIDIA's endpoint; a
    # plain ``openai`` provider must fall through to ChatOpenAI's own default.
    if provider in NVIDIA_DEFAULT_PROVIDERS:
        return DEFAULT_NVIDIA_BASE_URL
    return None


def build_chat_model(payload: dict[str, Any]) -> tuple[Any, str, str | None]:
    """Build a LangChain chat model from Fabric model config.

    The default path targets NVIDIA-hosted OpenAI-compatible endpoints. A generic
    hook falls back to ``langchain.chat_models.init_chat_model`` for any provider
    that is not OpenAI-compatible, so other backends can be added without
    reworking the adapter.
    """

    settings = common_utils.settings_payload(payload)
    model_config = selected_model_config(payload)
    model_name = settings.get("model_name") or model_config.get("model")
    if not model_name:
        raise RuntimeError("models.default.model is required for the Deep Agents adapter")

    api_key_env = resolve_api_key_env(settings, model_config)
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is required for the Deep Agents adapter")

    provider = (settings.get("provider") or model_config.get("provider") or "nvidia").lower()
    base_url = resolve_base_url(settings, model_config)
    temperature = settings.get("temperature", model_config.get("temperature"))

    if provider not in OPENAI_COMPATIBLE_PROVIDERS:
        # Generic provider hook: honor an explicit non-OpenAI-compatible provider.
        from langchain.chat_models import init_chat_model

        kwargs = {"model": model_name, "model_provider": provider, "api_key": api_key}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if base_url:
            kwargs["base_url"] = base_url
        return init_chat_model(**_supported_kwargs(init_chat_model, kwargs)), model_name, base_url

    from langchain_openai import ChatOpenAI

    kwargs = {"model": model_name, "api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatOpenAI(**_supported_kwargs(ChatOpenAI, kwargs)), model_name, base_url


def resolve_backend(payload: dict[str, Any]) -> Any:
    """Root the Deep Agents filesystem backend at the Fabric workspace, if set."""

    environment = common_utils.environment_payload(payload)
    workspace = environment.get("workspace") or common_utils.settings_payload(payload).get("workspace")
    if not workspace:
        return None
    root = Path(str(workspace))
    if not root.is_absolute():
        root = Path(common_utils.base_dir(payload)) / root
    from deepagents.backends import FilesystemBackend

    # virtual_mode=True confines the agent to root_dir; absolute paths and ``..``
    # cannot escape it (and it silences the deepagents 0.6 default-flip warning).
    return FilesystemBackend(root_dir=str(root), virtual_mode=True)


async def resolve_tools(payload: dict[str, Any]) -> list[Any] | None:
    """Resolve Fabric MCP servers into Deep Agents tools."""

    tools = await _mcp_tools(payload)
    return tools or None


def _blocked_tool_names(payload: dict[str, Any]) -> set[str]:
    return set(common_utils.blocked_tools(payload))


def _tool_gate_middleware(
    is_blocked: Callable[[Any], bool], message: Callable[[Any], str]
) -> ToolGateMiddleware:
    return ToolGateMiddleware(is_blocked, message)


def blocked_tools_middleware(blocked: set[str]) -> Any:
    """Middleware that blocks explicitly denied tool calls across the full tool surface."""

    return _tool_gate_middleware(
        lambda name: name in blocked,
        lambda name: f"Tool '{name}' is blocked by the configured tools policy.",
    )


def resolve_skills(payload: dict[str, Any]) -> list[str] | None:
    """Map routed ``native.skill_paths`` onto the Deep Agents ``skills`` sources."""

    native = common_utils.capability_plan(payload).get("native") or {}
    skills = [str(path) for path in (native.get("skill_paths") or [])]
    return skills or None


async def _mcp_tools(payload: dict[str, Any]) -> list[Any]:
    native = common_utils.capability_plan(payload).get("native") or {}
    servers = native.get("mcp_servers") or {}
    connections = {name: _mcp_connection(name, spec) for name, spec in servers.items()}
    if not connections:
        return []
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(connections)
    return list(await client.get_tools())


def _mcp_connection(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    # A misconfigured server must fail loudly, not be silently dropped.
    if not isinstance(spec, dict):
        raise AdapterConfigError(f"MCP server '{name}' must be a mapping.")
    transport = str(spec.get("transport") or "").strip().lower().replace("-", "_")
    # McpServerPlan carries the URL or command in ``url``; there is no ``command``.
    target = os.path.expandvars(str(spec.get("url") or "")).strip()
    if not target:
        raise AdapterConfigError(f"MCP server '{name}' requires a url (or command in url).")
    if transport in ("stdio", "command", "process"):
        parts = shlex.split(target)
        if not parts:
            raise AdapterConfigError(f"MCP server '{name}' has an empty stdio command.")
        return {"transport": "stdio", "command": parts[0], "args": parts[1:]}
    if transport in ("", "http", "streamable_http", "streamablehttp"):
        transport = "streamable_http"
    if transport not in VALID_MCP_TRANSPORTS:
        raise AdapterConfigError(f"MCP server '{name}' has unsupported transport '{transport}'.")
    return {"transport": transport, "url": target}


# --- runtime / resume state ------------------------------------------------
#
# Resume is keyed by the Fabric ``runtime_id`` (stable across ``invoke`` calls in
# a started runtime, fresh for each one-shot ``run``), mirroring the Codex
# adapter. LangGraph owns the transcript via a persistent SQLite checkpointer;
# Fabric owns the runtime-to-LangGraph-thread correlation record.


def state_dir(payload: dict[str, Any]) -> Path:
    base_dir = Path(common_utils.base_dir(payload)).resolve()
    settings = common_utils.settings_payload(payload)
    configured = settings.get("state_dir")
    if configured:
        path = Path(str(configured))
        return path if path.is_absolute() else base_dir / path
    artifacts = common_utils.runtime_context(payload).get("artifacts") or {}
    root = artifacts.get("root") or os.environ.get("FABRIC_ARTIFACTS")
    if root:
        return Path(str(root)).resolve() / ".fabric" / "deepagents"
    return base_dir / "artifacts" / "deepagents" / ".fabric"


def runtime_state_paths(payload: dict[str, Any], runtime_id: str) -> tuple[Path, Path]:
    key = hashlib.sha256(runtime_id.encode("utf-8")).hexdigest()
    base = state_dir(payload) / "runtimes"
    return base / f"{key}.json", base / f"{key}.sqlite"


def load_thread_id(payload: dict[str, Any], runtime_id: str) -> str | None:
    json_path, _ = runtime_state_paths(payload, runtime_id)
    if not json_path.is_file():
        return None
    value = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("runtime_id") != runtime_id or not value.get("thread_id"):
        raise RuntimeError(f"invalid Deep Agents runtime state in {json_path}")
    return str(value["thread_id"])


def save_thread_id(payload: dict[str, Any], runtime_id: str, thread_id: str) -> None:
    json_path, _ = runtime_state_paths(payload, runtime_id)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    invocation_id = common_utils.runtime_context(payload).get("invocation_id") or "pending"
    tmp = json_path.with_suffix(f".{invocation_id}.tmp")
    tmp.write_text(json.dumps({"runtime_id": runtime_id, "thread_id": thread_id}, indent=2), encoding="utf-8")
    os.replace(tmp, json_path)


async def open_checkpointer(state_sqlite: Path) -> Any:
    """Open a persistent async LangGraph checkpointer so resume works across processes.

    The agent is driven with ``astream``, so the checkpointer must be async: the
    synchronous ``SqliteSaver`` raises ``NotImplementedError`` from its async methods.
    """

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    state_sqlite.parent.mkdir(parents=True, exist_ok=True)
    saver_cm = AsyncSqliteSaver.from_conn_string(str(state_sqlite))
    saver = await saver_cm.__aenter__()
    # Keep the context manager alive for the duration of the invocation.
    saver._fabric_cm = saver_cm  # type: ignore[attr-defined]
    return saver


async def close_checkpointer(checkpointer: Any) -> None:
    saver_cm = getattr(checkpointer, "_fabric_cm", None)
    if saver_cm is not None:
        await saver_cm.__aexit__(None, None, None)


# --- invocation ------------------------------------------------------------


async def build_agent_kwargs(payload: dict[str, Any], model: Any, settings: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "tools": await resolve_tools(payload),
        # deepagents 0.5.x/0.6.x take the system prompt as ``system_prompt``.
        "system_prompt": settings.get("system_prompt"),
        "skills": resolve_skills(payload),
        "backend": resolve_backend(payload),
    }
    # Deep Agents-specific settings (e.g. subagents, interrupt_on) pass through,
    # after validation against the documented JSON-serializable allow-list.
    extra = settings.get("deepagents")
    if extra is not None:
        kwargs.update(_validated_passthrough(extra))
    blocked = _blocked_tool_names(payload)
    if blocked:
        middleware = list(kwargs.get("middleware") or [])
        middleware.append(blocked_tools_middleware(blocked))
        kwargs["middleware"] = middleware
        kwargs["subagents"] = _gated_subagents(kwargs.get("subagents"), blocked)
    return {key: value for key, value in kwargs.items() if value is not None}


def _validated_passthrough(extra: Any) -> dict[str, Any]:
    """Validate the harness.settings.deepagents passthrough and return the safe subset.

    Only documented, JSON-serializable create_deep_agent options are forwarded.
    Fabric-owned keys cannot be overridden (that would bypass the normalized model
    config, MCP tool resolution, workspace confinement, and tool gating), and unknown
    keys fail clearly instead of being silently dropped.
    """

    if not isinstance(extra, dict):
        raise AdapterConfigError(
            f"harness.settings.deepagents must be a mapping of JSON-serializable options, not {type(extra).__name__}."
        )
    reserved = sorted(FABRIC_OWNED_AGENT_KEYS.intersection(extra))
    if reserved:
        raise AdapterConfigError(
            f"harness.settings.deepagents cannot override Fabric-owned keys {reserved}; "
            "they are derived from the normalized Fabric config."
        )
    unknown = sorted(set(extra) - DEEPAGENTS_PASSTHROUGH_KEYS)
    if unknown:
        raise AdapterConfigError(
            f"harness.settings.deepagents has unsupported option(s) {unknown}; supported "
            f"passthrough keys are {sorted(DEEPAGENTS_PASSTHROUGH_KEYS)}."
        )
    return dict(extra)


def _block_subagent(subagent: dict[str, Any], blocked: set[str]) -> dict[str, Any]:
    gated = dict(subagent)
    gated["middleware"] = [*(gated.get("middleware") or []), blocked_tools_middleware(blocked)]
    return gated


def _gated_subagents(subagents: Any, blocked: set[str]) -> list[dict[str, Any]]:
    if subagents is None:
        configured: list[Any] = []
    elif isinstance(subagents, list):
        configured = subagents
    else:
        raise AdapterConfigError(
            "harness.settings.deepagents.subagents must be a list when tools.blocked is configured."
        )

    gated: list[dict[str, Any]] = []
    for subagent in configured:
        if not isinstance(subagent, dict):
            raise AdapterConfigError("Deep Agents subagents must be mappings when tools.blocked is configured.")
        name = str(subagent.get("name") or "<unnamed>")
        if "graph_id" in subagent:
            raise AdapterConfigError(f"tools.blocked cannot be enforced for remote Deep Agents subagent '{name}'.")
        if "runnable" in subagent:
            raise AdapterConfigError(f"tools.blocked cannot be enforced for precompiled Deep Agents subagent '{name}'.")
        gated.append(_block_subagent(subagent, blocked))

    if not any(subagent.get("name") == "general-purpose" for subagent in gated):
        from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT

        gated.insert(0, _block_subagent(dict(GENERAL_PURPOSE_SUBAGENT), blocked))
    return gated


async def run_deepagents(payload: dict[str, Any]) -> dict[str, Any]:
    settings = common_utils.settings_payload(payload)
    request = payload.get("request") or {}
    telemetry_providers = common_utils.telemetry_providers(payload)
    relay_enabled = common_utils.relay_enabled(payload)
    telemetry_provider = "relay" if relay_enabled else ("native" if "native" in telemetry_providers else "")

    user_message = request.get("input") or ""
    if not isinstance(user_message, str):
        user_message = json.dumps(user_message, sort_keys=True)

    runtime_id = common_utils.runtime_context(payload).get("runtime_id")
    model_name: str | None = None
    base_url: str | None = None
    prior_thread_id: str | None = None
    thread_id: str | None = None
    observability: Observability | None = None
    result_state: Any = None
    events: list[dict[str, Any]] = []
    turn_messages: list[dict[str, Any]] = []
    error: str | None = None
    checkpointer = None
    try:
        # Preflight, model construction, and resume-state load run inside the guarded
        # scope so a misconfiguration (e.g. a missing credential) returns a normalized
        # failure result rather than raising a raw traceback.
        preflight_check(payload)
        model, model_name, base_url = build_chat_model(payload)
        prior_thread_id = load_thread_id(payload, runtime_id) if runtime_id else None
        thread_id = (prior_thread_id or uuid.uuid4().hex) if runtime_id else None
        observability = resolve_observability(payload, telemetry_provider, relay_enabled)

        agent_kwargs = await build_agent_kwargs(payload, model, settings)
        # Acquire the async checkpointer inside the guarded scope so a setup failure
        # (tools, backend, observability) can never leak the SQLite connection.
        if runtime_id:
            _, state_sqlite = runtime_state_paths(payload, runtime_id)
            checkpointer = await open_checkpointer(state_sqlite)
            agent_kwargs["checkpointer"] = checkpointer

        if observability is not None:
            api_config = common_utils.relay_api_plugin_config(observability.plugin_config)
            from nemo_relay import plugin
            from nemo_relay.integrations.deepagents import add_nemo_relay_integration

            wrapped = add_nemo_relay_integration(agent_kwargs)
            async with plugin.plugin(api_config):
                result_state, events, turn_messages = await invoke_agent(wrapped, user_message, thread_id)
        else:
            result_state, events, turn_messages = await invoke_agent(agent_kwargs, user_message, thread_id)
    except Exception as exc:  # normalized adapter failure
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if checkpointer is not None:
            await close_checkpointer(checkpointer)

    if error is None and runtime_id and thread_id:
        save_thread_id(payload, runtime_id, thread_id)

    telemetry_runtime: dict[str, Any] | None = None
    relay_artifacts: list[dict[str, str]] | None = None
    if observability is not None:
        telemetry_runtime = {
            "enabled": True,
            "provider": telemetry_provider,
            "emitter": observability.emitter,
        }
        if observability.collect_artifacts:
            relay_artifacts = common_utils.collect_relay_artifacts(observability.plugin_config)

    return normalize_output(
        model_name=model_name,
        base_url=base_url,
        runtime_id=runtime_id,
        thread_id=thread_id,
        resumed=bool(prior_thread_id),
        result_state=result_state,
        events=events,
        turn_messages=turn_messages,
        error=error,
        telemetry_runtime=telemetry_runtime,
        relay_artifacts=relay_artifacts,
    )


async def invoke_agent(
    agent_kwargs: dict[str, Any], user_message: str, thread_id: str | None
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the agent; return the final state, per-step events, and this turn's messages.

    On a resumed run the final ``values`` state also replays prior-turn messages,
    so usage/cost must be aggregated from the messages emitted *this* turn — the
    ``updates`` deltas — rather than the full final state.
    """

    from deepagents import create_deep_agent

    agent = create_deep_agent(**_supported_kwargs(create_deep_agent, agent_kwargs))
    inputs = {"messages": [{"role": "user", "content": user_message}]}
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None

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


def resolve_observability(
    payload: dict[str, Any], telemetry_provider: str, relay_enabled: bool
) -> Observability | None:
    """Resolve the nemo_relay observability plugin config for relay or native telemetry.

    Relay telemetry loads its plugin config from ``FABRIC_RELAY_CONFIG_PATH`` and
    collects ATOF/ATIF artifacts. Native telemetry reads
    ``telemetry_plan.native_config`` from the payload (e.g. an
    OpenTelemetry/OpenInference exporter) and exports spans directly to the
    configured collector without writing relay artifacts.
    """

    if relay_enabled and telemetry_provider == "relay":
        return Observability(
            common_utils.load_relay_plugin_config(payload),
            "deepagents.observability/nemo_relay",
            True,
        )
    if telemetry_provider == "native":
        native_config = common_utils.native_telemetry_config(payload)
        if native_config.get("components"):
            return Observability(native_config, "deepagents.observability/native", False)
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
) -> dict[str, Any]:
    messages = _extract_messages(result_state)
    response = _final_response(messages)
    # Aggregate usage/cost from this turn's messages only; the resumed final state
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
    if telemetry_runtime is not None:
        output["telemetry"] = telemetry_runtime
    if relay_artifacts is not None:
        output["relay_artifacts"] = relay_artifacts
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
    role = getattr(message, "type", None) or getattr(message, "role", None) or "assistant"
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
            if isinstance(value, int):
                totals[target] = int(totals.get(target, 0)) + value
        # Cost is not part of LangChain's UsageMetadata; surface it only when a
        # model/provider reports it on the usage or response metadata.
        candidate = usage.get("total_cost") or usage.get("cost") or _metadata_cost(message)
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            cost += float(candidate)
            has_cost = True
    if has_cost:
        totals["cost"] = cost
    return totals or None


def _metadata_cost(message: dict[str, Any]) -> float | None:
    metadata = message.get("response_metadata")
    if isinstance(metadata, dict):
        value = metadata.get("cost")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
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


if __name__ == "__main__":
    main()
