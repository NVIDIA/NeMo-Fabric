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
import uuid
from pathlib import Path
from typing import Any, NamedTuple

import nemo_fabric_adapters.common.utils as common_utils

HARNESS = "deepagents"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
# Providers we serve through the OpenAI-compatible ``ChatOpenAI`` client.
OPENAI_COMPATIBLE_PROVIDERS = {"", "nvidia", "openai", "openai-compatible"}


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
    """Fail fast with clear errors before invoking the harness.

    Covers the same ground as ``fabric doctor`` for this adapter: the
    ``deepagents`` package must be importable and the configured model-provider
    credential must be present in the environment.
    """

    import importlib.util

    if importlib.util.find_spec("deepagents") is None:
        raise RuntimeError(
            "the 'deepagents' package is required for the Deep Agents adapter; install "
            "it with the 'deepagents' extra (pip install nemo-fabric-adapters-deepagents)."
        )

    settings = common_utils.settings_payload(payload)
    model_config = selected_model_config(payload)
    api_key_env = settings.get("api_key_env") or model_config.get("api_key_env") or "NVIDIA_API_KEY"
    if api_key_env not in os.environ:
        raise RuntimeError(
            f"api_key_env={api_key_env} is defined in the configuration but is not set in the "
            "environment. Please set it to your API key."
        )


def selected_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    settings = common_utils.settings_payload(payload)
    models = common_utils.models_payload(payload)
    return models.get(settings.get("model", "default"), {}) or {}


def resolve_base_url(settings: dict[str, Any], model_config: dict[str, Any]) -> str | None:
    base_url = (
        settings.get("base_url")
        or (model_config.get("settings") or {}).get("base_url")
        or model_config.get("base_url")
    )
    if base_url:
        return base_url
    provider = (settings.get("provider") or model_config.get("provider") or "").lower()
    if provider in OPENAI_COMPATIBLE_PROVIDERS:
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

    api_key_env = settings.get("api_key_env") or model_config.get("api_key_env") or "NVIDIA_API_KEY"
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
        root = Path(common_utils.config_root(payload)) / root
    from deepagents.backends import FilesystemBackend

    return FilesystemBackend(root_dir=str(root))


async def resolve_tools(payload: dict[str, Any]) -> list[Any] | None:
    """Resolve Fabric MCP servers into Deep Agents tools, filtered by allowed tools.

    ``config.tools`` (surfaced as ``capability_plan.native.tools``) is treated as an
    allow-list of tool names: when present, only tools whose name is listed are
    exposed to the agent.
    """

    tools = await _mcp_tools(payload)
    allowed = _allowed_tool_names(payload)
    if allowed is not None:
        tools = [tool for tool in tools if getattr(tool, "name", None) in allowed]
    return tools or None


def _allowed_tool_names(payload: dict[str, Any]) -> set[str] | None:
    native = common_utils.capability_plan(payload).get("native") or {}
    tools = native.get("tools")
    if tools is None:
        tools = common_utils.settings_payload(payload).get("tools")
    if not isinstance(tools, (list, str)):
        return None
    names = set(common_utils.normalize_list(tools))
    return names or None


async def _mcp_tools(payload: dict[str, Any]) -> list[Any]:
    native = common_utils.capability_plan(payload).get("native") or {}
    servers = native.get("mcp_servers") or {}
    connections = {
        name: connection
        for name, connection in ((name, _mcp_connection(spec)) for name, spec in servers.items())
        if connection
    }
    if not connections:
        return []
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(connections)
    return list(await client.get_tools())


def _mcp_connection(spec: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(spec, dict):
        return None
    transport = str(spec.get("transport") or "").strip().lower().replace("-", "_")
    url = spec.get("url")
    command = spec.get("command")
    if transport in ("stdio", "command", "process") or (command and not url):
        resolved = os.path.expandvars(str(command or "")).strip()
        if not resolved:
            return None
        connection: dict[str, Any] = {"transport": "stdio", "command": resolved}
        args = spec.get("args")
        if args:
            connection["args"] = [str(arg) for arg in args]
        return connection
    if not url:
        return None
    if transport in ("", "http", "streamable_http", "streamablehttp"):
        transport = "streamable_http"
    return {"transport": transport, "url": os.path.expandvars(str(url))}


# --- runtime / resume state ------------------------------------------------
#
# Resume is keyed by the Fabric ``runtime_id`` (stable across ``invoke`` calls in
# a started runtime, fresh for each one-shot ``run``), mirroring the codex-cli
# adapter. LangGraph owns the transcript via a persistent SQLite checkpointer;
# Fabric owns the runtime-to-LangGraph-thread correlation record.


def state_dir(payload: dict[str, Any]) -> Path:
    config_root = Path(common_utils.config_root(payload)).resolve()
    settings = common_utils.settings_payload(payload)
    configured = settings.get("state_dir")
    if configured:
        path = Path(str(configured))
        return path if path.is_absolute() else config_root / path
    artifacts = common_utils.runtime_context(payload).get("artifacts") or {}
    root = artifacts.get("root") or os.environ.get("FABRIC_ARTIFACTS")
    if root:
        return Path(str(root)).resolve() / ".fabric" / "deepagents"
    return config_root / "artifacts" / "deepagents" / ".fabric"


def runtime_state_paths(payload: dict[str, Any], runtime_id: str) -> tuple[Path, Path]:
    key = hashlib.sha256(runtime_id.encode("utf-8")).hexdigest()
    base = state_dir(payload) / "runtimes"
    return base / f"{key}.json", base / f"{key}.sqlite"


def load_thread_id(payload: dict[str, Any], runtime_id: str) -> str | None:
    json_path, _ = runtime_state_paths(payload, runtime_id)
    if not json_path.is_file():
        return None
    value = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("runtime_id") != runtime_id or not value.get(
        "thread_id"
    ):
        raise RuntimeError(f"invalid Deep Agents runtime state in {json_path}")
    return str(value["thread_id"])


def save_thread_id(payload: dict[str, Any], runtime_id: str, thread_id: str) -> None:
    json_path, _ = runtime_state_paths(payload, runtime_id)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    invocation_id = common_utils.runtime_context(payload).get("invocation_id") or "pending"
    tmp = json_path.with_suffix(f".{invocation_id}.tmp")
    tmp.write_text(
        json.dumps({"runtime_id": runtime_id, "thread_id": thread_id}, indent=2), encoding="utf-8"
    )
    os.replace(tmp, json_path)


def open_checkpointer(state_sqlite: Path) -> Any:
    """Open a persistent LangGraph checkpointer so resume works across processes."""

    from langgraph.checkpoint.sqlite import SqliteSaver

    state_sqlite.parent.mkdir(parents=True, exist_ok=True)
    saver_cm = SqliteSaver.from_conn_string(str(state_sqlite))
    saver = saver_cm.__enter__()
    # Keep the context manager alive for the duration of the invocation.
    saver._fabric_cm = saver_cm  # type: ignore[attr-defined]
    return saver


def close_checkpointer(checkpointer: Any) -> None:
    saver_cm = getattr(checkpointer, "_fabric_cm", None)
    if saver_cm is not None:
        saver_cm.__exit__(None, None, None)


# --- invocation ------------------------------------------------------------


async def run_deepagents(payload: dict[str, Any]) -> dict[str, Any]:
    preflight_check(payload)
    settings = common_utils.settings_payload(payload)
    request = payload.get("request") or {}
    telemetry_provider = common_utils.telemetry_provider(payload)
    relay_enabled = os.environ.get("FABRIC_RELAY_ENABLED", "").strip().lower() == "true"

    user_message = request.get("input") or ""
    if not isinstance(user_message, str):
        user_message = json.dumps(user_message, sort_keys=True)

    model, model_name, base_url = build_chat_model(payload)

    runtime_id = common_utils.runtime_context(payload).get("runtime_id")
    prior_thread_id = load_thread_id(payload, runtime_id) if runtime_id else None

    checkpointer = None
    thread_id = None
    if runtime_id:
        _, state_sqlite = runtime_state_paths(payload, runtime_id)
        thread_id = prior_thread_id or uuid.uuid4().hex
        checkpointer = open_checkpointer(state_sqlite)

    agent_kwargs: dict[str, Any] = {
        "model": model,
        "tools": await resolve_tools(payload),
        # deepagents 0.5.x/0.6.x take the system prompt as ``system_prompt``.
        "system_prompt": settings.get("system_prompt"),
        "backend": resolve_backend(payload),
    }
    if checkpointer is not None:
        agent_kwargs["checkpointer"] = checkpointer
    agent_kwargs = {key: value for key, value in agent_kwargs.items() if value is not None}

    observability = resolve_observability(payload, telemetry_provider, relay_enabled)
    result_state: Any = None
    error: str | None = None
    try:
        if observability is not None:
            api_config = common_utils.relay_api_plugin_config(observability.plugin_config)
            from nemo_relay import plugin
            from nemo_relay.integrations.deepagents import add_nemo_relay_integration

            wrapped = add_nemo_relay_integration(agent_kwargs)
            async with plugin.plugin(api_config):
                result_state = await invoke_agent(wrapped, user_message, thread_id)
        else:
            result_state = await invoke_agent(agent_kwargs, user_message, thread_id)
    except Exception as exc:  # normalized adapter failure
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if checkpointer is not None:
            close_checkpointer(checkpointer)

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
        error=error,
        telemetry_runtime=telemetry_runtime,
        relay_artifacts=relay_artifacts,
    )


async def invoke_agent(agent_kwargs: dict[str, Any], user_message: str, thread_id: str | None) -> Any:
    from deepagents import create_deep_agent

    agent = create_deep_agent(**_supported_kwargs(create_deep_agent, agent_kwargs))
    inputs = {"messages": [{"role": "user", "content": user_message}]}
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None

    if hasattr(agent, "ainvoke"):
        return await agent.ainvoke(inputs, config) if config else await agent.ainvoke(inputs)
    return agent.invoke(inputs, config) if config else agent.invoke(inputs)


class Observability(NamedTuple):
    plugin_config: dict[str, Any]
    emitter: str
    collect_artifacts: bool


def resolve_observability(
    payload: dict[str, Any], telemetry_provider: str, relay_enabled: bool
) -> Observability | None:
    """Resolve the nemo_relay observability plugin config for relay or native telemetry.

    Relay telemetry loads its plugin config from ``FABRIC_RELAY_CONFIG_PATH`` and
    collects ATOF/ATIF artifacts. Native telemetry reads ``telemetry.config`` from
    the payload (e.g. an OpenTelemetry/OpenInference exporter) and exports spans
    directly to the configured collector without writing relay artifacts.
    """

    if relay_enabled and telemetry_provider == "relay":
        return Observability(
            common_utils.load_relay_plugin_config(payload),
            "deepagents.observability/nemo_relay",
            True,
        )
    telemetry = common_utils.telemetry_payload(payload)
    if telemetry.get("enabled") and telemetry_provider == "native":
        native_config = telemetry.get("config") or {}
        if isinstance(native_config, dict) and native_config.get("components"):
            return Observability(native_config, "deepagents.observability/native", False)
    return None


# --- normalization ---------------------------------------------------------


def normalize_output(
    *,
    model_name: str,
    base_url: str | None,
    runtime_id: str | None,
    thread_id: str | None,
    resumed: bool,
    result_state: Any,
    error: str | None,
    telemetry_runtime: dict[str, Any] | None,
    relay_artifacts: list[dict[str, str]] | None,
) -> dict[str, Any]:
    messages = _extract_messages(result_state)
    response = _final_response(messages)
    usage = _aggregate_usage(messages)

    output: dict[str, Any] = {
        "harness": HARNESS,
        "adapter": "python",
        "mode": "deepagents",
        "model": model_name,
        "base_url": base_url,
        "response": response,
        "messages": messages,
        "message_count": len(messages),
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
    raw = result_state.get("messages") or []
    messages: list[dict[str, Any]] = []
    for item in raw:
        messages.append(item if isinstance(item, dict) else _message_to_dict(item))
    return messages


def _message_to_dict(message: Any) -> dict[str, Any]:
    role = getattr(message, "type", None) or getattr(message, "role", None) or "assistant"
    content = getattr(message, "content", "")
    entry: dict[str, Any] = {"role": role, "content": content}
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        entry["tool_calls"] = tool_calls
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        entry["usage"] = usage
    return entry


def _final_response(messages: list[dict[str, Any]]) -> Any:
    for message in reversed(messages):
        role = str(message.get("role", ""))
        if role in ("ai", "assistant"):
            return message.get("content")
    return messages[-1].get("content") if messages else None


def _aggregate_usage(messages: list[dict[str, Any]]) -> dict[str, int] | None:
    totals: dict[str, int] = {}
    for message in messages:
        usage = message.get("usage")
        if not isinstance(usage, dict):
            continue
        for source, target in (
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            value = usage.get(source)
            if isinstance(value, int):
                totals[target] = totals.get(target, 0) + value
    return totals or None


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
