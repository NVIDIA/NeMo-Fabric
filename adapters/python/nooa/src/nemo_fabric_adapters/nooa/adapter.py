#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run registered OO Agents ``InteractiveAgent`` targets through NeMo Fabric."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import os
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunError
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunResult
from nemo_fabric_adapter_contract.models import AgentRunStatus
from nemo_fabric_adapter_contract.models import AgentUsage
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common import lifecycle
import nemo_fabric_adapters.common.utils as common_utils
from nemo_fabric_adapters.nooa.model_support import build_models
from nemo_fabric_adapters.nooa.model_support import close_models
from nemo_fabric_adapters.nooa.telemetry import RelayReport
from nemo_fabric_adapters.nooa.telemetry import RelayTelemetry

ADAPTER = "python"
HARNESS = "nooa"
MODE = "interactive_agent"
WORKFLOW_FACTORY_KIND = "interactive_agent_factory"
_TERMINAL_REASONS = frozenset({"DONE", "NEED_INPUT", "GET_USER_INPUT"})
_MAX_TOKEN_COUNT = (1 << 64) - 1
_USAGE_COUNTS = ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InteractiveAgentBuildContext:
    """Fabric-owned values supplied to one registered target factory."""

    config: AgentConfig
    models: Mapping[str, Any]
    system_instruction: str | None
    settings: Mapping[str, Any]
    skill_paths: tuple[Path, ...]
    runtime_id: str
    base_dir: Path
    workspace: Path
    artifact_root: Path | None


@dataclass(slots=True)
class InteractiveAgentTarget:
    """An interactive agent with optional target-owned lifecycle policy."""

    agent: Any
    close: Callable[[], Awaitable[None] | None] | None = None
    continue_after: Callable[[Any, str, str], Awaitable[bool] | bool] | None = None


class _InvalidRespondResult(Exception):
    """An OO Agents turn returned a value outside the dispatcher contract."""


def main() -> None:
    """Serve the persistent local-host lifecycle protocol."""

    lifecycle.serve(NooaRuntime, config_loader=AgentConfig.from_mapping)


def _config_error(code: str, message: str, **metadata: Any) -> lifecycle.LifecycleError:
    return lifecycle.LifecycleError(code, message, metadata=metadata or None)


def _agent_config(payload: dict[str, Any]) -> AgentConfig:
    config = payload.get("config")
    if not isinstance(config, AgentConfig):
        raise _config_error(
            "nooa_invalid_agent_config",
            "OO Agents requires a validated AgentConfig start payload",
        )
    return config


def _factory_ref(config: AgentConfig) -> str:
    workflow = config.workflow
    if workflow is None:
        raise _config_error(
            "nooa_invalid_workflow",
            "AgentConfig.workflow is required",
            field="workflow",
        )
    if workflow.entrypoint.kind != WORKFLOW_FACTORY_KIND:
        raise _config_error(
            "nooa_invalid_workflow",
            f"workflow.entrypoint.kind must equal {WORKFLOW_FACTORY_KIND!r}",
            field="workflow.entrypoint.kind",
        )
    return workflow.entrypoint.ref


def _load_factory(ref: str) -> Callable[[InteractiveAgentBuildContext], Any]:
    module_name, separator, attribute = ref.partition(":")
    if (
        not separator
        or not module_name
        or not attribute.isidentifier()
        or any(not part.isidentifier() for part in module_name.split("."))
    ):
        raise _config_error(
            "nooa_invalid_factory_ref",
            "OO Agents factory refs must use the form 'package.module:factory'",
            field="workflow.entrypoint.ref",
        )
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute)
    except Exception as error:
        raise _config_error(
            "nooa_factory_not_found",
            "The registered OO Agents target factory could not be loaded",
            field="workflow.entrypoint.ref",
        ) from error
    if not callable(factory):
        raise _config_error(
            "nooa_factory_not_callable",
            "The registered OO Agents target factory is not callable",
            field="workflow.entrypoint.ref",
        )
    return factory


def _path(value: Any, *, default: Path | None = None) -> Path | None:
    if value is None:
        return default
    if not isinstance(value, (str, Path)):
        raise _config_error(
            "nooa_invalid_runtime_context",
            "OO Agents received an invalid runtime filesystem path",
        )
    return Path(value)


def _resolved_skill_paths(config: AgentConfig, base_dir: Path) -> tuple[Path, ...]:
    values = config.skills.paths if config.skills is not None else []
    paths: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        path = Path(value)
        resolved = path if path.is_absolute() else base_dir / path
        if resolved not in seen:
            paths.append(resolved)
            seen.add(resolved)
    return tuple(paths)


def _mcp_registry(agent: Any) -> Any:
    registry = getattr(agent, "skills", None)
    required = {
        "agent.skills.register": getattr(registry, "register", None),
        "agent.skills.activate": getattr(registry, "activate", None),
    }
    missing = [name for name, value in required.items() if not callable(value)]
    if missing:
        raise _config_error(
            "nooa_mcp_unsupported_target",
            "The selected OO Agents target cannot register MCP servers",
            missing=missing,
        )
    return registry


def _mcp_target(name: str, value: str) -> str:
    target = os.path.expandvars(value).strip()
    if not target:
        raise _config_error(
            "nooa_invalid_mcp_server",
            f"OO Agents MCP server {name!r} requires a non-empty url",
            server=name,
        )
    return target


def _validate_mcp_server(name: str, server: Any) -> str:
    if server.allowed_tools is not None or server.blocked_tools:
        raise _config_error(
            "nooa_mcp_tool_filters_unsupported",
            "OO Agents does not support per-server MCP tool filters",
            server=name,
        )
    if server.authentication is not None:
        raise _config_error(
            "nooa_mcp_authentication_unsupported",
            "OO Agents does not support normalized MCP authentication",
            server=name,
        )

    transport = server.transport.strip().lower().replace("_", "-")
    if transport not in {"stdio", "sse", "streamable-http"}:
        raise _config_error(
            "nooa_unsupported_mcp_transport",
            f"OO Agents MCP server {name!r} has unsupported transport {transport!r}",
            server=name,
            transport=transport,
        )
    if transport == "stdio" and server.custom_headers:
        raise _config_error(
            "nooa_invalid_mcp_server",
            "OO Agents stdio MCP servers do not accept custom headers",
            server=name,
        )
    if transport != "stdio" and server.args:
        raise _config_error(
            "nooa_invalid_mcp_server",
            "OO Agents network MCP servers do not accept command arguments",
            server=name,
        )
    return transport


async def _register_mcp_servers(agent: Any, config: AgentConfig) -> None:
    servers = config.mcp.servers if config.mcp is not None else {}
    if not servers:
        return

    validated = {
        name: (_validate_mcp_server(name, server), server)
        for name, server in sorted(servers.items())
    }
    registry = _mcp_registry(agent)
    try:
        from nooa.mcp import MCPManager
    except ImportError as error:
        raise _config_error(
            "nooa_mcp_dependency_missing",
            "OO Agents MCP support is not installed",
        ) from error

    registered: list[str] = []
    for name, (transport, server) in validated.items():
        target = _mcp_target(name, server.url)
        if transport == "stdio":
            tool = await MCPManager.create_stdio_server(
                name,
                command=target,
                args=list(server.args),
                env=dict(server.env),
            )
        else:
            try:
                headers = common_utils.expand_http_headers(
                    name,
                    dict(server.custom_headers),
                )
            except (TypeError, ValueError) as error:
                raise _config_error(
                    "nooa_invalid_mcp_server",
                    f"OO Agents MCP server {name!r} has invalid custom headers",
                    server=name,
                ) from error
            tool = await MCPManager.create_url_server(
                name,
                target,
                headers=headers,
                transport=transport,
            )

        registry_name = f"mcp.{name}"
        try:
            registry.register(registry_name, tool)
        except ValueError as error:
            raise _config_error(
                "nooa_mcp_registration_failed",
                f"OO Agents could not register MCP server {name!r}",
                server=name,
            ) from error
        registered.append(registry_name)

    try:
        registry.activate(registered)
    except ValueError as error:
        raise _config_error(
            "nooa_mcp_activation_failed",
            "OO Agents could not activate the configured MCP servers",
        ) from error


def _build_context(
    payload: dict[str, Any],
    config: AgentConfig,
    models: Mapping[str, Any],
) -> InteractiveAgentBuildContext:
    try:
        runtime_id = common_utils.runtime_id(payload)
        base_dir = Path(common_utils.base_dir(payload))
    except ValueError as error:
        raise _config_error(
            "nooa_invalid_runtime_context",
            "OO Agents lifecycle payload is missing required runtime context",
        ) from error

    environment = common_utils.environment_payload(payload)
    workspace = _path(environment.get("workspace"), default=base_dir)
    artifact_root = _path(environment.get("artifacts"))
    assert workspace is not None
    workflow = config.workflow
    assert workflow is not None
    system_instruction = (
        config.instructions.system.content
        if config.instructions is not None and config.instructions.system is not None
        else None
    )
    return InteractiveAgentBuildContext(
        config=config,
        models=MappingProxyType(dict(models)),
        system_instruction=system_instruction,
        settings=MappingProxyType(dict(workflow.settings)),
        skill_paths=_resolved_skill_paths(config, base_dir),
        runtime_id=runtime_id,
        base_dir=base_dir,
        workspace=workspace,
        artifact_root=artifact_root,
    )


async def _await_if_needed(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _unwrap_target(value: Any) -> InteractiveAgentTarget:
    if isinstance(value, InteractiveAgentTarget):
        return value
    return InteractiveAgentTarget(agent=value)


def _validate_agent(agent: Any) -> None:
    queue_manager = getattr(agent, "queue_manager", None)
    event_manager = getattr(agent, "event_manager", None)
    required = {
        "agent.handle": getattr(agent, "handle", None),
        "agent.event_manager.on": getattr(event_manager, "on", None),
        "agent.queue_manager.channels": getattr(queue_manager, "channels", None),
        "agent.queue_manager.get_channel": getattr(queue_manager, "get_channel", None),
        "agent.queue_manager.race": getattr(queue_manager, "race", None),
        "agent.queue_manager.shutdown": getattr(queue_manager, "shutdown", None),
    }
    missing = [name for name, value in required.items() if not callable(value)]
    if missing:
        raise _config_error(
            "nooa_invalid_interactive_agent",
            "The registered target factory did not return a compatible InteractiveAgent",
            missing=missing,
        )


def _validate_target(target: InteractiveAgentTarget) -> None:
    _validate_agent(target.agent)
    if target.close is not None and not callable(target.close):
        raise _config_error(
            "nooa_invalid_interactive_agent",
            "The registered target cleanup callback is not callable",
        )
    if target.continue_after is not None and not callable(target.continue_after):
        raise _config_error(
            "nooa_invalid_interactive_agent",
            "The registered target continuation predicate is not callable",
        )


async def _close_target(target: InteractiveAgentTarget) -> None:
    if target.close is not None:
        await _await_if_needed(target.close())
        return

    close = getattr(target.agent, "close", None)
    if callable(close):
        await _await_if_needed(close())
        return

    queue_manager = getattr(target.agent, "queue_manager", None)
    shutdown = getattr(queue_manager, "shutdown", None)
    if callable(shutdown):
        await _await_if_needed(shutdown())


async def _close_resources(
    target: InteractiveAgentTarget | None,
    models: Mapping[str, Any],
) -> None:
    primary: BaseException | None = None
    if target is not None:
        try:
            await _close_target(target)
        except BaseException as error:
            primary = error
    try:
        await close_models(models)
    except BaseException as error:
        if primary is None:
            primary = error
        else:
            LOGGER.error(
                "OO Agents model cleanup failed while preserving target cleanup failure "
                "(error_type=%s)",
                type(error).__name__,
            )
    if primary is not None:
        raise primary


def _reason(result: Any) -> tuple[str, str]:
    kind = getattr(result, "kind", None)
    value = getattr(kind, "value", kind)
    explanation = getattr(result, "explanation", None)
    if value not in _TERMINAL_REASONS | {"WAIT"}:
        raise _InvalidRespondResult("unsupported response reason")
    if not isinstance(explanation, str) or not explanation.strip():
        raise _InvalidRespondResult("missing response explanation")
    return value, explanation


async def dispatch(target: InteractiveAgentTarget, text: str) -> tuple[str, str]:
    """Submit one user message and run the standard InteractiveAgent wake loop."""

    agent = target.agent
    queue_manager = agent.queue_manager
    queue_manager.get_channel("user_messages").put(text)
    while True:
        wins = await queue_manager.race()
        notification: dict[str, list[Any]] = {}
        for name, item in wins:
            notification.setdefault(name, []).append(item)
        for name, channel in queue_manager.channels().items():
            drained = channel.drain()
            if drained:
                notification.setdefault(name, []).extend(drained)

        result = await agent.handle(notification)
        reason, explanation = _reason(result)
        continue_after = False
        if reason != "WAIT" and target.continue_after is not None:
            continue_after = bool(
                await _await_if_needed(
                    target.continue_after(agent, reason, explanation)
                )
            )
        if reason != "WAIT" and not continue_after:
            return reason, explanation


def _failure_output(code: str, message: str) -> AgentRunResult:
    return AgentRunResult(
        status=AgentRunStatus.FAILED,
        output={
            "harness": HARNESS,
            "adapter": ADAPTER,
            "mode": MODE,
            "response": None,
            "messages": [],
            "completed": False,
        },
        error=AgentRunError(code=code, message=message),
    )


def _success_output(
    messages: list[dict[str, str]],
    reason: str,
    explanation: str,
) -> AgentRunResult:
    response = messages[-1]["content"] if messages else explanation
    return AgentRunResult(
        status=AgentRunStatus.SUCCEEDED,
        output={
            "harness": HARNESS,
            "adapter": ADAPTER,
            "mode": MODE,
            "response": response,
            "messages": messages,
            "reason": reason,
            "explanation": explanation,
            "completed": reason == "DONE",
        },
    )


def _token_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value <= _MAX_TOKEN_COUNT else None


def _cached_tokens(details: Any) -> int | None:
    if not isinstance(details, dict):
        return None
    return _token_count(details.get("cached_tokens"))


def _responses_api_call_usage(usage: Any) -> dict[str, int | None]:
    """Read the Responses API usage that NOOA's ``ResponsesClient`` returns."""
    if not isinstance(usage, dict):
        return dict.fromkeys(_USAGE_COUNTS)
    return {
        "input_tokens": _token_count(usage.get("input_tokens")),
        "cached_input_tokens": _cached_tokens(usage.get("input_tokens_details")),
        "output_tokens": _token_count(usage.get("output_tokens")),
        "total_tokens": _token_count(usage.get("total_tokens")),
    }


def _completion_call_usage(usage: Any) -> dict[str, int | None]:
    """Read the Chat Completions usage that NOOA's ``CompletionClient`` returns."""
    if not isinstance(usage, dict):
        return dict.fromkeys(_USAGE_COUNTS)
    # LiteLLM zero-fills missing prompt, completion and total counts, so the
    # adapter cannot distinguish those missing counts from reported zeros.
    # Those zeros remain unknown; explicit cached-token zeros are preserved.
    return {
        "input_tokens": _token_count(usage.get("prompt_tokens")) or None,
        "cached_input_tokens": _cached_tokens(usage.get("prompt_tokens_details")),
        "output_tokens": _token_count(usage.get("completion_tokens")) or None,
        "total_tokens": _token_count(usage.get("total_tokens")) or None,
    }


def _invocation_usage(calls: list[dict[str, int | None]]) -> AgentUsage | None:
    """Sum the model calls of one invocation; a count any call omits stays unknown."""
    if not calls:
        return None
    totals: dict[str, int | None] = {}
    for name in _USAGE_COUNTS:
        values = [call[name] for call in calls]
        total = None if None in values else sum(values)
        totals[name] = _token_count(total)
    if all(value is None for value in totals.values()):
        return None
    return AgentUsage(
        **totals,
        # Both NOOA client APIs count cached tokens inside the input tokens.
        input_tokens_include_cache=True if totals["input_tokens"] is not None else None,
    )


class _InvocationUsage:
    """Model calls owned by one invocation.

    The owner is bound in a ContextVar for the invocation, so tasks that the
    invocation's turns create, such as NOOA's history summarizer, report to it.
    A call that returns after the owner closes is dropped, never charged to a
    later invocation.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, int | None]] = []
        self.in_flight = 0
        self.closed = False


_INVOCATION_USAGE: ContextVar[_InvocationUsage | None] = ContextVar(
    "nooa_invocation_usage", default=None
)
# Bounds how long an invocation waits for a history summary that its last turn
# started. The wait also ends at Fabric's invocation deadline when that is sooner.
_SUMMARIZER_WAIT_SECONDS = 120.0


def _owned_summarizer_tasks(
    agent: Any, owner: _InvocationUsage
) -> list[asyncio.Task[Any]]:
    """Return the running history summaries that ``owner``'s turns started.

    NOOA's summarizer runs each summary in an ``asyncio.Task`` on the agent's
    model client, so the task's context holds the invocation that started it.
    """
    tasks = []
    for summarizer in getattr(agent, "_summarizers", None) or []:
        task = getattr(summarizer, "_pending_task", None)
        if (
            isinstance(task, asyncio.Task)
            and not task.done()
            and task.get_context().get(_INVOCATION_USAGE) is owner
        ):
            tasks.append(task)
    return tasks


def _running_jobs(agent: Any) -> list[Any]:
    """Return the agent's running queue jobs (``QueueManager.running_handles``)."""
    running_handles = getattr(
        getattr(agent, "queue_manager", None), "running_handles", None
    )
    if not callable(running_handles):
        return []
    jobs = running_handles()
    return list(jobs) if isinstance(jobs, list) else []


def _has_new_running_job(agent: Any, jobs_before: list[Any]) -> bool:
    """Return whether a queue job started during the invocation is still running.

    A job that ``QueueManager.spawn`` started during the invocation runs in a
    task that copies the invocation's context, so it can still make model calls
    for the invocation, including its first one.
    """
    return any(
        all(job is not earlier for earlier in jobs_before)
        for job in _running_jobs(agent)
    )


async def _close_invocation_usage(
    agent: Any,
    owner: _InvocationUsage,
    deadline: float,
    jobs_before: list[Any],
) -> AgentUsage | None:
    """Wait for a running history summary, then close the invocation's usage.

    Usage is unknown when work that the invocation started could still make a
    model call: a call in flight, an unfinished summary, or a running queue job.
    The adapter never waits for queue jobs, and it never cancels owned work.
    """
    summaries: list[asyncio.Task[Any]] = []
    try:
        if _has_new_running_job(agent, jobs_before):
            return None
        summaries = _owned_summarizer_tasks(agent, owner)
        if summaries:
            remaining = deadline - asyncio.get_running_loop().time()
            await asyncio.wait(
                summaries,
                timeout=max(0.0, min(_SUMMARIZER_WAIT_SECONDS, remaining)),
            )
    finally:
        owner.closed = True
    if (
        owner.in_flight
        or any(not task.done() for task in summaries)
        or _has_new_running_job(agent, jobs_before)
    ):
        return None
    return _invocation_usage(owner.calls)


def _record_model_usage(model: Any, *, responses_api: bool) -> None:
    """Report each call of one NOOA model client to the invocation that owns it."""
    call = model.call
    acall = model.acall
    call_usage = _responses_api_call_usage if responses_api else _completion_call_usage

    def recorded_call(*args: Any, **kwargs: Any) -> Any:
        owner = _INVOCATION_USAGE.get()
        if owner is None:
            return call(*args, **kwargs)
        owner.in_flight += 1
        try:
            response = call(*args, **kwargs)
        finally:
            owner.in_flight -= 1
        if not owner.closed:
            owner.calls.append(call_usage(getattr(response, "usage", None)))
        return response

    async def recorded_acall(*args: Any, **kwargs: Any) -> Any:
        owner = _INVOCATION_USAGE.get()
        if owner is None:
            return await acall(*args, **kwargs)
        owner.in_flight += 1
        try:
            response = await acall(*args, **kwargs)
        finally:
            owner.in_flight -= 1
        if not owner.closed:
            owner.calls.append(call_usage(getattr(response, "usage", None)))
        return response

    model.call = recorded_call
    model.acall = recorded_acall


def _with_telemetry(
    result: AgentRunResult,
    report: RelayReport | None,
) -> AgentRunResult:
    if report is None:
        return result
    output = dict(result.output)
    telemetry: dict[str, Any] = {
        "enabled": report.enabled,
        "provider": "relay",
        "emitter": "nooa.nemo_relay_middleware",
    }
    if report.error is not None:
        telemetry["degraded"] = True
        telemetry["error"] = report.error
    if report.quarantine_cause is not None:
        telemetry["quarantine_cause"] = report.quarantine_cause
    output["telemetry"] = telemetry
    output["relay_artifacts"] = list(report.artifacts)
    result.output = output
    return result


class NooaRuntime:
    """One registered OO Agents target owned by one Fabric runtime."""

    def __init__(self) -> None:
        self._runtime_id: str | None = None
        self._target: InteractiveAgentTarget | None = None
        self._models: dict[str, Any] = {}
        self._telemetry: RelayTelemetry | None = None

    async def start(self, payload: dict[str, Any]) -> None:
        if self._target is not None:
            raise lifecycle.LifecycleError(
                "nooa_runtime_already_started",
                "OO Agents runtime is already started",
            )

        config = _agent_config(payload)
        factory = _load_factory(_factory_ref(config))
        models: dict[str, Any] = {}
        target: InteractiveAgentTarget | None = None
        telemetry: RelayTelemetry | None = None
        try:
            models = await build_models(config)
            if models:
                # build_models gets every client from NOOA's get_llm_client,
                # which returns a ResponsesClient or a CompletionClient.
                from nooa.unifiedllm import ResponsesClient

                for model in models.values():
                    _record_model_usage(
                        model,
                        responses_api=isinstance(model, ResponsesClient),
                    )
            context = _build_context(payload, config, models)
            telemetry = RelayTelemetry(
                agent_name=common_utils.agent_name(payload),
                base_dir=context.base_dir,
                config=config,
            )
            target = _unwrap_target(await _await_if_needed(factory(context)))
            _validate_target(target)
            await _register_mcp_servers(target.agent, config)
        except asyncio.CancelledError:
            await _close_resources(target, models)
            if telemetry is not None:
                await telemetry.close()
            raise
        except lifecycle.LifecycleError:
            try:
                await _close_resources(target, models)
            except Exception as error:
                LOGGER.error(
                    "OO Agents cleanup failed after start error (error_type=%s)",
                    type(error).__name__,
                )
            if telemetry is not None:
                await telemetry.close()
            raise
        except Exception as error:
            try:
                await _close_resources(target, models)
            except Exception as cleanup_error:
                LOGGER.error(
                    "OO Agents cleanup failed after start error (error_type=%s)",
                    type(cleanup_error).__name__,
                )
            if telemetry is not None:
                await telemetry.close()
            raise lifecycle.LifecycleError(
                "nooa_target_start_failed",
                "The registered OO Agents target failed to start",
            ) from error

        self._runtime_id = context.runtime_id
        self._target = target
        self._models = models
        self._telemetry = telemetry

    async def invoke(
        self,
        request: AgentRunRequest,
        runtime_context: RuntimeContext,
    ) -> AgentRunResult:
        if self._target is None or self._runtime_id is None:
            raise lifecycle.LifecycleError(
                "nooa_runtime_not_started",
                "OO Agents runtime is not started",
            )
        if runtime_context.runtime_id != self._runtime_id:
            raise lifecycle.LifecycleError(
                "nooa_runtime_mismatch",
                "OO Agents invocation does not match the active runtime",
            )
        if not isinstance(request.input, str):
            return _failure_output(
                "nooa_invalid_request",
                "OO Agents InteractiveAgent input must be a string",
            )

        agent = self._target.agent
        telemetry = self._telemetry
        assert telemetry is not None

        async def invoke_target() -> AgentRunResult:
            messages: list[dict[str, str]] = []

            def capture_message(event: Any) -> None:
                content = getattr(event, "content", None)
                if isinstance(content, str):
                    messages.append({"content": content})

            unsubscribe: Callable[[], None] | None = None
            try:
                unsubscribe = agent.event_manager.on("AgentMessage", capture_message)
                reason, explanation = await dispatch(self._target, request.input)
            except asyncio.CancelledError:
                raise
            except _InvalidRespondResult:
                return _failure_output(
                    "nooa_invalid_respond_result",
                    "OO Agents InteractiveAgent returned an invalid RespondResult",
                )
            except Exception as error:
                LOGGER.error(
                    "OO Agents invocation failed (error_type=%s)",
                    type(error).__name__,
                )
                return _failure_output(
                    "nooa_target_invoke_failed",
                    "OO Agents target invocation failed; inspect adapter stderr for details",
                )
            finally:
                if unsubscribe is not None:
                    try:
                        unsubscribe()
                    except Exception as error:
                        LOGGER.error(
                            "OO Agents event unsubscribe failed (error_type=%s)",
                            type(error).__name__,
                        )

            return _success_output(messages, reason, explanation)

        deadline = lifecycle.invocation_deadline(runtime_context)
        jobs_before = _running_jobs(agent)
        owner = _InvocationUsage()
        token = _INVOCATION_USAGE.set(owner)
        try:
            relay_invocation = await telemetry.invoke(
                agent=agent,
                runtime_context=runtime_context,
                call=invoke_target,
            )
        except BaseException:
            owner.closed = True
            raise
        finally:
            _INVOCATION_USAGE.reset(token)
        usage = await _close_invocation_usage(agent, owner, deadline, jobs_before)
        if not relay_invocation.called:
            result = _failure_output(
                "nooa_telemetry_setup_failed",
                "OO Agents Relay telemetry setup failed before target execution",
            )
        else:
            assert isinstance(relay_invocation.result, AgentRunResult)
            result = relay_invocation.result
        result.usage = usage
        return _with_telemetry(result, relay_invocation.report)

    async def stop(self) -> None:
        target = self._target
        models = self._models
        telemetry = self._telemetry
        self._runtime_id = None
        self._target = None
        self._models = {}
        self._telemetry = None
        if target is None and not models and telemetry is None:
            return
        telemetry_error: Exception | None = None
        if telemetry is not None:
            try:
                await telemetry.close()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                telemetry_error = error
        try:
            await _close_resources(target, models)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise lifecycle.LifecycleError(
                "nooa_runtime_stop_failed",
                "OO Agents runtime failed to stop cleanly",
            ) from error
        if telemetry_error is not None:
            raise lifecycle.LifecycleError(
                "nooa_runtime_stop_failed",
                "OO Agents runtime failed to stop cleanly",
            ) from telemetry_error


if __name__ == "__main__":
    main()
