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
from collections.abc import Callable
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any
from typing import NamedTuple

from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunError
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunResult
from nemo_fabric_adapter_contract.models import AgentRunStatus
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common import mcp_availability
import nemo_fabric_adapters.common.utils as common_utils
from nemo_fabric_adapters.nooa.interactive import InteractiveAgentBuildContext
from nemo_fabric_adapters.nooa.interactive import InteractiveAgentTarget
from nemo_fabric_adapters.nooa.model_support import build_models
from nemo_fabric_adapters.nooa.model_support import close_models
from nemo_fabric_adapters.nooa.telemetry import RelayReport
from nemo_fabric_adapters.nooa.telemetry import RelayTelemetry

ADAPTER = "python"
HARNESS = "nooa"
MODE = "interactive_agent"
WORKFLOW_FACTORY_KIND = "interactive_agent_factory"
_TERMINAL_REASONS = frozenset({"DONE", "NEED_INPUT", "GET_USER_INPUT"})
# Upper bound on connecting to one MCP server and listing its tools at startup.
# Unreachable servers usually fail fast; this bounds a server that accepts the
# connection but never answers, so it is reported instead of stalling startup.
MCP_CONNECT_TIMEOUT_SECONDS = 60.0
# How long a timed-out connection may spend closing before startup moves on.
MCP_CONNECT_CANCEL_GRACE_SECONDS = 5.0

LOGGER = logging.getLogger(__name__)


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


class _McpServer(NamedTuple):
    """One validated MCP server, ready for NOOA to connect."""

    name: str
    transport: str
    target: str
    args: list[str]
    env: dict[str, str]
    headers: dict[str, str]


def _resolve_mcp_server(name: str, server: Any) -> _McpServer:
    """Validate one Fabric MCP server for NOOA before any connection."""
    transport = _validate_mcp_server(name, server)
    target = _mcp_target(name, server.url)
    headers: dict[str, str] = {}
    if transport != "stdio":
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
    return _McpServer(
        name=name,
        transport=transport,
        target=target,
        args=list(server.args),
        env=dict(server.env),
        headers=headers,
    )


async def _connect_mcp_server(
    manager: Any,
    server: _McpServer,
) -> tuple[Any, dict[str, str] | None]:
    """Create one NOOA MCP tool, reporting an unreachable server instead of raising."""

    async def create() -> Any:
        """Ask NOOA to connect the server and list its tools."""
        if server.transport == "stdio":
            return await manager.create_stdio_server(
                server.name,
                command=server.target,
                args=server.args,
                env=server.env,
            )
        return await manager.create_url_server(
            server.name,
            server.target,
            headers=server.headers,
            transport=server.transport,
        )

    task, timed_out = await mcp_availability.wait_bounded(
        create(),
        timeout=MCP_CONNECT_TIMEOUT_SECONDS,
        grace=MCP_CONNECT_CANCEL_GRACE_SECONDS,
    )
    if timed_out:
        return None, mcp_availability.unavailable_server(
            server.name,
            server.transport,
            mcp_availability.TIMED_OUT,
            "TimeoutError",
        )
    if task.cancelled():
        return None, mcp_availability.unavailable_server(
            server.name, server.transport, mcp_availability.FAILED, "CancelledError"
        )
    try:
        tool = task.result()
    except Exception as error:
        # NOOA wraps connection failures in a RuntimeError whose cause is the
        # transport error; report that cause's type, never the message.
        cause = error.__cause__ if isinstance(error.__cause__, Exception) else error
        return None, mcp_availability.unavailable_server(
            server.name,
            server.transport,
            mcp_availability.FAILED,
            mcp_availability.exception_type(cause),
        )
    return tool, None


async def _register_mcp_servers(
    agent: Any, config: AgentConfig
) -> list[dict[str, str]]:
    """Register every reachable MCP server and return the unavailable ones.

    Invalid server configuration still fails startup. A server that NOOA cannot
    connect to, or that does not list its tools in time, is skipped and
    reported, so the runtime starts with the remaining servers. Connection runs
    again on every runtime start; a skipped server is not remembered.
    """

    servers = config.mcp.servers if config.mcp is not None else {}
    if not servers:
        return []

    resolved = [
        _resolve_mcp_server(name, server) for name, server in sorted(servers.items())
    ]
    registry = _mcp_registry(agent)
    try:
        from nooa.mcp import MCPManager
    except ImportError as error:
        raise _config_error(
            "nooa_mcp_dependency_missing",
            "OO Agents MCP support is not installed",
        ) from error

    connected = await asyncio.gather(
        *(_connect_mcp_server(MCPManager, server) for server in resolved)
    )
    registered: list[str] = []
    unavailable: list[dict[str, str]] = []
    for server, (tool, failure) in zip(resolved, connected, strict=True):
        if failure is not None:
            unavailable.append(failure)
            continue
        registry_name = f"mcp.{server.name}"
        try:
            registry.register(registry_name, tool)
        except ValueError as error:
            raise _config_error(
                "nooa_mcp_registration_failed",
                f"OO Agents could not register MCP server {server.name!r}",
                server=server.name,
            ) from error
        registered.append(registry_name)
    mcp_availability.warn("OO Agents", unavailable)

    if registered:
        try:
            registry.activate(registered)
        except ValueError as error:
            raise _config_error(
                "nooa_mcp_activation_failed",
                "OO Agents could not activate the configured MCP servers",
            ) from error
    return unavailable


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


def _reason(result: Any) -> tuple[str, str, str | None]:
    kind = getattr(result, "kind", None)
    value = getattr(kind, "value", kind)
    explanation = getattr(result, "explanation", None)
    message = None
    if value is None:
        try:
            from nooa.interactive import Done
            from nooa.interactive import NeedInput
            from nooa.interactive import Waiting
        except ImportError:
            pass
        else:
            if isinstance(result, Done):
                value, message = "DONE", result.message
            elif isinstance(result, NeedInput):
                value, message = "NEED_INPUT", result.question
                explanation = next(
                    (
                        text
                        for text in (result.explanation, result.reason, result.question)
                        if isinstance(text, str) and text.strip()
                    ),
                    None,
                )
            elif isinstance(result, Waiting):
                value, message = "WAIT", result.message
    if value not in _TERMINAL_REASONS | {"WAIT"}:
        raise _InvalidRespondResult("unsupported response reason")
    if not isinstance(explanation, str) or not explanation.strip():
        raise _InvalidRespondResult("missing response explanation")
    return value, explanation, message


async def dispatch(
    target: InteractiveAgentTarget,
    text: str,
    *,
    on_message: Callable[[str], None] | None = None,
) -> tuple[str, str]:
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
        reason, explanation, message = _reason(result)
        if message is not None and on_message is not None:
            on_message(message)
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


def _with_mcp_report(
    result: AgentRunResult,
    unavailable: list[dict[str, str]],
) -> AgentRunResult:
    """Report the MCP servers this runtime could not use, when there are any."""

    if mcp := mcp_availability.output_block(unavailable):
        result.output = {**result.output, "mcp": mcp}
    return result


class NooaRuntime:
    """One registered OO Agents target owned by one Fabric runtime."""

    def __init__(self) -> None:
        """Create an unstarted runtime with no NOOA target."""
        self._runtime_id: str | None = None
        self._target: InteractiveAgentTarget | None = None
        self._models: dict[str, Any] = {}
        self._telemetry: RelayTelemetry | None = None
        self._mcp_unavailable: list[dict[str, str]] = []

    async def start(self, payload: dict[str, Any]) -> None:
        """Build the NOOA target and register the reachable MCP servers."""
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
        mcp_unavailable: list[dict[str, str]] = []
        try:
            models = await build_models(config)
            context = _build_context(payload, config, models)
            telemetry = RelayTelemetry(
                agent_name=common_utils.agent_name(payload),
                base_dir=context.base_dir,
                config=config,
            )
            target = _unwrap_target(await _await_if_needed(factory(context)))
            _validate_target(target)
            mcp_unavailable = await _register_mcp_servers(target.agent, config)
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
        self._mcp_unavailable = mcp_unavailable

    async def invoke(
        self,
        request: AgentRunRequest,
        runtime_context: RuntimeContext,
    ) -> AgentRunResult:
        """Run one NOOA turn and report MCP availability with its result."""
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
            return _with_mcp_report(
                _failure_output(
                    "nooa_invalid_request",
                    "OO Agents InteractiveAgent input must be a string",
                ),
                self._mcp_unavailable,
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
                reason, explanation = await dispatch(
                    self._target,
                    request.input,
                    on_message=lambda content: messages.append({"content": content}),
                )
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

        relay_invocation = await telemetry.invoke(
            agent=agent,
            runtime_context=runtime_context,
            call=invoke_target,
            relay_session_root=request.relay_session_root,
        )
        if not relay_invocation.called:
            result = _failure_output(
                "nooa_telemetry_setup_failed",
                "OO Agents Relay telemetry setup failed before target execution",
            )
        else:
            assert isinstance(relay_invocation.result, AgentRunResult)
            result = relay_invocation.result
        return _with_mcp_report(
            _with_telemetry(result, relay_invocation.report),
            self._mcp_unavailable,
        )

    async def stop(self) -> None:
        """Close the NOOA target and forget this runtime's MCP report."""
        target = self._target
        models = self._models
        telemetry = self._telemetry
        self._runtime_id = None
        self._target = None
        self._models = {}
        self._telemetry = None
        self._mcp_unavailable = []
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


# Selected software reported by the shared lifecycle host; no dependency scan.
PROVENANCE_PACKAGES = (
    ("adapter", "nemo-fabric-adapters-nooa", "nemo_fabric_adapters.nooa.adapter"),
    ("harness", "nooa", "nooa"),
)

if __name__ == "__main__":
    main()
