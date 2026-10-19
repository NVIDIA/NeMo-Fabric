#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermes adapter for Fabric.

This adapter maps Fabric's normalized config into Hermes' native Python SDK
surface and invokes the installed Hermes runtime.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import sys
import weakref
from collections.abc import Callable
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import nemo_fabric_adapters.common.utils as common_utils
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentModelConfig
from nemo_fabric_adapter_contract.models import AgentRunError
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunResult
from nemo_fabric_adapter_contract.models import AgentRunStatus
from nemo_fabric_adapter_contract.models import AgentUsage
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common import instructions as common_instructions
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.hermes import configuration
from nemo_fabric_adapters.hermes import telemetry

# Default agent loop budget when FabricConfig.runtime.max_turns is unset.
# The actual internal hermes default  as of v0.21 is `sys.maxsize`
DEFAULT_MAX_ITERATIONS: int = 90
LOGGER = logging.getLogger(__name__)
hermes_mcp_server_config = configuration.hermes_mcp_server_config


def main() -> None:
    """Serve the persistent local-host lifecycle protocol."""

    lifecycle.serve(HermesRuntime, config_loader=AgentConfig.from_mapping)


class HermesRuntime:
    """One Hermes agent and session database owned by a Fabric runtime."""

    def __init__(self) -> None:
        self._started = False
        self._agent_config: AgentConfig | None = None
        self._runtime_id: str | None = None
        self._settings: dict[str, Any] = {}
        self._model_config: AgentModelConfig | None = None
        self._hermes_home: Path | None = None
        self._hermes_config_path: Path | None = None
        self._hermes_config: dict[str, Any] = {}
        self._enabled_toolsets: list[str] | None = None
        self._mcp_authentication_checked = False
        self._conversation_history: list[dict[str, Any]] | None = None
        self._session_db: Any = None
        self._agent: Any = None
        self._relay_plugin_config: dict[str, Any] | None = None
        self._relay_plugin_config_path: Path | None = None
        self._active_invoke_task: asyncio.Task[tuple[dict[str, Any], str]] | None = None
        # Hermes reports session totals; each invocation reports the change
        # from this baseline. A counter missing from it has no baseline.
        self._session_token_totals: dict[str, int] = _fresh_session_totals()
        self._uncounted_responses = 0
        # Provider responses whose usage left out a count, by count.
        self._unreported_counts = dict.fromkeys(_REPORTED_COUNTS, 0)

    async def start(self, payload: dict[str, Any]) -> None:
        if self._started:
            raise lifecycle.LifecycleError(
                "hermes_runtime_already_started",
                "Hermes runtime is already started",
            )

        try:
            agent_config = payload.get("config")
            if not isinstance(agent_config, AgentConfig):
                raise lifecycle.LifecycleError(
                    "hermes_invalid_config",
                    "Hermes requires a validated AgentConfig",
                )
            common_instructions.system_instruction(
                agent_config,
                adapter="Hermes",
                supported_modes={"replace"},
            )
            runtime_context = RuntimeContext.from_mapping(
                payload.get("runtime_context")
            )
            telemetry.validate_hermes_telemetry_provider(runtime_context)
            relay_enabled = bool(
                runtime_context.telemetry and runtime_context.telemetry.relay_enabled
            )
            if relay_enabled:
                telemetry.validate_hermes_relay_compatibility()
            self._agent_config = agent_config
            self._settings = configuration._settings(agent_config)
            model_config = configuration._selected_model(agent_config)
            self._model_config = model_config
            self._runtime_id = runtime_context.runtime_id
            self._hermes_home = (
                _artifact_root(runtime_context, common_utils.base_dir(payload))
                / ".fabric"
                / "hermes"
                / "runtimes"
                / runtime_context.runtime_id
            )
            self._hermes_home.mkdir(parents=True, exist_ok=True)
            os.environ["HOME"] = str(self._hermes_home)
            os.environ["HERMES_HOME"] = str(self._hermes_home)
            os.environ.setdefault("HERMES_YOLO_MODE", "1")
            os.environ.setdefault("HERMES_ACCEPT_HOOKS", "1")
            os.environ["HERMES_SESSION_SOURCE"] = "fabric"
            os.environ["TERMINAL_ENV"] = "local"
            os.environ.setdefault(
                "TERMINAL_TIMEOUT",
                str(self._settings.get("terminal_timeout", 60)),
            )

            if relay_enabled:
                relay_payload = {**payload, "config": agent_config.to_mapping()}
                (
                    self._relay_plugin_config_path,
                    self._relay_plugin_config,
                ) = telemetry.write_hermes_relay_plugin_config(relay_payload)
                for name in telemetry.HERMES_RELAY_ENV_NAMES:
                    os.environ.pop(name, None)
                os.environ["HERMES_NEMO_RELAY_PLUGINS_TOML"] = str(
                    self._relay_plugin_config_path
                )

            (
                self._hermes_config_path,
                self._hermes_config,
            ) = configuration.write_hermes_config(
                agent_config,
                self._hermes_home,
                # Workspace belongs to the per-runtime context, not AgentConfig.
                workspace=str(runtime_context.environment.workspace or "."),
                relay_enabled=relay_enabled,
            )
            api_key_env = configuration._api_key_env(model_config)
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise RuntimeError(f"{api_key_env} is required for Hermes mode")

            from hermes_cli.config import load_config
            from hermes_cli.plugins import discover_plugins
            from hermes_state import SessionDB
            from run_agent import AIAgent

            with redirect_stdout(StringIO()):
                if relay_enabled:
                    common_utils.reject_ambient_relay_plugin_config()
                discover_plugins(force=True)
                loaded_hermes_config = load_config()
                # Hermes 0.12+ no longer discovers MCP tools as an import side effect
                # (#16856). Fabric is a Hermes host: discover after config.yaml exists
                # and before AIAgent resolves mcp-* toolsets.
                # discover_mcp_tools uses a blocking 120s wait, wrapping it in
                # asyncio.to_thread to avoid blocking the loop.
                if self._hermes_config.get("mcp_servers"):
                    try:
                        from tools.mcp_tool_discovery import discover_mcp_tools
                    except ModuleNotFoundError as error:
                        if error.name != "tools.mcp_tool_discovery":
                            raise
                        from tools.mcp_tool import discover_mcp_tools

                    await asyncio.to_thread(discover_mcp_tools)

                self._enabled_toolsets = configuration.resolve_hermes_toolsets(
                    agent_config, loaded_hermes_config
                )
                self._session_db = SessionDB()
                self._conversation_history = None
                max_iterations = configuration._max_turns(agent_config)
                if max_iterations is None:
                    max_iterations = DEFAULT_MAX_ITERATIONS
                temperature = model_config.temperature
                top_p = model_config.top_p
                request_overrides = {
                    name: value
                    for name, value in (
                        ("temperature", temperature),
                        ("top_p", top_p),
                    )
                    if value is not None
                }
                self._agent = AIAgent(
                    **filter_supported_kwargs(
                        AIAgent,
                        base_url=model_config.base_url,
                        api_key=api_key,
                        provider=model_config.provider,
                        model=model_config.model,
                        max_iterations=int(max_iterations),
                        enabled_toolsets=self._enabled_toolsets,
                        disabled_toolsets=configuration.disabled_toolsets(agent_config)
                        or None,
                        quiet_mode=True,
                        skip_context_files=True,
                        skip_memory=True,
                        save_trajectories=bool(
                            self._settings.get("save_trajectories", False)
                        ),
                        max_tokens=model_config.max_tokens
                        or self._settings.get("max_tokens", 512),
                        request_overrides=request_overrides or None,
                        reasoning_config=self._settings.get(
                            "reasoning_config", {"effort": "none"}
                        ),
                        platform="fabric",
                        session_id=self._runtime_id,
                        session_db=self._session_db,
                    )
                )
                self._session_token_totals = _fresh_session_totals()
                self._observe_usage_reports(self._agent)
            self._agent_config = agent_config
            self._started = True
        except BaseException:
            await self.stop()
            raise

    async def invoke(
        self,
        request: AgentRunRequest,
        runtime_context: RuntimeContext,
    ) -> AgentRunResult:
        agent_config = self._agent_config
        model_config = self._model_config
        if (
            not self._started
            or self._agent is None
            or agent_config is None
            or model_config is None
        ):
            raise lifecycle.LifecycleError(
                "hermes_runtime_not_started",
                "Hermes runtime is not started",
            )
        if runtime_context.runtime_id != self._runtime_id:
            raise lifecycle.LifecycleError(
                "hermes_runtime_mismatch",
                "Hermes invocation does not match the active runtime",
            )

        user_message = common_utils.normalize_user_input(request.input)
        instructions = agent_config.instructions
        system_prompt = (
            instructions.system.content
            if instructions and instructions.system
            else None
        )

        await self._authenticate_mcp_servers()

        def run_hermes_turn() -> tuple[dict[str, Any], str]:
            if self._relay_plugin_config is not None:
                common_utils.reject_ambient_relay_plugin_config()
            try:
                return _invoke_hermes_turn(
                    agent=self._agent,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    conversation_history=self._conversation_history,
                    task_id=runtime_context.request_id,
                )
            finally:
                if self._relay_plugin_config is not None:
                    # Hermes writes TOML-configured ATIF at its session-finalization
                    # boundary for every supported Relay version. Fabric defines
                    # each invoke as an artifact-complete boundary, so finalize
                    # through Hermes' lifecycle instead of reaching into Relay
                    # directly.
                    telemetry.finalize_hermes_relay_session(str(self._agent.session_id))

        # Hermes' upstream Relay integration drives async Relay hooks from its
        # synchronous agent loop. Run that loop outside this lifecycle server's
        # event-loop thread so Hermes can own its Relay event loop.
        if self._active_invoke_task is not None:
            raise lifecycle.LifecycleError(
                "hermes_invocation_in_progress",
                "Hermes runtime already has an active invocation",
            )
        # Until this invocation's result arrives the baseline is unknown, so a
        # failed, cancelled or timed-out invocation never hands its tokens to
        # the next one.
        previous_totals, self._session_token_totals = self._session_token_totals, {}
        uncounted_before = self._uncounted_responses
        unreported_before = dict(self._unreported_counts)
        invoke_task = asyncio.create_task(asyncio.to_thread(run_hermes_turn))
        self._active_invoke_task = invoke_task

        def clear_active_invoke_task(
            completed_task: asyncio.Task[tuple[dict[str, Any], str]],
        ) -> None:
            if self._active_invoke_task is completed_task:
                self._active_invoke_task = None

        invoke_task.add_done_callback(clear_active_invoke_task)
        try:
            result, adapter_stdout = await asyncio.shield(invoke_task)
        finally:
            if invoke_task.done() and self._active_invoke_task is invoke_task:
                self._active_invoke_task = None
        messages = result.get("messages") or []
        if isinstance(messages, list):
            self._conversation_history = messages
        reported_error = result.get("error")
        failed = (
            bool(result.get("failed"))
            or bool(result.get("partial"))
            or bool(reported_error)
        )
        if isinstance(reported_error, str) and reported_error:
            reported_error = {
                "code": "hermes_reported_failure",
                "message": reported_error,
                "retryable": False,
                "metadata": {},
            }

        output = {
            "harness": "hermes",
            "adapter": "python",
            "mode": "hermes",
            "model": model_config.model,
            "base_url": model_config.base_url,
            "response": result.get("response") or result.get("final_response"),
            "completed": bool(result.get("completed")),
            "api_calls": result.get("api_calls"),
            "messages": messages,
            "message_count": len(messages),
            "adapter_stdout": adapter_stdout,
            "hermes_home": str(self._hermes_home),
            "hermes_config_path": str(self._hermes_config_path),
            "hermes_native_config": configuration.summarize_hermes_config(
                self._hermes_config
            ),
            "enabled_toolsets": self._enabled_toolsets,
        }
        if self._relay_plugin_config is not None:
            output["relay_runtime"] = {
                "enabled": True,
                "config_path": os.environ.get("FABRIC_RELAY_CONFIG_PATH"),
                "plugin_config_path": str(self._relay_plugin_config_path),
                "emitter": "hermes-agent/nemo-relay",
            }
            output["relay_artifacts"] = common_utils.collect_relay_artifacts(
                self._relay_plugin_config
            )
        error = None
        if failed:
            if isinstance(reported_error, dict):
                metadata = reported_error.get("metadata")
                provider = (
                    metadata.get("provider") if isinstance(metadata, dict) else None
                )
                error = AgentRunError(
                    code=str(reported_error.get("code") or "hermes_reported_failure"),
                    message=str(
                        reported_error.get("message")
                        or "Hermes did not complete the invocation"
                    ),
                    retryable=bool(reported_error.get("retryable", False)),
                    extensions={"provider": provider}
                    if isinstance(provider, str)
                    else {},
                )
            else:
                error = AgentRunError(
                    code="hermes_reported_failure",
                    message="Hermes did not complete the invocation",
                )
        usage, self._session_token_totals = _invocation_usage(
            result,
            previous_totals,
            unreported=frozenset(
                count
                for count, responses in self._unreported_counts.items()
                if responses != unreported_before[count]
            ),
        )
        if self._uncounted_responses != uncounted_before:
            # A provider response without usage, or one Hermes returned on
            # before recording its usage, added no tokens, so the change in
            # the counters is an undercount, not this invocation's.
            usage = None
        return AgentRunResult(
            status=AgentRunStatus.FAILED if failed else AgentRunStatus.SUCCEEDED,
            output=output,
            error=error,
            usage=usage,
        )

    def _observe_usage_reports(self, agent: Any) -> None:
        """Count the agent's provider responses that left usage unreported."""
        self._count_usage_less_responses(agent)
        self._count_unreported_counts(agent)

    def _count_usage_less_responses(self, agent: Any) -> None:
        """Count the agent's provider responses that carried no usage.

        Hermes counts such a response as an API call but leaves its token
        counters unchanged, so the counters cannot tell an unreported call
        from a reported zero. Hermes reports each of these responses to the
        context compressor's ``note_usage_less_response`` hook, so the
        adapter wraps that hook on the agent it owns.
        """
        compressor = agent.context_compressor
        note_usage_less_response = getattr(
            compressor, "note_usage_less_response", None
        )

        def count_and_note() -> None:
            self._uncounted_responses += 1
            if callable(note_usage_less_response):
                note_usage_less_response()

        compressor.note_usage_less_response = count_and_note

    def _count_unreported_counts(self, agent: Any) -> None:
        """Count the agent's provider responses whose usage left out a count.

        Hermes reads a missing count as 0, so its counters cannot tell an
        omitted count from a reported zero, and Hermes passes the raw usage to
        no per-agent hook. The adapter therefore registers the agent with
        process-wide wrappers around the functions Hermes' turn loop calls for
        each response; see ``_install_usage_observer``. An invalid, refused or
        truncated response that Hermes never records counts as a response
        without usage.
        """
        _install_usage_observer()

        def count_unreported(handler: str, response: Any, outcome: Any) -> None:
            if handler != "record_response_usage":
                # Hermes returns on an invalid, refused or truncated response
                # before recording its usage, unless the handler falls through.
                if getattr(outcome, "action", None) != "fallthrough":
                    self._uncounted_responses += 1
                return
            usage = getattr(response, "usage", None)
            # A response without usage is counted by the compressor hook.
            if usage:
                for count in _omitted_counts(agent, usage):
                    self._unreported_counts[count] += 1

        _USAGE_OBSERVERS[agent] = count_unreported

    async def _authenticate_mcp_servers(self) -> None:
        if self._mcp_authentication_checked:
            return

        oauth_server_names = {
            name
            for name, server in (self._hermes_config.get("mcp_servers") or {}).items()
            if server.get("auth") == "oauth"
        }
        if not oauth_server_names:
            self._mcp_authentication_checked = True
            return

        from tools.mcp_oauth import force_interactive_oauth
        from tools.mcp_tool import (
            discover_mcp_tools,
            get_mcp_status,
            refresh_agent_mcp_tools,
        )

        statuses = {
            status["name"]: status for status in await asyncio.to_thread(get_mcp_status)
        }
        disconnected = {
            name
            for name in oauth_server_names
            if not statuses.get(name, {}).get("connected")
        }
        if disconnected:

            def authenticate() -> None:
                lifecycle_stdin = sys.stdin
                try:
                    # Hermes forces interactive OAuth to enable the browser flow,
                    # which also starts an optional stdin paste reader. Fabric's
                    # stdin carries lifecycle messages, so give only that fallback
                    # an immediate EOF while the loopback callback remains active.
                    sys.stdin = StringIO()
                    with redirect_stdout(StringIO()), force_interactive_oauth():
                        discover_mcp_tools()
                finally:
                    sys.stdin = lifecycle_stdin

            try:
                await asyncio.to_thread(authenticate)
            except Exception as error:
                raise lifecycle.LifecycleError(
                    "hermes_mcp_authentication_failed",
                    "Hermes could not authenticate the configured MCP servers",
                    metadata={"servers": sorted(disconnected)},
                ) from error

            statuses = {
                status["name"]: status
                for status in await asyncio.to_thread(get_mcp_status)
            }
            disconnected = {
                name
                for name in oauth_server_names
                if not statuses.get(name, {}).get("connected")
            }
            if disconnected:
                raise lifecycle.LifecycleError(
                    "hermes_mcp_authentication_failed",
                    "Hermes could not authenticate the configured MCP servers",
                    metadata={"servers": sorted(disconnected)},
                )

            await asyncio.to_thread(
                refresh_agent_mcp_tools,
                self._agent,
                quiet_mode=True,
            )

        self._mcp_authentication_checked = True

    def _finalize_relay_session(self) -> None:
        if (
            self._relay_plugin_config is None
            or self._agent is None
            or self._invoke_hook is None
            or not self._relay_session_pending
        ):
            return
        if not self._relay_finalize_hook_invoked:
            self._invoke_hook(
                "on_session_finalize",
                session_id=getattr(self._agent, "session_id", ""),
                model=getattr(self._agent, "model", None) or self._relay_model_name,
                platform=getattr(self._agent, "platform", None) or "fabric",
            )
            self._relay_finalize_hook_invoked = True
        # Relay subscriber callbacks are queued. The long-lived plugin context
        # does not flush them until runtime shutdown, but invocation results
        # must include artifacts produced by this turn.
        from nemo_relay import subscribers

        subscribers.flush()
        self._relay_session_pending = False
        self._relay_finalize_hook_invoked = False

    async def stop(self) -> None:
        active_invoke_task = self._active_invoke_task
        errors: list[BaseException] = []
        if active_invoke_task is not None:
            try:
                await asyncio.shield(active_invoke_task)
            except BaseException as error:
                errors.append(error)
            finally:
                if self._active_invoke_task is active_invoke_task:
                    self._active_invoke_task = None

        agent = self._agent
        session_db = self._session_db
        if agent is not None:
            _USAGE_OBSERVERS.pop(agent, None)
        had_mcp_servers = bool(self._hermes_config.get("mcp_servers"))
        had_relay_plugin = self._relay_plugin_config_path is not None
        self._agent = None
        self._session_db = None
        self._agent_config = None
        self._runtime_id = None
        self._settings = {}
        self._model_config = None
        self._hermes_home = None
        self._hermes_config_path = None
        self._hermes_config = {}
        self._enabled_toolsets = None
        self._mcp_authentication_checked = False
        self._conversation_history = None
        self._relay_plugin_config = None
        self._relay_plugin_config_path = None
        self._started = False

        if had_relay_plugin:
            for name in telemetry.HERMES_RELAY_ENV_NAMES:
                os.environ.pop(name, None)

        if had_mcp_servers:
            try:
                try:
                    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
                except ModuleNotFoundError as error:
                    if error.name != "tools.mcp_tool_lifecycle":
                        raise
                    from tools.mcp_tool import shutdown_mcp_servers

                # wrapping this blocking call in asyncio.to_thread to avoid
                # blocking the loop.
                await asyncio.to_thread(shutdown_mcp_servers)
            except BaseException as error:
                errors.append(error)
        if agent is not None:
            try:
                agent.close()
            except BaseException as error:
                errors.append(error)
        if session_db is not None:
            try:
                session_db.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            for error in errors:
                if isinstance(error, asyncio.CancelledError):
                    raise error
                LOGGER.error(
                    "Hermes runtime cleanup failed",
                    exc_info=(type(error), error, error.__traceback__),
                )
            raise lifecycle.LifecycleError(
                "hermes_runtime_stop_failed",
                "Hermes runtime failed to stop cleanly",
            ) from errors[0]


def _artifact_root(runtime_context: RuntimeContext, base_dir: str) -> Path:
    root = runtime_context.artifacts.root
    if root:
        artifact_root = Path(str(root))
        if not artifact_root.is_absolute():
            artifact_root = Path(base_dir) / artifact_root
        return artifact_root.resolve()
    return Path(base_dir).resolve() / "artifacts"


def _invoke_hermes_turn(
    *,
    agent: Any,
    system_prompt: str | None,
    user_message: str,
    conversation_history: list[dict[str, Any]] | None,
    task_id: str | None,
) -> tuple[dict[str, Any], str]:
    hermes_stdout = StringIO()
    with redirect_stdout(hermes_stdout):
        conversation_kwargs = filter_supported_call_kwargs(
            agent.run_conversation,
            system_message=system_prompt,
            conversation_history=conversation_history,
            task_id=task_id,
            sync_honcho=False,
            dont_review=True,
        )
        result = agent.run_conversation(
            user_message,
            **conversation_kwargs,
        )
    return result, hermes_stdout.getvalue()


# Hermes' per-session token counters in a turn's result. Its input_tokens
# excludes cache reads and cache writes.
_SESSION_TOKEN_KEYS = (
    "input_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "output_tokens",
)


# Response observers by Hermes agent, called with each response Hermes'
# turn loop handles. Delegated subagents are never registered: their tokens
# never reach the registered agent's counters.
_USAGE_OBSERVERS: weakref.WeakKeyDictionary[Any, Callable[[str, Any, Any], None]] = (
    weakref.WeakKeyDictionary()
)
# The functions ``agent.turn_response_check.check_api_response`` calls with
# (agent, response, ...) for a provider response, by the module defining
# each. record_response_usage folds the usage; on an invalid response, a
# content-filter refusal or a reply cut off at the token limit, Hermes calls
# one of the other three and returns before recording the usage.
_RESPONSE_HANDLERS = {
    "record_response_usage": "agent.turn_usage",
    "retry_invalid_response": "agent.turn_response_check",
    "handle_content_policy_refusal": "agent.turn_truncation",
    "recover_from_truncation": "agent.turn_truncation",
}


def _observed_response_handler(name: str, handler: Callable[..., Any]) -> Any:
    """Wrap a Hermes response handler to show each call to the agent's observer."""

    def observed(agent: Any, response: Any, *args: Any, **kwargs: Any) -> Any:
        outcome = handler(agent, response, *args, **kwargs)
        observer = _USAGE_OBSERVERS.get(agent)
        if observer is not None:
            observer(name, response, outcome)
        return outcome

    observed.nemo_fabric_observed = True  # type: ignore[attr-defined]
    return observed


def _install_usage_observer() -> None:
    """Route Hermes' per-response handlers through the observers.

    ``check_api_response`` looks up each handler as a global of its module,
    and ``record_response_usage`` is the first place the raw usage reaches
    before Hermes normalizes it. The wrappers replace those module globals
    once per process and always delegate to Hermes' functions. If Hermes no
    longer calls a handler through that name, start fails rather than
    reporting an omitted or uncounted response as measured tokens.
    """
    import importlib

    from agent import turn_response_check

    called = turn_response_check.check_api_response.__code__.co_names
    handlers = {}
    for name, module in _RESPONSE_HANDLERS.items():
        current = getattr(turn_response_check, name, None)
        if getattr(current, "nemo_fabric_observed", False):
            continue
        if (
            current is None
            or name not in called
            or current is not getattr(importlib.import_module(module), name, None)
        ):
            raise lifecycle.LifecycleError(
                "hermes_usage_accounting_unsupported",
                f"This Hermes version handles responses outside "
                f"agent.turn_response_check.{name}",
            )
        handlers[name] = current
    for name, handler in handlers.items():
        setattr(turn_response_check, name, _observed_response_handler(name, handler))


_REPORTED_COUNTS = ("input", "cached", "output")
# The raw usage fields Hermes' normalize_usage reads for each count, by the
# usage shape it picks (agent/usage_pricing.py). A count is reported when every
# field of one alternative is present. Anthropic's input_tokens excludes cache
# reads and writes, so its input needs both cache fields, as in the Claude
# adapter. The other shapes' input count includes the cache, so a missing cache
# detail leaves only the cached count unknown.
_ANTHROPIC_FIELDS = {
    "input": (
        ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"),
    ),
    "cached": (("cache_read_input_tokens",),),
    "output": (("output_tokens",),),
}
_RESPONSES_FIELDS = {
    "input": (("input_tokens",),),
    "cached": (("input_tokens_details.cached_tokens",),),
    "output": (("output_tokens",),),
}
_CHAT_FIELDS = {
    "input": (("prompt_tokens",), ("input_tokens",)),
    "cached": (
        ("prompt_tokens_details.cached_tokens",),
        ("cache_read_input_tokens",),
        ("prompt_cache_hit_tokens",),
        ("cached_tokens",),
    ),
    "output": (("completion_tokens",), ("output_tokens",)),
}


def _omitted_counts(agent: Any, usage: Any) -> list[str]:
    """Return the counts that the provider's raw ``usage`` left out."""
    mode = str(getattr(agent, "api_mode", None) or "").strip().lower()
    provider = str(getattr(agent, "provider", None) or "").strip().lower()
    if mode == "anthropic_messages" or provider == "anthropic":
        fields = _ANTHROPIC_FIELDS
    elif mode == "codex_responses":
        fields = _RESPONSES_FIELDS
    else:
        fields = _CHAT_FIELDS
    return [
        count
        for count, alternatives in fields.items()
        if not any(
            all(_is_reported_count(usage, path) for path in alternative)
            for alternative in alternatives
        )
    ]


def _is_reported_count(usage: Any, path: str) -> bool:
    """Return whether the raw usage reports the dotted ``path`` as a count."""
    value = usage
    for field in path.split("."):
        value = (
            value.get(field) if isinstance(value, dict) else getattr(value, field, None)
        )
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _fresh_session_totals() -> dict[str, int]:
    """Return the baseline of a new Hermes agent, whose counters start at 0."""
    return dict.fromkeys(_SESSION_TOKEN_KEYS, 0)


def _invocation_usage(
    result: dict[str, Any],
    previous_totals: dict[str, int],
    *,
    unreported: frozenset[str] = frozenset(),
) -> tuple[AgentUsage | None, dict[str, int]]:
    """Return this invocation's usage and the session totals to keep as baseline.

    Hermes keeps one session across a runtime's invocations, so each counter is
    the change since ``previous_totals``. A counter that is missing from the
    result or from ``previous_totals`` (no baseline: the previous invocation
    failed or its result held no counters), or that went down, is unknown for
    this invocation. The usage therefore never holds a whole-session total or
    another invocation's tokens. ``unreported`` names the counts ("input",
    "cached", "output") that a provider response of this invocation left out
    and Hermes counted as 0. Those counts are unknown, and so is the total when
    input or output is.
    """
    totals = {
        key: value
        for key in _SESSION_TOKEN_KEYS
        if isinstance((value := result.get(key)), int)
        and not isinstance(value, bool)
        and 0 <= value <= (1 << 64) - 1
    }
    change = {
        key: value - previous_totals[key]
        for key, value in totals.items()
        if key in previous_totals and value >= previous_totals[key]
    }
    fields: dict[str, Any] = {}
    if "input" not in unreported and all(
        key in change
        for key in ("input_tokens", "cache_read_tokens", "cache_write_tokens")
    ):
        fields["input_tokens"] = (
            change["input_tokens"]
            + change["cache_read_tokens"]
            + change["cache_write_tokens"]
        )
        if "cached" not in unreported:
            fields["cached_input_tokens"] = change["cache_read_tokens"]
        fields["input_tokens_include_cache"] = True
    if "output" not in unreported and "output_tokens" in change:
        fields["output_tokens"] = change["output_tokens"]
    if "input_tokens" in fields and "output_tokens" in fields:
        fields["total_tokens"] = fields["input_tokens"] + fields["output_tokens"]
    return (AgentUsage(**fields) if fields else None), totals


def filter_supported_kwargs(callable_obj: Any, **kwargs: Any) -> dict[str, Any]:
    signature = inspect.signature(callable_obj.__init__)
    supported = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in supported}


def filter_supported_call_kwargs(func: Any, **kwargs: Any) -> dict[str, Any]:
    signature = inspect.signature(func)
    supported = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in supported}


if __name__ == "__main__":
    main()
