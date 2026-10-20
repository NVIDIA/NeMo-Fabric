#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermes adapter for Fabric.

This adapter maps Fabric's normalized config into Hermes' native Python SDK
surface and invokes the installed Hermes runtime.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import os
import sys
import threading
import weakref
from collections.abc import Callable
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
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
        self._uncounted_responses = 0
        # Provider responses whose usage left out a count, by count.
        self._unreported_counts = dict.fromkeys(_REPORTED_COUNTS, 0)
        # Tokens of provider responses Hermes returned on before adding them to
        # its session counters, in the counters' buckets.
        self._skipped_response_tokens = dict.fromkeys(_SESSION_TOKEN_KEYS, 0)
        # Compressions by a plugin context engine, whose model calls the
        # adapter cannot see.
        self._plugin_compressions = 0
        # Iteration-limit summary attempts started minus summary responses
        # observed; an attempt whose request raised leaves it above zero.
        self._unanswered_summary_attempts = 0
        # Subagents the agent built for delegated work; see _observe_delegation.
        self._delegated_subagents = 0

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
                _require_background_review_switch(AIAgent)
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
                        # Hermes' background skill review forks an agent after
                        # the turn returns and records its tokens outside the
                        # auxiliary client, so the invocation could not count
                        # them.
                        skip_background_review=True,
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

        auxiliary_calls = _AuxiliaryCalls(self._session_db)

        def run_hermes_turn() -> tuple[dict[str, Any], str]:
            if self._relay_plugin_config is not None:
                common_utils.reject_ambient_relay_plugin_config()
            # Hermes copies this context into the threads it starts for the
            # turn, so its auxiliary calls find the invocation that made them.
            owner = _AUXILIARY_CALLS.set(auxiliary_calls)
            try:
                return _invoke_hermes_turn(
                    agent=self._agent,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    conversation_history=self._conversation_history,
                    task_id=runtime_context.request_id,
                )
            finally:
                _AUXILIARY_CALLS.reset(owner)
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
        # No worker is running here, so the agent's session counters hold every
        # earlier invocation's tokens, including a failed or cancelled one's.
        previous_totals = _session_counters(self._agent)
        uncounted_before = self._uncounted_responses
        unreported_before = dict(self._unreported_counts)
        skipped_before = dict(self._skipped_response_tokens)
        plugin_compressions_before = self._plugin_compressions
        unanswered_summaries_before = self._unanswered_summary_attempts
        delegated_before = self._delegated_subagents
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
        auxiliary_tokens, auxiliary_unreported = auxiliary_calls.close()
        usage = _invocation_usage(
            _session_counters(self._agent),
            previous_totals,
            skipped={
                key: tokens - skipped_before[key] + auxiliary_tokens[key]
                for key, tokens in self._skipped_response_tokens.items()
            },
            unreported=auxiliary_unreported
            | frozenset(
                count
                for count, responses in self._unreported_counts.items()
                if responses != unreported_before[count]
            )
            # A plugin context engine's compression calls may bypass the
            # auxiliary client, a summary request that raised before its
            # response was observed may have been billed, and a subagent's
            # tokens go to its own counters, possibly after this invocation
            # returns, so every count they add to is unknown.
            | (
                frozenset(_REPORTED_COUNTS)
                if self._plugin_compressions != plugin_compressions_before
                or self._unanswered_summary_attempts != unanswered_summaries_before
                or self._delegated_subagents != delegated_before
                else frozenset()
            ),
        )
        if self._uncounted_responses != uncounted_before:
            # Hermes counted a provider response without usage as 0 tokens,
            # so the change in the counters is an undercount.
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
        self._count_plugin_compressions(agent)
        _install_auxiliary_usage_observer()
        _observe_background_reviews(agent)

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

    def _count_plugin_compressions(self, agent: Any) -> None:
        """Count the compressions of a plugin context engine on the agent.

        The built-in ``ContextCompressor`` sends every summary call through
        Hermes' auxiliary client, whose usage the auxiliary observer counts.
        A plugin context engine has no ``_call_summary_llm`` method and may
        call a model another way, so the adapter counts its ``compress``
        calls instead.
        """
        compressor = agent.context_compressor
        if callable(getattr(compressor, "_call_summary_llm", None)):
            return
        compress = compressor.compress

        def count_and_compress(*args: Any, **kwargs: Any) -> Any:
            self._plugin_compressions += 1
            return compress(*args, **kwargs)

        compressor.compress = count_and_compress

    def _count_unreported_counts(self, agent: Any) -> None:
        """Count the agent's provider responses whose usage left out a count.

        Hermes reads a missing count as 0, so its counters cannot tell an
        omitted count from a reported zero, and Hermes passes the raw usage to
        no per-agent hook. The adapter therefore registers the agent with
        process-wide wrappers around the functions Hermes' turn loop calls for
        each response; see ``_install_usage_observer``. Hermes returns on an
        invalid, refused or truncated response before adding its usage to the
        session counters, so the adapter adds that response's reported counts
        itself.
        """
        _install_usage_observer()

        def count_unreported(
            handler: str,
            response: Any,
            outcome: Any,
            provider: Any,
            api_mode: Any,
        ) -> None:
            usage = getattr(response, "usage", None)
            if handler == _ITERATION_SUMMARY_HANDLER:
                self._unanswered_summary_attempts -= 1
            if handler == "record_response_usage":
                # A response without usage is counted by the compressor hook.
                # An empty usage object is a report with every count left out.
                if usage is not None:
                    for count in _omitted_counts(usage, provider, api_mode):
                        self._unreported_counts[count] += 1
                return
            if not _skips_usage_recording(handler, outcome):
                return
            for count in _omitted_counts(usage, provider, api_mode):
                self._unreported_counts[count] += 1
            if usage:
                self._add_skipped_response_tokens(usage, provider, api_mode)

        def count_summary_attempt() -> None:
            self._unanswered_summary_attempts += 1

        _USAGE_OBSERVERS[agent] = count_unreported
        def count_delegated_subagent() -> None:
            self._delegated_subagents += 1

        _SUMMARY_ATTEMPT_OBSERVERS[agent] = count_summary_attempt
        _DELEGATION_OBSERVERS[agent] = count_delegated_subagent

    def _add_skipped_response_tokens(
        self, usage: Any, provider: Any, api_mode: Any
    ) -> None:
        """Add a response's usage to the tokens Hermes left out of its counters.

        Hermes' own normalization splits the usage into its counters' buckets.
        It fills an omitted field with 0; every count that needs an omitted
        field is already marked unreported.
        """
        for key, tokens in _counter_tokens(usage, provider, api_mode).items():
            self._skipped_response_tokens[key] += tokens

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
            _SUMMARY_ATTEMPT_OBSERVERS.pop(agent, None)
            _DELEGATION_OBSERVERS.pop(agent, None)
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
# turn loop handles, the handler's outcome, and the provider and API mode
# that served the response. Delegated subagents are never registered: their
# tokens never reach the registered agent's counters.
_USAGE_OBSERVERS: weakref.WeakKeyDictionary[
    Any, Callable[[str, Any, Any, Any, Any], None]
] = weakref.WeakKeyDictionary()
# Called before each iteration-limit summary request of the agent.
_SUMMARY_ATTEMPT_OBSERVERS: weakref.WeakKeyDictionary[Any, Callable[[], None]] = (
    weakref.WeakKeyDictionary()
)
# Called when the agent builds a subagent for delegated work.
_DELEGATION_OBSERVERS: weakref.WeakKeyDictionary[Any, Callable[[], None]] = (
    weakref.WeakKeyDictionary()
)
# The functions ``agent.turn_response_check.check_api_response`` calls with
# (agent, response, ...) for a provider response, by the module defining
# each. record_response_usage folds the usage; on an invalid response, a
# content-filter refusal or a reply cut off at the token limit, Hermes calls
# one of the other three and usually returns before recording the usage (see
# _skips_usage_recording).
_RESPONSE_HANDLERS = {
    "record_response_usage": "agent.turn_usage",
    "retry_invalid_response": "agent.turn_response_check",
    "handle_content_policy_refusal": "agent.turn_truncation",
    "recover_from_truncation": "agent.turn_truncation",
}


def _observed_response_handler(
    name: str, handler: Callable[..., Any], *, observe_first: bool = False
) -> Any:
    """Wrap a Hermes response handler to show each call to the agent's observer.

    Hermes passes the response positionally to some handlers and by keyword
    to others, so the wrapper forwards the call unchanged and reads the
    handler's ``response`` argument from its signature. With
    ``observe_first``, the observer sees the response, with no outcome,
    before the handler runs, so a handler that raises cannot hide its usage.
    """
    signature = inspect.signature(handler)

    def observed(agent: Any, *args: Any, **kwargs: Any) -> Any:
        # A handler can switch the agent to a fallback provider, so read the
        # route that served this response before the handler runs.
        provider = getattr(agent, "provider", None)
        api_mode = getattr(agent, "api_mode", None)
        if observe_first:
            observer = _USAGE_OBSERVERS.get(agent)
            if observer is not None:
                response = signature.bind(agent, *args, **kwargs).arguments["response"]
                observer(name, response, None, provider, api_mode)
            return handler(agent, *args, **kwargs)
        outcome = handler(agent, *args, **kwargs)
        observer = _USAGE_OBSERVERS.get(agent)
        if observer is not None:
            response = signature.bind(agent, *args, **kwargs).arguments["response"]
            observer(name, response, outcome, provider, api_mode)
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
            or "response" not in inspect.signature(current).parameters
        ):
            raise lifecycle.LifecycleError(
                "hermes_usage_accounting_unsupported",
                f"This Hermes version handles responses outside "
                f"agent.turn_response_check.{name}",
            )
        handlers[name] = current
    for name, handler in handlers.items():
        setattr(turn_response_check, name, _observed_response_handler(name, handler))
    _observe_iteration_summary()
    _observe_delegation()
    _observe_native_usage()


# The function each of Hermes' iteration-limit summary attempts passes its
# provider response to (agent/chat_completion_helpers.py), the attempt
# builders that call it, and where ``handle_max_iterations`` looks them up.
_ITERATION_SUMMARY_HANDLER = "_summary_text"
_ITERATION_SUMMARY_ATTEMPTS = (
    "_chat_summary_attempt",
    "_anthropic_summary_attempt",
    "_codex_summary_attempt",
)
_ITERATION_SUMMARY_BUILDERS = "_SUMMARY_ATTEMPT_BUILDERS"
_ITERATION_SUMMARY_DEFAULT_BUILDER = "_chat_summary_attempt"


def _observe_iteration_summary() -> None:
    """Route Hermes' iteration-limit summary requests through the observers.

    When a turn reaches ``max_iterations``, ``handle_max_iterations`` makes an
    extra summary request and never adds its usage to the session counters.
    Every summary attempt hands its response to ``_summary_text``, a global of
    ``agent.chat_completion_helpers``, so the adapter wraps that function once
    per process and counts the response like one Hermes skipped, before
    Hermes normalizes it. Hermes picks each attempt's builder from
    ``_SUMMARY_ATTEMPT_BUILDERS`` or the ``_chat_summary_attempt`` global, so
    the adapter also wraps the builders' attempts to note each request before
    it is sent: an attempt that raises before its response reaches
    ``_summary_text`` leaves the invocation's counts unknown. Start fails if
    Hermes no longer looks them up by those names.
    """
    from agent import chat_completion_helpers as helpers

    current = getattr(helpers, _ITERATION_SUMMARY_HANDLER, None)
    if getattr(current, "nemo_fabric_observed", False):
        return
    builders = getattr(helpers, _ITERATION_SUMMARY_BUILDERS, None)
    handle_max_iterations = getattr(helpers, "handle_max_iterations", None)
    attempts = {name: getattr(helpers, name, None) for name in _ITERATION_SUMMARY_ATTEMPTS}
    if (
        current is None
        or not all(
            _inner_function_calls(attempt, _ITERATION_SUMMARY_HANDLER)
            for attempt in attempts.values()
        )
        or "response" not in inspect.signature(current).parameters
        or not isinstance(builders, dict)
        or not all(builder in attempts.values() for builder in builders.values())
        or not {_ITERATION_SUMMARY_BUILDERS, _ITERATION_SUMMARY_DEFAULT_BUILDER}
        <= set(getattr(getattr(handle_max_iterations, "__code__", None), "co_names", ()))
    ):
        raise lifecycle.LifecycleError(
            "hermes_usage_accounting_unsupported",
            "This Hermes version makes its iteration-limit summary request "
            "outside agent.chat_completion_helpers._summary_text",
        )
    setattr(
        helpers,
        _ITERATION_SUMMARY_HANDLER,
        _observed_response_handler(
            _ITERATION_SUMMARY_HANDLER, current, observe_first=True
        ),
    )
    for mode, builder in list(builders.items()):
        builders[mode] = _counted_summary_attempts(builder)
    setattr(
        helpers,
        _ITERATION_SUMMARY_DEFAULT_BUILDER,
        _counted_summary_attempts(attempts[_ITERATION_SUMMARY_DEFAULT_BUILDER]),
    )


def _counted_summary_attempts(build: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a summary attempt builder so each attempt notes its request first."""

    def build_counted(agent: Any, *args: Any, **kwargs: Any) -> Any:
        attempt = build(agent, *args, **kwargs)

        def counted_attempt(*attempt_args: Any, **attempt_kwargs: Any) -> Any:
            note_attempt = _SUMMARY_ATTEMPT_OBSERVERS.get(agent)
            if note_attempt is not None:
                note_attempt()
            return attempt(*attempt_args, **attempt_kwargs)

        return counted_attempt

    return build_counted


def _observe_delegation() -> None:
    """Note each subagent an observed agent builds for delegated work.

    A subagent is its own ``AIAgent`` with its own session counters, and
    Hermes runs a top-level delegation in the background, so its calls can
    finish after the delegating invocation returns. Every subagent is built
    by ``tools.delegate_tool._build_child_agent``, which
    ``_build_child_preserving_parent_tools`` looks up at each call, so the
    adapter wraps it once per process. An invocation whose agent built a
    subagent has unknown counts; a subagent's own subagents belong to that
    delegated work. Start fails if Hermes no longer builds subagents there.
    """
    from tools import delegate_tool, delegate_tool_results

    current = getattr(delegate_tool, _DELEGATION_BUILDER, None)
    if getattr(current, "nemo_fabric_observed", False):
        return
    builder_caller = getattr(
        delegate_tool_results, "_build_child_preserving_parent_tools", None
    )
    if (
        current is None
        or "parent_agent" not in inspect.signature(current).parameters
        or _DELEGATION_BUILDER
        not in getattr(getattr(builder_caller, "__code__", None), "co_names", ())
    ):
        raise lifecycle.LifecycleError(
            "hermes_usage_accounting_unsupported",
            "This Hermes version builds subagents outside "
            "tools.delegate_tool._build_child_agent",
        )
    setattr(delegate_tool, _DELEGATION_BUILDER, _observed_subagent_builder(current))


def _observed_subagent_builder(build: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a subagent builder to note the build to its parent's observer."""
    signature = inspect.signature(build)

    def observed_build(*args: Any, **kwargs: Any) -> Any:
        parent = signature.bind(*args, **kwargs).arguments.get("parent_agent")
        note_delegation = (
            _DELEGATION_OBSERVERS.get(parent) if parent is not None else None
        )
        if note_delegation is not None:
            note_delegation()
        return build(*args, **kwargs)

    observed_build.nemo_fabric_observed = True  # type: ignore[attr-defined]
    return observed_build


_DELEGATION_BUILDER = "_build_child_agent"

# The attribute that holds the counts a native transport's raw usage reported,
# set on the usage object it builds before Hermes reads it. The leading
# underscore keeps it out of Hermes' Relay payloads (agent/relay_llm.py
# _jsonable skips private attributes).
_NATIVE_REPORTED_COUNTS = "_nemo_fabric_reported_counts"
# The raw usage fields of Hermes' native Bedrock Converse and Gemini
# transports, by the count each reports, in the shape of _CHAT_FIELDS. Bedrock
# documents its cache fields as optional, so its input needs only inputTokens;
# Gemini's output is visible and thinking tokens, either of which it reports.
_BEDROCK_FIELDS = {
    "input": (("inputTokens",),),
    "cached": (("cacheReadInputTokens",),),
    "output": (("outputTokens",),),
}
_GEMINI_FIELDS = {
    "input": (("promptTokenCount",),),
    "cached": (("cachedContentTokenCount",),),
    "output": (("candidatesTokenCount",), ("thoughtsTokenCount",)),
}


def _observe_native_usage() -> None:
    """Record which counts a native transport's raw usage reported.

    Hermes' Bedrock Converse and Gemini transports build an OpenAI-shaped
    usage object and fill every count the provider left out with 0
    (``agent/bedrock_adapter.py``, ``agent/gemini_native_adapter.py``), so the
    observer cannot tell an omitted count from a reported zero. The adapter
    wraps the functions that read the raw usage once per process and records
    the reported counts on the usage object they return; the streamed
    Bedrock path reads them from the stream's metadata event, and a stream
    without one reports no count. A transport whose module cannot be
    imported cannot serve a turn, so it is left as it is.
    """
    try:
        from agent import bedrock_adapter
    except ImportError:
        bedrock_adapter = None
    try:
        from agent import gemini_native_adapter
    except ImportError:
        gemini_native_adapter = None
    if bedrock_adapter is not None:
        _wrap_native_usage(
            bedrock_adapter,
            "normalize_converse_response",
            lambda args, kwargs: _raw_usage(_first_argument(args, kwargs, "response")),
            _BEDROCK_FIELDS,
        )
        _wrap_bedrock_stream(bedrock_adapter)
    if gemini_native_adapter is not None:
        _wrap_native_usage(
            gemini_native_adapter,
            "_usage_from_metadata",
            lambda args, kwargs: _first_argument(args, kwargs, "usage_meta"),
            _GEMINI_FIELDS,
            annotate_result=True,
        )


def _first_argument(args: tuple[Any, ...], kwargs: dict[str, Any], name: str) -> Any:
    return args[0] if args else kwargs.get(name)


def _raw_usage(response: Any) -> Any:
    return response.get("usage") if isinstance(response, dict) else None


def _native_reported_counts(raw: Any, fields: dict[str, Any]) -> frozenset[str]:
    """Return the counts that a native transport's raw ``usage`` reported."""
    if not isinstance(raw, dict):
        return frozenset()
    return frozenset(
        count
        for count, alternatives in fields.items()
        if any(
            all(_is_reported_count(raw, path) for path in alternative)
            for alternative in alternatives
        )
    )


def _annotate_usage(usage: Any, reported: frozenset[str]) -> None:
    if usage is not None:
        setattr(usage, _NATIVE_REPORTED_COUNTS, reported)


def _wrap_native_usage(
    module: Any,
    name: str,
    raw_usage: Callable[[tuple[Any, ...], dict[str, Any]], Any],
    fields: dict[str, Any],
    *,
    annotate_result: bool = False,
) -> None:
    """Wrap ``module.name`` to record its raw usage's reported counts.

    The wrapped function returns a response with ``usage``, or with
    ``annotate_result`` the usage object itself.
    """
    current = getattr(module, name, None)
    if getattr(current, "nemo_fabric_observed", False):
        return
    if not callable(current):
        raise lifecycle.LifecycleError(
            "hermes_usage_accounting_unsupported",
            f"This Hermes version reads native usage outside {module.__name__}.{name}",
        )

    def observed(*args: Any, **kwargs: Any) -> Any:
        reported = _native_reported_counts(raw_usage(args, kwargs), fields)
        result = current(*args, **kwargs)
        _annotate_usage(
            result if annotate_result else getattr(result, "usage", None), reported
        )
        return result

    observed.nemo_fabric_observed = True  # type: ignore[attr-defined]
    setattr(module, name, observed)


def _wrap_bedrock_stream(bedrock_adapter: Any) -> None:
    """Wrap Bedrock's ConverseStream reader to record its metadata usage."""
    name = "stream_converse_with_callbacks"
    current = getattr(bedrock_adapter, name, None)
    if getattr(current, "nemo_fabric_observed", False):
        return
    if not callable(current):
        raise lifecycle.LifecycleError(
            "hermes_usage_accounting_unsupported",
            f"This Hermes version reads native usage outside "
            f"agent.bedrock_adapter.{name}",
        )

    def observed(event_stream: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(event_stream, dict):
            return current(event_stream, *args, **kwargs)
        raw: dict[str, Any] = {}

        def events() -> Any:
            for event in event_stream.get("stream", []):
                metadata = event.get("metadata") if isinstance(event, dict) else None
                if isinstance(metadata, dict):
                    usage = metadata.get("usage")
                    raw.clear()
                    if isinstance(usage, dict):
                        raw.update(usage)
                yield event

        result = current({**event_stream, "stream": events()}, *args, **kwargs)
        _annotate_usage(
            getattr(result, "usage", None),
            _native_reported_counts(raw, _BEDROCK_FIELDS),
        )
        return result

    observed.nemo_fabric_observed = True  # type: ignore[attr-defined]
    setattr(bedrock_adapter, name, observed)


def _inner_function_calls(function: Any, name: str) -> bool:
    """Return whether a function defined inside ``function`` calls global ``name``."""
    code = getattr(function, "__code__", None)
    return code is not None and any(
        name in getattr(const, "co_names", ()) for const in code.co_consts
    )


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


def _skips_usage_recording(handler: str, outcome: Any) -> bool:
    """Return whether ``check_api_response`` returns after ``handler`` without
    recording the response's usage.

    It returns after a content-filter refusal, after an invalid response
    unless the handler's action is ``fallthrough``, and after a truncated
    response when the action is ``return``, ``break`` or ``continue``
    (``agent/turn_response_check.py``). Any other outcome falls through to
    ``record_response_usage``, which adds the response's usage itself. An
    iteration-limit summary response is never recorded.
    """
    action = getattr(outcome, "action", None)
    if handler == "recover_from_truncation":
        return action in ("return", "break", "continue")
    if handler == "retry_invalid_response":
        return action != "fallthrough"
    return True


def _omitted_counts(usage: Any, provider: Any, api_mode: Any) -> list[str]:
    """Return the counts that the provider's raw ``usage`` left out."""
    reported = getattr(usage, _NATIVE_REPORTED_COUNTS, None)
    if isinstance(reported, frozenset):
        return [count for count in _REPORTED_COUNTS if count not in reported]
    mode = str(api_mode or "").strip().lower()
    provider = str(provider or "").strip().lower()
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
    return _is_count(value)


# The largest count AgentUsage accepts.
_MAX_COUNT = (1 << 64) - 1


def _is_count(value: Any) -> bool:
    """Return whether ``value`` is a token count AgentUsage can hold."""
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= _MAX_COUNT


# No Anthropic usage was captured for a response.
_NOT_CAPTURED = object()


def _counter_tokens(usage: Any, provider: Any, api_mode: Any) -> dict[str, int]:
    """Return a response's usage in the session counters' buckets.

    Hermes' own normalization fills an omitted field with 0; callers mark
    every count that needs an omitted field as unreported.
    """
    from agent.usage_pricing import normalize_usage

    canonical = normalize_usage(usage, provider=provider, api_mode=api_mode)
    return {key: getattr(canonical, key) for key in _SESSION_TOKEN_KEYS}


class _AuxiliaryCalls:
    """The auxiliary model calls that one invocation's turn made.

    Hermes sends context-compression summaries, vision analysis and its other
    auxiliary requests through its auxiliary client, which records their
    usage in the session database but never adds it to the agent's session
    counters. Usage belongs to the invocation whose turn made the call. A
    request that raised, or that is still in flight when the invocation
    returns, may have been billed, so it leaves every count unknown. Once
    the invocation returns, its calls are closed: usage that arrives later
    is dropped, never moved to the next invocation.
    """

    def __init__(self, session_db: Any) -> None:
        self._session_db = session_db
        self._lock = threading.Lock()
        self._closed = False
        self._tokens = dict.fromkeys(_SESSION_TOKEN_KEYS, 0)
        self._unreported: set[str] = set()
        # Raw usage of Anthropic auxiliary responses, by the converted
        # response Hermes passes on, kept until that response is recorded.
        self._anthropic_usage: dict[int, tuple[Any, Any]] = {}
        # Auxiliary requests sent and not yet returned or raised.
        self._requests_in_flight = 0
        # Responses of returned requests that Hermes has not recorded yet,
        # by response.
        self._unrecorded_responses: dict[int, Any] = {}

    def owns_session(self, session_db: Any) -> bool:
        """Return whether Hermes bills the call to this runtime's session.

        A delegated subagent bills its own session database handle, and the
        adapter leaves subagents' tokens out, as their main calls never reach
        the runtime agent's counters.
        """
        return session_db is not None and session_db is self._session_db

    def add_response(self, response: Any, provider: Any) -> None:
        """Add an auxiliary response's reported counts; see ``record_response``."""
        self.record_response(response, provider, counted=True)

    def record_response(self, response: Any, provider: Any, *, counted: bool) -> None:
        """Record an auxiliary response that Hermes is recording.

        When ``counted`` is false, as for mixture-of-agents responses and
        other sessions' calls, the response is only resolved.

        Hermes normalizes auxiliary usage by provider alone, so the counts
        follow the same usage shape. For a response from Hermes' Anthropic
        auxiliary client, the counts come from the raw Anthropic usage the
        adapter captured before Hermes converted it. Hermes' converting clients
        (uncaptured Anthropic, Codex Responses and Copilot ACP) build the
        usage as a ``SimpleNamespace`` and fill an omitted count with 0, so
        such usage leaves every count unknown. The native Bedrock and Gemini
        usage carries the counts its raw usage reported, which decide the
        omitted counts as for a main-loop response. A response without
        usage, like one that left out a count, leaves those counts unknown.

        The response stays unresolved while its usage is read, and its counts
        and resolution are committed together, so an invocation that returns
        meanwhile reports every count unknown.
        """
        with self._lock:
            captured = self._anthropic_usage.get(id(response))
        anthropic_usage = (
            captured[1]
            if captured is not None and captured[0] is response
            else _NOT_CAPTURED
        )
        omitted: list[str] = []
        tokens: dict[str, int] | None = None
        if counted:
            try:
                omitted, tokens = self._read_usage(response, provider, anthropic_usage)
            except Exception:
                LOGGER.exception("Could not read Hermes auxiliary usage")
                omitted, tokens = list(_REPORTED_COUNTS), None
        with self._lock:
            if self._unrecorded_responses.get(id(response)) is response:
                del self._unrecorded_responses[id(response)]
            if captured is not None and captured[0] is response:
                del self._anthropic_usage[id(response)]
            if not counted:
                return
            if self._closed:
                LOGGER.warning(
                    "Hermes auxiliary usage arrived after its invocation returned"
                )
                return
            self._unreported.update(omitted)
            for key, value in (tokens or {}).items():
                self._tokens[key] += value

    @staticmethod
    def _read_usage(
        response: Any, provider: Any, anthropic_usage: Any
    ) -> tuple[list[str], dict[str, int] | None]:
        """Return a response's omitted counts and its tokens by counter."""
        api_mode = None
        usage = getattr(response, "usage", None)
        if anthropic_usage is not _NOT_CAPTURED:
            usage = anthropic_usage
            provider, api_mode = "anthropic", "anthropic_messages"
        # The native Bedrock and Gemini readers annotate the usage they build
        # with the counts their raw usage reported.
        annotated = isinstance(getattr(usage, _NATIVE_REPORTED_COUNTS, None), frozenset)
        if usage is None or (
            anthropic_usage is _NOT_CAPTURED
            and type(usage) is SimpleNamespace
            and not annotated
        ):
            return list(_REPORTED_COUNTS), None
        omitted = _omitted_counts(usage, provider, api_mode)
        return omitted, _counter_tokens(usage, provider, api_mode) if usage else None

    def add_anthropic_usage(self, response: Any, usage: Any) -> None:
        """Keep the raw usage of a converted Anthropic auxiliary response."""
        with self._lock:
            if not self._closed:
                self._anthropic_usage[id(response)] = (response, usage)

    def mark_unaccounted(self) -> None:
        """Leave every count unknown for a call the adapter could not read."""
        with self._lock:
            if not self._closed:
                self._unreported.update(_REPORTED_COUNTS)

    def start_request(self) -> None:
        """Note an auxiliary request sent to the provider."""
        with self._lock:
            if self._closed:
                LOGGER.warning(
                    "Hermes sent an auxiliary request after its invocation returned"
                )
            self._requests_in_flight += 1

    def finish_request(self, response: Any = None) -> None:
        """Note that an auxiliary request returned ``response`` or raised.

        A returned response stays unresolved until Hermes records it.
        """
        with self._lock:
            self._requests_in_flight -= 1
            if response is not None:
                self._unrecorded_responses[id(response)] = response

    def replace_unrecorded_response(self, response: Any, replacement: Any) -> None:
        """Track ``replacement`` in place of the returned ``response``."""
        with self._lock:
            if self._unrecorded_responses.get(id(response)) is response:
                del self._unrecorded_responses[id(response)]
                self._unrecorded_responses[id(replacement)] = replacement

    def close(self) -> tuple[dict[str, int], frozenset[str]]:
        """Return the calls' tokens and unknown counts, and drop later usage.

        A request still in flight, a returned response Hermes has not
        recorded, or a captured Anthropic usage whose response was never
        recorded, means the counts miss work this invocation started, so
        every count is unknown.
        """
        with self._lock:
            self._closed = True
            if (
                self._requests_in_flight
                or self._unrecorded_responses
                or self._anthropic_usage
            ):
                self._unreported.update(_REPORTED_COUNTS)
            self._anthropic_usage.clear()
            self._unrecorded_responses.clear()
            return dict(self._tokens), frozenset(self._unreported)


# The invocation whose Hermes turn runs in this context.
_AUXILIARY_CALLS: contextvars.ContextVar[_AuxiliaryCalls | None] = (
    contextvars.ContextVar("nemo_fabric_hermes_auxiliary_calls", default=None)
)


def _owned_auxiliary_calls() -> _AuxiliaryCalls | None:
    """Return the invocation that owns an auxiliary call made in this context.

    The call is owned when an invocation's turn runs in this context and
    Hermes bills the call to that runtime's session.
    """
    from agent import aux_accounting

    calls = _AUXILIARY_CALLS.get()
    session = aux_accounting._accounting.get()
    if calls is None or session is None or not calls.owns_session(session[0]):
        return None
    return calls


# Provider dispatches of the auxiliary request running in this context.
_REQUEST_DISPATCHES: contextvars.ContextVar[list[int] | None] = contextvars.ContextVar(
    "nemo_fabric_hermes_request_dispatches", default=None
)


def _observed_notify_aux_dispatch(notify: Callable[..., Any]) -> Any:
    """Wrap Hermes' dispatch notification to count each provider dispatch."""

    def observed(*args: Any, **kwargs: Any) -> Any:
        dispatches = _REQUEST_DISPATCHES.get()
        if dispatches is not None:
            dispatches[0] += 1
        return notify(*args, **kwargs)

    observed.nemo_fabric_observed = True  # type: ignore[attr-defined]
    return observed


def _settle_request(calls: _AuxiliaryCalls, response: Any, dispatches: int) -> None:
    """Settle a returned auxiliary request.

    Hermes rejects a ``None`` response before recording it, and a request
    that Hermes sent more than once, such as a streamed request it retried
    without streaming, returns only the last attempt's usage. Either may
    have been billed, so it leaves every count unknown.
    """
    if response is None or dispatches > 1:
        calls.mark_unaccounted()
    calls.finish_request(response)


def _observed_relay_completion(completion: Callable[..., Any]) -> Any:
    """Wrap Hermes' synchronous auxiliary request function to track each
    request. A request that raised may have been billed, so it leaves every
    count unknown, and so does a request still running, or a response not
    yet recorded, when the invocation returns."""

    def observed(*args: Any, **kwargs: Any) -> Any:
        calls = _owned_auxiliary_calls()
        if calls is None:
            return completion(*args, **kwargs)
        calls.start_request()
        dispatches = [0]
        counting = _REQUEST_DISPATCHES.set(dispatches)
        try:
            response = completion(*args, **kwargs)
        except BaseException:
            calls.mark_unaccounted()
            calls.finish_request()
            raise
        finally:
            _REQUEST_DISPATCHES.reset(counting)
        _settle_request(calls, response, dispatches[0])
        return response

    observed.nemo_fabric_observed = True  # type: ignore[attr-defined]
    return observed


def _observed_async_relay_completion(completion: Callable[..., Any]) -> Any:
    """Async twin of ``_observed_relay_completion``."""

    async def observed(*args: Any, **kwargs: Any) -> Any:
        calls = _owned_auxiliary_calls()
        if calls is None:
            return await completion(*args, **kwargs)
        calls.start_request()
        dispatches = [0]
        counting = _REQUEST_DISPATCHES.set(dispatches)
        try:
            response = await completion(*args, **kwargs)
        except BaseException:
            calls.mark_unaccounted()
            calls.finish_request()
            raise
        finally:
            _REQUEST_DISPATCHES.reset(counting)
        _settle_request(calls, response, dispatches[0])
        return response

    observed.nemo_fabric_observed = True  # type: ignore[attr-defined]
    return observed


def _observed_record_aux_usage(record_aux_usage: Callable[..., Any]) -> Any:
    """Wrap Hermes' auxiliary usage recorder to show each response to its turn's
    invocation before Hermes normalizes the usage."""
    from agent import aux_accounting

    def observed(response: Any, task: Any, *args: Any, **kwargs: Any) -> Any:
        calls = _AUXILIARY_CALLS.get()
        if calls is not None:
            # MoA advisor and aggregator usage is already in the session
            # counters, so such a response, like another session's, is only
            # resolved.
            calls.record_response(
                response,
                kwargs.get("provider"),
                counted=task not in aux_accounting._EXCLUDED_TASKS
                and _owned_auxiliary_calls() is calls,
            )
        return record_aux_usage(response, task, *args, **kwargs)

    observed.nemo_fabric_observed = True  # type: ignore[attr-defined]
    return observed


# The raw usages of the Anthropic messages that Hermes' Anthropic auxiliary
# client receives in this context, while it converts one response.
_ANTHROPIC_AUX_CAPTURE: contextvars.ContextVar[list[Any] | None] = (
    contextvars.ContextVar("nemo_fabric_hermes_anthropic_aux_capture", default=None)
)


def _observed_anthropic_aux_create(create: Callable[..., Any]) -> Any:
    """Wrap the Anthropic auxiliary adapter's ``create`` to keep the raw usage.

    The adapter converts the Anthropic message into an OpenAI-shaped response
    whose usage leaves out the cache counts and fills missing counts with 0.
    Only ``create_anthropic_message``, which the adapter calls, sees the raw
    message. A response that later reaches the observer as another object,
    such as one a Relay execution intercept replaced, leaves its capture
    unrecorded, so the invocation's counts are unknown.
    """

    def observed(self: Any, *args: Any, **kwargs: Any) -> Any:
        captured: list[Any] = []
        capture = _ANTHROPIC_AUX_CAPTURE.set(captured)
        try:
            response = create(self, *args, **kwargs)
        finally:
            _ANTHROPIC_AUX_CAPTURE.reset(capture)
        calls = _owned_auxiliary_calls()
        if calls is not None:
            if len(captured) == 1:
                calls.add_anthropic_usage(response, captured[0])
            else:
                calls.mark_unaccounted()
        return response

    observed.nemo_fabric_observed = True  # type: ignore[attr-defined]
    return observed


def _observed_create_anthropic_message(create_message: Callable[..., Any]) -> Any:
    """Wrap ``create_anthropic_message`` to hand each raw usage to the
    Anthropic auxiliary call that made it. Main-loop calls are unchanged."""

    def observed(*args: Any, **kwargs: Any) -> Any:
        message = create_message(*args, **kwargs)
        captured = _ANTHROPIC_AUX_CAPTURE.get()
        if captured is not None:
            captured.append(getattr(message, "usage", None))
        return message

    observed.nemo_fabric_observed = True  # type: ignore[attr-defined]
    return observed


def _install_anthropic_usage_capture() -> None:
    """Route Hermes' Anthropic auxiliary client through the capture wrappers.

    ``_AnthropicCompletionsAdapter.create`` looks up
    ``agent.anthropic_adapter.create_anthropic_message`` when called. If
    Hermes no longer calls it by that name, start fails rather than counting
    the converted usage.
    """
    from agent import anthropic_adapter
    from agent import auxiliary_client

    adapter_type = getattr(auxiliary_client, "_AnthropicCompletionsAdapter", None)
    create = getattr(adapter_type, "create", None)
    create_message = getattr(anthropic_adapter, "create_anthropic_message", None)
    if getattr(create, "nemo_fabric_observed", False):
        return
    called = getattr(getattr(create, "__code__", None), "co_names", ())
    if not (
        callable(create_message)
        and "create_anthropic_message" in called
        and "agent.anthropic_adapter" in called
    ):
        raise lifecycle.LifecycleError(
            "hermes_usage_accounting_unsupported",
            "This Hermes version calls Anthropic auxiliary models outside "
            "agent.anthropic_adapter.create_anthropic_message",
        )
    if not getattr(create_message, "nemo_fabric_observed", False):
        anthropic_adapter.create_anthropic_message = _observed_create_anthropic_message(
            create_message
        )
    adapter_type.create = _observed_anthropic_aux_create(create)


# Hermes' functions that send an auxiliary request and validate its response,
# by the request function each one looks up as a module global.
_AUXILIARY_REQUEST_CALLERS = {
    "_retry_same_provider_sync": "_relay_sync_completion",
    "_retry_same_provider_async": "_relay_async_completion",
    "_call_fallback_candidate_sync": "_relay_sync_completion",
    "_call_fallback_candidate_async": "_relay_async_completion",
    "_call_llm_impl": "_relay_sync_completion",
    "_async_call_llm_impl": "_relay_async_completion",
}


def _observed_unwrap_data_envelope(unwrap: Callable[..., Any]) -> Any:
    """Wrap Hermes' gateway envelope unwrapping so that the unwrapped
    completion, which Hermes records instead, resolves the returned
    response."""

    def observed(response: Any, *args: Any, **kwargs: Any) -> Any:
        unwrapped = unwrap(response, *args, **kwargs)
        calls = _AUXILIARY_CALLS.get()
        if calls is not None and unwrapped is not response:
            calls.replace_unrecorded_response(response, unwrapped)
        return unwrapped

    observed.nemo_fabric_observed = True  # type: ignore[attr-defined]
    return observed


def _global_names(function: Any) -> set[str]:
    """Return the global names a function and its nested functions look up."""
    names: set[str] = set()
    pending = [getattr(function, "__code__", None)]
    while pending:
        code = pending.pop()
        if code is None:
            continue
        names.update(code.co_names)
        pending.extend(
            const for const in code.co_consts if isinstance(const, type(code))
        )
    return names


def _install_request_tracking() -> None:
    """Route Hermes' auxiliary request functions through the request tracker.

    Each auxiliary request goes through ``_relay_sync_completion`` or
    ``_relay_async_completion``, looked up as a module global by the callers
    in ``_AUXILIARY_REQUEST_CALLERS``. ``_validate_llm_response`` records the
    completion that ``_unwrap_data_envelope`` returns, which differs from
    the returned response for gateway envelopes. If Hermes no longer uses
    those names, start fails rather than missing failed requests.
    """
    from agent import auxiliary_client

    if "_unwrap_data_envelope" not in _global_names(
        auxiliary_client._validate_llm_response
    ):
        raise lifecycle.LifecycleError(
            "hermes_usage_accounting_unsupported",
            "This Hermes version unwraps auxiliary responses outside "
            "agent.auxiliary_client._unwrap_data_envelope",
        )

    for caller, request in _AUXILIARY_REQUEST_CALLERS.items():
        if request not in _global_names(getattr(auxiliary_client, caller, None)):
            raise lifecycle.LifecycleError(
                "hermes_usage_accounting_unsupported",
                f"This Hermes version sends auxiliary requests outside "
                f"agent.auxiliary_client.{request}",
            )
    for dispatcher in ("_create_with_progress_once", "_acreate_with_progress"):
        if "_notify_aux_dispatch" not in _global_names(
            getattr(auxiliary_client, dispatcher, None)
        ):
            raise lifecycle.LifecycleError(
                "hermes_usage_accounting_unsupported",
                f"This Hermes version dispatches auxiliary requests in "
                f"agent.auxiliary_client.{dispatcher} without "
                f"_notify_aux_dispatch",
            )
    for name, wrap in (
        ("_relay_sync_completion", _observed_relay_completion),
        ("_relay_async_completion", _observed_async_relay_completion),
        ("_unwrap_data_envelope", _observed_unwrap_data_envelope),
        ("_notify_aux_dispatch", _observed_notify_aux_dispatch),
    ):
        current = getattr(auxiliary_client, name)
        if not getattr(current, "nemo_fabric_observed", False):
            setattr(auxiliary_client, name, wrap(current))


def _observe_background_reviews(agent: Any) -> None:
    """Leave the invocation's counts unknown when its turn starts a review.

    After a turn, Hermes' ``finalize_turn`` calls the agent's
    ``_spawn_background_review``, which forks an agent that runs after the
    invocation returns and records its tokens directly in the session
    database, outside the auxiliary client. If Hermes no longer starts the
    review through that method, start fails rather than missing it.
    """
    from agent import turn_finalizer
    from run_agent import AIAgent

    if not (
        callable(getattr(AIAgent, "_spawn_background_review", None))
        and "_spawn_background_review"
        in _global_names(getattr(turn_finalizer, "finalize_turn", None))
    ):
        raise lifecycle.LifecycleError(
            "hermes_usage_accounting_unsupported",
            "This Hermes version starts background reviews outside "
            "AIAgent._spawn_background_review",
        )
    spawn_review = getattr(agent, "_spawn_background_review", None)
    if spawn_review is None:
        return

    def mark_and_spawn(*args: Any, **kwargs: Any) -> Any:
        calls = _AUXILIARY_CALLS.get()
        if calls is not None:
            calls.mark_unaccounted()
        return spawn_review(*args, **kwargs)

    agent._spawn_background_review = mark_and_spawn


def _install_auxiliary_usage_observer() -> None:
    """Route Hermes' auxiliary usage recorder through the observer.

    Every auxiliary response passes ``_validate_llm_response``, which looks up
    ``agent.aux_accounting.record_aux_usage`` when called and passes it the raw
    response before Hermes normalizes the usage. The wrapper replaces that
    module attribute once per process and always delegates to Hermes'
    function. If Hermes no longer reaches it through that name, start fails
    rather than reporting auxiliary calls as absent.
    """
    from agent import aux_accounting
    from agent import auxiliary_client

    current = aux_accounting.record_aux_usage
    if getattr(current, "nemo_fabric_observed", False):
        return
    called = auxiliary_client._validate_llm_response.__code__.co_names
    if not (
        isinstance(getattr(aux_accounting, "_EXCLUDED_TASKS", None), frozenset)
        and isinstance(
            getattr(aux_accounting, "_accounting", None), contextvars.ContextVar
        )
        and "record_aux_usage" in called
        and "agent.aux_accounting" in called
    ):
        raise lifecycle.LifecycleError(
            "hermes_usage_accounting_unsupported",
            "This Hermes version records auxiliary usage outside "
            "agent.aux_accounting.record_aux_usage",
        )
    _install_anthropic_usage_capture()
    _install_request_tracking()
    aux_accounting.record_aux_usage = _observed_record_aux_usage(current)


def _session_counters(agent: Any) -> dict[str, int]:
    """Return the agent's session token counters that hold a valid count.

    Hermes stamps exactly these attributes into a finished turn's result, but
    its early failure results carry none, so the adapter reads the agent.
    """
    return {
        key: value
        for key in _SESSION_TOKEN_KEYS
        if _is_count(value := getattr(agent, f"session_{key}", None))
    }


def _invocation_usage(
    totals: dict[str, int],
    previous_totals: dict[str, int],
    *,
    skipped: dict[str, int] | None = None,
    unreported: frozenset[str] = frozenset(),
) -> AgentUsage | None:
    """Return this invocation's usage from the agent's session counters.

    Hermes keeps one session across a runtime's invocations, so each counter is
    the change from ``previous_totals``, read before the turn, to ``totals``,
    read after it, plus the ``skipped`` tokens of responses Hermes left out of
    its counters. A counter that is missing from either reading, or that went
    down (Hermes reset its session), is unknown for this invocation, so the
    usage never holds a whole-session total. ``unreported`` names the counts
    ("input", "cached", "output") that a provider response of this invocation
    left out and Hermes counted as 0. Those counts are unknown, and so is the
    total when input or output is; cached input stays known without input. A
    count above what AgentUsage holds is unknown too.
    """
    skipped = skipped or {}
    change = {
        key: value - previous_totals[key] + skipped.get(key, 0)
        for key, value in totals.items()
        if key in previous_totals and value >= previous_totals[key]
    }
    fields: dict[str, Any] = {}
    if "input" not in unreported and all(
        key in change
        for key in ("input_tokens", "cache_read_tokens", "cache_write_tokens")
    ):
        input_tokens = (
            change["input_tokens"]
            + change["cache_read_tokens"]
            + change["cache_write_tokens"]
        )
        if _is_count(input_tokens):
            fields["input_tokens"] = input_tokens
            fields["input_tokens_include_cache"] = True
    # Cache reads are their own count, known even when the input is not.
    if (
        "cached" not in unreported
        and "cache_read_tokens" in change
        and _is_count(change["cache_read_tokens"])
    ):
        fields["cached_input_tokens"] = change["cache_read_tokens"]
    if (
        "output" not in unreported
        and "output_tokens" in change
        and _is_count(change["output_tokens"])
    ):
        fields["output_tokens"] = change["output_tokens"]
    if "input_tokens" in fields and "output_tokens" in fields:
        total_tokens = fields["input_tokens"] + fields["output_tokens"]
        if _is_count(total_tokens):
            fields["total_tokens"] = total_tokens
    return AgentUsage(**fields) if fields else None


def _require_background_review_switch(agent_type: Any) -> None:
    """Fail start when Hermes' agent cannot skip its background review.

    ``filter_supported_kwargs`` drops an unsupported keyword silently, which
    would let the review run and leave invocations without usage.
    """
    parameters = inspect.signature(agent_type.__init__).parameters
    if "skip_background_review" not in parameters:
        raise lifecycle.LifecycleError(
            "hermes_usage_accounting_unsupported",
            "This Hermes version cannot skip its background review",
        )


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
