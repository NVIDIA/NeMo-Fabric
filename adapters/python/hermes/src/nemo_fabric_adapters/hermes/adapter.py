#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermes adapter for Fabric.

This adapter maps Fabric's normalized config into Hermes' native Python SDK
surface and invokes the installed Hermes runtime.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import logging
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
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
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common import instructions as common_instructions
from nemo_fabric_adapters.hermes import sessions
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common import mcp_availability
from nemo_fabric_adapters.hermes import configuration
from nemo_fabric_adapters.hermes import interfaces
from nemo_fabric_adapters.hermes import skills
from nemo_fabric_adapters.hermes import telemetry

# Default agent loop budget when FabricConfig.runtime.max_turns is unset.
# The actual internal hermes default  as of v0.21 is `sys.maxsize`
DEFAULT_MAX_ITERATIONS: int = configuration.DEFAULT_MAX_ITERATIONS
LOGGER = logging.getLogger(__name__)
hermes_mcp_server_config = configuration.hermes_mcp_server_config


def main() -> None:
    """Serve the persistent local-host lifecycle protocol."""

    lifecycle.serve(HermesRuntime, config_loader=AgentConfig.from_mapping)


class HermesRuntime:
    """SDK and native Hermes execution owned by one Fabric runtime."""

    def __init__(self) -> None:
        """Create an unstarted runtime with no Hermes agent."""
        self._started = False
        self._sdk_enabled = True
        self._owns_mcp = False
        self._stopping = False
        self._health_lock = asyncio.Lock()
        self._health_task: asyncio.Task[Any] | None = None
        self._agent_config: AgentConfig | None = None
        self._runtime_id: str | None = None
        self._settings: dict[str, Any] = {}
        self._model_config: AgentModelConfig | None = None
        self._hermes_home: Path | None = None
        self._hermes_config_path: Path | None = None
        self._hermes_config: dict[str, Any] = {}
        self._enabled_toolsets: list[str] | None = None
        self._mcp_authentication_checked = False
        self._mcp_unavailable: list[dict[str, str]] = []
        self._skills_report: dict[str, Any] | None = None
        self._conversation_history: list[dict[str, Any]] | None = None
        self._native_session: sessions.NativeSession | None = None
        self._session_db: Any = None
        self._agent: Any = None
        self._relay_plugin_config: dict[str, Any] | None = None
        self._relay_plugin_config_path: Path | None = None
        self._active_invoke_task: asyncio.Task[tuple[dict[str, Any], str]] | None = None
        self._interfaces = interfaces.InterfaceChildren()

    async def start(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Apply shared configuration and start the selected SDK/native backends.

        SDK execution remains enabled by default. A native-only runtime skips
        SDK agent, database, and MCP initialization; failed startup cleans up
        any resources already created.
        """
        if self._started:
            raise lifecycle.LifecycleError(
                "hermes_runtime_already_started",
                "Hermes runtime is already started",
            )

        self._stopping = False
        # startup_timeout_seconds bounds all of start, so Fabric's start limit
        # can't expire first.
        started_at = asyncio.get_running_loop().time()
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
            interface_ports = interfaces.interface_ports(self._settings)
            self._sdk_enabled = self._settings.get("sdk", {}).get("enabled", True)
            if runtime_context.session_ref is not None and not self._sdk_enabled:
                raise lifecycle.LifecycleError(
                    "session_resume_unsupported",
                    "SessionRef resume requires the Hermes SDK backend",
                )
            if not self._sdk_enabled and not interface_ports:
                raise lifecycle.LifecycleError(
                    "hermes_no_execution_path",
                    "Enable the Hermes SDK or at least one native interface",
                )
            model_config = configuration._selected_model(agent_config)
            self._model_config = model_config
            self._runtime_id = runtime_context.runtime_id
            self._hermes_home = _hermes_home(
                runtime_context, common_utils.base_dir(payload)
            )
            self._hermes_home.mkdir(parents=True, exist_ok=True)
            # An abruptly ended runtime can leave credentials and attachment
            # records in a kept Hermes home.
            token_path = self._hermes_home / "interfaces" / "api.token"
            try:
                for name in ("tui.json", "tui.token", "api.token", "agent-config.json"):
                    token_path.with_name(name).unlink(missing_ok=True)
            except OSError as error:
                raise lifecycle.LifecycleError(
                    "hermes_interface_cleanup_failed",
                    "Hermes interface files could not be removed",
                ) from error
            if interface_ports:
                # Refuse before Hermes's setup, whose MCP discovery can take
                # minutes and outlast the caller's start deadline.
                interfaces.require_supported_platform()
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

            # Check skills.directories and the bundled-skill names where the
            # task's folders are mounted, before anything starts.
            skill_plan = skills.plan_skills(
                agent_config,
                base_dir=common_utils.base_dir(payload),
                hermes_home=self._hermes_home,
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
                interfaces=interface_ports,
                skill_plan=skill_plan,
            )
            if skill_plan.structured:
                # Before the agent or any interface reads its skills.
                self._skills_report = await _prepare_skills(skill_plan)
            api_key_env = configuration._api_key_env(model_config)
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise RuntimeError(f"{api_key_env} is required for Hermes mode")

            if self._sdk_enabled:
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
                        self._owns_mcp = True
                        try:
                            from tools.mcp_tool_discovery import discover_mcp_tools
                        except ModuleNotFoundError as error:
                            if error.name != "tools.mcp_tool_discovery":
                                raise
                            from tools.mcp_tool import discover_mcp_tools

                        await asyncio.to_thread(discover_mcp_tools)
                        await self._refresh_unavailable_mcp_servers()

                    self._enabled_toolsets = configuration.resolve_hermes_toolsets(
                        agent_config, loaded_hermes_config
                    )
                    if (
                        runtime_context.session_ref is not None
                        and not (self._hermes_home / "state.db").is_file()
                    ):
                        raise lifecycle.LifecycleError(
                            "session_not_found",
                            "Native session database was not found in retained storage",
                        )
                    self._session_db = SessionDB()
                    self._native_session = sessions.open_session(
                        self._session_db, runtime_context
                    )
                    self._conversation_history = (
                        self._native_session.history if self._native_session else None
                    )
                    self._agent = AIAgent(
                        **filter_supported_kwargs(
                            AIAgent,
                            base_url=model_config.base_url,
                            api_key=api_key,
                            provider=model_config.provider,
                            model=model_config.model,
                            **configuration.agent_behavior_kwargs(
                                agent_config, self._enabled_toolsets
                            ),
                            quiet_mode=True,
                            skip_context_files=True,
                            skip_memory=True,
                            platform="fabric",
                            session_id=self._native_session.native_id
                            if self._native_session
                            else self._runtime_id,
                            session_db=self._session_db,
                        )
                    )
            if interface_ports:
                if "api" in interface_ports:
                    from hermes_cli.profile_channels import platform_ids

                    # This gateway serves the managed API, not ambient chat bots.
                    # Explicit disables beat Hermes's credential-based enablement.
                    self._hermes_config["platforms"] = {
                        name: {"enabled": False}
                        for name in platform_ids(self._hermes_home)
                        if name != "api_server"
                    }
                    self._hermes_config["gateway"] = {"multiplex_profiles": False}
                    self._hermes_config_path.write_text(
                        common_utils.dump_yaml(self._hermes_config), encoding="utf-8"
                    )
                environment = dict(os.environ)
                if (
                    str(model_config.provider or "").lower() == "openai"
                    and not model_config.base_url
                ):
                    # An ambient endpoint must not redirect the configured key.
                    environment.pop("OPENAI_BASE_URL", None)
                provider_key_env = configuration.PROVIDER_DEFAULT_API_KEY_ENV.get(
                    str(model_config.provider or "").lower()
                )
                if provider_key_env:
                    # Hermes's own processes read a provider's key from its
                    # default variable, not from api_key_env.
                    environment[provider_key_env] = api_key
                await self._interfaces.start(
                    interface_ports,
                    hermes_home=self._hermes_home,
                    token_path=token_path,
                    timeout=max(
                        interfaces.startup_timeout(self._settings)
                        - (asyncio.get_running_loop().time() - started_at),
                        0.0,
                    ),
                    environment=environment,
                    agent_config=agent_config,
                    workspace=str(runtime_context.environment.workspace or "."),
                    terminal_enabled=self._settings.get("interfaces", {})
                    .get("terminal", {})
                    .get("enabled", True)
                    if "terminal" in self._settings.get("interfaces", {})
                    else False,
                )
            self._agent_config = agent_config
            self._started = True
            return self._native_session.start_output() if self._native_session else None
        except BaseException:
            await self.stop()
            raise

    async def invoke(
        self,
        request: AgentRunRequest,
        runtime_context: RuntimeContext,
    ) -> AgentRunResult:
        """Invoke the embedded agent and return its normalized result.

        Native-only runtimes reject SDK invocation without constructing an
        agent. Enabled SDK runtimes retain their conversation between calls.
        """
        if self._started and not self._sdk_enabled:
            return AgentRunResult(
                status=AgentRunStatus.FAILED,
                output=self._skills_output(),
                error=AgentRunError(
                    code="hermes_sdk_disabled",
                    message="Embedded Hermes SDK execution is disabled; use an enabled native interface",
                ),
            )
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

        if self._native_session and self._native_session.recovery_required:
            if not request.acknowledge_recovery:
                return AgentRunResult(
                    status=AgentRunStatus.FAILED,
                    session_id=self._native_session.session_id,
                    output=None,
                    error=AgentRunError(
                        code="session_recovery_required",
                        message="Inspect the restored state and explicitly acknowledge recovery before a new invocation.",
                        retryable=False,
                    ),
                )
            self._native_session.recovery_required = False

        user_message = common_utils.normalize_user_input(request.input)
        instructions = agent_config.instructions
        system_prompt = (
            instructions.system.content
            if instructions and instructions.system
            else None
        )

        await self._authenticate_mcp_servers()

        def run_hermes_turn() -> tuple[dict[str, Any], str]:
            native_session = self._native_session
            completed = False
            if self._relay_plugin_config is not None:
                common_utils.reject_ambient_relay_plugin_config()
            try:
                result = _invoke_hermes_turn(
                    agent=self._agent,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    conversation_history=self._conversation_history,
                    task_id=runtime_context.request_id,
                )
                if native_session:
                    completed = native_session.finish_turn(
                        str(self._agent.session_id), result[0]
                    )
                return result
            finally:
                if native_session:
                    native_session.recovery_required = not completed
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
        if self._native_session:
            self._native_session.begin_turn(runtime_context)
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
        unavailable_mcp_servers = await self._refresh_unavailable_mcp_servers()
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
        # Absent when every configured server is usable, so clean runs keep their shape.
        if mcp := mcp_availability.output_block(unavailable_mcp_servers):
            output["mcp"] = mcp
        output.update(self._skills_output())
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
        return AgentRunResult(
            session_id=self._native_session.session_id
            if self._native_session
            else None,
            status=AgentRunStatus.FAILED if failed else AgentRunStatus.SUCCEEDED,
            output=output,
            error=error,
        )

    def _skills_output(self) -> dict[str, Any]:
        """Return ``{"skills": report}`` for a run result, or nothing.

        Present only with skills.directories or skills.bundled. Each result
        gets its own copy of the report.
        """

        if not self._skills_report:
            return {}
        return {"skills": copy.deepcopy(self._skills_report)}

    async def _refresh_unavailable_mcp_servers(self) -> list[dict[str, str]]:
        """Read Hermes's current status of the configured MCP servers.

        Hermes keeps running when a server fails to connect, so this reports
        its current state: a server that has since connected is no longer
        reported. Hermes records why a connect failed
        only as a message, which is never reported, so every failure is
        ``failed``. Servers that are still connecting, or that Hermes has not
        attempted yet, are not reported.
        """

        configured = self._hermes_config.get("mcp_servers") or {}
        agent_config = self._agent_config
        if not configured or agent_config is None or agent_config.mcp is None:
            self._mcp_unavailable = []
            return []

        try:
            try:
                from tools.mcp_tool_discovery import get_mcp_status
            except ModuleNotFoundError as error:
                if error.name != "tools.mcp_tool_discovery":
                    raise
                from tools.mcp_tool import get_mcp_status

            statuses = await asyncio.to_thread(get_mcp_status, configured)
        except Exception:
            # The report is diagnostic: omit it rather than fail start or a turn.
            LOGGER.warning("Hermes MCP server status could not be observed")
            self._mcp_unavailable = []
            return []
        servers = agent_config.mcp.servers
        unavailable = [
            mcp_availability.unavailable_server(
                name,
                servers[name].transport.strip().lower(),
                mcp_availability.FAILED,
            )
            for status in statuses
            if status.get("status") == "failed"
            and (name := status.get("name")) in servers
        ]
        reported = {entry["name"] for entry in self._mcp_unavailable}
        mcp_availability.warn(
            "Hermes",
            [entry for entry in unavailable if entry["name"] not in reported],
        )
        self._mcp_unavailable = unavailable
        return unavailable

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

    async def check(self, level: str) -> dict[str, Any]:
        """Observe SDK health when enabled, otherwise the native backends."""
        if (
            self._sdk_enabled
            or not self._started
            or self._stopping
            or level not in {"live", "active", "ready"}
        ):
            return await self._check_sdk(level)
        ports = interfaces.interface_ports(self._settings)
        if not ports:
            return await self._check_sdk(level)
        reason = await self._interfaces.check(level)
        if self._stopping or not self._started:
            reason = "observation_changed"
        return {
            "outcome": "passed" if reason is None else "failed",
            "reason_code": reason,
            "activity": "unknown",
            "source": {
                "live": "fabric.adapter.liveness",
                "active": "fabric.adapter.interface",
                "ready": "fabric.adapter.readiness",
            }[level],
            "observed_at": datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
        }

    async def _check_sdk(self, level: str) -> dict[str, Any]:
        """Observe the owned Hermes session through its native diagnostic interface."""
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
        agent, config, model = self._agent, self._agent_config, self._model_config
        runtime_id, database = self._runtime_id, self._session_db
        if (
            not self._started
            or agent is None
            or config is None
            or model is None
            or database is None
            or not runtime_id
        ):
            return {**result, "reason_code": "runtime_unavailable"}
        if self._stopping:
            return {**result, "reason_code": "observation_changed"}

        def unchanged() -> bool:
            return (
                self._started
                and not self._stopping
                and self._agent is agent
                and self._agent_config is config
                and self._model_config is model
                and self._session_db is database
                and self._runtime_id == runtime_id
            )

        def observed(outcome: str, reason: str | None) -> dict[str, Any]:
            if not unchanged():
                return {**result, "reason_code": "observation_changed"}
            if getattr(agent, "_memory_provider_shutdown", False):
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
        if getattr(agent, "_memory_provider_shutdown", False):
            return observed("failed", "session_closed")
        if level != "live" and not callable(agent.run_conversation):
            return observed("failed", "endpoint_unavailable")
        try:
            async with self._health_lock:
                pending = self._health_task
                if pending is not None and not pending.done():
                    await asyncio.wait({pending})
                if not unchanged():
                    return {**result, "reason_code": "observation_changed"}
                if getattr(agent, "_memory_provider_shutdown", False):
                    return observed("failed", "session_closed")
                # Retain one diagnostic worker when a caller's deadline expires.
                task = asyncio.create_task(
                    asyncio.to_thread(agent.get_activity_summary)
                )
                task.add_done_callback(
                    lambda done: None if done.cancelled() else done.exception()
                )
                self._health_task = task
                await asyncio.wait({task})
                summary = task.result()
                self._health_task = None
            if not isinstance(summary, dict) or any(
                type(summary.get(key)) is not int or summary[key] < 0
                for key in (
                    "api_call_count",
                    "budget_used",
                    "budget_max",
                    "max_iterations",
                )
            ):
                raise ValueError("invalid native diagnostic response")
        except Exception:
            if not unchanged():
                return {**result, "reason_code": "observation_changed"}
            if getattr(agent, "_memory_provider_shutdown", False):
                return observed("failed", "session_closed")
            return result
        return observed("passed", None)

    async def stop(self) -> None:
        """Stop owned native processes and close resources created by the SDK.

        Cleanup attempts every resource even after a failure, then reports
        errors without closing parent MCP resources that this runtime never
        initialized.
        """
        self._stopping = True
        active_invoke_task = self._active_invoke_task
        errors: list[BaseException] = []
        try:
            await self._interfaces.stop()
        except BaseException as error:
            errors.append(error)
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
        had_mcp_servers = self._owns_mcp
        self._owns_mcp = False
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
        self._mcp_unavailable = []
        self._skills_report = None
        self._conversation_history = None
        self._native_session = None
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


async def _prepare_skills(plan: skills.SkillPlan) -> dict[str, Any] | None:
    """Copy the selected bundled skills into the home and observe what loads.

    Hermes's own entry points copy its bundled skills into the home at start;
    the embedded agent does not, so the adapter does whenever
    ``skills.bundled`` is set. The copy seeds Hermes's essential skills even
    when no bundled skill is selected, and removes a stale copy that an
    external skill replaces. Returns the ``output.skills`` report and warns
    once.
    """

    sync_result = None
    if plan.sync:
        try:
            sync_result = await asyncio.to_thread(skills.sync_bundled)
        except Exception as error:
            raise lifecycle.LifecycleError(
                "hermes_skill_invalid",
                "Hermes could not copy its bundled skills into its home",
                metadata={"field": "skills.bundled", "reason": "sync_failed"},
            ) from error
    try:
        catalog = await asyncio.to_thread(skills.native_catalog)
    except Exception:
        # The report is diagnostic: keep the checked plan rather than fail start.
        LOGGER.warning("Hermes skill catalog could not be observed", exc_info=True)
        observation = skills.SkillObservation(plan.load or skills.DirectoryLoad())
    else:
        observation = skills.observe(plan, catalog, sync_result=sync_result)
    observation.warn()
    return observation.report()


def _hermes_home(runtime_context: RuntimeContext, base_dir: str) -> Path:
    """Return the agent's state folder when Fabric keeps state across runtimes,
    otherwise this runtime's own folder."""
    if runtime_context.state_dir:
        return Path(runtime_context.state_dir)
    return (
        _artifact_root(runtime_context, base_dir)
        / ".fabric"
        / "hermes"
        / "runtimes"
        / runtime_context.runtime_id
    )


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
        )
        result = agent.run_conversation(
            user_message,
            **conversation_kwargs,
        )
    return result, hermes_stdout.getvalue()


def filter_supported_kwargs(callable_obj: Any, **kwargs: Any) -> dict[str, Any]:
    signature = inspect.signature(callable_obj.__init__)
    supported = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in supported}


def filter_supported_call_kwargs(func: Any, **kwargs: Any) -> dict[str, Any]:
    signature = inspect.signature(func)
    supported = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in supported}


# Selected software reported by the shared lifecycle host; no dependency scan.
PROVENANCE_PACKAGES = (
    ("adapter", "nemo-fabric-adapters-hermes", "nemo_fabric_adapters.hermes.adapter"),
)
PROVENANCE_HARNESS = "hermes"

if __name__ == "__main__":
    main()
