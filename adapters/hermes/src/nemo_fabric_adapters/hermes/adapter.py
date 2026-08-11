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
import json
import logging
import os
import sys
from collections.abc import Mapping
from contextlib import redirect_stdout
from importlib.metadata import version as distribution_version
from io import StringIO
from pathlib import Path
from typing import Any

from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common import mcp_auth
import nemo_fabric_adapters.common.utils as common_utils

# Default agent loop budget when FabricConfig.runtime.max_turns is unset.
# Mirrors Hermes' own AIAgent default (agent/agent_init.py); a lower value such
# as 1 silently starves multi-step tasks (they run out of budget before
# answering while the trial still reports success). See FABRIC-85.
DEFAULT_MAX_ITERATIONS: int = 90
LOGGER = logging.getLogger(__name__)
PROVIDER_DEFAULT_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
# Hermes 0.16+ discovers Relay from this TOML path and falls back to direct
# ATIF/ATOF only when TOML initialization fails. Clear only those enable flags.
HERMES_RELAY_ENV_NAMES = (
    "HERMES_NEMO_RELAY_PLUGINS_TOML",
    "HERMES_NEMO_RELAY_ATIF_ENABLED",
    "HERMES_NEMO_RELAY_ATOF_ENABLED",
)


def finalize_hermes_relay_session(session_id: str) -> None:
    """Finalize one Relay session through the installed Hermes lifecycle API."""
    try:
        from hermes_cli.lifecycle import finalize_session
    except ModuleNotFoundError as error:
        if error.name != "hermes_cli.lifecycle":
            raise
        # Hermes 0.19 exposes the same finalization boundary as a plugin hook.
        from hermes_cli.plugins import invoke_hook

        invoke_hook("on_session_finalize", session_id=session_id, platform="fabric")
    else:
        finalize_session(session_id=session_id, platform="fabric")


def _api_key_env(model_config: dict[str, Any]) -> str:
    explicit = model_config.get("api_key_env")
    if isinstance(explicit, str) and explicit:
        return explicit
    provider = str(model_config.get("provider") or "").lower()
    default = PROVIDER_DEFAULT_API_KEY_ENV.get(provider)
    if default is None:
        raise ValueError(
            f"selected model api_key_env is required for provider {provider!r}"
        )
    return default


def validate_hermes_telemetry_provider(payload: dict[str, Any]) -> None:
    providers = common_utils.telemetry_providers(payload)
    if any(provider != "relay" for provider in providers):
        raise ValueError("only relay telemetry is supported for Hermes")


def disabled_toolsets(payload: dict[str, Any]) -> list[str]:
    return common_utils.blocked_tools(payload)


def build_hermes_config(
    payload: dict[str, Any], *, relay_enabled: bool = False
) -> dict[str, Any]:
    settings = common_utils.settings_payload(payload)
    model_config = common_utils.selected_model_config(payload)
    native = common_utils.capability_plan(payload).get("native") or {}
    environment = common_utils.environment_payload(payload)

    model_name = model_config.get("model", "")
    provider = model_config.get("provider")
    base_url = common_utils.get_base_url(model_config)
    blocked_toolsets = disabled_toolsets(payload)
    enabled_toolsets = common_utils.enabled_tools(payload)

    config: dict[str, Any] = {
        "model": common_utils.without_none(
            {
                "provider": provider,
                "default": model_name,
                "base_url": base_url,
            }
        ),
        "agent": common_utils.without_none(
            {
                "max_turns": common_utils.max_turns(payload),
                "disabled_toolsets": blocked_toolsets or None,
            }
        ),
        "terminal": common_utils.without_none(
            {
                "backend": "local",
                "cwd": str(environment.get("workspace") or "."),
                "timeout": settings.get("terminal_timeout", 60),
            }
        ),
    }

    skill_dirs = [str(path) for path in native.get("skill_paths", [])]
    if skill_dirs:
        config["skills"] = {"external_dirs": skill_dirs}

    mcp_servers = native.get("mcp_servers") or {}
    if mcp_servers:
        config["mcp_servers"] = {
            name: hermes_mcp_server_config(server, name=name)
            for name, server in sorted(mcp_servers.items())
        }

    if enabled_toolsets is not None:
        config["platform_toolsets"] = {"cli": enabled_toolsets}

    plugins = common_utils.normalize_list(settings.get("plugins_enabled"))
    if relay_enabled and "observability/nemo_relay" not in plugins:
        plugins.append("observability/nemo_relay")
    if plugins:
        config["plugins"] = {"enabled": plugins}

    return config


def write_hermes_config(
    payload: dict[str, Any],
    hermes_home: Path,
    *,
    relay_enabled: bool = False,
) -> tuple[Path, dict[str, Any]]:
    hermes_home.mkdir(parents=True, exist_ok=True)
    config = build_hermes_config(payload, relay_enabled=relay_enabled)
    config_path = hermes_home / "config.yaml"
    config_path.write_text(common_utils.dump_yaml(config), encoding="utf-8")
    return config_path, config


def write_hermes_relay_plugin_config(
    payload: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Stage Fabric's resolved Relay config for Hermes' bundled integration."""

    plugin_config = common_utils.load_relay_plugin_config(payload)
    hermes_plugin_config = copy.deepcopy(plugin_config)
    relay_version = distribution_version("nemo-relay")
    try:
        relay_major, relay_minor = (
            int(part) for part in relay_version.split(".", maxsplit=2)[:2]
        )
    except ValueError as error:
        raise RuntimeError(
            f"unsupported NeMo Relay version {relay_version!r}"
        ) from error
    observability_version = 3 if (relay_major, relay_minor) >= (0, 7) else 2
    for component in hermes_plugin_config.get("components", []):
        if component.get("kind") != "observability":
            continue
        observability = component.get("config")
        if not isinstance(observability, dict):
            continue

        if observability_version == 3 and observability.get("version") != 3:
            # Relay 0.7 combines Fabric's legacy OTLP and OpenInference exporter
            # settings into typed OpenTelemetry endpoints in its v3 schema.
            endpoints = []
            for config_name, endpoint_type in (
                ("opentelemetry", "full"),
                ("openinference", "openinference"),
            ):
                exporter = observability.pop(config_name, None)
                if not isinstance(exporter, dict) or not exporter.get("enabled"):
                    continue
                endpoint = {
                    key: value
                    for key, value in exporter.items()
                    if key != "enabled" and value is not None
                }
                endpoint["type"] = endpoint_type
                endpoints.append(endpoint)
            if endpoints:
                observability["opentelemetry"] = {
                    "enabled": True,
                    "endpoints": endpoints,
                }
            observability["version"] = 3

        # Fabric finalizes Hermes' Relay session after every invocation. Each
        # finalization reinitializes Relay for the next turn, so a file sink
        # cannot overwrite the runtime-scoped artifact it created previously.
        for sink in (observability.get("atof") or {}).get("sinks") or []:
            if isinstance(sink, dict) and sink.get("type") == "file":
                if sink.get("mode") == "overwrite":
                    sink["mode"] = "append"
    _, plugin_config_path = common_utils.write_relay_configs(
        plugin_config=hermes_plugin_config,
        observability_version=observability_version,
    )
    if plugin_config_path is None:
        raise RuntimeError("Hermes Relay plugin configuration was not generated")
    return plugin_config_path, plugin_config


def hermes_mcp_server_config(
    server: dict[str, Any], *, name: str = "configured"
) -> dict[str, Any]:
    transport = str(server.get("transport") or "").strip().lower()
    raw_target = server.get("url")
    target = os.path.expandvars(str(raw_target or "")).strip()
    if not target:
        raise ValueError("MCP server mapping requires a URL")

    if transport == "stdio":
        return common_utils.without_none(
            {
                "enabled": True,
                "command": target,
                "args": common_utils.normalize_list(server.get("args")) or None,
                "env": server.get("env"),
            }
        )

    result: dict[str, Any] = {
        "enabled": True,
        "url": target,
        "transport": transport,
    }
    if headers := server.get("custom_headers"):
        try:
            result["headers"] = mcp_auth.normalize_custom_headers(name, headers)
        except mcp_auth.McpAuthConfigError as error:
            raise ValueError(str(error)) from error
    if authentication := server.get("authentication"):
        raw_authentication = authentication
        try:
            if (
                isinstance(authentication, Mapping)
                and authentication.get("type") == "service_account"
            ):
                raise mcp_auth.McpAuthConfigError(
                    f"MCP server {name!r} service_account authentication is not supported by Hermes"
                )
            authentication = mcp_auth.parse_oauth2_config(name, authentication)
        except mcp_auth.McpAuthConfigError as error:
            raise ValueError(str(error)) from error
        if authentication.client_name:
            raise ValueError(
                f"MCP server {name!r} authentication.client_name is not supported by Hermes"
            )
        if authentication.token_endpoint_auth_method:
            raise ValueError(
                f"MCP server {name!r} authentication.token_endpoint_auth_method is not supported by Hermes"
            )
        if "authorization_timeout_seconds" in raw_authentication:
            raise ValueError(
                f"MCP server {name!r} authentication.authorization_timeout_seconds is not supported by Hermes"
            )
        oauth = common_utils.without_none(
            {
                "client_id": authentication.client_id,
                "scope": authentication.scope,
                "redirect_uri": authentication.redirect_uri,
            }
        )
        if secret_env := authentication.client_secret_env:
            try:
                mcp_auth.resolve_client_secret(name, authentication)
            except mcp_auth.McpAuthConfigError as error:
                raise ValueError(str(error)) from error
            oauth["client_secret"] = f"${{{secret_env}}}"
        result["auth"] = "oauth"
        result["oauth"] = oauth
    return result


def summarize_hermes_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": config.get("model", {}),
        "terminal": config.get("terminal", {}),
        "skill_dirs": (config.get("skills") or {}).get("external_dirs", []),
        "mcp_servers": sorted((config.get("mcp_servers") or {}).keys()),
        "plugins": (config.get("plugins") or {}).get("enabled", []),
        "platform_toolsets": config.get("platform_toolsets", {}),
        "disabled_toolsets": (config.get("agent") or {}).get("disabled_toolsets", []),
    }


def main() -> None:
    """Serve the persistent local-host lifecycle protocol."""

    lifecycle.serve(HermesRuntime)


def resolve_hermes_toolsets(
    payload: dict[str, Any], config: dict[str, Any]
) -> list[str] | None:
    enabled = common_utils.enabled_tools(payload)
    if enabled is not None:
        return enabled

    from hermes_cli.tools_config import _get_platform_tools

    return sorted(_get_platform_tools(config, "cli"))


def _artifact_root(payload: dict[str, Any]) -> Path:
    artifacts = common_utils.runtime_context(payload).get("artifacts") or {}
    root = artifacts.get("root") if isinstance(artifacts, dict) else None
    if root:
        artifact_root = Path(str(root))
        if not artifact_root.is_absolute():
            artifact_root = Path(common_utils.base_dir(payload)) / artifact_root
        return artifact_root.resolve()
    return Path(common_utils.base_dir(payload)).resolve() / "artifacts"


class HermesRuntime:
    """One Hermes agent and session database owned by a Fabric runtime."""

    def __init__(self) -> None:
        self._started = False
        self._start_payload: dict[str, Any] | None = None
        self._runtime_id: str | None = None
        self._settings: dict[str, Any] = {}
        self._model_config: dict[str, Any] = {}
        self._base_url: str | None = None
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

    async def start(self, payload: dict[str, Any]) -> None:
        if self._started:
            raise lifecycle.LifecycleError(
                "hermes_runtime_already_started",
                "Hermes runtime is already started",
            )

        try:
            validate_hermes_telemetry_provider(payload)
            self._settings = common_utils.settings_payload(payload)
            self._model_config = common_utils.selected_model_config(payload)
            self._runtime_id = common_utils.runtime_id(payload)
            self._hermes_home = common_utils.runtime_state_directory(
                _artifact_root(payload) / ".fabric" / "hermes", payload
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

            relay_enabled = common_utils.relay_enabled(payload)
            if relay_enabled:
                (
                    self._relay_plugin_config_path,
                    self._relay_plugin_config,
                ) = write_hermes_relay_plugin_config(payload)
                for name in HERMES_RELAY_ENV_NAMES:
                    os.environ.pop(name, None)
                os.environ["HERMES_NEMO_RELAY_PLUGINS_TOML"] = str(
                    self._relay_plugin_config_path
                )

            self._hermes_config_path, self._hermes_config = write_hermes_config(
                payload,
                self._hermes_home,
                relay_enabled=relay_enabled,
            )
            api_key_env = _api_key_env(self._model_config)
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise RuntimeError(f"{api_key_env} is required for Hermes mode")
            self._base_url = common_utils.get_base_url(self._model_config)

            from hermes_cli.config import load_config
            from hermes_cli.plugins import discover_plugins
            from hermes_state import SessionDB
            from run_agent import AIAgent

            with redirect_stdout(StringIO()):
                discover_plugins(force=True)
                loaded_hermes_config = load_config()
                # Hermes 0.12+ no longer discovers MCP tools as an import side effect
                # (#16856). Fabric is a Hermes host: discover after config.yaml exists
                # and before AIAgent resolves mcp-* toolsets.
                # discover_mcp_tools uses a blocking 120s wait, wrapping it in
                # asyncio.to_thread to avoid blocking the loop.
                if self._hermes_config.get("mcp_servers"):
                    from tools.mcp_tool import discover_mcp_tools

                    await asyncio.to_thread(discover_mcp_tools)

                self._enabled_toolsets = resolve_hermes_toolsets(
                    payload, loaded_hermes_config
                )
                self._session_db = SessionDB()
                self._conversation_history = None
                max_iterations = common_utils.max_turns(payload)
                if max_iterations is None:
                    max_iterations = DEFAULT_MAX_ITERATIONS
                temperature = self._model_config.get("temperature")
                self._agent = AIAgent(
                    **filter_supported_kwargs(
                        AIAgent,
                        base_url=self._base_url,
                        api_key=api_key,
                        provider=self._model_config.get("provider"),
                        model=self._model_config.get("model", ""),
                        max_iterations=int(max_iterations),
                        enabled_toolsets=self._enabled_toolsets,
                        disabled_toolsets=disabled_toolsets(payload) or None,
                        quiet_mode=True,
                        skip_context_files=True,
                        skip_memory=True,
                        save_trajectories=bool(
                            self._settings.get("save_trajectories", False)
                        ),
                        max_tokens=self._settings.get("max_tokens", 512),
                        request_overrides=(
                            {"temperature": temperature}
                            if temperature is not None
                            else None
                        ),
                        reasoning_config=self._settings.get(
                            "reasoning_config", {"effort": "none"}
                        ),
                        platform="fabric",
                        session_id=self._runtime_id,
                        session_db=self._session_db,
                    )
                )
            self._start_payload = payload
            self._started = True
        except BaseException:
            await self.stop()
            raise

    async def invoke(self, invocation: dict[str, Any]) -> dict[str, Any]:
        start_payload = self._start_payload
        if not self._started or self._agent is None or start_payload is None:
            raise lifecycle.LifecycleError(
                "hermes_runtime_not_started",
                "Hermes runtime is not started",
            )
        if common_utils.runtime_id(invocation) != self._runtime_id:
            raise lifecycle.LifecycleError(
                "hermes_runtime_mismatch",
                "Hermes invocation does not match the active runtime",
            )

        payload = {
            **start_payload,
            "runtime_context": invocation.get("runtime_context"),
            "request": invocation.get("request"),
        }
        request = common_utils.request_payload(payload)
        user_message = request.get("input") or ""
        if not isinstance(user_message, str):
            user_message = json.dumps(user_message, sort_keys=True)

        await self._authenticate_mcp_servers()

        def run_hermes_turn() -> tuple[dict[str, Any], str]:
            try:
                return _invoke_hermes_turn(
                    agent=self._agent,
                    system_prompt=common_utils.system_instruction(start_payload),
                    user_message=user_message,
                    conversation_history=self._conversation_history,
                    task_id=(
                        request["request_id"]
                        if isinstance(request.get("request_id"), str)
                        else None
                    ),
                )
            finally:
                if self._relay_plugin_config is not None:
                    # Hermes writes TOML-configured ATIF at its session-finalization
                    # boundary for every supported Relay version. Fabric defines
                    # each invoke as an artifact-complete boundary, so finalize
                    # through Hermes' lifecycle instead of reaching into Relay
                    # directly.
                    finalize_hermes_relay_session(str(self._agent.session_id))

        # Hermes' upstream Relay integration drives async Relay hooks from its
        # synchronous agent loop. Run that loop outside this lifecycle server's
        # event-loop thread so Hermes can own its Relay event loop.
        if self._active_invoke_task is not None:
            raise lifecycle.LifecycleError(
                "hermes_invocation_in_progress",
                "Hermes runtime already has an active invocation",
            )
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

        output = {
            "harness": "hermes",
            "adapter": "python",
            "mode": "hermes",
            "model": self._model_config.get("model"),
            "base_url": self._base_url,
            "response": result.get("response") or result.get("final_response"),
            "completed": bool(result.get("completed")),
            "failed": bool(result.get("failed")),
            "api_calls": result.get("api_calls"),
            "messages": messages,
            "message_count": len(messages),
            "error": result.get("error"),
            "adapter_stdout": adapter_stdout,
            "hermes_home": str(self._hermes_home),
            "hermes_config_path": str(self._hermes_config_path),
            "hermes_native_config": summarize_hermes_config(self._hermes_config),
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
        return output

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
            status["name"]: status
            for status in await asyncio.to_thread(get_mcp_status)
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
        had_mcp_servers = bool(self._hermes_config.get("mcp_servers"))
        had_relay_plugin = self._relay_plugin_config_path is not None
        self._agent = None
        self._session_db = None
        self._start_payload = None
        self._runtime_id = None
        self._settings = {}
        self._model_config = {}
        self._base_url = None
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
            for name in HERMES_RELAY_ENV_NAMES:
                os.environ.pop(name, None)

        if had_mcp_servers:
            try:
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
