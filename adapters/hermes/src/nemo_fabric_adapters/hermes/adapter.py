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
import json
import logging
import os
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common import relay_gateway
import nemo_fabric_adapters.common.utils as common_utils
from nemo_fabric_adapters.hermes import relay_cli

# Default agent loop budget when harness.settings.max_iterations is unset.
# Mirrors Hermes' own AIAgent default (agent/agent_init.py); a lower value such
# as 1 silently starves multi-step tasks (they run out of budget before
# answering while the trial still reports success). See FABRIC-85.
DEFAULT_MAX_ITERATIONS: int = 90
LOGGER = logging.getLogger(__name__)
NATIVE_RELAY_PLUGIN = "observability/nemo_relay"


def _fabric_stream_sink_enabled(config: dict[str, Any] | None) -> bool:
    if config is None:
        return False
    for component in config.get("components") or []:
        if not isinstance(component, dict) or component.get("kind") != "observability":
            continue
        component_config = component.get("config")
        if not isinstance(component_config, dict):
            continue
        atof = component_config.get("atof")
        if not isinstance(atof, dict):
            continue
        if any(
            isinstance(sink, dict) and sink.get("name") == "nemo-fabric-stream"
            for sink in atof.get("sinks") or []
        ):
            return True
    return False


def validate_hermes_telemetry_provider(payload: dict[str, Any]) -> None:
    providers = common_utils.telemetry_providers(payload)
    if any(provider != "relay" for provider in providers):
        raise ValueError("only relay telemetry is supported for Hermes")


def validate_hermes_relay_plugin_mode(payload: dict[str, Any]) -> None:
    """Reject the native Relay plugin outside Fabric-managed Relay telemetry."""

    plugins = common_utils.normalize_list(
        common_utils.settings_payload(payload).get("plugins_enabled")
    )
    if not common_utils.relay_enabled(payload) and NATIVE_RELAY_PLUGIN in plugins:
        raise ValueError(
            "observability/nemo_relay cannot be enabled directly; "
            "enable Fabric Relay telemetry instead"
        )


def disabled_toolsets(payload: dict[str, Any]) -> list[str]:
    settings = common_utils.settings_payload(payload)
    return common_utils.merge_unique(
        common_utils.blocked_tools(payload),
        settings.get("disabled_toolsets"),
    )


def hermes_working_directory(payload: dict[str, Any]) -> Path:
    """Resolve the normalized Hermes workspace relative to the Fabric root."""

    base_dir = payload.get("base_dir")
    root = Path(base_dir).resolve() if isinstance(base_dir, str) and base_dir else Path()
    environment = common_utils.environment_payload(payload)
    settings = common_utils.settings_payload(payload)
    workspace = environment.get("workspace") or settings.get("workspace")
    path = Path(workspace) if workspace else root
    return path if path.is_absolute() else root / path


def build_hermes_config(
    payload: dict[str, Any],
    *,
    relay_enabled: bool = False,
    exclude_relay_plugin: bool = False,
) -> dict[str, Any]:
    settings = common_utils.settings_payload(payload)
    model_config = common_utils.selected_model_config(payload)
    native = common_utils.capability_plan(payload).get("native") or {}

    model_name = settings.get("model_name") or model_config.get("model", "")
    provider = settings.get("provider") or model_config.get("provider")
    base_url = common_utils.get_base_url(settings, model_config)
    blocked_toolsets = disabled_toolsets(payload)

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
                "max_turns": settings.get("max_iterations"),
                "disabled_toolsets": blocked_toolsets or None,
            }
        ),
        "terminal": common_utils.without_none(
            {
                "backend": settings.get("terminal_backend", "local"),
                "cwd": str(hermes_working_directory(payload)),
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
            name: hermes_mcp_server_config(server)
            for name, server in sorted(mcp_servers.items())
        }

    if "enabled_toolsets" in settings:
        config["platform_toolsets"] = {
            settings.get("toolset_platform", "cli"): common_utils.normalize_list(
                settings.get("enabled_toolsets")
            )
        }

    plugins = common_utils.normalize_list(settings.get("plugins_enabled"))
    if exclude_relay_plugin:
        plugins = [plugin for plugin in plugins if plugin != NATIVE_RELAY_PLUGIN]
    if relay_enabled and NATIVE_RELAY_PLUGIN not in plugins:
        plugins.append(NATIVE_RELAY_PLUGIN)
    if plugins:
        config["plugins"] = {"enabled": plugins}

    return config


def write_hermes_config(
    payload: dict[str, Any],
    hermes_home: Path,
    *,
    relay_enabled: bool = False,
    exclude_relay_plugin: bool = False,
) -> tuple[Path, dict[str, Any]]:
    hermes_home.mkdir(parents=True, exist_ok=True)
    config = build_hermes_config(
        payload,
        relay_enabled=relay_enabled,
        exclude_relay_plugin=exclude_relay_plugin,
    )
    config_path = hermes_home / "config.yaml"
    config_path.write_text(common_utils.dump_yaml(config), encoding="utf-8")
    return config_path, config


def hermes_mcp_server_config(server: dict[str, Any]) -> dict[str, Any]:
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

    return {"enabled": True, "url": target, "transport": transport}


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
    settings: dict[str, Any], config: dict[str, Any]
) -> list[str] | None:
    if "enabled_toolsets" in settings:
        return common_utils.normalize_list(settings.get("enabled_toolsets"))

    from hermes_cli.tools_config import _get_platform_tools

    platform = settings.get("toolset_platform", "cli")
    return sorted(_get_platform_tools(config, platform))


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
        self._conversation_history: list[dict[str, Any]] | None = None
        self._session_db: Any = None
        self._agent: Any = None
        self._invoke_hook: Any = None
        self._relay_plugin_config: dict[str, Any] | None = None
        self._relay_session_pending = False
        self._relay_finalize_hook_invoked = False
        self._relay_model_name = "unknown"
        self._relay_runner: relay_cli.HermesRelayRunner | None = None
        self._hermes_cli_version: tuple[int, int, int] | None = None

    async def start(self, payload: dict[str, Any]) -> None:
        if self._started:
            raise lifecycle.LifecycleError(
                "hermes_runtime_already_started",
                "Hermes runtime is already started",
            )

        try:
            validate_hermes_telemetry_provider(payload)
            validate_hermes_relay_plugin_mode(payload)
            self._settings = common_utils.settings_payload(payload)
            self._model_config = common_utils.selected_model_config(payload)
            self._runtime_id = common_utils.runtime_id(payload)
            hermes_home_base = Path(common_utils.base_dir(payload)).joinpath(
                self._settings.get("hermes_home", "./artifacts/hermes-home")
            )
            self._hermes_home = common_utils.runtime_state_directory(
                hermes_home_base, payload
            )
            self._hermes_home.mkdir(parents=True, exist_ok=True)

            relay_enabled = common_utils.relay_enabled(payload)
            if relay_enabled:
                self._relay_plugin_config = common_utils.load_relay_plugin_config(
                    payload
                )

            self._hermes_config_path, self._hermes_config = write_hermes_config(
                payload,
                self._hermes_home,
                relay_enabled=False,
                exclude_relay_plugin=relay_enabled,
            )
            api_key_env = (
                self._settings.get("api_key_env")
                or self._model_config.get("api_key_env")
                or "NVIDIA_API_KEY"
            )
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise RuntimeError(f"{api_key_env} is required for Hermes mode")
            self._base_url = common_utils.get_base_url(
                self._settings, self._model_config
            )
            if relay_enabled:
                await self._start_relay_runtime(payload)
                self._start_payload = payload
                self._started = True
                return

            os.environ["HOME"] = str(self._hermes_home)
            os.environ["HERMES_HOME"] = str(self._hermes_home)
            os.environ.setdefault("HERMES_YOLO_MODE", "1")
            os.environ.setdefault("HERMES_ACCEPT_HOOKS", "1")
            os.environ["HERMES_SESSION_SOURCE"] = "fabric"
            os.environ.setdefault(
                "TERMINAL_ENV",
                self._settings.get("terminal_backend", "local"),
            )
            os.environ.setdefault(
                "TERMINAL_TIMEOUT",
                str(self._settings.get("terminal_timeout", 60)),
            )

            from hermes_cli.config import load_config
            from hermes_cli.plugins import discover_plugins
            from hermes_cli.plugins import invoke_hook
            from hermes_state import SessionDB
            from run_agent import AIAgent

            with redirect_stdout(StringIO()):
                discover_plugins(force=True)
                loaded_hermes_config = load_config()
                self._enabled_toolsets = resolve_hermes_toolsets(
                    self._settings, loaded_hermes_config
                )
                self._session_db = SessionDB()
                self._conversation_history = None
                max_iterations = self._settings.get("max_iterations")
                if max_iterations is None:
                    max_iterations = DEFAULT_MAX_ITERATIONS
                self._agent = AIAgent(
                    **filter_supported_kwargs(
                        AIAgent,
                        base_url=self._base_url,
                        api_key=api_key,
                        provider=self._settings.get("provider")
                        or self._model_config.get("provider"),
                        model=self._settings.get("model_name")
                        or self._model_config.get("model", ""),
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
                        temperature=self._settings.get(
                            "temperature",
                            self._model_config.get("temperature", 0.0),
                        ),
                        reasoning_config=self._settings.get(
                            "reasoning_config", {"effort": "none"}
                        ),
                        insert_reasoning=bool(
                            self._settings.get("insert_reasoning", False)
                        ),
                        platform="fabric",
                        session_id=self._runtime_id,
                        session_db=self._session_db,
                    )
                )
            self._invoke_hook = invoke_hook
            self._relay_model_name = common_utils.relay_model_name(payload)
            self._start_payload = payload
            self._started = True
        except BaseException:
            await self.stop()
            raise

    async def _start_relay_runtime(self, payload: dict[str, Any]) -> None:
        if (
            self._hermes_home is None
            or self._hermes_config_path is None
            or self._runtime_id is None
            or self._relay_plugin_config is None
        ):
            raise RuntimeError("Hermes Relay runtime state is incomplete")
        root = Path(common_utils.base_dir(payload)).resolve()
        relay_command = (
            self._settings.get("nemo_relay_command")
            or self._settings.get("relay_cli_command")
            or "nemo-relay"
        )
        hermes_command = (
            self._settings.get("hermes_command")
            or self._settings.get("command")
            or "hermes"
        )
        relay_executable = relay_cli.resolve_executable(
            root, relay_command, name="NeMo Relay CLI"
        )
        hermes_executable = relay_cli.resolve_executable(
            root, hermes_command, name="Hermes CLI"
        )
        relay_contract, hermes_version = await asyncio.gather(
            asyncio.to_thread(relay_gateway.relay_cli_contract, relay_executable),
            asyncio.to_thread(relay_cli.hermes_cli_version, hermes_executable),
        )
        self._hermes_cli_version = hermes_version
        base_url = relay_cli.validate_openai_upstream(
            self._settings, self._model_config, self._base_url
        )
        model = str(
            self._settings.get("model_name") or self._model_config.get("model") or ""
        )
        if not model:
            raise relay_cli.HermesRelayError("Hermes Relay execution requires a model")
        cwd = hermes_working_directory(payload)
        env = relay_cli.child_environment(
            self._settings,
            self._model_config,
            self._relay_plugin_config,
            self._hermes_home,
        )
        await asyncio.to_thread(
            _ensure_hermes_runtime_session,
            self._hermes_home,
            self._runtime_id,
            model,
            self._model_config,
            str(cwd.resolve()),
        )
        self._relay_runner = relay_cli.HermesRelayRunner(
            relay_cli.HermesRelayLaunch(
                relay_executable=relay_executable,
                relay_contract=relay_contract,
                hermes_executable=hermes_executable,
                hermes_config_path=self._hermes_config_path,
                hermes_home=self._hermes_home,
                cwd=cwd.resolve(),
                env=env,
                base_url=base_url,
                model=model,
                runtime_id=self._runtime_id,
                settings=self._settings,
                model_config=self._model_config,
                plugin_config=self._relay_plugin_config,
            )
        )

    async def invoke(self, invocation: dict[str, Any]) -> dict[str, Any]:
        start_payload = self._start_payload
        if (
            not self._started
            or start_payload is None
            or (self._agent is None and self._relay_runner is None)
        ):
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
        if self._relay_runner is not None:
            return await self._invoke_relay(payload, user_message)

        def invoke_turn() -> tuple[dict[str, Any], str]:
            return _invoke_hermes_turn(
                agent=self._agent,
                settings=self._settings,
                user_message=user_message,
                conversation_history=self._conversation_history,
            )

        self._relay_session_pending = self._relay_plugin_config is not None
        self._relay_finalize_hook_invoked = False
        if _fabric_stream_sink_enabled(self._relay_plugin_config):
            from nemo_relay import ScopeType, scope

            with scope.scope(
                "nemo-fabric-invocation",
                ScopeType.Agent,
                metadata={
                    "nemo_fabric_request_id": request.get("request_id"),
                },
            ):
                try:
                    result, adapter_stdout = invoke_turn()
                finally:
                    # The Hermes plugin pushes its session below this correlation
                    # scope, so finalize that session before popping the parent.
                    self._finalize_relay_session()
        else:
            try:
                result, adapter_stdout = invoke_turn()
            finally:
                # Hermes' Relay plugin materializes ATIF when its session-finalize
                # hook runs. Finalize the telemetry session for each Fabric
                # invocation while retaining the native AIAgent and SessionDB.
                self._finalize_relay_session()
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
                "emitter": "hermes.observability/nemo_relay",
            }
            output["relay_artifacts"] = common_utils.collect_relay_artifacts(
                self._relay_plugin_config
            )
        return output

    async def _invoke_relay(
        self, payload: dict[str, Any], user_message: str
    ) -> dict[str, Any]:
        if self._relay_runner is None:
            raise RuntimeError("Hermes Relay runtime is not initialized")
        context = common_utils.runtime_context(payload)
        invocation_id = str(
            context.get("invocation_id")
            or context.get("request_id")
            or f"{self._runtime_id}-invocation"
        )
        result = await self._relay_runner.invoke(user_message, invocation_id)
        response = relay_cli.quiet_response(result.stdout)
        failed = result.returncode != 0
        error = None
        if failed:
            error = (
                "NeMo Relay/Hermes exited with status "
                f"{result.returncode}; see {result.stderr_path}"
            )
        return {
            "harness": "hermes",
            "adapter": "cli",
            "mode": "hermes_relay_gateway",
            "model": self._model_config.get("model"),
            "base_url": self._base_url,
            "response": response,
            "completed": not failed or bool(response),
            "failed": failed,
            "api_calls": None,
            "messages": [],
            "message_count": 0,
            "error": error,
            "adapter_stdout": result.stdout,
            "hermes_home": str(self._hermes_home),
            "hermes_config_path": str(self._hermes_config_path),
            "hermes_native_config": summarize_hermes_config(self._hermes_config),
            "enabled_toolsets": common_utils.normalize_list(
                self._settings.get("enabled_toolsets")
            ),
            "command": result.command,
            "returncode": result.returncode,
            "stdout_path": str(result.stdout_path),
            "stderr_path": str(result.stderr_path),
            "output_truncated": result.truncated,
            "relay_runtime": {
                "enabled": True,
                "config_path": str(result.config_path),
                "plugin_config_path": str(result.plugin_config_path),
                "emitter": "nemo-relay.cli-gateway",
                "model_event_source": "gateway",
                "hook_event_policy": "lifecycle_context_only",
                "relay_version": ".".join(
                    str(value) for value in self._relay_runner.relay_version
                ),
                "hermes_version": ".".join(
                    str(value) for value in self._hermes_cli_version or ()
                ),
            },
            "relay_artifacts": common_utils.collect_relay_artifacts(
                result.plugin_config
            ),
        }

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
        from nemo_relay import subscribers

        subscribers.flush()
        self._relay_session_pending = False
        self._relay_finalize_hook_invoked = False

    async def stop(self) -> None:
        agent = self._agent
        session_db = self._session_db
        relay_runner = self._relay_runner
        errors: list[BaseException] = []
        if self._relay_plugin_config is not None and agent is not None:
            try:
                self._finalize_relay_session()
            except BaseException as error:
                errors.append(error)
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
        self._conversation_history = None
        self._invoke_hook = None
        self._relay_plugin_config = None
        self._relay_session_pending = False
        self._relay_finalize_hook_invoked = False
        self._relay_model_name = "unknown"
        self._relay_runner = None
        self._hermes_cli_version = None
        self._started = False

        if relay_runner is not None:
            try:
                await relay_runner.stop()
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
    settings: dict[str, Any],
    user_message: str,
    conversation_history: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], str]:
    hermes_stdout = StringIO()
    with redirect_stdout(hermes_stdout):
        conversation_kwargs = filter_supported_call_kwargs(
            agent.run_conversation,
            system_message=settings.get("system_prompt"),
            conversation_history=conversation_history,
            sync_honcho=False,
            dont_review=True,
        )
        result = agent.run_conversation(
            user_message,
            **conversation_kwargs,
        )
    return result, hermes_stdout.getvalue()


def _ensure_hermes_runtime_session(
    hermes_home: Path,
    runtime_id: str,
    model: str,
    model_config: dict[str, Any],
    cwd: str,
) -> None:
    """Create the stable session that ``hermes chat --continue`` resumes."""

    from hermes_state import SessionDB

    session_db = SessionDB(db_path=hermes_home / "state.db")
    try:
        session_db.create_session(
            runtime_id,
            "fabric",
            model=model,
            model_config=model_config,
            cwd=cwd,
        )
    finally:
        session_db.close()


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
