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

from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentMcpServerConfig
from nemo_fabric_adapter_contract.models import AgentModelConfig
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common import lifecycle
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


def _settings(config: AgentConfig) -> dict[str, Any]:
    return config.harness.settings if config.harness is not None else {}


def _selected_model(config: AgentConfig) -> AgentModelConfig:
    model = config.models.get("default")
    if model is None and len(config.models) == 1:
        model = next(iter(config.models.values()))
    if model is None:
        raise ValueError("Hermes requires a default model or exactly one model")
    return model


def _max_turns(config: AgentConfig) -> int | None:
    return config.runtime.max_turns if config.runtime is not None else None


def _api_key_env(model_config: AgentModelConfig) -> str:
    explicit = model_config.api_key_env
    if isinstance(explicit, str) and explicit:
        return explicit
    provider = str(model_config.provider or "").lower()
    default = PROVIDER_DEFAULT_API_KEY_ENV.get(provider)
    if default is None:
        raise ValueError(
            f"selected model api_key_env is required for provider {provider!r}"
        )
    return default


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


def validate_hermes_telemetry_provider(runtime_context: RuntimeContext) -> None:
    telemetry = runtime_context.telemetry
    providers = telemetry.metadata.get("telemetry_providers", []) if telemetry else []
    if any(provider != "relay" for provider in providers):
        raise ValueError("only relay telemetry is supported for Hermes")


def disabled_toolsets(config: AgentConfig) -> list[str]:
    return config.tools.blocked if config.tools is not None else []


def build_hermes_config(
    agent_config: AgentConfig,
    *,
    workspace: str,
    relay_enabled: bool = False,
) -> dict[str, Any]:
    settings = _settings(agent_config)
    model_config = _selected_model(agent_config)
    blocked_toolsets = disabled_toolsets(agent_config)
    enabled_toolsets = (
        agent_config.tools.enabled if agent_config.tools is not None else None
    )

    config: dict[str, Any] = {
        "model": common_utils.without_none(
            {
                "provider": model_config.provider,
                "default": model_config.model,
                "base_url": model_config.base_url,
            }
        ),
        "agent": common_utils.without_none(
            {
                "max_turns": _max_turns(agent_config),
                "disabled_toolsets": blocked_toolsets or None,
            }
        ),
        "terminal": common_utils.without_none(
            {
                "backend": "local",
                "cwd": workspace,
                "timeout": settings.get("terminal_timeout", 60),
            }
        ),
    }

    skill_dirs = (
        [str(path) for path in agent_config.skills.paths]
        if agent_config.skills is not None
        else []
    )
    if skill_dirs:
        config["skills"] = {"external_dirs": skill_dirs}

    mcp_servers = agent_config.mcp.servers if agent_config.mcp is not None else {}
    if mcp_servers:
        config["mcp_servers"] = {
            name: hermes_mcp_server_config(server)
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
    agent_config: AgentConfig,
    hermes_home: Path,
    *,
    workspace: str,
    relay_enabled: bool = False,
) -> tuple[Path, dict[str, Any]]:
    hermes_home.mkdir(parents=True, exist_ok=True)
    config = build_hermes_config(
        agent_config,
        workspace=workspace,
        relay_enabled=relay_enabled,
    )
    config_path = hermes_home / "config.yaml"
    config_path.write_text(common_utils.dump_yaml(config), encoding="utf-8")
    return config_path, config


def hermes_mcp_server_config(server: AgentMcpServerConfig) -> dict[str, Any]:
    transport = server.transport.strip().lower()
    target = os.path.expandvars(server.url).strip()
    if not target:
        raise ValueError("MCP server mapping requires a URL")

    if transport == "stdio":
        return common_utils.without_none(
            {
                "enabled": True,
                "command": target,
                "args": server.args or None,
                "env": server.env or None,
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

    lifecycle.serve(HermesRuntime, config_loader=AgentConfig.from_mapping)


def resolve_hermes_toolsets(
    agent_config: AgentConfig, config: dict[str, Any]
) -> list[str] | None:
    enabled = agent_config.tools.enabled if agent_config.tools is not None else None
    if enabled is not None:
        return enabled

    from hermes_cli.tools_config import _get_platform_tools

    return sorted(_get_platform_tools(config, "cli"))


def _artifact_root(runtime_context: RuntimeContext, base_dir: str) -> Path:
    root = runtime_context.artifacts.root
    if root:
        artifact_root = Path(str(root))
        if not artifact_root.is_absolute():
            artifact_root = Path(base_dir) / artifact_root
        return artifact_root.resolve()
    return Path(base_dir).resolve() / "artifacts"


class HermesRuntime:
    """One Hermes agent and session database owned by a Fabric runtime."""

    def __init__(self) -> None:
        self._started = False
        self._agent_config: AgentConfig | None = None
        self._runtime_id: str | None = None
        self._model_config: AgentModelConfig | None = None
        self._hermes_home: Path | None = None
        self._hermes_config_path: Path | None = None
        self._hermes_config: dict[str, Any] = {}
        self._enabled_toolsets: list[str] | None = None
        self._conversation_history: list[dict[str, Any]] | None = None
        self._session_db: Any = None
        self._agent: Any = None
        self._invoke_hook: Any = None
        self._relay_plugin_config: dict[str, Any] | None = None
        self._relay_context: Any = None
        self._relay_context_entered = False
        self._relay_session_pending = False
        self._relay_finalize_hook_invoked = False
        self._relay_model_name = "unknown"

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
            runtime_context = RuntimeContext.from_mapping(payload.get("runtime_context"))
            validate_hermes_telemetry_provider(runtime_context)
            self._agent_config = agent_config
            settings = _settings(agent_config)
            model_config = _selected_model(agent_config)
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
                str(settings.get("terminal_timeout", 60)),
            )

            relay_enabled = bool(
                runtime_context.telemetry and runtime_context.telemetry.relay_enabled
            )
            if relay_enabled:
                relay_payload = {**payload, "config": agent_config.to_mapping()}
                self._relay_plugin_config = common_utils.load_relay_plugin_config(
                    relay_payload
                )
                from nemo_relay import plugin

                self._relay_context = plugin.plugin(self._relay_plugin_config)
                await self._relay_context.__aenter__()
                self._relay_context_entered = True

            self._hermes_config_path, self._hermes_config = write_hermes_config(
                agent_config,
                self._hermes_home,
                workspace=str(runtime_context.environment.workspace or "."),
                relay_enabled=relay_enabled,
            )
            api_key_env = _api_key_env(model_config)
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise RuntimeError(f"{api_key_env} is required for Hermes mode")
            self._relay_model_name = model_config.model

            from hermes_cli.config import load_config
            from hermes_cli.plugins import discover_plugins
            from hermes_cli.plugins import invoke_hook
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
                    agent_config, loaded_hermes_config
                )
                self._session_db = SessionDB()
                max_iterations = _max_turns(agent_config)
                if max_iterations is None:
                    max_iterations = DEFAULT_MAX_ITERATIONS
                temperature = model_config.temperature
                self._agent = AIAgent(
                    **filter_supported_kwargs(
                        AIAgent.__init__,
                        base_url=model_config.base_url,
                        api_key=api_key,
                        provider=model_config.provider,
                        model=model_config.model,
                        max_iterations=int(max_iterations),
                        enabled_toolsets=self._enabled_toolsets,
                        disabled_toolsets=disabled_toolsets(agent_config) or None,
                        quiet_mode=True,
                        skip_context_files=True,
                        skip_memory=True,
                        save_trajectories=bool(
                            settings.get("save_trajectories", False)
                        ),
                        max_tokens=settings.get("max_tokens", 512),
                        request_overrides=(
                            {"temperature": temperature}
                            if temperature is not None
                            else None
                        ),
                        reasoning_config=settings.get(
                            "reasoning_config", {"effort": "none"}
                        ),
                        platform="fabric",
                        session_id=self._runtime_id,
                        session_db=self._session_db,
                    )
                )
            self._invoke_hook = invoke_hook
            self._started = True
        except BaseException:
            await self.stop()
            raise

    async def invoke(self, invocation: dict[str, Any]) -> dict[str, Any]:
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
        runtime_context = RuntimeContext.from_mapping(invocation.get("runtime_context"))
        if runtime_context.runtime_id != self._runtime_id:
            raise lifecycle.LifecycleError(
                "hermes_runtime_mismatch",
                "Hermes invocation does not match the active runtime",
            )

        request = common_utils.request_payload(invocation)
        user_message = request.get("input") or ""
        if not isinstance(user_message, str):
            user_message = json.dumps(user_message, sort_keys=True)
        instructions = agent_config.instructions
        system_prompt = (
            instructions.system.content if instructions and instructions.system else None
        )

        def invoke_turn() -> tuple[dict[str, Any], str]:
            return _invoke_hermes_turn(
                agent=self._agent,
                system_prompt=system_prompt,
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
                    "nemo_fabric_request_id": runtime_context.request_id,
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
            "model": model_config.model,
            "base_url": model_config.base_url,
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
        agent = self._agent
        session_db = self._session_db
        relay_context = self._relay_context
        relay_context_entered = self._relay_context_entered
        relay_plugin_config = self._relay_plugin_config
        had_mcp_servers = bool(self._hermes_config.get("mcp_servers"))
        errors: list[BaseException] = []
        if relay_plugin_config is not None and agent is not None:
            try:
                self._finalize_relay_session()
            except BaseException as error:
                errors.append(error)
        self._agent = None
        self._session_db = None
        self._agent_config = None
        self._runtime_id = None
        self._model_config = None
        self._hermes_home = None
        self._hermes_config_path = None
        self._hermes_config = {}
        self._enabled_toolsets = None
        self._conversation_history = None
        self._relay_context = None
        self._relay_context_entered = False
        self._relay_session_pending = False
        self._relay_finalize_hook_invoked = False
        self._invoke_hook = None
        self._relay_plugin_config = None
        self._relay_model_name = "unknown"
        self._started = False

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
        if relay_context is not None and relay_context_entered:
            try:
                await relay_context.__aexit__(None, None, None)
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
) -> tuple[dict[str, Any], str]:
    hermes_stdout = StringIO()
    with redirect_stdout(hermes_stdout):
        conversation_kwargs = filter_supported_kwargs(
            agent.run_conversation,
            system_message=system_prompt,
            conversation_history=conversation_history,
            sync_honcho=False,
            dont_review=True,
        )
        result = agent.run_conversation(
            user_message,
            **conversation_kwargs,
        )
    return result, hermes_stdout.getvalue()


def filter_supported_kwargs(func: Any, **kwargs: Any) -> dict[str, Any]:
    signature = inspect.signature(func)
    supported = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in supported}


if __name__ == "__main__":
    main()
