#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenHands SDK adapter for NVIDIA NeMo Fabric."""

from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path
from typing import Any
from typing import NamedTuple

from nemo_fabric_adapter_contract import models as contract
from nemo_fabric_adapters.common import instructions as common_instructions
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common import utils as common_utils

HARNESS = "OpenHands"
DEFAULT_MAX_ITERATIONS = 500
TOOL_ALIASES = {
    "terminal": "terminal",
    "bash": "terminal",
    "file_editor": "file_editor",
    "edit": "file_editor",
}
REMOTE_MCP_TRANSPORTS = {
    "http": "http",
    "streamable-http": "streamable-http",
    "streamable_http": "streamable-http",
    "sse": "sse",
}


class OpenHandsApi(NamedTuple):
    Agent: Any
    AgentContext: Any
    Conversation: Any
    ConversationExecutionStatus: Any
    LLM: Any
    MCPServer: Any
    SecretStr: Any
    Skill: Any
    Tool: Any
    TerminalTool: Any
    FileEditorTool: Any
    get_agent_final_response: Any


def _load_openhands_api() -> OpenHandsApi:
    try:
        available = importlib.util.find_spec("openhands.sdk") is not None
    except ModuleNotFoundError:
        available = False
    if not available:
        raise lifecycle.LifecycleError(
            "openhands_harness_unavailable",
            "OpenHands is not installed; install compatible openhands-sdk and "
            "openhands-tools packages.",
        )

    from pydantic import SecretStr

    from openhands.sdk import Agent
    from openhands.sdk import AgentContext
    from openhands.sdk import ConversationExecutionStatus
    from openhands.sdk import LLM
    from openhands.sdk import Tool
    from openhands.sdk.conversation import get_agent_final_response
    from openhands.sdk.conversation.impl.local_conversation import LocalConversation
    from openhands.sdk.mcp import MCPServer
    from openhands.sdk.skills import Skill
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.terminal import TerminalTool

    return OpenHandsApi(
        Agent=Agent,
        AgentContext=AgentContext,
        Conversation=LocalConversation,
        ConversationExecutionStatus=ConversationExecutionStatus,
        LLM=LLM,
        MCPServer=MCPServer,
        SecretStr=SecretStr,
        Skill=Skill,
        Tool=Tool,
        TerminalTool=TerminalTool,
        FileEditorTool=FileEditorTool,
        get_agent_final_response=get_agent_final_response,
    )


def _selected_model(config: contract.AgentConfig) -> contract.AgentModelConfig:
    model = config.models.get("default")
    if model is None and len(config.models) == 1:
        model = next(iter(config.models.values()))
    if model is None:
        raise lifecycle.LifecycleError(
            "openhands_model_required",
            "OpenHands requires a default model or exactly one configured model.",
        )
    return model


def _llm(api: OpenHandsApi, model: contract.AgentModelConfig) -> Any:
    wire_model = (
        model.model if "/" in model.model else f"{model.provider}/{model.model}"
    )
    kwargs: dict[str, Any] = {"model": wire_model, "usage_id": "agent"}
    if model.api_key_env is not None:
        api_key = os.environ.get(model.api_key_env)
        if not api_key:
            raise lifecycle.LifecycleError(
                "openhands_credential_missing",
                f"The model credential environment variable "
                f"{model.api_key_env!r} is not set.",
            )
        kwargs["api_key"] = api.SecretStr(api_key)
    if model.base_url is not None:
        kwargs["base_url"] = model.base_url
    if model.temperature is not None:
        kwargs["temperature"] = model.temperature
    if model.top_p is not None:
        kwargs["top_p"] = model.top_p
    if model.max_tokens is not None:
        kwargs["max_output_tokens"] = model.max_tokens
    return api.LLM(**kwargs)


def _workspace(context: contract.RuntimeContext, base_dir: str) -> Path:
    path = Path(context.environment.workspace or base_dir)
    if not path.is_absolute():
        path = Path(base_dir) / path
    path = path.resolve()
    if not path.is_dir():
        raise lifecycle.LifecycleError(
            "openhands_workspace_invalid",
            f"The OpenHands workspace is not a directory: {path}",
        )
    return path


def _skill_paths(config: contract.AgentConfig, base_dir: str) -> list[Path]:
    paths: list[Path] = []
    for value in config.skills.paths if config.skills is not None else []:
        path = Path(value)
        if not path.is_absolute():
            path = Path(base_dir) / path
        path = path.resolve()
        if not path.is_dir() or not (path / "SKILL.md").is_file():
            raise lifecycle.LifecycleError(
                "openhands_skill_invalid",
                "An OpenHands skill path must be a directory containing "
                f"SKILL.md: {path}",
            )
        paths.append(path)
    return paths


def _skills(
    api: OpenHandsApi, config: contract.AgentConfig, base_dir: str
) -> list[Any]:
    skills: list[Any] = []
    names: set[str] = set()
    for path in _skill_paths(config, base_dir):
        try:
            skill = api.Skill.load(path / "SKILL.md", strict=True, skip_mcp=True)
        except Exception as error:
            raise lifecycle.LifecycleError(
                "openhands_skill_invalid",
                f"OpenHands could not load the configured skill at {path}.",
            ) from error
        if skill.name in names:
            raise lifecycle.LifecycleError(
                "openhands_skill_duplicate",
                f"OpenHands skill names must be unique: {skill.name!r}.",
            )
        names.add(skill.name)
        skills.append(skill)
    return skills


def _tools(api: OpenHandsApi, config: contract.AgentConfig) -> list[Any]:
    enabled = config.tools.enabled if config.tools is not None else None
    blocked_names = set(config.tools.blocked if config.tools is not None else [])
    selected = ["terminal", "file_editor"] if enabled is None else enabled

    native: list[str] = []
    unknown: list[str] = []
    blocked: set[str] = set()
    for name in blocked_names:
        canonical = TOOL_ALIASES.get(name)
        if canonical is None:
            unknown.append(name)
        else:
            blocked.add(canonical)
    for name in selected:
        canonical = TOOL_ALIASES.get(name)
        if canonical is None:
            unknown.append(name)
        elif canonical not in blocked and canonical not in native:
            native.append(canonical)
    if unknown:
        raise lifecycle.LifecycleError(
            "openhands_tool_unsupported",
            f"OpenHands does not support configured tool name {sorted(set(unknown))[0]!r}; "
            f"supported names: {', '.join(sorted(TOOL_ALIASES))}.",
        )

    classes = {
        "terminal": api.TerminalTool,
        "file_editor": api.FileEditorTool,
    }
    return [api.Tool(name=classes[name].name) for name in native]


def _mcp_servers(api: OpenHandsApi, config: contract.AgentConfig) -> dict[str, Any]:
    result: dict[str, Any] = {}
    servers = config.mcp.servers if config.mcp is not None else {}
    for name, server in servers.items():
        if server.authentication is not None:
            raise lifecycle.LifecycleError(
                "openhands_mcp_auth_unsupported",
                f"MCP authentication is not supported for server {name!r}.",
            )
        if server.allowed_tools is not None or server.blocked_tools:
            raise lifecycle.LifecycleError(
                "openhands_mcp_tool_policy_unsupported",
                f"Per-server MCP tool policy is not supported for server {name!r}.",
            )

        transport = server.transport.strip().lower()
        target = os.path.expandvars(server.url).strip()
        if not target:
            raise lifecycle.LifecycleError(
                "openhands_mcp_target_invalid",
                f"MCP server {name!r} must have a non-empty URL or command.",
            )
        if transport in {"stdio", "command", "process"}:
            result[name] = api.MCPServer(
                transport="stdio",
                command=target,
                args=server.args,
                env=server.env or None,
            )
            continue
        native_transport = REMOTE_MCP_TRANSPORTS.get(transport)
        if native_transport is None:
            raise lifecycle.LifecycleError(
                "openhands_mcp_transport_unsupported",
                f"MCP server {name!r} has unsupported transport {server.transport!r}.",
            )
        headers = None
        if server.custom_headers:
            try:
                headers = common_utils.expand_http_headers(name, server.custom_headers)
            except Exception as error:
                raise lifecycle.LifecycleError(
                    "openhands_mcp_header_invalid",
                    f"MCP headers are invalid for server {name!r}.",
                ) from error
        result[name] = api.MCPServer(
            transport=native_transport,
            url=target,
            headers=headers,
        )
    return result


def _usage(api_llm: Any, baseline: Any) -> contract.AgentUsage | None:
    metrics = api_llm.metrics.diff(baseline)
    tokens = metrics.accumulated_token_usage
    if tokens is None:
        return None
    input_tokens = int(tokens.prompt_tokens)
    output_tokens = int(tokens.completion_tokens)
    return contract.AgentUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cost_usd=float(metrics.accumulated_cost),
    )


class OpenHandsRuntime:
    def __init__(self) -> None:
        self._api: OpenHandsApi | None = None
        self._llm_instance: Any = None
        self._conversation: Any = None
        self._profile_store_temp: tempfile.TemporaryDirectory[str] | None = None

    async def start(self, payload: dict[str, Any]) -> None:
        config: contract.AgentConfig = payload["config"]
        context = contract.RuntimeContext.from_mapping(payload.get("runtime_context"))
        base_dir = common_utils.base_dir(payload)
        instruction = common_instructions.system_instruction(
            config,
            adapter=HARNESS,
            supported_modes={"replace", "append"},
        )
        api = _load_openhands_api()
        llm = _llm(api, _selected_model(config))
        agent_context = api.AgentContext(
            skills=_skills(api, config, base_dir),
            system_message_suffix=(
                instruction.content
                if instruction is not None and instruction.mode == "append"
                else None
            ),
            load_user_skills=False,
            load_public_skills=False,
            load_project_skills=False,
            load_memory=False,
        )
        agent_kwargs: dict[str, Any] = {
            "llm": llm,
            "tools": _tools(api, config),
            "mcp_config": _mcp_servers(api, config),
            "agent_context": agent_context,
        }
        if instruction is not None and instruction.mode == "replace":
            agent_kwargs["system_prompt"] = instruction.content
        agent = api.Agent(**agent_kwargs)
        profile_store_temp = tempfile.TemporaryDirectory(
            prefix="nemo-fabric-openhands-"
        )
        try:
            conversation = api.Conversation(
                agent=agent,
                workspace=str(_workspace(context, base_dir)),
                max_iteration_per_run=(
                    config.runtime.max_turns
                    if config.runtime is not None
                    and config.runtime.max_turns is not None
                    else DEFAULT_MAX_ITERATIONS
                ),
                persistence_dir=None,
                profile_store_dir=profile_store_temp.name,
                delete_on_close=True,
                visualizer=None,
            )
        except BaseException:
            profile_store_temp.cleanup()
            raise
        self._api = api
        self._llm_instance = llm
        self._conversation = conversation
        self._profile_store_temp = profile_store_temp

    async def invoke(
        self,
        request: contract.AgentRunRequest,
        context: contract.RuntimeContext,
    ) -> contract.AgentRunResult:
        del context
        if (
            self._api is None
            or self._llm_instance is None
            or self._conversation is None
        ):
            raise lifecycle.LifecycleError(
                "openhands_not_started", "The OpenHands runtime is not started."
            )

        conversation = self._conversation
        event_count = len(conversation.state.events)
        baseline = self._llm_instance.metrics.deep_copy()
        conversation.send_message(common_utils.normalize_user_input(request.input))
        try:
            await conversation.arun()
        except Exception:
            return contract.AgentRunResult(
                status=contract.AgentRunStatus.FAILED,
                output=None,
                error=contract.AgentRunError(
                    code="openhands_invocation_failed",
                    message="OpenHands could not complete the invocation.",
                ),
                usage=_usage(self._llm_instance, baseline),
            )

        status = conversation.state.execution_status
        new_events = conversation.state.events[event_count:]
        response = self._api.get_agent_final_response(new_events)
        usage = _usage(self._llm_instance, baseline)
        if status != self._api.ConversationExecutionStatus.FINISHED:
            return contract.AgentRunResult(
                status=contract.AgentRunStatus.FAILED,
                output=None,
                error=contract.AgentRunError(
                    code=f"openhands_{str(status.value).replace('-', '_')}",
                    message=f"OpenHands ended the invocation with status {status.value!r}.",
                ),
                usage=usage,
            )
        if not response:
            return contract.AgentRunResult(
                status=contract.AgentRunStatus.FAILED,
                output=None,
                error=contract.AgentRunError(
                    code="openhands_no_assistant_response",
                    message="OpenHands completed without an assistant response.",
                ),
                usage=usage,
            )
        return contract.AgentRunResult(
            status=contract.AgentRunStatus.SUCCEEDED,
            output={"response": response},
            usage=usage,
        )

    async def stop(self) -> None:
        conversation, self._conversation = self._conversation, None
        profile_store_temp, self._profile_store_temp = self._profile_store_temp, None
        self._api = None
        self._llm_instance = None
        try:
            if conversation is not None:
                conversation.close()
        finally:
            if profile_store_temp is not None:
                profile_store_temp.cleanup()


def main() -> None:
    lifecycle.serve(OpenHandsRuntime, config_loader=contract.AgentConfig.from_mapping)


if __name__ == "__main__":
    main()
