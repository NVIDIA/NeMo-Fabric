#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenHands SDK adapter for NVIDIA NeMo Fabric."""

from __future__ import annotations

import importlib.util
import json
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
    default_condenser: Any


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
    from openhands.sdk.context.condenser import default_condenser
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
        default_condenser=default_condenser,
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


def _wire_model(model: contract.AgentModelConfig) -> str:
    """Return the LiteLLM model name, prefixed with its provider."""
    return model.model if "/" in model.model else f"{model.provider}/{model.model}"


def _omits_zero_cache_reads(wire_model: str) -> bool:
    """Whether the model's API leaves out a cache-read count of 0.

    Gemini usage counters are proto3 ``int32`` fields without presence, and
    proto3 JSON omits such a field when its value is 0.
    """
    provider, _, name = wire_model.partition("/")
    return provider in {"gemini", "vertex_ai"} and name.startswith("gemini")


def _llm(api: OpenHandsApi, model: contract.AgentModelConfig) -> Any:
    kwargs: dict[str, Any] = {"model": _wire_model(model), "usage_id": "agent"}
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


def _condenser_enabled(config: contract.AgentConfig) -> bool:
    """Return whether to condense history, rejecting values the descriptor forbids.

    Planning already validates the setting, but a host that skips planning must
    not turn an unknown value into model-billed summarization calls.
    """

    settings = config.harness.settings if config.harness is not None else {}
    condenser = settings.get("condenser", "default")
    if condenser not in ("default", "none"):
        raise lifecycle.LifecycleError(
            "openhands_condenser_invalid",
            "The OpenHands condenser setting must be 'default' or 'none'.",
        )
    return condenser == "default"


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
    if config.mcp is None:
        return result
    if config.mcp.extensions:
        raise lifecycle.LifecycleError(
            "openhands_mcp_configuration_unsupported",
            "OpenHands does not support MCP configuration extensions.",
        )
    servers = config.mcp.servers
    for name, server in servers.items():
        if server.extensions:
            raise lifecycle.LifecycleError(
                "openhands_mcp_configuration_unsupported",
                f"OpenHands does not support extensions for MCP server {name!r}.",
            )
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
            if server.custom_headers:
                raise lifecycle.LifecycleError(
                    "openhands_mcp_configuration_unsupported",
                    f"OpenHands stdio MCP server {name!r} does not support "
                    "custom_headers.",
                )
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
        if server.args or server.env:
            raise lifecycle.LifecycleError(
                "openhands_mcp_configuration_unsupported",
                f"OpenHands remote MCP server {name!r} does not support args or env.",
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


_UINT64_MAX = (1 << 64) - 1


class _CacheReadReports:
    """Record, per model response, whether the provider reported cache reads.

    OpenHands stores an omitted cache-read count as 0 in its usage records,
    but its completion log carries the LiteLLM response, whose
    ``cached_tokens`` field is present only when the provider reported it.
    For an API that omits a count of 0, an absent field is a reported 0.
    Each entry is ``(response_id, reported)``, matching the ``response_id``
    of the OpenHands usage record for the same response.
    """

    def __init__(self, omitted_means_zero: bool = False) -> None:
        self.reported: list[tuple[str, bool]] = []
        self._omitted_means_zero = omitted_means_zero

    def record(self, filename: str, log_data: str) -> None:
        # OpenHands logs failed requests through the same callback.
        if filename.endswith("-error.json"):
            return
        response = json.loads(log_data).get("response") or {}
        usage = response.get("usage") or {}
        details = (
            usage.get("prompt_tokens_details")
            or usage.get("input_tokens_details")
            or {}
        )
        self.reported.append(
            (
                str(response.get("id", "")),
                self._omitted_means_zero or "cached_tokens" in details,
            )
        )


def _bounded(count: int) -> int | None:
    """Return a count the contract can carry, or unknown past uint64."""
    return count if count <= _UINT64_MAX else None


def _sum_if_all_positive(counts: list[int]) -> int | None:
    """Return the sum of per-response counts, or unknown if any count is 0.

    LiteLLM stores an omitted prompt or completion count as 0, so a 0 cannot
    be told apart from a reported zero.
    """
    if not counts or any(count <= 0 for count in counts):
        return None
    return _bounded(sum(counts))


def _cached_input_tokens(
    records: list[Any],
    cache_read_reports: list[tuple[str, bool]],
    input_tokens: int | None,
) -> int | None:
    """Sum cache reads when every response reported them, else unknown.

    The reports must name the same responses, in order, as the usage records.
    A mismatch means the completion log missed a response, for example after
    OpenHands replaced the telemetry that holds the callback.
    """
    if [response_id for response_id, _ in cache_read_reports] != [
        r.response_id for r in records
    ] or not all(reported for _, reported in cache_read_reports):
        return None
    # Cache reads are part of input, so a count above the response's own
    # reported input, or above the summed input, is inconsistent.
    if any(0 < int(r.prompt_tokens) < int(r.cache_read_tokens) for r in records):
        return None
    cached = _bounded(sum(int(r.cache_read_tokens) for r in records))
    if cached is None or (input_tokens is not None and cached > input_tokens):
        return None
    return cached


def _usage(
    api_llm: Any, baseline: Any, cache_read_reports: list[tuple[str, bool]]
) -> contract.AgentUsage | None:
    """Return the tokens OpenHands recorded for this invocation, or unknown.

    OpenHands records a latency for every model response but token usage only
    for responses with a positive prompt or completion count, so fewer usage
    records than responses means usage is incomplete. Input and output are
    known only when every usage record has them above 0. Cache reads are known
    only when every response reported them, which may be 0. LiteLLM's
    ``prompt_tokens`` includes cache reads and writes. OpenHands cost mixes a
    LiteLLM proxy cost header with LiteLLM's price-table estimate, so it is
    not reported.
    """
    metrics = api_llm.metrics.diff(baseline)
    records = metrics.token_usages
    if not records or len(records) != len(metrics.response_latencies):
        return None
    input_tokens = _sum_if_all_positive([int(r.prompt_tokens) for r in records])
    output_tokens = _sum_if_all_positive([int(r.completion_tokens) for r in records])
    cached_input_tokens = _cached_input_tokens(
        records, cache_read_reports, input_tokens
    )
    if input_tokens is None and output_tokens is None and cached_input_tokens is None:
        return None
    return contract.AgentUsage(
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        input_tokens_include_cache=True if input_tokens is not None else None,
        output_tokens=output_tokens,
        total_tokens=(
            _bounded(input_tokens + output_tokens)
            if input_tokens is not None and output_tokens is not None
            else None
        ),
    )


class OpenHandsRuntime:
    def __init__(self) -> None:
        self._api: OpenHandsApi | None = None
        self._llm_instance: Any = None
        self._conversation: Any = None
        self._profile_store_temp: tempfile.TemporaryDirectory[str] | None = None
        self._runtime_id: str | None = None
        self._cache_read_reports = _CacheReadReports()

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
        model = _selected_model(config)
        llm = _llm(api, model)
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
        if _condenser_enabled(config):
            # Share the agent LLM instead of a copy so summarization calls
            # accrue to the metrics that invoke() reports as usage.
            agent_kwargs["condenser"] = api.default_condenser(llm)
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
        # The completion log is the only per-response record that keeps the
        # provider's cache fields. With a callback and no log directory,
        # OpenHands hands each log to the callback and writes no file. If
        # OpenHands replaces this telemetry, responses go unlogged and cache
        # reads become unknown rather than wrong.
        cache_read_reports = _CacheReadReports(
            omitted_means_zero=_omits_zero_cache_reads(_wire_model(model))
        )
        llm.telemetry.set_log_completions_callback(cache_read_reports.record)
        llm.telemetry.log_enabled = True
        self._cache_read_reports = cache_read_reports
        self._api = api
        self._llm_instance = llm
        self._conversation = conversation
        self._profile_store_temp = profile_store_temp
        self._runtime_id = context.runtime_id

    async def invoke(
        self,
        request: contract.AgentRunRequest,
        context: contract.RuntimeContext,
    ) -> contract.AgentRunResult:
        if (
            self._api is None
            or self._llm_instance is None
            or self._conversation is None
        ):
            raise lifecycle.LifecycleError(
                "openhands_not_started", "The OpenHands runtime is not started."
            )
        if context.runtime_id != self._runtime_id:
            raise lifecycle.LifecycleError(
                "openhands_runtime_mismatch",
                "OpenHands invocation does not match the active runtime.",
            )

        conversation = self._conversation
        event_count = len(conversation.state.events)
        baseline = self._llm_instance.metrics.deep_copy()
        reports = self._cache_read_reports.reported
        first_report = len(reports)
        try:
            conversation.send_message(common_utils.normalize_user_input(request.input))
            await conversation.arun()
        except Exception:
            return contract.AgentRunResult(
                status=contract.AgentRunStatus.FAILED,
                output=None,
                error=contract.AgentRunError(
                    code="openhands_invocation_failed",
                    message="OpenHands could not complete the invocation.",
                ),
                usage=_usage(self._llm_instance, baseline, reports[first_report:]),
            )

        status = conversation.state.execution_status
        new_events = conversation.state.events[event_count:]
        response = self._api.get_agent_final_response(new_events)
        usage = _usage(self._llm_instance, baseline, reports[first_report:])
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
        self._runtime_id = None
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
