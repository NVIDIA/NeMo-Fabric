#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import importlib.util
import os
from typing import Any

from nemo_fabric_adapter_contract import models as contract
from nemo_fabric_adapters.common import instructions as common_instructions
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common import utils as common_utils

SYSTEM_TEMPLATE = "{{system_instruction}}"
INSTANCE_TEMPLATE = """Solve this task in the current workspace:

{{task}}

Use the bash tool. When complete, run one final command that prints
`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` on its first line and your final response
on the following lines. The command output after the first line is submitted as
your final response, so include it in the same command.
"""
_UNREADABLE = object()
_MAX_TOKEN_COUNT = (1 << 64) - 1
_USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "total_tokens",
)


def _selected_model(config: contract.AgentConfig) -> contract.AgentModelConfig:
    model = config.models.get("default")
    if model is None and len(config.models) == 1:
        model = next(iter(config.models.values()))
    if model is None:
        raise lifecycle.LifecycleError(
            "mini_swe_agent_missing_model", "mini-SWE-agent needs a model"
        )
    return model


class MiniSweAgentRuntime:
    def __init__(self) -> None:
        self._model = self._environment = self._agent = None
        self._system_instruction = ""
        self._relay_enabled = False
        self._relay_plugin = self._relay_scope = self._relay_scope_type = None
        self._relay_plugin_config: dict[str, Any] | None = None
        self._telemetry_quarantine: str | None = None

    async def start(self, payload: dict[str, Any]) -> None:
        config: contract.AgentConfig = payload["config"]
        instruction = common_instructions.system_instruction(
            config,
            adapter="mini-SWE-agent",
            supported_modes={"replace"},
        )
        from minisweagent.environments.local import LocalEnvironment
        from nemo_fabric_adapters.mini_swe_agent.agents import (
            RelayRetainingDefaultAgent,
        )
        from nemo_fabric_adapters.mini_swe_agent.agents import RetainingDefaultAgent
        from nemo_fabric_adapters.mini_swe_agent.agents import (
            UsageRecordingLitellmModel,
        )

        runtime_context = contract.RuntimeContext.from_mapping(
            payload.get("runtime_context")
        )
        self._relay_enabled = bool(
            runtime_context.telemetry and runtime_context.telemetry.relay_enabled
        )
        agent_type = RetainingDefaultAgent
        if self._relay_enabled:
            if importlib.util.find_spec("nemo_relay") is None:
                raise lifecycle.LifecycleError(
                    "mini_swe_agent_relay_missing",
                    "telemetry is enabled but a compatible 'nemo-relay' package is "
                    "not installed; install the Relay extra (pip install "
                    "'nemo-fabric-adapters-mini-swe-agent[relay]').",
                )
            common_utils.reject_ambient_relay_plugin_config()
            relay_payload = {**payload, "config": config.to_mapping()}
            self._relay_plugin_config = common_utils.load_relay_plugin_config(
                relay_payload
            )
            from nemo_relay import ScopeType, plugin, scope

            self._relay_plugin = plugin
            self._relay_scope = scope
            self._relay_scope_type = ScopeType
            agent_type = RelayRetainingDefaultAgent

        model = _selected_model(config)
        model_kwargs: dict[str, Any] = {}
        if model.api_key_env and (api_key := os.environ.get(model.api_key_env)):
            model_kwargs["api_key"] = api_key
        if model.base_url is not None:
            model_kwargs["api_base"] = model.base_url
        if model.temperature is not None:
            model_kwargs["temperature"] = model.temperature
        if model.top_p is not None:
            model_kwargs["top_p"] = model.top_p
        if model.max_tokens is not None:
            model_kwargs["max_tokens"] = model.max_tokens
        self._model = UsageRecordingLitellmModel(
            model_name=(
                model.model if "/" in model.model else f"{model.provider}/{model.model}"
            ),
            model_kwargs=model_kwargs,
        )
        self._environment = LocalEnvironment(
            **(config.harness.settings if config.harness else {})
        )
        self._system_instruction = instruction.content if instruction else ""
        agent_kwargs: dict[str, Any] = {}
        if self._relay_enabled:
            agent_kwargs["relay_model_name"] = self._model.config.model_name
        self._agent = agent_type(
            self._model,
            self._environment,
            system_template=SYSTEM_TEMPLATE,
            instance_template=INSTANCE_TEMPLATE,
            step_limit=config.runtime.max_turns if config.runtime else 0,
            **agent_kwargs,
        )

    async def invoke(
        self,
        request: contract.AgentRunRequest,
        context: contract.RuntimeContext,
    ) -> contract.AgentRunResult:
        task = common_utils.normalize_user_input(request.input)
        inherited_quarantine = self._telemetry_quarantine is not None
        telemetry_errors: list[str] = []
        self._model.provider_usage.clear()
        if self._relay_enabled:
            result, telemetry_errors = await self._run_with_relay(
                task, context, request.relay_session_root
            )
        else:
            result = await self._run_agent(task)
        failed = result.get("exit_status") != "Submitted"
        output: dict[str, Any] = {
            "output": result.get("submission", ""),
            "usage": {"api_calls": self._agent.n_calls},
        }
        if self._relay_enabled:
            output["telemetry"] = {
                "enabled": True,
                "provider": "relay",
                "emitter": "mini-swe-agent.subclass/nemo_relay",
            }
            if not inherited_quarantine:
                try:
                    assert self._relay_plugin_config is not None
                    output["relay_artifacts"] = common_utils.collect_relay_artifacts(
                        self._relay_plugin_config
                    )
                except Exception as error:
                    telemetry_errors.append(_error_text(error))
            if telemetry_errors:
                output["telemetry"].update(
                    {
                        "degraded": True,
                        "error": "; ".join(telemetry_errors),
                    }
                )
        error = (
            contract.AgentRunError(
                code="mini_swe_agent_incomplete",
                message="mini-SWE-agent stopped before submitting a final output",
                retryable=False,
            )
            if failed
            else None
        )
        return contract.AgentRunResult(
            status=(
                contract.AgentRunStatus.FAILED
                if failed
                else contract.AgentRunStatus.SUCCEEDED
            ),
            output=output,
            error=error,
            usage=_invocation_usage(
                self._agent.invocation_responses,
                self._agent.n_calls,
                self._model.provider_usage,
            ),
        )

    async def _run_agent(self, task: str) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._agent.run, task, system_instruction=self._system_instruction
        )

    async def _run_with_relay(
        self,
        task: str,
        context: contract.RuntimeContext,
        session_root: str | None,
    ) -> tuple[dict[str, Any], list[str]]:
        if self._telemetry_quarantine is not None:
            self._agent.begin_relay_invocation(None)
            try:
                result = await self._run_agent(task)
            finally:
                self._agent.end_relay_invocation()
            return result, [self._telemetry_quarantine]

        baseline = _current_scope_handle()
        result: dict[str, Any] | None = None
        telemetry_errors: list[str] = []
        try:
            common_utils.reject_ambient_relay_plugin_config()
            assert self._relay_plugin_config is not None
            async with self._relay_plugin.activate(
                self._relay_plugin_config
            ) as activation:
                common_utils.reject_inherited_relay_plugin_config(activation.report)
                request_context, metadata = common_utils.relay_request_context(
                    context.request_id, session_root
                )
                metadata.update(common_utils.relay_correlation_metadata(context))
                with (
                    request_context,
                    self._relay_scope.scope(
                        "mini-swe-agent.request",
                        self._relay_scope_type.Agent,
                        metadata=metadata,
                    ) as handle,
                ):
                    self._agent.begin_relay_invocation(handle)
                    try:
                        result = await self._run_agent(task)
                    finally:
                        telemetry_errors.extend(self._agent.end_relay_invocation())
        except Exception as error:
            if result is None:
                raise
            telemetry_errors.append(_error_text(error))

        if telemetry_errors and not _scope_top_unchanged(baseline):
            self._telemetry_quarantine = (
                "telemetry unreliable for the rest of this runtime: an earlier "
                "turn left the Relay scope stack dirty"
            )
            telemetry_errors.append(self._telemetry_quarantine)
        assert result is not None
        return result, telemetry_errors

    async def stop(self) -> None:
        self.__init__()


def _invocation_usage(
    responses: list[Any],
    model_calls: int,
    provider_usage: dict[str, dict[str, Any]],
) -> contract.AgentUsage | None:
    """Sum the LiteLLM usage of one invocation's model calls.

    A count is known only when every model call stored a response that reported
    it, so a call without a stored response leaves every count unknown. LiteLLM
    returns OpenAI-shaped usage, where ``prompt_tokens`` includes cached tokens;
    its Anthropic conversion adds cache reads and writes to ``prompt_tokens``.
    ``provider_usage`` maps response ids to the provider's own usage, recorded
    before that conversion. mini-SWE-agent's cost is a LiteLLM price-table estimate, so it is not
    reported as ``cost_usd``.
    """

    if not responses or len(responses) != model_calls:
        return None
    calls = [_call_usage(response, provider_usage) for response in responses]
    totals: dict[str, int | None] = {}
    for field in _USAGE_FIELDS:
        values = [call[field] for call in calls]
        total = None if None in values else sum(values)
        totals[field] = total if _is_count(total) else None
    if all(value is None for value in totals.values()):
        return None
    return contract.AgentUsage(
        **totals,
        input_tokens_include_cache=(
            True
            if totals["input_tokens"] is not None
            or totals["cached_input_tokens"] is not None
            else None
        ),
    )


def _call_usage(
    response: Any, provider_usage: dict[str, dict[str, Any]]
) -> dict[str, int | None]:
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return dict.fromkeys(_USAGE_FIELDS)
    if any(
        usage.get(key) is not None
        for key in ("cache_read_input_tokens", "cache_creation_input_tokens")
    ):
        # LiteLLM's Anthropic and Bedrock conversions zero-fill omitted counts,
        # add cache reads and writes to ``prompt_tokens``, and may compute the
        # total, so read the provider's own usage instead. An OpenAI-shaped
        # body that carries these keys keeps the OpenAI rules below.
        raw = provider_usage.get(response.get("id"))
        raw = raw if isinstance(raw, dict) else {}
        if raw.keys() & {"input_tokens", "output_tokens", "iterations"}:
            return _anthropic_usage(raw)
        if raw.keys() & {"inputTokens", "outputTokens"}:
            return _bedrock_usage(raw)
        if "prompt_tokens" not in raw:
            return _normalized_cache_usage(usage)
    # LiteLLM stores an omitted prompt, completion, or total count as 0, so a
    # 0 from a call that ran is unknown. It keeps a cached count only when the
    # provider reported one, so a cached 0 is known.
    input_tokens = _nonzero_count(usage.get("prompt_tokens"))
    output_tokens = _nonzero_count(usage.get("completion_tokens"))
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    reported_total = usage.get("total_tokens")
    # A valid reported total is kept on its own. A missing or zero-filled total
    # is derived only from two known parts, and an invalid one stays unknown.
    total_tokens = _nonzero_count(reported_total)
    if reported_total is None or (_is_count(reported_total) and reported_total == 0):
        total_tokens = _sum_known(input_tokens, output_tokens)
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached if _is_count(cached) else None,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _anthropic_usage(raw: dict[str, Any]) -> dict[str, int | None]:
    iterations = raw.get("iterations")
    if not iterations:
        return _anthropic_counts(raw)
    # LiteLLM sums the per-iteration counts instead of the top-level ones.
    if not isinstance(iterations, list) or not all(
        isinstance(iteration, dict) for iteration in iterations
    ):
        return dict.fromkeys(_USAGE_FIELDS)
    counts = [_anthropic_counts(iteration) for iteration in iterations]
    return {
        field: _sum_known(*(count[field] for count in counts))
        for field in _USAGE_FIELDS
    }


def _anthropic_counts(raw: dict[str, Any]) -> dict[str, int | None]:
    # Anthropic always sends all four counts; cache counts may be null. Input
    # is known only when its uncached part and both cache parts are reported.
    cache_read = _count(raw.get("cache_read_input_tokens"))
    input_tokens = _sum_known(
        _count(raw.get("input_tokens")),
        cache_read,
        _count(raw.get("cache_creation_input_tokens")),
    )
    output_tokens = _count(raw.get("output_tokens"))
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cache_read,
        "output_tokens": output_tokens,
        "total_tokens": _sum_known(input_tokens, output_tokens),
    }


def _bedrock_usage(raw: dict[str, Any]) -> dict[str, int | None]:
    # Bedrock Converse always sends input, output, and total, and its cache
    # counts are optional, so an omitted cache count adds nothing to input but
    # leaves the cached count unknown.
    cache_parts = [
        raw[key]
        for key in ("cacheReadInputTokens", "cacheWriteInputTokens")
        if raw.get(key) is not None
    ]
    input_tokens = _sum_known(
        _count(raw.get("inputTokens")), *(_count(part) for part in cache_parts)
    )
    output_tokens = _count(raw.get("outputTokens"))
    reported_total = raw.get("totalTokens")
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": _count(raw.get("cacheReadInputTokens")),
        "output_tokens": output_tokens,
        "total_tokens": (
            _sum_known(input_tokens, output_tokens)
            if reported_total is None
            else _count(reported_total)
        ),
    }


def _normalized_cache_usage(usage: dict[str, Any]) -> dict[str, int | None]:
    # Without the provider's usage, LiteLLM's zero-filled parts hide whether a
    # cache count was omitted, so inclusive input and its total stay unknown,
    # and only a nonzero cached or output count is known.
    details = usage.get("prompt_tokens_details")
    details = details if isinstance(details, dict) else {}
    return {
        "input_tokens": None,
        "cached_input_tokens": _nonzero_count(details.get("cached_tokens")),
        "output_tokens": _nonzero_count(usage.get("completion_tokens")),
        "total_tokens": None,
    }


def _sum_known(*values: int | None) -> int | None:
    known = [value for value in values if value is not None]
    if len(known) != len(values):
        return None
    total = sum(known)
    return total if _is_count(total) else None


def _count(value: Any) -> int | None:
    return value if _is_count(value) else None


def _nonzero_count(value: Any) -> int | None:
    return value if _is_count(value) and value > 0 else None


def _is_count(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= _MAX_TOKEN_COUNT
    )


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _current_scope_handle() -> Any:
    try:
        import nemo_relay

        return nemo_relay.scope.get_handle()
    except Exception:
        return None


def _scope_top_unchanged(baseline: Any) -> bool:
    current = _current_scope_handle()
    if baseline is None:
        return current is None
    if current is None:
        return False
    baseline_uuid = getattr(baseline, "uuid", _UNREADABLE)
    current_uuid = getattr(current, "uuid", _UNREADABLE)
    if baseline_uuid is _UNREADABLE or current_uuid is _UNREADABLE:
        return False
    return bool(current_uuid == baseline_uuid)


def main() -> None:
    lifecycle.serve(
        MiniSweAgentRuntime, config_loader=contract.AgentConfig.from_mapping
    )


if __name__ == "__main__":
    main()
