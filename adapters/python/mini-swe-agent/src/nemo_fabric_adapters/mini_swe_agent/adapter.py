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
        from minisweagent.models.litellm_model import LitellmModel
        from nemo_fabric_adapters.mini_swe_agent.agents import (
            RelayRetainingDefaultAgent,
        )
        from nemo_fabric_adapters.mini_swe_agent.agents import RetainingDefaultAgent

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
        self._model = LitellmModel(
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
        if self._relay_enabled:
            result, telemetry_errors = await self._run_with_relay(task, context)
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
        )

    async def _run_agent(self, task: str) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._agent.run, task, system_instruction=self._system_instruction
        )

    async def _run_with_relay(
        self,
        task: str,
        context: contract.RuntimeContext,
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
            async with self._relay_plugin.plugin(
                self._relay_plugin_config
            ) as activation_report:
                common_utils.reject_inherited_relay_plugin_config(activation_report)
                request_context, metadata = common_utils.relay_request_context(
                    context.request_id
                )
                metadata["nemo_fabric_invocation_id"] = context.invocation_id
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
