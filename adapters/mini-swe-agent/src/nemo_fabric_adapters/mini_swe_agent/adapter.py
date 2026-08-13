#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from nemo_fabric_adapter_contract import models as contract
from nemo_fabric_adapters.common import lifecycle, utils as common_utils

DEFAULT_API_KEY_ENVS = {
    provider: f"{provider.upper()}_API_KEY"
    for provider in ("anthropic", "nvidia", "openai", "openrouter")
}
DEFAULT_SYSTEM_TEMPLATE = "You are a helpful software engineering assistant."
INSTANCE_TEMPLATE = """Solve this task in the current workspace:

{{task}}

Use the bash tool. When complete, run
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` by itself, then provide final output.
"""


def main() -> None:
    lifecycle.serve(
        MiniSweAgentRuntime, config_loader=contract.AgentConfig.from_mapping
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


def _model_api_key(
    model: contract.AgentModelConfig, environment: dict[str, str]
) -> str | None:
    name = model.api_key_env or DEFAULT_API_KEY_ENVS.get(model.provider)
    if name is None:
        return None
    value = environment.get(name) or os.environ.get(name)
    if not value:
        raise lifecycle.LifecycleError(
            "mini_swe_agent_missing_api_key",
            f"Model credential environment variable {name!r} is not set",
        )
    return value


class MiniSweAgentRuntime:
    def __init__(self) -> None:
        self._model = self._environment = None
        self._runtime_id: str | None = None
        self._agent_kwargs: dict[str, Any] = {}

    async def start(self, payload: dict[str, Any]) -> None:
        config = payload.get("config")
        if not isinstance(config, contract.AgentConfig):
            raise lifecycle.LifecycleError(
                "mini_swe_agent_invalid_config",
                "mini-SWE-agent requires a validated AgentConfig",
            )
        context = contract.RuntimeContext.from_mapping(payload.get("runtime_context"))
        workspace = context.environment.workspace
        if workspace is None:
            raise lifecycle.LifecycleError(
                "mini_swe_agent_missing_workspace", "environment.workspace is required"
            )
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.litellm_model import LitellmModel

        model = _selected_model(config)
        model_kwargs: dict[str, Any] = {}
        if api_key := _model_api_key(model, context.environment.env):
            model_kwargs["api_key"] = api_key
        if model.base_url is not None:
            model_kwargs["api_base"] = model.base_url
        if model.temperature is not None:
            model_kwargs["temperature"] = model.temperature
        settings = config.harness.settings if config.harness else {}
        timeout = settings.get("timeout_seconds", 180)
        self._model = LitellmModel(
            model_name=(
                model.model if "/" in model.model else f"{model.provider}/{model.model}"
            ),
            model_kwargs=model_kwargs,
            cost_tracking="ignore_errors",
        )
        self._environment = LocalEnvironment(
            cwd=str(workspace), env=context.environment.env, timeout=timeout
        )
        system = config.instructions.system if config.instructions else None
        template = system.content if system else DEFAULT_SYSTEM_TEMPLATE
        self._agent_kwargs = {
            "system_template": template,
            "instance_template": INSTANCE_TEMPLATE,
            "step_limit": config.runtime.max_turns if config.runtime else 0,
            "cost_limit": 0,
        }
        self._runtime_id = context.runtime_id

    async def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        context = contract.RuntimeContext.from_mapping(payload.get("runtime_context"))
        if self._model is None or self._environment is None:
            raise lifecycle.LifecycleError(
                "mini_swe_agent_not_started", "mini-SWE-agent runtime is not started"
            )
        if context.runtime_id != self._runtime_id:
            raise lifecycle.LifecycleError(
                "mini_swe_agent_runtime_mismatch",
                "mini-SWE-agent invocation does not match the active runtime",
            )
        raw_task = common_utils.request_payload(payload).get("input", "")
        task = raw_task if isinstance(raw_task, str) else json.dumps(raw_task)
        from minisweagent.agents.default import DefaultAgent

        agent = DefaultAgent(self._model, self._environment, **self._agent_kwargs)
        result = await asyncio.to_thread(agent.run, task)
        failed = result.get("exit_status") != "Submitted"
        output: dict[str, Any] = {
            "failed": failed,
            "output": result.get("submission", ""),
            "usage": {"cost_usd": agent.cost, "api_calls": agent.n_calls},
            "exit_status": result.get("exit_status"),
        }
        if failed:
            output["error"] = {
                "code": "mini_swe_agent_incomplete",
                "message": "mini-SWE-agent stopped before submitting a final output",
                "retryable": False,
            }
        return output

    async def stop(self) -> None:
        self.__init__()


if __name__ == "__main__":
    main()
