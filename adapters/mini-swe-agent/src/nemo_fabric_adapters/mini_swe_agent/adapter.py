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
SYSTEM_TEMPLATE = "{{system_instruction}}"
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
        self._agent_kwargs: dict[str, Any] = {}
        self._system_instruction = ""

    async def start(self, payload: dict[str, Any]) -> None:
        config: contract.AgentConfig = payload["config"]
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
        self._system_instruction = (
            config.instructions.system.content
            if config.instructions and config.instructions.system
            else "You are a helpful software engineering assistant."
        )
        self._agent_kwargs = {
            "system_template": SYSTEM_TEMPLATE,
            "instance_template": INSTANCE_TEMPLATE,
            "step_limit": config.runtime.max_turns if config.runtime else 0,
            "cost_limit": 0,
        }

    async def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._model is None or self._environment is None:
            raise lifecycle.LifecycleError(
                "mini_swe_agent_not_started", "mini-SWE-agent runtime is not started"
            )
        raw_task = common_utils.request_payload(payload).get("input", "")
        task = raw_task if isinstance(raw_task, str) else json.dumps(raw_task)
        from minisweagent.agents.default import DefaultAgent

        agent = DefaultAgent(self._model, self._environment, **self._agent_kwargs)
        result = await asyncio.to_thread(
            agent.run, task, system_instruction=self._system_instruction
        )
        failed = result.get("exit_status") != "Submitted"
        output: dict[str, Any] = {
            "failed": failed,
            "output": result.get("submission", ""),
            "usage": {"api_calls": agent.n_calls},
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
