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

SYSTEM_TEMPLATE = "{{system_instruction}}"
INSTANCE_TEMPLATE = """Solve this task in the current workspace:

{{task}}

Use the bash tool. When complete, run
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` by itself, then provide final output.
"""


def _continue_agent(agent: Any, task: str) -> dict[str, Any]:
    from minisweagent.exceptions import FormatError, InterruptAgentFlow

    agent.n_calls = agent.cost = 0
    format_error_limit = agent.config.model_dump().get(
        "max_consecutive_format_errors", 0
    )
    consecutive_format_errors = 0
    agent.messages[-1]["role"] = "assistant"
    agent.add_messages(agent.model.format_message(role="user", content=task))
    while True:
        try:
            agent.step()
            consecutive_format_errors = 0
        except FormatError as error:
            agent.cost += error.messages[0].get("extra", {}).get("cost", 0.0)
            consecutive_format_errors += 1
            if 0 < format_error_limit <= consecutive_format_errors:
                agent.add_messages(
                    *error.messages,
                    {
                        "role": "exit",
                        "content": "RepeatedFormatError",
                        "extra": {
                            "exit_status": "RepeatedFormatError",
                            "submission": "",
                        },
                    },
                )
            else:
                agent.add_messages(*error.messages)
        except InterruptAgentFlow as error:
            agent.add_messages(*error.messages)
        except Exception as error:
            agent.handle_uncaught_exception(error)
            raise
        if agent.messages[-1].get("role") == "exit":
            return agent.messages[-1].get("extra", {})


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

    async def start(self, payload: dict[str, Any]) -> None:
        config: contract.AgentConfig = payload["config"]
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.litellm_model import LitellmModel

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
        self._system_instruction = (
            config.instructions.system.content
            if config.instructions and config.instructions.system
            else ""
        )
        from minisweagent.agents.default import DefaultAgent

        self._agent = DefaultAgent(
            self._model,
            self._environment,
            system_template=SYSTEM_TEMPLATE,
            instance_template=INSTANCE_TEMPLATE,
            step_limit=config.runtime.max_turns if config.runtime else 0,
        )

    async def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw_task = common_utils.request_payload(payload).get("input", "")
        task = raw_task if isinstance(raw_task, str) else json.dumps(raw_task)
        if self._agent.messages:
            result = await asyncio.to_thread(_continue_agent, self._agent, task)
        else:
            result = await asyncio.to_thread(
                self._agent.run, task, system_instruction=self._system_instruction
            )
        failed = result.get("exit_status") != "Submitted"
        output: dict[str, Any] = {
            "failed": failed,
            "output": result.get("submission", ""),
            "usage": {"api_calls": self._agent.n_calls},
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


def main() -> None:
    lifecycle.serve(
        MiniSweAgentRuntime, config_loader=contract.AgentConfig.from_mapping
    )


if __name__ == "__main__":
    main()
