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
        from minisweagent.exceptions import Submitted
        from minisweagent.models.litellm_model import LitellmModel
        from minisweagent.agents.default import DefaultAgent

        class RetainingDefaultAgent(DefaultAgent):
            _retain_messages = _skip_initial_messages = False

            @property
            def messages(self) -> list[dict[str, Any]]:
                return self._messages

            @messages.setter
            def messages(self, messages: list[dict[str, Any]]) -> None:
                if self._retain_messages and not messages:
                    self._retain_messages = False
                    self._skip_initial_messages = True
                else:
                    self._messages = messages

            def add_messages(self, *messages: dict[str, Any]) -> list[dict[str, Any]]:
                if self._skip_initial_messages:
                    self._skip_initial_messages = False
                    return []
                return super().add_messages(*messages)

            def run(self, task: str = "", **kwargs: Any) -> dict[str, Any]:
                if self.messages:
                    self.n_calls = self.cost = self.n_consecutive_format_errors = 0
                    self.messages.pop()
                    self.add_messages(
                        self.model.format_message(role="user", content=task)
                    )
                    self._retain_messages = True
                return super().run(task, **kwargs)

            def execute_actions(self, message: dict[str, Any]) -> list[dict[str, Any]]:
                actions = message.get("extra", {}).get("actions", [])
                outputs = []
                for action in actions:
                    try:
                        outputs.append(self.env.execute(action))
                    except Submitted as error:
                        outputs.append(
                            {
                                "output": (
                                    "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n"
                                    f"{error.messages[0]['content']}"
                                ),
                                "returncode": 0,
                                "exception_info": "",
                            }
                        )
                        self.add_messages(
                            *self.model.format_observation_messages(
                                message, outputs, self.get_template_vars()
                            )
                        )
                        raise
                return self.add_messages(
                    *self.model.format_observation_messages(
                        message, outputs, self.get_template_vars()
                    )
                )

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
        self._agent = RetainingDefaultAgent(
            self._model,
            self._environment,
            system_template=SYSTEM_TEMPLATE,
            instance_template=INSTANCE_TEMPLATE,
            step_limit=config.runtime.max_turns if config.runtime else 0,
        )

    async def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw_task = common_utils.request_payload(payload).get("input", "")
        task = raw_task if isinstance(raw_task, str) else json.dumps(raw_task)
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
