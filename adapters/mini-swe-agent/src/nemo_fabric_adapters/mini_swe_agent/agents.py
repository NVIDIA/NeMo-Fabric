# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mini-SWE-agent subclasses used by the NVIDIA NeMo Fabric adapter."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import InterruptAgentFlow
from minisweagent.exceptions import Submitted


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _submitted_output(error: Submitted) -> dict[str, Any]:
    content = error.messages[0].get("content", "") if error.messages else ""
    return {
        "output": f"COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n{content}",
        "returncode": 0,
        "exception_info": "",
    }


class RetainingDefaultAgent(DefaultAgent):
    """Preserve mini-SWE-agent history across ordered Fabric invocations.

    Before a later run, remove the prior terminal exit message, append the new
    task as a user message, and suppress ``DefaultAgent`` history
    initialization. History remains in memory without truncation until the
    runtime stops.
    """

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
            self.add_messages(self.model.format_message(role="user", content=task))
            self._retain_messages = True
        return super().run(task, **kwargs)

    def _execute_action(self, action: dict[str, Any]) -> dict[str, Any]:
        return self.env.execute(action)

    def execute_actions(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        actions = message.get("extra", {}).get("actions", [])
        outputs = []
        for action in actions:
            try:
                outputs.append(self._execute_action(action))
            except Submitted as error:
                outputs.append(_submitted_output(error))
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


class RelayRetainingDefaultAgent(RetainingDefaultAgent):
    """Emit Relay scopes around mini-SWE-agent steps, model calls, and actions."""

    def __init__(self, *args: Any, relay_model_name: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._relay_model_name = relay_model_name
        self._relay_invocation_handle: Any = None
        self._relay_step_handle: Any = None
        self._relay_errors: list[str] = []

    def begin_relay_invocation(self, handle: Any) -> None:
        self._relay_invocation_handle = handle
        self._relay_step_handle = None
        self._relay_errors = []

    def end_relay_invocation(self) -> list[str]:
        errors = list(self._relay_errors)
        self._relay_invocation_handle = None
        self._relay_step_handle = None
        self._relay_errors = []
        return errors

    def _record_relay_error(self, error: BaseException) -> None:
        self._relay_errors.append(_error_text(error))

    def _parent_handle(self) -> Any:
        return self._relay_step_handle or self._relay_invocation_handle

    def step(self) -> list[dict[str, Any]]:
        parent = self._relay_invocation_handle
        if parent is None:
            return super().step()

        from nemo_relay import ScopeType
        from nemo_relay import scope

        try:
            handle = scope.push(
                "mini-swe-agent.step",
                ScopeType.Function,
                handle=parent,
                metadata={"step_index": self.n_calls + 1},
            )
        except Exception as error:
            self._record_relay_error(error)
            return super().step()

        self._relay_step_handle = handle
        outcome: dict[str, Any] = {"status": "succeeded"}
        try:
            return super().step()
        except InterruptAgentFlow as error:
            outcome = {
                "status": "interrupted",
                "interrupt_type": type(error).__name__,
            }
            raise
        except Exception as error:
            outcome = {"status": "error", "error_type": type(error).__name__}
            raise
        finally:
            self._relay_step_handle = None
            try:
                scope.pop(handle, output=outcome)
            except Exception as error:
                self._record_relay_error(error)

    def query(self) -> dict[str, Any]:
        parent = self._parent_handle()
        if parent is None:
            return super().query()

        from nemo_relay import LLMRequest
        from nemo_relay import llm

        request_messages = [
            {key: value for key, value in message.items() if key != "extra"}
            for message in self.messages
        ]
        try:
            handle = llm.call(
                "mini-swe-agent.model",
                LLMRequest(
                    {},
                    {
                        "model": self._relay_model_name,
                        "messages": request_messages,
                    },
                ),
                handle=parent,
                model_name=self._relay_model_name,
            )
        except Exception as error:
            self._record_relay_error(error)
            return super().query()

        response: Any = None
        metadata: dict[str, Any] = {"status": "succeeded"}
        try:
            message = super().query()
            response = _relay_response(message)
            return message
        except Exception as error:
            response = _relay_exception_response(error)
            metadata = {"status": "error", "error_type": type(error).__name__}
            raise
        finally:
            try:
                response_codec = None
                if isinstance(response, dict) and isinstance(
                    response.get("choices"), list
                ):
                    from nemo_relay.codecs import OpenAIChatCodec

                    response_codec = OpenAIChatCodec()
                llm.call_end(
                    handle,
                    response,
                    metadata=metadata,
                    response_codec=response_codec,
                )
            except Exception as error:
                self._record_relay_error(error)

    def _execute_action(self, action: dict[str, Any]) -> dict[str, Any]:
        parent = self._parent_handle()
        if parent is None:
            return super()._execute_action(action)

        from nemo_relay import tools

        tool_call_id = action.get("tool_call_id")
        arguments = {
            key: value for key, value in action.items() if key != "tool_call_id"
        }
        try:
            handle = tools.call(
                "bash",
                arguments,
                handle=parent,
                tool_call_id=(tool_call_id if isinstance(tool_call_id, str) else None),
            )
        except Exception as error:
            self._record_relay_error(error)
            return super()._execute_action(action)

        try:
            output = super()._execute_action(action)
        except Submitted as error:
            self._end_tool_call(
                handle, _submitted_output(error), metadata={"status": "submitted"}
            )
            raise
        except Exception as error:
            self._end_tool_call(
                handle,
                None,
                metadata={"status": "error", "error_type": type(error).__name__},
            )
            raise
        self._end_tool_call(handle, output, metadata={"status": "succeeded"})
        return output

    def _end_tool_call(
        self,
        handle: Any,
        output: Any,
        *,
        metadata: dict[str, Any],
    ) -> None:
        from nemo_relay import tools

        try:
            tools.call_end(handle, output, metadata=metadata)
        except Exception as error:
            self._record_relay_error(error)


def _relay_response(message: dict[str, Any]) -> Any:
    extra = message.get("extra", {})
    response = extra.get("response")
    if not isinstance(response, dict):
        return {key: value for key, value in message.items() if key != "extra"}

    response = deepcopy(response)
    cost = extra.get("cost")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        usage = response.setdefault("usage", {})
        if isinstance(usage, dict):
            usage.setdefault("cost", float(cost))
    return response


def _relay_exception_response(error: BaseException) -> Any:
    messages = getattr(error, "messages", None)
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        response = messages[0].get("extra", {}).get("response")
        if (
            isinstance(response, (dict, list, str, int, float, bool))
            or response is None
        ):
            return response
    return None
