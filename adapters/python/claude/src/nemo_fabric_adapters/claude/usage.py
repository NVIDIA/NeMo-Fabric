# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Presence-preserving Claude stream observations owned by one SDK query.

Claude's terminal usage zero-fills omissions and modelUsage is process-cumulative.
Neither can recover literal provider field presence. This collector keeps the
raw per-message counters and delegates reduction and normalization to Fabric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import AssistantMessage, ConversationResetMessage
from claude_agent_sdk import Message, ResultMessage, StreamEvent, SystemMessage
from claude_agent_sdk import ToolResultBlock, ToolUseBlock, UserMessage
from nemo_fabric_adapter_contract.models import AgentUsage, GapReason, ReportedUsage
from nemo_fabric_adapter_contract.usage import UsageObservation, usage_from_observations

_FIELDS = (
    "input_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "output_tokens",
)
_PRE_QUERY_REFUSALS = ("Sandbox required but unavailable", "not_claimed:")


def _count(value: Any) -> int | None:
    return value if type(value) is int and 0 <= value < 1 << 64 else None


def pre_query_refusal(result: ResultMessage) -> bool:
    """The pinned CLI emits these errors before entering its model loop."""
    return bool(result.errors) and all(
        isinstance(error, str) and error.startswith(_PRE_QUERY_REFUSALS)
        for error in result.errors
    )


def substitute_result(result: ResultMessage) -> bool:
    """Identify the CLI's zero-filled crash/internal-error substitute result."""
    return (
        result.subtype == "error_during_execution"
        and result.is_error is True
        and result.terminal_reason is None
        and all(
            type(value) is int and value == 0
            for value in (result.num_turns, result.duration_ms, result.duration_api_ms)
        )
        and not pre_query_refusal(result)
    )


def injected_turn(message: UserMessage | ResultMessage) -> bool:
    """SDK 0.2.165 labels task/channel/peer turns with a nonhuman origin."""
    origin = message.origin
    return origin is not None and (
        not isinstance(origin, dict) or origin.get("kind") != "human"
    )


@dataclass
class _Call:
    model: str | None
    counts: dict[str, int] = field(default_factory=dict)
    stopped: bool = False
    final_output: bool = False
    raw_message_only: bool = False


class ClaudeUsageSession:
    """Keep source route and session-wide background-estimate ambiguity."""

    def __init__(self, provider: str | None = None) -> None:
        self.provider = provider
        self.background_seen = False

    def start(self) -> ClaudeInvocationUsage:
        return ClaudeInvocationUsage(self)


class ClaudeInvocationUsage:
    """Freeze one invocation's source units at return, excluding injected turns."""

    def __init__(self, session: ClaudeUsageSession) -> None:
        self._session = session
        self._calls: dict[tuple[str | None, str], _Call] = {}
        self._current: dict[str | None, str] = {}
        self._children: set[str] = set()
        self._child_summaries: dict[str, ReportedUsage] = {}
        self._scope_gap = False
        self._retry_gap = False
        self._injected = False
        self._result: ResultMessage | None = None
        self._snapshot: AgentUsage | None = None

    def observe(self, message: Message) -> None:
        """Read public source fields; retain no prompt, response or tool text."""
        if self._snapshot is not None:
            return
        if isinstance(message, UserMessage):
            if message.parent_tool_use_id is None and message.origin is not None:
                self._injected = injected_turn(message)
            if not self._injected and isinstance(message.content, list):
                for block in message.content:
                    if (
                        isinstance(block, ToolResultBlock)
                        and block.tool_use_id in self._children
                    ):
                        raw = message.tool_use_result
                        raw = raw if isinstance(raw, dict) else {}
                        usage = raw.get("usage")
                        # A rejected permission/tool request need not start a
                        # child. Native task identity/usage establishes execution.
                        if raw.get("agentId") or isinstance(usage, dict):
                            self._scope_gap = True
                        if isinstance(usage, dict):
                            # This aggregate zero-fills omitted provider fields.
                            # Positive values remain observations, never complete
                            # counts, and its zeros cannot establish presence.
                            fields = {
                                target: value
                                for source, target in (
                                    ("input_tokens", "input_tokens"),
                                    ("cache_read_input_tokens", "cached_input_tokens"),
                                    (
                                        "cache_creation_input_tokens",
                                        "cache_write_input_tokens",
                                    ),
                                    ("output_tokens", "output_tokens"),
                                )
                                if (value := _count(usage.get(source))) is not None
                                and value > 0
                            }
                            if "input_tokens" in fields:
                                fields["input_tokens_include_cache"] = False
                            if "output_tokens" in fields:
                                fields["output_tokens_include_reasoning"] = True
                            self._child_summaries[block.tool_use_id] = (
                                ReportedUsage.from_mapping(fields)
                            )
            return
        if isinstance(message, ResultMessage):
            if injected_turn(message):
                # The SDK can omit the user echo. Terminal origin still proves
                # these preceding root frames belong to an injected turn.
                self._calls.clear()
                self._current.clear()
                self._children.clear()
                self._child_summaries.clear()
                self._scope_gap = False
                self._retry_gap = False
            else:
                self._result = message
            self._injected = False
            return
        if self._injected:
            return
        parent = getattr(message, "parent_tool_use_id", None)
        if parent is not None and parent not in self._children:
            return
        if isinstance(message, ConversationResetMessage):
            self._scope_gap = True
        elif isinstance(message, SystemMessage):
            if message.subtype in {
                "api_retry",
                "model_refusal_fallback",
                "model_refusal_no_fallback",
            }:
                self._retry_gap = True
            elif message.subtype == "compact_boundary":
                # Compaction's model request is not a raw root StreamEvent.
                self._scope_gap = True
            elif message.subtype == "task_started":
                tool_id = message.data.get("tool_use_id")
                if tool_id in self._children:
                    self._scope_gap = True
                    if message.data.get("is_backgrounded") is True:
                        self._session.background_seen = True
        elif isinstance(message, AssistantMessage):
            if (
                parent is not None
                and message.message_id is not None
                and message.model != "<synthetic>"
            ):
                key = (parent, message.message_id)
                call = self._calls.get(key)
                if call is None:
                    model = (
                        message.model
                        if isinstance(message.model, str) and message.model.strip()
                        else None
                    )
                    call = _Call(model, raw_message_only=True)
                    self._calls[key] = call
                if call.raw_message_only:
                    self._update(call, message.usage)
            for block in message.content:
                if isinstance(block, ToolUseBlock) and block.name in {"Agent", "Task"}:
                    self._children.add(block.id)
        elif isinstance(message, StreamEvent):
            self._stream(parent, message.event)

    def _stream(self, parent: str | None, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "message_start":
            data = event.get("message")
            if not isinstance(data, dict):
                return
            identity, model = data.get("id"), data.get("model")
            if not isinstance(identity, str) or not identity or model == "<synthetic>":
                self._current.pop(parent, None)
                return
            model = model if isinstance(model, str) and model.strip() else None
            key = (parent, identity)
            call = self._calls.setdefault(key, _Call(model))
            if call.model != model:
                self._scope_gap = True
            self._current[parent] = identity
            call.stopped = False
            call.final_output = False
            call.raw_message_only = False
            self._update(call, data.get("usage"))
        elif parent in self._current:
            call = self._calls[(parent, self._current[parent])]
            if kind == "message_delta":
                usage = event.get("usage")
                self._update(call, usage)
                if isinstance(usage, dict) and "output_tokens" in usage:
                    delta = event.get("delta")
                    delta = delta if isinstance(delta, dict) else {}
                    # The CLI also synthesizes a stop pair for an abandoned
                    # stream. Its null stop reason does not certify final usage.
                    call.final_output = (
                        _count(usage["output_tokens"]) is not None
                        and isinstance(delta.get("stop_reason"), str)
                        and bool(delta["stop_reason"])
                    )
            elif kind == "message_stop":
                call.stopped = True
                self._current.pop(parent)
            elif kind == "error":
                self._retry_gap = True

    @staticmethod
    def _update(call: _Call, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        for key in _FIELDS:
            if key in usage:
                value = _count(usage[key])
                if value is None:
                    call.counts.pop(key, None)
                else:
                    call.counts[key] = value

    def finish(self) -> AgentUsage:
        """Return the same detached snapshot after later stream callbacks."""
        if self._snapshot is not None:
            return self._snapshot
        records = []
        for (parent, _), call in self._calls.items():
            if parent in self._child_summaries:
                continue
            raw = call.counts
            reported = ReportedUsage(
                input_tokens=raw.get("input_tokens"),
                cached_input_tokens=raw.get("cache_read_input_tokens"),
                cache_write_input_tokens=raw.get("cache_creation_input_tokens"),
                input_tokens_include_cache=False if "input_tokens" in raw else None,
                output_tokens=raw.get("output_tokens"),
                output_tokens_include_reasoning=True
                if "output_tokens" in raw
                else None,
            )
            records.append(
                UsageObservation(
                    reported=reported,
                    model=call.model,
                    provider=self._session.provider,
                    is_model_call=True,
                    tokens_aligned=True,
                    gap=GapReason.SCOPE_UNVERIFIED
                    if call.raw_message_only
                    else GapReason.UNFINISHED_WORK
                    if not call.stopped or not call.final_output
                    else None,
                )
            )
        records.extend(
            UsageObservation(reported=reported, gap=GapReason.SCOPE_UNVERIFIED)
            for reported in self._child_summaries.values()
        )
        if self._result is not None and not substitute_result(self._result):
            # Only the root stream is in the CLI's per-turn terminal usage.
            root = [call for (parent, _), call in self._calls.items() if parent is None]
            terminal = self._result.usage
            terminal = terminal if isinstance(terminal, dict) else {}
            for field_name in _FIELDS:
                values = [call.counts.get(field_name) for call in root]
                value = _count(terminal.get(field_name))
                if value is not None and values and all(v is not None for v in values):
                    if sum(values) != value:
                        self._scope_gap = True
        elif self._result is None or substitute_result(self._result):
            records.append(
                UsageObservation(
                    reported=ReportedUsage(), gap=GapReason.UNFINISHED_WORK
                )
            )
        if self._result is not None and pre_query_refusal(self._result) and not records:
            records.append(
                UsageObservation(
                    reported=ReportedUsage(
                        input_tokens=0,
                        cached_input_tokens=0,
                        cache_write_input_tokens=0,
                        input_tokens_include_cache=False,
                        output_tokens=0,
                        output_tokens_include_reasoning=True,
                        total_tokens=0,
                    )
                )
            )
        if self._scope_gap or self._retry_gap:
            records.append(
                UsageObservation(
                    reported=ReportedUsage(),
                    gap=(
                        GapReason.SCOPE_UNVERIFIED
                        if self._scope_gap
                        else GapReason.MISSING_USAGE
                    ),
                )
            )
        self._snapshot = usage_from_observations(records)
        return self._snapshot
