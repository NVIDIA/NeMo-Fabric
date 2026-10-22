# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accounting tests include sanitized receipts from CLI 2.1.294 + SDK 0.2.165."""

import json
from pathlib import Path

from claude_agent_sdk import ResultMessage, StreamEvent, SystemMessage
from claude_agent_sdk._internal.message_parser import parse_message
import pytest

from nemo_fabric_adapters.claude.usage import ClaudeUsageSession, substitute_result

KEY = "nemo_fabric.usage_accounting.v1"
DETAIL = "nemo_fabric.detailed_token_usage.v1"
FIXTURE = Path(__file__).parents[1] / "fixtures/claude/usage-evidence.json"


def result(**changes):
    values = dict(
        subtype="success",
        duration_ms=10,
        duration_api_ms=8,
        is_error=False,
        num_turns=1,
        session_id="fixture",
        usage={
            "input_tokens": 10,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 2,
        },
    )
    values.update(changes)
    return ResultMessage(**values)


def event(kind, **fields):
    return StreamEvent(
        uuid="fixture", session_id="fixture", event={"type": kind, **fields}
    )


def call(owner, *, identity="m1", usage=None, output=2, stop=True):
    if usage is None:
        usage = {
            "input_tokens": 10,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 1,
        }
    owner.observe(
        event(
            "message_start",
            message={"id": identity, "model": "claude-test", "usage": usage},
        )
    )
    if output is not None:
        owner.observe(
            event(
                "message_delta",
                delta={"stop_reason": "end_turn"},
                usage={"output_tokens": output},
            )
        )
    if stop:
        owner.observe(event("message_stop"))


def test_literal_zero_and_omission_remain_independent():
    owner = ClaudeUsageSession("anthropic").start()
    call(owner, usage={"input_tokens": 0, "output_tokens": 1}, output=0)
    owner.observe(
        result(
            usage={
                "input_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "output_tokens": 0,
            }
        )
    )
    usage = owner.finish()
    assert usage.input_tokens == 0
    assert usage.cached_input_tokens is None
    assert usage.output_tokens == 0
    assert usage.cost_usd is None
    assert usage.extensions[DETAIL]["input_tokens_include_cache"] is False


def test_final_stream_output_overrides_initial_placeholder_without_double_count():
    owner = ClaudeUsageSession().start()
    call(owner)
    owner.observe(result())
    usage = owner.finish()
    assert (usage.input_tokens, usage.output_tokens, usage.cached_input_tokens) == (
        10,
        2,
        0,
    )
    assert usage.extensions[KEY]["models"][0]["observed"]["output_tokens"] == 2
    assert usage.extensions[KEY]["models"][0].get("provider") is None


def test_failed_terminal_keeps_settled_usage():
    owner = ClaudeUsageSession().start()
    call(owner)
    owner.observe(
        result(subtype="error_max_turns", is_error=True, terminal_reason="max_turns")
    )
    assert owner.finish().output_tokens == 2


def test_missing_terminal_preserves_observations_and_freezes():
    owner = ClaudeUsageSession().start()
    call(owner)
    usage = owner.finish()
    assert usage.input_tokens is None
    assert usage.extensions[KEY]["observed"]["input_tokens"] == 10
    owner.observe(result())
    call(owner, identity="late")
    assert owner.finish() is usage
    assert usage.extensions[KEY]["observed"]["input_tokens"] == 10


def test_retry_is_actual_missing_call_evidence():
    owner = ClaudeUsageSession().start()
    call(owner)
    owner.observe(SystemMessage(subtype="api_retry", data={"error_status": 429}))
    owner.observe(result())
    usage = owner.finish()
    assert usage.input_tokens is None
    assert usage.extensions[KEY]["observed"]["output_tokens"] == 2


def test_no_terminal_delta_cannot_certify_initial_output():
    owner = ClaudeUsageSession().start()
    call(owner, output=None)
    owner.observe(result())
    usage = owner.finish()
    assert usage.output_tokens is None
    assert usage.extensions[KEY]["observed"]["output_tokens"] == 1


def test_injected_result_rolls_back_preceding_root_frames_without_user_echo():
    owner = ClaudeUsageSession().start()
    call(owner, identity="previous-background")
    owner.observe(result(origin={"kind": "task-notification"}))
    call(
        owner,
        identity="owned",
        usage={
            "input_tokens": 30,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        output=4,
    )
    owner.observe(
        result(
            usage={
                "input_tokens": 30,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "output_tokens": 4,
            }
        )
    )
    usage = owner.finish()
    assert (usage.input_tokens, usage.output_tokens) == (30, 4)


def test_substitute_crash_cannot_report_zero_or_erase_observed_calls():
    owner = ClaudeUsageSession().start()
    call(owner)
    crash = result(
        subtype="error_during_execution",
        is_error=True,
        num_turns=0,
        duration_ms=0,
        duration_api_ms=0,
        terminal_reason=None,
        usage={"input_tokens": 0, "output_tokens": 0},
    )
    assert substitute_result(crash)
    owner.observe(crash)
    usage = owner.finish()
    assert usage.input_tokens is None
    assert usage.extensions[KEY]["observed"]["input_tokens"] == 10


def test_pre_query_refusal_proves_no_model_work():
    owner = ClaudeUsageSession().start()
    owner.observe(
        result(
            subtype="error_during_execution",
            is_error=True,
            num_turns=0,
            duration_ms=0,
            duration_api_ms=0,
            errors=["Sandbox required but unavailable: fixture"],
        )
    )
    usage = owner.finish()
    assert (usage.input_tokens, usage.output_tokens, usage.cached_input_tokens) == (
        0,
        0,
        0,
    )


@pytest.mark.parametrize(
    "scenario",
    [
        "normal",
        "foreground",
        "background",
        "compact",
        "background-followup",
        "background-overlap",
    ],
)
def test_pinned_cli_localhost_receipts(scenario):
    receipt = json.loads(FIXTURE.read_text())[scenario]
    assert (receipt["sdk_version"], receipt["cli_version"]) == ("0.2.165", "2.1.294")
    session = ClaudeUsageSession()
    owners = {}
    for item in receipt["events"]:
        owner = owners.setdefault(item["invocation"], session.start())
        raw = item["message"]
        if raw["type"] == "system":
            message = SystemMessage(subtype=raw["subtype"], data=raw)
        else:
            message = parse_message(raw)
        owner.observe(message)
        if isinstance(message, ResultMessage) and (
            message.origin is None or message.origin.get("kind") == "human"
        ):
            owner.finish()
    usages = {invocation: owner.finish() for invocation, owner in owners.items()}
    if scenario == "normal":
        assert (usages[0].input_tokens, usages[0].output_tokens) == (10, 2)
    elif scenario == "foreground":
        assert usages[0].input_tokens is None
        assert usages[0].extensions[KEY]["observed"]["input_tokens"] == 60
        assert usages[0].extensions[KEY]["observed"]["output_tokens"] == 6
    elif scenario == "background":
        assert usages[0].input_tokens is None
        assert usages[0].extensions[KEY]["observed"]["input_tokens"] == 60
        assert usages[0].extensions[KEY]["observed"]["output_tokens"] == 5
    elif scenario == "compact":
        assert usages[0].input_tokens == 10
        assert usages[1].input_tokens is None
        assert any(
            gap["reason"] == "scope_unverified"
            for gap in usages[1].extensions[KEY]["gaps"]
        )
    else:
        assert usages[0].input_tokens is None
        assert usages[0].extensions[KEY]["observed"]["input_tokens"] == 40
        expected_input = 50 if scenario == "background-followup" else 40
        assert (usages[1].input_tokens, usages[1].output_tokens) == (expected_input, 2)
        assert usages[1].cached_input_tokens == 0
        assert usages[1].extensions[KEY]["observed"]["input_tokens"] == expected_input


@pytest.mark.parametrize("invalid", [-1, True, 1.5, "10", 1 << 64])
def test_invalid_counter_never_becomes_known(invalid):
    owner = ClaudeUsageSession().start()
    call(
        owner,
        usage={
            "input_tokens": invalid,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    )
    owner.observe(result())
    usage = owner.finish()
    assert usage.input_tokens is None
    assert usage.output_tokens == 2
    assert usage.cached_input_tokens == 0


def test_synthetic_abandoned_stream_stop_cannot_certify_output():
    owner = ClaudeUsageSession().start()
    call(owner, output=None, stop=False)
    owner.observe(
        event("message_delta", delta={"stop_reason": None}, usage={"output_tokens": 1})
    )
    owner.observe(event("message_stop"))
    owner.observe(
        result(
            usage={
                "input_tokens": 10,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "output_tokens": 1,
            }
        )
    )
    usage = owner.finish()
    assert usage.output_tokens is None
    assert usage.extensions[KEY]["observed"]["output_tokens"] == 1


def test_unused_agent_capability_does_not_create_accounting_gap():
    owner = ClaudeUsageSession().start()
    owner.observe(SystemMessage(subtype="init", data={"tools": ["Agent", "Task"]}))
    call(owner)
    owner.observe(result())
    assert owner.finish().input_tokens == 10


def test_prior_invocations_parented_child_message_is_excluded():
    from claude_agent_sdk import AssistantMessage

    session = ClaudeUsageSession()
    previous = session.start()
    call(previous)
    previous.observe(result())
    snapshot = previous.finish()
    owner = session.start()
    owner.observe(
        AssistantMessage(
            content=[],
            model="claude-test",
            parent_tool_use_id="old-child",
            message_id="late-child",
            usage={"input_tokens": 500, "output_tokens": 20},
        )
    )
    call(owner)
    owner.observe(result())
    assert owner.finish().input_tokens == snapshot.input_tokens == 10


def test_rejected_agent_tool_request_without_dispatch_does_not_create_gap():
    from claude_agent_sdk import (
        AssistantMessage,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    owner = ClaudeUsageSession().start()
    call(owner)
    owner.observe(
        AssistantMessage(
            content=[ToolUseBlock(id="blocked", name="Agent", input={})],
            model="claude-test",
        )
    )
    owner.observe(
        UserMessage(
            content=[
                ToolResultBlock(
                    tool_use_id="blocked", content="permission denied", is_error=True
                )
            ]
        )
    )
    owner.observe(result())
    assert owner.finish().input_tokens == 10
