#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic Claude Code control-protocol fixture."""

import json
import os
import sys


SESSION_ID = "11111111-1111-4111-8111-111111111111"
# One reply per user message; later messages get "success". The failure replies
# mirror the result objects Claude Code 2.1.294 writes on each path.
TURNS = json.loads(os.environ.get("MOCK_CLAUDE_CLI_TURNS", "[]"))
# Claude Code's zero-filled empty usage, which its session-crash result reports.
EMPTY_USAGE = {
    "output_tokens_details": {"thinking_tokens": 0},
    "input_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "output_tokens": 0,
    "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
    "service_tier": "standard",
    "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
    "inference_geo": "",
    "iterations": [],
    "speed": "standard",
    "fallback_credit": None,
}
TURN_USAGE = {
    "input_tokens": 12,
    "cache_creation_input_tokens": 400,
    "cache_read_input_tokens": 3000,
    "output_tokens": 25,
}


def emit(message):
    print(json.dumps(message), flush=True)


MESSAGE_INDEX = 0


def emit_stream(event):
    emit(
        {
            "type": "stream_event",
            "event": event,
            "uuid": "fixture",
            "session_id": SESSION_ID,
            "parent_tool_use_id": None,
        }
    )


def emit_answer(usage):
    global MESSAGE_INDEX
    MESSAGE_INDEX += 1
    initial = {**usage, "output_tokens": 1}
    message = {
        "id": f"msg-{MESSAGE_INDEX}",
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": "claude-test-model",
        "usage": initial,
    }
    emit_stream({"type": "message_start", "message": message})
    emit(
        {
            "type": "assistant",
            "message": {
                **message,
                "content": [{"type": "text", "text": "mock Claude response"}],
            },
            "session_id": SESSION_ID,
        }
    )
    emit_stream(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": usage["output_tokens"]},
        }
    )
    emit_stream({"type": "message_stop"})


def emit_fallback_result(errors, **fields):
    """Write the result Claude Code builds outside its agent loop."""

    emit(
        {
            "type": "result",
            "subtype": "error_during_execution",
            "duration_ms": 0,
            "duration_api_ms": 0,
            "is_error": True,
            "num_turns": 0,
            "stop_reason": None,
            "session_id": SESSION_ID,
            "total_cost_usd": 0,
            "usage": EMPTY_USAGE,
            "modelUsage": {},
            "permission_denials": [],
            "errors": errors,
            **fields,
        }
    )


SESSION_CRASH_ERRORS = [
    "TypeError: mock crash",
    "[session_crash] error_name=TypeError api_error_status=none cause_name=none",
]

with open(os.environ["MOCK_CLAUDE_CLI_LOG"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\n")

if env_log := os.environ.get("MOCK_CLAUDE_CLI_ENV_LOG"):
    with open(env_log, "a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "ANTHROPIC_BASE_URL": os.environ.get("ANTHROPIC_BASE_URL"),
                    "NEMO_RELAY_GATEWAY_URL": os.environ.get("NEMO_RELAY_GATEWAY_URL"),
                }
            )
            + "\n"
        )

for line in sys.stdin:
    message = json.loads(line)
    if message.get("type") == "control_request":
        print(
            json.dumps(
                {
                    "type": "control_response",
                    "response": {
                        "subtype": "success",
                        "request_id": message["request_id"],
                        "response": {"commands": [], "output_style": "default"},
                    },
                }
            ),
            flush=True,
        )
        continue
    if message.get("type") != "user":
        continue
    turn = TURNS.pop(0) if TURNS else "success"
    if turn == "crash_before_answer":
        emit_fallback_result(SESSION_CRASH_ERRORS)
        continue
    if turn == "crash_after_answer":
        emit_answer(TURN_USAGE)
        emit_fallback_result(SESSION_CRASH_ERRORS)
        continue
    if turn == "internal_error":
        # The headless error handler adds result_index; it has no crash marker.
        emit_fallback_result(
            ["An internal error ended the session (TypeError: mock crash)."],
            result_index=0,
        )
        continue
    if turn == "sandbox_unavailable":
        emit_fallback_result(
            [
                "Sandbox required but unavailable: mock reason. Set "
                "sandbox.failIfUnavailable=false to allow unsandboxed execution."
            ]
        )
        continue
    if turn == "max_turns":
        emit_answer(TURN_USAGE)
        emit(
            {
                "type": "result",
                "subtype": "error_max_turns",
                "duration_ms": 10,
                "duration_api_ms": 8,
                "is_error": True,
                "num_turns": 1,
                "stop_reason": "tool_use",
                "session_id": SESSION_ID,
                "total_cost_usd": 0.002,
                "usage": TURN_USAGE,
                "modelUsage": {},
                "permission_denials": [],
                "terminal_reason": "max_turns",
                "errors": ["Reached maximum number of turns (1)"],
            }
        )
        continue
    if turn != "success":
        raise SystemExit(f"unknown mock turn: {turn}")
    emit_answer({"input_tokens": 1, "output_tokens": 2})
    print(
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "duration_ms": 10,
                "duration_api_ms": 8,
                "is_error": False,
                "num_turns": 1,
                "session_id": SESSION_ID,
                "total_cost_usd": 0.001,
                "usage": {"input_tokens": 1, "output_tokens": 2},
                "result": "mock Claude response",
            }
        ),
        flush=True,
    )
