#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic Claude Code control-protocol fixture."""

import json
import os
import sys


SESSION_ID = "11111111-1111-4111-8111-111111111111"

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
    print(
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "mock Claude response"}],
                    "model": "claude-test-model",
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                },
                "session_id": SESSION_ID,
            }
        ),
        flush=True,
    )
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
