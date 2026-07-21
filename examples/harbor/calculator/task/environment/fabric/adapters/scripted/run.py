#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Deterministic adapter used by the credential-free Harbor example."""

import json
import sys
from pathlib import Path
from typing import Any


def response(operation: str, *, output: Any = None) -> None:
    print(
        json.dumps(
            {
                "operation": operation,
                "outcome": {"status": "succeeded", "output": output},
            }
        ),
        flush=True,
    )


def invoke(payload: dict[str, Any]) -> dict[str, Any]:
    calculator = Path("/app/calculator.py")
    calculator.write_text(
        "def add(a, b):\n"
        "    return a + b\n\n\n"
        "def multiply(a, b):\n"
        "    return a * b\n",
        encoding="utf-8",
    )
    return {
        "harness": "scripted",
        "response": "Fixed multiply(a, b) in /app/calculator.py",
        "request_id": payload["request"]["request_id"],
    }


def main() -> None:
    runtime_id = None
    for line in sys.stdin:
        message = json.loads(line)
        operation = message["operation"]
        payload = message["payload"]
        if operation == "start":
            runtime_id = payload["runtime_context"]["runtime_id"]
            response("start")
        elif operation == "invoke":
            if payload["runtime_context"]["runtime_id"] != runtime_id:
                raise RuntimeError("invoke does not match the active runtime")
            response("invoke", output=invoke(payload))
        elif operation == "stop":
            if payload["runtime_id"] != runtime_id:
                raise RuntimeError("stop does not match the active runtime")
            response("stop")
            break


if __name__ == "__main__":
    main()
