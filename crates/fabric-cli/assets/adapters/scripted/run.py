#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic process adapter for credential-free CLI experiments."""

import json
import sys
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
            request = payload["request"]
            response(
                "invoke",
                output={
                    "response": request.get("input"),
                    "request_id": request["request_id"],
                },
            )
        elif operation == "stop":
            if payload["runtime_id"] != runtime_id:
                raise RuntimeError("stop does not match the active runtime")
            response("stop")
            break


if __name__ == "__main__":
    main()
