#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small process adapter that creates a real, temporary worker-memory spike."""

import json
import subprocess
import sys


def response(operation, output=None):
    print(
        json.dumps(
            {
                "operation": operation,
                "outcome": {"status": "succeeded", "output": output},
            }
        ),
        flush=True,
    )


def main():
    runtime_id = None
    for line in sys.stdin:
        message = json.loads(line)
        operation, payload = message["operation"], message["payload"]
        if operation == "start":
            runtime_id = payload["runtime_context"]["runtime_id"]
            response(operation)
        elif operation == "invoke":
            if payload["runtime_context"]["runtime_id"] != runtime_id:
                raise RuntimeError("invocation does not match the active runtime")
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    """
import time
memory = bytearray(128 * 1024 * 1024)
for offset in range(0, len(memory), 4096):
    memory[offset] = 1
started = time.process_time()
while time.process_time() - started < 0.3:
    pass
time.sleep(0.8)
""",
                ],
                check=True,
                timeout=10,
            )
            response(
                operation,
                {
                    "status": "succeeded",
                    "output": {"allocated_mib": 128, "worker_exited": True},
                },
            )
        elif operation == "stop":
            if payload["runtime_id"] != runtime_id:
                raise RuntimeError("stop does not match the active runtime")
            response(operation)
            break


if __name__ == "__main__":
    main()
