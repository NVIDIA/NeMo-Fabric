# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replace gateway exec with real fabric-agent-ctl subprocesses for SDK tests."""

import json
import os
from pathlib import Path
import subprocess
import sys


for line in sys.stdin:
    frame = json.loads(line)
    operation = frame["operation"]
    if operation in ("prepare", "attach"):
        environment = frame["environment"]
        output = {
            "workspace": environment["workspace"],
            "artifacts": environment["artifacts"],
            "connection": environment["connection"],
        }
    elif operation == "release":
        output = {"released": False, "detached": True}
    elif operation == "runtime_command":
        command = frame["command"]
        name = command["operation"]
        arguments = [sys.executable, "-m", "nemo_fabric.runtime_server", name]
        payload = None
        if name == "check":
            arguments.extend(["--timeout-seconds", str(command["timeout_seconds"])])
            if "runtime_id" in command:
                arguments.extend(
                    [
                        "--agent",
                        command["agent"],
                        "--runtime-id",
                        command["runtime_id"],
                        f"--{command['level']}",
                    ]
                )
        else:
            arguments.extend(["--agent", command["agent"]])
            if "runtime_id" in command:
                arguments.extend(["--runtime-id", command["runtime_id"]])
            field = (
                "config" if name == "start" else "request" if name == "invoke" else None
            )
            if field is not None:
                arguments.extend([f"--{field}", "-"])
                payload = json.dumps(command[field])
        if path := os.environ.get("FABRIC_CHECK_COMMAND_LOG"):
            routing = {
                key: value
                for key, value in command.items()
                if key
                in {"operation", "agent", "runtime_id", "level", "timeout_seconds"}
            }
            with Path(path).open("a") as stream:
                stream.write(json.dumps(routing) + "\n")
        result = subprocess.run(
            arguments,
            input=payload,
            text=True,
            capture_output=True,
            timeout=frame["timeout_seconds"] + 2,
        )
        response = json.loads(result.stdout)
        if (
            response["status"] == "succeeded"
            or name == "check"
            and response["result"] is not None
        ):
            output = {"status": "returned", "value": response["result"]}
        else:
            error = response["error"]
            output = {
                "status": "rejected",
                "error": {key: error[key] for key in ("code", "stage", "effects")},
            }
    else:
        raise ValueError("unsupported test provider operation")
    print(
        json.dumps(
            {
                "protocol_version": frame["protocol_version"],
                "request_id": frame["request_id"],
                "status": "succeeded",
                "output": output,
            }
        ),
        flush=True,
    )
