# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emulate the OpenShell provider protocol without a gateway or model.

The fixture receives host provider requests over stdio, executes lifecycle
operations with a separate real task SDK, and returns provider responses. It
also invokes the artifact reader and can inject malformed command results.
"""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

from nemo_fabric import Fabric, FabricConfig, RunRequest


async def main():
    fabric = Fabric()
    runtime = None
    fault_injected = False
    try:
        for line in sys.stdin:
            frame = json.loads(line)
            operation = frame["operation"]
            if operation in ("prepare", "attach"):
                environment = frame["environment"]
                connection = environment["connection"]
                output = {
                    "workspace": environment["workspace"],
                    "artifacts": environment["artifacts"],
                    "connection": {
                        **connection,
                        "sandbox_id": connection.get("sandbox_id", "fixture-created-id"),
                        "sandbox_name": connection.get(
                            "sandbox_name",
                            environment["settings"].get("sandbox_name", "echo"),
                        ),
                    },
                }
            elif operation == "runtime_command":
                command = frame["command"]
                if command["operation"] == "start":
                    task_root = os.environ.get("FABRIC_TEST_TASK_ROOT")
                    if task_root is None:
                        task_root = str(Path(frame["environment"]["workspace"]).parent)
                    runtime = await fabric.start_runtime(
                        FabricConfig.model_validate(command["config"]),
                        base_dir=Path(task_root),
                    )
                    value = runtime.handle.to_mapping()
                elif command["operation"] == "invoke":
                    value = (
                        await runtime.invoke(
                            request=RunRequest.model_validate(command["request"])
                        )
                    ).to_mapping()
                else:
                    await runtime.stop()
                    value = None
                fault = os.environ.get("FABRIC_TEST_PROVIDER_FAULT")
                if not fault_injected and (
                    (fault == "start" and command["operation"] == "start")
                    or (fault == "invoke" and command["operation"] == "invoke")
                ):
                    fault_injected = True
                    await runtime.stop()
                    value = {"PRIVATE": "malformed SDK value"}
                elif (
                    not fault_injected
                    and fault == "stop"
                    and command["operation"] == "stop"
                ):
                    fault_injected = True
                    value = {"stopped": True}
                output = {"status": "returned", "value": value}
            elif operation == "collect_artifacts":
                reader = os.environ["FABRIC_TEST_ARTIFACT_READER"]
                result = subprocess.run(
                    [sys.executable, reader],
                    input=json.dumps(
                        {
                            "root": os.environ.get(
                                "FABRIC_TEST_TASK_ARTIFACTS_ROOT",
                                frame["environment"]["artifacts"],
                            ),
                            "paths": frame["paths"],
                        }
                    ),
                    text=True,
                    capture_output=True,
                    timeout=5,
                    check=True,
                )
                output = json.loads(result.stdout)["files"]
            elif operation == "release":
                output = (
                    {"released": True, "detached": False}
                    if frame["environment"]["ownership"] == "fabric_owned"
                    else {"released": False, "detached": True}
                )
            else:
                raise ValueError("unsupported fixture operation")
            with Path(os.environ["FABRIC_TEST_PROVIDER_LOG"]).open("a") as stream:
                stream.write(json.dumps({"frame": frame, "output": output}) + "\n")
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
    finally:
        if runtime is not None:
            await runtime.stop()


asyncio.run(main())
