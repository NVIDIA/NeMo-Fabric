# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Fabric's generated hook policy with the released Relay CLI."""

import json
from pathlib import Path
import shlex
import shutil
import socket
import subprocess

import pytest

from nemo_fabric_adapters.common.relay_hooks import render_relay_hooks
from nemo_fabric_adapters.common.relay_gateway import relay_cli_contract


@pytest.mark.parametrize("agent", ["claude", "codex"])
@pytest.mark.parametrize(
    ("event", "expected_status"),
    [("PreToolUse", 2), ("PostToolUse", 0), ("SessionStart", 0)],
)
def test_generated_hook_delivery_failure_policy(
    tmp_path, agent, event, expected_status
):
    executable = shutil.which("nemo-relay")
    if executable is None:
        pytest.skip("NeMo Relay CLI is not installed")
    contract = relay_cli_contract(Path(executable))
    if event == "PreToolUse" and contract.version < (0, 11, 0):
        expected_status = 1
    # Reserve an unused loopback port, then close it so delivery fails promptly.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    hook = render_relay_hooks(agent, Path(executable))["hooks"][event][0]["hooks"][0]
    command = shlex.split(hook["command"])
    command += ["--forward-only", "--gateway-url", f"http://127.0.0.1:{port}"]
    completed = subprocess.run(
        command,
        input=json.dumps(
            {
                "hook_event_name": event,
                "session_id": "fabric-hook-policy-test",
                "cwd": str(tmp_path),
                "tool_name": "Read",
                "tool_input": {"file_path": "README.md"},
                "tool_response": "test result",
            }
        ),
        capture_output=True,
        text=True,
        timeout=hook["timeout"],
        check=False,
    )

    assert completed.returncode == expected_status
