# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render NeMo Relay hook documents for supported coding agents."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from sys import platform
from typing import Any, Literal


RELAY_TOOL_CHECK_PATH = Path(__file__).with_name("relay_tool_check.py").resolve()

RelayHookAgent = Literal["claude", "codex"]

RELAY_HOOK_EVENTS: dict[RelayHookAgent, tuple[str, ...]] = {
    "claude": (
        "SessionStart",
        "UserPromptSubmit",
        "UserPromptExpansion",
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "SubagentStart",
        "SubagentStop",
        "Notification",
        "Stop",
        "PreCompact",
        "PostCompact",
        "SessionEnd",
    ),
    "codex": (
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "SubagentStart",
        "SubagentStop",
        "Stop",
        "PreCompact",
        "PostCompact",
    ),
}
RELAY_TOOL_HOOK_EVENTS: dict[RelayHookAgent, frozenset[str]] = {
    "claude": frozenset(
        {
            "PreToolUse",
            "PostToolUse",
            "PostToolUseFailure",
        }
    ),
    "codex": frozenset(
        {
            "PreToolUse",
            "PostToolUse",
        }
    ),
}


def render_relay_hooks(
    agent: RelayHookAgent,
    executable: Path,
) -> dict[str, Any]:
    """Return the native hook document for one Relay-supported coding agent."""

    if agent not in ("claude", "codex"):
        raise ValueError(f"unsupported NeMo Relay hook agent {agent!r}")

    if platform == "win32":
        executable_path = str(executable).replace("\\", "/")
        executable_arg = (
            f'"{executable_path}"'
            if agent == "claude"
            else subprocess.list2cmdline([executable_path])
        )
    else:
        executable_arg = shlex.quote(str(executable))
    command = f"{executable_arg} hook-forward {agent}"
    python_arg = (
        subprocess.list2cmdline([sys.executable.replace("\\", "/")])
        if platform == "win32"
        else shlex.quote(sys.executable)
    )
    helper_arg = (
        subprocess.list2cmdline([str(RELAY_TOOL_CHECK_PATH).replace("\\", "/")])
        if platform == "win32"
        else shlex.quote(str(RELAY_TOOL_CHECK_PATH))
    )
    hooks: dict[str, list[dict[str, Any]]] = {}
    for event in RELAY_HOOK_EVENTS[agent]:
        # Relay reports tool-check delivery failures; the host handles process failures.
        failure_policy = "--fail-closed" if event == "PreToolUse" else "--fail-open"
        hook_command = f"{command} {failure_policy}"
        if agent == "codex" and event == "PreToolUse":
            # Keep workspace files and PYTHONPATH out of the approval check.
            hook_command = f"{python_arg} -I {helper_arg} {hook_command}"
        group: dict[str, Any] = {
            "hooks": [
                {
                    "type": "command",
                    "command": hook_command,
                    "timeout": 45,
                }
            ]
        }
        if agent == "claude" and event == "PreToolUse":
            group["hooks"][0]["onFailure"] = "block"
        if event in RELAY_TOOL_HOOK_EVENTS[agent]:
            group["matcher"] = "*"
        hooks[event] = [group]
    return {"hooks": hooks}
