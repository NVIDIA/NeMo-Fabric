# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render NeMo Relay hook documents for supported coding agents."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Literal


RelayHookAgent = Literal["claude", "codex"]

RELAY_HOOK_EVENTS: dict[RelayHookAgent, tuple[str, ...]] = {
    "claude": (
        "SessionStart",
        "UserPromptSubmit",
        "UserPromptExpansion",
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "PermissionRequest",
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
        "PermissionRequest",
        "SubagentStart",
        "SubagentStop",
        "Stop",
        "PreCompact",
        "PostCompact",
    ),
}
RELAY_TOOL_HOOK_EVENTS = {
    "claude": frozenset(
        {
            "PreToolUse",
            "PostToolUse",
            "PostToolUseFailure",
            "PermissionRequest",
        }
    ),
    "codex": frozenset(
        {
            "PreToolUse",
            "PostToolUse",
            "PermissionRequest",
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

    command = f"{shlex.quote(str(executable))} hook-forward {agent}"
    hooks: dict[str, list[dict[str, Any]]] = {}
    for event in RELAY_HOOK_EVENTS[agent]:
        group: dict[str, Any] = {
            "hooks": [
                {
                    "type": "command",
                    "command": command,
                    "timeout": 30,
                }
            ]
        }
        if event in RELAY_TOOL_HOOK_EVENTS[agent]:
            group["matcher"] = "*"
        hooks[event] = [group]
    return {"hooks": hooks}
