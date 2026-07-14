# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render NeMo Relay hook documents for supported coding agents."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Literal


RelayHookAgent = Literal["claude", "codex"]

CLAUDE_RELAY_HOOK_EVENTS = (
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
)
CODEX_RELAY_HOOK_EVENTS = (
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
)
RELAY_TOOL_HOOK_EVENTS = frozenset(
    {
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "PermissionRequest",
    }
)


def render_relay_hooks(
    agent: RelayHookAgent,
    executable: Path,
) -> dict[str, Any]:
    """Return the native hook document for one Relay-supported coding agent."""

    if agent not in ("claude", "codex"):
        raise ValueError(f"unsupported NeMo Relay hook agent {agent!r}")

    command = f"{shlex.quote(str(executable))} hook-forward {agent}"
    hooks: dict[str, list[dict[str, Any]]] = {}
    events = (
        CLAUDE_RELAY_HOOK_EVENTS
        if agent == "claude"
        else CODEX_RELAY_HOOK_EVENTS
    )
    for event in events:
        group: dict[str, Any] = {
            "hooks": [
                {
                    "type": "command",
                    "command": command,
                    "timeout": 30,
                }
            ]
        }
        if event in RELAY_TOOL_HOOK_EVENTS:
            group["matcher"] = "*"
        hooks[event] = [group]
    return {"hooks": hooks}
