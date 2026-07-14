# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render NeMo Relay hook documents for supported coding agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Literal


RelayHookAgent = Literal["claude", "codex"]

# NeMo Relay currently uses this union for both Claude Code and Codex. Keep the
# snapshot centralized until Relay exposes its hook renderer as a public API.
RELAY_HOOK_EVENTS = (
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

    command = f"{executable} hook-forward {agent}"
    hooks: dict[str, list[dict[str, Any]]] = {}
    for event in RELAY_HOOK_EVENTS:
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
