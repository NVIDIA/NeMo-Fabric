# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import cast

import pytest

import nemo_fabric_adapters.common.relay_hooks as relay_hooks


EXPECTED_EVENTS = (
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


@pytest.mark.parametrize("agent", ["claude", "codex"])
def test_render_relay_hooks_matches_relay_agent_contract(agent):
    executable = Path("/opt/nvidia relay/bin/nemo-relay")

    hooks = relay_hooks.render_relay_hooks(agent, executable)["hooks"]

    assert tuple(hooks) == EXPECTED_EVENTS
    assert hooks["SessionStart"] == [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": f"'{executable}' hook-forward {agent}",
                    "timeout": 30,
                }
            ]
        }
    ]
    assert {
        event for event, groups in hooks.items() if groups[0].get("matcher") == "*"
    } == {
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "PermissionRequest",
    }


def test_render_relay_hooks_rejects_unsupported_agent():
    with pytest.raises(ValueError, match="unsupported NeMo Relay hook agent"):
        relay_hooks.render_relay_hooks(
            cast(relay_hooks.RelayHookAgent, "other"),
            Path("nemo-relay"),
        )
