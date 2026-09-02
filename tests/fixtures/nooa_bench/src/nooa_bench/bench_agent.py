# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic BenchAgent contract fixture for Harbor-to-Fabric tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class _Shell:
    async def close(self) -> None:
        return None


class BenchAgent:
    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.shell = _Shell()
        self.event_manager = object()

    async def _run_evaluation(self, task_input: dict[str, Any]) -> dict[str, Any]:
        workspace = Path(task_input["working_dir"])
        instruction = task_input["user_message"]
        system_instruction = task_input.get("instructions", "")
        (workspace / "bench-agent-result.txt").write_text(
            f"{instruction}\n{system_instruction}\n",
            encoding="utf-8",
        )
        self.shell = _Shell()
        return {
            "response": "test -f bench-agent-result.txt",
            "success": True,
            "result": {
                "solution_description": "Created the requested task artifact.",
                "evidence": "bench-agent-result.txt exists in the task workspace",
                "command_to_verify": "test -f bench-agent-result.txt",
            },
        }
