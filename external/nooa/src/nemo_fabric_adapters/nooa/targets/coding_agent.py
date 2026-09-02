# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registered ``CodingAgent`` target construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nemo_fabric_adapters.nooa import InteractiveAgentBuildContext
from nemo_fabric_adapters.nooa.model_support import selected_model


def _register_skills(agent: Any, paths: tuple[Path, ...]) -> None:
    if not paths:
        return
    from nooa.skill import TextSkill

    names: list[str] = []
    for path in paths:
        skill_file = path / "SKILL.md"
        if not path.is_dir() or not skill_file.is_file():
            raise ValueError(
                "CodingAgent skill paths must be directories containing SKILL.md"
            )
        skill = TextSkill(path=path)
        name = f"cmd.{skill.id}"
        agent.skills.register(name, skill)
        names.append(name)
    agent.skills.activate(names)


def create_agent(context: InteractiveAgentBuildContext) -> Any:
    """Build one host-neutral OO Agents coding agent."""

    from nooa import Context
    from nooa_cli.coding import CodingAgent

    libs_dir = (
        context.artifact_root / "nooa-libs"
        if context.artifact_root is not None
        else context.base_dir / ".nooa" / "libs"
    )
    agent = CodingAgent(
        llm=selected_model(context.models, target_name="CodingAgent"),
        cwd=context.workspace,
        libs_dir=libs_dir,
    )
    if context.system_instruction is not None:
        agent.context["fabric_system_instruction"] = Context(
            context.system_instruction,
            prefix=True,
        )
    _register_skills(agent, context.skill_paths)
    return agent
