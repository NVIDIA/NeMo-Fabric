# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registered markdown-backed ARC-AGI-3 solver target construction."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from nemo_fabric_adapters.nooa import InteractiveAgentBuildContext
from nemo_fabric_adapters.nooa import InteractiveAgentTarget
from nemo_fabric_adapters.nooa.model_support import selected_model


def _skill_file(paths: tuple[Path, ...]) -> Path | None:
    if not paths:
        return None
    if len(paths) != 1:
        raise ValueError("ARC solver accepts at most one skill path")
    path = paths[0] / "SKILL.md"
    if not path.is_file():
        raise ValueError("ARC solver skill path must contain SKILL.md")
    return path


def _run_dir(context: InteractiveAgentBuildContext) -> Path:
    root = context.artifact_root or context.workspace
    return root / "nooa-arc"


def _latest_state(agent: Any) -> dict[str, Any] | None:
    latest_state = getattr(agent, "latest_state", None)
    if callable(latest_state):
        state = latest_state()
        return state if isinstance(state, dict) else None
    return None


def _continue_arc_session(agent: Any, reason: str, _explanation: str) -> bool:
    """Keep a premature ARC ``DONE`` inside the current Fabric invocation."""

    if reason != "DONE":
        return False
    state = _latest_state(agent)
    if state is None:
        return False
    if state.get("state") in {"WIN", "GAME_OVER"}:
        return False
    note = state.get("note")
    return not (isinstance(note, str) and note.startswith("harness stopped:"))


def create_agent(context: InteractiveAgentBuildContext) -> InteractiveAgentTarget:
    """Build the OO Agents markdown-backed ARC solver for an external harness."""

    try:
        solver_module = importlib.import_module("solver_agent")
        solver_class = solver_module.MdArcSolverAgent
    except (AttributeError, ImportError) as error:
        raise RuntimeError(
            "ARC solver target requires the OO Agents ARC example on PYTHONPATH"
        ) from error

    settings = context.settings
    alias = "the game"
    agent = solver_class(
        llm=selected_model(context.models, target_name="ARC solver"),
        run_dir=_run_dir(context),
        game_id=alias,
        alias=alias,
        reflect_every=int(settings.get("reflect_every", 8)),
        visual=str(settings.get("visual", "off")),
        png_scale=int(settings.get("png_scale", 8)),
        max_actions_per_turn=int(settings.get("max_actions_per_turn", 10)),
        skill_path=_skill_file(context.skill_paths),
    )
    if context.system_instruction is not None:
        from nooa import Context

        agent.context["fabric_system_instruction"] = Context(
            context.system_instruction,
            prefix=True,
        )
    return InteractiveAgentTarget(
        agent=agent,
        continue_after=_continue_arc_session,
    )
