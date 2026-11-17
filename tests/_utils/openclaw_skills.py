# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skill folders, configurations, and expected reports shared by the OpenClaw skill tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import RuntimeContext

COPY_AT = "Copy at {folder}."
"""The default description, which names the skill's folder."""

REFUSAL_REASONS = {
    "malformed": "invalid_frontmatter",
    "empty": "empty_frontmatter_field:description",
}
"""The skip reason of each kind of skill :func:`refused_skill` writes."""


def write_skill(
    directory: Path, name: str | None = None, description: str | None = COPY_AT
) -> Path:
    """Write ``SKILL.md`` into ``directory``; ``None`` leaves a field out.

    ``{folder}`` in ``description`` becomes the folder name.
    """

    directory.mkdir(parents=True, exist_ok=True)
    lines = ["---"]
    if name is not None:
        lines.append(f"name: {name}")
    if description is not None:
        lines.append(f"description: {description.replace('{folder}', directory.name)}")
    lines += ["---", "", "Body.", ""]
    (directory / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")
    return directory


def refused_skill(directory: Path, name: str, problem: str) -> Path:
    """Write a skill OpenClaw refuses: ``malformed`` YAML or an ``empty`` description."""

    directory.mkdir(parents=True, exist_ok=True)
    lines = {
        "malformed": [f"name: {name}", "description: Broken.", "metadata: [unclosed"],
        "empty": [f"name: {name}", 'description: ""'],
    }[problem]
    (directory / "SKILL.md").write_text(
        "\n".join(["---", *lines, "---", "", "Body.", ""]), encoding="utf-8"
    )
    return directory


def agent_config(
    skills: dict[str, Any],
    settings: dict[str, Any] | None = None,
    command: Path | None = None,
) -> AgentConfig:
    """Return an OpenClaw agent configuration with ``skills``."""

    payload: dict[str, Any] = {
        "models": {"default": {"provider": "test", "model": "fabric-echo"}},
        "skills": skills,
    }
    settings = dict(settings or {})
    if command is not None:
        settings["openclaw_command"] = str(command)
    if settings:
        payload["harness"] = {"settings": settings}
    return AgentConfig.from_mapping(payload)


def runtime_context(
    workspace: Path, env: dict[str, str] | None = None
) -> RuntimeContext:
    """Return a local runtime context whose workspace is ``workspace``."""

    return RuntimeContext.from_mapping(
        {
            "runtime_id": "openclaw-runtime",
            "invocation_id": "openclaw-invocation",
            "request_id": "openclaw-request",
            "environment": {
                "environment_id": "openclaw-environment",
                "provider": "local",
                "control_location": "in_env_control",
                "workspace": str(workspace),
                "env": env or {},
                "ownership": "caller_owned",
            },
            "artifacts": {},
        }
    )


def loaded(name: str, path: Path, entry: str) -> dict[str, Any]:
    """Return the report item of a loaded configured skill."""

    return {
        "name": name,
        "path": str(path.resolve()),
        "entry": entry,
        "origin": "external",
    }


def skipped(path: Path, entry: str, reason: str) -> dict[str, Any]:
    """Return the report item of a skipped candidate."""

    return {"path": str(path.resolve()), "entry": entry, "reason": reason}


def copy(origin: str, path: Path, entry: str | None = None) -> dict[str, Any]:
    """Return one copy of a duplicate record; ``entry`` is set for configured copies."""

    item: dict[str, Any] = {"origin": origin, "path": str(path.resolve())}
    if entry is not None:
        item["entry"] = entry
    return item


def duplicate(
    name: str, outcome: str, selected: Path | None, *copies: dict[str, Any]
) -> dict[str, Any]:
    """Return a duplicate record of ``copies``."""

    item: dict[str, Any] = {"name": name, "outcome": outcome}
    if selected is not None:
        item["selected"] = str(selected.resolve())
    item["copies"] = list(copies)
    return item


@dataclass(frozen=True)
class RefusedCopies:
    """Valid skills beside copies OpenClaw refuses, under every kind of entry.

    ``review`` has a valid copy between a malformed and an empty-description
    copy, one per ``skill`` entry, so the first and the last copy are both
    refused. ``triage`` has the same three copies inside one ``folder`` entry.
    """

    review: list[Path]
    team: Path
    triage: Path

    @property
    def directories(self) -> list[dict[str, str]]:
        return [
            *({"skill": str(path)} for path in self.review),
            {"folder": str(self.team)},
        ]

    @property
    def refused(self) -> list[tuple[Path, str, str]]:
        """``(folder, entry, reason)`` of each refused copy, in report order."""

        return [
            (self.review[0], "skills.directories[0]", REFUSAL_REASONS["malformed"]),
            (self.review[2], "skills.directories[2]", REFUSAL_REASONS["empty"]),
            (
                self.team / "a-broken",
                "skills.directories[3]",
                REFUSAL_REASONS["malformed"],
            ),
            (self.team / "c-empty", "skills.directories[3]", REFUSAL_REASONS["empty"]),
        ]

    def report(self) -> dict[str, Any]:
        """Return the expected report: the valid copies load, the others are skipped."""

        return {
            "loaded": [
                loaded("review", self.review[1], "skills.directories[1]"),
                loaded("triage", self.triage, "skills.directories[3]"),
            ],
            "skipped": [skipped(*item) for item in self.refused],
        }


def refused_copies(root: Path) -> RefusedCopies:
    """Write the :class:`RefusedCopies` scenario below ``root``."""

    team = root / "team"
    refused_skill(team / "a-broken", "triage", "malformed")
    triage = write_skill(team / "b-valid", "triage", "Valid triage.")
    refused_skill(team / "c-empty", "triage", "empty")
    return RefusedCopies(
        review=[
            refused_skill(root / "broken" / "review", "review", "malformed"),
            write_skill(root / "valid" / "review", "review", "Valid review."),
            refused_skill(root / "empty" / "review", "review", "empty"),
        ],
        team=team,
        triage=triage,
    )
