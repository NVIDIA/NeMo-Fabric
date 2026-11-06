# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve ``skills.paths`` entries in the task environment.

Each entry names one skill or a folder of skills:

1. An entry that contains ``SKILL.md`` is one skill.
2. Otherwise each folder directly inside the entry that contains ``SKILL.md``
   is a skill. Other files and folders are ignored.
3. An entry that exists but holds no skills loads nothing.
4. An entry that does not exist, or is not a folder, fails.
5. Two skills with the same name fail.

A skill's name is the ``name`` field of its ``SKILL.md`` front matter, or its
folder name when the front matter has none.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SKILL_FILE = "SKILL.md"

SkillPathFailure = Literal["not_found", "invalid", "duplicate"]

_NAME = re.compile(r"name:\s*(.*?)\s*")


@dataclass(frozen=True)
class Skill:
    """One skill found through ``skills.paths``."""

    name: str
    """Name from the skill's front matter, or its folder name."""
    path: Path
    """Resolved folder that contains the skill's ``SKILL.md``."""
    field: str
    """The ``skills.paths`` entry that supplied the skill, such as ``skills.paths[0]``."""


class SkillPathError(ValueError):
    """A ``skills.paths`` entry that cannot be loaded."""

    def __init__(
        self,
        reason: SkillPathFailure,
        message: str,
        *,
        field: str,
        paths: tuple[Path, ...],
    ) -> None:
        super().__init__(message)
        self.reason: SkillPathFailure = reason
        self.field = field
        self.paths = paths


def resolve_skills(
    paths: Iterable[str | os.PathLike[str]], base_dir: str | os.PathLike[str]
) -> list[Skill]:
    """Return the skills named by ``paths``, resolving relative entries against ``base_dir``.

    Raises:
        SkillPathError: If an entry does not exist or is not a folder, or two
            skills share a name.
    """

    skills: list[Skill] = []
    by_name: dict[str, Skill] = {}
    for index, value in enumerate(paths):
        field = f"skills.paths[{index}]"
        configured = Path(base_dir) / value
        try:
            entry = configured.resolve(strict=True)
        except (OSError, RuntimeError):
            raise SkillPathError(
                "not_found",
                f"NeMo Fabric skill path {field} does not exist: {configured}",
                field=field,
                paths=(configured,),
            ) from None
        if not entry.is_dir():
            raise SkillPathError(
                "invalid",
                f"NeMo Fabric skill path {field} is not a folder: {configured}",
                field=field,
                paths=(configured,),
            )
        for directory in _skill_directories(entry):
            skill = Skill(name=_skill_name(directory), path=directory, field=field)
            existing = by_name.get(skill.name)
            if existing is None:
                by_name[skill.name] = skill
                skills.append(skill)
            elif existing.path != skill.path:
                raise SkillPathError(
                    "duplicate",
                    f"NeMo Fabric skill {skill.name!r} is configured twice: "
                    f"{existing.path} ({existing.field}) and {skill.path} ({field})",
                    field=field,
                    paths=(existing.path, skill.path),
                )
    return skills


def _skill_directories(entry: Path) -> list[Path]:
    if (entry / SKILL_FILE).is_file():
        return [entry]
    return [
        child.resolve()
        for child in sorted(entry.iterdir(), key=lambda child: child.name)
        if child.is_dir() and (child / SKILL_FILE).is_file()
    ]


def _skill_name(directory: Path) -> str:
    text = (directory / SKILL_FILE).read_text(encoding="utf-8-sig", errors="replace")
    front_matter = text.splitlines()
    if front_matter[:1] == ["---"]:
        for line in front_matter[1:]:
            if line.strip() == "---":
                break
            match = _NAME.fullmatch(line)
            if match:
                name = match.group(1)
                if len(name) >= 2 and name[0] == name[-1] and name[0] in "\"'":
                    name = name[1:-1]
                if name:
                    return name
    return directory.name
