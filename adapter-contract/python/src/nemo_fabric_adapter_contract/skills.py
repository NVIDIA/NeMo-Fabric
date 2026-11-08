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

_NAME = re.compile(r"name:(?:[ \t]+(.*?))?[ \t]*")
_COMMENT = re.compile(r"[ \t]+#")


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

    A folder inside an entry that cannot be read is not a skill. The same
    folder reached through several entries loads once.

    Raises:
        SkillPathError: If an entry does not exist, is not a folder, or cannot
            be read, if a skill's ``SKILL.md`` cannot be read, or if two skills
            share a name.
    """

    skills: list[Skill] = []
    by_name: dict[str, Skill] = {}
    folders: set[tuple[int, int] | Path] = set()
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
        for directory in _skill_directories(entry, field, configured):
            folder = _folder_identity(directory)
            if folder in folders:
                continue
            folders.add(folder)
            skill = Skill(
                name=_skill_name(directory, field), path=directory, field=field
            )
            existing = by_name.get(skill.name)
            if existing is not None:
                raise SkillPathError(
                    "duplicate",
                    f"NeMo Fabric skill {skill.name!r} is configured twice: "
                    f"{existing.path} ({existing.field}) and {skill.path} ({field})",
                    field=field,
                    paths=(existing.path, skill.path),
                )
            by_name[skill.name] = skill
            skills.append(skill)
    return skills


def _skill_directories(entry: Path, field: str, configured: Path) -> list[Path]:
    try:
        if (entry / SKILL_FILE).is_file():
            return [entry]
        children = sorted(entry.iterdir(), key=lambda child: child.name)
    except OSError:
        raise SkillPathError(
            "invalid",
            f"NeMo Fabric skill path {field} cannot be read: {configured}",
            field=field,
            paths=(configured,),
        ) from None
    return [child.resolve() for child in children if _is_skill(child)]


def _is_skill(child: Path) -> bool:
    # A folder that cannot be read, such as a volume's lost+found, is not a skill.
    try:
        return child.is_dir() and (child / SKILL_FILE).is_file()
    except OSError:
        return False


def _folder_identity(directory: Path) -> tuple[int, int] | Path:
    # Compare folders by identity so that spellings resolve() keeps apart, such
    # as a different case on a case-insensitive file system, still match. An
    # inode of 0 identifies nothing, so fall back to the resolved path.
    stat = directory.stat()
    return (stat.st_dev, stat.st_ino) if stat.st_ino else directory


def _skill_name(directory: Path, field: str) -> str:
    skill_file = directory / SKILL_FILE
    try:
        with skill_file.open(encoding="utf-8-sig", errors="replace") as lines:
            name = _front_matter_name(lines)
    except OSError:
        raise SkillPathError(
            "invalid",
            f"NeMo Fabric skill path {field} has a {SKILL_FILE} that cannot be "
            f"read: {skill_file}",
            field=field,
            paths=(skill_file,),
        ) from None
    return name or directory.name


def _front_matter_name(lines: Iterable[str]) -> str | None:
    """Return the ``name`` scalar of a ``---`` front matter block, if any."""

    lines = iter(lines)
    if next(lines, "").rstrip() != "---":
        return None
    name: str | None = None
    for line in lines:
        line = line.rstrip()
        if line == "---":
            return name
        match = _NAME.fullmatch(line)
        if match and not name:
            name = _scalar(match.group(1) or "")
    # Without its closing line the block is not front matter.
    return None


def _scalar(value: str) -> str:
    if value[:1] in ("'", '"'):
        end = value.find(value[0], 1)
        rest = value[end + 1 :].strip() if end > 0 else ""
        if end > 0 and (not rest or rest.startswith("#")):
            return value[1:end]
        return value
    if value.startswith("#"):
        return ""
    return _COMMENT.split(value, maxsplit=1)[0]
