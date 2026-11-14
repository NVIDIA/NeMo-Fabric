# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scaffolding shared by the Hermes skill tests. Needs no Hermes installation."""

from __future__ import annotations

from pathlib import Path


def write_skill(
    folder: Path, name: str | None = None, *, text: str | None = None
) -> Path:
    """Write ``folder/SKILL.md`` and return the resolved folder.

    The file has a ``name`` line when ``name`` is given, and a description.
    ``text`` replaces the whole file.
    """
    folder.mkdir(parents=True, exist_ok=True)
    if text is None:
        name_line = f"name: {name}\n" if name else ""
        text = f"---\n{name_line}description: test skill\n---\nBody\n"
    (folder / "SKILL.md").write_text(text, encoding="utf-8")
    return folder.resolve()


def loaded_skill(name: str, path: Path, entry: int = 0) -> dict[str, str]:
    """An ``output.skills.loaded`` item for a skill from ``skills.directories``."""
    return {
        "name": name,
        "path": str(path),
        "entry": f"skills.directories[{entry}]",
        "origin": "external",
    }


def skill_copy(origin: str, path: Path, entry: int | None = None) -> dict[str, str]:
    """A ``copies`` item of an ``output.skills.duplicates`` record."""
    item = {"origin": origin, "path": str(path)}
    if entry is not None:
        item["entry"] = f"skills.directories[{entry}]"
    return item


def skipped_skill(path: Path, reason: str, entry: int = 0) -> dict[str, str]:
    """An ``output.skills.skipped`` item."""
    return {
        "path": str(path),
        "entry": f"skills.directories[{entry}]",
        "reason": reason,
    }
