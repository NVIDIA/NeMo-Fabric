# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve skills.paths entries that name a skill or a folder of skills."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import os
from pathlib import Path

import pytest

from nemo_fabric_adapter_contract.skills import Skill, SkillPathError, resolve_skills


def _skill(directory: Path, name: str | None = None) -> Path:
    directory.mkdir(parents=True)
    frontmatter = f"---\nname: {name}\ndescription: Test skill.\n---\n" if name else ""
    (directory / "SKILL.md").write_text(
        f"{frontmatter}# Test skill\n", encoding="utf-8"
    )
    return directory


def test_entries_name_skills_or_folders_of_skills(tmp_path: Path):
    shared = tmp_path / "shared"
    one = _skill(shared / "one", "one")
    two = _skill(shared / "two", "two")
    (shared / "notes.md").write_text("Not a skill.\n", encoding="utf-8")
    (shared / "SKILL.md").mkdir()
    _skill(shared / "group" / "nested", "nested")
    three = _skill(tmp_path / "mounted" / "three", "three")
    (tmp_path / "empty").mkdir()

    assert resolve_skills([shared, "mounted/three", "empty"], tmp_path) == [
        Skill(name="one", path=one.resolve(), field="skills.paths[0]"),
        Skill(name="two", path=two.resolve(), field="skills.paths[0]"),
        Skill(name="three", path=three.resolve(), field="skills.paths[1]"),
    ]


_POSIX_PERMISSIONS = pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="needs POSIX permissions enforced for a non-root user",
)


@pytest.fixture(name="lock")
def lock_fixture() -> Iterator[Callable[[Path], Path]]:
    """Remove all permissions from a path and restore them after the test."""
    locked: list[Path] = []

    def lock(path: Path) -> Path:
        path.chmod(0)
        locked.append(path)
        return path

    yield lock
    for path in reversed(locked):
        path.chmod(0o700)


def _missing(
    root: Path, _lock: Callable[[Path], Path]
) -> tuple[list[Path], tuple[Path, ...]]:
    return [_skill(root / "review"), root / "missing"], (root / "missing",)


def _file(
    root: Path, _lock: Callable[[Path], Path]
) -> tuple[list[Path], tuple[Path, ...]]:
    (root / "SKILL.md").write_text("# Not a folder\n", encoding="utf-8")
    return [_skill(root / "review"), root / "SKILL.md"], (root / "SKILL.md",)


def _duplicate(
    root: Path, _lock: Callable[[Path], Path]
) -> tuple[list[Path], tuple[Path, ...]]:
    first = _skill(root / "shared" / "review", "review")
    second = _skill(root / "mounted" / "review-copy", "review")
    return [root / "shared", second], (first.resolve(), second.resolve())


def _unreadable_entry(
    root: Path, lock: Callable[[Path], Path]
) -> tuple[list[Path], tuple[Path, ...]]:
    (root / "locked").mkdir()
    return [_skill(root / "review"), lock(root / "locked")], (root / "locked",)


def _unreadable_skill_file(
    root: Path, lock: Callable[[Path], Path]
) -> tuple[list[Path], tuple[Path, ...]]:
    _skill(root / "review")
    review = _skill(root / "shared" / "other")
    lock(review / "SKILL.md")
    return [root / "review", root / "shared"], (review.resolve() / "SKILL.md",)


@pytest.mark.parametrize(
    ("layout", "reason"),
    [
        (_missing, "not_found"),
        (_file, "invalid"),
        (_duplicate, "duplicate"),
        pytest.param(_unreadable_entry, "invalid", marks=_POSIX_PERMISSIONS),
        pytest.param(_unreadable_skill_file, "invalid", marks=_POSIX_PERMISSIONS),
    ],
)
def test_failures_name_the_entry_and_paths(
    tmp_path: Path, lock: Callable[[Path], Path], layout, reason: str
):
    entries, paths = layout(tmp_path, lock)

    with pytest.raises(SkillPathError) as caught:
        resolve_skills(entries, tmp_path)

    assert caught.value.reason == reason
    assert caught.value.field == "skills.paths[1]"
    assert caught.value.paths == paths
    assert "skills.paths[1]" in str(caught.value)
    assert all(str(path) in str(caught.value) for path in paths)


@_POSIX_PERMISSIONS
def test_unreadable_folder_inside_entry_is_ignored(
    tmp_path: Path, lock: Callable[[Path], Path]
):
    shared = tmp_path / "shared"
    one = _skill(shared / "one", "one")
    (shared / "lost+found").mkdir()
    lock(shared / "lost+found")

    assert resolve_skills([shared], tmp_path) == [
        Skill(name="one", path=one.resolve(), field="skills.paths[0]")
    ]


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ('---\r\nname: "code-review"\r\n---\r\n', "code-review"),
        ("---\nname: code-review # main skill\n---\n", "code-review"),
        ("---\nname: 'code-review' # quoted\n---\n", "code-review"),
        ("---\nname: code#review\n---\n", "code#review"),
        ("--- \nname: code-review\n---\n", "code-review"),
        ("\ufeff---\nname: code-review\n---\n", "code-review"),
        ("---\nname:\n---\n", "review"),
        (
            "---\ndescription: Front matter without an end.\nname: code-review\n",
            "review",
        ),
        ("# Review\n\nname: code-review\n", "review"),
    ],
)
def test_skill_name_follows_front_matter(tmp_path: Path, text: str, name: str):
    review = tmp_path / "review"
    review.mkdir()
    (review / "SKILL.md").write_text(text, encoding="utf-8")

    assert [skill.name for skill in resolve_skills([review], tmp_path)] == [name]


@pytest.mark.parametrize(
    "spelling",
    ["shared", "shared/review/", "./shared/review", "other/../shared/review", "link"],
)
def test_same_skill_reached_twice_loads_once(tmp_path: Path, spelling: str):
    review = _skill(tmp_path / "shared" / "review", "review")
    (tmp_path / "other").mkdir()
    (tmp_path / "link").symlink_to(review, target_is_directory=True)

    assert resolve_skills(["shared/review", spelling], tmp_path) == [
        Skill(name="review", path=review.resolve(), field="skills.paths[0]")
    ]


def test_same_skill_in_another_case_loads_once(tmp_path: Path):
    review = _skill(tmp_path / "Review")
    if not (tmp_path / "review").exists():
        pytest.skip("needs a case-insensitive file system")

    assert resolve_skills(["Review", "review"], tmp_path) == [
        Skill(name="Review", path=review.resolve(), field="skills.paths[0]")
    ]


def test_folders_without_inode_numbers_compare_by_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    shared = tmp_path / "shared"
    one = _skill(shared / "one", "one")
    two = _skill(shared / "two", "two")
    stat = Path.stat

    def without_inode(path: Path, *args, **kwargs) -> os.stat_result:
        # Some file systems report inode 0 for every folder.
        result = stat(path, *args, **kwargs)
        return os.stat_result((result.st_mode, 0, *tuple(result)[2:]))

    monkeypatch.setattr(Path, "stat", without_inode)

    assert resolve_skills([shared, one], tmp_path) == [
        Skill(name="one", path=one.resolve(), field="skills.paths[0]"),
        Skill(name="two", path=two.resolve(), field="skills.paths[0]"),
    ]
