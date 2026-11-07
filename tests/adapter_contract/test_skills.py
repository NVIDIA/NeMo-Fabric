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


def test_entry_with_skill_md_is_one_skill(tmp_path: Path):
    review = _skill(tmp_path / "review", "code-review")

    assert resolve_skills([review], tmp_path) == [
        Skill(name="code-review", path=review.resolve(), field="skills.paths[0]")
    ]


def test_skill_name_falls_back_to_its_directory(tmp_path: Path):
    review = _skill(tmp_path / "review")

    assert [skill.name for skill in resolve_skills([review], tmp_path)] == ["review"]


def test_quoted_frontmatter_name_is_unquoted(tmp_path: Path):
    review = tmp_path / "review"
    review.mkdir()
    (review / "SKILL.md").write_text(
        '---\r\nname: "code-review"\r\n---\r\n', encoding="utf-8"
    )

    assert [skill.name for skill in resolve_skills([review], tmp_path)] == [
        "code-review"
    ]


def test_folder_loads_skills_directly_inside_it(tmp_path: Path):
    shared = tmp_path / "shared"
    one = _skill(shared / "one", "one")
    two = _skill(shared / "two", "two")
    (shared / "notes.md").write_text("Not a skill.\n", encoding="utf-8")
    (shared / "docs").mkdir()
    (shared / "docs" / "README.md").write_text("Not a skill.\n", encoding="utf-8")
    _skill(shared / "group" / "nested", "nested")

    assert resolve_skills([shared], tmp_path) == [
        Skill(name="one", path=one.resolve(), field="skills.paths[0]"),
        Skill(name="two", path=two.resolve(), field="skills.paths[0]"),
    ]


def test_folder_and_separate_skill_mix(tmp_path: Path):
    shared = tmp_path / "shared"
    _skill(shared / "one", "one")
    _skill(shared / "two", "two")
    separate = _skill(tmp_path / "mounted" / "three", "three")

    skills = resolve_skills([shared, separate], tmp_path)

    assert [(skill.name, skill.field) for skill in skills] == [
        ("one", "skills.paths[0]"),
        ("two", "skills.paths[0]"),
        ("three", "skills.paths[1]"),
    ]


def test_existing_entry_without_skills_loads_nothing(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()

    assert resolve_skills([empty], tmp_path) == []


def test_relative_entry_resolves_against_base_dir(tmp_path: Path):
    review = _skill(tmp_path / "skills" / "review", "review")

    assert resolve_skills(["skills/review"], tmp_path) == [
        Skill(name="review", path=review.resolve(), field="skills.paths[0]")
    ]


def test_missing_entry_fails_naming_it(tmp_path: Path):
    review = _skill(tmp_path / "review", "review")
    missing = tmp_path / "missing"

    with pytest.raises(SkillPathError) as caught:
        resolve_skills([review, missing], tmp_path)

    assert caught.value.reason == "not_found"
    assert caught.value.field == "skills.paths[1]"
    assert caught.value.paths == (missing,)
    assert "skills.paths[1]" in str(caught.value)
    assert str(missing) in str(caught.value)


def test_file_entry_fails(tmp_path: Path):
    entry = tmp_path / "SKILL.md"
    entry.write_text("# Not a folder\n", encoding="utf-8")

    with pytest.raises(SkillPathError) as caught:
        resolve_skills([entry], tmp_path)

    assert caught.value.reason == "invalid"
    assert caught.value.field == "skills.paths[0]"
    assert str(entry) in str(caught.value)


def test_duplicate_skill_names_fail_naming_both_paths(tmp_path: Path):
    shared = tmp_path / "shared"
    first = _skill(shared / "review", "review")
    second = _skill(tmp_path / "mounted" / "review-copy", "review")

    with pytest.raises(SkillPathError) as caught:
        resolve_skills([shared, second], tmp_path)

    assert caught.value.reason == "duplicate"
    assert caught.value.field == "skills.paths[1]"
    assert caught.value.paths == (first.resolve(), second.resolve())
    assert str(first.resolve()) in str(caught.value)
    assert str(second.resolve()) in str(caught.value)


def test_same_skill_listed_twice_loads_once(tmp_path: Path):
    shared = tmp_path / "shared"
    review = _skill(shared / "review", "review")

    assert resolve_skills([shared, review], tmp_path) == [
        Skill(name="review", path=review.resolve(), field="skills.paths[0]")
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


@_POSIX_PERMISSIONS
def test_unreadable_entry_fails_naming_it(tmp_path: Path, lock: Callable[[Path], Path]):
    review = _skill(tmp_path / "review", "review")
    entry = tmp_path / "locked"
    entry.mkdir()
    lock(entry)

    with pytest.raises(SkillPathError) as caught:
        resolve_skills([review, entry], tmp_path)

    assert caught.value.reason == "invalid"
    assert caught.value.field == "skills.paths[1]"
    assert caught.value.paths == (entry,)
    assert "skills.paths[1]" in str(caught.value)
    assert str(entry) in str(caught.value)


@_POSIX_PERMISSIONS
def test_unreadable_skill_file_fails_naming_it(
    tmp_path: Path, lock: Callable[[Path], Path]
):
    shared = tmp_path / "shared"
    review = _skill(shared / "review", "review")
    lock(review / "SKILL.md")

    with pytest.raises(SkillPathError) as caught:
        resolve_skills([shared], tmp_path)

    assert caught.value.reason == "invalid"
    assert caught.value.field == "skills.paths[0]"
    assert caught.value.paths == (review.resolve() / "SKILL.md",)
    assert str(review.resolve() / "SKILL.md") in str(caught.value)


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("---\nname: code-review # main skill\n---\n", "code-review"),
        ("---\nname: 'code-review' # quoted\n---\n", "code-review"),
        ("---\nname: code#review\n---\n", "code#review"),
        ("--- \nname: code-review\n---\n", "code-review"),
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


def test_skill_md_folder_does_not_make_a_skill(tmp_path: Path):
    shared = tmp_path / "shared"
    (shared / "SKILL.md").mkdir(parents=True)
    one = _skill(shared / "one", "one")

    assert resolve_skills([shared], tmp_path) == [
        Skill(name="one", path=one.resolve(), field="skills.paths[0]")
    ]


@pytest.mark.parametrize("spelling", ["review/", "./review", "other/../review", "link"])
def test_same_skill_spelled_differently_loads_once(tmp_path: Path, spelling: str):
    review = _skill(tmp_path / "review", "review")
    (tmp_path / "other").mkdir()
    if spelling == "link":
        (tmp_path / "link").symlink_to(review, target_is_directory=True)

    assert resolve_skills(["review", spelling], tmp_path) == [
        Skill(name="review", path=review.resolve(), field="skills.paths[0]")
    ]


def test_same_skill_in_another_case_loads_once(tmp_path: Path):
    review = _skill(tmp_path / "Review")
    if not (tmp_path / "review").exists():
        pytest.skip("needs a case-insensitive file system")

    assert resolve_skills(["Review", "review"], tmp_path) == [
        Skill(name="Review", path=review.resolve(), field="skills.paths[0]")
    ]
