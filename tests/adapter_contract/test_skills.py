# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured skills contract models and the task-environment skills helper."""

from __future__ import annotations

import os
from collections.abc import Callable
from collections.abc import Iterator
from pathlib import Path

import pytest
from nemo_fabric_adapter_contract.codec import ContractValidationError
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentSkillConfig
from nemo_fabric_adapter_contract.models import BundledSkills
from nemo_fabric_adapter_contract.models import SkillDirectory
from nemo_fabric_adapter_contract.skills import DirectoryLoad
from nemo_fabric_adapter_contract.skills import DiscoveryRules
from nemo_fabric_adapter_contract.skills import DuplicateRecord
from nemo_fabric_adapter_contract.skills import FoundSkill
from nemo_fabric_adapter_contract.skills import SkillLoadError
from nemo_fabric_adapter_contract.skills import SkippedCandidate
from nemo_fabric_adapter_contract.skills import load_directories
from nemo_fabric_adapter_contract.skills import report_block
from nemo_fabric_adapter_contract.skills import warn


IMMEDIATE = DiscoveryRules(
    traversal="immediate_children", nested_skills="stop_at_skill", max_depth=1
)
RECURSIVE = DiscoveryRules(
    traversal="recursive", nested_skills="include", max_depth=None
)
STOP_AT_SKILL = DiscoveryRules(
    traversal="recursive", nested_skills="stop_at_skill", max_depth=None
)

_POSIX_PERMISSIONS = pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="needs POSIX permissions enforced for a non-root user",
)


@pytest.fixture(name="make_skill")
def make_skill_fixture() -> Callable[..., Path]:
    def make_skill(
        directory: Path, name: str | None = None, *, description: bool = True
    ) -> Path:
        directory.mkdir(parents=True)
        lines = []
        if name is not None:
            lines.append(f"name: {name}")
        if description:
            lines.append("description: Test skill.")
        frontmatter = "---\n" + "".join(f"{line}\n" for line in lines) + "---\n"
        (directory / "SKILL.md").write_text(
            f"{frontmatter}# Test skill\n", encoding="utf-8"
        )
        return directory

    return make_skill


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


# Contract models


def test_agent_skill_config_round_trips_structured_fields():
    value = {
        "skills": {
            "directories": [{"skill": "/opt/review"}, {"folder": "/sandbox/team"}],
            "bundled": {"enabled": True, "include": []},
            "duplicates": "first_wins",
            "extensions": {"acme": True},
        }
    }

    config = AgentConfig.from_mapping(value)

    assert config.skills.directories == [
        SkillDirectory(skill="/opt/review"),
        SkillDirectory(folder="/sandbox/team"),
    ]
    assert [(entry.kind, entry.path) for entry in config.skills.directories] == [
        ("skill", "/opt/review"),
        ("folder", "/sandbox/team"),
    ]
    assert config.skills.bundled == BundledSkills(include=[])
    assert config.to_mapping() == value


@pytest.mark.parametrize(
    ("skills", "omitted"),
    [
        ({"paths": ["/opt/review"]}, {"directories", "bundled", "duplicates"}),
        ({"directories": []}, {"paths", "bundled", "duplicates"}),
    ],
)
def test_agent_skill_config_omits_unset_fields(skills, omitted: set[str]):
    mapping = AgentSkillConfig.from_mapping(skills).to_mapping()

    assert mapping == skills
    assert omitted.isdisjoint(mapping)


@pytest.mark.parametrize(
    ("skills", "path", "message"),
    [
        (
            {"directories": [{"skill": "/a", "folder": "/b"}]},
            "skills.directories.0",
            "exactly one of skill or folder",
        ),
        ({"directories": [{}]}, "skills.directories.0", "exactly one"),
        (
            {"directories": [{"skill": "/a", "path": "/b"}]},
            "skills.directories.0",
            "unexpected field 'path'",
        ),
        ({"directories": [{"folder": 3}]}, "skills.directories.0.folder", "match"),
        ({"directories": [{"skill": " "}]}, "skills.directories.0.skill", "non-empty"),
        (
            {"paths": ["/a"], "directories": []},
            "skills.directories",
            "only when paths is empty",
        ),
        ({"duplicates": "reject"}, "skills.duplicates", "requires directories"),
        (
            {"directories": [], "duplicates": "newest"},
            "skills.duplicates",
            "must be one of",
        ),
        (
            {"bundled": {"include": [], "exclude": []}},
            "skills.bundled.exclude",
            "cannot both be set",
        ),
        (
            {"bundled": {"enabled": False, "include": ["a"]}},
            "skills.bundled.enabled",
            "enabled=false",
        ),
        (
            {"bundled": {"enabled": False, "exclude": []}},
            "skills.bundled.enabled",
            "enabled=false",
        ),
        (
            {"bundled": {"include": [""]}},
            "skills.bundled.include",
            "must not be empty",
        ),
        ({"bundled": {"exclude": [1]}}, "skills.bundled.exclude.0", "string"),
    ],
)
def test_agent_skill_config_rejects_invalid_structure(skills, path: str, message: str):
    with pytest.raises(ContractValidationError, match=message) as caught:
        AgentConfig.from_mapping({"skills": skills})

    assert ".".join(caught.value.path) == path


@pytest.mark.parametrize("value", ["", " ", Path(""), Path(".")])
def test_skill_directory_rejects_empty_paths(value):
    with pytest.raises(ContractValidationError, match="non-empty path"):
        SkillDirectory(skill=value)

    # The string "." names the current folder.
    assert SkillDirectory(folder=".").path == "."


def test_skill_directory_null_key_is_absent():
    entry = SkillDirectory.from_mapping({"skill": "/a", "folder": None})

    assert (entry.kind, entry.path) == ("skill", "/a")
    assert entry.to_mapping() == {"skill": "/a"}
    with pytest.raises(ContractValidationError, match="exactly one"):
        SkillDirectory.from_mapping({"skill": None, "folder": None})


def test_skill_directory_kind_is_none_without_a_key():
    entry = SkillDirectory(folder="/a")
    # Bypass validation to model a partially built entry.
    object.__setattr__(entry, "folder", None)

    assert entry.kind is None


# Task-environment checks


def test_skill_and_folder_entries_load_in_entry_order(
    tmp_path: Path, make_skill: Callable[..., Path]
):
    team = tmp_path / "team"
    debugging = make_skill(team / "debugging", "debugging")
    review = make_skill(team / "code-review", "code-review")
    (team / "drafts").mkdir()
    (team / "README.md").write_text("Not a skill.\n", encoding="utf-8")
    release = make_skill(tmp_path / "opt" / "release-review")

    load = load_directories(
        [
            {"folder": "team"},
            SkillDirectory(skill=str(release)),
            {"skill": Path("opt/release-review")},
        ],
        rules=IMMEDIATE,
        duplicates=None,
        base_dir=tmp_path,
    )

    assert load == DirectoryLoad(
        selected=[
            FoundSkill("code-review", review, "skills.directories[0]", "folder"),
            FoundSkill("debugging", debugging, "skills.directories[0]", "folder"),
            FoundSkill("release-review", release, "skills.directories[1]", "skill"),
        ],
        entries=[
            ("skills.directories[0]", "folder", team),
            ("skills.directories[1]", "skill", release),
            ("skills.directories[2]", "skill", release),
        ],
    )


def test_empty_folder_entry_is_valid_and_recorded(tmp_path: Path):
    (tmp_path / "empty").mkdir()

    load = load_directories(
        [{"folder": tmp_path / "empty"}], rules=RECURSIVE, duplicates="reject"
    )

    assert load.selected == []
    assert load.empty_entries == ["skills.directories[0]"]
    assert report_block(load) == {"empty_entries": ["skills.directories[0]"]}


def test_no_entries_load_nothing():
    load = load_directories([], rules=RECURSIVE, duplicates=None)

    assert load == DirectoryLoad()
    assert report_block(load) is None


def _missing(root: Path, make_skill, lock):
    return {"folder": root / "missing"}, ".folder", (root / "missing",)


def _missing_parent(root: Path, make_skill, lock):
    (root / "file").write_text("Not a folder.\n", encoding="utf-8")
    return {"skill": root / "file" / "review"}, ".skill", (root / "file" / "review",)


def _file(root: Path, make_skill, lock):
    (root / "SKILL.md").write_text("# Not a folder\n", encoding="utf-8")
    return {"skill": root / "SKILL.md"}, ".skill", (root / "SKILL.md",)


def _skill_without_skill_file(root: Path, make_skill, lock):
    (root / "review").mkdir()
    return {"skill": root / "review"}, ".skill", (root / "review",)


def _skill_file_is_a_folder(root: Path, make_skill, lock):
    (root / "review" / "SKILL.md").mkdir(parents=True)
    return {"skill": root / "review"}, ".skill", (root / "review",)


def _unreadable_skill_file(root: Path, make_skill, lock):
    review = make_skill(root / "review")
    lock(review / "SKILL.md")
    return {"skill": review}, ".skill", (review.resolve() / "SKILL.md",)


def _unlistable_folder(root: Path, make_skill, lock):
    (root / "locked").mkdir()
    lock(root / "locked")
    return {"folder": root / "locked"}, ".folder", (root / "locked",)


def _inaccessible_parent(root: Path, make_skill, lock):
    make_skill(root / "locked" / "review")
    lock(root / "locked")
    return (
        {"skill": root / "locked" / "review"},
        ".skill",
        (root / "locked" / "review",),
    )


def _unsearchable_skill_folder(root: Path, make_skill, lock):
    # The folder exists, but nothing inside it, SKILL.md included, can be seen.
    lock(make_skill(root / "review"))
    return {"skill": root / "review"}, ".skill", (root / "review",)


@pytest.mark.parametrize(
    ("layout", "reason", "message"),
    [
        (_missing, "not_found", "does not exist"),
        (_missing_parent, "not_found", "does not exist"),
        (_file, "invalid", "is not a folder"),
        (_skill_without_skill_file, "invalid", "has no SKILL.md"),
        (_skill_file_is_a_folder, "invalid", "has no SKILL.md"),
        pytest.param(
            _unreadable_skill_file,
            "invalid",
            "SKILL.md that cannot be read",
            marks=_POSIX_PERMISSIONS,
        ),
        pytest.param(
            _unlistable_folder, "invalid", "cannot be listed", marks=_POSIX_PERMISSIONS
        ),
        pytest.param(
            _inaccessible_parent,
            "invalid",
            "is not accessible",
            marks=_POSIX_PERMISSIONS,
        ),
        pytest.param(
            _unsearchable_skill_folder,
            "invalid",
            "is not accessible",
            marks=_POSIX_PERMISSIONS,
        ),
    ],
)
def test_entry_failures_name_the_key_and_path(
    tmp_path: Path,
    make_skill: Callable[..., Path],
    lock: Callable[[Path], Path],
    layout,
    reason: str,
    message: str,
):
    make_skill(tmp_path / "first")
    entry, key, paths = layout(tmp_path, make_skill, lock)

    with pytest.raises(SkillLoadError, match=message) as caught:
        load_directories(
            [{"skill": tmp_path / "first"}, entry], rules=RECURSIVE, duplicates=None
        )

    assert caught.value.reason == reason
    assert caught.value.field == f"skills.directories[1]{key}"
    assert caught.value.paths == paths
    assert caught.value.field in str(caught.value)
    assert all(str(path) in str(caught.value) for path in paths)


@pytest.mark.parametrize(
    ("entry", "field", "message"),
    [
        ({"skill": "/a", "folder": "/b"}, "skills.directories[0]", "exactly one"),
        ({"path": "/a"}, "skills.directories[0]", "unexpected field 'path'"),
        ({"folder": 3}, "skills.directories[0].folder", "declared types"),
        ("/a", "skills.directories[0]", "must be an object"),
    ],
)
def test_malformed_entries_are_invalid(entry, field: str, message: str):
    with pytest.raises(SkillLoadError, match=message) as caught:
        load_directories([entry], rules=RECURSIVE, duplicates=None)

    assert (caught.value.reason, caught.value.field) == ("invalid", field)


def test_relative_entries_resolve_against_base_dir_without_expansion(
    tmp_path: Path, make_skill: Callable[..., Path]
):
    os.environ["SKILLS_ROOT"] = str(tmp_path)
    make_skill(tmp_path / "review")

    with pytest.raises(SkillLoadError) as caught:
        load_directories(
            [{"skill": "$SKILLS_ROOT/review"}],
            rules=RECURSIVE,
            duplicates=None,
            base_dir=tmp_path,
        )

    assert caught.value.reason == "not_found"
    assert caught.value.paths == (tmp_path / "$SKILLS_ROOT" / "review",)


# Discovery


@pytest.fixture(name="nested_tree")
def nested_tree_fixture(tmp_path: Path, make_skill: Callable[..., Path]) -> Path:
    root = tmp_path / "skills"
    make_skill(root / "alpha", "alpha")
    make_skill(root / "alpha" / "inner", "alpha-inner")
    make_skill(root / "group" / "beta", "beta")
    make_skill(root / "group" / "deeper" / "gamma", "gamma")
    make_skill(root / ".git" / "hidden", "hidden")
    make_skill(root / "node_modules" / "package", "package")
    return root


@pytest.mark.parametrize(
    ("rules", "names"),
    [
        (IMMEDIATE, ["alpha"]),
        (RECURSIVE, ["alpha", "alpha-inner", "beta", "gamma"]),
        (STOP_AT_SKILL, ["alpha", "beta", "gamma"]),
        (
            DiscoveryRules(traversal="recursive", nested_skills="include", max_depth=2),
            ["alpha", "alpha-inner", "beta"],
        ),
        (
            DiscoveryRules(traversal="recursive", nested_skills="include", max_depth=1),
            ["alpha"],
        ),
        (
            DiscoveryRules(
                traversal="recursive",
                nested_skills="include",
                max_depth=None,
                excluded_dir_names=frozenset({"group"}),
            ),
            ["hidden", "alpha", "alpha-inner", "package"],
        ),
    ],
)
def test_folder_discovery_follows_rules(
    nested_tree: Path, rules: DiscoveryRules, names: list[str]
):
    load = load_directories([{"folder": nested_tree}], rules=rules, duplicates=None)

    assert [skill.name for skill in load.selected] == names
    assert all(skill.kind == "folder" for skill in load.selected)


@pytest.mark.parametrize(
    ("rules", "names"),
    [
        (RECURSIVE, ["alpha", "alpha-inner"]),
        (STOP_AT_SKILL, ["alpha"]),
        (IMMEDIATE, ["alpha"]),
    ],
)
def test_skill_entry_includes_nested_skills_only_when_rules_do(
    nested_tree: Path, rules: DiscoveryRules, names: list[str]
):
    load = load_directories(
        [{"skill": nested_tree / "alpha"}], rules=rules, duplicates=None
    )

    assert [(skill.name, skill.kind) for skill in load.selected] == [
        (name, "skill") for name in names
    ]


def test_symlink_loops_are_not_followed(
    tmp_path: Path, make_skill: Callable[..., Path]
):
    root = tmp_path / "skills"
    review = make_skill(root / "group" / "review", "review")
    (root / "group" / "loop").symlink_to(root, target_is_directory=True)
    (review / "back").symlink_to(root / "group", target_is_directory=True)

    load = load_directories([{"folder": root}], rules=RECURSIVE, duplicates="reject")

    assert load.selected == [
        FoundSkill("review", review.resolve(), "skills.directories[0]", "folder")
    ]


def test_symlinked_skill_folders_report_resolved_path(
    tmp_path: Path, make_skill: Callable[..., Path]
):
    target = make_skill(tmp_path / "store" / "real-review")
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills" / "review").symlink_to(target, target_is_directory=True)

    load = load_directories(
        [{"folder": tmp_path / "skills"}], rules=IMMEDIATE, duplicates=None
    )

    # The harness sees the link's name; the report gives the resolved folder.
    assert load.selected == [
        FoundSkill("review", target.resolve(), "skills.directories[0]", "folder")
    ]
    assert report_block(load)["loaded"][0]["path"] == str(target.resolve())


@pytest.mark.parametrize(
    "spelling",
    ["shared", "shared/review/", "./shared/review", "other/../shared/review", "link"],
)
def test_same_skill_reached_twice_loads_once(
    tmp_path: Path, make_skill: Callable[..., Path], spelling: str
):
    review = make_skill(tmp_path / "shared" / "review", "review")
    (tmp_path / "other").mkdir()
    (tmp_path / "link").symlink_to(review, target_is_directory=True)
    kind = "folder" if spelling == "shared" else "skill"

    load = load_directories(
        [{"skill": "shared/review"}, {kind: spelling}],
        rules=IMMEDIATE,
        duplicates="reject",
        base_dir=tmp_path,
    )

    assert load.selected == [
        FoundSkill("review", review.resolve(), "skills.directories[0]", "skill")
    ]
    assert load.duplicates == []
    assert load.empty_entries == []


def test_folders_without_inode_numbers_compare_by_path(
    tmp_path: Path, make_skill: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
):
    shared = tmp_path / "shared"
    one = make_skill(shared / "one", "one")
    two = make_skill(shared / "two", "two")
    stat = Path.stat

    def without_inode(path: Path, *args, **kwargs) -> os.stat_result:
        # Some file systems report inode 0 for every folder.
        result = stat(path, *args, **kwargs)
        return os.stat_result((result.st_mode, 0, *tuple(result)[2:]))

    monkeypatch.setattr(Path, "stat", without_inode)

    load = load_directories(
        [{"folder": shared}, {"skill": one}], rules=IMMEDIATE, duplicates="reject"
    )

    assert [skill.path for skill in load.selected] == [one.resolve(), two.resolve()]


@_POSIX_PERMISSIONS
def test_unreadable_folder_inside_entry_is_ignored(
    tmp_path: Path, make_skill: Callable[..., Path], lock: Callable[[Path], Path]
):
    shared = tmp_path / "shared"
    one = make_skill(shared / "one", "one")
    lock(make_skill(shared / "locked" / "nested").parent)

    load = load_directories([{"folder": shared}], rules=RECURSIVE, duplicates=None)

    assert [skill.path for skill in load.selected] == [one.resolve()]
    assert load.skipped == []


def test_discovery_does_not_depend_on_path_predicates_that_can_raise(
    tmp_path: Path, make_skill: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
):
    # Python 3.13 and earlier raise PermissionError from these predicates for
    # paths below an untraversable folder.
    team = tmp_path / "team"
    review = make_skill(team / "review", "review")
    single = make_skill(tmp_path / "single", "single")
    (tmp_path / "file").write_text("Not a folder.\n", encoding="utf-8")

    def denied(self: Path, *args, **kwargs) -> bool:
        raise PermissionError(13, "Permission denied", str(self))

    for predicate in ("is_dir", "is_file", "exists"):
        monkeypatch.setattr(Path, predicate, denied)

    load = load_directories(
        [{"folder": team}, {"skill": single}], rules=RECURSIVE, duplicates="reject"
    )

    assert [skill.path for skill in load.selected] == [
        review.resolve(),
        single.resolve(),
    ]
    with pytest.raises(SkillLoadError, match="is not a folder"):
        load_directories(
            [{"skill": tmp_path / "file"}], rules=RECURSIVE, duplicates=None
        )


def test_deeper_symlink_does_not_hide_the_real_folder(
    tmp_path: Path, make_skill: Callable[..., Path]
):
    root = tmp_path / "root"
    skill = make_skill(root / "m" / "n" / "s")
    (root / "a" / "x").mkdir(parents=True)
    # Reached at depth 3 through the link, but at depth 2 through m/n.
    (root / "a" / "x" / "link").symlink_to(root / "m" / "n", target_is_directory=True)
    rules = DiscoveryRules(
        traversal="recursive", nested_skills="stop_at_skill", max_depth=3
    )

    load = load_directories([{"folder": root}], rules=rules, duplicates=None)

    assert load.selected == [
        FoundSkill("s", skill.resolve(), "skills.directories[0]", "folder")
    ]


def test_breadth_first_discovery_keeps_name_order(
    tmp_path: Path, make_skill: Callable[..., Path]
):
    root = tmp_path / "root"
    for path in ("b", "a/z/deep", "a/c", "a"):
        if not (root / path).exists():
            make_skill(root / path)
    (root / "a" / "SKILL.md").write_text("---\n---\n", encoding="utf-8")

    load = load_directories([{"folder": root}], rules=RECURSIVE, duplicates=None)

    assert [skill.path.relative_to(root.resolve()) for skill in load.selected] == [
        Path("a"),
        Path("a/c"),
        Path("a/z/deep"),
        Path("b"),
    ]


def test_rules_can_differ_per_entry(tmp_path: Path, make_skill: Callable[..., Path]):
    shallow = make_skill(tmp_path / "one" / "group" / "deep").parent.parent
    deep = make_skill(tmp_path / "two" / "group" / "deep").parent.parent
    single = make_skill(tmp_path / "single")
    make_skill(single / "nested")
    calls = []

    def rules(index: int, kind: str, path: Path) -> DiscoveryRules:
        calls.append((index, kind, path))
        return STOP_AT_SKILL if index == 1 else IMMEDIATE

    load = load_directories(
        [{"folder": shallow}, {"folder": deep}, {"skill": single}],
        rules=rules,
        duplicates="reject",
    )

    assert calls == [
        (0, "folder", shallow.resolve()),
        (1, "folder", deep.resolve()),
        (2, "skill", single.resolve()),
    ]
    assert [(skill.path, skill.field) for skill in load.selected] == [
        (deep.resolve() / "group" / "deep", "skills.directories[1]"),
        (single.resolve(), "skills.directories[2]"),
    ]
    assert load.empty_entries == ["skills.directories[0]"]


def test_rules_callable_must_return_rules(tmp_path: Path):
    (tmp_path / "empty").mkdir()

    with pytest.raises(TypeError, match=r"skills.directories\[0\]"):
        load_directories(
            [{"folder": tmp_path / "empty"}],
            rules=lambda index, kind, path: None,
            duplicates=None,
        )
    with pytest.raises(TypeError, match="DiscoveryRules"):
        load_directories([], rules="recursive", duplicates=None)


@pytest.mark.parametrize(
    ("nested_skills", "names"),
    [("include", ["team", "inner"]), ("stop_at_skill", ["team"])],
)
def test_folder_root_holding_skill_md_can_be_a_skill(
    tmp_path: Path, make_skill: Callable[..., Path], nested_skills: str, names
):
    team = make_skill(tmp_path / "team", "team")
    make_skill(team / "inner", "inner")
    rules = DiscoveryRules(
        traversal="recursive",
        nested_skills=nested_skills,
        max_depth=None,
        folder_root_is_skill=True,
    )

    load = load_directories([{"folder": team}], rules=rules, duplicates="reject")

    assert [(skill.name, skill.kind) for skill in load.selected] == [
        (name, "folder") for name in names
    ]
    assert load.selected[0].path == team.resolve()
    # Without the rule the root is a container, as before.
    default = load_directories([{"folder": team}], rules=RECURSIVE, duplicates=None)
    assert [skill.name for skill in default.selected] == ["inner"]


def test_folder_root_skill_is_skipped_like_other_candidates(tmp_path: Path):
    team = tmp_path / "team"
    team.mkdir()
    (team / "SKILL.md").write_text("---\nname: team\n---\n", encoding="utf-8")
    rules = DiscoveryRules(
        traversal="recursive",
        nested_skills="include",
        max_depth=None,
        required_frontmatter_fields=("description",),
        folder_root_is_skill=True,
    )

    load = load_directories([{"folder": team}], rules=rules, duplicates=None)

    assert load.skipped == [
        SkippedCandidate(
            team.resolve(),
            "skills.directories[0]",
            "missing_frontmatter_field:description",
        )
    ]
    assert load.empty_entries == ["skills.directories[0]"]


def test_support_folders_inside_a_skill_are_not_searched(
    tmp_path: Path, make_skill: Callable[..., Path]
):
    team = tmp_path / "team"
    review = make_skill(team / "review", "review")
    # A SKILL.md template in a support folder is not a second skill.
    make_skill(review / "scripts" / "template", "review")
    other = make_skill(team / "other", "other")
    make_skill(team / "scripts" / "tool", "tool")
    single = make_skill(tmp_path / "single", "single")
    make_skill(single / "assets" / "example", "example")
    rules = DiscoveryRules(
        traversal="recursive",
        nested_skills="include",
        max_depth=None,
        support_dir_names=frozenset({"scripts", "assets"}),
    )

    load = load_directories(
        [{"folder": team}, {"skill": single}], rules=rules, duplicates="reject"
    )

    # Only folders holding SKILL.md prune their support folders.
    assert [skill.name for skill in load.selected] == [
        "other",
        "review",
        "tool",
        "single",
    ]
    assert other.resolve() in [skill.path for skill in load.selected]
    with pytest.raises(SkillLoadError, match="appears twice"):
        load_directories([{"folder": team}], rules=RECURSIVE, duplicates="reject")


def test_symlink_escape_is_skipped_when_rules_forbid_it(
    tmp_path: Path, make_skill: Callable[..., Path]
):
    outside = make_skill(tmp_path / "store" / "outside", "outside")
    team = tmp_path / "team"
    inside = make_skill(team / "inside", "inside")
    (team / "linked").symlink_to(outside, target_is_directory=True)
    (team / "alias").symlink_to(inside, target_is_directory=True)
    contained = DiscoveryRules(
        traversal="immediate_children",
        nested_skills="stop_at_skill",
        max_depth=1,
        allow_symlink_escape=False,
    )

    load = load_directories([{"folder": team}], rules=contained, duplicates="reject")

    assert [skill.path for skill in load.selected] == [inside.resolve()]
    assert load.skipped == [
        SkippedCandidate(outside.resolve(), "skills.directories[0]", "path_escape")
    ]
    allowed = load_directories([{"folder": team}], rules=IMMEDIATE, duplicates=None)
    assert {skill.path for skill in allowed.selected} == {
        inside.resolve(),
        outside.resolve(),
    }


def test_subroot_restarts_depth_counting(
    tmp_path: Path, make_skill: Callable[..., Path]
):
    root = tmp_path / "root"
    make_skill(root / "a" / "b" / "too-deep")
    make_skill(root / "a" / "shallow")
    make_skill(root / "skills" / "x" / "y" / "deep")
    make_skill(root / "skills" / "x" / "y" / "z" / "too-deep-subroot")
    make_skill(root / "other" / "skills" / "q" / "not-a-subroot")
    rules = DiscoveryRules(
        traversal="recursive",
        nested_skills="stop_at_skill",
        max_depth=2,
        subroot_depths={"skills": 3},
    )

    load = load_directories([{"folder": root}], rules=rules, duplicates=None)

    assert [skill.name for skill in load.selected] == ["shallow", "deep"]
    # A skill entry has no subroots.
    skill_entry = make_skill(tmp_path / "single")
    make_skill(skill_entry / "skills" / "x" / "y" / "deep")
    nested = DiscoveryRules(
        traversal="recursive",
        nested_skills="include",
        max_depth=2,
        subroot_depths={"skills": 3},
    )
    load = load_directories([{"skill": skill_entry}], rules=nested, duplicates=None)
    assert [skill.name for skill in load.selected] == ["single"]


@pytest.mark.parametrize(
    "subroot_depths", [{"skills": 0}, {"skills": True}, {"": 2}, ["skills"]]
)
def test_discovery_rules_reject_invalid_subroot_depths(subroot_depths):
    with pytest.raises(ValueError, match="subroot_depths"):
        DiscoveryRules(
            traversal="recursive",
            nested_skills="include",
            max_depth=None,
            subroot_depths=subroot_depths,
        )


def test_discovery_rules_stay_immutable_and_hashable():
    depths = {"skills": 6}
    rules = DiscoveryRules(
        traversal="recursive",
        nested_skills="include",
        max_depth=2,
        subroot_depths=depths,
    )
    depths["skills"] = 1

    assert rules.subroot_depths == {"skills": 6}
    with pytest.raises(TypeError):
        rules.subroot_depths["skills"] = 1  # type: ignore[index]
    assert hash(rules) == hash(
        DiscoveryRules(traversal="recursive", nested_skills="include", max_depth=2)
    )
    assert RECURSIVE == DiscoveryRules(
        traversal="recursive", nested_skills="include", max_depth=None
    )


# Parsing


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ('---\r\nname: "code-review"\r\n---\r\n', "code-review"),
        ("---\nname: code-review # main skill\n---\n", "code-review"),
        ("---\nname: 'code-review' # quoted\n---\n", "code-review"),
        ("---\nname: code#review\n---\n", "code#review"),
        ("--- \nname: code-review\n---\n", "code-review"),
        ("﻿---\nname: code-review\n---\n", "code-review"),
        ("---\nname:\n---\n", "review"),
        ("---\nname: 'it''s'\n---\n", "it's"),
        ('---\nname: "say \\"hi\\" \\u0041"\n---\n', 'say "hi" A'),
        ('---\nname: "a\\\\b"\n---\n', "a\\b"),
        ("---\nname: 'null'\n---\n", "null"),
        ("---\nname: ''\n---\n", "review"),
        ("---\nname: ~\n---\n", "review"),
        ("---\nname: null # unset\n---\n", "review"),
        ("---\nname: Null\n---\n", "review"),
        ("---\nname: NULL\n---\n", "review"),
        ("---\nname: >\n  folded\n---\n", "review"),
        ("---\nname: |-\n  literal\n---\n", "review"),
        ("---\nname: >2 # indented\n  folded\n---\n", "review"),
        ("---\nname: [a, b]\n---\n", "review"),
        ("---\nname: {first: a}\n---\n", "review"),
        ('---\n"name": quoted-key\n---\n', "quoted-key"),
        ("---\n'name' : quoted-spaced\n---\n", "quoted-spaced"),
        ("---\nname : spaced-key\n---\n", "spaced-key"),
        ("---\nnamespace: other\n---\n", "review"),
        ("---\nmetadata:\n  name: nested\n---\n", "review"),
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

    load = load_directories([{"skill": review}], rules=IMMEDIATE, duplicates=None)

    assert [skill.name for skill in load.selected] == [name]


@pytest.mark.parametrize(
    ("text", "loads"),
    [
        ("---\nname: review\ndescription: Reviews code.\n---\n", True),
        ("---\nname: review\ndescription:\n---\n", True),
        ("---\n'description': Quoted key.\n---\n", True),
        ("---\nname: review\n---\n", False),
        ("---\nmetadata:\n  description: Nested.\n---\n", False),
        ("---\n# description: Commented out.\n---\n", False),
        ("description: No front matter.\n", False),
    ],
)
def test_required_front_matter_fields_need_a_top_level_key(
    tmp_path: Path, text: str, loads: bool
):
    review = tmp_path / "skills" / "review"
    review.mkdir(parents=True)
    (review / "SKILL.md").write_text(text, encoding="utf-8")
    rules = DiscoveryRules(
        traversal="immediate_children",
        nested_skills="stop_at_skill",
        max_depth=1,
        required_frontmatter_fields=("description",),
    )

    load = load_directories(
        [{"folder": tmp_path / "skills"}], rules=rules, duplicates=None
    )

    assert bool(load.selected) is loads
    assert load.skipped == (
        []
        if loads
        else [
            SkippedCandidate(
                review.resolve(),
                "skills.directories[0]",
                "missing_frontmatter_field:description",
            )
        ]
    )


def test_skills_without_a_name_are_skipped_without_fallback(
    tmp_path: Path, make_skill: Callable[..., Path]
):
    root = tmp_path / "skills"
    named = make_skill(root / "named", "named")
    unnamed = make_skill(root / "unnamed")
    rules = DiscoveryRules(
        traversal="immediate_children",
        nested_skills="stop_at_skill",
        max_depth=1,
        name_fallback=False,
    )

    load = load_directories([{"folder": root}], rules=rules, duplicates=None)

    assert [skill.path for skill in load.selected] == [named.resolve()]
    assert load.skipped == [
        SkippedCandidate(unnamed.resolve(), "skills.directories[0]", "no_name")
    ]


@_POSIX_PERMISSIONS
def test_unreadable_candidate_is_skipped(
    tmp_path: Path, make_skill: Callable[..., Path], lock: Callable[[Path], Path]
):
    root = tmp_path / "skills"
    locked = make_skill(root / "locked")
    lock(locked / "SKILL.md")

    load = load_directories([{"folder": root}], rules=IMMEDIATE, duplicates=None)

    assert load.skipped == [
        SkippedCandidate(locked.resolve(), "skills.directories[0]", "unreadable")
    ]
    assert load.empty_entries == ["skills.directories[0]"]


def test_discovery_rules_reject_invalid_values():
    with pytest.raises(ValueError, match="max_depth"):
        DiscoveryRules(traversal="recursive", nested_skills="include", max_depth=0)
    with pytest.raises(ValueError, match="traversal"):
        DiscoveryRules(traversal="deep", nested_skills="include", max_depth=None)
    with pytest.raises(ValueError, match="nested"):
        DiscoveryRules(traversal="recursive", nested_skills="all", max_depth=None)


# Duplicates


@pytest.fixture(name="clashing_entries")
def clashing_entries_fixture(
    tmp_path: Path, make_skill: Callable[..., Path]
) -> tuple[list[dict[str, Path]], Path, Path]:
    team = tmp_path / "team"
    make_skill(team / "debugging", "debugging")
    first = make_skill(team / "review", "review")
    second = make_skill(tmp_path / "project" / "review-copy", "review")
    return [{"folder": team}, {"skill": second}], first.resolve(), second.resolve()


def test_native_duplicates_record_ambiguous_clash_and_keep_every_copy(
    clashing_entries,
):
    entries, first, second = clashing_entries

    load = load_directories(entries, rules=IMMEDIATE, duplicates=None)

    assert [skill.path for skill in load.selected][1:] == [first, second]
    assert load.duplicates == [
        DuplicateRecord(
            "review",
            (
                ("external", first, "skills.directories[0]"),
                ("external", second, "skills.directories[1]"),
            ),
            "ambiguous",
            None,
        )
    ]


@pytest.mark.parametrize(("policy", "winner"), [("first_wins", 0), ("last_wins", 1)])
def test_explicit_policy_selects_one_copy(clashing_entries, policy: str, winner: int):
    entries, first, second = clashing_entries
    selected = (first, second)[winner]

    load = load_directories(entries, rules=IMMEDIATE, duplicates=policy)

    assert [skill.name for skill in load.selected] == ["debugging", "review"]
    assert load.selected[1].path == selected
    assert load.duplicates[0].outcome == "selected"
    assert load.duplicates[0].selected == selected


def test_reject_policy_names_both_entries(clashing_entries):
    entries, first, second = clashing_entries

    with pytest.raises(SkillLoadError) as caught:
        load_directories(entries, rules=IMMEDIATE, duplicates="reject")

    assert caught.value.reason == "duplicate"
    assert caught.value.field == "skills.directories[1]"
    assert caught.value.paths == (first, second)
    assert "skills.directories[0]" in str(caught.value)
    assert "skills.directories[1]" in str(caught.value)


@pytest.mark.parametrize("policy", ["reject", "first_wins", "last_wins"])
def test_explicit_policy_rejects_duplicates_inside_one_folder(
    tmp_path: Path, make_skill: Callable[..., Path], policy: str
):
    first = make_skill(tmp_path / "team" / "a", "review").resolve()
    second = make_skill(tmp_path / "team" / "b", "review").resolve()

    with pytest.raises(SkillLoadError, match="appears twice") as caught:
        load_directories(
            [{"folder": tmp_path / "team"}], rules=IMMEDIATE, duplicates=policy
        )

    assert caught.value.reason == "duplicate"
    assert caught.value.field == "skills.directories[0]"
    assert caught.value.paths == (first, second)


def test_native_duplicates_inside_one_folder_are_recorded(
    tmp_path: Path, make_skill: Callable[..., Path]
):
    make_skill(tmp_path / "team" / "a", "review")
    make_skill(tmp_path / "team" / "b", "review")

    load = load_directories(
        [{"folder": tmp_path / "team"}], rules=IMMEDIATE, duplicates=None
    )

    assert len(load.selected) == 2
    assert load.duplicates[0].outcome == "ambiguous"


@pytest.mark.parametrize(
    ("policy", "field", "winner"),
    [
        ("last_wins", "skills.directories[2]", "s1"),
        ("first_wins", "skills.directories[0]", "s1"),
        (None, "skills.directories[0]", None),
    ],
)
def test_folder_reached_again_moves_to_the_later_entry_only_under_last_wins(
    tmp_path: Path,
    make_skill: Callable[..., Path],
    policy: str | None,
    field: str,
    winner: str | None,
):
    first = make_skill(tmp_path / "F1" / "s1", "x").resolve()
    second = make_skill(tmp_path / "F2" / "s2", "x").resolve()

    load = load_directories(
        [
            {"folder": tmp_path / "F1"},
            {"folder": tmp_path / "F2"},
            {"skill": tmp_path / "F1" / "s1"},
        ],
        rules=IMMEDIATE,
        duplicates=policy,
    )

    copies = {skill.path: skill for skill in load.selected}
    assert copies[first].field == field
    if winner is None:
        assert list(copies) == [first, second]
        assert load.duplicates[0].outcome == "ambiguous"
    else:
        assert list(copies) == [first]
        assert copies[first].kind == ("skill" if policy == "last_wins" else "folder")
        assert load.duplicates[0].selected == first
        assert [copy[2] for copy in load.duplicates[0].copies] == sorted(
            ["skills.directories[1]", field]
        )


# Reporting


def test_report_block_and_warnings_describe_the_load(
    tmp_path: Path, make_skill: Callable[..., Path], capsys
):
    team = tmp_path / "team"
    review = make_skill(team / "review", "review").resolve()
    plain = make_skill(team / "plain", description=False).resolve()
    copy = make_skill(tmp_path / "copy", "review").resolve()
    (tmp_path / "empty").mkdir()
    rules = DiscoveryRules(
        traversal="immediate_children",
        nested_skills="stop_at_skill",
        max_depth=1,
        required_frontmatter_fields=("description",),
    )

    load = load_directories(
        [{"folder": team}, {"skill": copy}, {"folder": tmp_path / "empty"}],
        rules=rules,
        duplicates="last_wins",
    )
    bundled = {"selected": ["pdf"], "unavailable": []}

    assert report_block(load, bundled=bundled) == {
        "loaded": [
            {
                "name": "review",
                "path": str(copy),
                "entry": "skills.directories[1]",
                "origin": "external",
            }
        ],
        "empty_entries": ["skills.directories[2]"],
        "skipped": [
            {
                "path": str(plain),
                "entry": "skills.directories[0]",
                "reason": "missing_frontmatter_field:description",
            }
        ],
        "duplicates": [
            {
                "name": "review",
                "outcome": "selected",
                "selected": str(copy),
                "copies": [
                    {
                        "origin": "external",
                        "path": str(review),
                        "entry": "skills.directories[0]",
                    },
                    {
                        "origin": "external",
                        "path": str(copy),
                        "entry": "skills.directories[1]",
                    },
                ],
            }
        ],
        "bundled": bundled,
    }

    warn("acme", load)
    lines = capsys.readouterr().err.splitlines()
    assert lines == [
        f"acme: skipped skill {plain} from skills.directories[0]: "
        "missing_frontmatter_field:description",
        "acme: skills.directories[2] contains no skills",
        f"acme: skill 'review' has 2 copies, selected; {copy} loads: "
        f"{review} (external, skills.directories[0]), "
        f"{copy} (external, skills.directories[1])",
    ]
    assert "Test skill" not in str(report_block(load))


def test_report_block_lists_copies_from_other_sources(tmp_path: Path):
    record = DuplicateRecord(
        "pdf",
        (
            ("external", tmp_path / "pdf", "skills.directories[0]"),
            ("bundled", tmp_path / "b", ""),
        ),
        "shadowed",
        tmp_path / "b",
    )

    block = report_block(DirectoryLoad(duplicates=[record]))

    assert block["duplicates"][0]["copies"][1] == {
        "origin": "bundled",
        "path": str(tmp_path / "b"),
    }
