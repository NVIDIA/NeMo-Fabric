# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured skills contract models and the task-environment skills helper."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
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


@pytest.mark.parametrize(
    "skills",
    [
        {"duplicates": "reject"},
        {"directories": None, "duplicates": "reject"},
        {"paths": ["/a"], "directories": []},
        {"bundled": {"include": [], "exclude": []}},
        {"bundled": {"enabled": False, "include": []}},
        {"duplicates": None},
        {"paths": ["/a"], "directories": None},
        {"directories": [], "duplicates": "reject"},
        {"bundled": {"include": [], "exclude": None}},
        {"bundled": {"enabled": False, "exclude": None}},
        {"bundled": {"exclude": []}},
    ],
)
def test_agent_skill_schema_matches_the_contract_model(skills: dict):
    """The committed schema applies the cross-field rules, null read as absent."""
    schema = json.loads(
        (
            Path(__file__).parents[2]
            / "schemas/adapter-contract/agent-config.schema.json"
        ).read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(
        {"$defs": schema["$defs"], "$ref": "#/$defs/AgentSkillConfig"}
    )
    try:
        AgentSkillConfig.from_mapping(skills)
    except ContractValidationError:
        accepted = False
    else:
        accepted = True

    assert validator.is_valid(skills) is accepted


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


@pytest.mark.skipif(
    os.name == "nt", reason="Windows collapses `..` before following a symlink"
)
def test_dot_dot_after_a_symlink_follows_the_file_system(
    tmp_path: Path, make_skill: Callable[..., Path]
):
    (tmp_path / "store" / "project").mkdir(parents=True)
    real = make_skill(tmp_path / "store" / "skills" / "real")
    make_skill(tmp_path / "cfg" / "skills" / "decoy")
    (tmp_path / "cfg" / "link").symlink_to(
        tmp_path / "store" / "project", target_is_directory=True
    )

    load = load_directories(
        [{"folder": "cfg/link/../skills"}, {"skill": "cfg/link/../skills/real"}],
        rules=RECURSIVE,
        duplicates=None,
        base_dir=tmp_path,
    )

    # The OS resolves ``link/..`` to ``store``, not lexically to ``cfg``.
    assert load.selected == [
        FoundSkill("real", real.resolve(), "skills.directories[0]", "folder")
    ]
    # A path ending in ``..`` falls back to the resolved folder's name.
    (real / "sub").mkdir()
    (tmp_path / "cfg" / "inner").symlink_to(real / "sub", target_is_directory=True)
    load = load_directories(
        [{"skill": "cfg/inner/.."}],
        rules=IMMEDIATE,
        duplicates=None,
        base_dir=tmp_path,
    )
    assert load.selected == [
        FoundSkill("real", real.resolve(), "skills.directories[0]", "skill")
    ]


def test_folder_is_entered_again_with_a_larger_depth_budget(
    tmp_path: Path, make_skill: Callable[..., Path]
):
    root = tmp_path / "root"
    review = make_skill(root / "skills" / "group" / "deep" / "review")
    # ``a`` sorts first and claims ``group`` with the outer, smaller budget.
    (root / "a").symlink_to(root / "skills" / "group", target_is_directory=True)
    rules = DiscoveryRules(
        traversal="recursive",
        nested_skills="stop_at_skill",
        max_depth=2,
        subroot_depths={"skills": 6},
    )

    load = load_directories([{"folder": root}], rules=rules, duplicates=None)

    assert load.selected == [
        FoundSkill("review", review.resolve(), "skills.directories[0]", "folder")
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
        ("---\nname: >\n  folded\n---\n", "folded"),
        ("---\nname: |-\n  literal\n---\n", "literal"),
        ("---\nname: >2 # indented\n  folded\n---\n", "folded"),
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


# Front matter as harnesses pass it to YAML: the text between the fences,
# without the final line break.
_YAML_NAMES = [
    # Plain scalars.
    "name: review",
    "name: code review",
    "name: code  review",
    "name: code#review",
    "name: code-review # comment",
    "name: -review",
    "name: review:v2",
    "name: review\ndescription: d",
    # Quoted scalars and escapes.
    "name: 'review'",
    "name: 'it''s'",
    "name: 'review' # comment",
    'name: "say \\"hi\\" \\u0041"',
    'name: "tab\\there"',
    'name: "\\x41\\U0001F600\\\\"',
    "name: 'null'",
    "name: '123'",
    "name: ''",
    'name: "multi\n  line"',
    "name: 'multi\n\n  paragraph'",
    'name: "joined\\\n  line"',
    'name: "starts\nat column 0"\ndescription: d',
    # Block scalars: literal and folded, each chomping indicator.
    "name: |\n  review",
    "name: |\n  review\ndescription: d",
    "name: |-\n  review\ndescription: d",
    "name: |+\n  review\n\ndescription: d",
    "name: |+\n  review\n\n",
    "name: >\n  code\n  review\ndescription: d",
    "name: >-\n  code\n  review",
    "name: >+\n  review\n\n\ndescription: d",
    "name: >-\n  a\n\n  b\n    more\n  c",
    "name: >-\n  a\n    more\n  b",
    "name: |-\n  literal\n   more",
    "name: |-\n\n  leading blank line",
    "name: |-\n \n  short blank line",
    "name: |\ndescription: d",
    "name: > # comment\n  review\n# comment\ndescription: d",
    # Explicit indentation indicators, before or after chomping.
    "name: |2-\n    indented\ndescription: d",
    "name: >-2\n  review",
    "name: >1\n review",
    "name: |2\n  review\n   more\ndescription: d",
    # Values on the lines below the key.
    "name: code\n  review",
    "name: code\n  review\n\n  again\ndescription: d",
    "name: code\n   review\n  again",
    "name: code # comment\ndescription: d",
    "name:\n  review",
    "name: # comment\n  review",
    "name:\n# comment\n  review",
    "name:\n  >-\n    review",
    "name:\n  'review'",
    # Null tokens.
    "name:",
    "name:\ndescription: d",
    "name: ~",
    "name: null",
    "name: Null # comment",
    "name: NULL",
    # Other scalars that are not strings.
    "name: 123",
    "name: 1.5",
    "name: 0x1F",
    "name: 1_000",
    "name: 1:20",
    "name: .inf",
    "name: true",
    "name: yes",
    "name: Off",
    "name: 2024-01-02",
    "name: 2024-01-02 10:00:00",
    # Tags and anchors.
    "name: !!str 123",
    "name: !!str |-\n  review",
    "name: &anchor review",
    "name: ! 12",
    "name: !!int 3",
    # Collections.
    "name: [a, b]",
    "name: {first: a}",
    "name:\n  - a",
    "name:\n- a",
    "name:\n  first: a",
    "name: &anchor\n  first: a",
    # Keys.
    '"name": quoted-key',
    "'name' : quoted-spaced",
    "name : spaced-key",
    "namespace: other",
    "metadata:\n  name: nested",
    "name: first\nname: second",
    "name: first\nname:",
]


@pytest.mark.parametrize("front_matter", _YAML_NAMES)
def test_skill_name_matches_what_yaml_reads(tmp_path: Path, front_matter: str):
    folder = tmp_path / "folder-name"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        f"---\n{front_matter}\n---\n# Skill\n", encoding="utf-8"
    )
    value = yaml.safe_load(front_matter).get("name")
    # A value that YAML reads as anything but a non-empty string, such as
    # null, a number, a boolean, a date, a sequence, or a mapping, is no name,
    # so the folder name is used. Hermes skips such names the same way, so
    # this deliberately differs from str(value).
    expected = value if isinstance(value, str) and value else "folder-name"

    load = load_directories([{"skill": folder}], rules=IMMEDIATE, duplicates=None)

    assert [skill.name for skill in load.selected] == [expected]


@pytest.mark.parametrize(
    ("front_matter", "name"),
    [
        ('name: "unclosed', '"unclosed'),
        ("name: 'unclosed\n  still open", "'unclosed"),
        ("name: a: b", "a: b"),
        ("name:\ttabbed", "tabbed"),
        ("name: @at # comment", "@at"),
        ("name: |0\n  review", "folder-name"),
        ("name: 'review' trailing", "'review' trailing"),
        ("name: code # comment\n  review", "code"),
    ],
)
def test_invalid_yaml_name_uses_the_text_on_its_line(
    tmp_path: Path, front_matter: str, name: str
):
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(front_matter)
    folder = tmp_path / "folder-name"
    folder.mkdir()
    (folder / "SKILL.md").write_text(f"---\n{front_matter}\n---\n", encoding="utf-8")

    load = load_directories([{"skill": folder}], rules=IMMEDIATE, duplicates=None)

    assert [skill.name for skill in load.selected] == [name]


OPENCLAW = DiscoveryRules(
    traversal="immediate_children",
    nested_skills="stop_at_skill",
    max_depth=1,
    name_style="openclaw",
)

# Front matter and the name OpenClaw 2026.9.34 lists for it, checked against
# its own parser: YAML 1.2 core schema, then trimmed strings, String() of
# numbers and booleans, JSON of collections, and the line text for a null.
_OPENCLAW_NAMES = [
    # The cases the end-to-end test checks against the installed OpenClaw.
    ("name: >-\n  folded-name", "folded-name"),
    ("name: |-\n  literal-name", "literal-name"),
    ('name: "escaped\\tname"', "escaped\tname"),
    ("name: plain # comment", "plain"),
    ("name:", "folder-name"),
    ("name: >\n  folded\n  clipped", "folded clipped"),
    ("name: '  quoted name '", "quoted name"),
    ("name: ~", "~"),
    ("name: 42", "42"),
    ("name: yes", "yes"),
    ("name: true", "true"),
    ("name: 2026-01-02", "2026-01-02"),
    ("name: [a, b]", '["a","b"]'),
    # Strings, trimmed as JavaScript trims them.
    ("name: review", "review"),
    ("name: review:v2", "review:v2"),
    ('name: "  padded \\t"', "padded"),
    ("name: |\n  literal\n", "literal"),
    ('name: "\\u3000wide\\u00a0"', "wide"),
    ('name: "\\x85kept"', "\x85kept"),
    # YAML 1.1 booleans, sexagesimals, binary, and timestamps are strings.
    ("name: no", "no"),
    ("name: On", "On"),
    ("name: y", "y"),
    ("name: 1_000", "1_000"),
    ("name: 1:20", "1:20"),
    ("name: 0b11", "0b11"),
    ("name: 2026-01-02 10:00:00", "2026-01-02 10:00:00"),
    # YAML 1.2 booleans and numbers, as String() writes them.
    ("name: True", "true"),
    ("name: FALSE", "false"),
    ("name: 0o17", "15"),
    ("name: 0x1F", "31"),
    ("name: 017", "17"),
    ("name: +5", "5"),
    ("name: -0", "0"),
    ("name: 1.50", "1.5"),
    ("name: 1.", "1"),
    ("name: .5", "0.5"),
    ("name: 1e3", "1000"),
    ("name: 1e21", "1e+21"),
    ("name: 1e-7", "1e-7"),
    ("name: 0.000001", "0.000001"),
    ("name: 12345678901234567890", "12345678901234567000"),
    ("name: .inf", "Infinity"),
    ("name: -.Inf", "-Infinity"),
    ("name: .nan", "NaN"),
    # Tags and anchors.
    ("name: !!str 42", "42"),
    ("name: ! 42", "42"),
    ("name: !!int 42", "42"),
    ("name: !!int '0x10'", "16"),
    ("name: !!int abc", "abc"),
    ("name: !!float 1.50", "1.5"),
    ("name: !!bool yes", "yes"),
    ("name: !foo bar", "bar"),
    ("name: !<tag:yaml.org,2002:int> 7", "7"),
    ("name: &anchor review", "review"),
    ("name: !!str", "folder-name"),
    # A null takes the text on an unquoted key's line, or the indented lines
    # below an empty one; an empty string does not.
    ("name: null # unset", "null # unset"),
    ("name: !!null", "!!null"),
    ("name: &anchor ~", "&anchor ~"),
    ("name:\n  # comment\n\n  below", "below"),
    ("name: ''", "folder-name"),
    ("name: '   '", "folder-name"),
    ('"name": ~', "folder-name"),
    ("name : ~", "folder-name"),
    # Collections become JSON, unless the key's line holds a colon.
    (
        'name: [1, 2.5, -0, 1e21, .inf, true, ~, "a\\"b\\n"]',
        '[1,2.5,0,1e+21,null,true,null,"a\\"b\\n"]',
    ),
    ("name: [ a\n   b , c]", '["a b","c"]'),
    ("name: [a,\n]", '["a"]'),
    ("name: {a}", '{"a":null}'),
    ("name: {a: b}", "{a: b}"),
    ('name: [a, "b:c"]', '[a, "b:c"]'),
    ("name:\n- a\n- b", '["a","b"]'),
    (
        "name:\n  - a\n  -\n  - - x\n    - y\n  - k: v\n    k2: [1]",
        '["a",null,["x","y"],{"k":"v","k2":[1]}]',
    ),
    ("name:\n  a:\n    b: c\n  d: e", '{"a":{"b":"c"},"d":"e"}'),
    ("name:\n  a:\n  - x\n  - y", '{"a":["x","y"]}'),
    ("name:\n  k: v w\n     x", '{"k":"v w x"}'),
    # JavaScript lists integer-like keys first; equal property names merge.
    ("name:\n  b: 1\n  2: x\n  1: y", '{"1":"y","2":"x","b":1}'),
    ("name:\n  z: 1\n  '0': 2\n  -1: 3", '{"0":2,"z":1,"-1":3}'),
    ('name:\n  1: a\n  "1": b', '{"1":"b"}'),
    # JSON escapes control characters and lone surrogates.
    ('name: ["\\x01\\x1f"]', '["\\u0001\\u001f"]'),
    ('name: ["\\ud83d\\ude00", "\\ud83d"]', '["\U0001f600","\\ud83d"]'),
]


@pytest.mark.parametrize(("front_matter", "name"), _OPENCLAW_NAMES)
def test_openclaw_style_reads_names_as_openclaw(
    tmp_path: Path, front_matter: str, name: str
):
    folder = tmp_path / "folder-name"
    folder.mkdir()
    (folder / "SKILL.md").write_text(f"---\n{front_matter}\n---\n", encoding="utf-8")

    load = load_directories([{"skill": folder}], rules=OPENCLAW, duplicates=None)

    assert [skill.name for skill in load.selected] == [name]


@pytest.mark.parametrize(
    ("front_matter", "name"),
    [
        # Values whose OpenClaw text the helper does not derive.
        ("x: &a foo\nname: *a", "folder-name"),
        ("name: !!timestamp 2026-01-02", "folder-name"),
        ("name: !!binary aGk=", "folder-name"),
        ("name: [? a]", "folder-name"),
        ("name:\n  - |\n    block", "folder-name"),
        # OpenClaw rejects front matter that is not valid YAML; the text on
        # the line is kept, as in the pyyaml style.
        ("name: a: b", "a: b"),
        ("name: 'a' trailing", "'a' trailing"),
        ("name:\n  k: v\n  k: w", "folder-name"),
    ],
)
def test_openclaw_style_falls_back_where_it_cannot_match(
    tmp_path: Path, front_matter: str, name: str
):
    folder = tmp_path / "folder-name"
    folder.mkdir()
    (folder / "SKILL.md").write_text(f"---\n{front_matter}\n---\n", encoding="utf-8")

    load = load_directories([{"skill": folder}], rules=OPENCLAW, duplicates=None)

    assert [skill.name for skill in load.selected] == [name]


@pytest.mark.skipif(
    os.name == "nt", reason="Windows folder names cannot end in a space"
)
def test_openclaw_style_trims_the_folder_name_fallback(tmp_path: Path):
    folder = tmp_path / "padded "
    folder.mkdir()
    (folder / "SKILL.md").write_text("---\nname: ''\n---\n", encoding="utf-8")

    pyyaml = load_directories([{"skill": folder}], rules=IMMEDIATE, duplicates=None)
    openclaw = load_directories([{"skill": folder}], rules=OPENCLAW, duplicates=None)

    assert [skill.name for skill in pyyaml.selected] == ["padded "]
    assert [skill.name for skill in openclaw.selected] == ["padded"]


@pytest.mark.parametrize(
    ("front_matter", "pyyaml", "openclaw"),
    [
        ("name: '  spaced '", "  spaced ", "spaced"),
        ("name: yes", "folder-name", "yes"),
        ("name: 42", "folder-name", "42"),
        ("name: ~", "folder-name", "~"),
    ],
)
def test_name_style_defaults_to_pyyaml(
    tmp_path: Path, front_matter: str, pyyaml: str, openclaw: str
):
    folder = tmp_path / "folder-name"
    folder.mkdir()
    (folder / "SKILL.md").write_text(f"---\n{front_matter}\n---\n", encoding="utf-8")

    default = load_directories([{"skill": folder}], rules=IMMEDIATE, duplicates=None)
    chosen = load_directories([{"skill": folder}], rules=OPENCLAW, duplicates=None)

    assert IMMEDIATE.name_style == "pyyaml"
    assert [skill.name for skill in default.selected] == [pyyaml]
    assert [skill.name for skill in chosen.selected] == [openclaw]


@pytest.mark.parametrize(
    ("policy", "selected"), [("first_wins", 0), ("last_wins", 1), ("reject", None)]
)
def test_block_scalar_names_take_part_in_duplicate_policies(
    tmp_path: Path, policy: str, selected: int | None
):
    folded = tmp_path / "folded"
    folded.mkdir()
    (folded / "SKILL.md").write_text("---\nname: >-\n  review\n---\n", "utf-8")
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "SKILL.md").write_text("---\nname: review\n---\n", "utf-8")
    entries = [{"skill": folded}, {"skill": plain}]

    if selected is None:
        with pytest.raises(SkillLoadError, match="configured twice"):
            load_directories(entries, rules=IMMEDIATE, duplicates=policy)
        return
    load = load_directories(entries, rules=IMMEDIATE, duplicates=policy)

    assert [(skill.name, skill.path) for skill in load.selected] == [
        ("review", (folded, plain)[selected].resolve())
    ]


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
    with pytest.raises(ValueError, match="name style"):
        DiscoveryRules(
            traversal="recursive",
            nested_skills="include",
            max_depth=None,
            name_style="yaml",  # type: ignore[arg-type]
        )


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


@pytest.mark.parametrize("policy", ["reject", "first_wins", "last_wins"])
@pytest.mark.parametrize("folder_first", [True, False])
def test_duplicates_inside_one_folder_fail_when_another_entry_reaches_one(
    tmp_path: Path, make_skill: Callable[..., Path], policy: str, folder_first: bool
):
    first = make_skill(tmp_path / "team" / "a", "review").resolve()
    second = make_skill(tmp_path / "team" / "b", "review").resolve()
    entries = [{"folder": tmp_path / "team"}, {"skill": tmp_path / "team" / "b"}]
    if not folder_first:
        entries.reverse()

    with pytest.raises(SkillLoadError, match="appears twice") as caught:
        load_directories(entries, rules=IMMEDIATE, duplicates=policy)

    assert caught.value.reason == "duplicate"
    assert caught.value.field == f"skills.directories[{0 if folder_first else 1}]"
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
