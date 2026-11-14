# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured skills configuration in the Python SDK."""

from __future__ import annotations

from pathlib import Path

import pytest
from nemo_fabric import BundledSkills
from nemo_fabric import FabricConfig
from nemo_fabric import FabricConfigError
from nemo_fabric import HarnessConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import SkillConfig
from nemo_fabric import SkillDirectory
from nemo_fabric.types import _FabricConfigSnapshot
from nemo_fabric.types import _SkillConfig
from pydantic import ValidationError


@pytest.fixture(name="config")
def config_fixture() -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="skills"),
        harness=HarnessConfig(adapter_id="test.fabric.shim"),
    )


def test_structured_skills_serialize_in_schema_shape(config: FabricConfig):
    config.skills = SkillConfig(
        directories=[
            SkillDirectory(folder="/sandbox/team-skills"),
            SkillDirectory(skill=Path("/opt/skills/release-review")),
        ],
        bundled=BundledSkills(exclude=["pdf"]),
        duplicates="last_wins",
    )

    assert config.to_mapping()["skills"] == {
        "paths": [],
        "directories": [
            {"folder": "/sandbox/team-skills"},
            # A Path serializes with the platform's separators.
            {"skill": str(Path("/opt/skills/release-review"))},
        ],
        "bundled": {"enabled": True, "exclude": ["pdf"]},
        "duplicates": "last_wins",
    }
    assert (
        FabricConfig.from_mapping(config.to_mapping()).to_mapping()
        == config.to_mapping()
    )


@pytest.mark.parametrize(
    ("skills", "expected"),
    [
        (SkillConfig(directories=[]), {"directories": []}),
        (SkillConfig(bundled=BundledSkills()), {"bundled": {"enabled": True}}),
        (
            SkillConfig(bundled=BundledSkills(include=[])),
            {"bundled": {"enabled": True, "include": []}},
        ),
        (
            SkillConfig(bundled=BundledSkills(enabled=False)),
            {"bundled": {"enabled": False}},
        ),
        (SkillConfig(paths=["./review"]), {"paths": ["./review"]}),
    ],
)
def test_empty_lists_and_unset_fields_stay_distinct(
    skills: SkillConfig, expected: dict
):
    assert skills.to_mapping() == expected


@pytest.mark.parametrize(
    ("bundled", "expected"),
    [
        (BundledSkills(), {"enabled": True}),
        (BundledSkills(include=[]), {"enabled": True, "include": []}),
        (BundledSkills(exclude=[]), {"enabled": True, "exclude": []}),
    ],
)
def test_bundled_skills_mapping_keeps_explicit_empty_lists(
    bundled: BundledSkills, expected: dict
):
    assert bundled.to_mapping() == expected
    assert BundledSkills.model_validate(bundled.to_mapping()) == bundled


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"skill": "/a", "folder": "/b"}, "exactly one of skill or folder"),
        ({}, "exactly one of skill or folder"),
        ({"skill": "/a", "path": "/b"}, "Extra inputs are not permitted"),
        ({"folder": 3}, "valid string"),
        ({"skill": ""}, "non-empty"),
        ({"skill": Path("")}, "non-empty"),
        ({"folder": Path(".")}, "non-empty"),
        ({"skill": None, "folder": None}, "exactly one of skill or folder"),
    ],
)
def test_skill_directory_requires_exactly_one_path(value: dict, message: str):
    with pytest.raises(ValidationError, match=message):
        SkillDirectory.model_validate(value)


def test_skill_directory_accepts_dot_as_a_string():
    # Path("") reads as ".", so only the string spells the current folder.
    assert SkillDirectory(folder=".").folder == "."


@pytest.mark.parametrize(
    "entry", [{"skill": "/a", "folder": None}, {"folder": None, "skill": "/a"}]
)
def test_null_skill_directory_key_is_absent_in_every_binding(entry: dict):
    model = SkillConfig.model_validate({"directories": [entry]})
    snapshot = _SkillConfig.from_mapping({"directories": [entry]})

    assert model.to_mapping() == {"directories": [{"skill": "/a"}]}
    assert snapshot.to_mapping() == {"directories": [{"skill": "/a"}]}


def test_duplicate_policies_share_one_definition():
    from typing import get_args

    from nemo_fabric import types

    assert types._SKILL_DUPLICATE_POLICIES == get_args(types.SkillDuplicatePolicy)
    assert SkillConfig.model_fields["duplicates"].annotation == (
        types.SkillDuplicatePolicy | None
    )


def test_snapshot_bundled_lists_accept_tuples():
    skills = _SkillConfig(bundled={"include": ("review",)})
    excluded = _SkillConfig.from_mapping({"bundled": BundledSkills(exclude=("a",))})

    assert skills.to_mapping() == {"bundled": {"enabled": True, "include": ["review"]}}
    assert excluded.to_mapping() == {"bundled": {"enabled": True, "exclude": ["a"]}}


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"include": [], "exclude": []}, "cannot set both include and exclude"),
        ({"include": ["a"], "exclude": ["b"]}, "cannot set both include and exclude"),
        ({"enabled": False, "include": []}, "enabled=false cannot set include"),
        ({"enabled": False, "exclude": ["a"]}, "enabled=false cannot set include"),
        ({"include": [" "]}, "non-empty strings"),
        ({"exclude": [1]}, "valid string"),
        ({"only": ["a"]}, "Extra inputs are not permitted"),
    ],
)
def test_bundled_skills_reject_conflicting_selection(value: dict, message: str):
    with pytest.raises(ValidationError, match=message):
        BundledSkills.model_validate(value)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (
            {"paths": ["./review"], "directories": []},
            "skills.directories may be set only when skills.paths is empty",
        ),
        ({"duplicates": "reject"}, "skills.duplicates requires skills.directories"),
        ({"directories": [], "duplicates": "newest"}, "duplicates"),
        ({"directories": [{"skill": "/a", "folder": "/b"}]}, "exactly one"),
    ],
)
def test_skill_config_rejects_invalid_combinations(value: dict, message: str):
    with pytest.raises(ValidationError, match=message):
        SkillConfig.model_validate(value)


def test_skill_config_accepts_explicit_empty_paths_with_directories():
    skills = SkillConfig(paths=[], directories=[SkillDirectory(skill="./review")])

    assert skills.to_mapping() == {"directories": [{"skill": "./review"}]}


def test_skill_config_assignment_keeps_rules(config: FabricConfig):
    config.add_skill_path("./review")

    with pytest.raises(ValidationError, match="only when skills.paths is empty"):
        config.skills.directories = []


def test_add_skill_path_refuses_structured_config(config: FabricConfig):
    config.skills = SkillConfig(directories=[SkillDirectory(skill="./review")])

    with pytest.raises(ValueError, match="add a SkillDirectory entry instead"):
        config.add_skill_path("./other")
    assert config.skills.paths == []


@pytest.mark.parametrize(
    "skills",
    [
        SkillConfig(directories=[]),
        SkillConfig(bundled=BundledSkills()),
        SkillConfig(directories=[], duplicates="reject"),
        SkillConfig.model_validate({"acme": {"mode": "strict"}}),
    ],
)
def test_remove_skill_path_keeps_other_skill_settings(
    config: FabricConfig, skills: SkillConfig
):
    config.skills = skills
    expected = skills.to_mapping()

    config.remove_skill_path("./missing")

    assert config.skills is not None
    assert config.skills.to_mapping() == expected


def test_remove_last_skill_path_drops_empty_section(config: FabricConfig):
    config.add_skill_path("./review").remove_skill_path("./review")

    assert config.skills is None


def test_snapshot_mapping_preserves_structured_skills(config: FabricConfig):
    config.skills = SkillConfig(
        directories=[SkillDirectory(folder="/team")],
        bundled=BundledSkills(include=[]),
        duplicates="first_wins",
        acme=True,
    )

    snapshot = _FabricConfigSnapshot.from_mapping(config.to_mapping())

    assert snapshot.skills.to_mapping() == {
        "directories": [{"folder": "/team"}],
        "bundled": {"enabled": True, "include": []},
        "duplicates": "first_wins",
        "acme": True,
    }
    assert snapshot.skills.extra_fields == {"acme": True}
    with pytest.raises(FabricConfigError, match="cannot be combined"):
        snapshot.add_skill_path("./review")


def test_snapshot_mapping_keeps_empty_directories():
    skills = _SkillConfig.from_mapping({"paths": [], "directories": []})

    assert skills.to_mapping() == {"directories": []}


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"paths": ["./a"], "directories": []}, "only when skills.paths is empty"),
        ({"directories": "./a"}, "skills.directories must be a list"),
        ({"directories": [{"skill": "/a", "folder": "/b"}]}, "exactly one"),
        ({"directories": [{"skill": 3}]}, r"skills.directories\[0\].skill"),
        ({"directories": [{"skill": None}]}, "exactly one"),
        ({"directories": [{"skill": "/a", "path": None}]}, "unknown field 'path'"),
        ({"duplicates": "reject"}, "requires skills.directories"),
        ({"directories": [], "duplicates": "newest"}, "must be one of"),
        ({"bundled": {"include": [], "exclude": []}}, "both include and exclude"),
        ({"bundled": {"enabled": False, "include": []}}, "enabled=false"),
        ({"bundled": {"enabled": "no"}}, "must be a boolean"),
        ({"bundled": {"include": [""]}}, "non-empty strings"),
        ({"bundled": {"only": []}}, "unknown field 'only'"),
    ],
)
def test_snapshot_mapping_rejects_invalid_skills(value: dict, message: str):
    with pytest.raises(FabricConfigError, match=message):
        _SkillConfig.from_mapping(value)
