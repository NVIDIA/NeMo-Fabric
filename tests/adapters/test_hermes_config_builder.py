# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free tests for Hermes configuration construction."""

import builtins
import json
import os
import sys
import tomllib
from pathlib import Path

import pytest
from _hermes_skills import loaded_skill
from _hermes_skills import skill_copy
from _hermes_skills import skipped_skill
from _hermes_skills import write_skill
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.skills import CandidateInfo
from nemo_fabric_adapter_contract.skills import SkippedCandidate
from nemo_fabric_adapter_contract.skills import front_matter_reader

if sys.version_info >= (3, 15):
    pytest.skip(
        "Hermes adapter requires Python 3.14 or earlier",
        allow_module_level=True,
    )

from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.hermes import configuration
from nemo_fabric_adapters.hermes import skills as hermes_skills
from nemo_fabric_adapters.hermes import telemetry


@pytest.mark.parametrize("with_observability", [False, True])
def test_dynamic_relay_plugins_reach_hermes_toml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    with_observability: bool,
):
    from nemo_fabric import FabricConfig, _native

    os.environ["XDG_CONFIG_HOME"] = str(tmp_path / "xdg")
    monkeypatch.chdir(tmp_path)
    plugins = {
        "dynamic": [
            {
                "manifest": "/opt/switchyard/relay-plugin.toml",
                "enabled": True,
                "config": {"route": "canary"},
            }
        ]
    }
    relay = {"plugins": plugins}
    if with_observability:
        relay["observability"] = {"atif": {"enabled": True}}
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "dynamic-plugin-test"},
            "harness": {"adapter_id": "nvidia.fabric.hermes"},
            "models": {"default": {"provider": "nvidia", "model": "test-model"}},
            "telemetry": {"providers": {"relay": {}}},
            "relay": relay,
        }
    )
    plan = json.loads(
        _native.plan_config(json.dumps(config.to_mapping()), str(tmp_path))
    )
    resolved = plan["telemetry_plan"]["relay_config"]
    assert resolved["plugins"] == plugins

    wrapper_path = tmp_path / "relay.json"
    wrapper_path.write_text(json.dumps({"relay": {"config": resolved}}))
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(wrapper_path)
    config_path, staged = telemetry.write_hermes_relay_plugin_config(
        {
            "agent_name": "dynamic-plugin-test",
            "base_dir": str(tmp_path),
            "runtime_context": {"runtime_id": "runtime-dynamic"},
        }
    )

    with config_path.open("rb") as stream:
        saved = tomllib.load(stream)
    assert staged["plugins"] == saved["plugins"] == plugins
    assert len(saved["components"]) == int(with_observability)


def test_build_hermes_config_omits_unset_values_without_hermes_agent():
    config = AgentConfig.from_mapping(
        {
            "harness": {"settings": {}},
            "models": {
                "default": {
                    "provider": "nvidia",
                    "model": "nvidia/test-model",
                }
            },
        }
    )

    config = configuration.build_hermes_config(config, workspace=".")

    assert config["model"] == {
        "provider": "nvidia",
        "default": "nvidia/test-model",
    }
    assert config["agent"] == {}


def test_write_hermes_config_round_trips_without_pyyaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    real_import = builtins.__import__

    def import_without_yaml(name: str, *args: object, **kwargs: object) -> object:
        if name == "yaml":
            raise ImportError("No module named yaml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_yaml)
    agent_config = AgentConfig.from_mapping(
        {
            "harness": {"settings": {}},
            "models": {"default": {"provider": "nvidia", "model": "nvidia/test-model"}},
        }
    )

    config_path, config = configuration.write_hermes_config(
        agent_config,
        tmp_path / "hermes-home",
        workspace=".",
    )

    assert json.loads(config_path.read_text(encoding="utf-8")) == config


def _native_controls_config(settings: dict[str, object]) -> AgentConfig:
    return AgentConfig.from_mapping(
        {
            "harness": {"settings": settings},
            "models": {"default": {"provider": "nvidia", "model": "nvidia/test-model"}},
            "tools": {"enabled": ["file"], "blocked": ["terminal"]},
        }
    )


@pytest.mark.parametrize("relay_enabled", [False, True])
@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"auxiliary": {}, "tools": {}},
        {"auxiliary": {"title_generation": {}, "background_review": {}}},
        {"tools": {"tool_search": {}}},
    ],
    ids=["absent", "empty-sections", "empty-tasks", "empty-tool-search"],
)
def test_omitted_native_controls_keep_the_existing_config(
    settings: dict[str, object], relay_enabled: bool
):
    config = configuration.build_hermes_config(
        _native_controls_config(settings),
        workspace=".",
        relay_enabled=relay_enabled,
    )

    # The configuration NeMo Fabric wrote before native controls existed.
    expected = {
        "model": {"provider": "nvidia", "default": "nvidia/test-model"},
        "agent": {"disabled_toolsets": ["terminal"]},
        "terminal": {"backend": "local", "cwd": ".", "timeout": 60},
        "platform_toolsets": {"cli": ["file"]},
    }
    if relay_enabled:
        expected["auxiliary"] = {"title_generation": {"enabled": False}}
    assert config == expected


@pytest.mark.parametrize("enabled", [True, False])
def test_authored_auxiliary_controls_reach_native_config(enabled: bool):
    config = configuration.build_hermes_config(
        _native_controls_config(
            {
                "auxiliary": {
                    "title_generation": {"enabled": enabled},
                    "background_review": {"enabled": enabled},
                }
            }
        ),
        workspace=".",
    )

    assert config["auxiliary"] == {
        "title_generation": {"enabled": enabled},
        "background_review": {"enabled": enabled},
    }
    assert "tools" not in config


@pytest.mark.parametrize("value", [True, False, "auto", "on", "off"])
def test_tool_search_value_passes_through_as_authored(value: object):
    config = configuration.build_hermes_config(
        _native_controls_config({"tools": {"tool_search": {"enabled": value}}}),
        workspace=".",
    )

    assert config["tools"] == {"tool_search": {"enabled": value}}
    # Normalized tool selection stays in its own native sections.
    assert config["platform_toolsets"] == {"cli": ["file"]}
    assert config["agent"]["disabled_toolsets"] == ["terminal"]
    assert "auxiliary" not in config


@pytest.mark.parametrize(
    ("title_generation", "relay_enabled", "expected"),
    [
        pytest.param(True, True, {"enabled": True}, id="explicit-true-relay"),
        pytest.param(False, True, {"enabled": False}, id="explicit-false-relay"),
        pytest.param(True, False, {"enabled": True}, id="explicit-true-no-relay"),
        pytest.param(False, False, {"enabled": False}, id="explicit-false-no-relay"),
        pytest.param(None, True, {"enabled": False}, id="omitted-relay"),
        pytest.param(None, False, None, id="omitted-no-relay"),
    ],
)
def test_title_generation_precedence_over_relay(
    title_generation: bool | None,
    relay_enabled: bool,
    expected: dict[str, bool] | None,
):
    auxiliary: dict[str, object] = {"background_review": {"enabled": False}}
    if title_generation is not None:
        auxiliary["title_generation"] = {"enabled": title_generation}

    config = configuration.build_hermes_config(
        _native_controls_config({"auxiliary": auxiliary}),
        workspace=".",
        relay_enabled=relay_enabled,
    )

    # The Relay default must not replace other authored auxiliary controls.
    assert config["auxiliary"]["background_review"] == {"enabled": False}
    assert config["auxiliary"].get("title_generation") == expected


@pytest.fixture(name="skill_tree")
def skill_tree_fixture(tmp_path: Path) -> Path:
    """Skill folders shaped like the #580 cases."""

    write_skill(tmp_path / "team" / "review")
    write_skill(tmp_path / "team" / "debugging")
    write_skill(tmp_path / "team" / "debugging" / "collect-logs")
    (tmp_path / "team" / "drafts").mkdir()
    (tmp_path / "team" / "README.md").write_text("Not a skill\n", encoding="utf-8")
    write_skill(tmp_path / "opt" / "release-review")
    write_skill(tmp_path / "opt" / "review")
    write_skill(tmp_path / "rw" / "notes", name="team-notes")
    (tmp_path / "empty").mkdir()
    return tmp_path.resolve()


def _skills_config(skills: dict[str, object]) -> AgentConfig:
    return AgentConfig.from_mapping(
        {
            "models": {"default": {"provider": "nvidia", "model": "nvidia/test-model"}},
            "skills": skills,
        }
    )


def _inventory(root: Path) -> dict[str, Path]:
    return {
        name: root / "bundled" / name
        for name in ("arxiv", "github", "hermes-agent", "pdf")
    }


def _plan(
    tmp_path: Path, skills: dict[str, object], *, home: Path | None = None
) -> hermes_skills.SkillPlan:
    """Plan ``skills`` with the bundled inventory of :func:`_inventory`, and
    ``home`` (by default ``tmp_path``) as the Hermes home."""
    return hermes_skills.plan_skills(
        _skills_config(skills),
        inventory=lambda: _inventory(tmp_path),
        hermes_home=tmp_path if home is None else home,
    )


@pytest.mark.parametrize(
    ("entries", "external", "loaded", "empty"),
    [
        pytest.param(
            [{"folder": "team"}],
            ["team"],
            [
                ("debugging", "skills.directories[0]"),
                # Hermes also loads a skill nested in a skill's folder.
                ("collect-logs", "skills.directories[0]"),
                ("review", "skills.directories[0]"),
            ],
            [],
            id="folder",
        ),
        pytest.param(
            [{"skill": "opt/release-review"}, {"skill": "team/review"}],
            ["opt/release-review", "team/review"],
            [
                ("release-review", "skills.directories[0]"),
                ("review", "skills.directories[1]"),
            ],
            [],
            id="chosen-skills",
        ),
        pytest.param(
            [{"folder": "rw"}, {"skill": "opt/release-review"}],
            ["rw", "opt/release-review"],
            [
                ("team-notes", "skills.directories[0]"),
                ("release-review", "skills.directories[1]"),
            ],
            [],
            id="mixed",
        ),
        pytest.param(
            [{"folder": "empty"}],
            ["empty"],
            [],
            ["skills.directories[0]"],
            id="empty-folder",
        ),
        pytest.param([], [], [], [], id="no-entries"),
    ],
)
def test_directories_map_entry_folders_to_external_dirs(
    skill_tree: Path,
    entries: list[dict[str, str]],
    external: list[str],
    loaded: list[tuple[str, str]],
    empty: list[str],
):
    agent_config = _skills_config(
        {
            "directories": [
                {kind: str(skill_tree / path) for kind, path in entry.items()}
                for entry in entries
            ]
        }
    )

    plan = hermes_skills.plan_skills(agent_config)
    config = configuration.build_hermes_config(
        agent_config, workspace=".", skill_plan=plan
    )

    assert config.get("skills", {}) == (
        {"external_dirs": [str(skill_tree / path) for path in external]}
        if external
        else {}
    )
    assert plan.load is not None
    assert [(skill.name, skill.field) for skill in plan.load.selected] == loaded
    assert plan.load.empty_entries == empty


def test_paths_pass_through_unchanged(tmp_path: Path):
    """skills.paths keeps its unchecked mapping, even for a missing path."""
    missing = tmp_path / "missing"
    agent_config = _skills_config({"paths": [str(missing), "relative/skill"]})

    plan = hermes_skills.plan_skills(agent_config)
    without_plan = configuration.build_hermes_config(agent_config, workspace=".")
    with_plan = configuration.build_hermes_config(
        agent_config, workspace=".", skill_plan=plan
    )

    assert not plan.structured
    assert without_plan == with_plan
    assert without_plan["skills"] == {"external_dirs": [str(missing), "relative/skill"]}
    assert "skills_disabled" not in configuration.summarize_hermes_config(with_plan)


def test_directories_require_a_skill_plan(skill_tree: Path):
    agent_config = _skills_config(
        {"directories": [{"folder": str(skill_tree / "team")}]}
    )

    with pytest.raises(ValueError, match="need a skill plan"):
        configuration.build_hermes_config(agent_config, workspace=".")


@pytest.mark.parametrize(
    ("policy", "external", "loaded_review"),
    [
        pytest.param("first_wins", ["team/debugging", "team/review"], "team/review"),
        pytest.param("last_wins", ["team/debugging", "opt/review"], "opt/review"),
    ],
)
def test_explicit_policy_passes_each_selected_skill(
    skill_tree: Path, policy: str, external: list[str], loaded_review: str
):
    agent_config = _skills_config(
        {
            "directories": [
                {"folder": str(skill_tree / "team")},
                {"skill": str(skill_tree / "opt" / "review")},
            ],
            "duplicates": policy,
        }
    )

    plan = hermes_skills.plan_skills(agent_config)
    config = configuration.build_hermes_config(
        agent_config, workspace=".", skill_plan=plan
    )

    # The nested collect-logs skill loads through its parent's folder.
    assert config["skills"]["external_dirs"] == [
        str(skill_tree / path) for path in external
    ]
    assert plan.load is not None
    [record] = plan.load.duplicates
    assert record.name == "review"
    assert record.outcome == "selected"
    assert record.selected == skill_tree / loaded_review


def test_reject_policy_fails_naming_both_entries(skill_tree: Path):
    agent_config = _skills_config(
        {
            "directories": [
                {"folder": str(skill_tree / "team")},
                {"skill": str(skill_tree / "opt" / "review")},
            ],
            "duplicates": "reject",
        }
    )

    with pytest.raises(lifecycle.LifecycleError) as caught:
        hermes_skills.plan_skills(agent_config)

    assert caught.value.code == "hermes_skill_invalid"
    assert caught.value.metadata == {
        "field": "skills.directories[1]",
        "reason": "duplicate",
        "path": str(skill_tree / "team" / "review"),
        "paths": [
            str(skill_tree / "team" / "review"),
            str(skill_tree / "opt" / "review"),
        ],
    }


@pytest.mark.parametrize("policy", ["first_wins", "last_wins"])
def test_policy_fails_when_a_shadowed_copy_is_nested_in_a_selected_skill(
    tmp_path: Path, policy: str
):
    """Hermes loads skills nested in a skill folder, so that copy cannot be dropped."""
    outer = write_skill(tmp_path / "outer")
    nested = write_skill(tmp_path / "outer" / "tools" / "lint")
    other = write_skill(tmp_path / "other" / "lint")
    # The policy selects the other copy of lint; outer still loads.
    entries = [{"skill": str(outer)}, {"skill": str(other)}]
    if policy == "first_wins":
        entries = [{"skill": str(other)}, {"skill": str(outer)}]
    agent_config = _skills_config({"directories": entries, "duplicates": policy})

    with pytest.raises(lifecycle.LifecycleError) as caught:
        hermes_skills.plan_skills(agent_config)

    assert caught.value.code == "hermes_skill_invalid"
    assert caught.value.metadata["reason"] == "duplicate"
    assert caught.value.metadata["paths"] == [str(nested), str(outer)]


def test_native_policy_keeps_every_copy_and_records_the_clash(skill_tree: Path):
    agent_config = _skills_config(
        {
            "directories": [
                {"folder": str(skill_tree / "team")},
                {"skill": str(skill_tree / "opt" / "review")},
            ]
        }
    )

    plan = hermes_skills.plan_skills(agent_config)

    assert plan.external_dirs == [
        str(skill_tree / "team"),
        str(skill_tree / "opt" / "review"),
    ]
    assert plan.load is not None
    [record] = plan.load.duplicates
    assert (record.name, record.outcome, record.selected) == (
        "review",
        "ambiguous",
        None,
    )


@pytest.mark.parametrize(
    ("entry", "code", "reason", "field"),
    [
        ({"folder": "missing"}, "hermes_skill_not_found", "not_found", "folder"),
        ({"skill": "team/README.md"}, "hermes_skill_invalid", "invalid", "skill"),
        ({"skill": "team/drafts"}, "hermes_skill_invalid", "invalid", "skill"),
    ],
    ids=["missing", "file", "no-skill-file"],
)
def test_entry_failures_map_to_hermes_codes(
    skill_tree: Path, entry: dict[str, str], code: str, reason: str, field: str
):
    [(kind, path)] = entry.items()
    agent_config = _skills_config(
        {
            "directories": [
                {"folder": str(skill_tree / "team")},
                {kind: str(skill_tree / path)},
            ]
        }
    )

    with pytest.raises(lifecycle.LifecycleError) as caught:
        hermes_skills.plan_skills(agent_config)

    assert caught.value.code == code
    assert caught.value.metadata == {
        "field": f"skills.directories[1].{field}",
        "reason": reason,
        "path": str(skill_tree / path),
    }


@pytest.mark.parametrize(
    ("bundled", "excluded", "selected"),
    [
        pytest.param({}, [], ["arxiv", "github", "hermes-agent", "pdf"], id="enable"),
        pytest.param(
            {"include": ["hermes-agent", "pdf"]},
            ["arxiv", "github"],
            ["hermes-agent", "pdf"],
            id="include",
        ),
        # Hermes always loads hermes-agent, so the smallest selection names it.
        pytest.param(
            {"include": ["hermes-agent"]},
            ["arxiv", "github", "pdf"],
            ["hermes-agent"],
            id="include-essential",
        ),
        pytest.param(
            {"exclude": ["arxiv"]},
            ["arxiv"],
            ["github", "hermes-agent", "pdf"],
            id="exclude",
        ),
        pytest.param(
            {"exclude": []},
            [],
            ["arxiv", "github", "hermes-agent", "pdf"],
            id="exclude-none",
        ),
    ],
)
def test_bundled_selection_plans_the_skills_to_stage(
    tmp_path: Path,
    bundled: dict[str, object],
    excluded: list[str],
    selected: list[str],
):
    agent_config = _skills_config({"bundled": bundled})

    plan = _plan(tmp_path, {"bundled": bundled}, home=tmp_path / "home")
    config = configuration.build_hermes_config(
        agent_config, workspace=".", skill_plan=plan
    )

    assert plan.selected_bundled == selected
    assert plan.excluded_bundled == excluded
    # The adapter copies the bundled skills into the home whenever bundled is set.
    assert plan.sync is True
    # Left-out skills never reach the home, so Hermes's skills.disabled, which
    # would hide an external or local skill of the same name, stays unset.
    assert config.get("skills", {}) == {}
    assert "skills_disabled" not in configuration.summarize_hermes_config(config)


def test_bundled_names_provided_by_external_skills_are_known(tmp_path: Path):
    """The report tells a bundled skill an external skill provides apart."""
    write_skill(tmp_path / "team" / "github")
    write_skill(tmp_path / "team" / "renamed", name="pdf")
    skills = {
        "directories": [{"folder": str(tmp_path / "team")}],
        "bundled": {"include": ["hermes-agent"]},
    }

    plan = _plan(tmp_path, skills, home=tmp_path / "home")
    config = configuration.build_hermes_config(
        _skills_config(skills), workspace=".", skill_plan=plan
    )

    assert config["skills"] == {"external_dirs": [str(tmp_path.resolve() / "team")]}
    assert plan.external_names == {"github", "renamed", "pdf"}
    assert plan.excluded_bundled == ["arxiv", "github", "pdf"]


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
def test_bundled_names_provided_by_skill_paths_are_known(
    tmp_path: Path, relative: bool
):
    """skills.paths is unchecked, but its skills' names are still read."""
    home = tmp_path / "home"
    mine = home / "mine" if relative else tmp_path / "mine"
    write_skill(mine / "github")
    write_skill(mine / "renamed", name="pdf")
    # Hermes's scan skips a skill's support folders.
    write_skill(mine / "github" / "scripts" / "arxiv")
    skills = {
        "paths": ["mine" if relative else str(mine)],
        "bundled": {"include": ["hermes-agent"]},
    }

    plan = _plan(tmp_path, skills, home=home)
    config = configuration.build_hermes_config(
        _skills_config(skills), workspace=".", skill_plan=plan
    )

    assert config["skills"] == {"external_dirs": ["mine" if relative else str(mine)]}
    assert plan.external_names == {"github", "renamed", "pdf"}


def test_stage_bundled_copies_only_the_selected_skills(tmp_path: Path):
    """The staged folder has the installation's layout, without left-out skills."""
    root = tmp_path / "bundled"
    write_skill(root / "research" / "arxiv")
    write_skill(root / "research" / "arxiv" / "nested")
    (root / "research" / "arxiv" / "scripts").mkdir()
    (root / "research" / "arxiv" / "scripts" / "run.py").write_text("x", "utf-8")
    write_skill(root / "devops" / "github")
    write_skill(root / "hermes-agent")
    (root / "research" / "DESCRIPTION.md").write_text("Research", "utf-8")
    (root / "devops" / "DESCRIPTION.md").write_text("DevOps", "utf-8")
    inventory = {
        "arxiv": root / "research" / "arxiv",
        "nested": root / "research" / "arxiv" / "nested",
        "github": root / "devops" / "github",
        "hermes-agent": root / "hermes-agent",
    }
    agent_config = _skills_config({"bundled": {"exclude": ["github", "nested"]}})
    plan = hermes_skills.plan_skills(
        agent_config, inventory=lambda: inventory, hermes_home=tmp_path / "home"
    )
    assert plan.bundled_root == root
    stage = tmp_path / "stage"

    hermes_skills.stage_bundled(plan, stage)

    files = sorted(
        path.relative_to(stage).as_posix()
        for path in stage.rglob("*")
        if path.is_file()
    )
    assert files == [
        "hermes-agent/SKILL.md",
        "research/DESCRIPTION.md",
        "research/arxiv/SKILL.md",
        "research/arxiv/scripts/run.py",
    ]


def test_remove_copies_drops_only_their_manifest_entries(tmp_path: Path):
    """A removed copy leaves the manifest, so a later sync copies it again."""
    home = tmp_path / "home"
    skills_dir = home / "skills"
    copy_path = write_skill(skills_dir / "research" / "arxiv")
    kept = write_skill(skills_dir / "devops" / "github")
    manifest = skills_dir / ".bundled_manifest"
    manifest.write_text("arxiv:aaa\ngithub:bbb\nhermes-agent:ccc\n", "utf-8")
    plan = hermes_skills.SkillPlan(stale_copies=[("arxiv", copy_path)])

    hermes_skills.remove_copies(plan, home)

    assert not copy_path.exists()
    assert kept.is_dir()
    assert manifest.read_text("utf-8") == "github:bbb\nhermes-agent:ccc\n"


def test_remove_copies_refuses_a_folder_outside_the_home(tmp_path: Path):
    outside = write_skill(tmp_path / "elsewhere" / "arxiv")
    plan = hermes_skills.SkillPlan(stale_copies=[("arxiv", outside)])

    with pytest.raises(ValueError, match="not inside"):
        hermes_skills.remove_copies(plan, tmp_path / "home")

    assert outside.is_dir()


@pytest.mark.parametrize(
    ("bundled", "field", "name", "reason"),
    [
        (
            {"include": ["hermes-agent", "pdf", "unknown"]},
            "skills.bundled.include",
            "unknown",
            "unknown_bundled_skill",
        ),
        (
            {"exclude": ["unknown"]},
            "skills.bundled.exclude",
            "unknown",
            "unknown_bundled_skill",
        ),
        (
            {"exclude": ["hermes-agent"]},
            "skills.bundled.exclude",
            "hermes-agent",
            "essential_bundled_skill",
        ),
        # Hermes always loads hermes-agent, so include must name it, and
        # nothing can turn every bundled skill off.
        (
            {"include": []},
            "skills.bundled.include",
            "hermes-agent",
            "essential_bundled_skill",
        ),
        (
            {"include": ["pdf"]},
            "skills.bundled.include",
            "hermes-agent",
            "essential_bundled_skill",
        ),
        (
            {"enabled": False},
            "skills.bundled.enabled",
            "hermes-agent",
            "essential_bundled_skill",
        ),
        # An unknown name is reported before a missing essential skill.
        (
            {"include": ["unknown"]},
            "skills.bundled.include",
            "unknown",
            "unknown_bundled_skill",
        ),
    ],
)
def test_bundled_names_are_checked(
    tmp_path: Path, bundled: dict[str, object], field: str, name: str, reason: str
):
    with pytest.raises(lifecycle.LifecycleError) as caught:
        _plan(tmp_path, {"bundled": bundled})

    assert caught.value.code == "hermes_skill_invalid"
    assert caught.value.metadata == {"field": field, "reason": reason, "name": name}


def test_bundled_inventory_without_an_essential_skill_allows_any_include(
    tmp_path: Path,
):
    """Only essential skills the installed Hermes ships are required."""
    inventory = {"pdf": tmp_path / "bundled" / "pdf"}
    agent_config = _skills_config({"bundled": {"include": []}})

    plan = hermes_skills.plan_skills(
        agent_config, inventory=lambda: inventory, hermes_home=tmp_path / "home"
    )

    assert plan.selected_bundled == []
    assert plan.excluded_bundled == ["pdf"]


def test_folder_entry_that_is_a_skill_loads_its_root(tmp_path: Path):
    """Hermes scans a folder from the folder itself, so a root SKILL.md is a skill."""
    pack = write_skill(tmp_path / "pack")
    write_skill(pack / "child")
    agent_config = _skills_config({"directories": [{"folder": str(pack)}]})

    plan = hermes_skills.plan_skills(agent_config)

    assert plan.external_dirs == [str(pack)]
    assert plan.load is not None
    assert [(skill.name, skill.path) for skill in plan.load.selected] == [
        ("pack", pack),
        ("child", pack / "child"),
    ]
    assert plan.load.empty_entries == []


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="needs POSIX permissions enforced for a non-root user",
)
def test_folder_entry_with_an_unreadable_root_skill_skips_it(tmp_path: Path):
    pack = write_skill(tmp_path / "pack")
    (pack / "SKILL.md").chmod(0)
    agent_config = _skills_config({"directories": [{"folder": str(pack)}]})

    try:
        plan = hermes_skills.plan_skills(agent_config)
    finally:
        (pack / "SKILL.md").chmod(0o644)

    # A folder entry's own SKILL.md is one of its candidates, like any other.
    assert plan.load is not None
    assert plan.load.skipped == [
        SkippedCandidate(pack, "skills.directories[0]", "unreadable")
    ]
    assert plan.load.empty_entries == ["skills.directories[0]"]


@pytest.mark.parametrize("policy", [None, "first_wins", "last_wins"])
def test_skill_support_folders_are_not_scanned(tmp_path: Path, policy: str | None):
    """Hermes never scans a skill's scripts, references, templates, or assets."""
    team = tmp_path.resolve() / "team"
    write_skill(team / "a", name="dup")
    write_skill(team / "b")
    for support in ("scripts", "references", "templates", "assets"):
        write_skill(team / "b" / support / "x", name="dup")
    # A support folder name below a folder without SKILL.md is scanned.
    write_skill(team / "scripts" / "tool")
    skills: dict[str, object] = {"directories": [{"folder": str(team)}]}
    if policy is not None:
        skills["duplicates"] = policy
    agent_config = _skills_config(skills)

    plan = hermes_skills.plan_skills(agent_config)

    assert plan.load is not None
    assert [(skill.name, skill.path) for skill in plan.load.selected] == [
        ("dup", team / "a"),
        ("b", team / "b"),
        ("tool", team / "scripts" / "tool"),
    ]
    assert plan.load.duplicates == []
    expected = (
        [team]
        if policy is None
        else [team / "a", team / "b", team / "scripts" / "tool"]
    )
    assert plan.external_dirs == [str(path) for path in expected]


@pytest.mark.parametrize(
    ("entries", "external"),
    [
        pytest.param(
            [{"folder": "team"}, {"skill": "team/review"}],
            ["team"],
            id="skill-in-folder",
        ),
        pytest.param(
            [{"skill": "team/review"}, {"folder": "team"}], ["team"], id="folder-after"
        ),
        pytest.param(
            [{"folder": "team"}, {"folder": "team/nested"}],
            ["team"],
            id="nested-folder",
        ),
        # Hermes's scan of team never reaches these, so they stay roots.
        pytest.param(
            [{"folder": "team"}, {"skill": "team/node_modules/vendored"}],
            ["team", "team/node_modules/vendored"],
            id="excluded-folder",
        ),
        pytest.param(
            [{"folder": "team"}, {"skill": "team/review/scripts/helper"}],
            ["team", "team/review/scripts/helper"],
            id="support-folder",
        ),
    ],
)
def test_native_policy_passes_each_root_once(
    tmp_path: Path, entries: list[dict[str, str]], external: list[str]
):
    """Hermes refuses a skill it reaches through two roots as ambiguous."""
    root = tmp_path.resolve()
    write_skill(root / "team" / "review")
    write_skill(root / "team" / "nested" / "debug")
    write_skill(root / "team" / "node_modules" / "vendored")
    write_skill(root / "team" / "review" / "scripts" / "helper")
    agent_config = _skills_config(
        {
            "directories": [
                {kind: str(root / path) for kind, path in entry.items()}
                for entry in entries
            ]
        }
    )

    plan = hermes_skills.plan_skills(agent_config)

    assert plan.external_dirs == [str(root / path) for path in external]
    assert plan.load is not None
    # Each skill loads once, attributed to the first entry that reaches it.
    assert plan.load.duplicates == []
    assert {plan.root_fields[root / path] for path in external} <= {
        "skills.directories[0]",
        "skills.directories[1]",
    }


def test_reader_falls_back_to_the_helper_without_hermes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The adapter runs beside Hermes; without it, the helper's reader applies."""
    real_import = builtins.__import__

    def import_without_hermes(name: str, *args: object, **kwargs: object) -> object:
        if name.partition(".")[0] in ("agent", "tools"):
            raise ImportError(f"No module named {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_hermes)
    plain = write_skill(tmp_path / "plain", name="review")
    folded = write_skill(
        tmp_path / "folded", text="---\nname: >-\n  notes\n---\nBody\n"
    )

    infos = hermes_skills.hermes_reader()([plain, folded])

    assert infos == front_matter_reader()([plain, folded])
    assert infos[plain] == CandidateInfo("review")
    # Hermes's parser reads a block scalar; the helper's reader cannot.
    assert infos[folded].name_uncertain


# Observation: Hermes's live catalog decides visibility, shadowing, and
# ambiguity. Its rows are written by hand here; the adapter tests run Hermes.


def _catalog_row(
    root: Path,
    folder: Path,
    *,
    tier: int,
    status: str,
    name: object = None,
    visible: bool = True,
) -> dict[str, object]:
    return {
        "name": folder.name if name is None else name,
        "tier": tier,
        "root": root,
        "path": folder / "SKILL.md",
        "status": status,
        "visible": visible,
    }


def test_observation_reports_hermes_outcomes(skill_tree: Path, tmp_path: Path):
    """Hermes's catalog decides what loads; the report follows it."""
    team, opt = skill_tree / "team", skill_tree / "opt" / "review"
    home = (tmp_path / "home" / "skills").resolve()
    write_skill(home / "debugging")
    write_skill(home / "pdf")
    write_skill(home / "hermes-agent")
    # Hermes's sync records the bundled skills it copied into the home.
    (home / ".bundled_manifest").write_text("pdf:0\nhermes-agent:0\n", encoding="utf-8")
    plan = _plan(
        tmp_path,
        {
            "directories": [{"folder": str(team)}, {"skill": str(opt)}],
            "bundled": {"include": ["arxiv", "hermes-agent", "pdf"]},
        },
    )
    catalog = [
        _catalog_row(team, team / "review", tier=3, status="ambiguous"),
        _catalog_row(opt, opt, tier=3, status="ambiguous"),
        _catalog_row(team, team / "debugging", tier=3, status="shadowed"),
        _catalog_row(home, home / "debugging", tier=1, status="unique"),
        _catalog_row(
            team, team / "debugging" / "collect-logs", tier=3, status="unique"
        ),
        _catalog_row(home, home / "pdf", tier=1, status="unique"),
        _catalog_row(home, home / "hermes-agent", tier=1, status="unique"),
    ]

    report = hermes_skills.observe(
        plan, catalog, sync_result={"suppressed": ["arxiv"]}
    ).report()

    assert report == {
        "loaded": [loaded_skill("collect-logs", team / "debugging" / "collect-logs")],
        "duplicates": [
            {
                "name": "review",
                "outcome": "ambiguous",
                "copies": [
                    skill_copy("external", team / "review", 0),
                    skill_copy("external", opt, 1),
                ],
            },
            {
                "name": "debugging",
                "outcome": "shadowed",
                "selected": str(home / "debugging"),
                "copies": [
                    skill_copy("local", home / "debugging"),
                    skill_copy("external", team / "debugging", 0),
                ],
            },
        ],
        "bundled": {
            "selected": ["hermes-agent", "pdf"],
            "unavailable": [{"name": "arxiv", "reason": "suppressed"}],
            "essential": ["hermes-agent"],
        },
    }


@pytest.mark.parametrize("own_place", ["mine/pdf", "pdf"], ids=["elsewhere", "beside"])
def test_observation_reports_home_clashes_without_an_external_copy(
    tmp_path: Path, own_place: str
):
    """A bundled copy and the agent's own skill in the home share a name."""
    home = (tmp_path / "home" / "skills").resolve()
    bundled_copy = write_skill(home / "bundled-cat" / "pdf")
    own = write_skill(home / own_place)
    (home / ".bundled_manifest").write_text("pdf:0\n", encoding="utf-8")
    plan = hermes_skills.plan_skills(_skills_config({"directories": []}))
    plan.inventory = {"pdf": tmp_path / "bundled" / "bundled-cat" / "pdf"}
    plan.bundled_root = tmp_path / "bundled"
    # Hermes refuses two same-tier copies of a name.
    catalog = [
        _catalog_row(home, own, tier=1, status="ambiguous"),
        _catalog_row(home, bundled_copy, tier=1, status="ambiguous"),
    ]

    report = hermes_skills.observe(plan, catalog).report()

    assert report == {
        "duplicates": [
            {
                "name": "pdf",
                "outcome": "ambiguous",
                "copies": [
                    skill_copy("local", own),
                    skill_copy("bundled", bundled_copy),
                ],
            }
        ]
    }


def test_observation_reports_a_project_skill_over_a_home_skill(tmp_path: Path):
    """A higher Hermes tier wins a name with no configured copy."""
    home = (tmp_path / "home" / "skills").resolve()
    project = (tmp_path / "project" / ".hermes" / "skills").resolve()
    winner = write_skill(project / "notes")
    loser = write_skill(home / "notes")
    plan = hermes_skills.plan_skills(_skills_config({"directories": []}))
    catalog = [
        _catalog_row(project, winner, tier=0, status="unique"),
        _catalog_row(home, loser, tier=1, status="shadowed"),
    ]

    report = hermes_skills.observe(plan, catalog).report()

    assert report == {
        "duplicates": [
            {
                "name": "notes",
                "outcome": "selected",
                "selected": str(winner),
                "copies": [skill_copy("local", winner), skill_copy("local", loser)],
            }
        ]
    }


def test_observation_reports_clashes_of_skill_paths_with_bundled(tmp_path: Path):
    """With skills.paths and bundled, a clash Hermes's catalog shows is reported."""
    home = (tmp_path / "home" / "skills").resolve()
    mine = (tmp_path / "mine").resolve()
    own = write_skill(mine / "notes")
    local = write_skill(home / "notes")
    plan = _plan(
        tmp_path, {"paths": [str(mine)], "bundled": {"include": ["hermes-agent"]}}
    )
    catalog = [
        _catalog_row(home, local, tier=1, status="unique"),
        _catalog_row(mine, own, tier=3, status="shadowed"),
    ]

    report = hermes_skills.observe(plan, catalog).report()

    assert report is not None
    assert report["duplicates"] == [
        {
            "name": "notes",
            "outcome": "shadowed",
            "selected": str(local),
            "copies": [skill_copy("local", local), skill_copy("external", own)],
        }
    ]


def test_observation_matches_copies_by_the_name_hermes_reads(tmp_path: Path):
    """A home skill shadows an external block-scalar-named copy.

    Hermes's sync index reads ``name: >-`` as ``>-``; Hermes's catalog reads the
    folded value. Clashes and origins follow the catalog and the copy's place.
    """
    home = (tmp_path / "home" / "skills").resolve()
    team = (tmp_path / "team").resolve()
    folded = write_skill(
        team / "folded",
        text="---\nname: >-\n  notes\ndescription: test skill\n---\nBody\n",
    )
    local = write_skill(home / "notes")
    plan = hermes_skills.plan_skills(
        _skills_config({"directories": [{"folder": str(team)}]})
    )
    catalog = [
        _catalog_row(home, local, tier=1, status="unique"),
        _catalog_row(team, folded, tier=3, status="shadowed", name="notes"),
    ]

    report = hermes_skills.observe(plan, catalog).report()

    assert report == {
        "duplicates": [
            {
                "name": "notes",
                "outcome": "shadowed",
                "selected": str(local),
                "copies": [
                    skill_copy("local", local),
                    skill_copy("external", folded, 0),
                ],
            }
        ]
    }


def test_observation_reports_bundled_copies_deferred_to_external(tmp_path: Path):
    team = write_skill(tmp_path / "team" / "github").parent
    plan = _plan(tmp_path, {"directories": [{"folder": str(team)}], "bundled": {}})
    catalog = [_catalog_row(team, team / "github", tier=3, status="unique")]

    report = hermes_skills.observe(
        plan, catalog, sync_result={"shadowed_by_external": ["github"]}
    ).report()

    assert report["duplicates"] == [
        {
            "name": "github",
            "outcome": "selected",
            "selected": str(team / "github"),
            "copies": [
                skill_copy("external", team / "github", 0),
                skill_copy("bundled", tmp_path / "bundled" / "github"),
            ],
        }
    ]
    assert {"name": "github", "reason": "provided_by_external_skill"} in report[
        "bundled"
    ]["unavailable"]


def test_observation_skips_a_skill_missing_from_hermes_catalog(tmp_path: Path):
    """A planned skill that Hermes's catalog does not list does not load."""
    team = write_skill(tmp_path / "team" / "tool").parent
    plan = hermes_skills.plan_skills(
        _skills_config({"directories": [{"folder": str(team)}]})
    )

    report = hermes_skills.observe(plan, []).report()

    assert report == {"skipped": [skipped_skill(team / "tool", "not_loaded_by_hermes")]}


def test_observation_skips_a_skill_hermes_sees_but_does_not_load(tmp_path: Path):
    """A front-matter name that is another skill's folder name is ambiguous."""
    team = tmp_path.resolve() / "team"
    write_skill(team / "foo", name="alpha")
    write_skill(team / "bar", name="foo")
    plan = hermes_skills.plan_skills(
        _skills_config({"directories": [{"folder": str(team)}]})
    )
    catalog = [
        _catalog_row(team, team / "foo", tier=3, status="unique", name="alpha"),
        _catalog_row(team, team / "bar", tier=3, status="ambiguous", name="foo"),
    ]

    report = hermes_skills.observe(plan, catalog).report()

    assert report == {
        "loaded": [loaded_skill("alpha", team / "foo")],
        "skipped": [skipped_skill(team / "bar", "ambiguous")],
    }


def test_observation_skips_external_skills_hermes_hides(tmp_path: Path):
    """A skill Hermes resolves but hides on this platform does not load."""
    team = tmp_path.resolve() / "team"
    home = (tmp_path / "home" / "skills").resolve()
    write_skill(team / "macos-only")
    write_skill(team / "debugging")
    write_skill(home / "debugging")
    plan = hermes_skills.plan_skills(
        _skills_config({"directories": [{"folder": str(team)}]})
    )
    catalog = [
        _catalog_row(team, team / "macos-only", tier=3, status="unique", visible=False),
        _catalog_row(team, team / "debugging", tier=3, status="unique", visible=False),
        _catalog_row(home, home / "debugging", tier=1, status="shadowed"),
    ]

    report = hermes_skills.observe(plan, catalog).report()

    assert "loaded" not in report
    assert report["skipped"] == [
        skipped_skill(team / "debugging", "native_conditions"),
        skipped_skill(team / "macos-only", "native_conditions"),
    ]
    # The configured copy wins the clash, but no copy loads.
    assert report["duplicates"] == [
        {
            "name": "debugging",
            "outcome": "selected",
            "copies": [
                skill_copy("local", home / "debugging"),
                skill_copy("external", team / "debugging", 0),
            ],
        }
    ]


@pytest.mark.parametrize(
    ("listed", "reason"),
    [(True, "native_conditions"), (False, "not_loaded_by_hermes")],
    ids=["hidden", "not-listed"],
)
def test_observation_drops_a_policy_winner_hermes_does_not_load(
    tmp_path: Path, listed: bool, reason: str
):
    """A policy's record names its winner only when Hermes loads and shows it."""
    team = tmp_path.resolve() / "team"
    winner = write_skill(team / "rev", name="review")
    other = write_skill(tmp_path / "opt" / "review")
    plan = hermes_skills.plan_skills(
        _skills_config(
            {
                "directories": [{"folder": str(team)}, {"skill": str(other)}],
                "duplicates": "first_wins",
            }
        )
    )
    assert plan.load is not None
    assert [record.selected for record in plan.load.duplicates] == [winner]
    hidden = _catalog_row(
        team, winner, tier=3, status="unique", name="review", visible=False
    )

    report = hermes_skills.observe(plan, [hidden] if listed else []).report()

    assert report["duplicates"] == [
        {
            "name": "review",
            "outcome": "selected",
            "copies": [
                skill_copy("external", winner, 0),
                skill_copy("external", other, 1),
            ],
        }
    ]
    assert report["skipped"] == [skipped_skill(winner, reason)]


def test_observation_does_not_report_the_agents_own_skill_as_bundled(
    tmp_path: Path,
):
    home = tmp_path / "home"
    skills_dir = (home / "skills").resolve()
    # The bundled manifest lacks pdf: the home's pdf is the agent's own skill.
    write_skill(skills_dir / "pdf")
    write_skill(skills_dir / "hermes-agent")
    (skills_dir / ".bundled_manifest").write_text("hermes-agent:0\n", encoding="utf-8")
    plan = _plan(tmp_path, {"bundled": {"include": ["hermes-agent", "pdf"]}}, home=home)
    catalog = [
        _catalog_row(skills_dir, skills_dir / "pdf", tier=1, status="unique"),
        _catalog_row(skills_dir, skills_dir / "hermes-agent", tier=1, status="unique"),
    ]

    observation = hermes_skills.observe(plan, catalog)

    assert observation.report() == {
        "bundled": {
            "selected": ["hermes-agent"],
            "unavailable": [{"name": "pdf", "reason": "provided_by_local_skill"}],
            "essential": ["hermes-agent"],
        }
    }
    assert observation.bundled_warnings == [
        "bundled skill 'pdf' does not load: provided_by_local_skill"
    ]
