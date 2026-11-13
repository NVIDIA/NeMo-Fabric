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
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.skills import SkippedCandidate

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


def _write_skill(folder: Path, name: str | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    front_matter = f"name: {name}\n" if name else ""
    (folder / "SKILL.md").write_text(
        f"---\n{front_matter}description: test skill\n---\nBody\n", encoding="utf-8"
    )
    return folder


@pytest.fixture(name="skill_tree")
def skill_tree_fixture(tmp_path: Path) -> Path:
    """Skill folders shaped like the #580 cases."""

    _write_skill(tmp_path / "team" / "review")
    _write_skill(tmp_path / "team" / "debugging")
    _write_skill(tmp_path / "team" / "debugging" / "collect-logs")
    (tmp_path / "team" / "drafts").mkdir()
    (tmp_path / "team" / "README.md").write_text("Not a skill\n", encoding="utf-8")
    _write_skill(tmp_path / "opt" / "release-review")
    _write_skill(tmp_path / "opt" / "review")
    _write_skill(tmp_path / "rw" / "notes", name="team-notes")
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
    outer = _write_skill(tmp_path / "outer").resolve()
    nested = _write_skill(tmp_path / "outer" / "tools" / "lint").resolve()
    other = _write_skill(tmp_path / "other" / "lint").resolve()
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
    ("bundled", "disabled", "selected"),
    [
        pytest.param({}, [], ["arxiv", "github", "pdf"], id="enable"),
        pytest.param({"enabled": False}, ["arxiv", "github", "pdf"], [], id="disable"),
        pytest.param({"include": ["pdf"]}, ["arxiv", "github"], ["pdf"], id="include"),
        pytest.param(
            {"include": []}, ["arxiv", "github", "pdf"], [], id="include-none"
        ),
        pytest.param(
            {"exclude": ["arxiv"]}, ["arxiv"], ["github", "pdf"], id="exclude"
        ),
        pytest.param(
            {"exclude": []}, [], ["arxiv", "github", "pdf"], id="exclude-none"
        ),
        # Hermes never disables hermes-agent, so naming it selects nothing more.
        pytest.param(
            {"include": ["hermes-agent"]},
            ["arxiv", "github", "pdf"],
            [],
            id="include-essential",
        ),
    ],
)
def test_bundled_selection_maps_to_skills_disabled(
    tmp_path: Path,
    bundled: dict[str, object],
    disabled: list[str],
    selected: list[str],
):
    agent_config = _skills_config({"bundled": bundled})

    plan = hermes_skills.plan_skills(
        agent_config,
        inventory=lambda: _inventory(tmp_path),
        hermes_home=tmp_path / "home",
    )
    config = configuration.build_hermes_config(
        agent_config, workspace=".", skill_plan=plan
    )

    assert plan.selected_bundled == selected
    # Hermes's sync seeds the essential skills even when none is selected.
    assert plan.sync is True
    assert config.get("skills", {}) == ({"disabled": disabled} if disabled else {})
    summary = configuration.summarize_hermes_config(config)
    assert summary.get("skills_disabled", []) == disabled


def test_bundled_names_provided_by_external_skills_stay_enabled(tmp_path: Path):
    """skills.disabled hides every copy of a name, so external names stay out of it."""
    _write_skill(tmp_path / "team" / "github")
    _write_skill(tmp_path / "team" / "renamed", name="pdf")
    agent_config = _skills_config(
        {
            "directories": [{"folder": str(tmp_path / "team")}],
            "bundled": {"enabled": False},
        }
    )

    plan = hermes_skills.plan_skills(
        agent_config,
        inventory=lambda: _inventory(tmp_path),
        hermes_home=tmp_path / "home",
    )
    config = configuration.build_hermes_config(
        agent_config, workspace=".", skill_plan=plan
    )

    assert config["skills"] == {
        "external_dirs": [str(tmp_path.resolve() / "team")],
        "disabled": ["arxiv"],
    }


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
def test_bundled_names_provided_by_skill_paths_stay_enabled(
    tmp_path: Path, relative: bool
):
    """skills.paths is unchecked, but its skills still keep their names enabled."""
    home = tmp_path / "home"
    mine = home / "mine" if relative else tmp_path / "mine"
    _write_skill(mine / "github")
    _write_skill(mine / "renamed", name="pdf")
    # Hermes's scan skips a skill's support folders.
    _write_skill(mine / "github" / "scripts" / "arxiv")
    agent_config = _skills_config(
        {
            "paths": ["mine" if relative else str(mine)],
            "bundled": {"enabled": False},
        }
    )

    plan = hermes_skills.plan_skills(
        agent_config, inventory=lambda: _inventory(tmp_path), hermes_home=home
    )
    config = configuration.build_hermes_config(
        agent_config, workspace=".", skill_plan=plan
    )

    assert config["skills"] == {
        "external_dirs": ["mine" if relative else str(mine)],
        "disabled": ["arxiv"],
    }


@pytest.mark.parametrize(
    ("manifest", "disabled", "kept"),
    [
        pytest.param(None, ["arxiv", "github"], ["pdf"], id="own-skill"),
        pytest.param("pdf:0123\n", ["arxiv", "github", "pdf"], [], id="bundled-copy"),
        pytest.param("pdf\n", ["arxiv", "github", "pdf"], [], id="v1-manifest"),
    ],
)
def test_bundled_names_of_the_agents_own_home_skills_stay_enabled(
    tmp_path: Path, manifest: str | None, disabled: list[str], kept: list[str]
):
    """skills.disabled would also hide the agent's own skill of that name."""
    home = tmp_path / "home"
    _write_skill(home / "skills" / "notes" / "pdf")
    if manifest is not None:
        (home / "skills" / ".bundled_manifest").write_text(manifest, encoding="utf-8")
    agent_config = _skills_config({"bundled": {"include": []}})

    plan = hermes_skills.plan_skills(
        agent_config, inventory=lambda: _inventory(tmp_path), hermes_home=home
    )

    assert plan.disabled == disabled
    assert plan.kept_local == kept


@pytest.mark.parametrize(
    ("bundled", "field", "name", "reason"),
    [
        (
            {"include": ["pdf", "unknown"]},
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
    ],
)
def test_bundled_names_are_checked(
    tmp_path: Path, bundled: dict[str, object], field: str, name: str, reason: str
):
    agent_config = _skills_config({"bundled": bundled})

    with pytest.raises(lifecycle.LifecycleError) as caught:
        hermes_skills.plan_skills(agent_config, inventory=lambda: _inventory(tmp_path))

    assert caught.value.code == "hermes_skill_invalid"
    assert caught.value.metadata == {"field": field, "reason": reason, "name": name}


def _catalog_row(
    root: Path, folder: Path, *, tier: int, status: str, name: object = None
) -> dict[str, object]:
    return {
        "name": folder.name if name is None else name,
        "tier": tier,
        "root": root,
        "path": folder / "SKILL.md",
        "status": status,
        "visible": True,
    }


def test_observation_reports_hermes_outcomes(skill_tree: Path, tmp_path: Path):
    """Hermes's catalog decides what loads; the report follows it."""
    team, opt = skill_tree / "team", skill_tree / "opt" / "review"
    home = (tmp_path / "home" / "skills").resolve()
    _write_skill(home / "debugging")
    _write_skill(home / "pdf")
    agent_config = _skills_config(
        {
            "directories": [{"folder": str(team)}, {"skill": str(opt)}],
            "bundled": {"include": ["pdf", "arxiv"]},
        }
    )
    plan = hermes_skills.plan_skills(
        agent_config, inventory=lambda: _inventory(tmp_path), hermes_home=tmp_path
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
    ]

    report = hermes_skills.observe(
        plan, catalog, sync_result={"suppressed": ["arxiv"]}
    ).report()

    assert report == {
        "loaded": [
            {
                "name": "collect-logs",
                "path": str(team / "debugging" / "collect-logs"),
                "entry": "skills.directories[0]",
                "origin": "external",
            }
        ],
        "duplicates": [
            {
                "name": "review",
                "outcome": "ambiguous",
                "copies": [
                    {
                        "origin": "external",
                        "path": str(team / "review"),
                        "entry": "skills.directories[0]",
                    },
                    {
                        "origin": "external",
                        "path": str(opt),
                        "entry": "skills.directories[1]",
                    },
                ],
            },
            {
                "name": "debugging",
                "outcome": "shadowed",
                "selected": str(home / "debugging"),
                "copies": [
                    {"origin": "local", "path": str(home / "debugging")},
                    {
                        "origin": "external",
                        "path": str(team / "debugging"),
                        "entry": "skills.directories[0]",
                    },
                ],
            },
        ],
        "bundled": {
            "selected": ["pdf"],
            "unavailable": [{"name": "arxiv", "reason": "suppressed"}],
        },
    }


def test_observation_reports_bundled_copies_deferred_to_external(tmp_path: Path):
    team = _write_skill(tmp_path / "team" / "github").parent.resolve()
    agent_config = _skills_config(
        {"directories": [{"folder": str(team)}], "bundled": {}}
    )
    plan = hermes_skills.plan_skills(
        agent_config, inventory=lambda: _inventory(tmp_path), hermes_home=tmp_path
    )
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
                {
                    "origin": "external",
                    "path": str(team / "github"),
                    "entry": "skills.directories[0]",
                },
                {"origin": "bundled", "path": str(tmp_path / "bundled" / "github")},
            ],
        }
    ]
    assert {"name": "github", "reason": "provided_by_external_skill"} in report[
        "bundled"
    ]["unavailable"]


def test_observation_reports_skills_hermes_does_not_scan(tmp_path: Path):
    """Hermes drops every skill below folders such as site-packages."""
    folder = _write_skill(tmp_path / "site-packages" / "tool").parent.resolve()
    agent_config = _skills_config({"directories": [{"folder": str(folder)}]})
    plan = hermes_skills.plan_skills(agent_config)

    report = hermes_skills.observe(plan, []).report()

    assert report == {
        "skipped": [
            {
                "path": str(folder / "tool"),
                "entry": "skills.directories[0]",
                "reason": "excluded_path",
            }
        ]
    }


def test_folder_entry_that_is_a_skill_loads_its_root(tmp_path: Path):
    """Hermes scans a folder from the folder itself, so a root SKILL.md is a skill."""
    pack = _write_skill(tmp_path / "pack").resolve()
    _write_skill(pack / "child")
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
    pack = _write_skill(tmp_path / "pack").resolve()
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
    _write_skill(team / "a", name="dup")
    _write_skill(team / "b")
    for support in ("scripts", "references", "templates", "assets"):
        _write_skill(team / "b" / support / "x", name="dup")
    # A support folder name below a folder without SKILL.md is scanned.
    _write_skill(team / "scripts" / "tool")
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
    _write_skill(root / "team" / "review")
    _write_skill(root / "team" / "nested" / "debug")
    _write_skill(root / "team" / "node_modules" / "vendored")
    _write_skill(root / "team" / "review" / "scripts" / "helper")
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


def test_observation_skips_a_skill_hermes_sees_but_does_not_load(tmp_path: Path):
    """A front-matter name that is another skill's folder name is ambiguous."""
    team = tmp_path.resolve() / "team"
    _write_skill(team / "foo", name="alpha")
    _write_skill(team / "bar", name="foo")
    agent_config = _skills_config({"directories": [{"folder": str(team)}]})
    plan = hermes_skills.plan_skills(agent_config)
    catalog = [
        _catalog_row(team, team / "foo", tier=3, status="unique", name="alpha"),
        _catalog_row(team, team / "bar", tier=3, status="ambiguous", name="foo"),
    ]

    report = hermes_skills.observe(plan, catalog).report()

    assert report == {
        "loaded": [
            {
                "name": "alpha",
                "path": str(team / "foo"),
                "entry": "skills.directories[0]",
                "origin": "external",
            }
        ],
        "skipped": [
            {
                "path": str(team / "bar"),
                "entry": "skills.directories[0]",
                "reason": "ambiguous",
            }
        ],
    }


def test_observation_skips_external_skills_hermes_hides(tmp_path: Path):
    """A skill Hermes resolves but hides on this platform does not load."""
    team = tmp_path.resolve() / "team"
    home = (tmp_path / "home" / "skills").resolve()
    _write_skill(team / "macos-only")
    _write_skill(team / "debugging")
    _write_skill(home / "debugging")
    agent_config = _skills_config({"directories": [{"folder": str(team)}]})
    plan = hermes_skills.plan_skills(agent_config)
    hidden = _catalog_row(team, team / "macos-only", tier=3, status="unique")
    hidden_winner = _catalog_row(team, team / "debugging", tier=3, status="unique")
    catalog = [
        {**hidden, "visible": False},
        {**hidden_winner, "visible": False},
        _catalog_row(home, home / "debugging", tier=1, status="shadowed"),
    ]

    report = hermes_skills.observe(plan, catalog).report()

    assert "loaded" not in report
    assert report["skipped"] == [
        {
            "path": str(team / "debugging"),
            "entry": "skills.directories[0]",
            "reason": "native_conditions",
        },
        {
            "path": str(team / "macos-only"),
            "entry": "skills.directories[0]",
            "reason": "native_conditions",
        },
    ]
    # The configured copy wins the clash, but no copy loads.
    assert report["duplicates"] == [
        {
            "name": "debugging",
            "outcome": "selected",
            "copies": [
                {"origin": "local", "path": str(home / "debugging")},
                {
                    "origin": "external",
                    "path": str(team / "debugging"),
                    "entry": "skills.directories[0]",
                },
            ],
        }
    ]


def test_observation_does_not_report_the_agents_own_skill_as_bundled(
    tmp_path: Path,
):
    home = tmp_path / "home"
    skills_dir = (home / "skills").resolve()
    # No bundled manifest: the home's pdf is the agent's own skill.
    _write_skill(skills_dir / "pdf")
    agent_config = _skills_config({"bundled": {"include": ["pdf"]}})
    plan = hermes_skills.plan_skills(
        agent_config, inventory=lambda: _inventory(tmp_path), hermes_home=home
    )
    assert plan.local_names == {"pdf"}
    catalog = [_catalog_row(skills_dir, skills_dir / "pdf", tier=1, status="unique")]

    observation = hermes_skills.observe(plan, catalog)

    assert observation.report() == {
        "bundled": {
            "selected": [],
            "unavailable": [{"name": "pdf", "reason": "provided_by_local_skill"}],
        }
    }
    assert observation.bundled_warnings == [
        "bundled skill 'pdf' does not load: provided_by_local_skill"
    ]


def test_observation_follows_the_names_hermes_reads(tmp_path: Path):
    """Hermes reads some front matter differently from the checks."""
    team = tmp_path.resolve() / "team"
    # Hermes needs a newline after the closing ---, so it uses the folder name.
    (team / "unclosed").mkdir(parents=True)
    (team / "unclosed" / "SKILL.md").write_text(
        "---\nname: closed\n---", encoding="utf-8"
    )
    # Hermes skips a null or non-string name; the checks fall back to the folder.
    for folder, value in (("blank", ""), ("number", "123")):
        (team / folder).mkdir()
        (team / folder / "SKILL.md").write_text(
            f"---\nname: {value}\n---\nBody\n", encoding="utf-8"
        )
    # The checks see a clash with the blank skill's folder name; Hermes doesn't.
    _write_skill(team / "other", name="blank")
    # Hermes lists an empty quoted name but cannot load it.
    (team / "quoted").mkdir()
    (team / "quoted" / "SKILL.md").write_text(
        '---\nname: ""\n---\nBody\n', encoding="utf-8"
    )
    agent_config = _skills_config({"directories": [{"folder": str(team)}]})
    plan = hermes_skills.plan_skills(agent_config)
    assert plan.load is not None
    assert [(skill.name, skill.path.name) for skill in plan.load.selected] == [
        ("blank", "blank"),
        ("123", "number"),
        ("blank", "other"),
        ("quoted", "quoted"),
        ("closed", "unclosed"),
    ]
    assert [record.name for record in plan.load.duplicates] == ["blank"]
    catalog = [
        _catalog_row(team, team / "other", tier=3, status="unique", name="blank"),
        {
            **_catalog_row(team, team / "quoted", tier=3, status="unique", name=""),
            "load_name": "",
        },
        _catalog_row(team, team / "unclosed", tier=3, status="unique"),
    ]

    report = hermes_skills.observe(plan, catalog).report()

    assert report == {
        "loaded": [
            {
                "name": "blank",
                "path": str(team / "other"),
                "entry": "skills.directories[0]",
                "origin": "external",
            },
            {
                "name": "unclosed",
                "path": str(team / "unclosed"),
                "entry": "skills.directories[0]",
                "origin": "external",
            },
        ],
        "skipped": [
            {
                "path": str(team / name),
                "entry": "skills.directories[0]",
                "reason": reason,
            }
            for name, reason in (
                ("blank", "not_loaded_by_hermes"),
                ("number", "not_loaded_by_hermes"),
                ("quoted", "no_name"),
            )
        ],
    }
