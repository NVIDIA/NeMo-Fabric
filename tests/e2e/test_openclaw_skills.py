# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured skill configuration checked against the installed OpenClaw loader.

Each test generates the adapter's OpenClaw configuration and asks the real
``openclaw skills list`` which skills load, without starting a Gateway or a
model. Set FABRIC_OPENCLAW_TEST_COMMAND to choose the command; otherwise
``openclaw`` on PATH is used, and the tests skip when neither is a supported
version.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from _utils.openclaw_skills import agent_config
from _utils.openclaw_skills import loaded
from _utils.openclaw_skills import refused_copies
from _utils.openclaw_skills import refused_skill
from _utils.openclaw_skills import runtime_context
from _utils.openclaw_skills import skipped
from _utils.openclaw_skills import write_skill
from _utils.utils import POSIX_PERMISSIONS
from nemo_fabric_adapters.openclaw import adapter
from nemo_fabric_adapters.openclaw import skills as openclaw_skills


@pytest.fixture(name="openclaw", scope="module")
def openclaw_fixture() -> Path:
    """Return a supported OpenClaw command, or skip."""

    configured = os.environ.get("FABRIC_OPENCLAW_TEST_COMMAND") or shutil.which(
        "openclaw"
    )
    if not configured:
        pytest.skip("OpenClaw is not installed")
    command = _command_path(configured)
    try:
        output = subprocess.run(
            [str(command), "--version"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout
        adapter._validate_openclaw_version(output)
    except (OSError, subprocess.SubprocessError, adapter.lifecycle.LifecycleError):
        pytest.skip("The installed OpenClaw version is not supported")
    return command


def _command_path(configured: str) -> Path:
    """Return a command path that still works from another working directory.

    A relative path becomes absolute; a bare name keeps its ``PATH`` lookup.
    """

    if os.path.dirname(configured):
        return Path(os.path.abspath(configured))
    return Path(configured)


def test_relative_command_paths_become_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)

    assert _command_path("./bin/openclaw") == tmp_path / "bin" / "openclaw"
    assert _command_path("openclaw") == Path("openclaw")


class _Native:
    """Generate the adapter's configuration and list OpenClaw's skills for it."""

    def __init__(self, tmp_path: Path, command: Path) -> None:
        self.tmp_path = tmp_path
        self.command = command
        self.workspace = tmp_path / "workspace"
        self.workspace.mkdir()
        self.runs = 0

    def plan(
        self, skills: dict[str, Any], settings: dict[str, Any] | None = None
    ) -> openclaw_skills.SkillPlan:
        return adapter._skill_plan(
            agent_config(skills, settings, self.command),
            runtime_context(self.workspace),
            base_dir=self.tmp_path,
            command=self.command,
            env=dict(os.environ),
        )

    def list(
        self, skills: dict[str, Any], settings: dict[str, Any] | None = None
    ) -> tuple[openclaw_skills.SkillPlan, dict]:
        """List OpenClaw's skills as start does.

        Like start, locate the copies the list leaves open with ``openclaw
        skills info``, and replace refused winners and list again. Every
        listing checks that the report names every skill OpenClaw loads.
        """
        plan = self.plan(skills, settings)
        while True:
            env = self._prepare(skills, settings, plan)
            status = self.run(env, "skills", "list", "--json", "--agent", "default")
            located = {}
            for name in openclaw_skills.names_to_locate(plan, status):
                where = openclaw_skills.located_copy(
                    self.run(
                        env, "skills", "info", name, "--json", "--agent", "default"
                    )
                )
                if where is not None:
                    located[name] = where
            if not openclaw_skills.apply_status(
                plan, status, located=located, fallback=True
            ):
                _assert_reports_every_loaded_skill(plan, status)
                return plan, status

    def _prepare(
        self,
        skills: dict[str, Any],
        settings: dict[str, Any] | None,
        plan: openclaw_skills.SkillPlan,
    ) -> dict[str, str]:
        """Write the generated configuration and return the environment to read it."""
        generated = adapter._openclaw_config(
            agent_config(skills, settings, self.command),
            runtime_context(self.workspace),
            base_dir=self.tmp_path,
            port=18_789,
            token_env="OPENCLAW_GATEWAY_TOKEN",
            skill_plan=plan,
        )
        self.runs += 1
        run = self.tmp_path / f"run-{self.runs}"
        (run / "home").mkdir(parents=True)
        config_path = run / "openclaw.json"
        config_path.write_text(json.dumps(generated), encoding="utf-8")
        env = {
            "PATH": os.environ["PATH"],
            "HOME": str(run / "home"),
            "OPENCLAW_CONFIG_PATH": str(config_path),
            "OPENCLAW_STATE_DIR": str(run / "state"),
            "OPENCLAW_GATEWAY_TOKEN": "TEST_ONLY_GATEWAY_TOKEN",
            "DO_NOT_TRACK": "1",
        }
        # As start does: OpenClaw reads only the selected shipped skills.
        staged = openclaw_skills.stage_bundled(plan, run / "bundled")
        if staged is not None:
            env[openclaw_skills.BUNDLED_SKILLS_ENV] = str(staged)
        return env

    def run(self, env: dict[str, str], *args: str) -> dict:
        """Run one OpenClaw command and return its JSON output."""
        result = subprocess.run(
            [str(self.command), *args],
            cwd=self.workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        return json.loads(result.stdout)


def _assert_reports_every_loaded_skill(
    plan: openclaw_skills.SkillPlan, status: dict
) -> None:
    """The report's loaded skills are the ones OpenClaw lists, from matching sources.

    A skill Fabric turned off with ``skills.entries`` is listed as disabled
    and does not load.
    """
    report = plan.report()
    assert report is not None
    rows = [
        row
        for row in status["skills"]
        if not (row.get("disabled") is True and row["name"] in plan.disabled_extensions)
    ]
    origins = {item["name"]: item["origin"] for item in report["loaded"]}
    assert sorted(item["name"] for item in report["loaded"]) == sorted(
        row["name"] for row in rows
    )
    for row in rows:
        allowed = {
            "openclaw-bundled": {"bundled"},
            "openclaw-custodian": {"bundled"},
            # A configured skill, or a skill an OpenClaw plugin adds.
            "openclaw-extra": {"external", "bundled", "local"},
        }.get(row["source"], {"local", "external"})
        assert origins[row["name"]] in allowed, (row, origins[row["name"]])


@pytest.fixture(name="native")
def native_fixture(tmp_path: Path, openclaw: Path) -> _Native:
    return _Native(tmp_path, openclaw)


@pytest.fixture(name="baseline", scope="module")
def baseline_fixture(
    tmp_path_factory: pytest.TempPathFactory, openclaw: Path
) -> set[str]:
    """Names of the extraDirs-source skills OpenClaw lists with nothing configured."""
    native = _Native(tmp_path_factory.mktemp("baseline"), openclaw)
    _, status = native.list({"directories": []})
    return set(_external(status, set()))


def _external(status: dict, baseline: set[str]) -> dict[str, str]:
    """Return ``name -> description`` of the skills.load.extraDirs skills."""

    return {
        row["name"]: row["description"]
        for row in status["skills"]
        if row["source"] == "openclaw-extra" and row["name"] not in baseline
    }


def _configured(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the report's loaded skills that ``skills.directories`` supplied."""
    return [
        item
        for item in (report or {}).get("loaded", [])
        if item["origin"] == "external"
    ]


def _loaded(plan: openclaw_skills.SkillPlan) -> dict[str, str]:
    """Return ``name -> description`` of the loaded configured skills."""
    return {
        item["name"]: f"Copy at {Path(item['path']).name}."
        for item in _configured(plan.report())
    }


def test_reported_skills_are_the_ones_openclaw_loads(native: _Native):
    """Folder, chosen-skill, and mixed entries report what OpenClaw loads."""
    root = native.tmp_path
    team = root / "team-skills"
    write_skill(team / "debugging", "debugging")
    write_skill(team / "group" / "code-review", "code-review")
    write_skill(team / "group" / "deep" / "too-deep", "too-deep")
    write_skill(team / "debugging" / "nested", "nested")
    write_skill(team / ".hidden" / "secret", "secret")
    write_skill(team / "node_modules" / "dependency", "dependency")
    write_skill(team / "undescribed", "undescribed", None)
    write_skill(team / "unnamed")
    refused_skill(team / "broken-yaml", "broken-yaml", "malformed")
    # Larger than the 256,000 bytes OpenClaw reads.
    write_skill(team / "too-large", "too-large", "x" * 256_001)
    named = root / "skills"
    write_skill(named / "a" / "b" / "c" / "deep-under-skills", "deep-under-skills")
    release = write_skill(root / "opt" / "release-review", "release-review")
    _, baseline_status = native.list({"directories": []})
    baseline = set(_external(baseline_status, set()))

    plan, status = native.list(
        {
            "directories": [
                {"folder": str(team)},
                {"folder": str(named)},
                {"skill": str(release)},
            ]
        }
    )

    assert _external(status, baseline) == _loaded(plan)
    assert set(_loaded(plan)) == {
        "debugging",
        "code-review",
        "unnamed",
        "deep-under-skills",
        "release-review",
    }
    report = plan.report()
    assert report is not None
    assert {item["reason"] for item in report["skipped"]} == {
        "missing_frontmatter_field:description",
        "invalid_frontmatter",
        "file_too_large",
    }


@pytest.mark.parametrize(
    ("front_matter", "name", "reason"),
    [
        pytest.param(["name: >-", "  folded-name"], "folded-name", None, id="block"),
        pytest.param(
            ["label: &label aliased", "name: *label"], "aliased", None, id="alias"
        ),
        pytest.param(
            ["name: '  trimmed name '"], "trimmed name", None, id="trimmed-name"
        ),
        pytest.param(
            ["name: named", "metadata: [unclosed"],
            None,
            "invalid_frontmatter",
            id="malformed",
        ),
        pytest.param(
            ["name: named", "description: '  '"],
            None,
            "empty_frontmatter_field:description",
            id="empty-description",
        ),
    ],
)
def test_reader_matches_openclaw(
    native: _Native,
    baseline: set[str],
    front_matter: list[str],
    name: str | None,
    reason: str | None,
):
    """The reader names and refuses skills as ``openclaw skills list`` does."""
    skill = native.tmp_path / "team" / "named"
    skill.mkdir(parents=True)
    if not any(line.startswith("description:") for line in front_matter):
        front_matter = ["description: Copy at named.", *front_matter]
    (skill / "SKILL.md").write_text(
        "\n".join(["---", *front_matter, "---", ""]), encoding="utf-8"
    )

    plan, status = native.list({"directories": [{"folder": str(skill.parent)}]})

    assert (
        _external(status, baseline)
        == _loaded(plan)
        == ({name: "Copy at named."} if name else {})
    )
    assert (plan.report() or {}).get("skipped", []) == (
        [skipped(skill, "skills.directories[0]", reason)] if reason else []
    )


@pytest.mark.parametrize("policy", [None, "first_wins", "last_wins"])
def test_duplicate_outcomes_match_openclaw(native: _Native, policy: str | None):
    """The reported copy of a duplicated name is the one OpenClaw loads."""
    root = native.tmp_path
    write_skill(root / "team" / "review-team", "review")
    write_skill(root / "team" / "debugging", "debugging")
    mounted = write_skill(root / "opt" / "review-mounted", "review")
    skills: dict[str, Any] = {
        "directories": [{"folder": str(root / "team")}, {"skill": str(mounted)}]
    }
    if policy is not None:
        skills["duplicates"] = policy

    plan, status = native.list(skills)

    loaded = _loaded(plan)
    assert _external(status, set())["review"] == loaded["review"]
    expected = "review-team" if policy == "first_wins" else "review-mounted"
    assert loaded["review"] == f"Copy at {expected}."


@pytest.mark.parametrize("policy", ["first_wins", "last_wins", "reject"])
def test_alias_names_take_part_in_duplicate_policies(native: _Native, policy: str):
    """A name written as an alias clashes, as in OpenClaw, under every policy."""
    root = native.tmp_path
    aliased = root / "opt" / "review-aliased"
    aliased.mkdir(parents=True)
    (aliased / "SKILL.md").write_text(
        "---\nlabel: &label review\nname: *label\n"
        "description: Copy at review-aliased.\n---\n",
        encoding="utf-8",
    )
    plain = write_skill(root / "team" / "review-plain", "review")
    skills = {
        "directories": [{"skill": str(aliased)}, {"skill": str(plain)}],
        "duplicates": policy,
    }

    if policy == "reject":
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            native.plan(skills)
        assert caught.value.metadata["reason"] == "duplicate"
        return
    plan, status = native.list(skills)

    loaded = _loaded(plan)
    assert _external(status, set())["review"] == loaded["review"]
    expected = "review-aliased" if policy == "first_wins" else "review-plain"
    assert loaded["review"] == f"Copy at {expected}."


def test_configured_workspace_skill_wins_as_reported(native: _Native):
    """A configured workspace skill outranks the shipped copy, as OpenClaw decides."""
    weather = write_skill(native.workspace / "skills" / "weather", "weather", "Mine.")

    plan, status = native.list(
        {"directories": [{"skill": str(weather)}], "bundled": {}}
    )

    rows = {row["name"]: row for row in status["skills"]}
    assert (rows["weather"]["source"], rows["weather"]["description"]) == (
        "openclaw-workspace",
        "Mine.",
    )
    report = plan.report()
    assert report is not None
    assert [(item["name"], item["path"]) for item in _configured(report)] == [
        ("weather", str(weather.resolve()))
    ]
    duplicates = {item["name"]: item for item in report["duplicates"]}
    assert duplicates["weather"]["outcome"] == "selected"
    assert duplicates["weather"]["selected"] == str(weather.resolve())
    unavailable = {item["name"]: item for item in report["bundled"]["unavailable"]}
    assert unavailable["weather"]["reason"] == "shadowed"


@pytest.mark.parametrize(
    "front_matter",
    [
        pytest.param(
            "name: review\ndescription: Workspace.\nmetadata: [unclosed", id="yaml"
        ),
        pytest.param("name: review\ndescription: ''", id="empty-description"),
    ],
)
@pytest.mark.parametrize("policy", [None, "first_wins"])
def test_invalid_workspace_skill_does_not_hide_a_configured_one(
    native: _Native, front_matter: str, policy: str | None
):
    """A workspace copy OpenClaw refuses hides nothing, as reported."""
    team = write_skill(native.tmp_path / "team" / "review", "review", "External.")
    local = native.workspace / "skills" / "review"
    local.mkdir(parents=True)
    (local / "SKILL.md").write_text(f"---\n{front_matter}\n---\n", encoding="utf-8")
    skills: dict[str, Any] = {"directories": [{"skill": str(team)}]}
    if policy is not None:
        skills["duplicates"] = policy

    plan, status = native.list(skills)

    rows = {row["name"]: row for row in status["skills"]}
    assert (rows["review"]["source"], rows["review"]["description"]) == (
        "openclaw-extra",
        "External.",
    )
    report = plan.report()
    assert report is not None
    assert [(item["name"], item["path"]) for item in _configured(report)] == [
        ("review", str(team.resolve()))
    ]
    assert "review" not in {item["name"] for item in report.get("duplicates", [])}


@pytest.mark.parametrize(
    ("bundled", "allowed"),
    [
        ({}, None),
        ({"enabled": False}, set()),
        ({"include": []}, set()),
        ({"include": ["weather", "github"]}, {"weather", "github"}),
        ({"exclude": ["weather"]}, "all-but-weather"),
    ],
)
def test_bundled_selection_matches_openclaw(
    native: _Native, bundled: dict[str, Any], allowed: set[str] | str | None
):
    """skills.bundled selects exactly the shipped skills OpenClaw loads."""
    plan, status = native.list({"bundled": bundled})

    shipped = {row["name"] for row in status["skills"] if row["bundled"]}
    report = plan.report()
    assert report is not None
    assert plan.inventory is not None
    # OpenClaw marks only its skills/ and custodian-skills/ copies bundled;
    # extension skills are tested on their own.
    folders = {*plan.inventory.skills, *plan.inventory.custodian}
    expected = (
        folders
        if allowed is None
        else folders - {"weather"}
        if allowed == "all-but-weather"
        else allowed
    )
    # A left-out shipped skill is not loaded at all, not merely blocked.
    assert shipped == expected
    assert not any(row["blockedByAllowlist"] for row in status["skills"])
    eligible = {
        row["name"] for row in status["skills"] if row["bundled"] and row["eligible"]
    }
    unavailable = {item["name"] for item in report["bundled"].get("unavailable", [])}
    assert unavailable & folders == expected - eligible
    # Selected lists only the selected bundled skills that load.
    assert set(report["bundled"]["selected"]) - set(plan.inventory.extensions) == (
        expected & eligible
    )


def _active_extension_skills(plan: openclaw_skills.SkillPlan, status: dict) -> set[str]:
    """Return the extension skills OpenClaw lists and has not turned off."""
    assert plan.inventory is not None
    return {
        row["name"]
        for row in status["skills"]
        if row["name"] in plan.inventory.extensions and row.get("disabled") is not True
    }


def test_extension_skills_follow_the_bundled_selection(native: _Native):
    """Skills OpenClaw's own extensions ship load only when the selection keeps
    them: none with bundled disabled, all but an excluded one otherwise."""
    plan, status = native.list({"bundled": {}})
    active = _active_extension_skills(plan, status)
    if not active:
        pytest.skip("This OpenClaw activates no extension that ships a skill")
    assert plan.inventory is not None
    left_out = sorted(active)[0]
    cases: list[tuple[dict[str, Any], set[str]]] = [
        ({}, active),
        ({"enabled": False}, set()),
        ({"include": []}, set()),
        ({"exclude": [left_out]}, active - {left_out}),
        ({"include": [left_out]}, {left_out}),
    ]
    for bundled, loads in cases:
        plan, status = native.list({"bundled": bundled})

        assert _active_extension_skills(plan, status) == loads, bundled
        report = plan.report()
        assert report is not None
        extension_loaded = {
            item["name"]: item["origin"]
            for item in report["loaded"]
            if item["name"] in plan.inventory.extensions
        }
        assert extension_loaded == dict.fromkeys(loads, "bundled"), bundled
        assert set(report["bundled"]["selected"]) & set(plan.inventory.extensions) == (
            loads
        ), bundled


def test_configured_skill_named_like_a_left_out_extension_skill_fails(
    native: _Native,
):
    """skills.entries would turn off both copies, and the active extension's
    copy hides the configured one, so start fails."""
    plan, status = native.list({"bundled": {}})
    active = sorted(_active_extension_skills(plan, status))
    if not active:
        pytest.skip("This OpenClaw activates no extension that ships a skill")
    name = active[0]
    write_skill(native.tmp_path / "team" / name, name, "Configured copy.")

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        native.list(
            {
                "directories": [{"folder": str(native.tmp_path / "team")}],
                "bundled": {"enabled": False},
            }
        )

    assert caught.value.metadata["reason"] == openclaw_skills.EXTENSION_CLASH
    assert caught.value.metadata["name"] == name


@pytest.mark.parametrize(
    "bundled",
    [{"exclude": ["weather"]}, {"include": ["github"]}, {"enabled": False}],
)
def test_left_out_shipped_skill_does_not_hide_a_configured_copy(
    native: _Native, bundled: dict[str, Any]
):
    """A configured weather loads when bundled leaves out the shipped weather."""
    write_skill(native.tmp_path / "team" / "weather", "weather", "Configured weather.")

    plan, status = native.list(
        {"directories": [{"folder": str(native.tmp_path / "team")}], "bundled": bundled}
    )

    rows = [row for row in status["skills"] if row["name"] == "weather"]
    assert [(row["source"], row["description"]) for row in rows] == [
        ("openclaw-extra", "Configured weather.")
    ]
    report = plan.report()
    assert report is not None
    assert [item["name"] for item in _configured(report)] == ["weather"]
    assert "duplicates" not in report


def test_higher_sources_hide_configured_skills_as_reported(native: _Native):
    """Workspace and selected shipped skills win over configured copies, as reported."""
    root = native.tmp_path
    team = root / "team"
    write_skill(team / "github", "github")
    write_skill(team / "weather", "weather")
    write_skill(team / "review", "review")
    write_skill(team / "kept", "kept")
    write_skill(native.workspace / "skills" / "review", "review", "Workspace copy.")

    plan, status = native.list(
        {"directories": [{"folder": str(team)}], "bundled": {"exclude": ["weather"]}}
    )

    rows = {row["name"]: row for row in status["skills"]}
    assert rows["review"]["source"] == "openclaw-workspace"
    assert rows["github"]["source"] == "openclaw-bundled"
    assert rows["weather"]["source"] == "openclaw-extra"
    report = plan.report()
    assert report is not None
    assert [item["name"] for item in _configured(report)] == ["kept", "weather"]
    outcomes = {
        item["name"]: (item["outcome"], item.get("selected"))
        for item in report["duplicates"]
    }
    assert outcomes["review"] == (
        "shadowed",
        str((native.workspace / "skills" / "review").resolve()),
    )
    assert outcomes["github"][0] == "shadowed"
    assert "weather" not in outcomes


def test_every_copy_of_a_clash_is_reported_with_openclaws_winner(native: _Native):
    """Configured, shipped, and workspace copies are all listed; OpenClaw's pick is selected."""
    root = native.tmp_path
    write_skill(root / "team" / "github", "github", "Configured github.")
    write_skill(native.workspace / "skills" / "github", "github", "Workspace github.")
    write_skill(
        native.workspace / ".agents" / "skills" / "tmux", "tmux", "Project tmux."
    )

    plan, status = native.list(
        {"directories": [{"folder": str(root / "team")}], "bundled": {}}
    )

    rows = {row["name"]: row for row in status["skills"]}
    assert (rows["github"]["source"], rows["github"]["description"]) == (
        "openclaw-workspace",
        "Workspace github.",
    )
    assert (rows["tmux"]["source"], rows["tmux"]["description"]) == (
        "agents-skills-project",
        "Project tmux.",
    )
    report = plan.report()
    assert report is not None
    duplicates = {item["name"]: item for item in report["duplicates"]}
    assert plan.inventory is not None
    assert duplicates["github"] == {
        "name": "github",
        "outcome": "shadowed",
        "selected": str((native.workspace / "skills" / "github").resolve()),
        "copies": [
            {
                "origin": "external",
                "path": str((root / "team" / "github").resolve()),
                "entry": "skills.directories[0]",
            },
            {"origin": "bundled", "path": str(plan.inventory.skills["github"])},
            {
                "origin": "local",
                "path": str((native.workspace / "skills" / "github").resolve()),
            },
        ],
    }
    # A clash with no configured copy is reported too.
    assert duplicates["tmux"] == {
        "name": "tmux",
        "outcome": "selected",
        "selected": str((native.workspace / ".agents/skills/tmux").resolve()),
        "copies": [
            {"origin": "bundled", "path": str(plan.inventory.skills["tmux"])},
            {
                "origin": "local",
                "path": str((native.workspace / ".agents/skills/tmux").resolve()),
            },
        ],
    }
    # The shipped copies hidden by the workspace are reported as such.
    unavailable = {
        item["name"]: item for item in report["bundled"].get("unavailable", [])
    }
    assert unavailable["github"] == {
        "name": "github",
        "reason": "shadowed",
        "by": "openclaw-workspace",
    }


@pytest.mark.skipif(os.name == "nt", reason="Requires symbolic links")
@pytest.mark.parametrize("policy", [None, "first_wins"])
def test_skills_linked_outside_their_folder_never_load(
    native: _Native, baseline: set[str], policy: str | None
):
    """A skill that links outside its folder entry never loads, as in OpenClaw."""
    root = native.tmp_path
    write_skill(root / "outside" / "escaped", "escaped")
    write_skill(root / "outside" / "skills" / "nested-escaped", "nested-escaped")
    write_skill(root / "team" / "inside", "inside")
    (root / "team" / "escaped").symlink_to(root / "outside" / "escaped")
    (root / "linked").mkdir()
    (root / "linked" / "skills").symlink_to(root / "outside" / "skills")
    skills: dict[str, Any] = {
        "directories": [
            {"folder": str(root / "team")},
            {"folder": str(root / "linked")},
        ]
    }
    if policy is not None:
        skills["duplicates"] = policy

    plan, status = native.list(skills)

    assert set(_external(status, baseline)) == {"inside"} == set(_loaded(plan))
    outside = (root / "outside").resolve()
    assert not any(
        Path(folder).resolve().is_relative_to(outside) for folder in plan.extra_dirs
    )
    report = plan.report()
    assert report is not None
    assert report["skipped"] == [
        {
            "path": str((root / "outside" / "escaped").resolve()),
            "entry": "skills.directories[0]",
            "reason": "path_escape",
        },
        {
            "path": str((root / "outside" / "skills").resolve()),
            "entry": "skills.directories[1]",
            "reason": "path_escape",
        },
    ]
    assert report["empty_entries"] == ["skills.directories[1]"]


@pytest.mark.parametrize("policy", [None, "first_wins"])
def test_skills_child_of_a_scanned_root_is_searched_six_levels(
    native: _Native, baseline: set[str], policy: str | None
):
    """A root's skills child keeps 6 levels while the root keeps 2."""
    root = native.tmp_path
    write_skill(root / "team" / "top", "top")
    write_skill(root / "team" / "skills" / "g" / "h" / "deep", "deep")
    write_skill(root / "team" / "a" / "b" / "too-deep", "too-deep")
    skills: dict[str, Any] = {"directories": [{"folder": str(root / "team")}]}
    if policy is not None:
        skills["duplicates"] = policy

    plan, status = native.list(skills)

    assert _external(status, baseline) == _loaded(plan)
    assert set(_loaded(plan)) == {"top", "deep"}


@pytest.mark.parametrize("policy", [None, "first_wins"])
def test_promoted_skills_folder_and_deep_hidden_folders_match_openclaw(
    native: _Native, baseline: set[str], policy: str | None
):
    """A promoted skills/ that is a skill loads alone; deep dot folders never load."""
    root = native.tmp_path
    write_skill(root / "packed" / "skills", "packed")
    write_skill(root / "packed" / "skills" / "nested", "nested")
    write_skill(root / "team" / "top", "top")
    write_skill(root / "team" / "skills" / "a" / "b" / "c" / "d" / "e" / ".hid", "hid")
    skills: dict[str, Any] = {
        "directories": [
            {"folder": str(root / "packed")},
            {"folder": str(root / "team")},
        ]
    }
    if policy is not None:
        skills["duplicates"] = policy

    plan, status = native.list(skills)

    assert _external(status, baseline) == _loaded(plan)
    assert set(_loaded(plan)) == {"packed", "top"}


def test_overlapping_folder_entries_report_each_entry_with_its_depth(
    native: _Native, baseline: set[str]
):
    """A skill reached only through the deeper entry is reported under it."""
    root = native.tmp_path
    write_skill(root / "a" / "top", "top")
    write_skill(root / "a" / "b" / "skills" / "x", "x")

    plan, status = native.list(
        {
            "directories": [
                {"folder": str(root / "a")},
                {"folder": str(root / "a" / "b" / "skills")},
            ],
            "duplicates": "first_wins",
        }
    )

    assert _external(status, baseline) == _loaded(plan)
    report = plan.report()
    assert report is not None
    assert [(item["name"], item["entry"]) for item in _configured(report)] == [
        ("top", "skills.directories[0]"),
        ("x", "skills.directories[1]"),
    ]


_SYMLINKS = pytest.mark.skipif(os.name == "nt", reason="Requires symbolic links")


@_SYMLINKS
@_SYMLINKS
@pytest.mark.parametrize("policy", [None, "reject", "first_wins", "last_wins"])
@pytest.mark.parametrize("layout", ["two_entries", "one_folder"])
def test_skill_file_linked_outside_its_folder_is_no_duplicate(
    native: _Native, policy: str | None, layout: str
):
    """OpenClaw refuses a SKILL.md that links outside its skill folder
    (``resolveSkillFilePath``), so that copy never counts as a duplicate.

    Start succeeds under reject, and the copy OpenClaw loads is reported.
    """
    root = native.tmp_path
    outside = write_skill(root / "outside" / "review", "review", "Outside.")
    if layout == "two_entries":
        good = write_skill(root / "good" / "review", "review", "Valid.")
        bad = root / "bad" / "review"
        directories = [{"folder": str(root / "good")}, {"folder": str(root / "bad")}]
        bad_field = "skills.directories[1]"
    else:
        good = write_skill(root / "team" / "good", "review", "Valid.")
        bad = root / "team" / "bad"
        directories = [{"folder": str(root / "team")}]
        bad_field = "skills.directories[0]"
    bad.mkdir(parents=True)
    (bad / "SKILL.md").symlink_to(outside / "SKILL.md")
    skills: dict[str, Any] = {"directories": directories}
    if policy is not None:
        skills["duplicates"] = policy

    plan, status = native.list(skills)

    rows = {row["name"]: row for row in status["skills"]}
    assert (rows["review"]["source"], rows["review"]["description"]) == (
        "openclaw-extra",
        "Valid.",
    )
    report = plan.report()
    assert report is not None
    assert _configured(report) == [loaded("review", good, "skills.directories[0]")]
    assert report["skipped"] == [skipped(bad, bad_field, "path_escape")]
    assert "duplicates" not in report


@pytest.mark.parametrize("policy", [None, "reject", "first_wins", "last_wins"])
@pytest.mark.parametrize("entry", ["folder", "broken_link"])
def test_no_skill_below_a_skill_md_entry_loads(
    native: _Native, policy: str | None, entry: str
):
    """OpenClaw stops at any entry named SKILL.md (``hasSkillFileCandidate``),
    so a skill below a SKILL.md folder or broken link never loads, and Fabric
    passes and reports none under any policy."""
    root = native.tmp_path
    boundary = root / "team" / "boundary"
    write_skill(boundary / "nested", "nested", "Nested.")
    if entry == "folder":
        (boundary / "SKILL.md").mkdir()
    else:
        if os.name == "nt":
            pytest.skip("Requires symbolic links")
        (boundary / "SKILL.md").symlink_to(root / "missing.md")
    skills: dict[str, Any] = {"directories": [{"folder": str(root / "team")}]}
    if policy is not None:
        skills["duplicates"] = policy

    plan, status = native.list(skills)

    assert "nested" not in {row["name"] for row in status["skills"]}
    report = plan.report()
    assert report is not None
    assert _configured(report) == []
    assert report["skipped"] == [
        skipped(boundary, "skills.directories[0]", "unreadable")
    ]


@pytest.mark.parametrize("owner", [None, "other"])
def test_custodian_skills_hide_configured_ones_only_for_their_owner(
    native: _Native, owner: str | None
):
    """A configured skill named like a Custodian skill loads unless the agent owns them."""
    inventory = native.plan({"bundled": {}}).inventory
    assert inventory is not None
    custodian_only = sorted(set(inventory.custodian) - set(inventory.skills))
    if not custodian_only:
        pytest.skip("The installed OpenClaw ships no Custodian skills")
    name = custodian_only[0]
    write_skill(native.tmp_path / "team" / name, name)
    # OpenClaw requires the owner to be a configured agent.
    settings = (
        {
            "agents": {
                "ownership": "explicit",
                "entries": {owner: {"contextInjection": "never"}},
                "defaults": {"systemAgent": {"agentId": owner}},
            }
        }
        if owner is not None
        else None
    )

    plan, status = native.list(
        {"directories": [{"folder": str(native.tmp_path / "team")}]}, settings
    )

    sources = {row["source"] for row in status["skills"] if row["name"] == name}
    report = plan.report() or {}
    if owner is None:
        assert sources == {"openclaw-custodian"}
        assert _configured(report) == []
        assert report["duplicates"][0]["outcome"] == "shadowed"
    else:
        assert sources == {"openclaw-extra"}
        assert set(_loaded(plan)) == {name}
        assert "duplicates" not in report


@POSIX_PERMISSIONS
@pytest.mark.parametrize("kind", ["folder", "skill"], ids=["folder", "chosen-skills"])
def test_read_only_skills_load_and_cannot_be_changed(
    native: _Native, baseline: set[str], make_read_only, kind: str
):
    """A read-only folder, or read-only chosen skills, load through OpenClaw as they are."""
    team = native.tmp_path / "team"
    debugging = write_skill(team / "debugging", "debugging")
    review = write_skill(team / "review", "review")
    make_read_only(team)
    entries = (
        [{"folder": str(team)}]
        if kind == "folder"
        else [{"skill": str(debugging)}, {"skill": str(review)}]
    )

    plan, status = native.list({"directories": entries})

    assert _external(status, baseline) == _loaded(plan)
    assert set(_loaded(plan)) == {"debugging", "review"}
    with pytest.raises(PermissionError):
        (review / "SKILL.md").write_text("changed", encoding="utf-8")
    with pytest.raises(PermissionError):
        write_skill(team / "new-skill", "new-skill")


def test_skill_written_into_a_writable_folder_loads_from_a_new_configuration(
    native: _Native, baseline: set[str]
):
    """A skill written after one generated configuration is listed with the next.

    This lists skills twice without a Gateway; the restart of a real runtime
    is covered by :func:`test_real_runtime_restart_loads_skills_written_meanwhile`.
    """
    agent_skills = native.tmp_path / "agent-skills"
    agent_skills.mkdir()
    skills = {"directories": [{"folder": str(agent_skills)}]}

    first, first_status = native.list(skills)
    # The agent saves a new skill into its read-write folder.
    write_skill(agent_skills / "triage", "triage")
    restarted, status = native.list(skills)

    assert _external(first_status, baseline) == {}
    assert _configured(first.report()) == []
    assert first.report()["empty_entries"] == ["skills.directories[0]"]
    assert (
        _external(status, baseline)
        == _loaded(restarted)
        == {"triage": "Copy at triage."}
    )


@POSIX_PERMISSIONS
def test_agent_adds_skills_beside_read_only_skills_it_cannot_change(
    native: _Native, baseline: set[str], make_read_only
):
    team = native.tmp_path / "team"
    protected = write_skill(team / "review", "review") / "SKILL.md"
    original = protected.read_text(encoding="utf-8")
    make_read_only(team)
    agent_skills = native.tmp_path / "agent-skills"
    agent_skills.mkdir()
    skills = {"directories": [{"folder": str(team)}, {"folder": str(agent_skills)}]}

    first, _ = native.list(skills)
    with pytest.raises(PermissionError):
        protected.write_text("changed", encoding="utf-8")
    with pytest.raises(PermissionError):
        write_skill(team / "triage", "triage")
    write_skill(agent_skills / "triage", "triage")
    restarted, status = native.list(skills)

    assert set(_loaded(first)) == {"review"}
    assert protected.read_text(encoding="utf-8") == original
    assert _external(status, baseline) == _loaded(restarted)
    report = restarted.report()
    assert report is not None
    assert [(item["name"], item["entry"]) for item in _configured(report)] == [
        ("review", "skills.directories[0]"),
        ("triage", "skills.directories[1]"),
    ]


@POSIX_PERMISSIONS
@pytest.mark.parametrize(
    ("entry", "mode"),
    [("skill", 0), ("folder", 0o300), ("folder", 0)],
    ids=["unreadable-skill-file", "unlistable-folder", "locked-folder"],
)
def test_unreadable_entries_are_invalid_with_openclaws_reader(
    native: _Native, entry: str, mode: int
):
    """With OpenClaw's own parser, an unreadable entry still fails as invalid."""
    good = write_skill(native.tmp_path / "good", "good")
    path = native.tmp_path / "entry"
    if entry == "skill":
        write_skill(path, "entry")
        locked = path / "SKILL.md"
    else:
        write_skill(path / "debugging", "debugging")
        locked = path
    original = locked.stat().st_mode & 0o777
    locked.chmod(mode)
    try:
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            native.plan({"directories": [{"skill": str(good)}, {entry: str(path)}]})
    finally:
        locked.chmod(original)

    assert caught.value.code == "openclaw_skill_invalid"
    assert caught.value.metadata == {
        "field": f"skills.directories[1].{entry}",
        "reason": "invalid",
        "path": str(locked),
    }


async def _start_runtime(
    native: _Native, skills: dict[str, Any]
) -> adapter.OpenClawRuntime:
    """Start a real OpenClaw Gateway for ``skills``; no model is called."""

    runtime = adapter.OpenClawRuntime()
    context = runtime_context(native.workspace).to_mapping()
    home = native.tmp_path / "home"
    home.mkdir(exist_ok=True)
    context["environment"]["env"] = {"HOME": str(home)}
    await runtime.start(
        {
            "config": agent_config(
                skills, {"startup_timeout_seconds": 120}, native.command
            ),
            "runtime_context": context,
            "base_dir": str(native.tmp_path),
        }
    )
    return runtime


def _running_skills(runtime: adapter.OpenClawRuntime) -> dict[str, dict]:
    """List a running runtime's skills with the environment its Gateway uses."""

    assert runtime._command is not None and runtime._command_env is not None
    result = subprocess.run(
        [str(runtime._command), "skills", "list", "--json", "--agent", "default"],
        env=runtime._command_env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return {row["name"]: row for row in json.loads(result.stdout)["skills"]}


async def test_real_runtime_restart_loads_skills_written_meanwhile(native: _Native):
    """Stop a running Gateway, add a skill, and start again: the report follows.

    Both starts keep the excluded shipped weather out, so the configured
    weather loads; the runtime's copy of the selected shipped skills goes away
    when it stops.
    """
    agent_skills = native.tmp_path / "agent-skills"
    agent_skills.mkdir()
    write_skill(agent_skills / "weather", "weather", "Configured weather.")
    skills = {
        "directories": [{"folder": str(agent_skills)}],
        "bundled": {"exclude": ["weather"]},
    }

    runtime = await _start_runtime(native, skills)
    try:
        assert runtime._process is not None and runtime._process.returncode is None
        first = runtime._skills_output()["skills"]
        rows = _running_skills(runtime)
        assert runtime._command_env is not None
        staged = Path(runtime._command_env[openclaw_skills.BUNDLED_SKILLS_ENV])
        assert staged.is_dir()
    finally:
        await runtime.stop()
    assert not staged.exists()
    assert [(item["name"], item["entry"]) for item in _configured(first)] == [
        ("weather", "skills.directories[0]")
    ]
    assert (rows["weather"]["source"], rows["weather"]["description"]) == (
        "openclaw-extra",
        "Configured weather.",
    )
    # The agent saves a new skill into its read-write folder while stopped.
    write_skill(agent_skills / "triage", "triage")

    runtime = await _start_runtime(native, skills)
    try:
        assert runtime._process is not None and runtime._process.returncode is None
        restarted = runtime._skills_output()["skills"]
        rows = _running_skills(runtime)
    finally:
        await runtime.stop()
    assert [(item["name"], item["entry"]) for item in _configured(restarted)] == [
        ("triage", "skills.directories[0]"),
        ("weather", "skills.directories[0]"),
    ]
    assert rows["triage"]["source"] == "openclaw-extra"
    assert rows["weather"]["description"] == "Configured weather."
    assert restarted["bundled"]["selected"] == first["bundled"]["selected"]
    assert "weather" not in restarted["bundled"]["selected"]


@pytest.mark.parametrize("policy", ["reject", "first_wins", "last_wins"])
async def test_copies_openclaw_refuses_never_count_as_duplicates(
    native: _Native, policy: str
):
    """Start succeeds under every policy and loads the copies OpenClaw accepts."""
    scenario = refused_copies(native.tmp_path)
    skills = {"directories": scenario.directories, "duplicates": policy}

    runtime = await _start_runtime(native, skills)
    try:
        assert runtime._process is not None and runtime._process.returncode is None
        report = runtime._skills_output()["skills"]
        rows = _running_skills(runtime)
    finally:
        await runtime.stop()
    assert (rows["review"]["source"], rows["review"]["description"]) == (
        "openclaw-extra",
        "Valid review.",
    )
    assert (rows["triage"]["source"], rows["triage"]["description"]) == (
        "openclaw-extra",
        "Valid triage.",
    )
    assert {**report, "loaded": _configured(report)} == scenario.report()


@pytest.mark.parametrize("policy", [None, "reject", "first_wins", "last_wins"])
def test_hard_linked_copy_is_skipped_as_openclaw_refuses_it(
    native: _Native, policy: str | None
):
    """A copy whose SKILL.md is hard-linked never loads, so it is no duplicate.

    Start succeeds under reject, and without a policy the valid copy is
    reported as the one that loads.
    """
    root = native.tmp_path
    valid = write_skill(root / "a" / "review", "review", "Valid.")
    linked = write_skill(root / "b" / "review", "review", "Linked.")
    os.link(linked / "SKILL.md", root / "SKILL.md.link")
    skills: dict[str, Any] = {
        "directories": [{"folder": str(root / "a")}, {"folder": str(root / "b")}]
    }
    if policy is not None:
        skills["duplicates"] = policy

    plan, status = native.list(skills)

    rows = {row["name"]: row for row in status["skills"]}
    assert (rows["review"]["source"], rows["review"]["description"]) == (
        "openclaw-extra",
        "Valid.",
    )
    report = plan.report()
    assert report is not None
    assert _configured(report) == [loaded("review", valid, "skills.directories[0]")]
    assert report["skipped"] == [
        skipped(linked, "skills.directories[1]", openclaw_skills.HARDLINKED)
    ]
    assert "duplicates" not in report


@_SYMLINKS
@pytest.mark.parametrize("policy", [None, "first_wins", "last_wins", "reject"])
def test_two_links_to_one_unnamed_skill_load_it_once(
    native: _Native, baseline: set[str], policy: str | None
):
    """Two configured links to one skill load it once, named after the kept link."""
    target = write_skill(native.tmp_path / "real" / "target")
    links = native.tmp_path / "links"
    links.mkdir()
    for name in ("alias-a", "alias-b"):
        (links / name).symlink_to(target, target_is_directory=True)
    skills: dict[str, Any] = {
        "directories": [
            {"skill": str(links / "alias-a")},
            {"skill": str(links / "alias-b")},
        ]
    }
    if policy is not None:
        skills["duplicates"] = policy

    plan, status = native.list(skills)

    kept = "alias-b" if policy == "last_wins" else "alias-a"
    assert set(_external(status, baseline)) == {kept} == set(_loaded(plan))


@_SYMLINKS
@pytest.mark.parametrize("policy", ["first_wins", "last_wins", "reject"])
def test_policy_keeps_a_linked_skill_named_after_its_link(
    native: _Native, baseline: set[str], policy: str
):
    """Under a policy an unnamed skill found through a link keeps the link's name."""
    team = native.tmp_path / "team"
    target = write_skill(team / "target")
    (team / "alias").symlink_to(target, target_is_directory=True)
    chosen = native.tmp_path / "chosen"
    chosen.mkdir()
    (chosen / "picked").symlink_to(target, target_is_directory=True)

    for entry, name in (
        ({"skill": str(chosen / "picked")}, "picked"),
        # Discovery reaches the target through the link first, by name order.
        ({"folder": str(team)}, "alias"),
    ):
        plan, status = native.list({"directories": [entry], "duplicates": policy})

        assert set(_external(status, baseline)) == {name} == set(_loaded(plan))
        report = plan.report()
        assert report is not None
        assert "skipped" not in report


@_SYMLINKS
def test_folder_passed_whole_reports_each_name_a_linked_skill_loads_under(
    native: _Native, baseline: set[str]
):
    """OpenClaw loads an unnamed skill once per path in a folder, as reported."""
    team = native.tmp_path / "team"
    target = write_skill(team / "target")
    (team / "alias").symlink_to(target, target_is_directory=True)
    named = write_skill(team / "named", "named")
    (team / "named-link").symlink_to(named, target_is_directory=True)

    plan, status = native.list({"directories": [{"folder": str(team)}]})

    assert set(_external(status, baseline)) == {"alias", "named", "target"}
    report = plan.report()
    assert report is not None
    assert _configured(report) == [
        loaded("alias", target, "skills.directories[0]"),
        loaded("named", named, "skills.directories[0]"),
        loaded("target", target, "skills.directories[0]"),
    ]


@_SYMLINKS
@pytest.mark.parametrize("policy", [None, "first_wins"])
def test_dot_dot_after_a_link_loads_the_folder_fabric_reports(
    native: _Native, baseline: set[str], policy: str | None
):
    """``link/../review`` loads the folder the link leads beside, under the name ``review``."""
    root = native.tmp_path
    (root / "data" / "sub").mkdir(parents=True)
    target = write_skill(root / "data" / "review", description="Through the link.")
    # A lexical normalization would reach this folder instead.
    write_skill(root / "review", description="Decoy.")
    (root / "link").symlink_to(root / "data" / "sub", target_is_directory=True)
    skills: dict[str, Any] = {"directories": [{"skill": "link/../review"}]}
    if policy is not None:
        skills["duplicates"] = policy

    plan, status = native.list(skills)

    assert _external(status, baseline) == {"review": "Through the link."}
    assert _configured(plan.report()) == [
        loaded("review", target, "skills.directories[0]")
    ]


def test_empty_directories_report_every_skill_openclaw_loads(native: _Native):
    """directories: [] reports workspace, shipped, and plugin skills with their folders."""
    notes = write_skill(native.workspace / "skills" / "notes", "notes", "Notes.")

    plan, status = native.list({"directories": []})

    report = plan.report()
    assert report is not None
    items = {item["name"]: item for item in report["loaded"]}
    assert items["notes"] == {
        "name": "notes",
        "path": str(notes.resolve()),
        "origin": "local",
    }
    rows = {row["name"]: row for row in status["skills"]}
    assert set(items) == set(rows)
    for name, row in rows.items():
        if row["source"] == "openclaw-extra":
            # A skill an OpenClaw plugin adds, from the installed package.
            assert items[name]["origin"] == "bundled"
        assert (Path(items[name]["path"]) / "SKILL.md").is_file()
