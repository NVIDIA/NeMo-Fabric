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
from _utils.utils import POSIX_PERMISSIONS
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import RuntimeContext
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


def _skill(directory: Path, name: str | None, description: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["---"]
    if name is not None:
        lines.append(f"name: {name}")
    if description is not False:
        lines.append(f"description: {description or f'Copy at {directory.name}.'}")
    lines += ["---", "", "Body.", ""]
    (directory / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")
    return directory


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
            self._config(skills, settings),
            self._context(),
            base_dir=self.tmp_path,
            command=self.command,
            env=dict(os.environ),
        )

    def list(
        self, skills: dict[str, Any], settings: dict[str, Any] | None = None
    ) -> tuple[openclaw_skills.SkillPlan, dict]:
        """List OpenClaw's skills; like start, replace refused winners and list again."""
        plan = self.plan(skills, settings)
        while True:
            status = self._status(skills, settings, plan)
            if not openclaw_skills.apply_status(plan, status, fallback=True):
                return plan, status

    def _status(
        self,
        skills: dict[str, Any],
        settings: dict[str, Any] | None,
        plan: openclaw_skills.SkillPlan,
    ) -> dict:
        generated = adapter._openclaw_config(
            self._config(skills, settings),
            self._context(),
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
        result = subprocess.run(
            [str(self.command), "skills", "list", "--json", "--agent", "default"],
            cwd=self.workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        return json.loads(result.stdout)

    def _config(
        self, skills: dict[str, Any], settings: dict[str, Any] | None = None
    ) -> AgentConfig:
        return AgentConfig.from_mapping(
            {
                "harness": {
                    "settings": {
                        "openclaw_command": str(self.command),
                        **(settings or {}),
                    }
                },
                "models": {"default": {"provider": "test", "model": "unused"}},
                "skills": skills,
            }
        )

    def _context(self) -> RuntimeContext:
        return RuntimeContext.from_mapping(
            {
                "runtime_id": "skill-loading",
                "invocation_id": "skill-loading",
                "request_id": "skill-loading",
                "environment": {
                    "environment_id": "skill-loading",
                    "provider": "local",
                    "control_location": "in_env_control",
                    "workspace": str(self.workspace),
                    "env": {},
                    "ownership": "caller_owned",
                },
                "artifacts": {},
            }
        )


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


def _loaded(plan: openclaw_skills.SkillPlan) -> dict[str, str]:
    report = plan.report() or {}
    return {
        item["name"]: f"Copy at {Path(item['path']).name}."
        for item in report.get("loaded", [])
    }


def test_reported_skills_are_the_ones_openclaw_loads(native: _Native):
    """Folder, chosen-skill, and mixed entries report what OpenClaw loads."""
    root = native.tmp_path
    team = root / "team-skills"
    _skill(team / "debugging", "debugging")
    _skill(team / "group" / "code-review", "code-review")
    _skill(team / "group" / "deep" / "too-deep", "too-deep")
    _skill(team / "debugging" / "nested", "nested")
    _skill(team / ".hidden" / "secret", "secret")
    _skill(team / "node_modules" / "dependency", "dependency")
    _skill(team / "undescribed", "undescribed", description=False)
    _skill(team / "unnamed", None)
    broken = team / "broken-yaml"
    broken.mkdir()
    (broken / "SKILL.md").write_text(
        "---\nname: broken-yaml\ndescription: Broken.\nmetadata: [unclosed\n---\n",
        encoding="utf-8",
    )
    named = root / "skills"
    _skill(named / "a" / "b" / "c" / "deep-under-skills", "deep-under-skills")
    release = _skill(root / "opt" / "release-review", "release-review")
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
        "rejected_by_openclaw",
    }


@pytest.mark.parametrize(
    "name_lines",
    [
        pytest.param(["name: >-", "  folded-name"], id="folded-strip"),
        pytest.param(["name: |-", "  literal-name"], id="literal-strip"),
        pytest.param(['name: "escaped\\tname"'], id="double-quoted"),
        pytest.param(["name: plain # comment"], id="comment"),
        pytest.param(["name:"], id="empty"),
        pytest.param(["name: >", "  folded", "  clipped"], id="folded-clip"),
        pytest.param(["name: '  quoted name '"], id="single-quoted"),
        pytest.param(["name: ~"], id="null"),
        pytest.param(["name: 42"], id="number"),
        pytest.param(["name: yes"], id="yes"),
        pytest.param(["name: true"], id="true"),
        pytest.param(["name: 2026-01-02"], id="date"),
        pytest.param(["name: [a, b]"], id="sequence"),
        pytest.param(["label: &label aliased", "name: *label"], id="alias"),
        pytest.param(["label: &l 0x1F", "name: *l"], id="alias-number"),
        pytest.param(["label: &l [a, b]", "name: *l"], id="alias-sequence"),
        pytest.param(["label: &l ~", "name: *l"], id="alias-null"),
        pytest.param(["meta:", "  x: &l nested", "name: *l"], id="alias-nested"),
        pytest.param(["name:", "  - |", "    block"], id="block-in-sequence"),
    ],
)
def test_reported_names_match_openclaw(
    native: _Native, baseline: set[str], name_lines: list[str]
):
    """The name Fabric reads from front matter is the one OpenClaw lists."""
    skill = native.tmp_path / "team" / "named"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "\n".join(["---", *name_lines, "description: Copy at named.", "---", ""]),
        encoding="utf-8",
    )

    plan, status = native.list({"directories": [{"folder": str(skill.parent)}]})

    assert _external(status, baseline) == _loaded(plan)
    assert len(_loaded(plan)) == 1


@pytest.mark.parametrize("policy", [None, "first_wins", "last_wins"])
def test_duplicate_outcomes_match_openclaw(native: _Native, policy: str | None):
    """The reported copy of a duplicated name is the one OpenClaw loads."""
    root = native.tmp_path
    _skill(root / "team" / "review-team", "review")
    _skill(root / "team" / "debugging", "debugging")
    mounted = _skill(root / "opt" / "review-mounted", "review")
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
    plain = _skill(root / "team" / "review-plain", "review")
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
    weather = _skill(native.workspace / "skills" / "weather", "weather", "Mine.")

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
    assert [(item["name"], item["path"]) for item in report["loaded"]] == [
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
    team = _skill(native.tmp_path / "team" / "review", "review", "External.")
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
    assert [(item["name"], item["path"]) for item in report["loaded"]] == [
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
    expected = (
        plan.inventory.names
        if allowed is None
        else plan.inventory.names - {"weather"}
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
    assert unavailable == expected - eligible
    # Selected lists only the selected bundled skills that load.
    assert set(report["bundled"]["selected"]) == expected & eligible


@pytest.mark.parametrize(
    "bundled",
    [{"exclude": ["weather"]}, {"include": ["github"]}, {"enabled": False}],
)
def test_left_out_shipped_skill_does_not_hide_a_configured_copy(
    native: _Native, bundled: dict[str, Any]
):
    """A configured weather loads when bundled leaves out the shipped weather."""
    _skill(native.tmp_path / "team" / "weather", "weather", "Configured weather.")

    plan, status = native.list(
        {"directories": [{"folder": str(native.tmp_path / "team")}], "bundled": bundled}
    )

    rows = [row for row in status["skills"] if row["name"] == "weather"]
    assert [(row["source"], row["description"]) for row in rows] == [
        ("openclaw-extra", "Configured weather.")
    ]
    report = plan.report()
    assert report is not None
    assert [item["name"] for item in report["loaded"]] == ["weather"]
    assert "duplicates" not in report


def test_higher_sources_hide_configured_skills_as_reported(native: _Native):
    """Workspace and selected shipped skills win over configured copies, as reported."""
    root = native.tmp_path
    team = root / "team"
    _skill(team / "github", "github")
    _skill(team / "weather", "weather")
    _skill(team / "review", "review")
    _skill(team / "kept", "kept")
    _skill(native.workspace / "skills" / "review", "review", "Workspace copy.")

    plan, status = native.list(
        {"directories": [{"folder": str(team)}], "bundled": {"exclude": ["weather"]}}
    )

    rows = {row["name"]: row for row in status["skills"]}
    assert rows["review"]["source"] == "openclaw-workspace"
    assert rows["github"]["source"] == "openclaw-bundled"
    assert rows["weather"]["source"] == "openclaw-extra"
    report = plan.report()
    assert report is not None
    assert [item["name"] for item in report["loaded"]] == ["kept", "weather"]
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
    _skill(root / "team" / "github", "github", "Configured github.")
    _skill(native.workspace / "skills" / "github", "github", "Workspace github.")
    _skill(native.workspace / ".agents" / "skills" / "tmux", "tmux", "Project tmux.")

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
    _skill(root / "outside" / "escaped", "escaped")
    _skill(root / "outside" / "skills" / "nested-escaped", "nested-escaped")
    _skill(root / "team" / "inside", "inside")
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
    _skill(root / "team" / "top", "top")
    _skill(root / "team" / "skills" / "g" / "h" / "deep", "deep")
    _skill(root / "team" / "a" / "b" / "too-deep", "too-deep")
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
    _skill(root / "packed" / "skills", "packed")
    _skill(root / "packed" / "skills" / "nested", "nested")
    _skill(root / "team" / "top", "top")
    _skill(root / "team" / "skills" / "a" / "b" / "c" / "d" / "e" / ".hid", "hid")
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
    _skill(root / "a" / "top", "top")
    _skill(root / "a" / "b" / "skills" / "x", "x")

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
    assert [(item["name"], item["entry"]) for item in report["loaded"]] == [
        ("top", "skills.directories[0]"),
        ("x", "skills.directories[1]"),
    ]


@pytest.mark.parametrize(
    ("policy", "refused"), [("first_wins", "a"), ("last_wins", "b")]
)
def test_refused_winner_falls_back_to_the_next_copy(
    native: _Native, policy: str, refused: str
):
    """When OpenClaw refuses the winning copy, the next copy loads instead."""
    root = native.tmp_path
    for name in ("a", "b"):
        _skill(root / name / "review", "review")
    broken = root / refused / "review" / "SKILL.md"
    broken.write_text(
        "---\nname: review\ndescription: Broken.\nmetadata: [unclosed\n---\n",
        encoding="utf-8",
    )
    kept = root / ("b" if refused == "a" else "a") / "review"

    plan, status = native.list(
        {
            "directories": [{"folder": str(root / "a")}, {"folder": str(root / "b")}],
            "duplicates": policy,
        }
    )

    assert "review" in _external(status, set())
    assert plan.extra_dirs == [str(kept.resolve())]
    report = plan.report()
    assert report is not None
    assert report["loaded"][0]["path"] == str(kept.resolve())
    assert report["duplicates"][0]["selected"] == str(kept.resolve())
    assert report["skipped"] == [
        {
            "path": str(broken.parent.resolve()),
            "entry": f"skills.directories[{0 if refused == 'a' else 1}]",
            "reason": "rejected_by_openclaw",
        }
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
    _skill(native.tmp_path / "team" / name, name)
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
        assert "loaded" not in report
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
    debugging = _skill(team / "debugging", "debugging")
    review = _skill(team / "review", "review")
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
        _skill(team / "new-skill", "new-skill")


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
    _skill(agent_skills / "triage", "triage")
    restarted, status = native.list(skills)

    assert _external(first_status, baseline) == {}
    assert first.report() == {"empty_entries": ["skills.directories[0]"]}
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
    protected = _skill(team / "review", "review") / "SKILL.md"
    original = protected.read_text(encoding="utf-8")
    make_read_only(team)
    agent_skills = native.tmp_path / "agent-skills"
    agent_skills.mkdir()
    skills = {"directories": [{"folder": str(team)}, {"folder": str(agent_skills)}]}

    first, _ = native.list(skills)
    with pytest.raises(PermissionError):
        protected.write_text("changed", encoding="utf-8")
    with pytest.raises(PermissionError):
        _skill(team / "triage", "triage")
    _skill(agent_skills / "triage", "triage")
    restarted, status = native.list(skills)

    assert set(_loaded(first)) == {"review"}
    assert protected.read_text(encoding="utf-8") == original
    assert _external(status, baseline) == _loaded(restarted)
    report = restarted.report()
    assert report is not None
    assert [(item["name"], item["entry"]) for item in report["loaded"]] == [
        ("review", "skills.directories[0]"),
        ("triage", "skills.directories[1]"),
    ]


async def _start_runtime(
    native: _Native, skills: dict[str, Any]
) -> adapter.OpenClawRuntime:
    """Start a real OpenClaw Gateway for ``skills``; no model is called."""

    runtime = adapter.OpenClawRuntime()
    context = native._context().to_mapping()
    home = native.tmp_path / "home"
    home.mkdir(exist_ok=True)
    context["environment"]["env"] = {"HOME": str(home)}
    await runtime.start(
        {
            "config": native._config(skills, {"startup_timeout_seconds": 120}),
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
    _skill(agent_skills / "weather", "weather", "Configured weather.")
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
    assert [(item["name"], item["entry"]) for item in first["loaded"]] == [
        ("weather", "skills.directories[0]")
    ]
    assert (rows["weather"]["source"], rows["weather"]["description"]) == (
        "openclaw-extra",
        "Configured weather.",
    )
    # The agent saves a new skill into its read-write folder while stopped.
    _skill(agent_skills / "triage", "triage")

    runtime = await _start_runtime(native, skills)
    try:
        assert runtime._process is not None and runtime._process.returncode is None
        restarted = runtime._skills_output()["skills"]
        rows = _running_skills(runtime)
    finally:
        await runtime.stop()
    assert [(item["name"], item["entry"]) for item in restarted["loaded"]] == [
        ("triage", "skills.directories[0]"),
        ("weather", "skills.directories[0]"),
    ]
    assert rows["triage"]["source"] == "openclaw-extra"
    assert rows["weather"]["description"] == "Configured weather."
    assert restarted["bundled"]["selected"] == first["bundled"]["selected"]
    assert "weather" not in restarted["bundled"]["selected"]
