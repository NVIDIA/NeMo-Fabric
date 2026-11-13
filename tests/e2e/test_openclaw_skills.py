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
    command = Path(configured)
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
    """skills.bundled selects exactly the shipped skills OpenClaw allows."""
    plan, status = native.list({"bundled": bundled})

    shipped = {row["name"] for row in status["skills"] if row["bundled"]}
    unblocked = {
        row["name"]
        for row in status["skills"]
        if row["bundled"] and not row["blockedByAllowlist"]
    }
    report = plan.report()
    assert report is not None
    assert plan.inventory is not None
    assert shipped == plan.inventory.names
    expected = (
        shipped
        if allowed is None
        else shipped - {"weather"}
        if allowed == "all-but-weather"
        else allowed
    )
    assert unblocked == expected
    eligible = {
        row["name"]
        for row in status["skills"]
        if row["bundled"] and row["eligible"] and not row["blockedByAllowlist"]
    }
    unavailable = {item["name"] for item in report["bundled"].get("unavailable", [])}
    assert unavailable == expected - eligible
    # Selected lists only the selected bundled skills that load.
    assert set(report["bundled"]["selected"]) == expected & eligible


def test_higher_sources_hide_configured_skills_as_reported(native: _Native):
    """Workspace and shipped skills win over configured copies, as reported."""
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
    assert rows["weather"]["source"] == "openclaw-bundled"
    assert rows["weather"]["blockedByAllowlist"] is True
    report = plan.report()
    assert report is not None
    assert [item["name"] for item in report["loaded"]] == ["kept"]
    outcomes = {
        item["name"]: (item["outcome"], item.get("selected"))
        for item in report["duplicates"]
    }
    assert outcomes["review"] == (
        "shadowed",
        str((native.workspace / "skills" / "review").resolve()),
    )
    assert outcomes["github"][0] == "shadowed"
    # The excluded shipped copy still hides the configured one: neither loads.
    assert outcomes["weather"] == ("shadowed", None)


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
