# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured skill configuration for the OpenClaw adapter."""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from _utils.openclaw_skills import agent_config
from _utils.openclaw_skills import copy
from _utils.openclaw_skills import duplicate
from _utils.openclaw_skills import loaded
from _utils.openclaw_skills import refused_copies
from _utils.openclaw_skills import refused_skill
from _utils.openclaw_skills import runtime_context
from _utils.openclaw_skills import skipped
from _utils.openclaw_skills import write_skill
from _utils.utils import POSIX_PERMISSIONS
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.skills import CandidateInfo
from nemo_fabric_adapter_contract.skills import front_matter_reader
from nemo_fabric_adapters.openclaw import adapter
from nemo_fabric_adapters.openclaw import skills as openclaw_skills


class FakeReader:
    """Reads the plain front matter these tests write, and refuses registered skills.

    OpenClaw's own parser is checked against ``openclaw skills list`` in
    ``tests/e2e/test_openclaw_skills.py``.
    """

    def __init__(self) -> None:
        self.refused: dict[Path, str] = {}
        self.calls: list[list[Path]] = []
        self.requests: list[tuple[Path | None, float]] = []
        """``(command, timeout)`` of each plan that asked for this reader."""

    def refuse(self, folder: Path, reason: str) -> None:
        self.refused[folder.resolve()] = reason

    def __call__(self, folders: Sequence[Path]) -> dict[Path, CandidateInfo]:
        self.calls.append(list(folders))
        simple = front_matter_reader(required_fields=("description",))(folders)
        return {
            folder: CandidateInfo(None, valid=False, reason=self.refused[folder])
            if folder in self.refused
            else simple[folder]
            for folder in folders
        }


@pytest.fixture(name="reader", autouse=True)
def reader_fixture(monkeypatch: pytest.MonkeyPatch) -> FakeReader:
    """Every plan reads skills with a :class:`FakeReader` instead of node."""

    reader = FakeReader()

    def candidate_reader(
        command: Path | None, *, env: Any, timeout: float
    ) -> FakeReader:
        reader.requests.append((command, timeout))
        return reader

    monkeypatch.setattr(openclaw_skills, "candidate_reader", candidate_reader)
    return reader


@pytest.fixture(name="package")
def package_fixture(tmp_path: Path) -> Path:
    """A minimal installed OpenClaw package with shipped and Custodian skills."""

    root = tmp_path / "node_modules/openclaw"
    root.mkdir(parents=True)
    (root / "package.json").write_text('{"name": "openclaw"}', encoding="utf-8")
    (root / "openclaw.mjs").write_text("", encoding="utf-8")
    for name in ("github", "weather", "tmux"):
        write_skill(root / "skills" / name, name)
    (root / "skills/pyproject.toml").write_text("", encoding="utf-8")
    write_skill(root / "custodian-skills/diagnose-gateway", "diagnose-gateway")
    return root


@pytest.fixture(name="workspace")
def workspace_fixture(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


def _plan(
    tmp_path: Path,
    workspace: Path,
    skills: dict[str, Any],
    package: Path | None = None,
) -> openclaw_skills.SkillPlan:
    return adapter._skill_plan(
        agent_config(skills),
        runtime_context(workspace),
        base_dir=tmp_path,
        command=package / "openclaw.mjs" if package is not None else None,
        env={},
    )


def _generated(
    tmp_path: Path, workspace: Path, skills: dict[str, Any], package: Path | None = None
) -> tuple[dict[str, Any], openclaw_skills.SkillPlan]:
    plan = _plan(tmp_path, workspace, skills, package)
    generated = adapter._openclaw_config(
        agent_config(skills),
        runtime_context(workspace),
        base_dir=tmp_path,
        port=20_000,
        token_env="OPENCLAW_GATEWAY_TOKEN",
        skill_plan=plan,
    )
    return generated, plan


def _loaded(plan: openclaw_skills.SkillPlan) -> list[tuple[str, str]]:
    report = plan.report() or {}
    return [(item["name"], item["entry"]) for item in report.get("loaded", [])]


def test_folder_entry_passes_the_folder_and_reports_its_skills(
    tmp_path: Path, workspace: Path
):
    """A read-only folder is passed as given and reported with OpenClaw's rules."""
    team = tmp_path / "team-skills"
    write_skill(team / "debugging", "debugging")
    write_skill(team / "group" / "code-review", "code-review")
    write_skill(team / "group" / "deep" / "too-deep", "too-deep")
    write_skill(team / "debugging" / "nested", "nested")
    write_skill(team / ".hidden" / "secret", "secret")
    write_skill(team / "node_modules" / "dependency", "dependency")
    write_skill(team / "undescribed", "undescribed", description=None)
    write_skill(team / "unnamed")
    (team / "drafts").mkdir()
    (team / "README.md").write_text("Not a skill.\n", encoding="utf-8")

    generated, plan = _generated(
        tmp_path, workspace, {"directories": [{"folder": str(team)}]}
    )

    assert generated["skills"] == {"load": {"extraDirs": [str(team)]}}
    assert _loaded(plan) == [
        ("debugging", "skills.directories[0]"),
        ("code-review", "skills.directories[0]"),
        ("unnamed", "skills.directories[0]"),
    ]
    report = plan.report()
    assert report is not None
    assert report["skipped"] == [
        skipped(
            team / "undescribed",
            "skills.directories[0]",
            "missing_frontmatter_field:description",
        )
    ]
    assert report["loaded"][0] == loaded(
        "debugging", team / "debugging", "skills.directories[0]"
    )


def test_folder_named_skills_is_searched_six_levels_deep(
    tmp_path: Path, workspace: Path
):
    """OpenClaw searches a folder named skills, or its skills subfolder, more deeply."""
    named = tmp_path / "skills"
    write_skill(named / "a" / "b" / "c" / "d" / "e" / "six", "six")
    write_skill(named / "a" / "b" / "c" / "d" / "e" / "f" / "seven", "seven")
    repository = tmp_path / "repository"
    write_skill(repository / "skills" / "x" / "y" / "z" / "nested-root", "nested-root")

    _, plan = _generated(
        tmp_path,
        workspace,
        {"directories": [{"folder": str(named)}, {"folder": str(repository)}]},
    )

    assert _loaded(plan) == [
        ("six", "skills.directories[0]"),
        ("nested-root", "skills.directories[1]"),
    ]


def test_folders_with_different_depths_each_use_their_own(
    tmp_path: Path, workspace: Path
):
    """A folder named skills keeps 6 levels while another folder keeps 2."""
    named = tmp_path / "skills"
    write_skill(named / "a" / "b" / "deep", "deep")
    team = tmp_path / "team"
    write_skill(team / "a" / "b" / "too-deep", "too-deep")
    write_skill(team / "a" / "shallow", "shallow")

    _, plan = _generated(
        tmp_path,
        workspace,
        {
            "directories": [{"folder": str(named)}, {"folder": str(team)}],
            "duplicates": "reject",
        },
    )

    assert _loaded(plan) == [
        ("deep", "skills.directories[0]"),
        ("shallow", "skills.directories[1]"),
    ]


def test_skill_entries_pass_each_chosen_skill(tmp_path: Path, workspace: Path):
    """Chosen skills are passed individually, read-only or not."""
    review = write_skill(tmp_path / "opt/release-review", "release-review")
    setup = write_skill(tmp_path / "opt/project-setup")

    generated, plan = _generated(
        tmp_path,
        workspace,
        {"directories": [{"skill": str(review)}, {"skill": str(setup)}]},
    )

    assert generated["skills"]["load"]["extraDirs"] == [str(review), str(setup)]
    assert _loaded(plan) == [
        ("release-review", "skills.directories[0]"),
        ("project-setup", "skills.directories[1]"),
    ]


def test_mixed_entries_keep_entry_order(tmp_path: Path, workspace: Path):
    """Folders and chosen skills mix in one list."""
    team = tmp_path / "team-skills"
    write_skill(team / "debugging", "debugging")
    project = tmp_path / "project-skills"
    write_skill(project / "integration-testing", "integration-testing")
    review = write_skill(tmp_path / "opt/release-review", "release-review")

    generated, plan = _generated(
        tmp_path,
        workspace,
        {
            "directories": [
                {"folder": str(team)},
                {"folder": str(project)},
                {"skill": str(review)},
            ]
        },
    )

    assert generated["skills"]["load"]["extraDirs"] == [
        str(team),
        str(project),
        str(review),
    ]
    assert [name for name, _ in _loaded(plan)] == [
        "debugging",
        "integration-testing",
        "release-review",
    ]


def test_empty_folder_and_empty_list(tmp_path: Path, workspace: Path):
    """An empty folder is valid and reported; an empty list loads nothing."""
    empty = tmp_path / "empty"
    empty.mkdir()

    generated, plan = _generated(
        tmp_path, workspace, {"directories": [{"folder": str(empty)}]}
    )
    assert generated["skills"] == {"load": {"extraDirs": [str(empty)]}}
    assert plan.report() == {"empty_entries": ["skills.directories[0]"]}

    generated, plan = _generated(tmp_path, workspace, {"directories": []})
    assert "skills" not in generated
    assert plan.report() is None


@POSIX_PERMISSIONS
@pytest.mark.parametrize("kind", ["folder", "skill"], ids=["folder", "chosen-skills"])
def test_read_only_entries_load_and_cannot_be_changed(
    tmp_path: Path, workspace: Path, make_read_only, kind: str
):
    """Start checks need only read access, and never change a skill."""
    team = tmp_path / "team"
    debugging = write_skill(team / "debugging", "debugging")
    review = write_skill(team / "review", "review")
    make_read_only(team)
    entries = (
        [{"folder": str(team)}]
        if kind == "folder"
        else [{"skill": str(debugging)}, {"skill": str(review)}]
    )

    generated, plan = _generated(tmp_path, workspace, {"directories": entries})

    assert generated["skills"]["load"]["extraDirs"] == [
        entry[kind] for entry in entries
    ]
    assert [name for name, _ in _loaded(plan)] == ["debugging", "review"]
    with pytest.raises(PermissionError):
        (review / "SKILL.md").write_text("changed", encoding="utf-8")
    with pytest.raises(PermissionError):
        write_skill(team / "new-skill", "new-skill")


def test_skill_written_into_a_writable_folder_loads_at_the_next_start(
    tmp_path: Path, workspace: Path
):
    """The plan is made at start: a skill written later loads after a restart."""
    agent_skills = tmp_path / "agent-skills"
    agent_skills.mkdir()
    skills = {"directories": [{"folder": str(agent_skills)}]}

    first = _plan(tmp_path, workspace, skills)
    # The agent saves a new skill into its read-write folder.
    write_skill(agent_skills / "triage", "triage")

    assert first.report() == {"empty_entries": ["skills.directories[0]"]}
    assert _loaded(_plan(tmp_path, workspace, skills)) == [
        ("triage", "skills.directories[0]")
    ]


@POSIX_PERMISSIONS
def test_agent_adds_skills_beside_read_only_skills_it_cannot_change(
    tmp_path: Path, workspace: Path, make_read_only
):
    team = tmp_path / "team"
    protected = write_skill(team / "review", "review") / "SKILL.md"
    original = protected.read_text(encoding="utf-8")
    make_read_only(team)
    agent_skills = tmp_path / "agent-skills"
    agent_skills.mkdir()
    skills = {"directories": [{"folder": str(team)}, {"folder": str(agent_skills)}]}

    first = _plan(tmp_path, workspace, skills)
    with pytest.raises(PermissionError):
        protected.write_text("changed", encoding="utf-8")
    with pytest.raises(PermissionError):
        write_skill(team / "triage", "triage")
    write_skill(agent_skills / "triage", "triage")
    restarted = _plan(tmp_path, workspace, skills)

    assert _loaded(first) == [("review", "skills.directories[0]")]
    assert protected.read_text(encoding="utf-8") == original
    assert _loaded(restarted) == [
        ("review", "skills.directories[0]"),
        ("triage", "skills.directories[1]"),
    ]


def test_native_duplicates_pass_every_copy_and_report_the_later_entry(
    tmp_path: Path, workspace: Path
):
    """Without a policy OpenClaw keeps the copy from the later root."""
    first = write_skill(tmp_path / "team/review", "review")
    second = write_skill(tmp_path / "opt/review", "review")

    generated, plan = _generated(
        tmp_path,
        workspace,
        {"directories": [{"folder": str(tmp_path / "team")}, {"skill": str(second)}]},
    )

    assert generated["skills"]["load"]["extraDirs"] == [
        str(tmp_path / "team"),
        str(second),
    ]
    report = plan.report()
    assert report is not None
    assert report["loaded"] == [loaded("review", second, "skills.directories[1]")]
    assert report["duplicates"] == [
        duplicate(
            "review",
            "selected",
            second,
            copy("external", first, "skills.directories[0]"),
            copy("external", second, "skills.directories[1]"),
        )
    ]


def test_native_duplicates_inside_one_folder_are_ambiguous(
    tmp_path: Path, workspace: Path
):
    """Two copies in one folder resolve in OpenClaw's path order."""
    write_skill(tmp_path / "team/x/review", "review")
    write_skill(tmp_path / "team/y/review", "review")

    _, plan = _generated(
        tmp_path, workspace, {"directories": [{"folder": str(tmp_path / "team")}]}
    )

    report = plan.report()
    assert report is not None
    assert [item["outcome"] for item in report["duplicates"]] == ["ambiguous"]


@pytest.mark.parametrize(
    ("policy", "winner"), [("first_wins", "first"), ("last_wins", "second")]
)
def test_explicit_policy_passes_only_selected_skill_folders(
    tmp_path: Path, workspace: Path, policy: str, winner: str
):
    """An explicit policy passes each selected skill so the shadowed copy never loads."""
    team = tmp_path / "team"
    copies = {
        "first": write_skill(team / "review", "review"),
        "second": write_skill(tmp_path / "opt/review", "review"),
    }
    other = write_skill(team / "debugging", "debugging")

    generated, plan = _generated(
        tmp_path,
        workspace,
        {
            "directories": [{"folder": str(team)}, {"skill": str(copies["second"])}],
            "duplicates": policy,
        },
    )

    assert generated["skills"]["load"]["extraDirs"] == [
        str(other.resolve()),
        str(copies[winner].resolve()),
    ]
    report = plan.report()
    assert report is not None
    assert report["duplicates"][0]["outcome"] == "selected"
    assert report["duplicates"][0]["selected"] == str(copies[winner].resolve())


def test_reject_policy_fails_naming_both_entries(tmp_path: Path, workspace: Path):
    """reject fails start with the second entry and both paths."""
    first = write_skill(tmp_path / "team/review", "review")
    second = write_skill(tmp_path / "opt/review", "review")

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        _plan(
            tmp_path,
            workspace,
            {
                "directories": [
                    {"folder": str(tmp_path / "team")},
                    {"skill": str(second)},
                ],
                "duplicates": "reject",
            },
        )

    assert caught.value.code == "openclaw_skill_invalid"
    assert caught.value.metadata == {
        "field": "skills.directories[1]",
        "reason": "duplicate",
        "path": str(first.resolve()),
        "paths": [str(first.resolve()), str(second.resolve())],
    }


@pytest.mark.parametrize("policy", ["reject", "first_wins", "last_wins"])
def test_explicit_policy_rejects_duplicates_inside_one_folder(
    tmp_path: Path, workspace: Path, policy: str
):
    """Two copies inside one folder entry fail under any explicit policy."""
    write_skill(tmp_path / "team/x/review", "review")
    write_skill(tmp_path / "team/y/review", "review")

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        _plan(
            tmp_path,
            workspace,
            {
                "directories": [{"folder": str(tmp_path / "team")}],
                "duplicates": policy,
            },
        )

    assert caught.value.code == "openclaw_skill_invalid"
    assert caught.value.metadata["field"] == "skills.directories[0]"
    assert caught.value.metadata["reason"] == "duplicate"


@pytest.mark.parametrize(
    ("entry", "setup", "code", "reason", "field"),
    [
        ("folder", "missing", "openclaw_skill_not_found", "not_found", "folder"),
        ("folder", "file", "openclaw_skill_invalid", "invalid", "folder"),
        ("skill", "missing", "openclaw_skill_not_found", "not_found", "skill"),
        ("skill", "no_manifest", "openclaw_skill_invalid", "invalid", "skill"),
        ("folder", "is_skill", "openclaw_skill_invalid", "folder_is_skill", "folder"),
    ],
)
def test_entry_failures_name_the_field_reason_and_path(
    tmp_path: Path,
    workspace: Path,
    entry: str,
    setup: str,
    code: str,
    reason: str,
    field: str,
):
    """Task-environment failures map to the adapter's error codes."""
    good = write_skill(tmp_path / "good", "good")
    path = tmp_path / "entry"
    if setup == "file":
        path.write_text("not a folder", encoding="utf-8")
    elif setup == "no_manifest":
        path.mkdir()
    elif setup == "is_skill":
        write_skill(path, "entry")

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        _plan(
            tmp_path,
            workspace,
            {"directories": [{"skill": str(good)}, {entry: str(path)}]},
        )

    assert caught.value.code == code
    assert caught.value.metadata == {
        "field": f"skills.directories[1].{field}",
        "reason": reason,
        "path": str(path),
    }


def test_relative_entries_resolve_against_the_base_directory(
    tmp_path: Path, workspace: Path
):
    """Relative entries resolve like paths do."""
    write_skill(tmp_path / "team/debugging", "debugging")

    generated, _ = _generated(
        tmp_path, workspace, {"directories": [{"folder": "team"}]}
    )

    assert generated["skills"]["load"]["extraDirs"] == [str(tmp_path / "team")]


def test_paths_behavior_is_unchanged(tmp_path: Path, workspace: Path, package: Path):
    """Legacy paths still pass resolved folders and combine with bundled selection."""
    skill = write_skill(tmp_path / "skills", "legacy")

    generated, plan = _generated(
        tmp_path,
        workspace,
        {"paths": ["skills"], "bundled": {"include": ["weather"]}},
        package,
    )

    assert generated["skills"] == {"load": {"extraDirs": [str(skill.resolve())]}}
    assert plan.stage_bundled is True
    assert plan.report() == {"bundled": {"selected": ["weather"]}}
    generated, plan = _generated(tmp_path, workspace, {"paths": ["skills"]})
    assert generated["skills"] == {"load": {"extraDirs": [str(skill.resolve())]}}
    assert plan.report() is None


@pytest.mark.parametrize(
    ("bundled", "staged", "selected"),
    [
        ({}, None, ["diagnose-gateway", "github", "tmux", "weather"]),
        ({"exclude": []}, None, ["diagnose-gateway", "github", "tmux", "weather"]),
        (
            {"include": ["diagnose-gateway", "github", "tmux", "weather"]},
            None,
            ["diagnose-gateway", "github", "tmux", "weather"],
        ),
        ({"enabled": False}, ([], []), []),
        ({"include": []}, ([], []), []),
        (
            {"include": ["weather", "github"]},
            (["github", "weather"], []),
            ["github", "weather"],
        ),
        (
            {"exclude": ["tmux"]},
            (["github", "weather"], ["diagnose-gateway"]),
            ["diagnose-gateway", "github", "weather"],
        ),
        (
            {"exclude": ["diagnose-gateway", "github", "tmux", "weather"]},
            ([], []),
            [],
        ),
    ],
)
def test_bundled_selection_stages_only_the_selected_shipped_skills(
    tmp_path: Path,
    workspace: Path,
    package: Path,
    bundled: dict[str, Any],
    staged: tuple[list[str], list[str]] | None,
    selected: list[str],
):
    """A selection that leaves out a shipped skill copies only the selected ones.

    Selecting every shipped skill keeps OpenClaw's own folder. OpenClaw's
    ``skills.allowBundled`` is never written.
    """
    generated, plan = _generated(tmp_path, workspace, {"bundled": bundled}, package)

    assert "allowBundled" not in generated.get("skills", {})
    assert plan.report() == {"bundled": {"selected": selected}}
    folder = openclaw_skills.stage_bundled(plan, tmp_path / "stage")
    if staged is None:
        assert folder is None
        assert not (tmp_path / "stage").exists()
        return
    assert folder == tmp_path / "stage/skills"
    custodian = tmp_path / "stage/custodian-skills"
    assert (
        sorted(child.name for child in folder.iterdir()),
        sorted(child.name for child in custodian.iterdir()),
    ) == staged
    for name in staged[0]:
        assert (folder / name / "SKILL.md").read_bytes() == (
            package / "skills" / name / "SKILL.md"
        ).read_bytes()


def test_staged_shipped_skills_keep_their_layout_and_files(
    tmp_path: Path, workspace: Path, package: Path
):
    """A staged skill keeps its path below the shipped folder and its support files."""
    nested = write_skill(package / "skills/group/nested", "nested")
    script = nested / "scripts/run.sh"
    script.parent.mkdir()
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    script.chmod(0o755)

    plan = _plan(tmp_path, workspace, {"bundled": {"include": ["nested"]}}, package)
    folder = openclaw_skills.stage_bundled(plan, tmp_path / "stage")

    assert folder is not None
    copied = folder / "group/nested/scripts/run.sh"
    assert copied.read_text(encoding="utf-8") == "#!/bin/sh\n"
    assert os.access(copied, os.X_OK) or os.name == "nt"
    assert not copied.is_symlink()
    assert sorted(path.name for path in folder.iterdir()) == ["group"]


def test_staging_failure_names_the_bundled_field(
    tmp_path: Path, workspace: Path, package: Path
):
    """A staged folder that cannot be written fails start naming skills.bundled."""
    plan = _plan(tmp_path, workspace, {"bundled": {"exclude": ["tmux"]}}, package)
    blocked = tmp_path / "stage"
    blocked.write_text("not a folder", encoding="utf-8")

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        openclaw_skills.stage_bundled(plan, blocked)

    assert caught.value.code == "openclaw_skill_invalid"
    assert caught.value.metadata == {
        "field": "skills.bundled",
        "reason": "staging_failed",
    }


@pytest.mark.parametrize("field", ["include", "exclude"])
def test_unknown_bundled_name_fails_start(
    tmp_path: Path, workspace: Path, package: Path, field: str
):
    """An unknown bundled name names its field and the name."""
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        _plan(tmp_path, workspace, {"bundled": {field: ["weather", "nope"]}}, package)

    assert caught.value.code == "openclaw_skill_invalid"
    assert caught.value.metadata == {
        "field": f"skills.bundled.{field}",
        "reason": "unknown_bundled_skill",
        "name": "nope",
    }


def test_bundled_selection_needs_the_shipped_skills_folder(
    tmp_path: Path, workspace: Path
):
    """Without the installed package, bundled selection cannot be checked."""
    command = tmp_path / "bin/openclaw"
    command.parent.mkdir()
    command.write_text("", encoding="utf-8")

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        adapter._skill_plan(
            agent_config({"bundled": {"enabled": False}}),
            runtime_context(workspace),
            base_dir=tmp_path,
            command=command,
            env={},
        )

    assert caught.value.metadata == {
        "field": "skills.bundled",
        "reason": "inventory_unavailable",
    }


def test_bundled_folder_follows_openclaw_override(tmp_path: Path, package: Path):
    """OPENCLAW_BUNDLED_SKILLS_DIR wins over the package folder, as in OpenClaw."""
    command = package / "openclaw.mjs"
    link = tmp_path / "bin/openclaw"
    link.parent.mkdir()
    link.symlink_to(command)

    assert (
        openclaw_skills.bundled_skills_dir(link, {}) == (package / "skills").resolve()
    )
    assert openclaw_skills.bundled_skills_dir(
        link, {"OPENCLAW_BUNDLED_SKILLS_DIR": str(tmp_path / "other")}
    ) == (tmp_path / "other")


def test_higher_sources_hide_configured_skills(
    tmp_path: Path, workspace: Path, package: Path
):
    """Workspace and selected shipped skills outrank skills.directories."""
    team = tmp_path / "team"
    mine = write_skill(team / "review", "review")
    github = write_skill(team / "github", "github")
    weather = write_skill(team / "weather", "weather")
    kept = write_skill(team / "kept", "kept")
    local = write_skill(workspace / "skills" / "review", "review")

    _, plan = _generated(
        tmp_path,
        workspace,
        {"directories": [{"folder": str(team)}], "bundled": {"exclude": ["weather"]}},
        package,
    )

    report = plan.report()
    assert report is not None
    # The excluded shipped weather is not staged, so the configured copy loads.
    assert [item["name"] for item in report["loaded"]] == ["kept", "weather"]
    assert report["loaded"][0]["path"] == str(kept.resolve())
    assert report["loaded"][1]["path"] == str(weather.resolve())
    duplicates = {item["name"]: item for item in report["duplicates"]}
    assert set(duplicates) == {"github", "review"}
    assert duplicates["review"] == duplicate(
        "review",
        "shadowed",
        local,
        copy("external", mine, "skills.directories[0]"),
        copy("local", local),
    )
    assert duplicates["github"] == duplicate(
        "github",
        "shadowed",
        package / "skills/github",
        copy("external", github, "skills.directories[0]"),
        copy("bundled", package / "skills/github"),
    )


def test_collisions_list_every_copy_from_every_source(
    tmp_path: Path, workspace: Path, package: Path
):
    """Each copy is listed with its origin, whichever sources clash."""
    team = tmp_path / "team"
    mine = write_skill(team / "github", "github")
    agents = write_skill(workspace / ".agents/skills/github", "github")
    local = write_skill(workspace / "skills/github", "github")
    tmux = write_skill(workspace / ".agents/skills/tmux", "tmux")
    first = write_skill(workspace / "skills/one/twin", "twin")
    second = write_skill(workspace / "skills/two/twin", "twin")
    shipped = (package / "skills/github").resolve()

    _, plan = _generated(
        tmp_path, workspace, {"directories": [{"folder": str(team)}]}, package
    )

    report = plan.report()
    assert report is not None
    assert "loaded" not in report
    duplicates = {item["name"]: item for item in report["duplicates"]}
    assert duplicates["github"] == duplicate(
        "github",
        "shadowed",
        local,
        copy("external", mine, "skills.directories[0]"),
        copy("bundled", shipped),
        copy("local", agents),
        copy("local", local),
    )
    # No configured copy: the workspace copy beats the shipped one.
    assert duplicates["tmux"] == duplicate(
        "tmux",
        "selected",
        tmux,
        copy("bundled", package / "skills/tmux"),
        copy("local", tmux),
    )
    # Two copies in the winning folder: OpenClaw's scan order decides.
    assert duplicates["twin"] == duplicate(
        "twin", "ambiguous", None, copy("local", first), copy("local", second)
    )
    assert "diagnose-gateway" not in duplicates


def test_bundled_only_selection_reports_shipped_and_local_clashes(
    tmp_path: Path, workspace: Path, package: Path
):
    """Without directories, a selected shipped skill a workspace skill hides is reported."""
    local = write_skill(workspace / "skills/weather", "weather")
    write_skill(workspace / "skills/tmux", "tmux")

    _, plan = _generated(
        tmp_path, workspace, {"bundled": {"exclude": ["tmux"]}}, package
    )

    report = plan.report()
    assert report is not None
    # tmux is left out, so only weather clashes.
    assert report["duplicates"] == [
        duplicate(
            "weather",
            "selected",
            local,
            copy("bundled", package / "skills/weather"),
            copy("local", local),
        )
    ]
    assert report["bundled"] == {"selected": ["diagnose-gateway", "github", "weather"]}


def test_explicit_policy_applies_only_between_entries(
    tmp_path: Path, workspace: Path, package: Path
):
    """first_wins picks between entries; a higher source still hides the winner."""
    first = write_skill(tmp_path / "a/github", "github")
    second = write_skill(tmp_path / "b/github", "github")

    generated, plan = _generated(
        tmp_path,
        workspace,
        {
            "directories": [
                {"folder": str(tmp_path / "a")},
                {"folder": str(tmp_path / "b")},
            ],
            "duplicates": "first_wins",
        },
        package,
    )

    assert generated["skills"]["load"]["extraDirs"] == [str(first.resolve())]
    report = plan.report()
    assert report is not None
    assert "loaded" not in report
    assert report["duplicates"] == [
        duplicate(
            "github",
            "shadowed",
            package / "skills/github",
            copy("external", first, "skills.directories[0]"),
            copy("external", second, "skills.directories[1]"),
            copy("bundled", package / "skills/github"),
        )
    ]


def test_status_reports_rejected_and_unavailable_skills(
    tmp_path: Path, workspace: Path, package: Path
):
    """OpenClaw's own list refines the report."""
    team = tmp_path / "team"
    write_skill(team / "kept", "kept")
    rejected = write_skill(team / "rejected", "rejected")
    write_skill(workspace / "skills" / "tmux", "tmux")
    _, plan = _generated(
        tmp_path,
        workspace,
        {"directories": [{"folder": str(team)}], "bundled": {}},
        package,
    )

    openclaw_skills.apply_status(
        plan,
        {
            "skills": [
                {"name": "kept", "source": "openclaw-extra", "eligible": True},
                {
                    "name": "github",
                    "source": "openclaw-bundled",
                    "bundled": True,
                    "eligible": False,
                    "disabled": False,
                    "missing": {"bins": ["gh"], "env": [], "os": []},
                },
                {
                    "name": "weather",
                    "source": "openclaw-bundled",
                    "bundled": True,
                    "eligible": True,
                    "disabled": True,
                },
                {"name": "tmux", "source": "openclaw-workspace", "eligible": True},
            ]
        },
    )

    report = plan.report()
    assert report is not None
    assert [item["name"] for item in report["loaded"]] == ["kept"]
    assert report["skipped"] == [
        skipped(rejected, "skills.directories[0]", "rejected_by_openclaw")
    ]
    # Only the selected bundled skills that load are listed as selected.
    assert report["bundled"]["selected"] == []
    assert report["bundled"]["unavailable"] == [
        {"name": "diagnose-gateway", "reason": "not_loaded"},
        {
            "name": "github",
            "reason": "missing_requirements",
            "missing": {"bins": ["gh"]},
        },
        {"name": "tmux", "reason": "shadowed", "by": "openclaw-workspace"},
        {"name": "weather", "reason": "disabled"},
    ]


@pytest.mark.parametrize("policy", [None, "first_wins", "last_wins", "reject"])
def test_configured_workspace_skill_keeps_its_workspace_rank(
    tmp_path: Path, workspace: Path, package: Path, policy: str | None
):
    """A configured folder that is also a workspace skill outranks a shipped copy."""
    weather = write_skill(workspace / "skills" / "weather", "weather", "Workspace.")
    skills: dict[str, Any] = {"directories": [{"skill": str(weather)}], "bundled": {}}
    if policy is not None:
        skills["duplicates"] = policy

    plan = _plan(tmp_path, workspace, skills, package)

    report = plan.report()
    assert report is not None
    assert _loaded(plan) == [("weather", "skills.directories[0]")]
    assert report["duplicates"] == [
        duplicate(
            "weather",
            "selected",
            weather,
            copy("external", weather, "skills.directories[0]"),
            copy("bundled", package / "skills/weather"),
        )
    ]

    # OpenClaw's list agrees, so the configured skill stays the winner.
    openclaw_skills.apply_status(
        plan, {"skills": [{"name": "weather", "source": "openclaw-workspace"}]}
    )
    refined = plan.report()
    assert refined is not None
    assert (refined["loaded"], refined["duplicates"]) == (
        report["loaded"],
        report["duplicates"],
    )


def test_configured_workspace_skill_wins_over_another_configured_copy(
    tmp_path: Path, workspace: Path
):
    """A policy's pick cannot outrank a configured copy OpenClaw loads natively."""
    local = write_skill(workspace / "skills" / "review", "review", "Workspace.")
    team = write_skill(tmp_path / "team" / "review", "review", "Team.")

    plan = _plan(
        tmp_path,
        workspace,
        {
            "directories": [{"skill": str(team)}, {"skill": str(local)}],
            "duplicates": "first_wins",
        },
    )

    report = plan.report()
    assert report is not None
    assert _loaded(plan) == [("review", "skills.directories[1]")]
    assert report["duplicates"][0]["outcome"] == "selected"
    assert report["duplicates"][0]["selected"] == str(local.resolve())
    assert plan.extra_dirs == [str(team.resolve())]


@pytest.mark.parametrize("policy", [None, "first_wins"])
def test_invalid_workspace_skill_does_not_hide_a_configured_one(
    tmp_path: Path, workspace: Path, reader: FakeReader, policy: str | None
):
    """A workspace copy OpenClaw refuses is not reported as the winner."""
    team = write_skill(tmp_path / "team" / "review", "review", "External.")
    local = write_skill(workspace / "skills" / "review", "review", "Workspace.")
    reader.refuse(local, "invalid_frontmatter")
    skills: dict[str, Any] = {"directories": [{"skill": str(team)}]}
    if policy is not None:
        skills["duplicates"] = policy

    plan = _plan(tmp_path, workspace, skills)

    report = plan.report()
    assert report is not None
    assert report["loaded"][0]["path"] == str(team.resolve())
    assert "duplicates" not in report
    openclaw_skills.apply_status(
        plan, {"skills": [{"name": "review", "source": "openclaw-extra"}]}
    )
    assert plan.report() == report


def test_status_settles_winners_fabric_could_not_predict(
    tmp_path: Path, workspace: Path
):
    """OpenClaw's listed source decides a clash Fabric's own check got wrong."""
    team = write_skill(tmp_path / "team" / "review", "review", "External.")
    local = write_skill(workspace / "skills" / "review", "review", "Workspace.")
    project = write_skill(workspace / ".agents/skills/review", "review", "Project.")
    plan = _plan(tmp_path, workspace, {"directories": [{"skill": str(team)}]})
    report = plan.report()
    assert report is not None
    assert report["duplicates"][0]["outcome"] == "shadowed"
    assert report["duplicates"][0]["selected"] == str(local.resolve())

    # OpenClaw refused both workspace copies, for example for a file limit
    # Fabric does not check: the configured copy wins.
    openclaw_skills.apply_status(
        plan, {"skills": [{"name": "review", "source": "openclaw-extra"}]}
    )

    report = plan.report()
    assert report is not None
    assert report["loaded"] == [loaded("review", team, "skills.directories[0]")]
    assert "duplicates" not in report

    # The next listing shows the project copy winning instead.
    openclaw_skills.apply_status(
        plan, {"skills": [{"name": "review", "source": "agents-skills-project"}]}
    )

    report = plan.report()
    assert report is not None
    assert "loaded" not in report
    assert report["duplicates"] == [
        duplicate(
            "review",
            "shadowed",
            project,
            copy("external", team, "skills.directories[0]"),
            copy("local", project),
        )
    ]


@pytest.mark.parametrize("policy", [None, "reject", "first_wins", "last_wins"])
def test_copies_openclaw_refuses_never_count_as_duplicates(
    tmp_path: Path, workspace: Path, reader: FakeReader, policy: str | None
):
    """Refused copies are skipped before the policy applies, so start succeeds."""
    scenario = refused_copies(tmp_path)
    for folder, _, reason in scenario.refused:
        reader.refuse(folder, reason)
    skills: dict[str, Any] = {"directories": scenario.directories}
    if policy is not None:
        skills["duplicates"] = policy

    generated, plan = _generated(tmp_path, workspace, skills)

    if policy is None:
        # OpenClaw gets every entry and refuses the same copies itself.
        assert generated["skills"]["load"]["extraDirs"] == [
            *(str(path) for path in scenario.review),
            str(scenario.team),
        ]
    else:
        assert generated["skills"]["load"]["extraDirs"] == [
            str(scenario.review[1].resolve()),
            str(scenario.triage.resolve()),
        ]
    expected = scenario.report()
    assert plan.report() == expected
    # OpenClaw's listing agrees, so nothing is reported as rejected by it.
    listed = {
        "skills": [
            {"name": name, "source": "openclaw-extra"} for name in ("review", "triage")
        ]
    }
    assert openclaw_skills.apply_status(plan, listed, fallback=True) is False
    assert plan.report() == expected


def test_one_reader_call_reads_every_source(
    tmp_path: Path, workspace: Path, package: Path, reader: FakeReader
):
    """Configured, shipped, and workspace skills are read in one call."""
    write_skill(tmp_path / "team" / "review", "review")
    write_skill(workspace / "skills" / "local", "local")
    write_skill(workspace / ".agents" / "skills" / "project", "project")

    _plan(
        tmp_path,
        workspace,
        {"directories": [{"folder": str(tmp_path / "team")}], "bundled": {}},
        package,
    )

    assert [sorted(folder.name for folder in call) for call in reader.calls] == [
        [
            "diagnose-gateway",
            "github",
            "local",
            "project",
            "review",
            "tmux",
            "weather",
        ]
    ]


def _without_node(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Plan with the real reader where node cannot be found."""
    monkeypatch.setattr(
        openclaw_skills,
        "candidate_reader",
        lambda command, *, env, timeout: openclaw_skills.OpenClawReader(
            None, env={"PATH": str(tmp_path / "no-node")}, timeout=timeout
        ),
    )


def test_reader_without_node_reads_simple_front_matter_and_warns_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Names come from simple front matter, and whether OpenClaw loads them is unknown."""
    named = write_skill(tmp_path / "named", "review").resolve()
    padded = write_skill(tmp_path / "  padded").resolve()
    undescribed = write_skill(tmp_path / "undescribed", "x", None).resolve()
    large = write_skill(tmp_path / "large", "large", "x" * 256_001).resolve()
    missing = (tmp_path / "missing").resolve()
    reader = openclaw_skills.OpenClawReader(
        None, env={"PATH": str(tmp_path / "no-node")}, timeout=5
    )

    infos = reader([named, padded, undescribed, large, missing])
    again = reader([named])

    assert infos == {
        named: CandidateInfo("review", valid=None),
        # OpenClaw trims the folder name.
        padded: CandidateInfo("padded", valid=None),
        undescribed: CandidateInfo("x", valid=None),
        # Refusals that need no parser stay certain.
        large: CandidateInfo(None, valid=False, reason="file_too_large"),
        missing: CandidateInfo(None, valid=False, reason="unreadable"),
    }
    assert again == {named: infos[named]}
    assert capsys.readouterr().err.splitlines() == [
        "OpenClaw: NeMo Fabric cannot run OpenClaw's front matter parser (node was "
        "not found), so it cannot tell which skills OpenClaw refuses"
    ]


def _stand_in_package(package: Path, parser: str) -> None:
    """Give the package hashed front matter modules; one exports ``parser``."""
    dist = package / "dist"
    dist.mkdir()
    (dist / "frontmatter-A1b2.mjs").write_text(
        "function resolveHookKey() {}\nexport { resolveHookKey as n };\n",
        encoding="utf-8",
    )
    (dist / "frontmatter-Z9y8.mjs").write_text(
        f"{parser}\nexport {{ resolveSkillKey as r, parseSkillFrontmatter as t }};\n"
        "function resolveSkillKey() {}\n",
        encoding="utf-8",
    )


@pytest.mark.skipif(sys.platform == "win32", reason="Requires a POSIX shell script")
@pytest.mark.parametrize(
    ("script", "problem"),
    [
        ("echo broken >&2; exit 3", "node failed: broken"),
        ("exec /bin/sleep 5", "it did not finish within 0.5 seconds"),
        ("echo '[]'", "node returned an unexpected result"),
    ],
    ids=["fails", "times-out", "wrong-output"],
)
def test_reader_falls_back_when_node_fails(
    tmp_path: Path,
    package: Path,
    capsys: pytest.CaptureFixture[str],
    script: str,
    problem: str,
):
    """A node call that fails or passes the deadline leaves validity unknown."""
    _stand_in_package(package, "function parseSkillFrontmatter() {}")
    node = tmp_path / "bin" / "node"
    node.parent.mkdir()
    node.write_text(f"#!/bin/sh\n{script}\n", encoding="utf-8")
    node.chmod(0o755)
    skill = write_skill(tmp_path / "review", "review").resolve()
    reader = openclaw_skills.OpenClawReader(
        package, env={"PATH": str(node.parent)}, timeout=0.5
    )

    assert reader([skill]) == {skill: CandidateInfo("review", valid=None)}
    assert f"parser ({problem})" in capsys.readouterr().err


@pytest.mark.skipif(shutil.which("node") is None, reason="Requires node")
def test_reader_runs_the_packages_parser_through_node(tmp_path: Path, package: Path):
    """The parser is found by its export; the loader's checks apply to its result."""
    _stand_in_package(
        package,
        """
function parseSkillFrontmatter(raw) {
  if (raw.includes("[unclosed")) throw new Error("invalid frontmatter");
  const fields = {};
  for (const line of raw.split("\\n")) {
    const match = line.match(/^(\\w+): ?(.*)$/);
    if (match) fields[match[1]] = match[2].replace(/^'(.*)'$/, "$1");
  }
  return fields;
}""",
    )
    skills = {
        "named": write_skill(tmp_path / "named", "'  spaced name '"),
        "unnamed": write_skill(tmp_path / "  padded"),
        "plain": write_skill(tmp_path / "plain", ""),
        "malformed": refused_skill(tmp_path / "malformed", "x", "malformed"),
        "empty": write_skill(tmp_path / "empty", "x", "'  '"),
        "undescribed": write_skill(tmp_path / "undescribed", "x", None),
        "large": write_skill(tmp_path / "large", "x", "x" * 256_001),
    }
    folders = {key: path.resolve() for key, path in skills.items()}
    reader = openclaw_skills.OpenClawReader(package, env=dict(os.environ), timeout=30)

    infos = reader(list(folders.values()))

    assert {key: infos[folder] for key, folder in folders.items()} == {
        "named": CandidateInfo("spaced name"),
        "unnamed": CandidateInfo("padded"),
        # The helper falls back to the folder name as discovery reached it.
        "plain": CandidateInfo(None),
        "malformed": CandidateInfo(None, valid=False, reason="invalid_frontmatter"),
        "empty": CandidateInfo(
            None, valid=False, reason="empty_frontmatter_field:description"
        ),
        "undescribed": CandidateInfo(
            None, valid=False, reason="missing_frontmatter_field:description"
        ),
        "large": CandidateInfo(None, valid=False, reason="file_too_large"),
    }


def test_policy_never_decides_on_a_guess_without_node(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
):
    """Without OpenClaw's parser, skills load, but no explicit policy picks a copy."""
    copies = [write_skill(tmp_path / name / "review", "review") for name in "ab"]
    entries = [{"skill": str(path)} for path in copies]
    _without_node(monkeypatch, tmp_path)

    assert _loaded(_plan(tmp_path, workspace, {"directories": entries})) == [
        ("review", "skills.directories[1]")
    ]
    for policy, deciding in (("reject", 0), ("first_wins", 0), ("last_wins", 1)):
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            _plan(tmp_path, workspace, {"directories": entries, "duplicates": policy})
        assert caught.value.code == "openclaw_skill_invalid"
        assert caught.value.metadata == {
            "field": f"skills.directories[{deciding}]",
            "reason": "unsupported_frontmatter",
            "path": str(copies[deciding].resolve()),
        }


def test_uncertain_name_fails_under_a_policy_without_node(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
):
    """A name the simple reader cannot read never silently picks a copy."""
    team = tmp_path / "team"
    unsupported = team / "unsupported"
    unsupported.mkdir(parents=True)
    (unsupported / "SKILL.md").write_text(
        "---\n? name\n: review\ndescription: d\n---\n", encoding="utf-8"
    )
    write_skill(team / "review", "review")
    _without_node(monkeypatch, tmp_path)

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        _plan(
            tmp_path,
            workspace,
            {"directories": [{"folder": str(team)}], "duplicates": "reject"},
        )

    assert caught.value.code == "openclaw_skill_invalid"
    assert caught.value.metadata == {
        "field": "skills.directories[0]",
        "reason": "unsupported_frontmatter_name",
        "path": str(unsupported.resolve()),
    }


def test_warn_writes_skipped_and_unavailable_once(
    tmp_path: Path,
    workspace: Path,
    package: Path,
    capsys: pytest.CaptureFixture[str],
):
    """Start warnings list each problem in one stderr line."""
    empty = tmp_path / "empty"
    empty.mkdir()
    _, plan = _generated(
        tmp_path,
        workspace,
        {"directories": [{"folder": str(empty)}], "bundled": {"include": ["github"]}},
        package,
    )
    plan.bundled_unavailable = [{"name": "github", "reason": "missing_requirements"}]

    plan.warn()

    assert capsys.readouterr().err.splitlines() == [
        "OpenClaw: skills.directories[0] contains no skills",
        "OpenClaw: 1 selected bundled skills are unavailable: github "
        "(missing_requirements)",
    ]


async def test_skill_status_uses_openclaw_list_for_the_agent(
    tmp_path: Path, workspace: Path, package: Path, monkeypatch: pytest.MonkeyPatch
):
    """The status command lists the agent's skills; a failure keeps the report."""
    _, plan = _generated(
        tmp_path, workspace, {"bundled": {"include": ["github"]}}, package
    )
    status = AsyncMock(
        return_value=json.dumps(
            {
                "skills": [
                    {
                        "name": "github",
                        "bundled": True,
                        "source": "openclaw-bundled",
                        "eligible": False,
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(adapter, "_command_output", status)

    await adapter._apply_skill_status(
        plan, tmp_path / "openclaw", agent_id="Reviewer", env={"A": "1"}, timeout=5
    )

    assert status.await_args.args[1:] == (
        "skills",
        "list",
        "--json",
        "--agent",
        "Reviewer",
    )
    assert status.await_args.kwargs["env"] == {"A": "1"}
    assert plan.bundled_unavailable == [
        {"name": "github", "reason": "missing_requirements"}
    ]

    plan.bundled_unavailable = []
    monkeypatch.setattr(
        adapter,
        "_command_output",
        AsyncMock(side_effect=adapter.lifecycle.LifecycleError("x", "failed")),
    )
    await adapter._apply_skill_status(
        plan, tmp_path / "openclaw", agent_id="default", env={}, timeout=5
    )
    assert plan.bundled_unavailable == []
    assert plan.report() == {"bundled": {"selected": ["github"]}}


def _reporting_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, report: dict[str, Any]
) -> adapter.OpenClawRuntime:
    runtime = adapter.OpenClawRuntime()
    process = MagicMock()
    process.returncode = None
    runtime._client = MagicMock()
    runtime._config = agent_config({"directories": []})
    runtime._context = runtime_context(tmp_path)
    runtime._port = 12345
    runtime._process = process
    runtime._skills_report = report
    monkeypatch.setattr(
        adapter,
        "_invoke_gateway",
        AsyncMock(return_value=("OpenClaw response", adapter.contract.AgentUsage())),
    )
    return runtime


async def test_every_run_result_reports_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Succeeded and failed results carry the skills block next to mcp."""
    report = {
        "loaded": [{"name": "kept", "path": "/x", "entry": "e", "origin": "external"}]
    }
    runtime = _reporting_runtime(tmp_path, monkeypatch, report)
    context = runtime._context
    assert context is not None

    result = await runtime.invoke(AgentRunRequest(input="Hello."), context)
    assert result.output == {"response": "OpenClaw response", "skills": report}
    result.output["skills"]["loaded"].clear()
    assert runtime._skills_report == report and report["loaded"]

    request = adapter.httpx.Request(
        "POST", "http://127.0.0.1:12345/v1/chat/completions"
    )
    monkeypatch.setattr(
        adapter,
        "_invoke_gateway",
        AsyncMock(
            side_effect=adapter.httpx.HTTPStatusError(
                "bad", request=request, response=adapter.httpx.Response(502)
            )
        ),
    )
    result = await runtime.invoke(AgentRunRequest(input="Hello."), context)
    assert result.output == {"skills": report}

    monkeypatch.setattr(adapter, "_invoke_gateway", AsyncMock(side_effect=TimeoutError))
    result = await runtime.invoke(AgentRunRequest(input="Hello."), context)
    assert result.output == {"skills": report}


def test_attach_refuses_structured_skills(tmp_path: Path):
    """attach_service cannot verify skills the deployment owns."""
    for skills in ({"directories": [{"folder": "x"}]}, {"bundled": {}}):
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            adapter._validate_attach_config(agent_config(skills))
        assert caught.value.metadata == {"fields": ["skills"]}
    adapter._validate_attach_config(agent_config({"directories": []}))


def test_descriptor_advertises_implemented_skill_support(repo_root: Path):
    """The descriptor's skills block matches what the adapter implements."""
    descriptor = json.loads(
        (repo_root / "adapters/python/openclaw/openclaw.fabric-adapter.json").read_text(
            encoding="utf-8"
        )
    )

    assert "skills" in descriptor["config"]["accepts"]
    skills = descriptor["skills"]
    assert skills["directory_loading"] == {
        "supported_entries": ["skill", "folder"],
        "skill_file": "SKILL.md",
        "discovery": {
            "traversal": "recursive",
            "nested_skills": "stop_at_skill",
            "depth_limit": {"kind": "adapter_defined"},
        },
        "parsing": {
            "required_frontmatter_fields": ["description"],
            "name_fallback": "directory_name",
            "frontmatter_errors": "reject",
        },
    }
    assert skills["bundled"] == {
        "default": "enabled",
        "supports_enable": True,
        "supports_disable": True,
        "supports_include": True,
        "supports_exclude": True,
    }
    assert skills["duplicates"]["supported_policies"] == [
        "reject",
        "first_wins",
        "last_wins",
    ]
    assert skills["duplicates"]["name_source"] == "frontmatter"


@pytest.mark.skipif(sys.platform == "win32", reason="Requires a POSIX shell script")
async def test_start_writes_the_plan_and_reports_it(
    tmp_path: Path,
    workspace: Path,
    package: Path,
    mock_openclaw: Path,
    reader: FakeReader,
    monkeypatch: pytest.MonkeyPatch,
):
    """start applies the plan, lists OpenClaw's skills, and keeps the report."""
    team = tmp_path / "team"
    write_skill(team / "kept", "kept")
    capture = tmp_path / "config.json"
    status = tmp_path / "status.json"
    status.write_text(
        json.dumps(
            {
                "skills": [
                    {"name": "kept", "source": "openclaw-extra", "eligible": True},
                    {
                        "name": "github",
                        "source": "openclaw-bundled",
                        "bundled": True,
                        "eligible": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    command = tmp_path / "openclaw"
    command.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = skills ]; then printf "%s\\n" "$*" > "$FAKE_SKILLS_ARGS"; '
        'printf "%s\\n" "$OPENCLAW_BUNDLED_SKILLS_DIR" > "$FAKE_SKILLS_STAGED"; '
        'ls "$OPENCLAW_BUNDLED_SKILLS_DIR" >> "$FAKE_SKILLS_STAGED"; '
        'exec cat "$FAKE_SKILLS_STATUS"; fi\n'
        f'exec "{sys.executable}" "{mock_openclaw}" "$@"\n',
        encoding="utf-8",
    )
    command.chmod(0o755)
    monkeypatch.setenv("FAKE_OPENCLAW_CAPTURE", str(capture))
    monkeypatch.setenv("FAKE_SKILLS_STATUS", str(status))
    monkeypatch.setenv("FAKE_SKILLS_ARGS", str(tmp_path / "args.txt"))
    monkeypatch.setenv("FAKE_SKILLS_STAGED", str(tmp_path / "staged.txt"))
    config = agent_config(
        {"directories": [{"folder": str(team)}], "bundled": {"include": ["github"]}},
        {"startup_timeout_seconds": 12},
        command,
    )
    runtime = adapter.OpenClawRuntime()

    await runtime.start(
        {
            "config": config,
            "runtime_context": runtime_context(
                workspace, {"OPENCLAW_BUNDLED_SKILLS_DIR": str(package / "skills")}
            ).to_mapping(),
            "base_dir": str(tmp_path),
        }
    )
    try:
        report = runtime._skills_report
    finally:
        await runtime.stop()

    # Skills were read with the configured command's package, within what is
    # left of the 12-second startup deadline.
    [(read_with, timeout)] = reader.requests
    assert read_with == command.resolve()
    assert 0 < timeout < 12 < openclaw_skills.READER_TIMEOUT_SECONDS
    generated = json.loads(capture.read_text(encoding="utf-8"))
    assert generated["skills"] == {"load": {"extraDirs": [str(team)]}}
    assert (tmp_path / "args.txt").read_text(encoding="utf-8").split() == [
        "skills",
        "list",
        "--json",
        "--agent",
        "default",
    ]
    # OpenClaw read the shipped skills from a per-runtime copy of the selection,
    # which stop removes with the rest of the runtime's state.
    staged, *listed = (tmp_path / "staged.txt").read_text(encoding="utf-8").split()
    assert listed == ["github"]
    assert Path(staged).name == "skills"
    assert not Path(staged).is_relative_to(package)
    assert not Path(staged).exists()
    assert report == {
        "loaded": [loaded("kept", team / "kept", "skills.directories[0]")],
        "bundled": {"selected": ["github"]},
    }
    assert runtime._skills_report is None


@pytest.fixture(name="mock_openclaw")
def mock_openclaw_fixture(repo_root: Path) -> Path:
    return repo_root / "tests/_utils/mock_openclaw.py"


def test_hidden_dir_names_cover_reachable_dot_folders(tmp_path: Path):
    """Folders whose names start with a dot are excluded wherever they appear."""
    (tmp_path / "a" / ".cache").mkdir(parents=True)
    (tmp_path / ".git").mkdir()

    assert openclaw_skills._hidden_dir_names([tmp_path]) == {".cache", ".git"}
    assert os.path.isdir(tmp_path / "a")


def test_hidden_dir_names_reach_the_deepest_skills_level(tmp_path: Path):
    """The entry's skills child is searched 6 levels below itself."""
    (tmp_path / "skills" / "a" / "b" / "c" / "d" / "e" / ".private").mkdir(parents=True)

    assert openclaw_skills._hidden_dir_names([tmp_path]) == {".private"}


@pytest.mark.skipif(sys.platform == "win32", reason="Requires symbolic links")
def test_hidden_dir_names_list_each_linked_folder_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Links back to a folder do not multiply the folders listed."""
    root = tmp_path / "loop"
    (root / ".cache").mkdir(parents=True)
    for index in range(6):
        (root / f"link{index}").symlink_to(root, target_is_directory=True)
    listed: list[str] = []
    scandir = os.scandir

    def counting_scandir(path: Any) -> Any:
        listed.append(os.fspath(path))
        return scandir(path)

    monkeypatch.setattr(openclaw_skills.os, "scandir", counting_scandir)

    assert openclaw_skills._hidden_dir_names([root]) == {".cache"}
    assert listed == [os.fspath(root)]


@pytest.mark.parametrize("policy", [None, "first_wins"])
def test_hidden_skill_at_the_deepest_skills_level_is_not_loaded(
    tmp_path: Path, workspace: Path, policy: str | None
):
    """OpenClaw skips dot folders at every level discovery reaches."""
    team = tmp_path / "team"
    top = write_skill(team / "top", "top")
    write_skill(team / "skills" / "a" / "b" / "c" / "d" / "e" / ".private", "private")
    skills: dict[str, Any] = {"directories": [{"folder": str(team)}]}
    if policy is not None:
        skills["duplicates"] = policy

    generated, plan = _generated(tmp_path, workspace, skills)

    assert _loaded(plan) == [("top", "skills.directories[0]")]
    assert generated["skills"]["load"]["extraDirs"] == (
        [str(team)] if policy is None else [str(top.resolve())]
    )


@pytest.mark.parametrize("policy", [None, "first_wins"])
def test_promoted_skills_folder_that_is_a_skill_loads_as_one(
    tmp_path: Path, workspace: Path, policy: str | None
):
    """OpenClaw scans skills/ when only it holds skills, and loads it as one skill."""
    repository = tmp_path / "repository"
    packed = write_skill(repository / "skills", "packed")
    write_skill(packed / "nested", "nested")
    skills: dict[str, Any] = {"directories": [{"folder": str(repository)}]}
    if policy is not None:
        skills["duplicates"] = policy

    generated, plan = _generated(tmp_path, workspace, skills)

    assert plan.report() == {
        "loaded": [loaded("packed", packed, "skills.directories[0]")]
    }
    assert generated["skills"]["load"]["extraDirs"] == (
        [str(repository)] if policy is None else [str(packed.resolve())]
    )


@pytest.mark.skipif(sys.platform == "win32", reason="Requires symbolic links")
@pytest.mark.parametrize("policy", [None, "first_wins"])
def test_skills_linked_outside_their_folder_are_skipped(
    tmp_path: Path, workspace: Path, policy: str | None
):
    """OpenClaw refuses skills outside the folder, so they are never passed or reported."""
    outside = tmp_path / "outside"
    write_skill(outside / "escaped", "escaped")
    write_skill(outside / "skills" / "nested-escaped", "nested-escaped")
    inside = write_skill(tmp_path / "team" / "inside", "inside")
    (tmp_path / "team" / "escaped").symlink_to(outside / "escaped")
    (tmp_path / "linked").mkdir()
    (tmp_path / "linked" / "skills").symlink_to(outside / "skills")
    skills: dict[str, Any] = {
        "directories": [
            {"folder": str(tmp_path / "team")},
            {"folder": str(tmp_path / "linked")},
        ]
    }
    if policy is not None:
        skills["duplicates"] = policy

    generated, plan = _generated(tmp_path, workspace, skills)

    assert generated["skills"]["load"]["extraDirs"] == (
        [str(tmp_path / "team"), str(tmp_path / "linked")]
        if policy is None
        else [str(inside.resolve())]
    )
    assert plan.report() == {
        "loaded": [loaded("inside", inside, "skills.directories[0]")],
        "empty_entries": ["skills.directories[1]"],
        "skipped": [
            skipped(outside / "escaped", "skills.directories[0]", "path_escape"),
            skipped(outside / "skills", "skills.directories[1]", "path_escape"),
        ],
    }


@pytest.mark.parametrize("policy", [None, "first_wins"])
def test_skills_child_of_a_scanned_root_is_searched_six_levels(
    tmp_path: Path, workspace: Path, policy: str | None
):
    """The root keeps 2 levels while its skills child is searched 6 levels deep."""
    team = tmp_path / "team"
    write_skill(team / "top", "top")
    deep = write_skill(team / "skills" / "g" / "h" / "deep", "deep")
    write_skill(team / "a" / "b" / "too-deep", "too-deep")
    skills: dict[str, Any] = {"directories": [{"folder": str(team)}]}
    if policy is not None:
        skills["duplicates"] = policy

    generated, plan = _generated(tmp_path, workspace, skills)

    assert _loaded(plan) == [
        ("deep", "skills.directories[0]"),
        ("top", "skills.directories[0]"),
    ]
    if policy is not None:
        assert str(deep.resolve()) in generated["skills"]["load"]["extraDirs"]


def test_overlapping_folder_entries_use_each_entry_depth(
    tmp_path: Path, workspace: Path
):
    """A skill too deep for one entry still loads through a deeper-searched entry."""
    write_skill(tmp_path / "a" / "top", "top")
    x = write_skill(tmp_path / "a" / "b" / "skills" / "x", "x")

    generated, plan = _generated(
        tmp_path,
        workspace,
        {
            "directories": [
                {"folder": str(tmp_path / "a")},
                {"folder": str(tmp_path / "a" / "b" / "skills")},
            ],
            "duplicates": "first_wins",
        },
    )

    assert _loaded(plan) == [
        ("top", "skills.directories[0]"),
        ("x", "skills.directories[1]"),
    ]
    assert generated["skills"]["load"]["extraDirs"][-1] == str(x.resolve())


def test_entry_errors_follow_entry_order(tmp_path: Path, workspace: Path):
    """A later folder_is_skill entry does not hide an earlier missing entry."""
    write_skill(tmp_path / "is-skill", "is-skill")

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        _plan(
            tmp_path,
            workspace,
            {
                "directories": [
                    {"folder": str(tmp_path / "missing")},
                    {"folder": str(tmp_path / "is-skill")},
                ]
            },
        )

    assert caught.value.code == "openclaw_skill_not_found"
    assert caught.value.metadata["field"] == "skills.directories[0].folder"


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="Requires POSIX permissions that apply to the user",
)
def test_untraversable_folder_is_invalid_on_older_python(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
):
    """A folder under an untraversable parent fails as invalid, not with an exception."""

    def strict_is_dir(self: Path, *, follow_symlinks: bool = True) -> bool:
        # Path.is_dir() before Python 3.14: only a missing path returns False.
        try:
            return (self.stat().st_mode & 0o170000) == 0o040000
        except (FileNotFoundError, NotADirectoryError):
            return False

    locked = tmp_path / "locked"
    (locked / "team").mkdir(parents=True)
    locked.chmod(0)
    monkeypatch.setattr(Path, "is_dir", strict_is_dir)
    try:
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            _plan(
                tmp_path,
                workspace,
                {"directories": [{"folder": str(locked / "team")}]},
            )
    finally:
        locked.chmod(0o755)

    assert caught.value.code == "openclaw_skill_invalid"
    assert caught.value.metadata["field"] == "skills.directories[0].folder"


@pytest.mark.parametrize(
    ("policy", "refused", "kept"), [("first_wins", 0, 1), ("last_wins", 2, 1)]
)
def test_refused_winner_falls_back_to_the_next_copy(
    tmp_path: Path, workspace: Path, policy: str, refused: int, kept: int
):
    """A winner OpenClaw refuses is replaced by the next copy in policy order."""
    copies = [
        write_skill(tmp_path / f"copy-{index}" / "review", "review")
        for index in range(3)
    ]
    other = write_skill(tmp_path / "copy-1" / "other", "other")
    _, plan = _generated(
        tmp_path,
        workspace,
        {
            "directories": [{"folder": str(copy.parent)} for copy in copies],
            "duplicates": policy,
        },
    )
    assert str(copies[refused].resolve()) in plan.extra_dirs
    status = {"skills": [{"name": "other", "source": "openclaw-extra"}]}

    assert openclaw_skills.apply_status(plan, status, fallback=True) is True

    assert sorted(plan.extra_dirs) == sorted(
        [str(copies[kept].resolve()), str(other.resolve())]
    )
    report = plan.report()
    assert report is not None
    assert {item["name"]: item["path"] for item in report["loaded"]} == {
        "review": str(copies[kept].resolve()),
        "other": str(other.resolve()),
    }
    assert report["duplicates"][0]["selected"] == str(copies[kept].resolve())
    assert report["skipped"] == [
        skipped(
            copies[refused], f"skills.directories[{refused}]", "rejected_by_openclaw"
        )
    ]
    # The next listing shows the replacement, so nothing changes again.
    listed = {
        "skills": [*status["skills"], {"name": "review", "source": "openclaw-extra"}]
    }
    assert openclaw_skills.apply_status(plan, listed, fallback=True) is False
    assert plan.report() == report


def test_refused_winner_without_fallback_loads_no_copy(tmp_path: Path, workspace: Path):
    """Without a fallback the report says no copy of the name loads."""
    first = write_skill(tmp_path / "a" / "review", "review")
    write_skill(tmp_path / "b" / "review", "review")
    _, plan = _generated(
        tmp_path,
        workspace,
        {
            "directories": [
                {"folder": str(tmp_path / "a")},
                {"folder": str(tmp_path / "b")},
            ],
            "duplicates": "first_wins",
        },
    )

    assert openclaw_skills.apply_status(plan, {"skills": []}) is False

    report = plan.report()
    assert report is not None
    assert "loaded" not in report
    assert "selected" not in report["duplicates"][0]
    assert report["skipped"][0]["path"] == str(first.resolve())
    assert plan.extra_dirs == [str(first.resolve())]


async def test_skill_status_lists_again_after_a_fallback(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
):
    """start rewrites the configuration and lists again after replacing a winner."""
    write_skill(tmp_path / "a" / "review", "review")
    second = write_skill(tmp_path / "b" / "review", "review")
    _, plan = _generated(
        tmp_path,
        workspace,
        {
            "directories": [
                {"folder": str(tmp_path / "a")},
                {"folder": str(tmp_path / "b")},
            ],
            "duplicates": "first_wins",
        },
    )
    listings = AsyncMock(
        side_effect=[
            json.dumps({"skills": []}),
            json.dumps({"skills": [{"name": "review", "source": "openclaw-extra"}]}),
        ]
    )
    monkeypatch.setattr(adapter, "_command_output", listings)
    written: list[list[str]] = []

    await adapter._apply_skill_status(
        plan,
        tmp_path / "openclaw",
        agent_id="default",
        env={},
        timeout=5,
        rewrite=lambda: written.append(list(plan.extra_dirs)),
    )

    assert listings.await_count == 2
    assert written == [[str(second.resolve())]]
    assert _loaded(plan) == [("review", "skills.directories[1]")]


@pytest.mark.parametrize(
    ("agents", "owner"),
    [
        ({"entries": {"default": {}}}, True),
        ({"entries": {"Reviewer": {}}}, True),
        ({"entries": {"default": {}, "other": {}}}, False),
        ({"entries": {"default": {}, "other": {"default": True}}}, False),
        ({"entries": {"default": {"default": True}, "other": {}}}, True),
        (
            {
                "ownership": "explicit",
                "entries": {"default": {"default": True}, "other": {}},
            },
            False,
        ),
        (
            {
                "defaults": {"systemAgent": {"agentId": "other"}},
                "entries": {"default": {}, "other": {}},
            },
            False,
        ),
        (
            {
                "defaults": {"systemAgent": {"agentId": " DEFAULT "}},
                "entries": {"default": {}, "other": {}},
            },
            True,
        ),
    ],
)
def test_custodian_owner_follows_openclaw(agents: dict[str, Any], owner: bool):
    """Custodian skills belong to the system agent, a default marker, or the only agent."""
    agent_id = next(iter(agents["entries"]))
    assert openclaw_skills.owns_custodian_skills(agents, agent_id) is owner


def test_custodian_skills_hide_configured_ones_only_for_their_owner(
    tmp_path: Path, workspace: Path, package: Path
):
    """For another agent a configured skill named like a Custodian skill loads."""
    mine = write_skill(tmp_path / "team" / "diagnose-gateway", "diagnose-gateway")
    skills = {
        "directories": [{"folder": str(tmp_path / "team")}],
        "bundled": {"include": ["diagnose-gateway", "github"]},
    }
    settings = {
        "agents": {
            "ownership": "explicit",
            "entries": {"other": {"contextInjection": "never"}},
            "defaults": {"systemAgent": {"agentId": "other"}},
        },
    }
    plan = adapter._skill_plan(
        agent_config(skills, settings),
        runtime_context(workspace),
        base_dir=tmp_path,
        command=package / "openclaw.mjs",
        env={},
    )

    assert plan.custodian is False
    assert plan.report() == {
        "loaded": [loaded("diagnose-gateway", mine, "skills.directories[0]")],
        "bundled": {
            "selected": ["github"],
            "unavailable": [
                {"name": "diagnose-gateway", "reason": "custodian_agent_only"}
            ],
        },
    }

    owned = _plan(tmp_path, workspace, skills, package)
    assert owned.custodian is True
    report = owned.report()
    assert report is not None
    assert report["duplicates"][0]["outcome"] == "shadowed"
    assert report["bundled"] == {"selected": ["diagnose-gateway", "github"]}


async def test_connected_runtime_reports_the_prepared_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A runtime that connects to a prepared Gateway carries its skill report."""
    report = {
        "loaded": [{"name": "kept", "path": "/x", "entry": "e", "origin": "external"}]
    }
    connection = tmp_path / "private" / "connection.json"
    adapter._write_service_connection(
        connection,
        gateway_url="http://127.0.0.1:18789",
        token="TEST_ONLY_GATEWAY_TOKEN",
        agent_id="default",
        skills=report,
    )
    runtime = adapter.OpenClawRuntime()
    monkeypatch.setattr(runtime, "_probe_gateway", AsyncMock(return_value="2026.9.34"))

    await runtime._connect_service_runtime(
        agent_config({"directories": []}),
        runtime_context(tmp_path),
        {},
        {"operation": "connect", "connection_path": str(connection)},
    )
    try:
        assert runtime._skills_output() == {"skills": report}
    finally:
        await runtime.stop()
