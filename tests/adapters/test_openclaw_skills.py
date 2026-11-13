# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured skill configuration for the OpenClaw adapter."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.openclaw import adapter
from nemo_fabric_adapters.openclaw import skills as openclaw_skills


def _skill(
    directory: Path, name: str | None = None, description: str | None = "Use it."
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["---"]
    if name is not None:
        lines.append(f"name: {name}")
    if description is not None:
        lines.append(f"description: {description}")
    lines += ["---", "", "Body.", ""]
    (directory / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")
    return directory


def _context(workspace: Path, env: dict[str, str] | None = None) -> RuntimeContext:
    return RuntimeContext.from_mapping(
        {
            "runtime_id": "openclaw-runtime",
            "invocation_id": "openclaw-invocation",
            "request_id": "openclaw-request",
            "environment": {
                "environment_id": "openclaw-environment",
                "provider": "local",
                "control_location": "in_env_control",
                "workspace": str(workspace),
                "env": env or {},
                "ownership": "caller_owned",
            },
            "artifacts": {},
        }
    )


def _config(skills: dict[str, Any], command: Path | None = None) -> AgentConfig:
    payload: dict[str, Any] = {
        "models": {"default": {"provider": "test", "model": "fabric-echo"}},
        "skills": skills,
    }
    if command is not None:
        payload["harness"] = {"settings": {"openclaw_command": str(command)}}
    return AgentConfig.from_mapping(payload)


@pytest.fixture(name="package")
def package_fixture(tmp_path: Path) -> Path:
    """A minimal installed OpenClaw package with shipped and Custodian skills."""

    root = tmp_path / "node_modules/openclaw"
    root.mkdir(parents=True)
    (root / "package.json").write_text('{"name": "openclaw"}', encoding="utf-8")
    (root / "openclaw.mjs").write_text("", encoding="utf-8")
    for name in ("github", "weather", "tmux"):
        _skill(root / "skills" / name, name)
    (root / "skills/pyproject.toml").write_text("", encoding="utf-8")
    _skill(root / "custodian-skills/diagnose-gateway", "diagnose-gateway")
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
        _config(skills),
        _context(workspace),
        base_dir=tmp_path,
        command=package / "openclaw.mjs" if package is not None else None,
        env={},
    )


def _generated(
    tmp_path: Path, workspace: Path, skills: dict[str, Any], package: Path | None = None
) -> tuple[dict[str, Any], openclaw_skills.SkillPlan]:
    plan = _plan(tmp_path, workspace, skills, package)
    generated = adapter._openclaw_config(
        _config(skills),
        _context(workspace),
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
    _skill(team / "debugging", "debugging")
    _skill(team / "group" / "code-review", "code-review")
    _skill(team / "group" / "deep" / "too-deep", "too-deep")
    _skill(team / "debugging" / "nested", "nested")
    _skill(team / ".hidden" / "secret", "secret")
    _skill(team / "node_modules" / "dependency", "dependency")
    _skill(team / "undescribed", "undescribed", description=None)
    _skill(team / "unnamed")
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
        {
            "path": str((team / "undescribed").resolve()),
            "entry": "skills.directories[0]",
            "reason": "missing_frontmatter_field:description",
        }
    ]
    assert report["loaded"][0] == {
        "name": "debugging",
        "path": str((team / "debugging").resolve()),
        "entry": "skills.directories[0]",
        "origin": "external",
    }


def test_folder_named_skills_is_searched_six_levels_deep(
    tmp_path: Path, workspace: Path
):
    """OpenClaw searches a folder named skills, or its skills subfolder, more deeply."""
    named = tmp_path / "skills"
    _skill(named / "a" / "b" / "c" / "d" / "e" / "six", "six")
    _skill(named / "a" / "b" / "c" / "d" / "e" / "f" / "seven", "seven")
    repository = tmp_path / "repository"
    _skill(repository / "skills" / "x" / "y" / "z" / "nested-root", "nested-root")

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
    _skill(named / "a" / "b" / "deep", "deep")
    team = tmp_path / "team"
    _skill(team / "a" / "b" / "too-deep", "too-deep")
    _skill(team / "a" / "shallow", "shallow")

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
    review = _skill(tmp_path / "opt/release-review", "release-review")
    setup = _skill(tmp_path / "opt/project-setup")

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
    _skill(team / "debugging", "debugging")
    project = tmp_path / "project-skills"
    _skill(project / "integration-testing", "integration-testing")
    review = _skill(tmp_path / "opt/release-review", "release-review")

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


def test_native_duplicates_pass_every_copy_and_report_the_later_entry(
    tmp_path: Path, workspace: Path
):
    """Without a policy OpenClaw keeps the copy from the later root."""
    first = _skill(tmp_path / "team/review", "review")
    second = _skill(tmp_path / "opt/review", "review")

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
    assert report["loaded"] == [
        {
            "name": "review",
            "path": str(second.resolve()),
            "entry": "skills.directories[1]",
            "origin": "external",
        }
    ]
    assert report["duplicates"] == [
        {
            "name": "review",
            "outcome": "selected",
            "selected": str(second.resolve()),
            "copies": [
                {
                    "origin": "external",
                    "path": str(first.resolve()),
                    "entry": "skills.directories[0]",
                },
                {
                    "origin": "external",
                    "path": str(second.resolve()),
                    "entry": "skills.directories[1]",
                },
            ],
        }
    ]


def test_native_duplicates_inside_one_folder_are_ambiguous(
    tmp_path: Path, workspace: Path
):
    """Two copies in one folder resolve in OpenClaw's path order."""
    _skill(tmp_path / "team/x/review", "review")
    _skill(tmp_path / "team/y/review", "review")

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
        "first": _skill(team / "review", "review"),
        "second": _skill(tmp_path / "opt/review", "review"),
    }
    other = _skill(team / "debugging", "debugging")

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
    first = _skill(tmp_path / "team/review", "review")
    second = _skill(tmp_path / "opt/review", "review")

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
    _skill(tmp_path / "team/x/review", "review")
    _skill(tmp_path / "team/y/review", "review")

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
    good = _skill(tmp_path / "good", "good")
    path = tmp_path / "entry"
    if setup == "file":
        path.write_text("not a folder", encoding="utf-8")
    elif setup == "no_manifest":
        path.mkdir()
    elif setup == "is_skill":
        _skill(path, "entry")

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
    _skill(tmp_path / "team/debugging", "debugging")

    generated, _ = _generated(
        tmp_path, workspace, {"directories": [{"folder": "team"}]}
    )

    assert generated["skills"]["load"]["extraDirs"] == [str(tmp_path / "team")]


def test_paths_behavior_is_unchanged(tmp_path: Path, workspace: Path, package: Path):
    """Legacy paths still pass resolved folders and combine with bundled selection."""
    skill = _skill(tmp_path / "skills", "legacy")

    generated, plan = _generated(
        tmp_path,
        workspace,
        {"paths": ["skills"], "bundled": {"include": ["weather"]}},
        package,
    )

    assert generated["skills"] == {
        "load": {"extraDirs": [str(skill.resolve())]},
        "allowBundled": ["weather"],
    }
    assert plan.report() == {"bundled": {"selected": ["weather"]}}
    generated, plan = _generated(tmp_path, workspace, {"paths": ["skills"]})
    assert generated["skills"] == {"load": {"extraDirs": [str(skill.resolve())]}}
    assert plan.report() is None


@pytest.mark.parametrize(
    ("bundled", "allow", "selected"),
    [
        ({}, None, ["diagnose-gateway", "github", "tmux", "weather"]),
        ({"exclude": []}, None, ["diagnose-gateway", "github", "tmux", "weather"]),
        ({"enabled": False}, [openclaw_skills.NO_BUNDLED_SKILLS], []),
        ({"include": []}, [openclaw_skills.NO_BUNDLED_SKILLS], []),
        (
            {"include": ["weather", "github"]},
            ["github", "weather"],
            ["github", "weather"],
        ),
        (
            {"exclude": ["tmux"]},
            ["diagnose-gateway", "github", "weather"],
            ["diagnose-gateway", "github", "weather"],
        ),
        (
            {"exclude": ["diagnose-gateway", "github", "tmux", "weather"]},
            [openclaw_skills.NO_BUNDLED_SKILLS],
            [],
        ),
    ],
)
def test_bundled_selection_uses_allow_bundled(
    tmp_path: Path,
    workspace: Path,
    package: Path,
    bundled: dict[str, Any],
    allow: list[str] | None,
    selected: list[str],
):
    """Bundled selection maps onto skills.allowBundled with empty-list semantics."""
    generated, plan = _generated(tmp_path, workspace, {"bundled": bundled}, package)

    assert generated.get("skills", {}).get("allowBundled") == allow
    assert plan.report() == {"bundled": {"selected": selected}}


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
            _config({"bundled": {"enabled": False}}),
            _context(workspace),
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
    """Workspace and shipped skills outrank skills.directories."""
    team = tmp_path / "team"
    mine = _skill(team / "review", "review")
    github = _skill(team / "github", "github")
    weather = _skill(team / "weather", "weather")
    kept = _skill(team / "kept", "kept")
    local = _skill(workspace / "skills" / "review", "review")

    _, plan = _generated(
        tmp_path,
        workspace,
        {"directories": [{"folder": str(team)}], "bundled": {"exclude": ["weather"]}},
        package,
    )

    report = plan.report()
    assert report is not None
    assert [item["name"] for item in report["loaded"]] == ["kept"]
    assert report["loaded"][0]["path"] == str(kept.resolve())
    duplicates = {item["name"]: item for item in report["duplicates"]}
    assert duplicates["review"] == {
        "name": "review",
        "outcome": "shadowed",
        "selected": str(local.resolve()),
        "copies": [
            {
                "origin": "external",
                "path": str(mine.resolve()),
                "entry": "skills.directories[0]",
            },
            {"origin": "local", "path": str(local.resolve())},
        ],
    }
    assert duplicates["github"]["selected"] == str(
        (package / "skills/github").resolve()
    )
    assert duplicates["github"]["copies"][0]["path"] == str(github.resolve())
    # An excluded shipped skill still hides the configured copy, so neither loads.
    assert duplicates["weather"]["outcome"] == "shadowed"
    assert "selected" not in duplicates["weather"]
    assert duplicates["weather"]["copies"][0]["path"] == str(weather.resolve())


def test_status_reports_rejected_and_unavailable_skills(
    tmp_path: Path, workspace: Path, package: Path
):
    """OpenClaw's own list refines the report."""
    team = tmp_path / "team"
    _skill(team / "kept", "kept")
    rejected = _skill(team / "rejected", "rejected")
    _skill(workspace / "skills" / "tmux", "tmux")
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
        {
            "path": str(rejected.resolve()),
            "entry": "skills.directories[0]",
            "reason": "rejected_by_openclaw",
        }
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
    runtime._config = _config({"directories": []})
    runtime._context = _context(tmp_path)
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
            adapter._validate_attach_config(_config(skills))
        assert caught.value.metadata == {"fields": ["skills"]}
    adapter._validate_attach_config(_config({"directories": []}))


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
    monkeypatch: pytest.MonkeyPatch,
):
    """start applies the plan, lists OpenClaw's skills, and keeps the report."""
    team = tmp_path / "team"
    _skill(team / "kept", "kept")
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
        'exec cat "$FAKE_SKILLS_STATUS"; fi\n'
        f'exec "{sys.executable}" "{mock_openclaw}" "$@"\n',
        encoding="utf-8",
    )
    command.chmod(0o755)
    monkeypatch.setenv("FAKE_OPENCLAW_CAPTURE", str(capture))
    monkeypatch.setenv("FAKE_SKILLS_STATUS", str(status))
    monkeypatch.setenv("FAKE_SKILLS_ARGS", str(tmp_path / "args.txt"))
    config = _config(
        {"directories": [{"folder": str(team)}], "bundled": {"include": ["github"]}},
        command,
    )
    runtime = adapter.OpenClawRuntime()

    await runtime.start(
        {
            "config": config,
            "runtime_context": _context(
                workspace, {"OPENCLAW_BUNDLED_SKILLS_DIR": str(package / "skills")}
            ).to_mapping(),
            "base_dir": str(tmp_path),
        }
    )
    try:
        report = runtime._skills_report
    finally:
        await runtime.stop()

    generated = json.loads(capture.read_text(encoding="utf-8"))
    assert generated["skills"] == {
        "load": {"extraDirs": [str(team)]},
        "allowBundled": ["github"],
    }
    assert (tmp_path / "args.txt").read_text(encoding="utf-8").split() == [
        "skills",
        "list",
        "--json",
        "--agent",
        "default",
    ]
    assert report == {
        "loaded": [
            {
                "name": "kept",
                "path": str((team / "kept").resolve()),
                "entry": "skills.directories[0]",
                "origin": "external",
            }
        ],
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
    (tmp_path / "skills" / "a" / "b" / "c" / "d" / "e" / ".private").mkdir(
        parents=True
    )

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
    top = _skill(team / "top", "top")
    _skill(team / "skills" / "a" / "b" / "c" / "d" / "e" / ".private", "private")
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
    packed = _skill(repository / "skills", "packed")
    _skill(packed / "nested", "nested")
    skills: dict[str, Any] = {"directories": [{"folder": str(repository)}]}
    if policy is not None:
        skills["duplicates"] = policy

    generated, plan = _generated(tmp_path, workspace, skills)

    assert plan.report() == {
        "loaded": [
            {
                "name": "packed",
                "path": str(packed.resolve()),
                "entry": "skills.directories[0]",
                "origin": "external",
            }
        ]
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
    _skill(outside / "escaped", "escaped")
    _skill(outside / "skills" / "nested-escaped", "nested-escaped")
    inside = _skill(tmp_path / "team" / "inside", "inside")
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
        "loaded": [
            {
                "name": "inside",
                "path": str(inside.resolve()),
                "entry": "skills.directories[0]",
                "origin": "external",
            }
        ],
        "empty_entries": ["skills.directories[1]"],
        "skipped": [
            {
                "path": str((outside / "escaped").resolve()),
                "entry": "skills.directories[0]",
                "reason": "path_escape",
            },
            {
                "path": str((outside / "skills").resolve()),
                "entry": "skills.directories[1]",
                "reason": "path_escape",
            },
        ],
    }


@pytest.mark.parametrize("policy", [None, "first_wins"])
def test_skills_child_of_a_scanned_root_is_searched_six_levels(
    tmp_path: Path, workspace: Path, policy: str | None
):
    """The root keeps 2 levels while its skills child is searched 6 levels deep."""
    team = tmp_path / "team"
    _skill(team / "top", "top")
    deep = _skill(team / "skills" / "g" / "h" / "deep", "deep")
    _skill(team / "a" / "b" / "too-deep", "too-deep")
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
    _skill(tmp_path / "a" / "top", "top")
    x = _skill(tmp_path / "a" / "b" / "skills" / "x", "x")

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
    _skill(tmp_path / "is-skill", "is-skill")

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
        _skill(tmp_path / f"copy-{index}" / "review", "review") for index in range(3)
    ]
    other = _skill(tmp_path / "copy-1" / "other", "other")
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
        {
            "path": str(copies[refused].resolve()),
            "entry": f"skills.directories[{refused}]",
            "reason": "rejected_by_openclaw",
        }
    ]
    # The next listing shows the replacement, so nothing changes again.
    listed = {
        "skills": [*status["skills"], {"name": "review", "source": "openclaw-extra"}]
    }
    assert openclaw_skills.apply_status(plan, listed, fallback=True) is False
    assert plan.report() == report


def test_refused_winner_without_fallback_loads_no_copy(tmp_path: Path, workspace: Path):
    """Without a fallback the report says no copy of the name loads."""
    first = _skill(tmp_path / "a" / "review", "review")
    _skill(tmp_path / "b" / "review", "review")
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
    _skill(tmp_path / "a" / "review", "review")
    second = _skill(tmp_path / "b" / "review", "review")
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
    mine = _skill(tmp_path / "team" / "diagnose-gateway", "diagnose-gateway")
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
    config = AgentConfig.from_mapping(
        {
            "harness": {"settings": settings},
            "models": {"default": {"provider": "test", "model": "fabric-echo"}},
            "skills": skills,
        }
    )

    plan = adapter._skill_plan(
        config,
        _context(workspace),
        base_dir=tmp_path,
        command=package / "openclaw.mjs",
        env={},
    )

    assert plan.custodian is False
    assert plan.report() == {
        "loaded": [
            {
                "name": "diagnose-gateway",
                "path": str(mine.resolve()),
                "entry": "skills.directories[0]",
                "origin": "external",
            }
        ],
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
        _config({"directories": []}),
        _context(tmp_path),
        {},
        {"operation": "connect", "connection_path": str(connection)},
    )
    try:
        assert runtime._skills_output() == {"skills": report}
    finally:
        await runtime.stop()
