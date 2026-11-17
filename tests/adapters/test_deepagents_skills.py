# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deep Agents loads configured skills through its backend (real deepagents SDK)."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from deepagents.backends import FilesystemBackend
from deepagents.middleware.skills import _list_skills_with_errors
from nemo_fabric_adapter_contract.models import AgentSkillConfig
from nemo_fabric_adapter_contract.models import SkillDirectory
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.deepagents import adapter

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="skills mount at their POSIX task-environment paths"
)


def write_skill(root: Path, name: str, *, description: bool = True) -> Path:
    (root / "references").mkdir(parents=True, exist_ok=True)
    lines = ["---", f"name: {name}"]
    if description:
        lines.append(f"description: The {name} skill.")
    (root / "SKILL.md").write_text("\n".join([*lines, "---", f"# {name}", ""]))
    (root / "references" / "example.md").write_text(f"{root} reference\n")
    return root


def context(workspace: Path | None) -> SimpleNamespace:
    return SimpleNamespace(
        environment=SimpleNamespace(
            workspace=str(workspace) if workspace is not None else None, env={}
        )
    )


def resolve(tmp_path, skills, *, workspace=None, local_shell=False):
    runtime_context = context(workspace)
    if local_shell:
        backend = adapter._local_shell_backend(
            root_dir=str(workspace), virtual_mode=True, env={}, inherit_env=False
        )
    else:
        backend = adapter.resolve_backend(runtime_context, str(tmp_path))
    report: list = []
    sources, backend = adapter.resolve_skills(
        SimpleNamespace(skills=skills), runtime_context, str(tmp_path), backend, report
    )
    return sources, backend, (report[0] if report else None)


def paths(*entries) -> AgentSkillConfig:
    return AgentSkillConfig(paths=[str(entry) for entry in entries])


def directories(*entries, duplicates=None) -> AgentSkillConfig:
    return AgentSkillConfig(
        directories=[SkillDirectory(**{kind: str(path)}) for kind, path in entries],
        duplicates=duplicates,
    )


def loaded(sources, backend) -> dict[str, str]:
    """Load the sources the way SkillsMiddleware does: the later source wins."""

    skills = {}
    for source in sources:
        found, error = _list_skills_with_errors(backend, source)
        assert error is None, error
        skills.update({skill["name"]: skill["path"] for skill in found})
    return skills


def test_single_skill_outside_the_workspace_loads_at_its_own_path(tmp_path):
    skill = write_skill(tmp_path / "skills" / "report-writing", "report-writing")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    sources, backend, report = resolve(tmp_path, paths(skill), workspace=workspace)

    assert sources == [f"{skill.parent}/"]
    assert loaded(sources, backend) == {"report-writing": f"{skill}/SKILL.md"}
    assert backend.read(f"{skill}/references/example.md").error is None
    assert report["loaded"] == [
        {
            "name": "report-writing",
            "path": str(skill.resolve()),
            "entry": "skills.paths[0]",
            "origin": "external",
        }
    ]
    # Nothing is copied into the workspace.
    assert list(workspace.iterdir()) == []


def test_folder_of_skills_inside_the_workspace_loads_every_child(tmp_path):
    write_skill(tmp_path / "skills" / "files", "files")
    write_skill(tmp_path / "skills" / "evals", "evals")
    write_skill(tmp_path / "skills" / "broken", "broken", description=False)
    (tmp_path / "skills" / "notes").mkdir()

    sources, backend, report = resolve(
        tmp_path, paths(tmp_path / "skills"), workspace=tmp_path
    )

    assert sorted(loaded(sources, backend)) == ["evals", "files"]
    assert [item["reason"] for item in report["skipped"]] == ["invalid_frontmatter"]
    # The workspace is still served by the default backend.
    assert backend.ls("/skills").error is None


def test_skills_load_without_a_workspace(tmp_path):
    skill = write_skill(tmp_path / "skills" / "report-writing", "report-writing")

    sources, backend, _ = resolve(tmp_path, paths(skill), workspace=None)

    assert list(loaded(sources, backend)) == ["report-writing"]


def test_writes_reach_the_mounted_folder(tmp_path):
    skill = write_skill(tmp_path / "skills" / "report-writing", "report-writing")

    _, backend, _ = resolve(tmp_path, paths(skill), workspace=tmp_path / "ws")

    assert backend.write(f"{skill}/notes.md", "learned").error is None
    assert (skill / "notes.md").read_text() == "learned"


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores permissions"
)
def test_a_read_only_mount_refuses_writes(tmp_path):
    skill = write_skill(tmp_path / "skills" / "report-writing", "report-writing")
    original = (skill / "SKILL.md").read_text()
    _, backend, _ = resolve(tmp_path, paths(skill), workspace=tmp_path / "ws")
    (skill / "SKILL.md").chmod(0o444)
    skill.chmod(0o555)
    try:
        assert backend.edit(f"{skill}/SKILL.md", "name", "nom").error
        assert backend.write(f"{skill}/notes.md", "x").error
    finally:
        skill.chmod(0o755)
        (skill / "SKILL.md").chmod(0o644)
    assert (skill / "SKILL.md").read_text() == original


def test_paths_duplicates_follow_deep_agents_later_source_wins(tmp_path):
    first = write_skill(tmp_path / "a" / "review", "review")
    second = write_skill(tmp_path / "b" / "review", "review")

    sources, backend, report = resolve(
        tmp_path, paths(first, second), workspace=tmp_path
    )

    assert loaded(sources, backend) == {"review": f"{second}/SKILL.md"}
    assert report["duplicates"] == [
        {
            "name": "review",
            "outcome": "selected",
            "selected": str(second.resolve()),
            "copies": [
                {
                    "origin": "external",
                    "path": str(first.resolve()),
                    "entry": "skills.paths[0]",
                },
                {
                    "origin": "external",
                    "path": str(second.resolve()),
                    "entry": "skills.paths[1]",
                },
            ],
        }
    ]
    assert [skill["entry"] for skill in report["loaded"]] == ["skills.paths[1]"]


@pytest.mark.parametrize(
    ("policy", "winner"), [("first_wins", "a"), ("last_wins", "b")]
)
def test_directories_duplicate_policy_picks_the_copy(tmp_path, policy, winner):
    write_skill(tmp_path / "a" / "review", "review")
    write_skill(tmp_path / "b" / "review", "review")

    sources, backend, report = resolve(
        tmp_path,
        directories(
            ("folder", tmp_path / "a"), ("folder", tmp_path / "b"), duplicates=policy
        ),
        workspace=tmp_path,
    )

    expected = tmp_path / winner / "review"
    assert loaded(sources, backend) == {"review": f"{expected}/SKILL.md"}
    assert report["duplicates"][0]["selected"] == str(expected.resolve())


def test_directories_reject_fails_start_naming_both_entries(tmp_path):
    write_skill(tmp_path / "a" / "review", "review")
    second = write_skill(tmp_path / "b" / "review", "review")

    with pytest.raises(lifecycle.LifecycleError) as error:
        resolve(
            tmp_path,
            directories(
                ("folder", tmp_path / "a"), ("skill", second), duplicates="reject"
            ),
            workspace=tmp_path,
        )

    assert error.value.code == "deepagents_skill_invalid"
    assert error.value.metadata["reason"] == "duplicate"
    assert error.value.metadata["field"] == "skills.directories[1]"


def test_skill_and_folder_entries_keep_config_order(tmp_path):
    single = write_skill(tmp_path / "one" / "a", "a")
    write_skill(tmp_path / "pack" / "b", "b")
    write_skill(tmp_path / "pack" / "c", "c")

    sources, backend, report = resolve(
        tmp_path,
        directories(("skill", single), ("folder", tmp_path / "pack")),
        workspace=tmp_path,
    )

    assert sources == [f"{single.parent}/", f"{tmp_path / 'pack'}/"]
    assert sorted(loaded(sources, backend)) == ["a", "b", "c"]
    assert [skill["entry"] for skill in report["loaded"]] == [
        "skills.directories[0]",
        "skills.directories[1]",
        "skills.directories[1]",
    ]


def test_a_skill_entry_hides_its_siblings(tmp_path):
    chosen = write_skill(tmp_path / "skills" / "chosen", "chosen")
    write_skill(tmp_path / "skills" / "other", "other")

    sources, backend, _ = resolve(
        tmp_path, directories(("skill", chosen)), workspace=tmp_path / "ws"
    )

    assert list(loaded(sources, backend)) == ["chosen"]


def test_empty_folder_entry_is_reported(tmp_path):
    (tmp_path / "empty").mkdir()

    sources, _, report = resolve(
        tmp_path, directories(("folder", tmp_path / "empty")), workspace=tmp_path
    )

    assert sources is None
    assert report == {"empty_entries": ["skills.directories[0]"]}


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_invalid_paths_entries_fail_start_naming_the_entry(tmp_path, kind):
    write_skill(tmp_path / "ok" / "fine", "fine")
    entry = tmp_path / "skills"
    if kind == "file":
        entry.write_text("not a folder")

    with pytest.raises(lifecycle.LifecycleError) as error:
        resolve(tmp_path, paths(tmp_path / "ok", entry), workspace=tmp_path / "ws")

    assert error.value.code == (
        "deepagents_skill_not_found"
        if kind == "missing"
        else "deepagents_skill_invalid"
    )
    assert error.value.metadata["field"] == "skills.paths[1]"
    assert "skills.paths[1]" in str(error.value)


def test_workspace_skills_folder_passes_through(tmp_path):
    # Skills already staged under the workspace, named by their workspace path.
    workspace = tmp_path / "ws"
    write_skill(workspace / ".agents" / "skills" / "report-writing", "report-writing")

    sources, backend, report = resolve(
        tmp_path, paths("/.agents/skills"), workspace=workspace
    )

    assert sources == ["/.agents/skills"]
    assert list(loaded(sources, backend)) == ["report-writing"]
    assert isinstance(backend, FilesystemBackend)
    assert report["loaded"][0]["entry"] == "skills.paths[0]"


def test_workspace_skills_folder_is_compared_by_name(tmp_path):
    # A routed skill and a staged workspace skill with the same name: Deep
    # Agents keeps the later source, and the report records the clash.
    workspace = tmp_path / "ws"
    staged = write_skill(workspace / ".agents" / "skills" / "review", "review")
    external = write_skill(tmp_path / "skills" / "review", "review")

    sources, backend, report = resolve(
        tmp_path, paths(external, "/.agents/skills"), workspace=workspace
    )

    # The earlier, shadowed copy is not routed at all.
    assert sources == ["/.agents/skills"]
    assert loaded(sources, backend) == {"review": "/.agents/skills/review/SKILL.md"}
    assert report["duplicates"][0]["outcome"] == "selected"
    assert report["duplicates"][0]["selected"] == str(staged.resolve())
    assert [copy["entry"] for copy in report["duplicates"][0]["copies"]] == [
        "skills.paths[0]",
        "skills.paths[1]",
    ]
    assert [skill["entry"] for skill in report["loaded"]] == ["skills.paths[1]"]


def test_local_shell_runs_skill_scripts_at_the_listed_path(tmp_path):
    skill = write_skill(tmp_path / "skills" / "report-writing", "report-writing")
    (skill / "scripts").mkdir()
    (skill / "scripts" / "token.sh").write_text("echo SCRIPT-OK\n")
    workspace = tmp_path / "ws"
    workspace.mkdir()

    sources, backend, _ = resolve(
        tmp_path, paths(skill), workspace=workspace, local_shell=True
    )

    skill_md = loaded(sources, backend)["report-writing"]
    script = Path(skill_md).parent / "scripts" / "token.sh"
    result = backend.execute(f"sh {script}")
    assert result.exit_code == 0
    assert "SCRIPT-OK" in result.output
    assert list(workspace.iterdir()) == []


def test_no_skills_leaves_the_backend_alone(tmp_path):
    runtime_context = context(tmp_path)
    backend = adapter.resolve_backend(runtime_context, str(tmp_path))

    assert adapter.resolve_skills(
        SimpleNamespace(skills=None), runtime_context, str(tmp_path), backend
    ) == (None, backend)


def test_descriptor_advertises_implemented_skill_support():
    """The descriptor's skills block matches what the adapter implements."""

    root = Path(__file__).resolve().parents[2]
    descriptor = json.loads(
        (root / "adapters/python/deepagents/deepagents.fabric-adapter.json").read_text(
            encoding="utf-8"
        )
    )

    assert "skills" in descriptor["config"]["accepts"]
    skills = descriptor["skills"]
    assert skills["directory_loading"] == {
        "supported_entries": ["skill", "folder"],
        "skill_file": "SKILL.md",
        "discovery": {
            "traversal": "immediate_children",
            "nested_skills": "stop_at_skill",
            "depth_limit": {"kind": "fixed", "levels": 1},
        },
        "parsing": {
            "required_frontmatter_fields": ["name", "description"],
            "name_fallback": "none",
            "frontmatter_errors": "reject",
        },
    }
    assert "bundled" not in skills
    assert skills["duplicates"]["supported_policies"] == [
        "reject",
        "first_wins",
        "last_wins",
    ]
    assert skills["duplicates"]["name_source"] == "frontmatter"


def test_planning_accepts_directories_and_rejects_bundled(tmp_path):
    from nemo_fabric import Fabric
    from nemo_fabric import FabricConfig

    def config(skills: dict) -> FabricConfig:
        return FabricConfig.model_validate(
            {
                "metadata": {"name": "skills"},
                "harness": {
                    "adapter_id": "nvidia.fabric.langchain.deepagents",
                    "resolution": "preinstalled",
                },
                "models": {
                    "default": {
                        "provider": "nvidia",
                        "model": "m",
                        "api_key_env": "NVIDIA_API_KEY",
                    }
                },
                "skills": skills,
                "runtime": {"input_schema": "chat", "output_schema": "message"},
                "environment": {"provider": "local", "workspace": "."},
            }
        )

    Fabric().plan(
        config(
            {
                "directories": [{"folder": "skills"}, {"skill": "one/review"}],
                "duplicates": "last_wins",
            }
        ),
        base_dir=tmp_path,
    )
    with pytest.raises(Exception, match="skills.bundled"):
        Fabric().plan(config({"bundled": {}}), base_dir=tmp_path)


def test_workspace_path_to_one_skill_loads_that_skill(tmp_path):
    workspace = tmp_path / "ws"
    write_skill(workspace / ".agents" / "skills" / "review", "review")

    sources, backend, report = resolve(
        tmp_path, paths("/.agents/skills/review"), workspace=workspace
    )

    assert list(loaded(sources, backend)) == ["review"]
    assert report["loaded"][0]["entry"] == "skills.paths[0]"


def test_symlinks_to_folders_with_one_name_keep_their_own_routes(tmp_path):
    first = write_skill(tmp_path / "vendor-a" / "review", "alpha")
    second = write_skill(tmp_path / "vendor-b" / "review", "beta")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "alpha").symlink_to(first)
    (bundle / "beta").symlink_to(second)

    sources, backend, report = resolve(
        tmp_path, directories(("folder", bundle)), workspace=tmp_path / "ws"
    )

    assert loaded(sources, backend) == {
        "alpha": f"{bundle}/alpha/SKILL.md",
        "beta": f"{bundle}/beta/SKILL.md",
    }
    assert sorted(skill["name"] for skill in report["loaded"]) == ["alpha", "beta"]


def test_a_skill_inside_another_skill_entry_stays_reachable(tmp_path):
    outer = write_skill(tmp_path / "skills" / "outer", "outer")
    inner = write_skill(outer / "inner", "inner")

    sources, backend, _ = resolve(
        tmp_path,
        directories(("skill", outer), ("skill", inner)),
        workspace=tmp_path / "ws",
    )

    found = loaded(sources, backend)
    assert sorted(found) == ["inner", "outer"]
    for skill_md in found.values():
        companion = str(Path(skill_md).parent / "references" / "example.md")
        assert backend.read(companion).error is None, companion
