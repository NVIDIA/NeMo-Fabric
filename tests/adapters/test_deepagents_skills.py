# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deep Agents loads ``skills.paths`` through its backend (real deepagents SDK)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from deepagents.backends import FilesystemBackend
from deepagents.middleware.skills import _list_skills_with_errors
from nemo_fabric_adapters.deepagents import adapter
from nemo_fabric_adapters.deepagents import skills as skill_mounts


def write_skill(root: Path, name: str, *, front_matter_name: str | None = None) -> Path:
    (root / "references").mkdir(parents=True, exist_ok=True)
    (root / "SKILL.md").write_text(
        f"---\nname: {front_matter_name or name}\ndescription: The {name} skill.\n---\n"
        f"# {name}\n"
    )
    (root / "references" / "example.md").write_text(f"{name} reference\n")
    return root


def context(workspace: Path | None) -> SimpleNamespace:
    return SimpleNamespace(
        environment=SimpleNamespace(
            workspace=str(workspace) if workspace is not None else None, env={}
        )
    )


def config(*paths: Path | str) -> SimpleNamespace:
    return SimpleNamespace(skills=SimpleNamespace(paths=[str(p) for p in paths]))


def resolve(
    tmp_path: Path, *paths: Path | str, workspace: Path | None = None, local_shell=False
):
    runtime_context = context(workspace)
    backend = None
    if local_shell:
        backend = adapter._local_shell_backend(
            root_dir=str(workspace), virtual_mode=True, env={}, inherit_env=False
        )
    elif workspace is not None:
        backend = adapter.resolve_backend(runtime_context, str(tmp_path))
    return adapter.resolve_skills(
        config(*paths),
        runtime_context,
        str(tmp_path),
        backend,
        local_shell=local_shell,
    )


def loaded(sources, backend) -> list[str]:
    names = []
    for source in sources:
        skills, error = _list_skills_with_errors(backend, source)
        assert error is None, error
        names.extend(skill["name"] for skill in skills)
    return names


def test_single_skill_outside_the_workspace_is_readable(tmp_path):
    skill = write_skill(tmp_path / "skills" / "report-writing", "report-writing")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    sources, backend = resolve(tmp_path, skill, workspace=workspace)

    assert sources == ["/.fabric/skills/"]
    assert loaded(sources, backend) == ["report-writing"]
    read = backend.read("/.fabric/skills/report-writing/references/example.md")
    assert read.error is None
    assert "report-writing reference" in str(read.file_data or read)
    # Nothing is copied into the workspace.
    assert not (workspace / ".fabric").exists()


def test_folder_of_skills_inside_the_workspace_loads_every_child(tmp_path):
    write_skill(tmp_path / "skills" / "files", "files")
    write_skill(tmp_path / "skills" / "evals", "evals")
    (tmp_path / "skills" / "notes").mkdir()

    sources, backend = resolve(tmp_path, tmp_path / "skills", workspace=tmp_path)

    assert sorted(loaded(sources, backend)) == ["evals", "files"]
    # The workspace itself is still served by the default backend.
    assert backend.ls("/skills").error is None


def test_mixed_entries_keep_config_order(tmp_path):
    write_skill(tmp_path / "pack" / "b", "b")
    write_skill(tmp_path / "pack" / "c", "c")
    single = write_skill(tmp_path / "one" / "a", "a")

    sources, backend = resolve(tmp_path, single, tmp_path / "pack", workspace=tmp_path)

    resolved = skill_mounts.resolve_skill_paths(
        [str(single), str(tmp_path / "pack")], tmp_path
    )
    assert [(s.name, s.entry_index) for s in resolved.skills] == [
        ("a", 0),
        ("b", 1),
        ("c", 1),
    ]
    assert sorted(loaded(sources, backend)) == ["a", "b", "c"]


def test_skills_load_without_a_workspace(tmp_path):
    skill = write_skill(tmp_path / "skills" / "report-writing", "report-writing")

    sources, backend = resolve(tmp_path, skill, workspace=None)

    assert loaded(sources, backend) == ["report-writing"]


def test_mounted_skills_are_read_only(tmp_path):
    skill = write_skill(tmp_path / "skills" / "report-writing", "report-writing")
    original = (skill / "SKILL.md").read_text()

    _, backend = resolve(tmp_path, skill, workspace=tmp_path / "ws")

    path = "/.fabric/skills/report-writing/SKILL.md"
    assert "read-only" in backend.write(path, "changed").error
    assert "read-only" in backend.edit(path, "name", "nom").error
    assert (skill / "SKILL.md").read_text() == original


def test_front_matter_name_is_the_mount_name(tmp_path):
    skill = write_skill(
        tmp_path / "skills" / "folder-name",
        "folder-name",
        front_matter_name="real-name",
    )

    sources, backend = resolve(tmp_path, skill, workspace=tmp_path)

    assert loaded(sources, backend) == ["real-name"]


def test_duplicate_names_fail_start_naming_both_entries(tmp_path):
    first = write_skill(tmp_path / "a" / "review", "review")
    second = write_skill(tmp_path / "b" / "review", "review")

    with pytest.raises(adapter.AdapterConfigError) as error:
        resolve(tmp_path, first, second, workspace=tmp_path)

    assert "skills.paths[0]" in str(error.value)
    assert "skills.paths[1]" in str(error.value)


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_invalid_entries_fail_start_naming_the_entry(tmp_path, kind):
    entry = tmp_path / "skills"
    if kind == "file":
        entry.write_text("not a folder")

    with pytest.raises(adapter.AdapterConfigError, match=r"skills\.paths\[0\]"):
        resolve(tmp_path, entry, workspace=tmp_path / "ws")


def test_workspace_skills_folder_passes_through(tmp_path):
    # Skills already staged under the workspace and named by their workspace path.
    workspace = tmp_path / "ws"
    write_skill(workspace / ".agents" / "skills" / "report-writing", "report-writing")

    sources, backend = resolve(tmp_path, "/.agents/skills", workspace=workspace)

    assert sources == ["/.agents/skills"]
    assert loaded(sources, backend) == ["report-writing"]
    assert isinstance(backend, FilesystemBackend)


def test_local_shell_copies_skills_into_the_workspace(tmp_path):
    skill = write_skill(tmp_path / "skills" / "report-writing", "report-writing")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    stale = workspace / ".fabric" / "skills" / "removed"
    stale.mkdir(parents=True)

    sources, backend = resolve(tmp_path, skill, workspace=workspace, local_shell=True)

    assert sources == ["/.fabric/skills/"]
    assert loaded(sources, backend) == ["report-writing"]
    copied = workspace / ".fabric" / "skills" / "report-writing"
    assert (copied / "references" / "example.md").is_file()
    assert not stale.exists()


def test_no_skill_paths_leaves_the_backend_alone(tmp_path):
    runtime_context = context(tmp_path)
    backend = adapter.resolve_backend(runtime_context, str(tmp_path))

    assert adapter.resolve_skills(
        SimpleNamespace(skills=None), runtime_context, str(tmp_path), backend
    ) == (None, backend)
