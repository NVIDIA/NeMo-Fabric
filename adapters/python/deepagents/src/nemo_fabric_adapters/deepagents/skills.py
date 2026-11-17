# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make NeMo Fabric ``skills.paths`` readable through the Deep Agents backend.

NeMo Fabric resolves each ``skills.paths`` entry to a path in the task
environment. Deep Agents reads skills only through its backend, which confines
file access to the workspace and treats every path as relative to it, so those
paths cannot be read directly. This module resolves each entry on disk and
exposes the skills it finds at a reserved virtual path:

- an entry that contains ``SKILL.md`` is one skill;
- any other folder is a folder of skills, and each immediate child that
  contains ``SKILL.md`` is a skill;
- an absolute entry that does not exist in the task environment but does exist
  under the workspace is a workspace path that is passed to Deep Agents
  unchanged (for configurations that stage skills into the workspace
  themselves);
- anything else fails start, naming the ``skills.paths`` entry.

The default backends mount the skills read-only through a routed backend, so
nothing is copied. The ``local_shell`` backend copies them into the workspace
instead, so shell commands can run skill scripts.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Reserved virtual folder that holds the configured skills. Deep Agents lists
# its immediate children as skills.
SKILLS_MOUNT = "/.fabric/skills/"
_SKILL_FILE = "SKILL.md"
_READ_ONLY_ERROR = (
    "Skills loaded from skills.paths are read-only; changes to them would not persist."
)


class SkillsConfigError(RuntimeError):
    """Raised when a ``skills.paths`` entry cannot be loaded."""


@dataclass(frozen=True)
class Skill:
    """One skill folder found in the task environment."""

    name: str
    directory: Path
    entry_index: int


@dataclass
class ResolvedSkills:
    """Skills resolved from ``skills.paths``, ready for Deep Agents."""

    skills: list[Skill] = field(default_factory=list)
    workspace_sources: list[str] = field(default_factory=list)

    def report(self) -> dict[str, Any]:
        """Describe what was loaded, for logs and run evidence."""

        return {
            "loaded": [
                {"name": skill.name, "entry": f"skills.paths[{skill.entry_index}]"}
                for skill in self.skills
            ],
            "workspace_sources": list(self.workspace_sources),
        }


def _front_matter_name(skill_file: Path) -> str | None:
    """Return the ``name`` from the skill's YAML front matter, if it has one."""

    try:
        text = skill_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    try:
        data = yaml.safe_load(text[3:end])
    except yaml.YAMLError:
        return None
    name = data.get("name") if isinstance(data, dict) else None
    return name.strip() if isinstance(name, str) and name.strip() else None


def _skill_name(directory: Path) -> str:
    # Deep Agents keys skills by their front-matter name; fall back to the folder
    # name so a malformed skill is still mounted and Deep Agents reports it.
    name = _front_matter_name(directory / _SKILL_FILE)
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        return directory.name
    return name


def _workspace_source(entry: str, workspace: Path | None) -> bool:
    if workspace is None or not entry.startswith("/"):
        return False
    candidate = workspace / entry.lstrip("/")
    return candidate.is_dir()


def resolve_skill_paths(entries: list[str], workspace: Path | None) -> ResolvedSkills:
    """Resolve ``skills.paths`` entries in the task environment.

    Fails with :class:`SkillsConfigError` naming the entry when an entry does
    not exist or is not a folder, or when two skills share a name.
    """

    resolved = ResolvedSkills()
    seen: dict[str, Skill] = {}
    for index, entry in enumerate(entries):
        path = Path(entry)
        if not path.exists():
            if _workspace_source(entry, workspace):
                resolved.workspace_sources.append(entry)
                continue
            raise SkillsConfigError(
                f"skills.paths[{index}] ({entry!r}) was not found in the task environment."
            )
        if not path.is_dir():
            raise SkillsConfigError(
                f"skills.paths[{index}] ({entry!r}) is not a folder; each entry must be "
                f"a skill folder containing {_SKILL_FILE} or a folder of skills."
            )
        if (path / _SKILL_FILE).is_file():
            directories = [path]
        else:
            directories = sorted(
                child
                for child in path.iterdir()
                if child.is_dir() and (child / _SKILL_FILE).is_file()
            )
            if not directories:
                logger.warning(
                    "skills.paths[%d] (%r) contains no skills.", index, entry
                )
        for directory in directories:
            skill = Skill(_skill_name(directory), directory.resolve(), index)
            if skill.name in seen:
                first = seen[skill.name]
                raise SkillsConfigError(
                    f"Skill name {skill.name!r} is defined twice: in "
                    f"skills.paths[{first.entry_index}] ({str(first.directory)!r}) and "
                    f"skills.paths[{index}] ({str(skill.directory)!r}). Skill names must "
                    "be unique."
                )
            seen[skill.name] = skill
            resolved.skills.append(skill)
    return resolved


def skill_sources(resolved: ResolvedSkills) -> list[str] | None:
    """Return the Deep Agents ``skills`` sources for the resolved entries."""

    sources = [SKILLS_MOUNT] if resolved.skills else []
    sources.extend(resolved.workspace_sources)
    return sources or None


def log_loaded(resolved: ResolvedSkills) -> None:
    """Log which skills each ``skills.paths`` entry provided."""

    for skill in resolved.skills:
        logger.info(
            "Loaded skill %r from skills.paths[%d] at %s%s",
            skill.name,
            skill.entry_index,
            SKILLS_MOUNT,
            skill.name,
        )
    for source in resolved.workspace_sources:
        logger.info("Passing workspace skills folder %r to Deep Agents.", source)


def _read_only_backend(directory: Path) -> Any:
    from deepagents.backends import FilesystemBackend
    from deepagents.backends.protocol import EditResult
    from deepagents.backends.protocol import FileUploadResponse
    from deepagents.backends.protocol import WriteResult

    class ReadOnlySkillBackend(FilesystemBackend):
        """A skill folder that the agent can read but not change."""

        def write(self, file_path: str, content: str) -> Any:
            return WriteResult(error=_READ_ONLY_ERROR)

        def edit(self, file_path: str, *args: Any, **kwargs: Any) -> Any:
            return EditResult(error=_READ_ONLY_ERROR)

        def delete(self, file_path: str) -> Any:
            from deepagents.backends.protocol import DeleteResult

            return DeleteResult(error=_READ_ONLY_ERROR)

        def upload_files(self, files: list[tuple[str, bytes]]) -> list[Any]:
            return [
                FileUploadResponse(path=path, error="permission_denied")
                for path, _content in files
            ]

    return ReadOnlySkillBackend(root_dir=str(directory), virtual_mode=True)


def _empty_backend() -> Any:
    from deepagents.backends.protocol import BackendProtocol
    from deepagents.backends.protocol import GlobResult
    from deepagents.backends.protocol import GrepResult
    from deepagents.backends.protocol import LsResult
    from deepagents.backends.protocol import ReadResult

    class EmptyBackend(BackendProtocol):
        """The skills folder itself; its children are the routed skills."""

        def ls(self, path: str) -> Any:
            return LsResult(entries=[])

        def read(self, file_path: str, *args: Any, **kwargs: Any) -> Any:
            return ReadResult(error=f"File '{file_path}' not found")

        def grep(self, pattern: str, *args: Any, **kwargs: Any) -> Any:
            return GrepResult(matches=[])

        def glob(self, pattern: str, *args: Any, **kwargs: Any) -> Any:
            return GlobResult(matches=[])

    return EmptyBackend()


def mounted_backend(default: Any, resolved: ResolvedSkills) -> Any:
    """Route ``SKILLS_MOUNT`` to read-only views of the resolved skills.

    ``default`` serves every other path. Returns ``default`` unchanged when no
    skills need mounting.
    """

    if not resolved.skills:
        return default
    from deepagents.backends import CompositeBackend
    from deepagents.backends import StateBackend

    skills_folder = CompositeBackend(
        default=_empty_backend(),
        routes={
            f"/{skill.name}/": _read_only_backend(skill.directory)
            for skill in resolved.skills
        },
    )
    return CompositeBackend(
        default=default if default is not None else StateBackend(),
        routes={SKILLS_MOUNT: skills_folder},
    )


def copy_into_workspace(workspace: Path, resolved: ResolvedSkills) -> None:
    """Copy the resolved skills to ``SKILLS_MOUNT`` under the workspace.

    Used for the ``local_shell`` backend, whose shell commands only see real
    files. The folder is replaced on every start so it matches ``skills.paths``.
    """

    target = workspace / SKILLS_MOUNT.strip("/")
    if target.exists():
        shutil.rmtree(target)
    if not resolved.skills:
        return
    target.mkdir(parents=True)
    for skill in resolved.skills:
        shutil.copytree(skill.directory, target / skill.name, symlinks=False)
