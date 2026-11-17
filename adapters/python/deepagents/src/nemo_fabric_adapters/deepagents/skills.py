# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make configured skills readable through the Deep Agents backend.

NeMo Fabric resolves ``skills.paths`` and ``skills.directories`` to folders in
the task environment. Deep Agents reads skills only through its backend, which
confines file access to the workspace and reads every path relative to it, so
it cannot read those folders directly. Deep Agents also treats each ``skills``
source as a folder of skills: it lists the source's immediate child folders
and loads each one that holds ``SKILL.md`` with a ``name`` and a
``description`` in its front matter. The later source wins a name clash.

This module checks the entries with the adapter contract's
:func:`~nemo_fabric_adapter_contract.skills.load_directories` and routes each
entry's skills into the backend at the path where they live in the task
environment. Each entry is a Deep Agents source, in configuration order. The
routes read and write the folders in place, so the mount decides whether the
agent can change a skill, and shell commands of the ``local_shell`` backend
reach the same files at the same paths.

``skills.paths`` keeps its meaning: an entry that holds ``SKILL.md`` is one
skill, and any other folder is a folder of skills. An absolute entry that does
not exist in the task environment, but does under the workspace, names skills
already staged in the workspace and is passed to Deep Agents unchanged.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from nemo_fabric_adapter_contract.models import AgentSkillConfig
from nemo_fabric_adapter_contract.models import SkillDirectory
from nemo_fabric_adapter_contract.skills import SKILL_FILE
from nemo_fabric_adapter_contract.skills import CandidateInfo
from nemo_fabric_adapter_contract.skills import CandidateReader
from nemo_fabric_adapter_contract.skills import DirectoryLoad
from nemo_fabric_adapter_contract.skills import DiscoveryRules
from nemo_fabric_adapter_contract.skills import DuplicateRecord
from nemo_fabric_adapter_contract.skills import FoundSkill
from nemo_fabric_adapter_contract.skills import SkillLoadError
from nemo_fabric_adapter_contract.skills import front_matter_reader
from nemo_fabric_adapter_contract.skills import load_directories
from nemo_fabric_adapter_contract.skills import report_block
from nemo_fabric_adapter_contract.skills import warn
from nemo_fabric_adapters.common import lifecycle

ADAPTER = "Deep Agents"

# deepagents/middleware/skills.py _list_skills: a source's immediate child
# folders are its skills, nothing deeper loads, and every child is listed.
DEEPAGENTS_DISCOVERY = DiscoveryRules(
    traversal="immediate_children",
    nested_skills="stop_at_skill",
    max_depth=1,
    name_fallback=False,
    excluded_dir_names=frozenset(),
)

# Mount point for an entry whose own folder cannot be a backend path.
_FALLBACK_MOUNT = "/.fabric/skills/{index}/"


@dataclass
class SkillMounts:
    """The Deep Agents ``skills`` sources and the backend routes behind them."""

    sources: list[str] = dataclass_field(default_factory=list)
    """Deep Agents ``skills`` sources, in configuration order."""
    routes: dict[str, dict[str, Path]] = dataclass_field(default_factory=dict)
    """Source path to the skill folders it lists, by folder name."""
    load: DirectoryLoad | None = None

    def report(self) -> dict[str, Any] | None:
        """Return the ``output.skills`` block, or ``None`` when empty."""

        return report_block(self.load) if self.load is not None else None

    def warn(self) -> None:
        """Write the start warnings to stderr."""

        if self.load is not None:
            warn(ADAPTER, self.load)


def plan_skills(skills: AgentSkillConfig | None, workspace: Path | None) -> SkillMounts:
    """Check the configured skills and plan the Deep Agents sources.

    Raises:
        lifecycle.LifecycleError: ``deepagents_skill_not_found`` or
            ``deepagents_skill_invalid`` for an entry that cannot load, or a
            duplicate under an explicit policy.
    """

    mounts = SkillMounts()
    if skills is None:
        return mounts
    passthrough: dict[int, str] = {}
    if skills.directories is not None:
        directories: list[SkillDirectory] = list(skills.directories)
        fields = {
            index: f"skills.directories[{index}]" for index in range(len(directories))
        }
    elif skills.paths:
        directories, fields, passthrough = _paths_as_directories(
            [os.fspath(path) for path in skills.paths], workspace
        )
    else:
        return mounts
    try:
        load = load_directories(
            directories,
            rules=DEEPAGENTS_DISCOVERY,
            duplicates=skills.duplicates,
            reader=deepagents_reader(),
        )
    except SkillLoadError as error:
        raise skill_error(_renamed_error(error, fields)) from None
    if skills.duplicates is None:
        load = _native_duplicates(load)
    _plan_sources(mounts, load, directories, fields, passthrough)
    mounts.load = _renamed_load(load, fields)
    return mounts


def _paths_as_directories(
    paths: Sequence[str], workspace: Path | None
) -> tuple[list[SkillDirectory], dict[int, str], dict[int, str]]:
    """Read each ``skills.paths`` entry as a ``skill`` or ``folder`` entry.

    Returns the entries, the ``skills.paths`` field of each entry by its
    index, and the workspace paths passed through unchanged, by their
    position in ``skills.paths``.
    """

    directories: list[SkillDirectory] = []
    fields: dict[int, str] = {}
    passthrough: dict[int, str] = {}
    for index, entry in enumerate(paths):
        path = Path(entry)
        if not path.exists() and _in_workspace(entry, workspace):
            passthrough[index] = entry
            continue
        fields[len(directories)] = f"skills.paths[{index}]"
        if (path / SKILL_FILE).is_file():
            directories.append(SkillDirectory(skill=entry))
        else:
            directories.append(SkillDirectory(folder=entry))
    return directories, fields, passthrough


def _in_workspace(entry: str, workspace: Path | None) -> bool:
    if workspace is None or not Path(entry).is_absolute():
        return False
    return (workspace / entry.lstrip("/\\")).is_dir()


def _position(field: str) -> int:
    return int(field.rsplit("[", 1)[1].rstrip("]"))


def _plan_sources(
    mounts: SkillMounts,
    load: DirectoryLoad,
    directories: Sequence[SkillDirectory],
    fields: dict[int, str],
    passthrough: dict[int, str],
) -> None:
    by_entry: dict[str, list[FoundSkill]] = {}
    for skill in load.selected:
        by_entry.setdefault(skill.field, []).append(skill)
    planned: list[tuple[int, str]] = list(passthrough.items())
    for index, directory in enumerate(directories):
        found = by_entry.get(f"skills.directories[{index}]", [])
        if not found:
            continue
        # The source lists the entry's skills by folder name, as Deep Agents
        # reads them, at the path that holds them in the task environment.
        parent = Path(os.path.abspath(directory.path))
        if directory.kind == "skill":
            parent = parent.parent
        source = _mount_point(parent, index)
        routed = mounts.routes.setdefault(source, {})
        for skill in found:
            routed[Path(skill.path).name] = Path(skill.path)
        if all(existing != source for _, existing in planned):
            planned.append((_position(fields[index]), source))
    mounts.sources = [source for _, source in sorted(planned)]


def _mount_point(parent: Path, index: int) -> str:
    """Return the backend path for a folder: its own path when it can be one."""

    if os.name == "nt" or parent.anchor != "/" or parent == Path("/"):
        return _FALLBACK_MOUNT.format(index=index)
    return str(PurePosixPath(parent)).rstrip("/") + "/"


def deepagents_reader() -> CandidateReader:
    """Return a reader that names and checks each candidate as Deep Agents does.

    It runs Deep Agents' own front-matter parser, which requires a ``name``
    and a ``description`` and skips a skill whose front matter does not
    parse. Without Deep Agents, as in dependency-free tests, it is the
    helper's :func:`front_matter_reader` with the same required fields.
    """

    try:
        from deepagents.middleware import skills as native
    except ImportError:
        native = None
    parse = getattr(native, "_parse_skill_metadata", None)
    if parse is None:
        return front_matter_reader(required_fields=("name", "description"))

    def read_one(folder: Path) -> CandidateInfo:
        skill_md = folder / SKILL_FILE
        try:
            text = skill_md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return CandidateInfo(None, valid=False, reason="unreadable")
        metadata = parse(text, str(skill_md), folder.name)
        if metadata is None:
            return CandidateInfo(None, valid=False, reason="invalid_frontmatter")
        return CandidateInfo(str(metadata["name"]))

    def read(folders: Sequence[Path]) -> dict[Path, CandidateInfo]:
        return {folder: read_one(folder) for folder in folders}

    return read


def _native_duplicates(load: DirectoryLoad) -> DirectoryLoad:
    """Apply Deep Agents' own rule when ``duplicates`` is left out.

    Each entry is a source and the later source wins, so a name supplied by
    several entries resolves to the last one. Copies inside one entry stay
    ``ambiguous``.
    """

    shadowed: set[int] = set()
    records: list[DuplicateRecord] = []
    for record in load.duplicates:
        fields = [field for _, _, field in record.copies]
        if record.outcome != "ambiguous" or len(set(fields)) != len(fields):
            records.append(record)
            continue
        winner = record.copies[-1][1]
        shadowed.update(
            id(skill)
            for skill in load.selected
            if skill.name == record.name and skill.path != winner
        )
        records.append(dataclasses.replace(record, outcome="selected", selected=winner))
    return dataclasses.replace(
        load,
        selected=[skill for skill in load.selected if id(skill) not in shadowed],
        duplicates=records,
    )


def _rename(field: str, fields: dict[int, str]) -> str:
    prefix = "skills.directories["
    if not field.startswith(prefix):
        return field
    index, _, rest = field[len(prefix) :].partition("]")
    if not index.isdigit() or int(index) not in fields:
        return field
    name = fields[int(index)]
    # A skills.paths entry is a plain path, without a skill or folder key.
    if name.startswith("skills.paths[") and rest in (".skill", ".folder"):
        rest = ""
    return name + rest


def _renamed_load(load: DirectoryLoad, fields: dict[int, str]) -> DirectoryLoad:
    """Name each entry by the configuration field it came from."""

    return dataclasses.replace(
        load,
        selected=[
            dataclasses.replace(skill, field=_rename(skill.field, fields))
            for skill in load.selected
        ],
        entries=[
            (_rename(field, fields), kind, path) for field, kind, path in load.entries
        ],
        empty_entries=[_rename(field, fields) for field in load.empty_entries],
        skipped=[
            dataclasses.replace(item, field=_rename(item.field, fields))
            for item in load.skipped
        ],
        duplicates=[
            dataclasses.replace(
                record,
                copies=tuple(
                    (origin, path, _rename(field, fields))
                    for origin, path, field in record.copies
                ),
            )
            for record in load.duplicates
        ],
    )


def _renamed_error(error: SkillLoadError, fields: dict[int, str]) -> SkillLoadError:
    message = str(error)
    # Replace higher indexes first so [1] never rewrites part of [10].
    for index in sorted(fields, reverse=True):
        for key in (".skill", ".folder", ""):
            field = f"skills.directories[{index}]{key}"
            message = message.replace(field, _rename(field, fields))
    return SkillLoadError(
        error.reason,
        message,
        field=_rename(error.field, fields),
        paths=error.paths,
    )


def skill_error(error: SkillLoadError) -> lifecycle.LifecycleError:
    """Map a skill entry failure to the adapter's error codes."""

    metadata: dict[str, Any] = {"field": error.field, "reason": error.reason}
    if error.paths:
        metadata["path"] = str(error.paths[0])
    if len(error.paths) > 1:
        metadata["paths"] = [str(path) for path in error.paths]
    code = (
        "deepagents_skill_not_found"
        if error.reason == "not_found"
        else "deepagents_skill_invalid"
    )
    return lifecycle.LifecycleError(code, str(error), metadata=metadata)


def _empty_backend() -> Any:
    from deepagents.backends.protocol import BackendProtocol
    from deepagents.backends.protocol import GlobResult
    from deepagents.backends.protocol import GrepResult
    from deepagents.backends.protocol import LsResult
    from deepagents.backends.protocol import ReadResult

    class EmptyBackend(BackendProtocol):
        """A source folder itself; its children are the routed skills."""

        def ls(self, path: str) -> Any:
            return LsResult(entries=[])

        def read(self, file_path: str, *args: Any, **kwargs: Any) -> Any:
            return ReadResult(error=f"File '{file_path}' not found")

        def grep(self, pattern: str, *args: Any, **kwargs: Any) -> Any:
            return GrepResult(matches=[])

        def glob(self, pattern: str, *args: Any, **kwargs: Any) -> Any:
            return GlobResult(matches=[])

    return EmptyBackend()


def mounted_backend(default: Any, mounts: SkillMounts) -> Any:
    """Route each source to its skill folders; ``default`` serves the rest.

    Returns ``default`` unchanged when nothing needs routing.
    """

    if not mounts.routes:
        return default
    from deepagents.backends import CompositeBackend
    from deepagents.backends import FilesystemBackend
    from deepagents.backends import StateBackend

    routes = {
        source: CompositeBackend(
            default=_empty_backend(),
            routes={
                f"/{name}/": FilesystemBackend(root_dir=str(folder), virtual_mode=True)
                for name, folder in skills.items()
            },
        )
        for source, skills in mounts.routes.items()
    }
    return CompositeBackend(
        default=default if default is not None else StateBackend(),
        routes=routes,
    )
