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
not exist in the task environment, but does under the workspace, names that
workspace folder, for skills already staged there, and loads like any other.
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
    if skills.directories is not None:
        directories: list[SkillDirectory] = list(skills.directories)
        fields = {
            index: f"skills.directories[{index}]" for index in range(len(directories))
        }
    elif skills.paths:
        directories, fields = _paths_as_directories(
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
    _plan_sources(mounts, load, directories, fields)
    mounts.load = _renamed_load(load, fields)
    return mounts


def _paths_as_directories(
    paths: Sequence[str], workspace: Path | None
) -> tuple[list[SkillDirectory], dict[int, str]]:
    """Read each ``skills.paths`` entry as a ``skill`` or ``folder`` entry.

    Returns the entries and the ``skills.paths`` field of each entry by its
    index. An absolute entry that does not exist in the task environment but
    does under the workspace names that folder of the workspace.
    """

    directories: list[SkillDirectory] = []
    fields: dict[int, str] = {}
    for index, entry in enumerate(paths):
        path = Path(entry)
        fields[len(directories)] = f"skills.paths[{index}]"
        if not path.exists() and _in_workspace(entry, workspace):
            assert workspace is not None
            path = workspace / entry.lstrip("/\\")
        if (path / SKILL_FILE).is_file():
            directories.append(SkillDirectory(skill=str(path)))
        else:
            directories.append(SkillDirectory(folder=str(path)))
    return directories, fields


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
) -> None:
    by_entry: dict[str, list[FoundSkill]] = {}
    for skill in load.selected:
        by_entry.setdefault(skill.field, []).append(skill)
    planned: list[tuple[int, str]] = []
    for index, directory in enumerate(directories):
        found = by_entry.get(f"skills.directories[{index}]", [])
        if not found:
            continue
        # The source lists the entry's skills by the folder names they are
        # reached through, at the path that holds them in the task environment.
        configured = Path(os.path.abspath(directory.path))
        parent = configured.parent if directory.kind == "skill" else configured
        names = _reached_names(directory.kind, configured, found)
        source = mount_point(parent, index)
        if any(
            mounts.routes.get(source, {}).get(name, path) != path
            for name, path in names.items()
        ):
            # Another entry lists a different skill under one of these names here.
            source = _FALLBACK_MOUNT.format(index=index)
        mounts.routes.setdefault(source, {}).update(names)
        if all(existing != source for _, existing in planned):
            planned.append((_position(fields[index]), source))
    mounts.sources = [source for _, source in sorted(planned)]


def _reached_names(
    kind: str, configured: Path, found: Sequence[FoundSkill]
) -> dict[str, Path]:
    """Return each skill's folder name as reached from the entry, to its folder.

    Discovery reports resolved folders, so a symbolic link is listed under its
    own name, as Deep Agents lists it.
    """

    if kind == "skill":
        return {configured.name: Path(found[0].path)}
    by_target: dict[Path, str] = {}
    try:
        children = sorted(configured.iterdir())
    except OSError:
        children = []
    for child in children:
        try:
            by_target.setdefault(child.resolve(), child.name)
        except (OSError, RuntimeError):
            # A symbolic link loop raises RuntimeError before Python 3.13.
            continue
    names: dict[str, Path] = {}
    for skill in found:
        target = Path(skill.path)
        name = by_target.get(target, target.name)
        if name in names:
            name = (
                skill.name if skill.name not in names else f"{skill.name}-{len(names)}"
            )
        names[name] = target
    return names


def mount_point(parent: Path, index: int) -> str:
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


def _with_paths(result: Any, field: str, offset: str) -> Any:
    """Drop ``offset`` from the paths in a listing or search result."""

    def strip(item: Any) -> Any:
        path = item.get("path") if isinstance(item, dict) else None
        if isinstance(path, str) and path.startswith(offset):
            return {**item, "path": path[len(offset) :] or "/"}
        return item

    if isinstance(result, list):
        return [strip(item) for item in result]
    items = getattr(result, field, None)
    if items is None:
        return result
    return dataclasses.replace(result, **{field: [strip(item) for item in items]})


def _without_other_skills(
    result: Any, names: Sequence[str], folder: Path | None
) -> Any:
    """Drop the skill folders of a listing that are not configured."""

    keep = set(names)

    def shown(item: Any) -> bool:
        if not isinstance(item, dict) or not item.get("is_dir"):
            return True
        name = PurePosixPath(str(item.get("path", "")).rstrip("/")).name
        if name in keep or folder is None:
            return True
        return not (folder / name / SKILL_FILE).exists()

    if isinstance(result, list):
        return [item for item in result if shown(item)]
    items = getattr(result, "entries", None)
    if items is None:
        return result
    return dataclasses.replace(result, entries=[item for item in items if shown(item)])


def _listing_backend(
    names: Sequence[str],
    inner: Any = None,
    offset: str = "",
    folder: Path | None = None,
) -> Any:
    """A source folder whose listing shows only the configured skills.

    When the folder is, or lies inside, another routed skill, ``inner`` is
    that skill's backend, ``offset`` the folder's path in it, and ``folder``
    the folder on disk. The listing then shows the folder's own entries
    except skill folders that were not configured, and every other operation
    goes to ``inner``, so the containing skill's files stay reachable.
    Otherwise the folder holds only the configured skills.
    """

    from deepagents.backends.protocol import BackendProtocol
    from deepagents.backends.protocol import EditResult
    from deepagents.backends.protocol import FileDownloadResponse
    from deepagents.backends.protocol import FileUploadResponse
    from deepagents.backends.protocol import GlobResult
    from deepagents.backends.protocol import GrepResult
    from deepagents.backends.protocol import LsResult
    from deepagents.backends.protocol import ReadResult
    from deepagents.backends.protocol import WriteResult

    entries = [
        {"path": f"/{name}/", "is_dir": True, "size": 0, "modified_at": ""}
        for name in sorted(names)
    ]

    class ListingBackend(BackendProtocol):
        def ls(self, path: str) -> Any:
            if path.rstrip("/") == "" and inner is not None:
                result = _with_paths(inner.ls(offset or "/"), "entries", offset)
                return _without_other_skills(result, names, folder)
            if path.rstrip("/") == "":
                return LsResult(entries=[dict(entry) for entry in entries])
            if inner is None:
                return LsResult(error=f"Directory '{path}' not found")
            return _with_paths(inner.ls(offset + path), "entries", offset)

        def read(self, file_path: str, *args: Any, **kwargs: Any) -> Any:
            if inner is None:
                return ReadResult(error=f"File '{file_path}' not found")
            return inner.read(offset + file_path, *args, **kwargs)

        def write(self, file_path: str, content: str) -> Any:
            if inner is None:
                return WriteResult(error=f"Cannot write '{file_path}'")
            return inner.write(offset + file_path, content)

        def edit(self, file_path: str, *args: Any, **kwargs: Any) -> Any:
            if inner is None:
                return EditResult(error=f"File '{file_path}' not found")
            return inner.edit(offset + file_path, *args, **kwargs)

        def download_files(self, paths: list[str]) -> list[Any]:
            if inner is None:
                return [
                    FileDownloadResponse(path=path, error="file_not_found")
                    for path in paths
                ]
            responses = inner.download_files([offset + path for path in paths])
            return [
                dataclasses.replace(response, path=path)
                for path, response in zip(paths, responses, strict=True)
            ]

        def upload_files(self, files: list[tuple[str, bytes]]) -> list[Any]:
            if inner is None:
                return [
                    FileUploadResponse(path=path, error="permission_denied")
                    for path, _ in files
                ]
            responses = inner.upload_files(
                [(offset + path, content) for path, content in files]
            )
            return [
                dataclasses.replace(response, path=path)
                for (path, _), response in zip(files, responses, strict=True)
            ]

        def grep(
            self,
            pattern: str,
            path: str | None = None,
            glob: str | None = None,
            **kwargs: Any,
        ) -> Any:
            if inner is None:
                return GrepResult(matches=[])
            result = inner.grep(pattern, offset + (path or "/"), glob, **kwargs)
            return _with_paths(result, "matches", offset)

        def glob(self, pattern: str, path: str | None = None) -> Any:
            if inner is None:
                return GlobResult(matches=[])
            result = inner.glob(pattern, offset + (path or "/"))
            return _with_paths(result, "matches", offset)

    return ListingBackend()


def _offset_in(root: Path, route: str, source: str) -> str | None:
    """Return ``source``'s path inside the skill folder ``root`` routed at ``route``.

    The skill's backend reports resolved paths, so the offset is taken
    between resolved folders. ``None`` when the source resolves outside it.
    """

    try:
        inside = (root / source[len(route) :]).resolve()
        relative = inside.relative_to(Path(root).resolve())
    except (OSError, RuntimeError, ValueError):
        return None
    return "/" + relative.as_posix() if relative.parts else ""


def mounted_backend(default: Any, mounts: SkillMounts) -> Any:
    """Route each source and each of its skills; ``default`` serves the rest.

    Every skill is routed at its own full path and every source at its own
    path, so the longest matching route always reaches the right folder, even
    when one entry's path lies inside another's skill. Returns ``default``
    unchanged when nothing needs routing.
    """

    if not mounts.routes:
        return default
    from deepagents.backends import CompositeBackend
    from deepagents.backends import FilesystemBackend
    from deepagents.backends import StateBackend

    skill_routes: dict[str, Any] = {}
    skill_folders: dict[str, Path] = {}
    for source, skills in mounts.routes.items():
        for name, folder in skills.items():
            skill_routes[f"{source}{name}/"] = FilesystemBackend(
                root_dir=str(folder), virtual_mode=True
            )
            skill_folders[f"{source}{name}/"] = folder
    routes = dict(skill_routes)
    for source, skills in mounts.routes.items():
        # A source that is, or lies inside, a routed skill's folder is listed
        # from that folder, without its other skills.
        containing = max(
            (route for route in skill_routes if source.startswith(route)),
            key=len,
            default=None,
        )
        offset = (
            None
            if containing is None
            else _offset_in(skill_folders[containing], containing, source)
        )
        if containing is None or offset is None:
            routes[source] = _listing_backend(list(skills))
        else:
            routes[source] = _listing_backend(
                list(skills),
                skill_routes[containing],
                offset,
                Path(skill_folders[containing]).resolve() / offset.lstrip("/"),
            )
    return CompositeBackend(
        default=default if default is not None else StateBackend(),
        routes=routes,
    )
