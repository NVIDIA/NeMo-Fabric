# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check and discover ``skills.directories`` entries in the task environment.

Adapters call :func:`load_directories` at start, where the configured folders
are mounted. Each entry names one skill (``skill``) or a folder of skills
(``folder``):

1. A path that does not exist fails with ``not_found``. A path that is not a
   folder or cannot be accessed fails with ``invalid``.
2. A ``skill`` entry must hold a readable ``SKILL.md``; a ``folder`` entry
   must be listable. Otherwise the entry fails with ``invalid``.
3. Skills inside a ``folder`` entry are discovered according to the adapter's
   :class:`DiscoveryRules`, which may differ per entry. A candidate is a
   folder holding ``SKILL.md``. Candidates that fail the parsing rules are
   skipped and recorded.
4. An empty ``folder`` entry is valid and recorded in ``empty_entries``.
5. The same folder reached through several entries loads once.
6. Skills that share a name are recorded, or fail or are resolved according
   to the configured duplicate policy.

A skill's name is the ``name`` field of its ``SKILL.md`` front matter, or its
folder name when the front matter has none and the rules allow the fallback.
The module reads only front matter and never reports file contents.
"""

from __future__ import annotations

import os
import re
import stat
import sys
from collections import deque
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from types import MappingProxyType
from typing import Any
from typing import Literal
from typing import get_args

from nemo_fabric_adapter_contract.codec import ContractValidationError
from nemo_fabric_adapter_contract.models import SkillDirectory
from nemo_fabric_adapter_contract.models import SkillDuplicatePolicy

SKILL_FILE = "SKILL.md"

SkillFailure = Literal["not_found", "invalid", "duplicate"]
DuplicatePolicy = SkillDuplicatePolicy
"""Alias of :data:`nemo_fabric_adapter_contract.models.SkillDuplicatePolicy`."""
SkillEntryKind = Literal["skill", "folder"]
SkillOrigin = Literal["external", "bundled", "local"]
DuplicateOutcome = Literal["selected", "shadowed", "ambiguous"]

# A top-level ``key: value`` line. The key may be quoted or followed by spaces.
_KEY = re.compile(r"([^\s#:\-?][^:]*?)[ \t]*:(?:[ \t]+(.*?))?[ \t]*")
_COMMENT = re.compile(r"[ \t]+#")
# Plain scalars that YAML reads as null.
_NULLS = frozenset({"", "~", "null", "Null", "NULL"})
_DOUBLE_QUOTED_ESCAPES = {
    "0": "\0",
    "a": "\a",
    "b": "\b",
    "t": "\t",
    "\t": "\t",
    "n": "\n",
    "v": "\v",
    "f": "\f",
    "r": "\r",
    "e": "\x1b",
    " ": " ",
    '"': '"',
    "/": "/",
    "\\": "\\",
    "N": "\x85",
    "_": "\xa0",
    "L": " ",
    "P": " ",
}
_HEX_ESCAPE_LENGTHS = {"x": 2, "u": 4, "U": 8}


@dataclass(frozen=True)
class DiscoveryRules:
    """How an adapter's harness discovers skills inside a ``folder`` entry."""

    traversal: Literal["immediate_children", "recursive"]
    """Search only the folder's direct children, or every level below it."""
    nested_skills: Literal["include", "stop_at_skill"]
    """Whether discovery continues below a folder that is already a skill."""
    max_depth: int | None
    """Deepest candidate level, or ``None`` for no fixed limit. The configured
    folder is depth 0 and its direct children are depth 1."""
    required_frontmatter_fields: tuple[str, ...] = ()
    """Top-level front matter keys a candidate must define to load."""
    name_fallback: bool = True
    """Use the folder name when the front matter has no name. When false, a
    skill without a front-matter name is skipped."""
    excluded_dir_names: frozenset[str] = frozenset({".git", "node_modules"})
    """Folder names discovery never enters."""
    folder_root_is_skill: bool = False
    """Treat a ``folder`` entry whose own folder holds ``SKILL.md`` as a skill
    candidate at depth 0, as harnesses that scan from the folder itself do.
    Skills below it are discovered as usual, except that ``stop_at_skill``
    stops at the root."""
    support_dir_names: frozenset[str] = frozenset()
    """Names of a skill's support folders (such as ``scripts``). Discovery
    does not enter a direct child with one of these names of a folder that
    holds ``SKILL.md``."""
    allow_symlink_escape: bool = True
    """When false, a candidate whose resolved folder is outside the entry's
    resolved folder does not load and is skipped with reason ``path_escape``."""
    subroot_depths: Mapping[str, int] = dataclass_field(
        default_factory=lambda: MappingProxyType({}), hash=False
    )
    """Direct children of a ``folder`` entry's folder, by name, that restart
    depth counting: the child's own children are at depth 1 again, down to
    the mapped maximum depth. ``max_depth`` applies everywhere else."""

    def __post_init__(self) -> None:
        if self.traversal not in ("immediate_children", "recursive"):
            raise ValueError(f"unknown skill traversal: {self.traversal!r}")
        if self.nested_skills not in ("include", "stop_at_skill"):
            raise ValueError(f"unknown nested skill rule: {self.nested_skills!r}")
        if not _is_depth(self.max_depth, allow_none=True):
            raise ValueError("max_depth must be None or an integer of at least 1")
        if not isinstance(self.subroot_depths, Mapping):
            raise ValueError("subroot_depths must be a mapping of names to depths")
        for name, depth in self.subroot_depths.items():
            if not isinstance(name, str) or not name or not _is_depth(depth):
                raise ValueError(
                    "subroot_depths must map non-empty folder names to integers "
                    "of at least 1"
                )
        # Detach the mapping so that the rules stay immutable.
        object.__setattr__(
            self, "subroot_depths", MappingProxyType(dict(self.subroot_depths))
        )


RulesForEntry = Callable[[int, SkillEntryKind, Path], DiscoveryRules]
"""Return the rules for one entry from its index, kind, and resolved folder."""


def _is_depth(value: object, *, allow_none: bool = False) -> bool:
    if value is None:
        return allow_none
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


@dataclass(frozen=True)
class FoundSkill:
    """One skill discovered through ``skills.directories``."""

    name: str
    """Name from the skill's front matter, or its folder name."""
    path: Path
    """Resolved folder that holds the skill's ``SKILL.md``."""
    field: str
    """The entry that supplied the skill, such as ``skills.directories[2]``."""
    kind: Literal["skill", "folder"]
    """The kind of entry that supplied the skill."""


@dataclass(frozen=True)
class SkippedCandidate:
    """A folder holding ``SKILL.md`` that does not load."""

    path: Path
    """Resolved candidate folder."""
    field: str
    """The entry the candidate was found through."""
    reason: str
    """Why it was skipped: ``unreadable``, ``no_name``, ``path_escape``, or
    ``missing_frontmatter_field:<field>``."""


@dataclass(frozen=True)
class DuplicateRecord:
    """Skills that share a name, and how the clash resolves."""

    name: str
    copies: tuple[tuple[str, Path, str], ...]
    """``(origin, path, field)`` per copy; origin is ``external``,
    ``bundled``, or ``local``; field is empty for copies outside
    ``skills.directories``."""
    outcome: DuplicateOutcome
    """Outcome for the configured copies: one of them is ``selected``, they
    are ``shadowed`` by another source, or the result is ``ambiguous``."""
    selected: Path | None
    """The copy that loads, when known."""


@dataclass(frozen=True)
class DirectoryLoad:
    """The result of checking and discovering ``skills.directories``."""

    selected: list[FoundSkill] = dataclass_field(default_factory=list)
    """Skills that load, in entry order."""
    entries: list[tuple[str, str, Path]] = dataclass_field(default_factory=list)
    """``(field, kind, resolved path)`` per entry."""
    empty_entries: list[str] = dataclass_field(default_factory=list)
    """Fields of ``folder`` entries that found no skills."""
    skipped: list[SkippedCandidate] = dataclass_field(default_factory=list)
    duplicates: list[DuplicateRecord] = dataclass_field(default_factory=list)


class SkillLoadError(ValueError):
    """A ``skills.directories`` entry that cannot be loaded."""

    def __init__(
        self,
        reason: SkillFailure,
        message: str,
        *,
        field: str,
        paths: tuple[Path, ...],
    ) -> None:
        super().__init__(message)
        self.reason: SkillFailure = reason
        self.field = field
        self.paths = paths


@dataclass(frozen=True)
class _Candidate:
    reached: Path
    resolved: Path
    identity: tuple[int, int] | Path


def load_directories(
    directories: Sequence[Mapping[str, str | os.PathLike[str]]]
    | Sequence[SkillDirectory],
    *,
    rules: DiscoveryRules | RulesForEntry,
    duplicates: DuplicatePolicy | None,
    base_dir: str | os.PathLike[str] | None = None,
) -> DirectoryLoad:
    """Check ``directories`` and return the skills they supply.

    Relative entries resolve against ``base_dir``, or the current directory
    when it is ``None``. ``~`` and environment variables are not expanded.

    ``rules`` is either one :class:`DiscoveryRules` for every entry, or a
    callable that receives each entry's index, kind (``"skill"`` or
    ``"folder"``), and resolved folder after the entry passes its checks, and
    returns the rules for that entry.

    A folder reached through several entries loads once. Under ``last_wins``
    it loads at the position of, and is attributed to, the last entry that
    reaches it; otherwise it keeps its first position.

    Raises:
        SkillLoadError: If an entry is malformed, missing, not a folder, not
            accessible, or cannot be read, or if skills share a name under an
            explicit ``duplicates`` policy.
    """

    if duplicates is not None and duplicates not in get_args(SkillDuplicatePolicy):
        raise ValueError(f"unknown skill duplicate policy: {duplicates!r}")
    if not isinstance(rules, DiscoveryRules) and not callable(rules):
        raise TypeError("rules must be DiscoveryRules or a callable returning them")
    load = DirectoryLoad()
    found: list[FoundSkill | None] = []
    # Position in ``found`` of each folder that loads, by identity.
    loaded: dict[tuple[int, int] | Path, int] = {}
    for index, value in enumerate(directories):
        entry_field = f"skills.directories[{index}]"
        directory = _entry(value, entry_field)
        kind = directory.kind
        assert kind is not None
        key_field = f"{entry_field}.{kind}"
        configured = Path(os.fspath(directory.path))
        if base_dir is not None:
            configured = Path(base_dir) / configured
        root, root_identity = _check_entry(configured, kind, key_field)
        load.entries.append((entry_field, kind, root))
        entry_rules = _entry_rules(rules, index, kind, root)

        reached = Path(os.path.normpath(configured))
        if kind == "skill":
            candidates = [_Candidate(reached, root, root_identity)]
            if (
                entry_rules.traversal == "recursive"
                and entry_rules.nested_skills == "include"
            ):
                candidates += _discover(
                    root, root_identity, entry_rules, reached, key_field, kind=kind
                )
        else:
            candidates = _discover(
                root, root_identity, entry_rules, reached, key_field, kind=kind
            )

        entry_skills = 0
        for candidate in candidates:
            if not entry_rules.allow_symlink_escape and not _within(
                candidate.resolved, root
            ):
                load.skipped.append(
                    SkippedCandidate(candidate.resolved, entry_field, "path_escape")
                )
                continue
            name = _candidate_name(
                candidate,
                entry_rules,
                entry_field,
                load.skipped,
                required=kind == "skill" and candidate.resolved == root,
                key_field=key_field,
            )
            if name is None:
                continue
            entry_skills += 1
            earlier = loaded.get(candidate.identity)
            if earlier is not None:
                if duplicates != "last_wins":
                    continue
                # The later entry takes the folder over.
                found[earlier] = None
            loaded[candidate.identity] = len(found)
            found.append(
                FoundSkill(
                    name=name, path=candidate.resolved, field=entry_field, kind=kind
                )
            )
        if kind == "folder" and not entry_skills:
            load.empty_entries.append(entry_field)

    load.selected.extend(
        _resolve_duplicates(
            [skill for skill in found if skill is not None],
            duplicates,
            load.duplicates,
        )
    )
    return load


def _entry_rules(
    rules: DiscoveryRules | RulesForEntry,
    index: int,
    kind: SkillEntryKind,
    root: Path,
) -> DiscoveryRules:
    if isinstance(rules, DiscoveryRules):
        return rules
    entry_rules = rules(index, kind, root)
    if not isinstance(entry_rules, DiscoveryRules):
        raise TypeError(
            f"rules for skills.directories[{index}] must be DiscoveryRules, "
            f"not {type(entry_rules).__name__}"
        )
    return entry_rules


def report_block(
    load: DirectoryLoad, *, bundled: Mapping[str, Any] | None = None
) -> dict[str, Any] | None:
    """Return the ``output.skills`` report for ``load``, or ``None`` when empty.

    Empty parts are omitted. ``bundled`` is the adapter's bundled-skill report
    and is copied as given. The report never includes file contents.
    """

    block: dict[str, Any] = {}
    if load.selected:
        block["loaded"] = [
            {
                "name": skill.name,
                "path": str(skill.path),
                "entry": skill.field,
                "origin": "external",
            }
            for skill in load.selected
        ]
    if load.empty_entries:
        block["empty_entries"] = list(load.empty_entries)
    if load.skipped:
        block["skipped"] = [
            {"path": str(item.path), "entry": item.field, "reason": item.reason}
            for item in load.skipped
        ]
    if load.duplicates:
        block["duplicates"] = [_duplicate_report(item) for item in load.duplicates]
    if bundled:
        block["bundled"] = dict(bundled)
    return block or None


def warn(adapter: str, load: DirectoryLoad) -> None:
    """Write one stderr line per skipped candidate, empty entry, and duplicate."""

    lines = [
        f"{adapter}: skipped skill {item.path} from {item.field}: {item.reason}"
        for item in load.skipped
    ]
    lines += [f"{adapter}: {field} contains no skills" for field in load.empty_entries]
    for item in load.duplicates:
        copies = ", ".join(
            f"{path} ({origin}{f', {field}' if field else ''})"
            for origin, path, field in item.copies
        )
        selected = f"; {item.selected} loads" if item.selected is not None else ""
        lines.append(
            f"{adapter}: skill {item.name!r} has {len(item.copies)} copies, "
            f"{item.outcome}{selected}: {copies}"
        )
    for line in lines:
        print(line, file=sys.stderr, flush=True)


def _entry(
    value: SkillDirectory | Mapping[str, str | os.PathLike[str]], field: str
) -> SkillDirectory:
    if isinstance(value, SkillDirectory):
        return value
    if not isinstance(value, Mapping):
        raise SkillLoadError(
            "invalid",
            f"NeMo Fabric skill entry {field} must be an object with skill or folder",
            field=field,
            paths=(),
        )
    try:
        return SkillDirectory.from_mapping(
            {
                key: os.fspath(item) if isinstance(item, os.PathLike) else item
                for key, item in value.items()
            }
        )
    except ContractValidationError as error:
        location = ".".join((field, *error.path))
        raise SkillLoadError(
            "invalid",
            f"NeMo Fabric skill entry {location}: {error.message}",
            field=location,
            paths=(),
        ) from None


def _check_entry(
    configured: Path, kind: str, field: str
) -> tuple[Path, tuple[int, int] | Path]:
    """Return the entry's resolved folder and its identity."""

    try:
        status = os.stat(configured)
    except (FileNotFoundError, NotADirectoryError):
        raise SkillLoadError(
            "not_found",
            f"NeMo Fabric skill entry {field} does not exist: {configured}",
            field=field,
            paths=(configured,),
        ) from None
    except OSError:
        raise _not_accessible(field, configured) from None
    try:
        entry = configured.resolve(strict=True)
        identity = _identity(entry)
    except (OSError, RuntimeError):
        raise _not_accessible(field, configured) from None
    if not stat.S_ISDIR(status.st_mode):
        raise SkillLoadError(
            "invalid",
            f"NeMo Fabric skill entry {field} is not a folder: {configured}",
            field=field,
            paths=(configured,),
        )
    if kind == "skill":
        skill_file = entry / SKILL_FILE
        try:
            skill_status = os.stat(skill_file)
        except (FileNotFoundError, NotADirectoryError):
            skill_status = None
        except OSError:
            # The folder exists but cannot be searched for SKILL.md.
            raise _not_accessible(field, configured) from None
        if skill_status is None or not stat.S_ISREG(skill_status.st_mode):
            raise SkillLoadError(
                "invalid",
                f"NeMo Fabric skill entry {field} has no {SKILL_FILE}: {configured}",
                field=field,
                paths=(configured,),
            )
        try:
            with skill_file.open("rb"):
                pass
        except OSError:
            raise SkillLoadError(
                "invalid",
                f"NeMo Fabric skill entry {field} has a {SKILL_FILE} that cannot "
                f"be read: {skill_file}",
                field=field,
                paths=(skill_file,),
            ) from None
    else:
        try:
            with os.scandir(entry):
                pass
        except OSError:
            raise SkillLoadError(
                "invalid",
                f"NeMo Fabric skill entry {field} cannot be listed: {configured}",
                field=field,
                paths=(configured,),
            ) from None
    return entry, identity


def _not_accessible(field: str, configured: Path) -> SkillLoadError:
    return SkillLoadError(
        "invalid",
        f"NeMo Fabric skill entry {field} is not accessible: {configured}",
        field=field,
        paths=(configured,),
    )


@dataclass(frozen=True)
class _Visit:
    folder: _Candidate
    level: int
    """Levels below the entry's folder."""
    depth: int
    """Depth counted against ``limit``; a subroot restarts it."""
    limit: int | None
    order: tuple[str, ...]
    """Folder names from the entry's folder, for name-ordered output."""


def _discover(
    root: Path,
    identity: tuple[int, int] | Path,
    rules: DiscoveryRules,
    reached: Path,
    field: str,
    *,
    kind: SkillEntryKind,
) -> list[_Candidate]:
    """Return candidate folders below ``root`` in name order, parents first.

    The walk is breadth first, so each folder is claimed by its shallowest
    path; a symlink reached deeper never hides the real folder's subtree.
    Each folder is entered once, so symlinked folders cannot loop. The root
    is a candidate only for a ``folder`` entry with ``folder_root_is_skill``.
    """

    candidates: list[tuple[tuple[str, ...], _Candidate]] = []
    visited = {identity}
    # Walk the configured spelling so that names follow what the harness sees.
    queue = deque(
        [_Visit(_Candidate(reached, root, identity), 0, 0, rules.max_depth, ())]
    )
    while queue:
        visit = queue.popleft()
        current = visit.folder
        is_skill = _is_file(current.reached / SKILL_FILE)
        if visit.level:
            if is_skill:
                candidates.append((visit.order, current))
            if (
                rules.traversal != "recursive"
                or (is_skill and rules.nested_skills == "stop_at_skill")
                or (visit.limit is not None and visit.depth >= visit.limit)
            ):
                continue
        elif is_skill and kind == "folder" and rules.folder_root_is_skill:
            candidates.append((visit.order, current))
            if rules.nested_skills == "stop_at_skill":
                continue
        try:
            with os.scandir(current.reached) as entries:
                children = sorted(entries, key=lambda child: child.name)
        except OSError:
            if not visit.level:
                raise SkillLoadError(
                    "invalid",
                    f"NeMo Fabric skill entry {field} cannot be listed: {reached}",
                    field=field,
                    paths=(reached,),
                ) from None
            # A folder that cannot be read, such as a volume's lost+found,
            # holds no skills.
            continue
        for child in children:
            if child.name in rules.excluded_dir_names or (
                is_skill and child.name in rules.support_dir_names
            ):
                continue
            try:
                if not child.is_dir():
                    continue
                path = Path(child.path)
                child_identity = _identity(path)
                resolved = path.resolve()
            except (OSError, RuntimeError):
                continue
            if child_identity in visited:
                continue
            visited.add(child_identity)
            depth, limit = visit.depth + 1, visit.limit
            if not visit.level and kind == "folder":
                subroot_limit = rules.subroot_depths.get(child.name)
                if subroot_limit is not None:
                    depth, limit = 0, subroot_limit
            queue.append(
                _Visit(
                    _Candidate(path, resolved, child_identity),
                    visit.level + 1,
                    depth,
                    limit,
                    (*visit.order, child.name),
                )
            )
    # Name order with each folder before its subfolders, as a depth-first
    # walk in name order would list them.
    candidates.sort(key=lambda item: item[0])
    return [candidate for _, candidate in candidates]


def _candidate_name(
    candidate: _Candidate,
    rules: DiscoveryRules,
    field: str,
    skipped: list[SkippedCandidate],
    *,
    required: bool,
    key_field: str,
) -> str | None:
    skill_file = candidate.resolved / SKILL_FILE
    try:
        with skill_file.open(encoding="utf-8-sig", errors="replace") as lines:
            front_matter = _front_matter(lines)
    except OSError:
        if required:
            raise SkillLoadError(
                "invalid",
                f"NeMo Fabric skill entry {key_field} has a {SKILL_FILE} that "
                f"cannot be read: {skill_file}",
                field=key_field,
                paths=(skill_file,),
            ) from None
        skipped.append(SkippedCandidate(candidate.resolved, field, "unreadable"))
        return None
    keys = front_matter[1] if front_matter is not None else frozenset()
    for name in rules.required_frontmatter_fields:
        if name not in keys:
            skipped.append(
                SkippedCandidate(
                    candidate.resolved, field, f"missing_frontmatter_field:{name}"
                )
            )
            return None
    name = front_matter[0] if front_matter is not None else None
    if not name and rules.name_fallback:
        name = candidate.reached.name or candidate.resolved.name
    if not name:
        skipped.append(SkippedCandidate(candidate.resolved, field, "no_name"))
        return None
    return name


def _resolve_duplicates(
    found: list[FoundSkill],
    policy: DuplicatePolicy | None,
    records: list[DuplicateRecord],
) -> list[FoundSkill]:
    by_name: dict[str, list[FoundSkill]] = {}
    for skill in found:
        by_name.setdefault(skill.name, []).append(skill)
    shadowed: set[int] = set()
    for name, copies in by_name.items():
        if len(copies) < 2:
            continue
        record_copies = tuple(("external", skill.path, skill.field) for skill in copies)
        if policy is None:
            records.append(DuplicateRecord(name, record_copies, "ambiguous", None))
            continue
        seen: dict[str, FoundSkill] = {}
        for skill in copies:
            earlier = seen.get(skill.field)
            if earlier is not None:
                raise SkillLoadError(
                    "duplicate",
                    f"NeMo Fabric skill {name!r} appears twice in {skill.field}: "
                    f"{earlier.path} and {skill.path}",
                    field=skill.field,
                    paths=(earlier.path, skill.path),
                )
            seen[skill.field] = skill
        if policy == "reject":
            first, second = copies[0], copies[1]
            raise SkillLoadError(
                "duplicate",
                f"NeMo Fabric skill {name!r} is configured twice: "
                f"{first.path} ({first.field}) and {second.path} ({second.field})",
                field=second.field,
                paths=(first.path, second.path),
            )
        winner = copies[0] if policy == "first_wins" else copies[-1]
        shadowed.update(id(skill) for skill in copies if skill is not winner)
        records.append(DuplicateRecord(name, record_copies, "selected", winner.path))
    return [skill for skill in found if id(skill) not in shadowed]


def _duplicate_report(item: DuplicateRecord) -> dict[str, Any]:
    copies = []
    for origin, path, field in item.copies:
        copy: dict[str, Any] = {"origin": origin, "path": str(path)}
        if field:
            copy["entry"] = field
        copies.append(copy)
    report: dict[str, Any] = {"name": item.name, "outcome": item.outcome}
    if item.selected is not None:
        report["selected"] = str(item.selected)
    report["copies"] = copies
    return report


def _is_file(path: Path) -> bool:
    # Path.is_file() raises PermissionError below an untraversable folder on
    # Python 3.13 and earlier; a file that cannot be reached is not a skill.
    try:
        return stat.S_ISREG(os.stat(path).st_mode)
    except (OSError, ValueError):
        return False


def _identity(directory: Path) -> tuple[int, int] | Path:
    # Compare folders by identity so that spellings resolve() keeps apart, such
    # as a different case on a case-insensitive file system, still match. An
    # inode of 0 identifies nothing, so fall back to the resolved path.
    status = directory.stat()
    return (status.st_dev, status.st_ino) if status.st_ino else directory.resolve()


def _front_matter(lines: Iterable[str]) -> tuple[str | None, frozenset[str]] | None:
    """Return the ``name`` scalar and top-level keys of a ``---`` front matter block."""

    lines = iter(lines)
    if next(lines, "").rstrip() != "---":
        return None
    name: str | None = None
    keys: set[str] = set()
    for line in lines:
        line = line.rstrip()
        if line == "---":
            return name, frozenset(keys)
        match = _KEY.fullmatch(line)
        if not match:
            continue
        key = _unquote(match.group(1))
        keys.add(key)
        if key == "name" and not name:
            name = _scalar(match.group(2) or "")
    # Without its closing line the block is not front matter.
    return None


def _unquote(key: str) -> str:
    if len(key) >= 2 and key[0] == key[-1] and key[0] in ("'", '"'):
        quoted = _quoted(key)
        if quoted is not None and quoted[1] == len(key):
            return quoted[0]
        return key[1:-1]
    return key


def _scalar(value: str) -> str | None:
    """Return a one-line YAML scalar as a name, or ``None`` when it has none.

    Null, empty, block (``|``, ``>``), and flow collection (``[``, ``{``)
    values give no usable name.
    """

    if value[:1] in ("'", '"'):
        quoted = _quoted(value)
        if quoted is None:
            # Not a closed one-line quoted scalar; keep the text as written.
            return value
        text, end = quoted
        rest = value[end:].strip()
        if rest and not rest.startswith("#"):
            return value
        return text or None
    if value[:1] in ("|", ">", "[", "{"):
        return None
    text = "" if value.startswith("#") else _COMMENT.split(value, maxsplit=1)[0]
    return None if text in _NULLS else text


def _quoted(value: str) -> tuple[str, int] | None:
    """Decode the quoted scalar at the start of ``value``.

    Returns the text and the index after the closing quote, or ``None`` when
    the scalar is not closed on this line or has an invalid escape.
    """

    quote = value[0]
    text: list[str] = []
    index = 1
    while index < len(value):
        char = value[index]
        if char == quote:
            if quote == "'" and value[index + 1 : index + 2] == "'":
                text.append("'")
                index += 2
                continue
            return "".join(text), index + 1
        if char == "\\" and quote == '"':
            escape = value[index + 1 : index + 2]
            if escape in _DOUBLE_QUOTED_ESCAPES:
                text.append(_DOUBLE_QUOTED_ESCAPES[escape])
                index += 2
                continue
            length = _HEX_ESCAPE_LENGTHS.get(escape)
            digits = value[index + 2 : index + 2 + length] if length else ""
            if not length or len(digits) != length:
                return None
            try:
                text.append(chr(int(digits, 16)))
            except ValueError:
                return None
            index += 2 + length
            continue
        text.append(char)
        index += 1
    return None


def _within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)
