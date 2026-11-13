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

A skill's name is the ``name`` field of its ``SKILL.md`` front matter, read as
the harness reads it (see :data:`NameStyle`), or its folder name when the
front matter has none and the rules allow the fallback. Under the default
``pyyaml`` style, a value that YAML reads as anything but a non-empty string,
such as null or a number, counts as no name.
The module reads only front matter and never reports file contents.
"""

from __future__ import annotations

import json
import math
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
NameStyle = Literal["pyyaml", "openclaw"]
"""How a harness turns the front matter ``name`` value into a skill name.

``pyyaml``
    As PyYAML (YAML 1.1) reads it, as Hermes does: only a non-empty string
    is a name.
``openclaw``
    As OpenClaw reads it: YAML 1.2 core schema, then strings are trimmed,
    numbers and booleans become their JavaScript text, sequences and mappings
    become JSON, and a null takes the text on the key's line.
"""

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
_HEX_DIGITS = "0123456789ABCDEFabcdef"
# What may follow an indicator such as ``:`` or ``-`` for it to count.
_BLANK = ("", " ", "\t", "\n")
# A block scalar header: style, then chomping and indentation indicators in
# either order, then an optional comment.
_BLOCK_HEADER = re.compile(
    r"[|>](?:([+-])([1-9])?|([1-9])([+-])?)?(?: +(?:#[^\n]*)?)?(?=\n|\Z)"
)
# Tags that make a scalar a string.
_STRING_TAGS = frozenset({"!!str", "!<tag:yaml.org,2002:str>"})
# Plain scalars that PyYAML reads as null, a boolean, an integer, a float, a
# timestamp, or a merge or value key: anything but a string.
_NON_STRING_PLAIN = re.compile(
    r"""
    ~|null|Null|NULL
    |yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF
    |[-+]?0b[0-1_]+|[-+]?0[0-7_]+|[-+]?(?:0|[1-9][0-9_]*)|[-+]?0x[0-9a-fA-F_]+
    |[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+
    |[-+]?[0-9][0-9_]*\.[0-9_]*(?:[eE][-+][0-9]+)?|\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
    |[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN)
    |[0-9]{4}-[0-9]{2}-[0-9]{2}
    |[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}(?:[Tt]|[ \t]+)[0-9]{1,2}:[0-9]{2}:[0-9]{2}
     (?:\.[0-9]*)?(?:[ \t]*(?:Z|[-+][0-9]{1,2}(?::[0-9]{2})?))?
    |<<|=
    """,
    re.VERBOSE,
)


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
    name_style: NameStyle = "pyyaml"
    """How the front matter ``name`` value becomes the skill name; see
    :data:`NameStyle`."""

    def __post_init__(self) -> None:
        if self.traversal not in ("immediate_children", "recursive"):
            raise ValueError(f"unknown skill traversal: {self.traversal!r}")
        if self.nested_skills not in ("include", "stop_at_skill"):
            raise ValueError(f"unknown nested skill rule: {self.nested_skills!r}")
        if not _is_depth(self.max_depth, allow_none=True):
            raise ValueError("max_depth must be None or an integer of at least 1")
        if self.name_style not in get_args(NameStyle):
            raise ValueError(f"unknown skill name style: {self.name_style!r}")
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

        # Keep ``..`` as configured: after a symlink it leads where the file
        # system says, which a lexical normpath would change.
        reached = configured
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

        named: list[tuple[str, _Candidate]] = []
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
            if name is not None:
                named.append((name, candidate))
        if kind == "folder" and not named:
            load.empty_entries.append(entry_field)
        if duplicates is not None:
            # Check the entry's own skills before another entry can take
            # one of them over.
            _reject_duplicates_in_entry(named, entry_field)
        for name, candidate in named:
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

    load.selected.extend(
        _resolve_duplicates(
            [skill for skill in found if skill is not None],
            duplicates,
            load.duplicates,
        )
    )
    return load


def _reject_duplicates_in_entry(
    named: list[tuple[str, _Candidate]], field: str
) -> None:
    """Fail when two skills found through one entry share a name."""

    seen: dict[str, Path] = {}
    for name, candidate in named:
        earlier = seen.get(name)
        if earlier is not None:
            raise SkillLoadError(
                "duplicate",
                f"NeMo Fabric skill {name!r} appears twice in {field}: "
                f"{earlier} and {candidate.resolved}",
                field=field,
                paths=(earlier, candidate.resolved),
            )
        seen[name] = candidate.resolved


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
    A folder is entered again only through a path that leaves it a larger
    depth budget, such as one inside a subroot, so symlinked folders cannot
    loop. The root is a candidate only for a ``folder`` entry with
    ``folder_root_is_skill``.
    """

    candidates: list[tuple[tuple[str, ...], _Candidate]] = []
    claimed = {identity}
    # The largest depth budget each folder has been entered with. The root
    # is never entered again.
    visited: dict[tuple[int, int] | Path, float] = {identity: float("inf")}
    # Walk the configured spelling so that names follow what the harness sees.
    queue = deque(
        [_Visit(_Candidate(reached, root, identity), 0, 0, rules.max_depth, ())]
    )
    while queue:
        visit = queue.popleft()
        current = visit.folder
        is_skill = _is_file(current.reached / SKILL_FILE)
        if visit.level:
            if is_skill and current.identity not in claimed:
                claimed.add(current.identity)
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
            depth, limit = visit.depth + 1, visit.limit
            if not visit.level and kind == "folder":
                subroot_limit = rules.subroot_depths.get(child.name)
                if subroot_limit is not None:
                    depth, limit = 0, subroot_limit
            budget = float("inf") if limit is None else limit - depth
            if visited.get(child_identity, -1) >= budget:
                continue
            visited[child_identity] = budget
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
            front_matter = _front_matter(lines, rules.name_style)
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
        reached_name = candidate.reached.name
        name = (
            reached_name if reached_name not in ("", "..") else candidate.resolved.name
        )
        if rules.name_style == "openclaw":
            # OpenClaw trims the folder name it falls back to.
            name = _js_trim(name)
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


def _front_matter(
    lines: Iterable[str], style: NameStyle = "pyyaml"
) -> tuple[str | None, frozenset[str]] | None:
    """Return the ``name`` value and top-level keys of a ``---`` front matter block."""

    lines = iter(lines)
    if next(lines, "").rstrip() != "---":
        return None
    block: list[str] = []
    for line in lines:
        line = line.rstrip("\r\n")
        if line.rstrip() == "---":
            break
        block.append(line)
    else:
        # Without its closing line the block is not front matter.
        return None
    name: str | None = None
    keys: set[str] = set()
    # Lines up to this index belong to the value of the ``name`` key.
    value_end = 0
    for index, line in enumerate(block):
        if index < value_end:
            continue
        match = _KEY.fullmatch(line.rstrip())
        if not match:
            continue
        key = _unquote(match.group(1))
        if style == "openclaw":
            # OpenClaw trims each key.
            key = _js_trim(key)
        keys.add(key)
        if key == "name":
            # When the key repeats, YAML loaders such as PyYAML keep the last.
            if style == "openclaw":
                name, value_end = _openclaw_name(block, index, match)
            else:
                name, value_end = _name(block, index, match)
    return name, frozenset(keys)


class _NotYaml(Exception):
    """The ``name`` value is not valid YAML."""


def _name(block: list[str], index: int, match: re.Match[str]) -> tuple[str | None, int]:
    """Return the string YAML reads for the ``name`` key on ``block[index]``.

    The value is read as PyYAML reads the front matter: a plain, quoted, or
    block scalar, over several lines where YAML allows it. A value that YAML
    reads as anything but a non-empty string, such as null, a number, a
    boolean, a date, a sequence, or a mapping, gives no name. When the value
    is not valid YAML, the text on the key's line is used instead, as
    harnesses that recover from front matter errors line by line read it.

    Also returns the index of the first line after the value.
    """

    # The text harnesses pass to YAML: the lines between the fences, without
    # a final line break.
    source = "\n".join(block[index:])
    start = source.index(":", match.end(1)) + 1
    try:
        name, end = _node(source, start)
    except _NotYaml:
        return _scalar(match.group(2) or ""), index + 1
    return name, index + 1 + source.count("\n", 0, end)


def _node(source: str, index: int) -> tuple[str | None, int]:
    """Return the string value after ``index``, and the end of a quoted value.

    Only a quoted value can run onto lines that start at column 0, so the end
    is ``index`` for other values.
    """

    position = _next_token(source, index)
    tag: str | None = None
    while position is not None and source[position] in "&!":
        # Node properties: an anchor, or a tag.
        end = position
        while end < len(source) and source[end] not in " \t\n":
            end += 1
        if source[position] == "!":
            tag = source[position:end]
            if tag not in _STRING_TAGS and tag != "!":
                # Any other tag makes the value something other than a string.
                return None, index
        position = _next_token(source, end)
    if position is None:
        return None, index
    char, following = source[position], source[position + 1 : position + 2]
    # PyYAML resolves the type of an untagged plain scalar, and of any scalar
    # with the non-specific tag ``!``, from its text.
    resolve = tag == "!"
    if char in "|>":
        text, end = _block_scalar(source, position)
    elif char in "'\"":
        quoted = _quoted(source, position)
        if quoted is None:
            raise _NotYaml
        text, end = quoted
    elif char in "[{*" or (char in "-?:" and following in _BLANK):
        # A collection, or an alias of another node.
        return None, index
    elif char in ",]}%@`\t":
        raise _NotYaml
    else:
        text, end = _plain(source, position)
        resolve = resolve or tag is None
        if source[end:].lstrip(" ")[:1] == ":" and "\n" in source[:position]:
            # A key on a line below ``name:`` starts a nested mapping.
            return None, index
    if not _only_comments_follow(source, end):
        raise _NotYaml
    if char not in "'\"":
        end = index
    if resolve and _NON_STRING_PLAIN.fullmatch(text):
        return None, end
    return text or None, end


def _next_token(source: str, index: int) -> int | None:
    """Return where the value after ``index`` starts, or ``None`` when empty."""

    while True:
        while source.startswith(" ", index):
            index += 1
        if source.startswith("#", index):
            index = _line_end(source, index)
        if index >= len(source):
            return None
        if source[index] != "\n":
            # A line that starts at column 0 holds the next key.
            return None if _column(source, index) == 0 else index
        index += 1


def _only_comments_follow(source: str, index: int) -> bool:
    """Whether only spaces and comments follow ``index`` up to the next key."""

    while index < len(source):
        if _column(source, index) == 0 and source[index] not in " \t\n#":
            return True
        while source.startswith(" ", index):
            index += 1
        if source.startswith("#", index):
            index = _line_end(source, index)
        if index >= len(source):
            return True
        if source[index] != "\n":
            return False
        index += 1
    return True


def _plain(source: str, index: int, indent: int = 0) -> tuple[str, int]:
    """Read a plain scalar, folding its continuation lines as YAML does.

    Continuation lines must start after column ``indent``.
    """

    chunks: list[str] = []
    spaces: list[str] | None = []
    end = index
    while not source.startswith("#", index):
        stop = index
        while stop < len(source):
            char = source[stop]
            if char in " \t\n" or (
                char == ":" and source[stop + 1 : stop + 2] in _BLANK
            ):
                break
            stop += 1
        if stop == index:
            break
        chunks.extend(spaces or ())
        chunks.append(source[index:stop])
        end = stop
        index, spaces = _plain_spaces(source, stop)
        # Continuation lines must be indented more than the key.
        if not spaces or _column(source, index) <= indent:
            break
    if not chunks:
        raise _NotYaml
    return "".join(chunks), end


def _plain_spaces(source: str, index: int) -> tuple[int, list[str] | None]:
    start = index
    while source.startswith(" ", index):
        index += 1
    if not source.startswith("\n", index):
        return index, [source[start:index]] if index > start else []
    index += 1
    if _document_marker(source, index):
        return index, None
    breaks: list[str] = []
    while source[index : index + 1] in (" ", "\n"):
        index += 1
        if source[index - 1] == "\n":
            breaks.append("\n")
            if _document_marker(source, index):
                return index, None
    # A single line break folds to a space; blank lines keep their breaks.
    return index, breaks or [" "]


def _block_scalar(source: str, index: int) -> tuple[str, int]:
    """Read a literal (``|``) or folded (``>``) block scalar."""

    header = _BLOCK_HEADER.match(source, index)
    if header is None:
        raise _NotYaml
    folded = source[index] == ">"
    chomping = header.group(1) or header.group(4)
    increment = header.group(2) or header.group(3)
    index = min(header.end() + 1, len(source))
    breaks: list[str] = []
    if increment:
        # The key is at column 0, so the indicator is the indentation.
        indent = int(increment)
        breaks, index = _block_breaks(source, index, indent)
    else:
        # The first non-empty line sets the indentation, which is never less
        # than the longest leading line of spaces.
        widest = 0
        while source[index : index + 1] in (" ", "\n"):
            if source[index] == "\n":
                breaks.append("\n")
            index += 1
            widest = max(widest, _column(source, index))
        indent = max(1, widest)
    chunks: list[str] = []
    line_break = ""
    while index < len(source) and _column(source, index) == indent:
        chunks.extend(breaks)
        starts_with_text = source[index] not in " \t"
        end = _line_end(source, index)
        chunks.append(source[index:end])
        line_break = "\n" if end < len(source) else ""
        breaks, index = _block_breaks(source, min(end + 1, len(source)), indent)
        if index >= len(source) or _column(source, index) != indent:
            break
        if folded and line_break and starts_with_text and source[index] not in " \t":
            # Fold the break between two text lines unless blank lines
            # separate them.
            if not breaks:
                chunks.append(" ")
        else:
            chunks.append(line_break)
    if chomping != "-":
        chunks.append(line_break)
    if chomping == "+":
        chunks.extend(breaks)
    return "".join(chunks), index


def _block_breaks(source: str, index: int, indent: int) -> tuple[list[str], int]:
    """Skip indentation and blank lines; return their line breaks and the end."""

    breaks: list[str] = []
    while True:
        while _column(source, index) < indent and source.startswith(" ", index):
            index += 1
        if not source.startswith("\n", index):
            return breaks, index
        breaks.append("\n")
        index += 1


def _document_marker(source: str, index: int) -> bool:
    return source[index : index + 3] in ("---", "...") and (
        source[index + 3 : index + 4] in _BLANK
    )


def _line_end(source: str, index: int) -> int:
    end = source.find("\n", index)
    return len(source) if end < 0 else end


def _column(source: str, index: int) -> int:
    return index - source.rfind("\n", 0, index) - 1


def _unquote(key: str) -> str:
    if len(key) >= 2 and key[0] == key[-1] and key[0] in ("'", '"'):
        quoted = _quoted(key, 0)
        if quoted is not None and quoted[1] == len(key):
            return quoted[0]
        return key[1:-1]
    return key


def _scalar(value: str) -> str | None:
    """Return the text of a ``name`` line whose value is not valid YAML.

    Null, empty, block (``|``, ``>``), and flow collection (``[``, ``{``)
    values give no usable name.
    """

    if value[:1] in ("'", '"'):
        quoted = _quoted(value, 0)
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


def _quoted(source: str, index: int) -> tuple[str, int] | None:
    """Decode the quoted scalar that starts at ``source[index]``.

    Line breaks inside the quotes fold as YAML folds them. Returns the text
    and the index after the closing quote, or ``None`` when the scalar is not
    closed or has an invalid escape.
    """

    quote = source[index]
    double = quote == '"'
    chunks: list[str] = []
    index += 1
    while True:
        # Text up to the next quote, escape, space, or line break.
        stop = index
        while stop < len(source) and source[stop] not in "'\"\\ \t\n":
            stop += 1
        chunks.append(source[index:stop])
        index = stop
        char = source[index : index + 1]
        if char == quote:
            if not double and source.startswith("'", index + 1):
                chunks.append("'")
                index += 2
                continue
            return "".join(chunks), index + 1
        if char in ("'", '"') or (char == "\\" and not double):
            chunks.append(char)
            index += 1
            continue
        if char == "\\":
            escape = source[index + 1 : index + 2]
            if escape in _DOUBLE_QUOTED_ESCAPES:
                chunks.append(_DOUBLE_QUOTED_ESCAPES[escape])
                index += 2
                continue
            if escape == "\n":
                # An escaped line break joins the lines without a space.
                breaks, index = _quoted_breaks(source, index + 2)
                if breaks is None:
                    return None
                chunks.extend(breaks)
                continue
            length = _HEX_ESCAPE_LENGTHS.get(escape)
            digits = source[index + 2 : index + 2 + length] if length else ""
            if not length or len(digits) != length or digits.strip(_HEX_DIGITS):
                return None
            try:
                chunks.append(chr(int(digits, 16)))
            except ValueError:
                return None
            index += 2 + length
            continue
        # Spaces and tabs, then more text or a line break.
        stop = index
        while source[stop : stop + 1] in (" ", "\t"):
            stop += 1
        if stop >= len(source):
            return None
        if source[stop] != "\n":
            chunks.append(source[index:stop])
            index = stop
            continue
        breaks, index = _quoted_breaks(source, stop + 1)
        if breaks is None:
            return None
        # A single line break folds to a space; blank lines keep their breaks.
        chunks.extend(breaks or [" "])


def _quoted_breaks(source: str, index: int) -> tuple[list[str] | None, int]:
    """Skip white space and blank lines at the start of a quoted line."""

    breaks: list[str] = []
    while True:
        if _document_marker(source, index):
            return None, index
        while source[index : index + 1] in (" ", "\t"):
            index += 1
        if not source.startswith("\n", index):
            return breaks, index
        breaks.append("\n")
        index += 1


# OpenClaw names (``name_style="openclaw"``).
#
# OpenClaw parses front matter with the ``yaml`` npm package and the YAML 1.2
# core schema, then turns each top-level value into text: a string is
# trimmed, a number or boolean becomes ``String(value)``, and a sequence or
# mapping becomes ``JSON.stringify(value)``. A null is dropped, and the text
# on the key's line takes its place. ``loadSingleSkillDirectory`` trims the
# name again and uses the trimmed folder name when the result is empty.
# Values are held here as JavaScript holds them: ``None`` for null, ``bool``,
# ``float`` for every number, ``str``, ``list``, and ``dict`` with string
# keys.


class _Unsupported(Exception):
    """A ``name`` value whose OpenClaw text this module does not derive.

    Aliases, explicit (``?``) and collection keys, block scalars inside
    collections, and the ``!!timestamp``, ``!!binary``, and ``!!merge`` tags.
    Such a value gives no name.
    """


# The characters String.prototype.trim() removes.
_JS_SPACE = (
    "\t\n\v\f\r \xa0 " + "".join(chr(code) for code in range(0x2000, 0x200B)) + "    　﻿"
)
# OpenClaw's line fallback (parseLineFrontmatter): an unquoted key at the
# start of the line, its colon, and the rest of the line.
_LINE_FIELD = re.compile(rf"([A-Za-z0-9_-]+):[{re.escape(_JS_SPACE)}]*([^\n\r  ]*)")
_SURROGATE = re.compile("[\ud800-\udfff]")
# What may follow ``:`` or another indicator in a flow collection for it to
# count.
_FLOW_END = (*_BLANK, ",", "[", "]", "{", "}")
_YAML_TAG_PREFIX = "tag:yaml.org,2002:"
_ARRAY_INDEX_LIMIT = 2**32 - 1


def _js_int(digits: str, base: int) -> float:
    try:
        return float(int(digits, base))
    except OverflowError:
        return math.inf


def _js_special_float(text: str) -> float:
    if text[-3:].lower() == "nan":
        return float("nan")
    return -math.inf if text[0] == "-" else math.inf


# The YAML 1.2 core schema as the ``yaml`` package resolves plain scalars, in
# the order it tries them. Anything else is a string.
_CORE_SCALARS: tuple[tuple[str, re.Pattern[str], Callable[[str], object]], ...] = (
    ("null", re.compile(r"(?:~|[Nn]ull|NULL)?"), lambda text: None),
    ("bool", re.compile(r"[Tt]rue|TRUE|[Ff]alse|FALSE"), lambda text: text[0] in "tT"),
    ("int", re.compile(r"0o[0-7]+"), lambda text: _js_int(text[2:], 8)),
    ("int", re.compile(r"[-+]?[0-9]+"), lambda text: _js_int(text, 10)),
    ("int", re.compile(r"0x[0-9a-fA-F]+"), lambda text: _js_int(text[2:], 16)),
    (
        "float",
        re.compile(r"[-+]?\.(?:inf|Inf|INF)|\.nan|\.NaN|\.NAN"),
        _js_special_float,
    ),
    (
        "float",
        re.compile(r"[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)[eE][-+]?[0-9]+"),
        float,
    ),
    ("float", re.compile(r"[-+]?(?:\.[0-9]+|[0-9]+\.[0-9]*)"), float),
)


def _openclaw_name(
    block: list[str], index: int, match: re.Match[str]
) -> tuple[str | None, int]:
    """Return the name OpenClaw reads for the ``name`` key on ``block[index]``.

    Also returns the index of the first line after the value. OpenClaw
    rejects front matter that is not valid YAML; for such a value the text on
    the key's line is returned, as for the ``pyyaml`` style.
    """

    source = "\n".join(block[index:])
    start = source.index(":", match.end(1)) + 1
    try:
        value, end = _js_node(source, start)
    except _NotYaml:
        return _scalar(match.group(2) or ""), index + 1
    except _Unsupported:
        return None, index + 1
    value_end = index + 1 + source.count("\n", 0, end)
    if value is None:
        text = _line_value(block, index)
    elif isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, float):
        text = _js_number(value)
    elif isinstance(value, str):
        text = value
    else:
        text = _js_json(value)
        # A collection written on the key's line with a colon in it keeps
        # that line's text instead.
        line = _LINE_FIELD.fullmatch(block[index])
        if line is not None and line.group(1) == "name" and ":" in line.group(2):
            text = _line_value(block, index) or text
    return _js_trim(text or "") or None, value_end


def _line_value(block: list[str], index: int) -> str | None:
    """Return OpenClaw's line fallback for the ``name`` key on ``block[index]``.

    The rest of an unquoted ``name:`` line without one pair of enclosing
    quotes or, when it is empty and the next line is indented, the indented
    lines that follow.
    """

    match = _LINE_FIELD.fullmatch(block[index])
    if match is None or match.group(1) != "name":
        return None
    value = _js_trim(match.group(2))
    following = block[index + 1] if index + 1 < len(block) else ""
    if not value and following[:1] in (" ", "\t"):
        lines: list[str] = []
        for line in block[index + 1 :]:
            if line and line[0] not in " \t":
                break
            lines.append(line)
        value = _js_trim("\n".join(lines))
    elif len(value) >= 2 and value[0] in "'\"" and value[-1] == value[0]:
        value = value[1:-1]
    return value or None


def _js_node(source: str, index: int) -> tuple[object, int]:
    """Return the YAML 1.2 value after ``index``, and the end of a quoted value.

    Like :func:`_node`, the end is ``index`` for values that are not quoted.
    """

    position = _next_token(source, index)
    tag: str | None = None
    after = index
    while position is not None and source[position] in "&!":
        after = _property_end(source, position)
        if source[position] == "!":
            tag = source[position:after]
        position = _next_token(source, after)
    if position is None:
        # A block sequence may start at column 0 below the key.
        following = _js_content(source, after)
        if following is not None and _sequence_entry(source, following):
            value, end = _js_block_sequence(source, following)
            if not _only_comments_follow(source, end):
                raise _NotYaml
            return value, index
        return _js_scalar("", tag, plain=True), index
    char, following = source[position], source[position + 1 : position + 2]
    below = "\n" in source[index:position]
    quoted = False
    if char in "|>":
        text, end = _block_scalar(source, position)
        value = _js_scalar(text, tag, plain=False)
    elif char in "'\"":
        scalar = _quoted(source, position)
        if scalar is None:
            raise _NotYaml
        text, end = scalar
        if below and _key_follows(source, end):
            value, end = _js_block_mapping(source, position)
        else:
            value, quoted = _js_scalar(text, tag, plain=False), True
    elif char == "*":
        raise _Unsupported
    elif char in "[{":
        value, end = _js_flow(source, position, 0)
    elif char == "-" and following in _BLANK:
        if not below:
            raise _NotYaml
        value, end = _js_block_sequence(source, position)
    elif char in "?:" and following in _BLANK:
        raise _Unsupported
    elif char in ",]}%@`\t":
        raise _NotYaml
    else:
        text, end = _plain(source, position)
        if _key_follows(source, end):
            if not below:
                raise _NotYaml
            value, end = _js_block_mapping(source, position)
        else:
            value = _js_scalar(text, tag, plain=True)
    if not _only_comments_follow(source, end):
        raise _NotYaml
    return value, end if quoted else index


def _property_end(source: str, index: int) -> int:
    """Return the end of the anchor or tag that starts at ``index``."""

    if source.startswith("!<", index):
        # A verbatim tag may hold flow indicators.
        index = source.find(">", index)
        if index < 0:
            raise _NotYaml
    while index < len(source) and source[index] not in " \t\n,[]{}":
        index += 1
    return index


def _key_follows(source: str, index: int) -> bool:
    """Whether a mapping value indicator follows ``index`` on its line."""

    rest = source[index:].lstrip(" ")
    return rest[:1] == ":" and rest[1:2] in _BLANK


def _sequence_entry(source: str, index: int) -> bool:
    return source[index] == "-" and source[index + 1 : index + 2] in _BLANK


def _js_content(source: str, index: int) -> int | None:
    """Return where the next token after ``index`` starts, on any line."""

    while index < len(source):
        char = source[index]
        if char in " \t\n":
            index += 1
        elif char == "#" and (index == 0 or source[index - 1] in " \t\n"):
            index = _line_end(source, index)
        else:
            return index
    return None


def _js_block_sequence(source: str, index: int) -> tuple[list[object], int]:
    """Read the block sequence whose first ``-`` is at ``index``."""

    indent = _column(source, index)
    items: list[object] = []
    while True:
        value, index = _js_entry(source, index + 1, indent, sequence=True)
        items.append(value)
        following = _js_content(source, index)
        if following is None or _column(source, following) < indent:
            return items, index
        if _column(source, following) > indent:
            raise _NotYaml
        if not _sequence_entry(source, following):
            # A key of the mapping that holds the sequence.
            return items, index
        index = following


def _js_block_mapping(source: str, index: int) -> tuple[dict[str, object], int]:
    """Read the block mapping whose first key starts at ``index``."""

    indent = _column(source, index)
    pairs: dict[str, object] = {}
    seen: set[tuple[str, object]] = set()
    while True:
        key, index = _js_block_key(source, index, indent)
        value, index = _js_entry(source, index, indent, sequence=False)
        _js_add(pairs, seen, key, value)
        following = _js_content(source, index)
        if following is None or _column(source, following) < indent:
            return pairs, index
        if _column(source, following) > indent or _sequence_entry(source, following):
            raise _NotYaml
        index = following


def _js_block_key(source: str, index: int, indent: int) -> tuple[object, int]:
    """Read a one-line block mapping key; return it and the end of its ``:``."""

    char, following = source[index], source[index + 1 : index + 2]
    if char in "'\"":
        scalar = _quoted(source, index)
        if scalar is None or "\n" in source[index : scalar[1]]:
            raise _NotYaml
        key: object = _js_text(scalar[0])
        end = scalar[1]
    elif char in "[{*&!" or (char == "?" and following in _BLANK):
        raise _Unsupported
    elif char in ",]}#|>%@`" or (char in "-:" and following in _BLANK):
        raise _NotYaml
    else:
        text, end = _plain(source, index, indent)
        if "\n" in source[index:end]:
            raise _NotYaml
        key = _js_scalar(text, None, plain=True)
    if not _key_follows(source, end):
        raise _NotYaml
    return key, source.index(":", end) + 1


def _js_entry(
    source: str, index: int, indent: int, *, sequence: bool
) -> tuple[object, int]:
    """Read the value after a block ``-`` or a block key's ``:``.

    ``indent`` is the column of the entry's ``-`` or key.
    """

    while source[index : index + 1] in (" ", "\t"):
        index += 1
    tag: str | None = None
    while source[index : index + 1] in ("&", "!"):
        end = _property_end(source, index)
        if source[index] == "!":
            tag = source[index:end]
        index = end
        while source[index : index + 1] in (" ", "\t"):
            index += 1
    if index < len(source) and source[index] not in "\n#":
        if _sequence_entry(source, index) and not sequence:
            raise _NotYaml
        return _js_inline(source, index, indent, tag, compact=sequence)
    following = _js_content(source, index)
    if following is not None:
        column = _column(source, following)
        if column > indent:
            return _js_inline(source, following, indent, tag, compact=True)
        if column == indent and not sequence and _sequence_entry(source, following):
            # A mapping value may be a sequence at the key's own indentation.
            return _js_block_sequence(source, following)
    return _js_scalar("", tag, plain=True), index


def _js_inline(
    source: str, index: int, indent: int, tag: str | None, *, compact: bool
) -> tuple[object, int]:
    """Read the node at ``index`` inside a block collection at ``indent``.

    With ``compact``, a key at ``index`` starts a block mapping there.
    """

    char, following = source[index], source[index + 1 : index + 2]
    if _sequence_entry(source, index):
        return _js_block_sequence(source, index)
    if char in "[{":
        value, end = _js_flow(source, index, indent)
        if _key_follows(source, end):
            raise _Unsupported
        return value, end
    if char in "'\"":
        scalar = _quoted(source, index)
        if scalar is None:
            raise _NotYaml
        text, end = scalar
        if _key_follows(source, end):
            if not compact or "\n" in source[index:end]:
                raise _NotYaml
            return _js_block_mapping(source, index)
        return _js_scalar(text, tag, plain=False), end
    if char in "|>*" or (char in "?:" and following in _BLANK):
        raise _Unsupported
    if char in ",]}#%@`":
        raise _NotYaml
    text, end = _plain(source, index, indent)
    if _key_follows(source, end):
        if not compact or "\n" in source[index:end]:
            raise _NotYaml
        return _js_block_mapping(source, index)
    return _js_scalar(text, tag, plain=True), end


def _js_flow(
    source: str, index: int, indent: int
) -> tuple[list[object] | dict[str, object], int]:
    """Read the flow sequence or mapping that opens at ``index``.

    Lines inside it must start after column ``indent``.
    """

    closing = "]" if source[index] == "[" else "}"
    items: list[object] = []
    pairs: dict[str, object] = {}
    seen: set[tuple[str, object]] = set()
    index = _js_flow_space(source, index + 1, indent)
    while source[index] != closing:
        if source[index] == "?" and source[index + 1 : index + 2] in _FLOW_END:
            raise _Unsupported
        node, index, plain = _js_flow_node(source, index, indent)
        index = _js_flow_space(source, index, indent)
        if source[index] == ":" and (
            not plain or source[index + 1 : index + 2] in _FLOW_END
        ):
            index = _js_flow_space(source, index + 1, indent)
            value: object = None
            if source[index] not in ",]}":
                value, index, _ = _js_flow_node(source, index, indent)
                index = _js_flow_space(source, index, indent)
            if closing == "}":
                _js_add(pairs, seen, node, value)
            else:
                items.append({_js_key(node): value})
        elif closing == "}":
            _js_add(pairs, seen, node, None)
        else:
            items.append(node)
        if source[index] == ",":
            index = _js_flow_space(source, index + 1, indent)
        elif source[index] != closing:
            raise _NotYaml
    return (pairs if closing == "}" else items), index + 1


def _js_flow_space(source: str, index: int, indent: int) -> int:
    """Skip white space, line breaks, and comments inside a flow collection."""

    crossed = False
    while index < len(source):
        char = source[index]
        if char in " \t":
            index += 1
        elif char == "\n":
            index += 1
            crossed = True
        elif char == "#" and source[index - 1] in " \t\n":
            index = _line_end(source, index)
        else:
            # Lines inside the collection start after ``indent``, except for
            # one that closes it.
            if crossed and _column(source, index) <= indent and char not in "]}":
                raise _NotYaml
            return index
    # The collection is not closed.
    raise _NotYaml


def _js_flow_node(source: str, index: int, indent: int) -> tuple[object, int, bool]:
    """Read one node inside a flow collection.

    Returns the value, its end, and whether it is a plain scalar.
    """

    tag: str | None = None
    properties = False
    while source[index] in "&!":
        end = _property_end(source, index)
        if source[index] == "!":
            tag = source[index:end]
        properties = True
        index = _js_flow_space(source, end, indent)
    char, following = source[index], source[index + 1 : index + 2]
    if char in ",]}" or (char == ":" and following in _FLOW_END):
        if not properties:
            raise _NotYaml
        return _js_scalar("", tag, plain=True), index, True
    if char in "[{":
        value, end = _js_flow(source, index, indent)
        return value, end, False
    if char in "'\"":
        scalar = _quoted(source, index)
        if scalar is None:
            raise _NotYaml
        return _js_scalar(scalar[0], tag, plain=False), scalar[1], False
    if char == "*":
        raise _Unsupported
    if char in "#|>%@`" or (char in "-?:" and following in _FLOW_END):
        raise _NotYaml
    text, end = _js_flow_plain(source, index, indent)
    return _js_scalar(text, tag, plain=True), end, True


def _js_flow_plain(source: str, index: int, indent: int) -> tuple[str, int]:
    """Read a plain scalar inside a flow collection, folding its lines."""

    chunks: list[str] = []
    separator = ""
    end = index
    while True:
        stop = index
        while stop < len(source) and not _flow_plain_stop(source, stop):
            stop += 1
        text = source[index:stop].rstrip(" \t")
        if text:
            chunks += [separator, text]
            end = index + len(text)
        if stop >= len(source) or source[stop] != "\n":
            break
        breaks = 0
        index = stop
        while source[index : index + 1] in (" ", "\t", "\n"):
            breaks += source[index] == "\n"
            index += 1
        if (
            index >= len(source)
            or source[index] == "#"
            or _flow_plain_stop(source, index)
            or _column(source, index) <= indent
        ):
            break
        # A single line break folds to a space; blank lines keep their breaks.
        separator = "\n" * (breaks - 1) or " "
    return "".join(chunks), end


def _flow_plain_stop(source: str, index: int) -> bool:
    char = source[index]
    if char in "\n,[]{}":
        return True
    if char == ":":
        return source[index + 1 : index + 2] in _FLOW_END
    return char == "#" and source[index - 1] in " \t"


def _js_add(
    pairs: dict[str, object], seen: set[tuple[str, object]], key: object, value: object
) -> None:
    """Add a mapping entry; equal keys are a YAML error."""

    if isinstance(key, (list, dict)):
        raise _Unsupported
    identity = (type(key).__name__, key)
    if identity in seen:
        raise _NotYaml
    seen.add(identity)
    pairs[_js_key(key)] = value


def _js_scalar(text: str, tag: str | None, *, plain: bool) -> object:
    """Return the value of a scalar as the YAML 1.2 core schema resolves it."""

    if tag is None:
        return _js_resolve(text, None) if plain else _js_text(text)
    if tag.startswith("!<") and tag.endswith(">"):
        name = tag[2:-1]
    elif tag.startswith("!!"):
        name = _YAML_TAG_PREFIX + tag[2:]
    else:
        name = tag
    kind = name[len(_YAML_TAG_PREFIX) :] if name.startswith(_YAML_TAG_PREFIX) else None
    if kind in ("timestamp", "binary", "merge"):
        raise _Unsupported
    if kind in ("null", "bool", "int", "float"):
        # An explicit tag whose pattern does not match leaves a string.
        return _js_resolve(text, kind)
    return _js_text(text)


def _js_resolve(text: str, kind: str | None) -> object:
    for scalar_kind, pattern, convert in _CORE_SCALARS:
        if kind in (None, scalar_kind) and pattern.fullmatch(text):
            return convert(text)
    return _js_text(text)


def _js_text(text: str) -> str:
    """Join escaped surrogate pairs into characters, as JavaScript strings do."""

    if _SURROGATE.search(text):
        text = text.encode("utf-16-le", "surrogatepass").decode(
            "utf-16-le", "surrogatepass"
        )
    return text


def _js_trim(text: str) -> str:
    return text.strip(_JS_SPACE)


def _js_key(key: object) -> str:
    """Return the property name a mapping key becomes in JavaScript."""

    if key is None:
        return ""
    if isinstance(key, bool):
        return "true" if key else "false"
    if isinstance(key, float):
        return _js_number(key)
    if isinstance(key, str):
        return key
    raise _Unsupported


def _js_number(value: float) -> str:
    """Return ``String(value)`` for a JavaScript number."""

    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    if value == 0:
        return "0"
    if value < 0:
        return "-" + _js_number(-value)
    # repr() gives the shortest digits that round-trip, as JavaScript does.
    mantissa, _, exponent = repr(value).partition("e")
    whole, _, fraction = mantissa.partition(".")
    digits = (whole + fraction).lstrip("0")
    # The value is 0.<digits> times 10 to the power ``point``.
    point = len(whole) + int(exponent or 0) - (len(whole + fraction) - len(digits))
    digits = digits.rstrip("0")
    count = len(digits)
    if count <= point <= 21:
        return digits + "0" * (point - count)
    if 0 < point <= 21:
        return f"{digits[:point]}.{digits[point:]}"
    if -6 < point <= 0:
        return f"0.{'0' * -point}{digits}"
    power = point - 1
    body = digits if count == 1 else f"{digits[0]}.{digits[1:]}"
    return f"{body}e{'+' if power >= 0 else '-'}{abs(power)}"


def _js_json(value: object) -> str:
    """Return ``JSON.stringify(value)`` for a value held as JavaScript holds it."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return _js_number(value) if math.isfinite(value) else "null"
    if isinstance(value, str):
        return _js_json_string(value)
    if isinstance(value, list):
        return "[" + ",".join(_js_json(item) for item in value) + "]"
    assert isinstance(value, dict)
    # JavaScript lists integer-like property names first, in numeric order.
    indices = sorted((key for key in value if _array_index(key)), key=int)
    keys = indices + [key for key in value if not _array_index(key)]
    return (
        "{"
        + ",".join(f"{_js_json_string(key)}:{_js_json(value[key])}" for key in keys)
        + "}"
    )


def _js_json_string(text: str) -> str:
    # JSON.stringify escapes the same characters as json.dumps, plus lone
    # surrogates.
    return _SURROGATE.sub(
        lambda match: f"\\u{ord(match.group()):04x}",
        json.dumps(text, ensure_ascii=False),
    )


def _array_index(key: str) -> bool:
    return (
        key.isascii()
        and key.isdigit()
        and (key == "0" or key[0] != "0")
        and int(key) < _ARRAY_INDEX_LIMIT
    )


def _within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)
