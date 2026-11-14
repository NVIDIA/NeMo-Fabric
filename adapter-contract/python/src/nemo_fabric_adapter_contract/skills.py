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
4. Each candidate's front matter is checked against the harness's
   requirements in the rules, such as a non-empty ``description`` or front
   matter that parses (:func:`check_front_matter`), before any duplicate
   check. A candidate the harness refuses is skipped and recorded, so it
   never counts as a copy of a name.
5. An empty ``folder`` entry is valid and recorded in ``empty_entries``.
6. The same folder reached through several entries loads once.
7. Skills that share a name are recorded, or fail or are resolved according
   to the configured duplicate policy.

A skill's name is the ``name`` field of its ``SKILL.md`` front matter, read as
the harness reads it (see :data:`NameStyle`), or its folder name when the
front matter has none and the rules allow the fallback. Under the default
``pyyaml`` style, a value that YAML reads as anything but a non-empty string,
such as null or a number, counts as no name. Aliases resolve to the anchored
value under each style's rules.

Some front matter names this module cannot derive reliably, such as an alias
whose anchor is in a value it does not read, an explicit (``?``) key, or a
tag it does not resolve. Such a skill is marked ``name_uncertain``. Without a
duplicate policy it keeps the folder name fallback and is reported; under an
explicit policy, where a wrong name could pick the wrong copy, the entry fails
with ``unsupported_frontmatter_name``.

Likewise, when the rules require front matter that parses or non-empty
fields and this module cannot tell whether the harness accepts a
candidate's front matter, the candidate is kept, as the harness may load
it. Under an explicit policy, when the outcome for a name would depend on
whether that candidate loads, the entry fails with
``unsupported_frontmatter`` instead of deciding on a guess.

The module reads only front matter and never reports file contents.
"""

from __future__ import annotations

import base64
import binascii
import datetime
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

SkillFailure = Literal[
    "not_found",
    "invalid",
    "duplicate",
    "unsupported_frontmatter_name",
    "unsupported_frontmatter",
]
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
    """Top-level front matter keys a candidate must define to load. A
    candidate without one is skipped with ``missing_frontmatter_field:<key>``."""
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
    nonempty_frontmatter_fields: tuple[str, ...] = ()
    """Top-level front matter keys a candidate must define with text that is
    not empty after trimming, as the harness reads it. A candidate without
    one is skipped with ``missing_frontmatter_field:<key>``, and one whose
    value is empty with ``empty_frontmatter_field:<key>``. Requires the
    ``openclaw`` name style, which reads field text."""
    frontmatter_must_parse: bool = False
    """Whether the harness refuses a candidate whose front matter its YAML
    parser rejects. Such a candidate is skipped with ``invalid_frontmatter``.
    A ``SKILL.md`` without a complete front matter block is left to the
    field requirements. Requires the ``openclaw`` name style, which tells
    whether the block parses."""

    def __post_init__(self) -> None:
        if self.traversal not in ("immediate_children", "recursive"):
            raise ValueError(f"unknown skill traversal: {self.traversal!r}")
        if self.nested_skills not in ("include", "stop_at_skill"):
            raise ValueError(f"unknown nested skill rule: {self.nested_skills!r}")
        if not _is_depth(self.max_depth, allow_none=True):
            raise ValueError("max_depth must be None or an integer of at least 1")
        if self.name_style not in get_args(NameStyle):
            raise ValueError(f"unknown skill name style: {self.name_style!r}")
        if (
            self.nonempty_frontmatter_fields or self.frontmatter_must_parse
        ) and self.name_style != "openclaw":
            raise ValueError(
                "nonempty_frontmatter_fields and frontmatter_must_parse require "
                "the openclaw name style"
            )
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
    name_uncertain: bool = False
    """Whether the front matter name could not be derived reliably, so the
    name may differ from the harness's."""
    validity_uncertain: bool = False
    """Whether Fabric cannot tell if the harness accepts the skill's front
    matter (:func:`check_front_matter`), so the harness may refuse it."""


@dataclass(frozen=True)
class FrontMatter:
    """What :func:`read_front_matter` reads from a ``SKILL.md`` front matter block."""

    name: str | None
    """The ``name`` value as the name style reads it, before any folder
    fallback."""
    name_uncertain: bool
    """Whether the name could not be derived reliably."""
    keys: frozenset[str]
    """Top-level keys."""
    parses: bool | None
    """Under the ``openclaw`` style, whether OpenClaw's YAML parser accepts
    the block, or ``None`` when this module cannot tell. Always ``None``
    under ``pyyaml``."""
    fields: Mapping[str, str]
    """Under the ``openclaw`` style, the text OpenClaw reads for each
    top-level key, including its line fallback. Empty under ``pyyaml``."""


@dataclass(frozen=True)
class SkippedCandidate:
    """A folder holding ``SKILL.md`` that does not load."""

    path: Path
    """Resolved candidate folder."""
    field: str
    """The entry the candidate was found through."""
    reason: str
    """Why it was skipped: ``unreadable``, ``no_name``, ``path_escape``,
    ``invalid_frontmatter``, ``missing_frontmatter_field:<field>``, or
    ``empty_frontmatter_field:<field>``."""


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

    A skill whose front matter name cannot be derived reliably keeps the
    folder name fallback and is marked ``name_uncertain`` when ``duplicates``
    is ``None``; under an explicit policy its entry fails with
    ``unsupported_frontmatter_name``.

    Each candidate is checked against the rules' front matter requirements
    (:func:`check_front_matter`) before any duplicate check, so a candidate
    the harness refuses is skipped and never counts as a copy. A candidate
    whose acceptance cannot be told is kept and marked
    ``validity_uncertain``; under an explicit policy, when the outcome for
    its name depends on whether it loads, its entry fails with
    ``unsupported_frontmatter``.

    Raises:
        SkillLoadError: If an entry is malformed, missing, not a folder, not
            accessible, or cannot be read, if skills share a name under an
            explicit ``duplicates`` policy, if a skill's name cannot be
            derived reliably under one, or if the outcome of one depends on
            front matter Fabric cannot check.
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
        uncertain: set[tuple[int, int] | Path] = set()
        # Candidates the harness may refuse: Fabric cannot tell.
        unchecked: set[tuple[int, int] | Path] = set()
        for candidate in candidates:
            if not entry_rules.allow_symlink_escape and not _within(
                candidate.resolved, root
            ):
                load.skipped.append(
                    SkippedCandidate(candidate.resolved, entry_field, "path_escape")
                )
                continue
            read = _read_candidate(
                candidate,
                entry_rules,
                entry_field,
                load.skipped,
                required=kind == "skill" and candidate.resolved == root,
                key_field=key_field,
            )
            if read is None:
                continue
            name, name_uncertain, validity_uncertain = read
            if validity_uncertain:
                unchecked.add(candidate.identity)
            if name_uncertain:
                if duplicates is not None:
                    # A wrong name could pick the wrong copy under the policy.
                    skill_file = candidate.resolved / SKILL_FILE
                    raise SkillLoadError(
                        "unsupported_frontmatter_name",
                        f"NeMo Fabric cannot read the front matter name of "
                        f"{skill_file} reliably, so the {duplicates} duplicate "
                        f"policy in {entry_field} cannot be applied "
                        f"(unsupported_frontmatter_name); write the name as a "
                        f"plain name: value",
                        field=entry_field,
                        paths=(candidate.resolved,),
                    )
                uncertain.add(candidate.identity)
            named.append((name, candidate))
        if kind == "folder" and not named:
            load.empty_entries.append(entry_field)
        if duplicates is not None:
            # Check the entry's own skills before another entry can take
            # one of them over.
            _reject_duplicates_in_entry(named, entry_field, unchecked, duplicates)
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
                    name=name,
                    path=candidate.resolved,
                    field=entry_field,
                    kind=kind,
                    name_uncertain=candidate.identity in uncertain,
                    validity_uncertain=candidate.identity in unchecked,
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
    named: list[tuple[str, _Candidate]],
    field: str,
    unchecked: set[tuple[int, int] | Path],
    policy: DuplicatePolicy,
) -> None:
    """Fail when two skills found through one entry share a name.

    A copy in ``unchecked`` may not load, so a name it shares fails with
    ``unsupported_frontmatter`` unless two other copies already clash.
    """

    seen: dict[str, Path] = {}
    for name, candidate in named:
        if candidate.identity in unchecked:
            continue
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
    counts: dict[str, int] = {}
    for name, _ in named:
        counts[name] = counts.get(name, 0) + 1
    for name, candidate in named:
        if counts[name] > 1 and candidate.identity in unchecked:
            raise _unchecked_copy(candidate.resolved, name, policy, field)


def _unchecked_copy(
    path: Path, name: str, policy: DuplicatePolicy, field: str
) -> SkillLoadError:
    """Return the failure for a copy whose acceptance would decide a policy."""

    skill_file = path / SKILL_FILE
    return SkillLoadError(
        "unsupported_frontmatter",
        f"NeMo Fabric cannot tell whether the harness accepts the front matter "
        f"of {skill_file}, a copy of skill {name!r}, so the {policy} duplicate "
        f"policy in {field} cannot be applied (unsupported_frontmatter); write "
        f"its front matter as plain key: value lines and quote values that "
        f"contain ': '",
        field=field,
        paths=(path,),
    )


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
    and is copied as given. A loaded skill whose name could not be derived
    reliably carries ``"name_uncertain": true``. The report never includes
    file contents.
    """

    block: dict[str, Any] = {}
    if load.selected:
        block["loaded"] = [_loaded_report(skill) for skill in load.selected]
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


def _loaded_report(skill: FoundSkill) -> dict[str, Any]:
    item: dict[str, Any] = {
        "name": skill.name,
        "path": str(skill.path),
        "entry": skill.field,
        "origin": "external",
    }
    if skill.name_uncertain:
        item["name_uncertain"] = True
    return item


def warn(adapter: str, load: DirectoryLoad) -> None:
    """Write one stderr line per skipped candidate, empty entry, duplicate, and
    skill whose name could not be derived reliably."""

    lines = [
        f"{adapter}: skipped skill {item.path} from {item.field}: {item.reason}"
        for item in load.skipped
    ]
    lines += [
        f"{adapter}: skill {skill.path} from {skill.field} loads as {skill.name!r}, "
        "but its front matter name could not be read reliably "
        "(unsupported_frontmatter_name)"
        for skill in load.selected
        if skill.name_uncertain
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


def _read_candidate(
    candidate: _Candidate,
    rules: DiscoveryRules,
    field: str,
    skipped: list[SkippedCandidate],
    *,
    required: bool,
    key_field: str,
) -> tuple[str, bool, bool] | None:
    """Return the candidate's name, whether the name is uncertain, and
    whether its acceptance is, or ``None`` when the candidate is skipped."""

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
    accepted, reason = check_front_matter(front_matter, rules)
    if accepted is False:
        assert reason is not None
        skipped.append(SkippedCandidate(candidate.resolved, field, reason))
        return None
    name = front_matter.name if front_matter is not None else None
    uncertain = front_matter is not None and front_matter.name_uncertain
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
    return name, uncertain, accepted is None


def check_front_matter(
    front_matter: FrontMatter | None, rules: DiscoveryRules
) -> tuple[bool | None, str | None]:
    """Return whether the harness accepts a skill with ``front_matter``.

    Applies the rules' front matter requirements: front matter that parses
    (``frontmatter_must_parse``), then ``required_frontmatter_fields``, then
    ``nonempty_frontmatter_fields``. ``front_matter`` is read with the rules'
    name style (:func:`read_front_matter`), or ``None`` for a ``SKILL.md``
    without a complete front matter block.

    Returns:
        ``(True, None)`` when the skill meets the requirements, ``(False,
        reason)`` with the skip reason when it does not, and ``(None, None)``
        when Fabric cannot tell, because the rules require front matter that
        parses or non-empty fields and it cannot tell whether the block parses.
    """

    if (
        rules.frontmatter_must_parse
        and front_matter is not None
        and front_matter.parses is False
    ):
        return False, "invalid_frontmatter"
    keys = front_matter.keys if front_matter is not None else frozenset()
    for name in (
        *rules.required_frontmatter_fields,
        *rules.nonempty_frontmatter_fields,
    ):
        if name not in keys:
            return False, f"missing_frontmatter_field:{name}"
    if front_matter is None or not (
        rules.frontmatter_must_parse or rules.nonempty_frontmatter_fields
    ):
        return True, None
    if front_matter.parses is None:
        return None, None
    for name in rules.nonempty_frontmatter_fields:
        if not front_matter.fields.get(name):
            return False, f"empty_frontmatter_field:{name}"
    return True, None


def _resolve_duplicates(
    found: list[FoundSkill],
    policy: DuplicatePolicy | None,
    records: list[DuplicateRecord],
) -> list[FoundSkill]:
    by_name: dict[str, list[FoundSkill]] = {}
    for skill in found:
        by_name.setdefault(skill.name, []).append(skill)
    shadowed: set[int] = set()
    # The first copy whose acceptance would decide the outcome of a policy.
    # A clash between checked copies fails first.
    deciding: FoundSkill | None = None
    for name, copies in by_name.items():
        if len(copies) < 2:
            continue
        record_copies = tuple(("external", skill.path, skill.field) for skill in copies)
        if policy is None:
            records.append(DuplicateRecord(name, record_copies, "ambiguous", None))
            continue
        if policy == "reject":
            checked = [skill for skill in copies if not skill.validity_uncertain]
            if len(checked) < 2:
                # Whether the name is duplicated depends on an unchecked copy.
                deciding = deciding or next(
                    skill for skill in copies if skill.validity_uncertain
                )
                continue
            first, second = checked[0], checked[1]
            raise SkillLoadError(
                "duplicate",
                f"NeMo Fabric skill {name!r} is configured twice: "
                f"{first.path} ({first.field}) and {second.path} ({second.field})",
                field=second.field,
                paths=(first.path, second.path),
            )
        winner = copies[0] if policy == "first_wins" else copies[-1]
        if winner.validity_uncertain:
            # If the harness refused it, the next copy would win instead.
            deciding = deciding or winner
            continue
        shadowed.update(id(skill) for skill in copies if skill is not winner)
        records.append(DuplicateRecord(name, record_copies, "selected", winner.path))
    if deciding is not None:
        assert policy is not None
        raise _unchecked_copy(deciding.path, deciding.name, policy, deciding.field)
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


def read_front_matter(
    skill_file: str | os.PathLike[str], *, name_style: NameStyle = "pyyaml"
) -> FrontMatter | None:
    """Read the front matter of a ``SKILL.md`` file as ``name_style`` reads it.

    Returns ``None`` when the file has no complete ``---`` front matter block.

    Raises:
        OSError: If the file cannot be read.
    """

    if name_style not in get_args(NameStyle):
        raise ValueError(f"unknown name style: {name_style!r}")
    with open(skill_file, encoding="utf-8-sig", errors="replace") as lines:
        return _front_matter(lines, name_style)


def _front_matter(
    lines: Iterable[str], style: NameStyle = "pyyaml"
) -> FrontMatter | None:
    """Read a ``---`` front matter block, or return ``None`` when there is none."""

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
    return _read_block(block, style)


# Keys whose invalid value OpenClaw may rewrite as a quoted string and parse
# again (normalizeFreeformFieldAtError).
_FREEFORM_FIELDS = frozenset({"description", "read_when", "summary"})
_PARSE_ORDER = {False: 0, None: 1, True: 2}


def _read_block(block: list[str], style: NameStyle) -> FrontMatter:
    """Read the lines between the fences of a front matter block."""

    text = "\n".join(block)
    offsets: list[int] = []
    position = 0
    for line in block:
        offsets.append(position)
        position += len(line) + 1
    anchors = _Anchors(text, style)
    # The pyyaml style reads the other values only for anchors an alias may
    # need; OpenClaw's rules need every value.
    read_all = style == "openclaw" or "&" in text
    keys: set[str] = set()
    values: dict[str, object] = {}
    key_lines: dict[str, int] = {}
    name: str | None = None
    name_from_yaml = False
    name_uncertain = False
    layout_uncertain = _indented_mapping(block)
    parses: bool | None = None if style == "pyyaml" or layout_uncertain else True
    # Lines up to this index belong to the value of the last key read.
    value_end = 0
    for index, line in enumerate(block):
        if index < value_end:
            continue
        if _unsupported_key(line):
            layout_uncertain = True
            parses = _least(parses, None)
            continue
        match = _KEY.fullmatch(line.rstrip())
        if not match:
            continue
        key = _unquote(match.group(1))
        source = "\n".join(block[index:])
        start = source.index(":", match.end(1)) + 1
        anchors.base = offsets[index]
        if style == "pyyaml":
            keys.add(key)
            if key == "name":
                # When the key repeats, PyYAML keeps the last.
                try:
                    name, value_end = _name(block, index, match, anchors)
                    name_uncertain = False
                except _Unsupported:
                    name, value_end, name_uncertain = None, index + 1, True
            if read_all:
                try:
                    _js_node(source, start, anchors)
                except (_NotYaml, _Unsupported):
                    # Anchors in this value stay unknown; an alias of one is
                    # unsupported.
                    pass
            continue
        # OpenClaw trims each key and rejects a repeated one.
        key = _js_trim(key)
        if key in keys:
            parses = False
        keys.add(key)
        key_lines[key] = index
        values.pop(key, None)
        value_end = index + 1
        try:
            value, end = _js_node(source, start, anchors)
        except _NotYaml:
            # OpenClaw rejects the block, unless it can recover the field. The
            # yaml package accepts tabs in places this reader does not.
            recoverable = key in _FREEFORM_FIELDS or "\t" in text
            parses = _least(parses, None if recoverable else False)
            if key == "name":
                # The text on the line, as for the pyyaml style.
                name, name_from_yaml = _scalar(match.group(2) or ""), False
                name_uncertain = False
            continue
        except _Unsupported:
            parses = _least(parses, None)
            if key == "name":
                name, name_from_yaml, name_uncertain = None, False, True
            continue
        values[key] = value
        value_end += source.count("\n", 0, end)
        if key == "name":
            name_from_yaml, name_uncertain = True, False
    fields: dict[str, str] = {}
    if style == "openclaw":
        line_fields = _line_fields(block)
        for key, value in values.items():
            field_text = _openclaw_text(
                value, key, block[key_lines[key]], line_fields.get(key)
            )
            if field_text is not None:
                fields[key] = field_text
        # Keys YAML reads as null, or does not see, take the line fallback.
        for key, field_text in line_fields.items():
            fields.setdefault(key, field_text)
        if name_from_yaml or "name" not in keys:
            name = _js_trim(fields.get("name", "")) or None
    elif "<<" in keys and "name" not in keys:
        # PyYAML merges the keys of the ``<<`` mapping in.
        name_uncertain = True
    return FrontMatter(
        name=name,
        name_uncertain=name_uncertain or layout_uncertain,
        keys=frozenset(keys),
        parses=parses,
        fields=MappingProxyType(fields),
    )


def _least(current: bool | None, new: bool | None) -> bool | None:
    """Combine what two parts say about whether the block parses."""

    return current if _PARSE_ORDER[current] <= _PARSE_ORDER[new] else new


def _indented_mapping(block: list[str]) -> bool:
    """Whether the first content line is indented, as when the whole mapping is."""

    for line in block:
        content = line.lstrip(" \t")
        if content and not content.startswith("#"):
            return content != line
    return False


def _unsupported_key(line: str) -> bool:
    """Whether a top-level line starts a key this module does not read.

    An explicit (``?``) key, a key with an anchor or tag, an alias key, or a
    flow collection.
    """

    return line[:1] in ("&", "!", "*", "{", "[") or (
        line[:1] == "?" and line[1:2] in ("", " ", "\t")
    )


class _NotYaml(Exception):
    """The ``name`` value is not valid YAML."""


class _Unsupported(Exception):
    """A value whose text this module does not derive.

    For example an alias whose anchor is in a value this module does not
    read, an explicit (``?``) or collection key, or a tag it does not
    resolve. The name is then uncertain.
    """


# A scalar's text, tag, and whether it is plain, for an anchor of it.
_Raw = tuple[str, "str | None", bool]


class _Anchors:
    """The anchors read so far in a front matter block, to resolve aliases.

    Each anchor keeps its offset in the block, its value as JavaScript holds
    it, and for a scalar its text, tag, and whether it is plain, so that each
    name style resolves the alias by its own rules. An alias resolves only
    when the last ``&name`` before it in the text is the recorded anchor;
    otherwise the anchor is in a value this module did not read, or the text
    only looks like one, and the alias is unsupported.
    """

    def __init__(self, text: str, style: NameStyle) -> None:
        self.text = text
        self.style = style
        self.base = 0
        """Offset in the block of the source being read."""
        self.nodes: dict[str, tuple[int, object, _Raw | None]] = {}
        self.duplicate = False
        """Whether an anchor name was defined twice, which PyYAML rejects."""

    def define(
        self,
        anchor: tuple[str, int] | None,
        value: object,
        raw: _Raw | None,
    ) -> None:
        if anchor is None or not anchor[0]:
            return
        label, position = anchor
        if label in self.nodes:
            self.duplicate = True
        self.nodes[label] = (self.base + position, value, raw)

    def resolve(self, label: str, position: int) -> tuple[object, _Raw | None]:
        """Return the value and raw scalar of the alias at ``position``.

        Raises:
            _NotYaml: If no anchor of that name comes before the alias.
            _Unsupported: If the anchor before it was not read.
        """

        position += self.base
        pattern = re.compile("&" + re.escape(label) + r"(?=[\s,\[\]{}]|\Z)")
        last = None
        for match in pattern.finditer(self.text, 0, position):
            last = match.start()
        if last is None:
            raise _NotYaml
        node = self.nodes.get(label)
        if node is None or node[0] != last:
            raise _Unsupported
        return node[1], node[2]


def _name(
    block: list[str], index: int, match: re.Match[str], anchors: _Anchors
) -> tuple[str | None, int]:
    """Return the string YAML reads for the ``name`` key on ``block[index]``.

    The value is read as PyYAML reads the front matter: a plain, quoted, or
    block scalar, over several lines where YAML allows it, or an alias of
    one. A value that YAML reads as anything but a non-empty string, such as
    null, a number, a boolean, a date, a sequence, or a mapping, gives no
    name. When the value is not valid YAML, the text on the key's line is
    used instead, as harnesses that recover from front matter errors line by
    line read it.

    Also returns the index of the first line after the value.

    Raises:
        _Unsupported: If the name cannot be derived reliably.
    """

    # The text harnesses pass to YAML: the lines between the fences, without
    # a final line break.
    source = "\n".join(block[index:])
    start = source.index(":", match.end(1)) + 1
    try:
        name, end = _node(source, start, anchors)
    except _NotYaml:
        return _scalar(match.group(2) or ""), index + 1
    return name, index + 1 + source.count("\n", 0, end)


# PyYAML's anchor and alias names.
_PYYAML_ANCHOR = re.compile(r"[0-9A-Za-z_-]+")


def _node(
    source: str, index: int, anchors: _Anchors | None = None
) -> tuple[str | None, int]:
    """Return the string value after ``index``, and the end of a quoted value.

    Only a quoted value can run onto lines that start at column 0, so the end
    is ``index`` for other values.
    """

    position = _next_token(source, index)
    tag: str | None = None
    properties = False
    while position is not None and source[position] in "&!":
        # Node properties: an anchor, or a tag.
        end = position
        while end < len(source) and source[end] not in " \t\n":
            end += 1
        if source[position] == "!":
            tag = source[position:end]
        properties = True
        position = _next_token(source, end)
    if position is None:
        return _pyyaml_scalar("", tag, plain=True), index
    char, following = source[position], source[position + 1 : position + 2]
    plain = False
    if char in "|>":
        text, end = _block_scalar(source, position)
    elif char in "'\"":
        quoted = _quoted(source, position)
        if quoted is None:
            raise _NotYaml
        text, end = quoted
    elif char == "*":
        # An alias, which cannot have properties of its own.
        end = position + 1
        while end < len(source) and source[end] not in " \t\n,[]{}":
            end += 1
        label = source[position + 1 : end]
        if (
            properties
            or not _PYYAML_ANCHOR.fullmatch(label)
            or not _only_comments_follow(source, end)
        ):
            raise _NotYaml
        if anchors is None:
            raise _Unsupported
        _, raw = anchors.resolve(label, position)
        if anchors.duplicate:
            raise _NotYaml
        if raw is None:
            # An alias of a collection.
            return None, index
        return _pyyaml_scalar(*raw), index
    elif char in "[{" or (char in "-?:" and following in _BLANK):
        # A collection.
        return None, index
    elif char in ",]}%@`\t":
        raise _NotYaml
    else:
        text, end = _plain(source, position)
        plain = True
        if source[end:].lstrip(" ")[:1] == ":" and "\n" in source[:position]:
            # A key on a line below ``name:`` starts a nested mapping.
            return None, index
    if not _only_comments_follow(source, end):
        raise _NotYaml
    if char not in "'\"":
        end = index
    return _pyyaml_scalar(text, tag, plain), end


def _pyyaml_scalar(text: str, tag: str | None, plain: bool) -> str | None:
    """Return the string PyYAML reads for a scalar, or ``None`` for another type.

    Raises:
        _Unsupported: If the tag is one this module does not resolve, or
            PyYAML would fail to build the tagged value.
    """

    if tag is not None and tag not in _STRING_TAGS and tag != "!":
        _pyyaml_typed(text, tag)
        return None
    # PyYAML resolves the type of an untagged plain scalar, and of any scalar
    # with the non-specific tag ``!``, from its text.
    resolve = tag == "!" or (plain and tag is None)
    if resolve and _NON_STRING_PLAIN.fullmatch(text):
        return None
    return text or None


_PYYAML_TYPES = {
    "bool": re.compile(r"(?i:yes|no|true|false|on|off)"),
    "int": re.compile(
        r"[-+]?0b[0-1_]+|[-+]?0[0-7_]+|[-+]?(?:0|[1-9][0-9_]*)|[-+]?0x[0-9a-fA-F_]+"
        r"|[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+"
    ),
    "float": re.compile(
        r"""
        [-+]?[0-9][0-9_]*(?:\.[0-9_]*)?(?:[eE][-+][0-9]+)?
        |\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
        |[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN)
        """,
        re.VERBOSE,
    ),
}
_PYYAML_TIMESTAMP = re.compile(
    r"""
    ([0-9]{4})-([0-9]{1,2})-([0-9]{1,2})
    (?:(?:[Tt]|[ \t]+)([0-9]{1,2}):([0-9]{2}):([0-9]{2})(?:\.[0-9]*)?
     (?:[ \t]*(?:Z|[-+]([0-9]{1,2})(?::[0-9]{2})?))?)?
    """,
    re.VERBOSE,
)


def _pyyaml_typed(text: str, tag: str) -> None:
    """Check that PyYAML builds a value of ``tag`` from ``text``.

    Raises:
        _Unsupported: If it may not, or the tag is not a YAML 1.1 scalar type.
    """

    if tag.startswith("!!"):
        kind = tag[2:]
    elif tag.startswith(f"!<{_YAML_TAG_PREFIX}") and tag.endswith(">"):
        kind = tag[len(_YAML_TAG_PREFIX) + 2 : -1]
    else:
        raise _Unsupported
    if kind == "null":
        return
    pattern = _PYYAML_TYPES.get(kind)
    if pattern is not None:
        if not pattern.fullmatch(text):
            raise _Unsupported
        return
    if kind == "timestamp":
        match = _PYYAML_TIMESTAMP.fullmatch(text)
        if match is None:
            raise _Unsupported
        year, month, day, hour, minute, second, zone = match.groups()
        try:
            datetime.datetime(
                int(year),
                int(month),
                int(day),
                int(hour or 0),
                int(minute or 0),
                int(second or 0),
            )
        except ValueError:
            raise _Unsupported from None
        if zone is not None and int(zone) >= 24:
            raise _Unsupported
        return
    if kind == "binary":
        try:
            base64.decodebytes(text.encode("ascii"))
        except (UnicodeEncodeError, binascii.Error):
            raise _Unsupported from None
        return
    raise _Unsupported


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


def _block_scalar(
    source: str, index: int, parent: int = 0, *, final_break: bool = False
) -> tuple[str, int]:
    """Read a literal (``|``) or folded (``>``) block scalar.

    ``parent`` is the column of the key or ``-`` that holds it. With
    ``final_break``, a last line at the end of ``source`` keeps a line break,
    as the ``yaml`` package reads it; PyYAML does not add one.
    """

    header = _BLOCK_HEADER.match(source, index)
    if header is None:
        raise _NotYaml
    folded = source[index] == ">"
    chomping = header.group(1) or header.group(4)
    increment = header.group(2) or header.group(3)
    index = min(header.end() + 1, len(source))
    breaks: list[str] = []
    if increment:
        # The indicator counts from the column of what holds the scalar.
        indent = parent + int(increment)
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
        indent = max(parent + 1, widest)
    chunks: list[str] = []
    line_break = ""
    while index < len(source) and _column(source, index) == indent:
        chunks.extend(breaks)
        starts_with_text = source[index] not in " \t"
        end = _line_end(source, index)
        chunks.append(source[index:end])
        line_break = "\n" if end < len(source) or final_break else ""
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
# mapping becomes ``JSON.stringify(value)``. A null is dropped, and the line
# fallback (``parseLineFrontmatter``) takes its place. Aliases give the
# anchored value. ``loadSingleSkillDirectory`` trims the name again and uses
# the trimmed folder name when the result is empty. Values are held here as
# JavaScript holds them: ``None`` for null, ``bool``, ``float`` for every
# number, ``str``, ``list``, and ``dict`` with string keys. Explicit (``?``)
# and collection keys, block scalars with an indentation indicator inside
# collections, and the ``!!timestamp``, ``!!binary``, and ``!!merge`` tags
# are not derived; a name that uses them is uncertain.


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


def _openclaw_text(
    value: object, key: str, line: str, fallback: str | None
) -> str | None:
    """Return the text OpenClaw keeps for a top-level value, or ``None`` for null.

    ``line`` is the key's line and ``fallback`` the key's line fallback.
    """

    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return _js_number(value)
    if isinstance(value, str):
        return _js_trim(value)
    # A collection written on the key's line with a colon in it keeps the
    # line fallback instead.
    match = _LINE_FIELD.fullmatch(line)
    if (
        fallback is not None
        and match is not None
        and match.group(1) == key
        and ":" in match.group(2)
    ):
        return fallback
    return _js_json(value)


def _line_fields(block: list[str]) -> dict[str, str]:
    """Return OpenClaw's line fallback for every key (``parseLineFrontmatter``).

    Each unquoted ``key:`` line gives the rest of the line without one pair of
    enclosing quotes or, when it is empty and the next line is indented, the
    lines up to the next one that is not. A later line wins.
    """

    fields: dict[str, str] = {}
    index = 0
    while index < len(block):
        match = _LINE_FIELD.fullmatch(block[index])
        if match is not None:
            value = _js_trim(match.group(2))
            following = block[index + 1] if index + 1 < len(block) else ""
            if not value and following[:1] in (" ", "\t"):
                lines: list[str] = []
                while index + 1 < len(block):
                    line = block[index + 1]
                    if line and line[0] not in " \t":
                        break
                    lines.append(line)
                    index += 1
                value = _js_trim("\n".join(lines))
            elif value and value[0] in "'\"" and value[-1] == value[0]:
                value = value[1:-1]
            if value:
                fields[match.group(1)] = value
        index += 1
    return fields


def _js_node(
    source: str, index: int, anchors: _Anchors | None = None
) -> tuple[object, int]:
    """Return the YAML 1.2 value after ``index``, and the end of a quoted value.

    Like :func:`_node`, the end is ``index`` for values that are not quoted.
    Anchors are recorded in ``anchors``, and aliases resolved from it.
    """

    position = _next_token(source, index)
    tag: str | None = None
    anchor: tuple[str, int] | None = None
    after = index
    while position is not None and source[position] in "&!":
        after = _property_end(source, position)
        if source[position] == "!":
            tag = source[position:after]
        else:
            anchor = (source[position + 1 : after], position)
        position = _next_token(source, after)
    raw: _Raw | None = None
    if position is None:
        # A block sequence may start at column 0 below the key.
        following = _js_content(source, after)
        if following is not None and _sequence_entry(source, following):
            value, end = _js_block_sequence(source, following, anchors)
            if not _only_comments_follow(source, end):
                raise _NotYaml
            _define(anchors, anchor, value, None)
            return value, index
        raw = ("", tag, True)
        value = _js_scalar_node(anchors, raw)
        _define(anchors, anchor, value, raw)
        return value, index
    char, following = source[position], source[position + 1 : position + 2]
    below = "\n" in source[index:position]
    # Properties on the line of an implicit key belong to the key.
    on_key = anchor is not None and "\n" not in source[anchor[1] : position]
    quoted = False
    if char in "|>":
        text, end = _block_scalar(source, position, final_break=_yaml_package(anchors))
        raw = (text, tag, False)
        value = _js_scalar_node(anchors, raw)
    elif char in "'\"":
        scalar = _quoted(source, position)
        if scalar is None or re.search(r"\n\S", source[position : scalar[1]]):
            # The yaml package needs every line of a quoted scalar indented.
            raise _NotYaml
        text, end = scalar
        if below and _key_follows(source, end):
            if on_key:
                raise _Unsupported
            value, end = _js_block_mapping(source, position, anchors)
        else:
            raw = (text, tag, False)
            value, quoted = _js_scalar_node(anchors, raw), True
    elif char == "*":
        if tag is not None or anchor is not None:
            raise _NotYaml
        value, end = _js_alias(source, position, anchors)
        if _key_follows(source, end):
            raise _Unsupported
    elif char in "[{":
        value, end = _js_flow(source, position, 0, anchors)
    elif char == "-" and following in _BLANK:
        if not below:
            raise _NotYaml
        value, end = _js_block_sequence(source, position, anchors)
    elif char in "?:" and following in _BLANK:
        raise _Unsupported
    elif char in ",]}%@`\t":
        raise _NotYaml
    else:
        text, end = _plain(source, position)
        if _key_follows(source, end):
            if not below:
                raise _NotYaml
            if on_key:
                raise _Unsupported
            value, end = _js_block_mapping(source, position, anchors)
        else:
            raw = (text, tag, True)
            value = _js_scalar_node(anchors, raw)
    if not _only_comments_follow(source, end):
        raise _NotYaml
    _define(anchors, anchor, value, raw)
    return value, end if quoted else index


def _define(
    anchors: _Anchors | None,
    anchor: tuple[str, int] | None,
    value: object,
    raw: _Raw | None,
) -> None:
    if anchors is not None:
        anchors.define(anchor, value, raw)


def _js_alias(source: str, index: int, anchors: _Anchors | None) -> tuple[object, int]:
    """Return the value of the alias at ``index``, and its end."""

    end = _property_end(source, index)
    label = source[index + 1 : end]
    if not label:
        raise _NotYaml
    if anchors is None:
        raise _Unsupported
    value, _ = anchors.resolve(label, index)
    return value, end


def _yaml_package(anchors: _Anchors | None) -> bool:
    """Whether values are read as the ``yaml`` package reads them."""

    return anchors is None or anchors.style == "openclaw"


def _js_scalar_node(anchors: _Anchors | None, raw: _Raw) -> object:
    """Return the value of a scalar node.

    Under the pyyaml style only anchors are needed from this parser, so a
    scalar it does not resolve is kept as ``None``.
    """

    text, tag, plain = raw
    try:
        return _js_scalar(text, tag, plain=plain)
    except _Unsupported:
        if anchors is not None and anchors.style == "pyyaml":
            return None
        raise


def _property_end(source: str, index: int) -> int:
    """Return the end of the anchor, tag, or alias that starts at ``index``."""

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


def _js_block_sequence(
    source: str, index: int, anchors: _Anchors | None = None
) -> tuple[list[object], int]:
    """Read the block sequence whose first ``-`` is at ``index``."""

    indent = _column(source, index)
    items: list[object] = []
    while True:
        value, index = _js_entry(source, index + 1, indent, True, anchors)
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


def _js_block_mapping(
    source: str, index: int, anchors: _Anchors | None = None
) -> tuple[dict[str, object], int]:
    """Read the block mapping whose first key starts at ``index``."""

    indent = _column(source, index)
    pairs: dict[str, object] = {}
    seen: set[tuple[str, object]] = set()
    while True:
        key, index = _js_block_key(source, index, indent)
        value, index = _js_entry(source, index, indent, False, anchors)
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
    source: str,
    index: int,
    indent: int,
    sequence: bool,
    anchors: _Anchors | None,
) -> tuple[object, int]:
    """Read the value after a block ``-`` (``sequence``) or a block key's ``:``.

    ``indent`` is the column of the entry's ``-`` or key.
    """

    while source[index : index + 1] in (" ", "\t"):
        index += 1
    tag: str | None = None
    anchor: tuple[str, int] | None = None
    while index < len(source) and source[index] in "&!":
        end = _property_end(source, index)
        if source[index] == "!":
            tag = source[index:end]
        else:
            anchor = (source[index + 1 : end], index)
        index = end
        while source[index : index + 1] in (" ", "\t"):
            index += 1
    properties = tag is not None or anchor is not None
    if index < len(source) and source[index] not in "\n#":
        if _sequence_entry(source, index) and not sequence:
            raise _NotYaml
        value, end, raw = _js_inline(
            source, index, indent, tag, sequence, anchors, properties=properties
        )
        if anchor is not None and isinstance(value, dict) and source[index] != "{":
            # The anchor belongs to the first key of a compact mapping.
            raise _Unsupported
        _define(anchors, anchor, value, raw)
        return value, end
    following = _js_content(source, index)
    if following is not None:
        column = _column(source, following)
        if column > indent:
            value, end, raw = _js_inline(
                source, following, indent, tag, True, anchors, properties=properties
            )
            _define(anchors, anchor, value, raw)
            return value, end
        if column == indent and not sequence and _sequence_entry(source, following):
            # A mapping value may be a sequence at the key's own indentation.
            value, end = _js_block_sequence(source, following, anchors)
            _define(anchors, anchor, value, None)
            return value, end
    raw = ("", tag, True)
    value = _js_scalar_node(anchors, raw)
    _define(anchors, anchor, value, raw)
    return value, index


def _js_inline(
    source: str,
    index: int,
    indent: int,
    tag: str | None,
    compact: bool,
    anchors: _Anchors | None,
    *,
    properties: bool,
) -> tuple[object, int, _Raw | None]:
    """Read the node at ``index`` inside a block collection at ``indent``.

    With ``compact``, a key at ``index`` starts a block mapping there.
    Returns the value, its end, and for a scalar its raw parts.
    """

    char, following = source[index], source[index + 1 : index + 2]
    if _sequence_entry(source, index):
        return (*_js_block_sequence(source, index, anchors), None)
    if char in "[{":
        value, end = _js_flow(source, index, indent, anchors)
        if _key_follows(source, end):
            raise _Unsupported
        return value, end, None
    if char in "'\"":
        scalar = _quoted(source, index)
        if scalar is None:
            raise _NotYaml
        text, end = scalar
        if _key_follows(source, end):
            if not compact or "\n" in source[index:end]:
                raise _NotYaml
            return (*_js_block_mapping(source, index, anchors), None)
        raw: _Raw = (text, tag, False)
        return _js_scalar_node(anchors, raw), end, raw
    if char == "*":
        if properties:
            raise _NotYaml
        value, end = _js_alias(source, index, anchors)
        if _key_follows(source, end):
            raise _Unsupported
        return value, end, None
    if char in "|>":
        text, end = _block_scalar(
            source, index, indent, final_break=_yaml_package(anchors)
        )
        raw = (text, tag, False)
        return _js_scalar_node(anchors, raw), end, raw
    if char in "?:" and following in _BLANK:
        raise _Unsupported
    if char in ",]}#%@`":
        raise _NotYaml
    text, end = _plain(source, index, indent)
    if _key_follows(source, end):
        if not compact or "\n" in source[index:end]:
            raise _NotYaml
        return (*_js_block_mapping(source, index, anchors), None)
    raw = (text, tag, True)
    return _js_scalar_node(anchors, raw), end, raw


def _js_flow(
    source: str, index: int, indent: int, anchors: _Anchors | None = None
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
        node, index, plain = _js_flow_node(source, index, indent, anchors)
        index = _js_flow_space(source, index, indent)
        if source[index] == ":" and (
            not plain or source[index + 1 : index + 2] in _FLOW_END
        ):
            index = _js_flow_space(source, index + 1, indent)
            value: object = None
            if source[index] not in ",]}":
                value, index, _ = _js_flow_node(source, index, indent, anchors)
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


def _js_flow_node(
    source: str, index: int, indent: int, anchors: _Anchors | None = None
) -> tuple[object, int, bool]:
    """Read one node inside a flow collection.

    Returns the value, its end, and whether it is a plain scalar.
    """

    tag: str | None = None
    anchor: tuple[str, int] | None = None
    properties = False
    while source[index] in "&!":
        end = _property_end(source, index)
        if source[index] == "!":
            tag = source[index:end]
        else:
            anchor = (source[index + 1 : end], index)
        properties = True
        index = _js_flow_space(source, end, indent)
    char, following = source[index], source[index + 1 : index + 2]
    if char in ",]}" or (char == ":" and following in _FLOW_END):
        if not properties:
            raise _NotYaml
        raw: _Raw = ("", tag, True)
        value = _js_scalar_node(anchors, raw)
        _define(anchors, anchor, value, raw)
        return value, index, True
    if char in "[{":
        value, end = _js_flow(source, index, indent, anchors)
        _define(anchors, anchor, value, None)
        return value, end, False
    if char in "'\"":
        scalar = _quoted(source, index)
        if scalar is None:
            raise _NotYaml
        raw = (scalar[0], tag, False)
        value = _js_scalar_node(anchors, raw)
        _define(anchors, anchor, value, raw)
        return value, scalar[1], False
    if char == "*":
        if properties:
            raise _NotYaml
        value, end = _js_alias(source, index, anchors)
        return value, end, False
    if char in "#|>%@`" or (char in "-?:" and following in _FLOW_END):
        raise _NotYaml
    text, end = _js_flow_plain(source, index, indent)
    raw = (text, tag, True)
    value = _js_scalar_node(anchors, raw)
    _define(anchors, anchor, value, raw)
    return value, end, True


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
