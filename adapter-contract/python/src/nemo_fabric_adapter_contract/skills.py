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
   folder holding ``SKILL.md``.
4. A :data:`CandidateReader` then reads every candidate in one call and
   returns a :class:`CandidateInfo` for each: its name, and whether the
   harness accepts it. Adapters pass a reader backed by the harness's own
   parser, so names and validity match the harness; the default
   :func:`front_matter_reader` reads only simple front matter. A candidate
   the harness refuses is skipped and recorded before any duplicate check,
   so it never counts as a copy of a name.
5. An empty ``folder`` entry is valid and recorded in ``empty_entries``.
6. The same folder reached through several entries loads once.
7. Skills that share a name are recorded, or fail or are resolved according
   to the configured duplicate policy.

A skill's name is the name the reader returns, or its folder name when the
reader returns none and the rules allow the fallback.

A reader that cannot tell a skill's name marks it ``name_uncertain``. Without
a duplicate policy the skill keeps its name, or the folder name fallback, and
is reported; under an explicit policy, where a wrong name could pick the wrong
copy, the entry fails with ``unsupported_frontmatter_name``.

Likewise, a reader that cannot tell whether the harness accepts a skill
returns ``valid=None``. The skill is kept, as the harness may load it. Under
an explicit policy, when the outcome for a name would depend on whether that
skill loads, the entry fails with ``unsupported_frontmatter`` instead of
deciding on a guess.

The module never reports file contents.
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


@dataclass(frozen=True)
class CandidateInfo:
    """What the harness reads from one candidate's ``SKILL.md``."""

    name: str | None
    """The name the harness reads from the front matter, or ``None`` when it
    reads none, so the folder name fallback applies if the rules allow it."""
    valid: bool | None = True
    """Whether the harness loads the skill: ``True`` when it does, ``False``
    when it refuses it, and ``None`` when the reader cannot tell."""
    reason: str | None = None
    """Why the harness refuses the skill, recorded as the skip reason.
    Required when ``valid`` is ``False``. Use ``unreadable`` for a
    ``SKILL.md`` that cannot be read."""
    name_uncertain: bool = False
    """Whether the reader cannot tell the name the harness reads. ``name`` is
    then a best guess, or ``None`` for the folder name fallback."""

    def __post_init__(self) -> None:
        if self.name is not None and not isinstance(self.name, str):
            raise ValueError("CandidateInfo.name must be a string or None")
        if self.valid is not None and not isinstance(self.valid, bool):
            raise ValueError("CandidateInfo.valid must be True, False, or None")
        if self.valid is False and not self.reason:
            raise ValueError("CandidateInfo.reason is required when valid is False")


CandidateReader = Callable[[Sequence[Path]], Mapping[Path, CandidateInfo]]
"""Read every candidate in one call.

The reader receives each candidate's resolved skill folder (the folder that
holds ``SKILL.md``) once, in discovery order, and returns a
:class:`CandidateInfo` for each of those folders. :func:`load_directories`
calls it at most once, and not at all when there are no candidates.
"""


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
    name_fallback: bool = True
    """Use the folder name when the reader returns no name. When false, a
    skill without a name is skipped with ``no_name``."""
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
    name_uncertain: bool = False
    """Whether the front matter name could not be derived reliably, so the
    name may differ from the harness's."""
    validity_uncertain: bool = False
    """Whether the reader could not tell if the harness accepts the skill
    (:attr:`CandidateInfo.valid` is ``None``), so the harness may refuse it."""


@dataclass(frozen=True)
class SkippedCandidate:
    """A folder holding ``SKILL.md`` that does not load."""

    path: Path
    """Resolved candidate folder."""
    field: str
    """The entry the candidate was found through."""
    reason: str
    """Why it was skipped: ``no_name``, ``path_escape``, or the
    :attr:`CandidateInfo.reason` of a candidate the harness refuses, such as
    ``unreadable`` or ``missing_frontmatter_field:<field>``."""


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


@dataclass(frozen=True)
class _Entry:
    """An entry after its checks and discovery, before any candidate is read."""

    field: str
    kind: SkillEntryKind
    key_field: str
    root: Path
    rules: DiscoveryRules
    candidates: list[tuple[_Candidate, bool]]
    """Each candidate, and whether it escapes the entry's folder."""


def load_directories(
    directories: Sequence[Mapping[str, str | os.PathLike[str]]]
    | Sequence[SkillDirectory],
    *,
    rules: DiscoveryRules | RulesForEntry,
    duplicates: DuplicatePolicy | None,
    base_dir: str | os.PathLike[str] | None = None,
    reader: CandidateReader | None = None,
) -> DirectoryLoad:
    """Check ``directories`` and return the skills they supply.

    Relative entries resolve against ``base_dir``, or the current directory
    when it is ``None``. ``~`` and environment variables are not expanded.

    ``rules`` is either one :class:`DiscoveryRules` for every entry, or a
    callable that receives each entry's index, kind (``"skill"`` or
    ``"folder"``), and resolved folder after the entry passes its checks, and
    returns the rules for that entry.

    Every entry is checked and its candidates discovered first. Then
    ``reader`` reads every candidate in one call (see :data:`CandidateReader`);
    the default is :func:`front_matter_reader`. Candidates the reader says the
    harness refuses are skipped before any duplicate check, so they never
    count as a copy of a name.

    A folder reached through several entries loads once. Under ``last_wins``
    it loads at the position of, and is attributed to, the last entry that
    reaches it; otherwise it keeps its first position.

    A skill whose name the reader marks uncertain keeps its name, or the
    folder name fallback, and is marked ``name_uncertain`` when
    ``duplicates`` is ``None``; under an explicit policy its entry fails with
    ``unsupported_frontmatter_name``. A skill whose validity the reader cannot
    tell is kept and marked ``validity_uncertain``; under an explicit policy,
    when the outcome for its name depends on whether it loads, its entry
    fails with ``unsupported_frontmatter``.

    Raises:
        SkillLoadError: If an entry is malformed, missing, not a folder, not
            accessible, or cannot be read, if skills share a name under an
            explicit ``duplicates`` policy, if a skill's name is uncertain
            under one, or if the outcome of one depends on a skill whose
            validity is uncertain.
        TypeError: If ``rules`` or ``reader`` is not callable as documented,
            or the reader returns no :class:`CandidateInfo` for a candidate.
    """

    if duplicates is not None and duplicates not in get_args(SkillDuplicatePolicy):
        raise ValueError(f"unknown skill duplicate policy: {duplicates!r}")
    if not isinstance(rules, DiscoveryRules) and not callable(rules):
        raise TypeError("rules must be DiscoveryRules or a callable returning them")
    if reader is None:
        reader = front_matter_reader()
    elif not callable(reader):
        raise TypeError("reader must be a callable returning CandidateInfo")
    load = DirectoryLoad()
    entries = [
        _discover_entry(index, value, rules, base_dir, load)
        for index, value in enumerate(directories)
    ]
    infos = _read_candidates(reader, entries)

    found: list[FoundSkill | None] = []
    # Position in ``found`` of each folder that loads, by identity.
    loaded: dict[tuple[int, int] | Path, int] = {}
    for entry in entries:
        named = _name_candidates(entry, infos, duplicates, load.skipped)
        if entry.kind == "folder" and not named:
            load.empty_entries.append(entry.field)
        if duplicates is not None:
            # Check the entry's own skills before another entry can take
            # one of them over.
            _reject_duplicates_in_entry(
                [(name, candidate) for name, candidate, _ in named],
                entry.field,
                {
                    candidate.identity
                    for _, candidate, info in named
                    if info.valid is None
                },
                duplicates,
            )
        for name, candidate, info in named:
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
                    field=entry.field,
                    kind=entry.kind,
                    name_uncertain=info.name_uncertain,
                    validity_uncertain=info.valid is None,
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


def _discover_entry(
    index: int,
    value: SkillDirectory | Mapping[str, str | os.PathLike[str]],
    rules: DiscoveryRules | RulesForEntry,
    base_dir: str | os.PathLike[str] | None,
    load: DirectoryLoad,
) -> _Entry:
    """Check one entry and discover its candidates, without reading them."""

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
    escapes = not entry_rules.allow_symlink_escape
    return _Entry(
        entry_field,
        kind,
        key_field,
        root,
        entry_rules,
        [
            (candidate, escapes and not _within(candidate.resolved, root))
            for candidate in candidates
        ],
    )


def _read_candidates(
    reader: CandidateReader, entries: list[_Entry]
) -> Mapping[Path, CandidateInfo]:
    """Read every candidate that does not escape its entry in one call."""

    folders = list(
        dict.fromkeys(
            candidate.resolved
            for entry in entries
            for candidate, escaped in entry.candidates
            if not escaped
        )
    )
    if not folders:
        return {}
    infos = reader(folders)
    if not isinstance(infos, Mapping):
        raise TypeError("the skill reader must return a mapping of CandidateInfo")
    for folder in folders:
        if not isinstance(infos.get(folder), CandidateInfo):
            raise TypeError(f"the skill reader returned no CandidateInfo for {folder}")
    return infos


def _name_candidates(
    entry: _Entry,
    infos: Mapping[Path, CandidateInfo],
    duplicates: DuplicatePolicy | None,
    skipped: list[SkippedCandidate],
) -> list[tuple[str, _Candidate, CandidateInfo]]:
    """Return the entry's candidates that load, with their names.

    Records the candidates that are skipped, and fails on an uncertain name
    under an explicit duplicate policy, where a wrong name could pick the
    wrong copy.
    """

    named: list[tuple[str, _Candidate, CandidateInfo]] = []
    for candidate, escaped in entry.candidates:
        if escaped:
            skipped.append(
                SkippedCandidate(candidate.resolved, entry.field, "path_escape")
            )
            continue
        info = infos[candidate.resolved]
        if info.valid is False:
            assert info.reason is not None
            if (
                info.reason == "unreadable"
                and entry.kind == "skill"
                and candidate.resolved == entry.root
            ):
                skill_file = candidate.resolved / SKILL_FILE
                raise SkillLoadError(
                    "invalid",
                    f"NeMo Fabric skill entry {entry.key_field} has a {SKILL_FILE} "
                    f"that cannot be read: {skill_file}",
                    field=entry.key_field,
                    paths=(skill_file,),
                )
            skipped.append(
                SkippedCandidate(candidate.resolved, entry.field, info.reason)
            )
            continue
        name = info.name
        if not name and entry.rules.name_fallback:
            reached_name = candidate.reached.name
            name = (
                reached_name
                if reached_name not in ("", "..")
                else candidate.resolved.name
            )
        if not name:
            skipped.append(SkippedCandidate(candidate.resolved, entry.field, "no_name"))
            continue
        if info.name_uncertain and duplicates is not None:
            skill_file = candidate.resolved / SKILL_FILE
            raise SkillLoadError(
                "unsupported_frontmatter_name",
                f"NeMo Fabric cannot read the front matter name of {skill_file} "
                f"reliably, so the {duplicates} duplicate policy in {entry.field} "
                f"cannot be applied (unsupported_frontmatter_name); write the "
                f"name as a plain name: value",
                field=entry.field,
                paths=(candidate.resolved,),
            )
        named.append((name, candidate, info))
    return named


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
        f"policy in {field} cannot be applied (unsupported_frontmatter); make "
        f"the harness's own skill parser available to the adapter, or remove "
        f"the policy",
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


# A top-level ``key: value`` line. The key may be quoted or followed by spaces.
_KEY = re.compile(r"([^\s#:\-?][^:]*?)[ \t]*:(?:[ \t]+(.*?))?[ \t]*")
_COMMENT = re.compile(r"[ \t]+#")
# Plain values YAML reads as null.
_NULLS = frozenset({"", "~", "null", "Null", "NULL"})
# Top-level lines that may define a key this reader does not read: an
# explicit key, an anchored, tagged, or alias key, or a flow collection.
_UNREAD_KEYS = tuple("?&!*{[")
# Value starts this reader does not read: a block scalar, an anchor, alias,
# or tag, a flow collection, a sequence, or a reserved indicator.
_UNREAD_VALUES = (*"|>&*![{@`%", "- ")


def front_matter_reader(*, required_fields: Sequence[str] = ()) -> CandidateReader:
    """Return a stdlib :data:`CandidateReader` for simple front matter.

    The reader takes a single-line ``name:`` value from the ``---`` front
    matter block, plain or quoted, and treats a missing, empty, or null value
    as no name. A candidate without a top-level key in ``required_fields`` is
    refused with ``missing_frontmatter_field:<key>``, and an unreadable
    ``SKILL.md`` with ``unreadable``. It never refuses front matter otherwise.

    YAML it does not read, such as a block scalar, anchor, alias, tag, flow
    collection, continuation line, escape sequence, or a ``name`` key it
    cannot see, marks the name uncertain rather than guessing it. Adapters
    whose harness parses YAML should pass a reader backed by that parser.
    """

    required = tuple(required_fields)

    def read(folders: Sequence[Path]) -> Mapping[Path, CandidateInfo]:
        return {
            folder: _read_simple(folder / SKILL_FILE, required) for folder in folders
        }

    return read


def _read_simple(skill_file: Path, required: tuple[str, ...]) -> CandidateInfo:
    try:
        with skill_file.open(encoding="utf-8-sig", errors="replace") as lines:
            block = _front_matter(lines)
    except OSError:
        return CandidateInfo(None, valid=False, reason="unreadable")
    keys: set[str] = set()
    name: str | None = None
    uncertain = False
    key: str | None = None
    for line in block:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[0] in " \t":
            # A continuation of the name, or an indented mapping that may hold it.
            uncertain = uncertain or key in (None, "name")
            continue
        match = None if line.startswith(_UNREAD_KEYS) else _KEY.fullmatch(line)
        if match is None:
            uncertain = uncertain or line.startswith(_UNREAD_KEYS)
            key = ""
            continue
        key = match.group(1)
        if len(key) > 1 and key[0] == key[-1] and key[0] in "'\"":
            key = key[1:-1]
        if key == "name":
            # PyYAML keeps a repeated key's last value; other parsers refuse it.
            repeated = "name" in keys
            name, value_uncertain = _simple_value(match.group(2) or "")
            uncertain = uncertain or repeated or value_uncertain
        # PyYAML merges a ``<<`` mapping's keys in.
        uncertain = uncertain or key == "<<"
        keys.add(key)
    for field in required:
        if field not in keys:
            return CandidateInfo(
                name, valid=False, reason=f"missing_frontmatter_field:{field}"
            )
    return CandidateInfo(name, name_uncertain=uncertain)


def _simple_value(text: str) -> tuple[str | None, bool]:
    """Return a single-line value, or ``None`` for none, and whether it is uncertain."""

    if text[:1] in ("'", '"'):
        quote = text[0]
        end = text.rfind(quote)
        rest = text[end + 1 :].strip()
        value = text[1:end]
        if end == 0 or (rest and not rest.startswith("#")):
            return None, True
        if quote == "'":
            if "'" in value.replace("''", ""):
                return None, True
            value = value.replace("''", "'")
        elif "\\" in value or '"' in value:
            return None, True
        return value or None, False
    if text.startswith("#"):
        return None, False
    value = _COMMENT.split(text, maxsplit=1)[0]
    # A plain value with ``: `` or a trailing ``:`` is not valid YAML.
    if value.startswith(_UNREAD_VALUES) or value == "-" or ": " in f"{value} ":
        return None, True
    return (None if value in _NULLS else value), False


def _front_matter(lines: Iterable[str]) -> list[str]:
    """Return the lines of a ``---`` front matter block, or none without one."""

    lines = iter(lines)
    if next(lines, "").rstrip() != "---":
        return []
    block: list[str] = []
    for line in lines:
        line = line.rstrip("\r\n")
        if line.rstrip() == "---":
            return block
        block.append(line.rstrip())
    # Without its closing line the block is not front matter.
    return []


def _within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)
