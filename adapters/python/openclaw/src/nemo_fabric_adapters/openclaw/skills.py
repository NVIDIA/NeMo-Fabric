# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map ``skills.directories`` and ``skills.bundled`` onto OpenClaw's native settings.

OpenClaw 2026.9.34 loads skills from these sources, highest precedence first:
``<workspace>/skills``, ``<workspace>/.agents/skills``, ``<state>/skills``, the
agent's Skill Workshop folder, the shipped (bundled and Custodian) skills, and
``skills.load.extraDirs`` with plugin skills. A higher source wins a name;
inside ``extraDirs`` the later root wins. See
``dist/workspace-skill-sources-*.mjs`` and ``mergeSkillRecords`` in
``dist/workspace-skill-loader-*.mjs`` of the installed package.

Discovery inside an ``extraDirs`` root (``discoverSkillCandidates`` in
``dist/skill-root-discovery-*.mjs``), reproduced with one
:func:`~nemo_fabric_adapter_contract.skills.load_directories` call whose
rules are chosen per entry:

- A root that holds ``SKILL.md`` is one skill; nothing below it loads. A
  ``folder`` entry like that fails start with ``folder_is_skill``.
- Otherwise OpenClaw scans the root, or its ``skills/`` subfolder when only
  that subfolder holds skills (``resolveNestedSkillsRoot``). It stops at the
  first ``SKILL.md`` on each path and skips folders whose name starts with
  ``.`` and ``node_modules``.
- The scanned folder is searched 2 levels deep, or 6 when it is named
  ``skills``. Its own child named ``skills`` restarts the count and is
  searched 6 levels deep, whatever the scanned folder is called
  (``max_depth`` plus ``subroot_depths``).
- A candidate whose real path is outside the scanned folder is skipped
  (``resolveContainedSkillPath``), and so is every candidate when the
  ``skills/`` subfolder itself leads outside the root. Both are reported with
  ``path_escape``, with or without a duplicate policy, so passing a selected
  skill's resolved folder never loads a skill OpenClaw refuses. The adapter
  writes no ``skills.load.allowSymlinkTargets``, so no target is trusted.
- A skill needs a non-empty ``description``; its name is the front-matter
  ``name`` or the folder name. Front matter that YAML rejects is skipped.
  ``openclaw skills list`` at start reports the skills OpenClaw refuses; under
  ``first_wins`` or ``last_wins`` the next copy replaces a refused winner.

``skills.allowBundled`` filters bundled and Custodian skills after names are
resolved, so a filtered-out bundled skill still hides a lower-precedence
copy. An empty list means no filter, so selecting no bundled skills uses a
list that names no shipped skill. Custodian skills load only for the agent
that owns them (``resolveCustodianSkillAgentId``).
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from dataclasses import replace
from pathlib import Path
from typing import Any

from nemo_fabric_adapter_contract import models as contract
from nemo_fabric_adapter_contract import skills as contract_skills
from nemo_fabric_adapters.common import lifecycle


SKILL_FILE = contract_skills.SKILL_FILE
# MAX_CONFIGURED_ROOT_GROUPED_SKILL_SCAN_DEPTH and MAX_GROUPED_SKILL_SCAN_DEPTH.
EXTRA_ROOT_DEPTH = 2
SKILLS_ROOT_DEPTH = 6
# A non-empty allowBundled list that matches no shipped skill selects none.
NO_BUNDLED_SKILLS = "nemo-fabric:no-bundled-skills"
BUNDLED_SOURCES = frozenset({"openclaw-bundled", "openclaw-custodian"})
# normalizeAgentId: invalid characters become "-", outer dashes are trimmed.
_AGENT_ID_INVALID = re.compile(r"[^a-z0-9_-]+")
# Native sources that outrank skills.load.extraDirs, lowest first.
_LOCAL_SKILL_FOLDERS = (Path(".agents") / "skills", Path("skills"))


@dataclass(frozen=True)
class BundledInventory:
    """Skills shipped with the installed OpenClaw package, by name."""

    skills: dict[str, Path]
    """``skills/`` of the package: loaded for every agent."""
    custodian: dict[str, Path]
    """``custodian-skills/``: loaded only for the configured Custodian agent."""

    @property
    def names(self) -> set[str]:
        """Return every shipped skill name."""

        return {*self.skills, *self.custodian}

    def path(self, name: str) -> Path | None:
        """Return the folder of a shipped skill."""

        return self.skills.get(name) or self.custodian.get(name)


@dataclass
class SkillPlan:
    """OpenClaw settings and the ``output.skills`` report for one runtime."""

    extra_dirs: list[str] = dataclass_field(default_factory=list)
    allow_bundled: list[str] | None = None
    load: contract_skills.DirectoryLoad | None = None
    bundled_selected: list[str] | None = None
    bundled_unavailable: list[dict[str, Any]] = dataclass_field(default_factory=list)
    inventory: BundledInventory | None = None
    custodian: bool = True
    """Whether OpenClaw loads the Custodian skills for the configured agent."""
    duplicates: contract.SkillDuplicatePolicy | None = None
    rejected: set[Path] = dataclass_field(default_factory=set)
    """Configured skill folders that ``openclaw skills list`` showed it refuses."""

    @property
    def needs_status(self) -> bool:
        """Return whether OpenClaw's own skill list can refine the report."""

        return self.load is not None or self.bundled_selected is not None

    def report(self) -> dict[str, Any] | None:
        """Return the ``output.skills`` block, or ``None`` when there is nothing to say."""

        bundled: dict[str, Any] | None = None
        if self.bundled_selected is not None:
            unavailable = {item["name"] for item in self.bundled_unavailable}
            bundled = {
                "selected": [
                    name for name in self.bundled_selected if name not in unavailable
                ]
            }
            if self.bundled_unavailable:
                bundled["unavailable"] = list(self.bundled_unavailable)
        load = self.load if self.load is not None else contract_skills.DirectoryLoad()
        return contract_skills.report_block(load, bundled=bundled)

    def warn(self) -> None:
        """Write one stderr line per skipped, empty, duplicate, or unavailable item."""

        if self.load is not None:
            contract_skills.warn("OpenClaw", self.load)
        if self.bundled_unavailable:
            items = ", ".join(
                f"{item['name']} ({item['reason']})"
                for item in self.bundled_unavailable
            )
            print(
                f"OpenClaw: {len(self.bundled_unavailable)} selected bundled skills "
                f"are unavailable: {items}",
                file=sys.stderr,
                flush=True,
            )


def plan(
    config: contract.AgentConfig,
    *,
    base_dir: Path,
    workspace: Path,
    bundled_dir: Path | None,
    custodian: bool = True,
) -> SkillPlan:
    """Check the structured skill settings and return what OpenClaw should load.

    ``custodian`` says whether the configured agent owns the Custodian skills
    (:func:`owns_custodian_skills`).

    Raises:
        lifecycle.LifecycleError: If an entry cannot be loaded, a bundled
            skill name is unknown, or the bundled inventory cannot be found.
    """

    skills = config.skills
    result = SkillPlan(custodian=custodian)
    if skills is None:
        return result
    result.duplicates = skills.duplicates
    if skills.bundled is not None:
        if bundled_dir is None:
            raise lifecycle.LifecycleError(
                "openclaw_skill_invalid",
                "OpenClaw's shipped skills folder was not found, so skills.bundled "
                "cannot be applied",
                metadata={"field": "skills.bundled", "reason": "inventory_unavailable"},
            )
        result.inventory = bundled_inventory(bundled_dir)
        result.allow_bundled, result.bundled_selected = _select_bundled(
            skills.bundled, result.inventory
        )
        if not custodian:
            result.bundled_unavailable = [
                {"name": name, "reason": "custodian_agent_only"}
                for name in result.bundled_selected
                if name in result.inventory.custodian
                and name not in result.inventory.skills
            ]
    if skills.directories is None:
        return result
    load = _load_directories(
        skills.directories, duplicates=skills.duplicates, base_dir=base_dir
    )
    if skills.duplicates is None:
        load = _with_native_extra_outcomes(load)
        result.extra_dirs = _unique(
            path for _, _, path in _configured(load, base_dir, skills.directories)
        )
    else:
        result.extra_dirs = _unique(str(skill.path) for skill in load.selected)
    if result.inventory is None and bundled_dir is not None:
        result.inventory = bundled_inventory(bundled_dir)
    result.load = _with_native_tiers(
        load,
        workspace=workspace,
        inventory=result.inventory,
        allow_bundled=result.allow_bundled,
        custodian=custodian,
    )
    return result


def owns_custodian_skills(agents: Mapping[str, Any], agent_id: str) -> bool:
    """Return whether OpenClaw loads its Custodian skills for ``agent_id``.

    ``agents`` is the generated ``agents`` section. OpenClaw loads them for
    the ambient owner only (``tryResolveAmbientOwnerAgentId``):
    ``agents.defaults.systemAgent.agentId``, else the one entry marked
    ``default`` unless ownership is explicit, else the only entry.
    """

    defaults = agents.get("defaults")
    system = defaults.get("systemAgent") if isinstance(defaults, Mapping) else None
    owner = system.get("agentId") if isinstance(system, Mapping) else None
    if not isinstance(owner, str) or not owner.strip():
        entries = agents.get("entries")
        ids = (
            [key for key, value in entries.items() if isinstance(value, Mapping)]
            if isinstance(entries, Mapping)
            else []
        )
        marked = (
            [key for key in ids if entries[key].get("default") is True]
            if agents.get("ownership") != "explicit"
            else []
        )
        owner = marked[0] if len(marked) == 1 else ids[0] if len(ids) == 1 else None
    return owner is not None and _agent_key(owner) == _agent_key(agent_id)


def _agent_key(value: str) -> str:
    normalized = value.strip().lower()
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", normalized):
        return normalized
    return _AGENT_ID_INVALID.sub("-", normalized).strip("-")[:64] or "main"


def bundled_skills_dir(command: Path, env: Mapping[str, str]) -> Path | None:
    """Return the shipped skills folder OpenClaw resolves (``bundled-dir.ts``).

    OpenClaw honors ``OPENCLAW_BUNDLED_SKILLS_DIR`` first, then the ``skills``
    folder of the package that holds its command.
    """

    override = env.get("OPENCLAW_BUNDLED_SKILLS_DIR", "").strip()
    if override:
        return Path(override)
    try:
        resolved = command.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    for parent in resolved.parents:
        manifest = parent / "package.json"
        try:
            name = json.loads(manifest.read_text(encoding="utf-8")).get("name")
        except (OSError, ValueError, AttributeError):
            continue
        if name == "openclaw":
            candidate = parent / "skills"
            return candidate if _is_dir(candidate) else None
    return None


def bundled_inventory(bundled_dir: Path) -> BundledInventory:
    """Return the shipped skills by name, read with OpenClaw's parsing rules."""

    def names(folder: Path) -> dict[str, Path]:
        if not _is_dir(folder):
            return {}
        try:
            load = contract_skills.load_directories(
                [{"folder": folder}],
                rules=_rules(SKILLS_ROOT_DEPTH, _hidden_dir_names([folder])),
                duplicates=None,
            )
        except contract_skills.SkillLoadError:
            return {}
        return {skill.name: skill.path for skill in load.selected}

    return BundledInventory(
        skills=names(bundled_dir),
        custodian=names(bundled_dir.parent / "custodian-skills"),
    )


def apply_status(plan_: SkillPlan, status: Any, *, fallback: bool = False) -> bool:
    """Refine the report with ``openclaw skills list --json`` for the agent.

    Configured skills OpenClaw does not list were rejected by its loader, for
    example for front matter YAML cannot parse, and are reported as skipped
    with ``rejected_by_openclaw``. Selected bundled skills that do not load
    move to ``unavailable`` with OpenClaw's reason.

    Under ``first_wins`` or ``last_wins`` only the winning copy of a name is
    passed to OpenClaw, so a rejected winner would leave the name without a
    skill. With ``fallback``, the next copy in policy order replaces it in
    ``extra_dirs`` and the report; the caller must then write the new
    ``skills.load.extraDirs`` and list again. Without it, the name is
    reported with no selected copy.

    Returns:
        Whether ``extra_dirs`` changed.
    """

    rows = status.get("skills") if isinstance(status, dict) else None
    if not isinstance(rows, list):
        return False
    listed: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("name"), str):
            listed.setdefault(row["name"], []).append(row)
    changed = False
    load = plan_.load
    if load is not None:
        shadowed = {
            path
            for record in load.duplicates
            if record.outcome == "shadowed"
            for _, path, _ in record.copies
        }
        rejected = [
            skill
            for skill in load.selected
            if skill.name not in listed and skill.path not in shadowed
        ]
        if rejected:
            changed = _reject(plan_, load, rejected, fallback=fallback)
    unavailable: list[dict[str, Any]] = []
    for name in plan_.bundled_selected or []:
        candidates = listed.get(name, [])
        row = next(
            (item for item in candidates if item.get("source") in BUNDLED_SOURCES),
            None,
        )
        if row is None:
            if candidates:
                unavailable.append(
                    {
                        "name": name,
                        "reason": "shadowed",
                        "by": str(candidates[0].get("source", "unknown")),
                    }
                )
            elif (
                not plan_.custodian
                and plan_.inventory is not None
                and name in plan_.inventory.custodian
            ):
                unavailable.append({"name": name, "reason": "custodian_agent_only"})
            else:
                unavailable.append({"name": name, "reason": "not_loaded"})
        elif row.get("disabled") is True:
            unavailable.append({"name": name, "reason": "disabled"})
        elif row.get("eligible") is False:
            item: dict[str, Any] = {"name": name, "reason": "missing_requirements"}
            missing = row.get("missing")
            if isinstance(missing, dict):
                missing = {key: value for key, value in missing.items() if value}
                if missing:
                    item["missing"] = missing
            unavailable.append(item)
    plan_.bundled_unavailable = unavailable
    return changed


def _reject(
    plan_: SkillPlan,
    load: contract_skills.DirectoryLoad,
    rejected: list[contract_skills.FoundSkill],
    *,
    fallback: bool,
) -> bool:
    """Record skills OpenClaw refused, replacing a refused winner when allowed."""

    plan_.rejected.update(skill.path for skill in rejected)
    kinds = {field: kind for field, kind, _ in load.entries}
    order = {field: index for index, (field, _, _) in enumerate(load.entries)}
    selected = [skill for skill in load.selected if skill not in rejected]
    duplicates = list(load.duplicates)
    changed = False
    for skill in rejected:
        index = next(
            (
                position
                for position, record in enumerate(duplicates)
                if record.name == skill.name
                and record.outcome == "selected"
                and record.selected == skill.path
            ),
            None,
        )
        if index is None:
            continue
        record = duplicates[index]
        replacement = (
            _next_copy(record, skill.path, plan_.duplicates, plan_.rejected)
            if fallback
            else None
        )
        duplicates[index] = replace(
            record, selected=replacement[0] if replacement else None
        )
        if replacement is None:
            continue
        path, field = replacement
        selected.append(
            contract_skills.FoundSkill(
                skill.name,
                path,
                field,
                "skill" if kinds[field] == "skill" else "folder",
            )
        )
        plan_.extra_dirs = [
            str(path) if folder == str(skill.path) else folder
            for folder in plan_.extra_dirs
        ]
        changed = True
    selected.sort(key=lambda skill: order[skill.field])
    plan_.load = contract_skills.DirectoryLoad(
        selected=selected,
        entries=load.entries,
        empty_entries=load.empty_entries,
        skipped=[
            *load.skipped,
            *(
                contract_skills.SkippedCandidate(
                    skill.path, skill.field, "rejected_by_openclaw"
                )
                for skill in rejected
            ),
        ],
        duplicates=duplicates,
    )
    return changed


def _next_copy(
    record: contract_skills.DuplicateRecord,
    current: Path,
    policy: contract.SkillDuplicatePolicy | None,
    rejected: set[Path],
) -> tuple[Path, str] | None:
    """Return the copy the policy picks once ``current`` is refused."""

    copies = [
        (path, field) for origin, path, field in record.copies if origin == "external"
    ]
    position = next(
        (index for index, (path, _) in enumerate(copies) if path == current), None
    )
    if position is None or policy not in ("first_wins", "last_wins"):
        return None
    following = (
        copies[position + 1 :]
        if policy == "first_wins"
        else list(reversed(copies[:position]))
    )
    return next((copy for copy in following if copy[0] not in rejected), None)


def skill_error(error: contract_skills.SkillLoadError) -> lifecycle.LifecycleError:
    """Map a task-environment failure to the adapter's error codes."""

    metadata: dict[str, Any] = {"field": error.field, "reason": error.reason}
    if error.paths:
        metadata["path"] = str(error.paths[0])
    if len(error.paths) > 1:
        metadata["paths"] = [str(path) for path in error.paths]
    return lifecycle.LifecycleError(
        "openclaw_skill_not_found"
        if error.reason == "not_found"
        else "openclaw_skill_invalid",
        str(error),
        metadata=metadata,
    )


def _rules(max_depth: int, hidden: frozenset[str]) -> contract_skills.DiscoveryRules:
    return contract_skills.DiscoveryRules(
        traversal="recursive",
        nested_skills="stop_at_skill",
        max_depth=max_depth,
        required_frontmatter_fields=("description",),
        name_fallback=True,
        excluded_dir_names=frozenset({"node_modules"}) | hidden,
        allow_symlink_escape=False,
        subroot_depths={"skills": SKILLS_ROOT_DEPTH},
    )


def _select_bundled(
    bundled: contract.BundledSkills, inventory: BundledInventory
) -> tuple[list[str] | None, list[str]]:
    """Return ``skills.allowBundled`` and the selected names for ``bundled``."""

    known = inventory.names
    for name, values in (("include", bundled.include), ("exclude", bundled.exclude)):
        for value in values or []:
            if value not in known:
                raise lifecycle.LifecycleError(
                    "openclaw_skill_invalid",
                    f"OpenClaw does not ship a skill named {value!r}",
                    metadata={
                        "field": f"skills.bundled.{name}",
                        "reason": "unknown_bundled_skill",
                        "name": value,
                    },
                )
    if not bundled.enabled:
        return [NO_BUNDLED_SKILLS], []
    if bundled.include is not None:
        selected = sorted(set(bundled.include))
    elif bundled.exclude:
        selected = sorted(known - set(bundled.exclude))
    else:
        return None, sorted(known)
    return (selected or [NO_BUNDLED_SKILLS]), selected


def _load_directories(
    directories: Sequence[contract.SkillDirectory],
    *,
    duplicates: contract.SkillDuplicatePolicy | None,
    base_dir: Path,
) -> contract_skills.DirectoryLoad:
    """Discover the configured skills the way OpenClaw scans each root.

    Each ``folder`` entry is replaced by the folder OpenClaw scans and gets
    that folder's depth; entries are checked and reported in list order.
    """

    entries: list[contract.SkillDirectory] = []
    folders: dict[int, Path] = {}
    depths: dict[int, int] = {}
    # Entries whose skills/ subfolder OpenClaw would scan but which leads
    # outside the entry's folder: OpenClaw loads nothing from them.
    escaped: dict[int, Path] = {}
    for index, entry in enumerate(directories):
        if entry.folder is None:
            entries.append(entry)
            continue
        folder = base_dir / entry.folder
        folders[index] = folder
        scanned = _scanned_folder(folder)
        if scanned != folder and not _contained(scanned, folder):
            escaped[index] = scanned
            scanned = folder
        entries.append(contract.SkillDirectory(folder=str(scanned)))
        depths[index] = (
            SKILLS_ROOT_DEPTH if scanned.name == "skills" else EXTRA_ROOT_DEPTH
        )
    hidden = _hidden_dir_names(
        [base_dir / entry.path for entry in entries if entry.folder is not None]
    )

    def rules(
        index: int, kind: contract_skills.SkillEntryKind, _: Path
    ) -> contract_skills.DiscoveryRules:
        # Called in entry order once the entry exists and can be listed.
        if kind == "skill":
            return _rules(EXTRA_ROOT_DEPTH, hidden)
        folder = folders[index]
        if os.path.lexists(folder / SKILL_FILE):
            raise lifecycle.LifecycleError(
                "openclaw_skill_invalid",
                f"NeMo Fabric skill entry skills.directories[{index}].folder holds "
                f"{SKILL_FILE}, so OpenClaw loads it as one skill; use skill instead: "
                f"{folder}",
                metadata={
                    "field": f"skills.directories[{index}].folder",
                    "reason": "folder_is_skill",
                    "path": str(folder),
                },
            )
        if index in escaped:
            # Only skills/ holds skills here, so nothing else is found.
            return _rules(depths[index], hidden | {"skills"})
        return _rules(depths[index], hidden)

    try:
        load = contract_skills.load_directories(
            entries, rules=rules, duplicates=duplicates, base_dir=base_dir
        )
    except contract_skills.SkillLoadError as error:
        raise skill_error(error) from None
    if escaped:
        for index, nested in escaped.items():
            load.skipped.append(
                contract_skills.SkippedCandidate(
                    _resolved(nested), f"skills.directories[{index}]", "path_escape"
                )
            )
        order = {field: index for index, (field, _, _) in enumerate(load.entries)}
        load.skipped.sort(key=lambda item: order[item.field])
    return load


def _with_native_extra_outcomes(
    load: contract_skills.DirectoryLoad,
) -> contract_skills.DirectoryLoad:
    """Resolve clashes between ``extraDirs`` roots the way OpenClaw does.

    Every copy is passed to OpenClaw, which keeps the copy from the later root.
    Copies inside one root resolve in OpenClaw's path order, which is reported
    as ambiguous.
    """

    order = {field: index for index, (field, _, _) in enumerate(load.entries)}
    records = []
    losers: set[Path] = set()
    for record in load.duplicates:
        last = max(order.get(field, -1) for _, _, field in record.copies)
        in_last = [copy for copy in record.copies if order.get(copy[2], -1) == last]
        if record.outcome != "ambiguous" or len(in_last) != 1:
            records.append(record)
            continue
        winner = in_last[0][1]
        losers.update(path for _, path, _ in record.copies if path != winner)
        records.append(
            contract_skills.DuplicateRecord(
                record.name, record.copies, "selected", winner
            )
        )
    return contract_skills.DirectoryLoad(
        selected=[skill for skill in load.selected if skill.path not in losers],
        entries=load.entries,
        empty_entries=load.empty_entries,
        skipped=load.skipped,
        duplicates=records,
    )


def _with_native_tiers(
    load: contract_skills.DirectoryLoad,
    *,
    workspace: Path,
    inventory: BundledInventory | None,
    allow_bundled: list[str] | None,
    custodian: bool,
) -> contract_skills.DirectoryLoad:
    """Record configured skills that a higher OpenClaw source hides.

    The workspace's ``skills/`` and ``.agents/skills/`` folders and the shipped
    skills, with the Custodian skills when the agent owns them, are checked. The state folder and Skill Workshop folder are new
    and empty at start, so they hide nothing yet.
    """

    native: dict[str, tuple[str, Path, bool]] = {}
    if inventory is not None:
        allowed = set(allow_bundled) if allow_bundled is not None else None
        # Custodian skills load only for the agent that owns them.
        shipped = (
            {**inventory.custodian, **inventory.skills}
            if custodian
            else inventory.skills
        )
        for name in sorted(shipped):
            # A filtered-out shipped skill still hides the lower copy.
            native[name] = (
                "bundled",
                shipped[name],
                allowed is None or name in allowed,
            )
    for relative in _LOCAL_SKILL_FOLDERS:
        folder = workspace / relative
        if not _is_dir(folder):
            continue
        try:
            local = contract_skills.load_directories(
                [{"folder": folder}],
                rules=_rules(SKILLS_ROOT_DEPTH, _hidden_dir_names([folder])),
                duplicates=None,
            )
        except contract_skills.SkillLoadError:
            continue
        for skill in local.selected:
            native[skill.name] = ("local", skill.path, True)
    hidden = {skill.name for skill in load.selected if skill.name in native}
    if not hidden:
        return load
    records = [record for record in load.duplicates if record.name not in hidden]
    for name in sorted(hidden):
        origin, path, loads = native[name]
        existing = next(
            (record for record in load.duplicates if record.name == name), None
        )
        copies = (
            existing.copies
            if existing is not None
            else tuple(
                ("external", skill.path, skill.field)
                for skill in load.selected
                if skill.name == name
            )
        )
        records.append(
            contract_skills.DuplicateRecord(
                name,
                (*copies, (origin, path, "")),
                "shadowed",
                path if loads else None,
            )
        )
    return contract_skills.DirectoryLoad(
        selected=[skill for skill in load.selected if skill.name not in hidden],
        entries=load.entries,
        empty_entries=load.empty_entries,
        skipped=load.skipped,
        duplicates=records,
    )


def _configured(
    load: contract_skills.DirectoryLoad,
    base_dir: Path,
    directories: Sequence[contract.SkillDirectory],
) -> list[tuple[str, str, str]]:
    """Return ``(field, kind, path)`` per entry, with paths as configured."""

    return [
        (field, kind, os.path.normpath(base_dir / entry.path))
        for (field, kind, _), entry in zip(load.entries, directories, strict=True)
    ]


def _scanned_folder(folder: Path) -> Path:
    """Return the folder OpenClaw scans for an ``extraDirs`` root (``resolveNestedSkillsRoot``)."""

    nested = folder / "skills"
    if not _is_dir(nested):
        return folder
    if _contains_skill(folder, skip="skills", symlinks=True):
        return folder
    return nested if _contains_skill(nested, include_root=True) else folder


def _contains_skill(
    directory: Path,
    *,
    include_root: bool = False,
    skip: str | None = None,
    symlinks: bool = False,
) -> bool:
    queue = [(directory, 0)]
    for current, depth in queue:
        if (include_root or depth) and _has_skill_file(current):
            return True
        if depth >= SKILLS_ROOT_DEPTH:
            continue
        try:
            with os.scandir(current) as listing:
                children = sorted(listing, key=lambda child: child.name)
        except OSError:
            continue
        for child in children:
            if (
                child.name.startswith(".")
                or child.name == "node_modules"
                or (depth == 0 and child.name == skip)
            ):
                continue
            try:
                if symlinks and child.is_symlink():
                    return True
                if child.is_dir(follow_symlinks=False):
                    queue.append((Path(child.path), depth + 1))
            except OSError:
                continue
    return False


def _is_dir(path: Path) -> bool:
    # Path.is_dir() raises PermissionError below an untraversable folder on
    # Python 3.13 and earlier.
    try:
        return path.is_dir()
    except OSError:
        return False


def _resolved(path: Path) -> Path:
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError):
        return Path(os.path.normpath(path))


def _contained(path: Path, root: Path) -> bool:
    """Return whether ``path`` resolves inside ``root``; unresolvable paths count as inside."""

    try:
        resolved, resolved_root = path.resolve(strict=True), root.resolve(strict=True)
    except (OSError, RuntimeError):
        return True
    return resolved == resolved_root or resolved.is_relative_to(resolved_root)


def _has_skill_file(directory: Path) -> bool:
    # OpenClaw tests for any entry named SKILL.md (hasSkillFileCandidate).
    try:
        os.lstat(directory / SKILL_FILE)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError:
        return True
    return True


def _hidden_dir_names(folders: Sequence[Path]) -> frozenset[str]:
    """Return the names of folders starting with ``.`` that discovery could reach.

    OpenClaw skips every such folder; the shared discovery helper skips folders
    by exact name.
    """

    names: set[str] = set()
    for folder in folders:
        stack = [(folder, 0)]
        while stack:
            current, depth = stack.pop()
            try:
                with os.scandir(current) as listing:
                    children = list(listing)
            except OSError:
                continue
            for child in children:
                try:
                    if not child.is_dir():
                        continue
                except OSError:
                    continue
                if child.name.startswith("."):
                    names.add(child.name)
                elif child.name != "node_modules" and depth + 1 < SKILLS_ROOT_DEPTH:
                    stack.append((Path(child.path), depth + 1))
    return frozenset(names)


def _unique(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
