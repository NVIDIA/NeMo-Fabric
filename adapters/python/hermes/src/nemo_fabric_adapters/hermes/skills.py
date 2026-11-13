# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map ``skills.directories``, ``skills.bundled``, and ``skills.duplicates`` to Hermes.

Hermes reads external skills from ``skills.external_dirs`` and hides skills by
name through ``skills.disabled``. Its loader ranks skill roots in tiers:

1. trusted project folders (never configured by this adapter);
2. the Hermes home's ``skills`` folder, which holds the bundled skills Hermes
   copies in from its installation and the skills the agent creates;
3. ``skills.create_dir`` (never configured by this adapter);
4. ``skills.external_dirs``.

A higher tier wins a name clash. Two different skills with the same name in
one tier are refused as ambiguous.

The planning functions here need no Hermes installation. The functions that
read Hermes's bundled inventory, copy the bundled skills into the home, and
observe Hermes's own skill catalog import Hermes lazily.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from collections.abc import Callable
from collections.abc import Mapping
from contextlib import redirect_stdout
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from io import StringIO
from pathlib import Path
from typing import Any

from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import BundledSkills
from nemo_fabric_adapter_contract.skills import SKILL_FILE
from nemo_fabric_adapter_contract.skills import DirectoryLoad
from nemo_fabric_adapter_contract.skills import DiscoveryRules
from nemo_fabric_adapter_contract.skills import DuplicateRecord
from nemo_fabric_adapter_contract.skills import SkillLoadError
from nemo_fabric_adapter_contract.skills import SkippedCandidate
from nemo_fabric_adapter_contract.skills import load_directories
from nemo_fabric_adapter_contract.skills import report_block
from nemo_fabric_adapter_contract.skills import warn
from nemo_fabric_adapters.common import lifecycle

ADAPTER = "Hermes"

# agent/skill_utils.py EXCLUDED_SKILL_DIRS: folders Hermes's skill scan never
# enters. Hermes also drops every skill whose path contains one of them.
#
# Supported Hermes releases differ in this list, so read it from the installed
# Hermes when it can be imported, and fall back to the newest known list.
_FALLBACK_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".github",
        ".hub",
        ".archive",
        ".curator_backups",
        ".locks",
        ".hermes-dist",
        ".venv",
        "venv",
        "node_modules",
        "site-packages",
        "__pycache__",
        ".tox",
        ".nox",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
)


def _installed_excluded_dirs() -> frozenset[str]:
    try:
        from agent.skill_utils import EXCLUDED_SKILL_DIRS
    except Exception:  # Hermes isn't importable here, or has no such constant.
        return _FALLBACK_EXCLUDED_DIRS
    return frozenset(EXCLUDED_SKILL_DIRS)


HERMES_EXCLUDED_DIRS = _installed_excluded_dirs()

# agent/skill_utils.py SKILL_SUPPORT_DIRS: a skill's support folders. Hermes's
# skill scan never enters one that sits directly in a folder holding SKILL.md.
HERMES_SUPPORT_DIRS = frozenset({"references", "templates", "assets", "scripts"})

# How Hermes's skill scan (agent/skill_utils.py iter_skill_index_files) walks a
# skill root: from the root itself, every level, below skill folders too
# except their support folders, with no depth limit. A missing front-matter
# name falls back to the folder name and no front-matter field is required
# (tools/skills_tool.py _skill_catalog).
HERMES_DISCOVERY = DiscoveryRules(
    traversal="recursive",
    nested_skills="include",
    max_depth=None,
    required_frontmatter_fields=(),
    name_fallback=True,
    excluded_dir_names=HERMES_EXCLUDED_DIRS,
    folder_root_is_skill=True,
    support_dir_names=HERMES_SUPPORT_DIRS,
)

# tools/skills_sync.py: the file that records the skills Hermes copied into its
# home from its installation, one ``name:hash`` (or plain ``name``) per line.
BUNDLED_MANIFEST = ".bundled_manifest"

# agent/skill_utils.py ESSENTIAL_SKILLS: Hermes removes these names from
# skills.disabled and seeds them even into homes that opt out of bundled skills.
ESSENTIAL_SKILLS = frozenset({"hermes-agent"})

# agent/skill_utils.py tier numbers.
TIER_LOCAL = 1
TIER_EXTERNAL = 3


@dataclass
class SkillPlan:
    """Hermes-native skill settings planned at start."""

    load: DirectoryLoad | None = None
    """Checked ``skills.directories``, or ``None`` when it is not set."""
    external_dirs: list[str] | None = None
    """``skills.external_dirs`` for ``directories``; ``None`` keeps ``paths``."""
    disabled: list[str] = dataclass_field(default_factory=list)
    """Bundled skill names to add to ``skills.disabled``."""
    bundled: BundledSkills | None = None
    inventory: dict[str, Path] = dataclass_field(default_factory=dict)
    """Bundled skill name to its folder in the Hermes installation. Read at
    planning for ``bundled``, and at start for reporting origins."""
    bundled_root: Path | None = None
    """The Hermes installation's bundled-skills folder that ``inventory`` is in."""
    selected_bundled: list[str] = dataclass_field(default_factory=list)
    """Bundled skills the configuration selects, in name order."""
    sync: bool = False
    """Whether the adapter copies the bundled skills into the Hermes home."""
    root_fields: dict[Path, str] = dataclass_field(default_factory=dict)
    """Resolved ``external_dirs`` root to the entry that supplied it."""
    external_names: set[str] = dataclass_field(default_factory=set)
    """Names Hermes indexes for the external skills, as its bundled sync does."""
    local_names: set[str] = dataclass_field(default_factory=set)
    """Names of the agent's own skills in the Hermes home, which Hermes's
    bundled sync did not copy there."""
    kept_local: list[str] = dataclass_field(default_factory=list)
    """Unselected bundled names left out of ``skills.disabled`` because the
    home holds the agent's own skill by that name."""

    @property
    def structured(self) -> bool:
        """Whether ``directories`` or ``bundled`` is configured."""

        return self.load is not None or self.bundled is not None


@dataclass
class SkillObservation:
    """What Hermes loads, observed at start."""

    load: DirectoryLoad
    bundled: dict[str, Any] | None = None
    bundled_warnings: list[str] = dataclass_field(default_factory=list)

    def report(self) -> dict[str, Any] | None:
        """Return the ``output.skills`` block."""

        return report_block(self.load, bundled=self.bundled)

    def warn(self) -> None:
        """Write the start warnings to stderr."""

        warn(ADAPTER, self.load)
        for line in self.bundled_warnings:
            print(f"{ADAPTER}: {line}", file=sys.stderr, flush=True)


def plan_skills(
    agent_config: AgentConfig,
    *,
    base_dir: str | os.PathLike[str] | None = None,
    inventory: Callable[[], dict[str, Path]] | None = None,
    hermes_home: str | os.PathLike[str] | None = None,
) -> SkillPlan:
    """Check the structured skill configuration and plan Hermes's settings.

    ``hermes_home`` is the Hermes home the runtime uses; it defaults to
    ``HERMES_HOME``. Relative ``skills.paths`` resolve against it, as Hermes
    resolves them, and the agent's own skills in it stay enabled.

    Raises:
        lifecycle.LifecycleError: ``hermes_skill_not_found`` or
            ``hermes_skill_invalid`` for an entry that cannot load, a duplicate
            under an explicit policy, or an unknown bundled skill name.
    """

    skills = agent_config.skills
    plan = SkillPlan()
    if skills is None:
        return plan
    if skills.directories is not None:
        try:
            plan.load = load_directories(
                skills.directories,
                rules=HERMES_DISCOVERY,
                duplicates=skills.duplicates,
                base_dir=base_dir,
            )
            plan.external_dirs, plan.root_fields = external_dirs(
                plan.load, skills.duplicates
            )
        except SkillLoadError as error:
            raise skill_error(error) from None
    if skills.bundled is not None:
        home = _home(hermes_home)
        plan.bundled = skills.bundled
        if inventory is None:
            load_inventory(plan)
        else:
            plan.inventory = inventory()
            plan.bundled_root = _common_root(plan.inventory)
        roots = (
            plan.external_dirs
            if plan.external_dirs is not None
            else [os.fspath(path) for path in skills.paths]
        )
        plan.external_names = external_skill_names(roots, home)
        _plan_bundled(plan, skills.bundled, home)
    return plan


def skill_error(error: SkillLoadError) -> lifecycle.LifecycleError:
    """Map a ``skills.directories`` failure to the adapter's error codes."""

    metadata: dict[str, Any] = {"field": error.field, "reason": error.reason}
    if error.paths:
        metadata["path"] = str(error.paths[0])
    if len(error.paths) > 1:
        metadata["paths"] = [str(path) for path in error.paths]
    code = (
        "hermes_skill_not_found"
        if error.reason == "not_found"
        else "hermes_skill_invalid"
    )
    return lifecycle.LifecycleError(code, str(error), metadata=metadata)


def external_dirs(
    load: DirectoryLoad, policy: str | None
) -> tuple[list[str], dict[Path, str]]:
    """Return ``skills.external_dirs`` and the entry behind each root.

    Hermes refuses one skill reached through two roots as ambiguous, and its
    scan reaches every skill below a root except under the folders it skips.
    So a root that Hermes's scan of another passed root reaches is not passed.

    Without a policy each entry's folder is passed, resolved as Hermes
    resolves it, so Hermes sees every copy. With a policy each selected
    skill's folder is passed, so a shadowed copy never loads. A shadowed copy
    that Hermes's scan of a selected skill's folder reaches fails with
    ``duplicate``.
    """

    if policy is None:
        roots = [(path, field) for field, _kind, path in load.entries]
    else:
        roots = [(skill.path, skill.field) for skill in load.selected]
    folders: list[tuple[Path, str]] = []
    for path, field in roots:
        if any(_reaches(other, path) for other, _ in folders):
            continue
        folders = [item for item in folders if not _reaches(path, item[0])]
        folders.append((path, field))
    if policy is not None:
        for record in load.duplicates:
            for _origin, path, field in record.copies:
                if path == record.selected:
                    continue
                for folder, _ in folders:
                    if _reaches(folder, path):
                        raise SkillLoadError(
                            "duplicate",
                            f"NeMo Fabric skill {record.name!r} at {path} ({field}) "
                            f"would still load through {folder}, because Hermes "
                            "loads skills nested inside a skill folder",
                            field=field,
                            paths=(path, folder),
                        )
    # Keep the configured order of the roots that remain.
    kept = {path for path, _ in folders}
    values: list[str] = []
    fields: dict[Path, str] = {}
    for path, field in roots:
        if path in kept and path not in fields:
            fields[path] = field
            values.append(str(path))
    return values, fields


def _reaches(root: Path, path: Path) -> bool:
    """Whether Hermes's scan of ``root`` reaches the folder ``path``.

    The scan enters every folder below ``root`` except the folders it
    excludes and a skill's support folders.
    """

    if path == root:
        return True
    if root not in path.parents:
        return False
    folder = root
    for part in path.relative_to(root).parts:
        if part in HERMES_EXCLUDED_DIRS:
            return False
        if part in HERMES_SUPPORT_DIRS and _is_file(folder / SKILL_FILE):
            return False
        folder = folder / part
    return True


def _home(hermes_home: str | os.PathLike[str] | None) -> Path | None:
    if hermes_home is not None:
        return Path(hermes_home)
    value = os.environ.get("HERMES_HOME", "").strip()
    return Path(os.path.expanduser(os.path.expandvars(value))) if value else None


def _plan_bundled(plan: SkillPlan, bundled: BundledSkills, home: Path | None) -> None:
    inventory = plan.inventory
    # Hermes loads its essential skills whatever skills.disabled says, so a
    # request that leaves one out cannot be honored and is refused.
    essential = sorted(ESSENTIAL_SKILLS.intersection(inventory))
    if not bundled.enabled and essential:
        raise _essential_error("skills.bundled.enabled", essential[0])
    for field_name in ("include", "exclude"):
        names = getattr(bundled, field_name)
        for name in names or ():
            if name not in inventory:
                raise lifecycle.LifecycleError(
                    "hermes_skill_invalid",
                    f"Hermes has no bundled skill named {name!r}",
                    metadata={
                        "field": f"skills.bundled.{field_name}",
                        "reason": "unknown_bundled_skill",
                        "name": name,
                    },
                )
            if field_name == "exclude" and name in ESSENTIAL_SKILLS:
                raise _essential_error("skills.bundled.exclude", name)
    if bundled.include is not None:
        missing = [name for name in essential if name not in bundled.include]
        if missing:
            raise _essential_error("skills.bundled.include", missing[0])
        selected = set(bundled.include)
    else:
        selected = set(inventory) - set(bundled.exclude or ())
    plan.selected_bundled = sorted(selected)
    # skills.disabled hides every copy of a name, not only the bundled one, so
    # names that an external skill or the agent's own skill in the home
    # provides stay out of the list. Hermes's sync never copies a bundled
    # skill whose name an external skill provides, nor over the agent's own
    # skill at the bundled copy's place in the home. An own skill elsewhere in
    # the home gets the bundled copy beside it, and Hermes refuses both as
    # ambiguous; the report's warning says so.
    unselected = set(inventory) - selected - ESSENTIAL_SKILLS
    plan.local_names = local_skill_names(home)
    own = plan.local_names & unselected
    plan.kept_local = sorted(own - plan.external_names)
    plan.disabled = sorted(unselected - plan.external_names - own)
    # Hermes's sync seeds its essential skills even into a home that opts out
    # of bundled skills, and removes a stale copy an external skill replaces.
    plan.sync = True


def _essential_error(field: str, name: str) -> lifecycle.LifecycleError:
    """Refuse a bundled selection that leaves out a skill Hermes always loads."""

    if field == "skills.bundled.include":
        detail = f"{field} must name it"
    elif field == "skills.bundled.exclude":
        detail = f"{field} cannot name it"
    else:
        detail = f"{field} cannot be false; use include to name the skills to keep"
    return lifecycle.LifecycleError(
        "hermes_skill_invalid",
        f"Hermes always loads its bundled skill {name!r}, so {detail}",
        metadata={"field": field, "reason": "essential_bundled_skill", "name": name},
    )


def external_skill_names(roots: list[str], home: Path | None) -> set[str]:
    """Return the names Hermes indexes for the skills below ``roots``.

    Mirrors tools/skills_sync.py ``_build_external_skill_index``: each
    ``skills.external_dirs`` value is expanded and resolved as Hermes does,
    and every ``SKILL.md`` Hermes scans contributes its folder name and its
    front-matter name.
    """

    names: set[str] = set()
    local = (home / "skills").resolve() if home is not None else None
    seen: set[Path] = set()
    for value in roots:
        path = Path(os.path.expanduser(os.path.expandvars(str(value))))
        if not path.is_absolute():
            if home is None:
                continue
            path = home / path
        try:
            path = path.resolve()
            if path == local or path in seen or not path.is_dir():
                continue
        except OSError:
            continue
        seen.add(path)
        for skill_md in _skill_files(path):
            names.update({skill_md.parent.name, read_skill_name(skill_md, "")})
    names.discard("")
    return names


def local_skill_names(home: Path | None) -> set[str]:
    """Return the names of the agent's own skills in the Hermes home.

    Hermes records each skill its sync copied into the home in the bundled
    manifest. A skill whose name the manifest does not hold is the agent's
    (or user's) own, and Hermes's sync never replaces it with the bundled
    copy.
    """

    if home is None:
        return set()
    skills_dir = home / "skills"
    names = {
        read_skill_name(skill_md, skill_md.parent.name)
        for skill_md in _skill_files(skills_dir)
    }
    return names - manifest_names(skills_dir)


def manifest_names(skills_dir: Path) -> set[str]:
    """Return the bundled skills Hermes's sync recorded copying into ``skills_dir``.

    Mirrors tools/skills_sync.py ``_read_manifest``.
    """

    try:
        lines = (skills_dir / BUNDLED_MANIFEST).read_text(encoding="utf-8-sig")
    except OSError:
        return set()
    return {
        name for line in lines.splitlines() if (name := line.partition(":")[0].strip())
    }


def _skill_files(root: Path) -> list[Path]:
    """Every ``SKILL.md`` below ``root`` that Hermes's sync index reads."""

    try:
        if not root.is_dir():
            return []
        found = sorted(root.rglob(SKILL_FILE))
    except OSError:
        return []
    return [skill_md for skill_md in found if not _excluded_skill_path(skill_md)]


def _excluded_skill_path(path: Path) -> bool:
    # agent/skill_utils.py is_excluded_skill_path for an absolute path.
    parts = path.parts
    if any(part in HERMES_EXCLUDED_DIRS for part in parts):
        return True
    return any(
        part in HERMES_SUPPORT_DIRS and _is_file(Path(*parts[:index]) / SKILL_FILE)
        for index, part in enumerate(parts[:-1])
        if index > 0
    )


def read_skill_name(skill_md: Path, fallback: str) -> str:
    """Return the ``name:`` line of a ``SKILL.md``, as Hermes's sync reads it.

    Mirrors tools/skill_usage.py ``read_skill_name``.
    """

    try:
        text = skill_md.read_text(encoding="utf-8-sig", errors="replace")[:4000]
    except OSError:
        return fallback
    lines = [line.strip() for line in text.split("\n")]
    if "---" not in lines:
        return fallback
    block = lines[lines.index("---") + 1 :]
    block = block[: block.index("---")] if "---" in block else block
    values = (
        line.split(":", 1)[1].strip().strip("\"'")
        for line in block
        if line.startswith("name:")
    )
    return next((value for value in values if value), fallback)


def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def bundled_inventory() -> dict[str, Path]:
    """Return the bundled skills Hermes copies into its home, by name."""

    with redirect_stdout(StringIO()):
        from tools import skills_sync

    bundled_dir = skills_sync._get_bundled_dir()
    return {
        name: path for name, path in skills_sync._discover_bundled_skills(bundled_dir)
    }


def bundled_dir() -> Path:
    """Return the Hermes installation's bundled-skills folder."""

    with redirect_stdout(StringIO()):
        from tools import skills_sync

    return Path(skills_sync._get_bundled_dir())


def load_inventory(plan: SkillPlan) -> None:
    """Read Hermes's bundled inventory into ``plan``, once."""

    if plan.bundled_root is None:
        plan.inventory = bundled_inventory()
        plan.bundled_root = bundled_dir()


def _common_root(inventory: Mapping[str, Path]) -> Path | None:
    """The folder that holds every skill of an inventory read some other way."""

    parents = [os.fspath(Path(path).parent) for path in inventory.values()]
    return Path(os.path.commonpath(parents)) if parents else None


def sync_bundled() -> dict[str, Any]:
    """Copy the bundled skills into the Hermes home, as Hermes's own entry points do."""

    with redirect_stdout(StringIO()):
        from tools import skills_sync

        return skills_sync.sync_skills(quiet=True)


def native_catalog() -> list[dict[str, Any]]:
    """Return every skill copy Hermes resolves, with its tier and status.

    Status is ``unique`` (it loads), ``ambiguous`` (a same-tier clash Hermes
    refuses), or ``shadowed`` (a higher tier owns the name). ``visible`` is
    false for a skill Hermes hides on this platform or environment.
    """

    with redirect_stdout(StringIO()):
        from agent import skill_utils
        from tools import skills_tool

        skill_utils._external_dirs_cache_clear()
        skills_tool.clear_skills_cache()
        rows = skills_tool._skill_catalog(skip_disabled=True, include_hidden=True)
        visible = {
            str(row["path"]) for row in skills_tool._skill_catalog(skip_disabled=True)
        }
        skills_tool.clear_skills_cache()
    return [{**row, "visible": str(row["path"]) in visible} for row in rows]


def observe(
    plan: SkillPlan,
    catalog: list[dict[str, Any]],
    *,
    sync_result: Mapping[str, Any] | None = None,
) -> SkillObservation:
    """Combine the plan with Hermes's resolved catalog.

    Hermes's catalog decides what loads. External skills that Hermes does not
    resolve as ``unique`` leave the loaded list, and loaded skills carry the
    name Hermes reads, which can differ from the checked name for unusual
    front matter. A skill that Hermes sees but does not load for another
    reason, such as a clash with another skill's folder name, is skipped with
    Hermes's status as the reason.

    Every name that more than one skill folder in Hermes's catalog provides is
    reported as a duplicate, with each copy's origin (``external``,
    ``bundled``, or ``local``) and path, and Hermes's outcome. When an
    external copy takes part, the outcome is ``selected`` when it loads,
    ``shadowed`` when a copy in the Hermes home wins, and ``ambiguous`` when
    Hermes refuses the name. Otherwise it is ``selected`` when Hermes loads
    one copy and ``ambiguous`` when it refuses them all. Copies are matched
    by the name Hermes reads, never by the checks' reading of the front
    matter.
    """

    load = plan.load or DirectoryLoad()
    places = _bundled_places(plan, catalog)
    rows = [_row(row, places) for row in catalog]
    external_rows = [row for row in rows if row["tier"] == TIER_EXTERNAL]
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["name"]:
            by_name.setdefault(row["name"], []).append(row)

    fields = _skill_fields(load)
    records = {record.name: record for record in load.duplicates}
    duplicates: list[DuplicateRecord] = []
    for name, copies in by_name.items():
        external = [row for row in copies if row["tier"] == TIER_EXTERNAL]
        # A policy's record keeps the copies it dropped, which Hermes never
        # sees. Without a policy every copy reaches Hermes, so its catalog
        # alone says which copies clash.
        record = records.get(name)
        if record is not None and record.outcome == "ambiguous":
            record = None
        if len({row["dir"] for row in copies}) < 2 and record is None:
            continue
        records.pop(name, None)
        reported = list(record.copies) if record is not None else []
        seen = {(path, field) for _origin, path, field in reported}
        for row in sorted(copies, key=lambda row: row["tier"]):
            origin = row["origin"]
            field = _field(row, fields, plan) if origin == "external" else ""
            if (row["dir"], field) in seen:
                continue
            seen.add((row["dir"], field))
            reported.append((origin, row["dir"], field))
        loads = [row for row in external if row["status"] == "unique"]
        owner = [row for row in copies if row["status"] == "unique"]
        if loads:
            # A configured copy wins; it loads unless Hermes hides it.
            outcome, selected = "selected", _visible_dir(loads)
        elif any(row["status"] == "ambiguous" for row in external) or not owner:
            outcome, selected = "ambiguous", None
        elif external:
            outcome, selected = "shadowed", _visible_dir(owner)
        else:
            # No configured copy: Hermes's own order picks one of the home's.
            outcome, selected = "selected", _visible_dir(owner)
        duplicates.append(DuplicateRecord(name, tuple(reported), outcome, selected))

    # Hermes's status, name, and visibility per external skill folder.
    statuses: dict[Path, tuple[str, str, bool]] = {}
    for row in external_rows:
        if statuses.get(row["dir"], ("", "", True))[0] != "unique":
            statuses[row["dir"]] = (row["status"], row["name"], row["visible"])

    # A clash the checks predicted without a policy that Hermes's catalog does
    # not show did not happen; each copy is reported as loaded under Hermes's
    # name or skipped. A policy's choice still stands, but its selected copy
    # is named only when Hermes loads it and shows it, whatever name Hermes
    # reads for it.
    for record in records.values():
        if record.outcome == "ambiguous":
            continue
        loaded = (
            record.selected is not None and statuses.get(record.selected) is not None
        )
        if loaded:
            status, _name, visible = statuses[record.selected]
            loaded = status == "unique" and visible
        duplicates.append(
            dataclasses.replace(record, selected=record.selected if loaded else None)
        )

    # Bundled copies Hermes's sync left out of the home because an external
    # skill provides the name. Hermes's catalog never shows them.
    deferred = (sync_result or {}).get("shadowed_by_external") or ()
    for name in deferred:
        loads = [row for row in external_rows if row["name"] == name]
        if (
            loads
            and name in plan.inventory
            and name not in {item.name for item in duplicates}
        ):
            copies = tuple(
                ("external", row["dir"], _field(row, fields, plan)) for row in loads
            ) + (("bundled", plan.inventory[name], ""),)
            unique = [row for row in loads if row["status"] == "unique"]
            duplicates.append(
                DuplicateRecord(
                    name,
                    copies,
                    "selected" if unique else "ambiguous",
                    _visible_dir(unique),
                )
            )

    reported_dirs = {path for item in duplicates for _origin, path, _ in item.copies}
    selected = []
    skipped = list(load.skipped)
    for skill in load.selected:
        status, name, visible = statuses.get(skill.path, (None, skill.name, True))
        if status == "unique" and visible:
            selected.append(dataclasses.replace(skill, name=name))
            continue
        if status is None:
            excluded = HERMES_EXCLUDED_DIRS.intersection(skill.path.parts)
            reason = "excluded_path" if excluded else "not_loaded_by_hermes"
        elif status == "unique":
            # Hermes resolves it but hides it on this platform or environment.
            reason = "native_conditions"
        elif skill.path in reported_dirs:
            # The duplicate record explains it.
            continue
        else:
            reason = str(status)
        skipped.append(SkippedCandidate(skill.path, skill.field, reason))
    observed = dataclasses.replace(
        load, selected=selected, skipped=skipped, duplicates=duplicates
    )

    bundled_report = None
    warnings: list[str] = []
    if plan.bundled is not None:
        bundled_report, warnings = _bundled_report(plan, rows, sync_result)
    return SkillObservation(observed, bundled_report, warnings)


def _visible_dir(rows: list[dict[str, Any]]) -> Path | None:
    """The folder of the first copy that loads, or ``None`` when Hermes hides
    every copy."""

    return next((row["dir"] for row in rows if row["visible"]), None)


def _row(row: Mapping[str, Any], places: Mapping[Path, str]) -> dict[str, Any]:
    skill_md = Path(row["path"])
    directory = _resolved(skill_md.parent)
    root = _resolved(Path(row["root"]))
    status = row.get("status")
    if status == "unique" and "load_name" in row and not row["load_name"]:
        # Hermes lists a skill with an empty name but cannot load it.
        status = "no_name"
    tier = int(row["tier"])
    bundled = places.get(directory) if tier == TIER_LOCAL else None
    if tier == TIER_EXTERNAL:
        origin = "external"
    else:
        origin = "bundled" if bundled is not None else "local"
    return {
        "name": str(row["name"]),
        "tier": tier,
        "root": root,
        "dir": directory,
        "status": status,
        "visible": row.get("visible", True),
        "origin": origin,
        "bundled": bundled,
    }


def _resolved(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _bundled_places(plan: SkillPlan, catalog: list[dict[str, Any]]) -> dict[Path, str]:
    """Folders in the Hermes home that hold Hermes's copy of a bundled skill.

    Hermes's sync copies a bundled skill to the same place below the home's
    ``skills`` folder as below its bundled folder, and records the name in the
    home's manifest. Any other skill in the home is the agent's (or user's)
    own, whatever its front matter says, so origins never depend on how a
    name is read.
    """

    if not plan.inventory or plan.bundled_root is None:
        return {}
    roots = {
        _resolved(Path(row["root"]))
        for row in catalog
        if int(row["tier"]) == TIER_LOCAL
    }
    places: dict[Path, str] = {}
    for root in roots:
        manifest = manifest_names(root)
        for name, source in plan.inventory.items():
            if name not in manifest:
                continue
            try:
                place = Path(source).relative_to(plan.bundled_root)
            except ValueError:
                continue
            places[_resolved(root / place)] = name
    return places


def _field(row: Mapping[str, Any], fields: Mapping[Path, str], plan: SkillPlan) -> str:
    """The entry that supplied an external copy: the checked skill's entry,
    else the entry whose folder Hermes scanned."""

    return fields.get(row["dir"]) or plan.root_fields.get(row["root"], "")


def _skill_fields(load: DirectoryLoad) -> dict[Path, str]:
    fields: dict[Path, str] = {}
    for record in load.duplicates:
        for _origin, path, field in record.copies:
            fields.setdefault(path, field)
    for skill in load.selected:
        fields.setdefault(skill.path, skill.field)
    return fields


def _bundled_report(
    plan: SkillPlan,
    rows: list[dict[str, Any]],
    sync_result: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    local = [row for row in rows if row["tier"] == TIER_LOCAL]
    external = {row["name"] for row in rows if row["tier"] == TIER_EXTERNAL}
    suppressed = set((sync_result or {}).get("suppressed") or ())
    opted_out = bool((sync_result or {}).get("skipped_opt_out"))
    loaded: list[str] = []
    unavailable: list[dict[str, str]] = []
    for name in plan.selected_bundled:
        # Hermes's copies of this bundled skill, matched by place, and every
        # skill in the home that Hermes reads under its name.
        own_copies = [row for row in local if row["bundled"] == name]
        copies = [row for row in local if row["name"] == name]
        if any(row["status"] == "unique" and row["visible"] for row in own_copies):
            loaded.append(name)
            continue
        candidates = own_copies + copies
        if name in external or name in plan.external_names:
            reason = "provided_by_external_skill"
        elif any(
            row["origin"] == "local" and row["status"] == "unique" for row in copies
        ):
            # The agent's own skill in the home holds the name instead.
            reason = "provided_by_local_skill"
        elif any(row["status"] == "unique" for row in candidates):
            reason = "native_conditions"
        elif any(row["status"] == "ambiguous" for row in candidates):
            reason = "ambiguous"
        elif name in suppressed:
            reason = "suppressed"
        elif opted_out:
            reason = "home_opted_out"
        else:
            reason = "not_in_home"
        unavailable.append({"name": name, "reason": reason})
    # Hermes always loads these, so every selection names them.
    essential = [name for name in loaded if name in ESSENTIAL_SKILLS]
    report: dict[str, Any] = {"selected": loaded}
    if unavailable:
        report["unavailable"] = unavailable
    if essential:
        report["essential"] = essential
    warnings = [
        f"bundled skill {item['name']!r} does not load: {item['reason']}"
        for item in unavailable
        if item["reason"] != "provided_by_external_skill"
    ]
    for name in plan.kept_local:
        statuses = {row["status"] for row in local if row["name"] == name}
        if "unique" in statuses:
            warnings.append(
                f"bundled skill {name!r} is not selected, but the agent's own "
                f"skill {name!r} in the Hermes home still loads"
            )
        elif "ambiguous" in statuses:
            warnings.append(
                f"bundled skill {name!r} is not selected, but Hermes copied it "
                f"into its home beside the agent's own skill {name!r}; Hermes "
                "refuses both as ambiguous"
            )
    return report, warnings
