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
- A skill needs front matter that OpenClaw's parser accepts and a
  ``description`` that is not empty after trimming, in a ``SKILL.md`` of at
  most 256,000 bytes (``loadSingleSkillDirectory`` in
  ``dist/skill-paths-*.mjs``); its name is the trimmed front-matter ``name``
  or the trimmed folder name. :class:`OpenClawReader` runs OpenClaw's own
  ``parseSkillFrontmatter`` through ``node`` to tell, so names and refusals
  match OpenClaw's. Refused skills are skipped before any duplicate check,
  with ``invalid_frontmatter``, ``missing_frontmatter_field:description``,
  ``empty_frontmatter_field:description``, or ``file_too_large``, so they
  never count as a copy. When the parser cannot run, Fabric reads simple
  front matter itself and cannot tell which skills OpenClaw refuses; under a
  duplicate policy whose outcome depends on one, start fails with
  ``unsupported_frontmatter``. ``openclaw skills list`` at start reports any
  other skill OpenClaw refuses, such as a hard-linked ``SKILL.md``; under
  ``first_wins`` or ``last_wins`` the next copy replaces a refused winner.

OpenClaw's own ``skills.allowBundled`` filters bundled and Custodian skills
after names are resolved, so a filtered-out shipped skill would still hide a
configured or workspace copy. ``skills.bundled`` therefore never uses it.
When the selection leaves out a shipped skill, the adapter copies the
selected shipped skills into the runtime's own folder and points
``OPENCLAW_BUNDLED_SKILLS_DIR`` at it (:func:`stage_bundled`). OpenClaw reads
the Custodian skills from ``custodian-skills`` beside that folder
(``resolveWorkspaceSkillSourcePlan``), so the copy holds the selected ones
there too. A left-out skill is then never loaded and hides nothing. The
skills are copied rather than linked because OpenClaw refuses a shipped
skill whose real path is outside that folder (``bundled-symlink-escape``)
and hard-linked shipped files (``shouldRejectHardlinkedPluginFiles``).
Custodian skills load only for the agent that owns them
(``resolveCustodianSkillAgentId``).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from dataclasses import replace
from pathlib import Path
from typing import Any

from nemo_fabric_adapter_contract import models as contract
from nemo_fabric_adapter_contract import skills as contract_skills
from nemo_fabric_adapter_contract.skills import CandidateInfo
from nemo_fabric_adapters.common import lifecycle


SKILL_FILE = contract_skills.SKILL_FILE
# DEFAULT_MAX_SKILL_FILE_BYTES: OpenClaw refuses a larger SKILL.md.
MAX_SKILL_FILE_BYTES = 256_000
# The longest one read of every candidate's front matter may take.
READER_TIMEOUT_SECONDS = 15.0
# MAX_CONFIGURED_ROOT_GROUPED_SKILL_SCAN_DEPTH and MAX_GROUPED_SKILL_SCAN_DEPTH.
EXTRA_ROOT_DEPTH = 2
SKILLS_ROOT_DEPTH = 6
# The deepest level below an entry's folder that discovery reaches: the
# scanned folder's ``skills`` child, searched SKILLS_ROOT_DEPTH levels deep.
HIDDEN_SCAN_DEPTH = SKILLS_ROOT_DEPTH + 1
BUNDLED_SKILLS_ENV = "OPENCLAW_BUNDLED_SKILLS_DIR"
CUSTODIAN_SKILLS_DIR = "custodian-skills"
BUNDLED_SOURCES = frozenset({"openclaw-bundled", "openclaw-custodian"})
# normalizeAgentId: invalid characters become "-", outer dashes are trimmed.
_AGENT_ID_INVALID = re.compile(r"[^a-z0-9_-]+")
# Native sources that outrank skills.load.extraDirs, lowest first.
_LOCAL_SKILL_FOLDERS = (Path(".agents") / "skills", Path("skills"))
# The rank of each source ``openclaw skills list`` names, as
# resolveWorkspaceSkillSourcePlan orders them: a higher source wins a name.
# The Skill Workshop, state, and personal folders are empty at start, so no
# copy of the adapter's own has their ranks.
_SOURCE_RANKS: dict[str, float] = {
    "openclaw-extra": -1,
    "openclaw-bundled": 0,
    "openclaw-custodian": 1,
    "openclaw-workshop": 1.25,
    "openclaw-managed": 1.5,
    "agents-skills-personal": 1.75,
    "agents-skills-project": 2,
    "openclaw-workspace": 3,
}


@dataclass(frozen=True)
class NativeCopy:
    """A skill OpenClaw loads at start from one of its own sources."""

    origin: contract_skills.SkillOrigin
    """``bundled`` for a shipped skill, ``local`` for a workspace one."""
    path: Path
    rank: float
    """The source's precedence: a higher rank wins a name."""


@dataclass(frozen=True)
class BundledInventory:
    """Skills shipped with the installed OpenClaw package, by name."""

    skills: dict[str, Path]
    """``skills/`` of the package: loaded for every agent."""
    custodian: dict[str, Path]
    """``custodian-skills/``: loaded only for the configured Custodian agent."""
    root: Path | None = None
    """The resolved shipped skills folder; ``custodian-skills`` is beside it."""

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
    stage_bundled: bool = False
    """Whether ``bundled`` leaves out a shipped skill, so OpenClaw must read
    the selected ones from a staged folder (:func:`stage_bundled`)."""
    load: contract_skills.DirectoryLoad | None = None
    bundled_selected: list[str] | None = None
    bundled_unavailable: list[dict[str, Any]] = dataclass_field(default_factory=list)
    inventory: BundledInventory | None = None
    custodian: bool = True
    """Whether OpenClaw loads the Custodian skills for the configured agent."""
    duplicates: contract.SkillDuplicatePolicy | None = None
    rejected: set[Path] = dataclass_field(default_factory=set)
    """Configured skill folders that ``openclaw skills list`` showed it refuses."""
    configured: contract_skills.DirectoryLoad | None = None
    """``skills.directories`` as discovered, before OpenClaw's own sources
    are taken into account; ``load`` is derived from it."""
    native: dict[str, list[NativeCopy]] = dataclass_field(default_factory=dict)
    """Copies OpenClaw loads from its own sources, by name, lowest rank first."""
    native_refused: set[Path] | None = None
    """Native copies ``openclaw skills list`` showed it does not load, or
    ``None`` before it ran."""
    report_directories: bool = False
    """Whether ``skills.directories`` is set, so the report always says what
    it loads."""

    def merge_native(self) -> None:
        """Derive ``load`` from ``configured`` and the native copies."""

        if self.configured is None:
            return
        load = _with_native_sources(self.configured, self.native, self.native_refused)
        self.load = load if self.report_directories or load.duplicates else None

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
    reader: contract_skills.CandidateReader,
    custodian: bool = True,
) -> SkillPlan:
    """Check the structured skill settings and return what OpenClaw should load.

    ``reader`` reads each candidate as OpenClaw does (:func:`candidate_reader`);
    every candidate the plan meets is passed to it in one call.
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
    if skills.directories is None and skills.bundled is None:
        return result
    if skills.bundled is not None and bundled_dir is None:
        raise lifecycle.LifecycleError(
            "openclaw_skill_invalid",
            "OpenClaw's shipped skills folder was not found, so skills.bundled "
            "cannot be applied",
            metadata={"field": "skills.bundled", "reason": "inventory_unavailable"},
        )
    reader = _read_ahead(
        reader,
        skills.directories,
        base_dir=base_dir,
        bundled_dir=bundled_dir,
        workspace=workspace,
    )
    if bundled_dir is not None:
        result.inventory = bundled_inventory(bundled_dir, reader)
    if skills.bundled is not None:
        assert result.inventory is not None
        result.stage_bundled, result.bundled_selected = _select_bundled(
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
        load = contract_skills.DirectoryLoad()
    else:
        load = _load_directories(
            skills.directories,
            duplicates=skills.duplicates,
            base_dir=base_dir,
            reader=reader,
        )
        if skills.duplicates is None:
            load = _with_native_extra_outcomes(load)
            result.extra_dirs = _unique(
                path for _, _, path in _configured(load, base_dir, skills.directories)
            )
        else:
            result.extra_dirs = _unique(str(skill.path) for skill in load.selected)
    result.configured = load
    result.native = _native_copies(
        workspace=workspace,
        inventory=result.inventory,
        bundled=(
            set(result.bundled_selected)
            if result.bundled_selected is not None
            else None
        ),
        custodian=custodian,
        reader=reader,
    )
    result.report_directories = skills.directories is not None
    result.merge_native()
    return result


def stage_bundled(plan_: SkillPlan, root: Path) -> Path | None:
    """Copy the selected shipped skills into ``root`` for OpenClaw to load.

    Returns the folder to pass as ``OPENCLAW_BUNDLED_SKILLS_DIR``, or ``None``
    when ``bundled`` selects every shipped skill, so OpenClaw keeps its own
    folder. Selected Custodian skills go to ``custodian-skills`` beside it,
    where OpenClaw looks for them. Each skill keeps its path relative to the
    shipped folder.

    Raises:
        lifecycle.LifecycleError: If the copy cannot be written.
    """

    inventory = plan_.inventory
    if not plan_.stage_bundled or inventory is None:
        return None
    skills_dir = root / "skills"
    selected = set(plan_.bundled_selected or ())
    try:
        for target, shipped, source_root in (
            (skills_dir, inventory.skills, inventory.root),
            (
                root / CUSTODIAN_SKILLS_DIR,
                inventory.custodian,
                inventory.root.parent / CUSTODIAN_SKILLS_DIR
                if inventory.root is not None
                else None,
            ),
        ):
            target.mkdir(parents=True, exist_ok=True)
            for name in sorted(selected & set(shipped)):
                source = shipped[name]
                relative = (
                    source.relative_to(source_root)
                    if source_root is not None and source.is_relative_to(source_root)
                    else Path(source.name)
                )
                shutil.copytree(source, target / relative, symlinks=True)
    except OSError as error:
        raise lifecycle.LifecycleError(
            "openclaw_skill_invalid",
            f"NeMo Fabric could not copy the selected shipped OpenClaw skills: {error}",
            metadata={"field": "skills.bundled", "reason": "staging_failed"},
        ) from None
    return skills_dir


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
    package = package_root(command)
    if package is None:
        return None
    candidate = package / "skills"
    return candidate if _is_dir(candidate) else None


def package_root(command: Path) -> Path | None:
    """Return the installed ``openclaw`` package that holds ``command``."""

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
            return parent
    return None


def bundled_inventory(
    bundled_dir: Path, reader: contract_skills.CandidateReader
) -> BundledInventory:
    """Return the shipped skills OpenClaw loads, by name, read with ``reader``."""

    def names(folder: Path) -> dict[str, Path]:
        if not _is_dir(folder):
            return {}
        try:
            load = contract_skills.load_directories(
                [{"folder": folder}],
                rules=_rules(SKILLS_ROOT_DEPTH, _hidden_dir_names([folder])),
                duplicates=None,
                reader=reader,
            )
        except contract_skills.SkillLoadError:
            return {}
        return {skill.name: skill.path for skill in load.selected}

    return BundledInventory(
        skills=names(bundled_dir),
        custodian=names(bundled_dir.parent / CUSTODIAN_SKILLS_DIR),
        root=_resolved(bundled_dir),
    )


def candidate_reader(
    command: Path | None, *, env: Mapping[str, str], timeout: float
) -> OpenClawReader:
    """Return a reader backed by the OpenClaw package that holds ``command``."""

    return OpenClawReader(
        package_root(command) if command is not None else None,
        env=env,
        timeout=timeout,
    )


# Reads each SKILL.md named on stdin with OpenClaw's parseSkillFrontmatter and
# applies the checks loadSingleSkillDirectory makes after it.
_PARSE_SCRIPT = """
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
const [modulePath, exportName] = process.argv.slice(-2);
const parse = (await import(pathToFileURL(modulePath).href))[exportName];
if (typeof parse !== "function") throw new Error(`${exportName} is not a function`);
const rows = JSON.parse(readFileSync(0, "utf8")).map((file) => {
  let raw;
  try {
    raw = readFileSync(file, "utf8");
  } catch {
    return { reason: "unreadable" };
  }
  let frontmatter;
  try {
    frontmatter = parse(raw);
  } catch {
    return { reason: "invalid_frontmatter" };
  }
  const description = frontmatter.description;
  if (description === undefined) {
    return { reason: "missing_frontmatter_field:description" };
  }
  if (!String(description).trim()) {
    return { reason: "empty_frontmatter_field:description" };
  }
  return { name: frontmatter.name?.trim() || null };
});
process.stdout.write(JSON.stringify(rows));
"""
_PARSER_EXPORT = re.compile(r"export\s*\{[^}]*\bparseSkillFrontmatter as ([\w$]+)")


class _ParserUnavailable(Exception):
    """OpenClaw's front matter parser cannot be run."""


class OpenClawReader:
    """A :data:`~nemo_fabric_adapter_contract.skills.CandidateReader` that
    reads front matter with the installed OpenClaw's own parser.

    One ``node`` process per call imports ``parseSkillFrontmatter`` from the
    package's ``dist/frontmatter-*.mjs`` (file names are hashed, so the module
    is found by its export) and reads every candidate's ``SKILL.md``. A
    ``SKILL.md`` larger than OpenClaw reads is refused without parsing.

    When ``node`` or the module cannot be found, or the call fails or passes
    the deadline, the reader warns once and reads simple front matter with
    :func:`~nemo_fabric_adapter_contract.skills.front_matter_reader` instead.
    Those skills' validity is then unknown (``valid=None``), so an explicit
    duplicate policy never decides on a guess.

    Without a front matter name, OpenClaw uses the trimmed name of the folder
    it reached. The reader receives resolved folders, so it leaves the name
    to the helper's folder name fallback, which uses the spelling discovery
    reached, as OpenClaw does through a symbolic link. Only a folder name with
    outer whitespace is trimmed here, from the resolved name.
    """

    def __init__(
        self, package: Path | None, *, env: Mapping[str, str], timeout: float
    ) -> None:
        self._package = package
        self._env = dict(env)
        self._deadline = time.monotonic() + timeout
        self._unavailable = False

    def __call__(self, folders: Sequence[Path]) -> dict[Path, CandidateInfo]:
        infos: dict[Path, CandidateInfo] = {}
        parse: list[Path] = []
        for folder in folders:
            try:
                size = os.stat(folder / SKILL_FILE).st_size
            except OSError:
                infos[folder] = CandidateInfo(None, valid=False, reason="unreadable")
                continue
            if size > MAX_SKILL_FILE_BYTES:
                infos[folder] = CandidateInfo(
                    None, valid=False, reason="file_too_large"
                )
            else:
                parse.append(folder)
        if parse:
            infos.update(self._parse(parse))
        return {folder: infos[folder] for folder in folders}

    def _parse(self, folders: list[Path]) -> dict[Path, CandidateInfo]:
        if not self._unavailable:
            try:
                rows = self._run(folders)
            except _ParserUnavailable as error:
                self._unavailable = True
                print(
                    f"OpenClaw: NeMo Fabric cannot run OpenClaw's front matter "
                    f"parser ({error}), so it cannot tell which skills OpenClaw "
                    f"refuses",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                return {
                    folder: (
                        CandidateInfo(None, valid=False, reason=row["reason"])
                        if row.get("reason")
                        else _named(folder, row.get("name"), valid=True)
                    )
                    for folder, row in zip(folders, rows, strict=True)
                }
        simple = contract_skills.front_matter_reader(required_fields=("description",))
        return {
            folder: (
                info
                if info.reason == "unreadable"
                else _named(
                    folder,
                    (info.name or "").strip() or None,
                    valid=None,
                    name_uncertain=info.name_uncertain,
                )
            )
            for folder, info in simple(folders).items()
        }

    def _run(self, folders: list[Path]) -> list[dict[str, Any]]:
        """Parse each folder's ``SKILL.md`` in one ``node`` process."""

        node = shutil.which("node", path=self._env.get("PATH"))
        if node is None:
            raise _ParserUnavailable("node was not found")
        module = _parser_module(self._package) if self._package else None
        if module is None:
            raise _ParserUnavailable("its parseSkillFrontmatter module was not found")
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise _ParserUnavailable("no time is left before the startup deadline")
        try:
            completed = subprocess.run(
                [node, "--input-type=module", "-e", _PARSE_SCRIPT, "--", *module],
                input=json.dumps([str(folder / SKILL_FILE) for folder in folders]),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=self._env,
                timeout=remaining,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise _ParserUnavailable(
                f"it did not finish within {remaining:.1f} seconds"
            ) from None
        except OSError as error:
            raise _ParserUnavailable(str(error)) from None
        if completed.returncode != 0:
            detail = (completed.stderr.strip().splitlines() or ["no output"])[-1]
            raise _ParserUnavailable(f"node failed: {detail}")
        try:
            rows = json.loads(completed.stdout)
        except ValueError:
            rows = None
        if not (
            isinstance(rows, list)
            and len(rows) == len(folders)
            and all(isinstance(row, dict) for row in rows)
        ):
            raise _ParserUnavailable("node returned an unexpected result")
        return rows


def _parser_module(package: Path) -> tuple[str, str] | None:
    """Return the module that exports ``parseSkillFrontmatter``, and its export name."""

    try:
        modules = sorted((package / "dist").glob("frontmatter-*.mjs"))
    except OSError:
        return None
    for module in modules:
        try:
            match = _PARSER_EXPORT.search(module.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
        if match:
            return str(module), match.group(1)
    return None


def _named(
    folder: Path,
    name: str | None,
    *,
    valid: bool | None,
    name_uncertain: bool = False,
) -> CandidateInfo:
    """Return ``name``, or apply OpenClaw's trimmed folder name fallback."""

    if not name and folder.name.strip() != folder.name:
        name = folder.name.strip()
        if not name:
            # OpenClaw refuses a skill without a name ("name is required").
            return CandidateInfo(None, valid=False, reason="no_name")
    return CandidateInfo(name or None, valid=valid, name_uncertain=name_uncertain)


def apply_status(plan_: SkillPlan, status: Any, *, fallback: bool = False) -> bool:
    """Refine the report with ``openclaw skills list --json`` for the agent.

    Each name's listed source says which copy won it, so workspace and
    shipped copies OpenClaw refused for a reason Fabric does not check, such
    as a hard-linked ``SKILL.md``, stop hiding other copies, and the clash
    records name OpenClaw's winner. Configured skills OpenClaw does not list
    were rejected by its loader and are reported as skipped with
    ``rejected_by_openclaw``. Selected bundled skills that do not load move
    to ``unavailable`` with OpenClaw's reason.

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
    if plan_.configured is not None:
        plan_.native_refused = _refused_native(plan_.native, listed)
        plan_.merge_native()
        load = plan_.load or contract_skills.DirectoryLoad()
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
            changed = _reject(plan_, plan_.configured, rejected, fallback=fallback)
            plan_.merge_native()
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
    """Record skills OpenClaw refused, replacing a refused winner when allowed.

    ``load`` is the configured load, before native sources; the result
    replaces ``plan_.configured``.
    """

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
    plan_.configured = contract_skills.DirectoryLoad(
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
    """Return OpenClaw's discovery rules."""

    return contract_skills.DiscoveryRules(
        traversal="recursive",
        nested_skills="stop_at_skill",
        max_depth=max_depth,
        name_fallback=True,
        excluded_dir_names=frozenset({"node_modules"}) | hidden,
        allow_symlink_escape=False,
        subroot_depths={"skills": SKILLS_ROOT_DEPTH},
    )


def _select_bundled(
    bundled: contract.BundledSkills, inventory: BundledInventory
) -> tuple[bool, list[str]]:
    """Return whether to stage the shipped skills, and the names ``bundled`` selects."""

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
        selected: list[str] = []
    elif bundled.include is not None:
        selected = sorted(set(bundled.include))
    else:
        selected = sorted(known - set(bundled.exclude or ()))
    return set(selected) != known, selected


def _load_directories(
    directories: Sequence[contract.SkillDirectory],
    *,
    duplicates: contract.SkillDuplicatePolicy | None,
    base_dir: Path,
    reader: contract_skills.CandidateReader,
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
    # Entries whose scanned skills/ subfolder holds SKILL.md: OpenClaw loads
    # that subfolder as one skill.
    promoted_skill: set[int] = set()
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
        elif scanned != folder and _has_skill_file(scanned):
            promoted_skill.add(index)
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
        if index in promoted_skill:
            return replace(_rules(depths[index], hidden), folder_root_is_skill=True)
        return _rules(depths[index], hidden)

    try:
        load = contract_skills.load_directories(
            entries,
            rules=rules,
            duplicates=duplicates,
            base_dir=base_dir,
            reader=reader,
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


def _native_copies(
    *,
    workspace: Path,
    inventory: BundledInventory | None,
    bundled: set[str] | None,
    custodian: bool,
    reader: contract_skills.CandidateReader,
) -> dict[str, list[NativeCopy]]:
    """Return the skills OpenClaw loads at start from its own sources, by name.

    Those are the workspace's ``skills/`` and ``.agents/skills/`` and the
    shipped skills. ``bundled`` is the selected shipped names, or ``None``
    when every shipped skill loads; a left-out shipped skill is not staged,
    so it hides nothing. Custodian skills count only when the agent owns
    them. The state folder and Skill Workshop folder are new and empty at
    start, so they provide nothing yet. Skills ``reader`` says OpenClaw
    refuses are left out, so they hide nothing.
    """

    native: dict[str, list[NativeCopy]] = {}
    if inventory is not None:
        shipped = [(inventory.skills, 0)]
        if custodian:
            shipped.append((inventory.custodian, 1))
        for skills, rank in shipped:
            for name, path in skills.items():
                if bundled is None or name in bundled:
                    native.setdefault(name, []).append(
                        NativeCopy("bundled", path, rank)
                    )
    for rank, skill in _workspace_skills(workspace, reader):
        native.setdefault(skill.name, []).append(NativeCopy("local", skill.path, rank))
    return native


def _workspace_skills(
    workspace: Path, reader: contract_skills.CandidateReader
) -> list[tuple[int, contract_skills.FoundSkill]]:
    """Return the skills of the workspace's ``.agents/skills/`` and ``skills/``, with their ranks."""

    found: list[tuple[int, contract_skills.FoundSkill]] = []
    for rank, relative in enumerate(_LOCAL_SKILL_FOLDERS, start=2):
        folder = workspace / relative
        if not _is_dir(folder):
            continue
        try:
            local = contract_skills.load_directories(
                [{"folder": folder}],
                rules=_rules(SKILLS_ROOT_DEPTH, _hidden_dir_names([folder])),
                duplicates=None,
                reader=reader,
            )
        except contract_skills.SkillLoadError:
            continue
        found += [(rank, skill) for skill in local.selected]
    return found


def _read_ahead(
    reader: contract_skills.CandidateReader,
    directories: Sequence[contract.SkillDirectory] | None,
    *,
    base_dir: Path,
    bundled_dir: Path | None,
    workspace: Path,
) -> contract_skills.CandidateReader:
    """Return ``reader`` with every candidate of the plan's scans already read.

    The plan scans the configured entries, the shipped skills, and the
    workspace in separate ``load_directories`` calls. A first pass discovers
    their candidates with a placeholder reader so that ``reader`` reads them
    all in one call, such as one ``node`` process. The returned reader
    answers from that result.
    """

    cache: dict[Path, CandidateInfo] = {}

    def cached(folders: Sequence[Path]) -> dict[Path, CandidateInfo]:
        missing = [folder for folder in folders if folder not in cache]
        if missing:
            cache.update(reader(missing))
        return {folder: cache[folder] for folder in folders}

    found: list[Path] = []

    def record(folders: Sequence[Path]) -> dict[Path, CandidateInfo]:
        found.extend(folders)
        # A distinct name per folder, so that no scan stops at a duplicate.
        return {folder: CandidateInfo(str(folder)) for folder in folders}

    with contextlib.suppress(lifecycle.LifecycleError):
        if directories is not None:
            _load_directories(
                directories, duplicates=None, base_dir=base_dir, reader=record
            )
    if bundled_dir is not None:
        bundled_inventory(bundled_dir, record)
    _workspace_skills(workspace, record)
    if found:
        cached(list(dict.fromkeys(found)))
    return cached


def _refused_native(
    native: Mapping[str, Sequence[NativeCopy]], listed: Mapping[str, list[Any]]
) -> set[Path]:
    """Return the native copies OpenClaw's list shows it does not load.

    The source a name is listed with outranks every copy it hides, so a copy
    of a higher source was refused. A name that is not listed lost every
    copy. Where the listed source is not one Fabric models, no copy is known
    to be refused.
    """

    refused: set[Path] = set()
    for name, copies in native.items():
        rows = listed.get(name)
        if not rows:
            refused.update(copy.path for copy in copies)
            continue
        rank = _SOURCE_RANKS.get(str(rows[0].get("source")))
        if rank is not None:
            refused.update(copy.path for copy in copies if copy.rank > rank)
    return refused


def _with_native_sources(
    load: contract_skills.DirectoryLoad,
    native: Mapping[str, Sequence[NativeCopy]],
    refused: set[Path] | None,
) -> contract_skills.DirectoryLoad:
    """Record every name that more than one source OpenClaw loads at start provides.

    OpenClaw keeps the copy from the highest source (``mergeSkillRecords``):
    the workspace's ``skills/`` over its ``.agents/skills/``, both over the
    shipped skills, and those over ``skills.load.extraDirs``. Between shipped
    skills with one name, the Custodian copy wins. Native copies in
    ``refused`` hide nothing.

    A configured folder that is also a native copy, such as a skill in the
    workspace's ``skills/``, is one copy with the native source's rank, so it
    can win the name. A configured copy that a higher source hides is
    ``shadowed``; when a configured copy wins, the outcome is ``selected``.
    A clash with no configured copy is ``selected`` when one copy wins, or
    ``ambiguous`` when the winning source holds several copies. Clashes only
    between configured entries outside OpenClaw's own sources keep the
    outcome already recorded.
    """

    existing = {record.name: record for record in load.duplicates}
    external: dict[str, list[tuple[str, Path, str]]] = {}
    for name, record in existing.items():
        external[name] = [copy for copy in record.copies if copy[0] == "external"]
    for skill in load.selected:
        if skill.name not in existing:
            external.setdefault(skill.name, []).append(
                ("external", skill.path, skill.field)
            )
    updated: dict[str, contract_skills.DuplicateRecord] = {}
    # The configured copies that load, by name, for each updated record.
    winners: dict[str, list[tuple[Path, str]]] = {}
    for name in sorted(native):
        copies = [
            copy for copy in native[name] if refused is None or copy.path not in refused
        ]
        configured = external.get(name, [])
        # A configured folder that is also a native one is the same copy,
        # with the native source's rank.
        native_rank: dict[Path, float] = {}
        for copy in copies:
            native_rank[copy.path] = max(native_rank.get(copy.path, -1), copy.rank)
        same = {path for _, path, _ in configured}
        others = sorted(
            (copy for copy in copies if copy.path not in same),
            key=lambda copy: copy.rank,
        )
        ranked = [
            (native_rank.get(path, -1), ("external", path, field))
            for _, path, field in configured
        ] + [(copy.rank, (copy.origin, copy.path, "")) for copy in others]
        if len(ranked) < 2 or all(rank < 0 for rank, _ in ranked):
            continue
        top_rank = max(rank for rank, _ in ranked)
        top = [copy for rank, copy in ranked if rank == top_rank]
        winner = top[0][1] if len(top) == 1 else None
        top_configured = [(path, field) for origin, path, field in top if field]
        outcome: contract_skills.DuplicateOutcome
        if top_configured:
            outcome = "selected" if winner else "ambiguous"
        else:
            outcome = (
                "shadowed" if configured else "selected" if winner else "ambiguous"
            )
        winners[name] = top_configured
        updated[name] = contract_skills.DuplicateRecord(
            name,
            (*configured, *((copy.origin, copy.path, "") for copy in others)),
            outcome,
            winner,
        )
    if not updated:
        return load
    kinds = {field: kind for field, kind, _ in load.entries}
    order = {field: index for index, (field, _, _) in enumerate(load.entries)}
    selected = [skill for skill in load.selected if skill.name not in updated]
    for name, copies in winners.items():
        for path, field in copies:
            found = next(
                (
                    skill
                    for skill in load.selected
                    if skill.name == name and skill.path == path
                ),
                None,
            )
            # A copy an explicit policy dropped still loads from its source.
            selected.append(
                found
                or contract_skills.FoundSkill(
                    name,
                    path,
                    field,
                    "skill" if kinds.get(field) == "skill" else "folder",
                )
            )
    selected.sort(key=lambda skill: order.get(skill.field, len(order)))
    return contract_skills.DirectoryLoad(
        selected=selected,
        entries=load.entries,
        empty_entries=load.empty_entries,
        skipped=load.skipped,
        duplicates=[
            *(record for record in load.duplicates if record.name not in updated),
            *updated.values(),
        ],
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
    by exact name. The scan follows symlinks as discovery does, but lists a
    folder again only when it is reached with more levels left to search.
    """

    names: set[str] = set()
    # Levels left to search below each folder already listed, by identity.
    listed: dict[tuple[int, int] | Path, int] = {}
    for folder in folders:
        stack = [(folder, HIDDEN_SCAN_DEPTH)]
        while stack:
            current, levels = stack.pop()
            try:
                status = os.stat(current)
                identity = (
                    (status.st_dev, status.st_ino)
                    if status.st_ino
                    else current.resolve()
                )
            except (OSError, RuntimeError):
                continue
            if listed.get(identity, 0) >= levels:
                continue
            listed[identity] = levels
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
                elif child.name != "node_modules" and levels > 1:
                    stack.append((Path(child.path), levels - 1))
    return frozenset(names)


def _unique(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
