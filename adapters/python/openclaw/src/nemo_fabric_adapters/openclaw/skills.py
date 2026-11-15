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
  ``unsupported_frontmatter``.
- OpenClaw opens a ``SKILL.md`` outside its shipped skills with hard links
  refused (``shouldRejectHardlinkedPluginFiles``), unless ``OPENCLAW_NIX_MODE``
  is ``1`` and the skill is in the Nix store. Such a skill is skipped with
  ``hardlinked_skill_file`` before any duplicate check too.
- OpenClaw names a skill without a front matter name after the path it
  reaches the skill through, and loads a folder once per such path, while
  Fabric's discovery loads a folder once. So a skill found through a
  symbolic link is passed through that link, never its target, and a second
  ``skill`` entry that reaches an already loaded folder is not passed. A
  ``folder`` entry passed whole loads a linked skill once per name, and each
  name is reported (:func:`_with_alias_copies`).

At start ``openclaw skills list`` reports any other skill OpenClaw refuses,
such as one whose ``SKILL.md`` links outside its folder; under ``first_wins``
or ``last_wins`` the next copy replaces a refused winner. The list names each
skill's source but not its folder, so ``openclaw skills info`` locates the
loaded copy of each name the list leaves open (:func:`names_to_locate`): a
clash between copies of one source, or a name Fabric knows no copy of, such
as a skill an OpenClaw plugin adds. The report's ``loaded`` lists every skill
OpenClaw loads: the configured ones, and the shipped, plugin, and workspace
ones with their folders.

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
import stat
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
from typing import Literal

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
# The skip reason of a skill whose SKILL.md OpenClaw refuses as hard-linked.
HARDLINKED = "hardlinked_skill_file"
# NIX_STORE_ROOT of hardlink-policy: hard links are allowed below it in Nix mode.
NIX_STORE = "/nix/store"
# The rank of a skill an OpenClaw plugin adds: it is read with the extraDirs
# roots but after them, so it hides a configured copy and loses to the
# shipped skills (loadWorkspaceSkillSourceEntries).
PLUGIN_RANK = -0.5
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

    origin: Literal["bundled", "local"]
    """``bundled`` for a shipped skill, ``local`` for a workspace one."""
    path: Path
    rank: float
    """The source's precedence: a higher rank wins a name."""


@dataclass(frozen=True)
class LocatedCopy:
    """The copy of a name OpenClaw loads, from ``openclaw skills info --json``."""

    source: str
    """OpenClaw's source label, such as ``openclaw-extra``."""
    path: Path
    """The resolved folder that holds the loaded ``SKILL.md`` (``baseDir``)."""


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
    reached: dict[tuple[str, Path], str] = dataclass_field(default_factory=dict)
    """The path OpenClaw discovers each configured skill through, by entry
    field and resolved folder: an alias stays an alias, because OpenClaw
    names a skill without a front matter name after that path's folder."""
    located: dict[str, LocatedCopy] = dataclass_field(default_factory=dict)
    """The copy OpenClaw loads for each name ``openclaw skills info`` located."""
    package: Path | None = None
    """The installed OpenClaw package, to tell a plugin's shipped skill."""

    def merge_native(self) -> None:
        """Derive ``load`` from ``configured`` and the native copies."""

        if self.configured is None:
            return
        self.load = _with_native_sources(
            self.configured, self.native, self.native_refused, self.located
        )

    @property
    def needs_status(self) -> bool:
        """Return whether OpenClaw's own skill list can refine the report."""

        return self.configured is not None

    def other_loaded(self) -> list[contract_skills.LoadedSkill]:
        """Return the skills OpenClaw loads from its own sources, by name.

        A name that a configured skill wins is left out, and so is a name
        whose winning copy Fabric cannot tell: several copies in the winning
        folder that ``openclaw skills info`` did not locate.
        """

        external = {skill.name for skill in self.load.selected} if self.load else set()
        refused = self.native_refused or set()
        result: list[contract_skills.LoadedSkill] = []
        for name in sorted(self.native):
            copies = [copy for copy in self.native[name] if copy.path not in refused]
            if name in external or not copies:
                continue
            top = max(copy.rank for copy in copies)
            winners = [copy for copy in copies if copy.rank == top]
            if len(winners) > 1:
                where = self.located.get(name)
                winners = [
                    copy for copy in winners if where and copy.path == where.path
                ]
            if len(winners) == 1:
                result.append(
                    contract_skills.LoadedSkill(
                        name, winners[0].origin, winners[0].path
                    )
                )
        return result

    def report(self) -> dict[str, Any] | None:
        """Return the ``output.skills`` block, or ``None`` without structured skills.

        ``loaded`` lists every skill OpenClaw loads: the configured ones in
        entry order, then the shipped, plugin, and workspace ones by name.
        """

        if self.configured is None:
            return None
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
        return contract_skills.report_block(
            load, bundled=bundled, other_loaded=self.other_loaded()
        )

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
    nix_mode: bool = False,
    package: Path | None = None,
) -> SkillPlan:
    """Check the structured skill settings and return what OpenClaw should load.

    ``reader`` reads each candidate as OpenClaw does (:func:`candidate_reader`);
    every candidate the plan meets is passed to it in one call.
    ``custodian`` says whether the configured agent owns the Custodian skills
    (:func:`owns_custodian_skills`). ``nix_mode`` says whether OpenClaw runs
    with ``OPENCLAW_NIX_MODE=1``, which allows hard-linked skills in the Nix
    store. ``package`` is the installed OpenClaw package (:func:`package_root`).

    Raises:
        lifecycle.LifecycleError: If an entry cannot be loaded, a bundled
            skill name is unknown, or the bundled inventory cannot be found.
    """

    skills = config.skills
    result = SkillPlan(custodian=custodian, package=package)
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
    # OpenClaw refuses a hard-linked SKILL.md outside its shipped skills.
    unshipped = _refusing_hardlinks(reader, nix_mode=nix_mode)
    if skills.directories is None:
        load = contract_skills.DirectoryLoad()
    else:
        load = _load_directories(
            skills.directories,
            duplicates=skills.duplicates,
            base_dir=base_dir,
            reader=unshipped,
        )
        discovered = _discovered_paths(skills.directories, base_dir)
        for field, candidates in discovered:
            for reached, resolved in candidates:
                result.reached.setdefault((field, resolved), str(reached))
        if skills.duplicates is None:
            result.extra_dirs = _passed_entries(load, base_dir, skills.directories)
            load = _with_alias_copies(load, discovered, unshipped)
            load = _with_native_extra_outcomes(load)
        else:
            # Each selected skill once, through the path it was discovered
            # by, so OpenClaw names it as Fabric does.
            result.extra_dirs = _unique(
                result.reached.get((skill.field, skill.path), str(skill.path))
                for skill in load.selected
            )
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
        reader=unshipped,
    )
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


def located_copy(info: Any) -> LocatedCopy | None:
    """Return the copy ``openclaw skills info <name> --json`` says loads."""

    if not isinstance(info, dict):
        return None
    source, base_dir = info.get("source"), info.get("baseDir")
    if not isinstance(source, str) or not isinstance(base_dir, str) or not base_dir:
        return None
    return LocatedCopy(source, _resolved(Path(base_dir)))


def names_to_locate(plan_: SkillPlan, status: Any) -> list[str]:
    """Return the listed names whose loaded copy ``openclaw skills list`` leaves open.

    The list gives each name's source but not its folder. That settles a
    name with one known copy of that source. Locate a name with several
    copies of the listed source that OpenClaw is given, such as copies from
    two entries passed without a policy, and a listed name Fabric knows no
    copy of, such as a skill an OpenClaw plugin adds.
    """

    listed = _listed(status)
    if plan_.configured is None or not listed:
        return []
    load = plan_.load or contract_skills.DirectoryLoad()
    known = {skill.name for skill in load.selected} | set(plan_.native)
    known |= {record.name for record in load.duplicates}
    passed = _passed_paths(plan_)
    names: list[str] = []
    for name, rows in listed.items():
        if name not in known:
            names.append(name)
            continue
        rank = _SOURCE_RANKS.get(str(rows[0].get("source")))
        record = next((item for item in load.duplicates if item.name == name), None)
        if record is None or rank is None:
            continue
        native_rank = {copy.path: copy.rank for copy in plan_.native.get(name, [])}
        same_rank = {
            path
            for origin, path, _ in record.copies
            if native_rank.get(path, -1 if origin == "external" else None) == rank
            and (
                origin != "external"
                or path in native_rank
                or passed is None
                or path in passed
            )
        }
        if len(same_rank) != 1:
            names.append(name)
    return sorted(names)


def apply_status(
    plan_: SkillPlan,
    status: Any,
    *,
    located: Mapping[str, LocatedCopy] | None = None,
    fallback: bool = False,
) -> bool:
    """Refine the report with ``openclaw skills list --json`` for the agent.

    Each name's listed source says which source won it, so workspace and
    shipped copies OpenClaw refused for a reason Fabric does not check stop
    hiding other copies, and the clash records name OpenClaw's winner.
    ``located`` gives the copy ``openclaw skills info`` says loads for the
    names :func:`names_to_locate` returns, which settles which copy of one
    source loads. A configured winner OpenClaw does not list, or whose name
    it loads from another configured copy, was rejected by its loader and is
    reported as skipped with ``rejected_by_openclaw``; the copy that loads
    instead is reported as selected. A located copy Fabric does not know,
    such as a plugin's skill, is added to the copies of its name. Selected
    bundled skills that do not load move to ``unavailable`` with OpenClaw's
    reason.

    Under ``first_wins`` or ``last_wins`` only the winning copy of a name is
    passed to OpenClaw, so a rejected winner would leave the name without a
    skill. With ``fallback``, the next copy in policy order replaces it in
    ``extra_dirs`` and the report; the caller must then write the new
    ``skills.load.extraDirs`` and list again. Without it, the name is
    reported with no selected copy.

    Returns:
        Whether ``extra_dirs`` changed.
    """

    listed = _listed(status)
    if listed is None:
        return False
    changed = False
    if plan_.configured is not None:
        plan_.located.update(located or {})
        _add_located(plan_)
        _settle_ambiguous(plan_)
        plan_.native_refused = _refused_native(plan_, listed)
        plan_.merge_native()
        load = plan_.load or contract_skills.DirectoryLoad()
        shadowed = {
            (record.name, path)
            for record in load.duplicates
            if record.outcome == "shadowed"
            for _, path, _ in record.copies
        }
        rejected: list[contract_skills.FoundSkill] = []
        replacements: dict[str, tuple[Path, str]] = {}
        for skill in load.selected:
            if (skill.name, skill.path) in shadowed:
                continue
            if skill.name not in listed:
                rejected.append(skill)
                continue
            where = plan_.located.get(skill.name)
            if (
                where is None
                or where.source != "openclaw-extra"
                or where.path == skill.path
            ):
                continue
            other = _configured_copy(plan_.configured, skill.name, where.path)
            if other is not None:
                rejected.append(skill)
                replacements[skill.name] = other
        if rejected:
            changed = _reject(
                plan_,
                plan_.configured,
                rejected,
                replacements=replacements,
                fallback=fallback,
            )
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


def _listed(status: Any) -> dict[str, list[dict[str, Any]]] | None:
    """Return the rows of ``openclaw skills list --json`` by name."""

    rows = status.get("skills") if isinstance(status, dict) else None
    if not isinstance(rows, list):
        return None
    listed: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("name"), str):
            listed.setdefault(row["name"], []).append(row)
    return listed


def _passed_paths(plan_: SkillPlan) -> set[Path] | None:
    """Return the configured folders passed one by one, or ``None`` for whole entries."""

    if plan_.duplicates is None:
        return None
    passed = set(plan_.extra_dirs)
    return {path for (_, path), reached in plan_.reached.items() if reached in passed}


def _configured_copy(
    load: contract_skills.DirectoryLoad, name: str, path: Path
) -> tuple[Path, str] | None:
    """Return the configured copy of ``name`` at ``path``, with its entry."""

    for skill in load.selected:
        if skill.name == name and skill.path == path:
            return skill.path, skill.field
    for record in load.duplicates:
        if record.name == name:
            for origin, copy_path, field in record.copies:
                if origin == "external" and copy_path == path:
                    return copy_path, field
    return None


def _add_located(plan_: SkillPlan) -> None:
    """Add each located copy Fabric does not know to the native copies of its name.

    A plugin's skill is read with the ``extraDirs`` roots but is not
    configured; it is ``bundled`` when it lives in the OpenClaw package.
    """

    configured = plan_.configured or contract_skills.DirectoryLoad()
    for name, where in plan_.located.items():
        copies = plan_.native.setdefault(name, [])
        if any(copy.path == where.path for copy in copies) or _configured_copy(
            configured, name, where.path
        ):
            continue
        if where.source == "openclaw-extra":
            rank: float | None = PLUGIN_RANK
        else:
            rank = _SOURCE_RANKS.get(where.source)
        if rank is None:
            continue
        shipped = where.source in BUNDLED_SOURCES or (
            plan_.package is not None
            and where.path.is_relative_to(_resolved(plan_.package))
        )
        copies.append(NativeCopy("bundled" if shipped else "local", where.path, rank))
        copies.sort(key=lambda copy: copy.rank)
    for name in [name for name, copies in plan_.native.items() if not copies]:
        del plan_.native[name]


def _settle_ambiguous(plan_: SkillPlan) -> None:
    """Pick the located copy of each clash between configured copies of one entry.

    Without a policy, OpenClaw loads one of several copies inside one entry
    in its own path order, which Fabric reports as ``ambiguous``; the copy
    ``openclaw skills info`` located is the one that loads.
    """

    load = plan_.configured
    if load is None:
        return
    records = []
    losers: set[tuple[str, Path]] = set()
    for record in load.duplicates:
        where = plan_.located.get(record.name)
        paths = [path for origin, path, _ in record.copies if origin == "external"]
        if record.outcome != "ambiguous" or where is None or where.path not in paths:
            records.append(record)
            continue
        losers.update((record.name, path) for path in paths if path != where.path)
        records.append(replace(record, outcome="selected", selected=where.path))
    if losers:
        plan_.configured = replace(
            load,
            selected=[
                skill
                for skill in load.selected
                if (skill.name, skill.path) not in losers
            ],
            duplicates=records,
        )


def _reject(
    plan_: SkillPlan,
    load: contract_skills.DirectoryLoad,
    rejected: list[contract_skills.FoundSkill],
    *,
    replacements: Mapping[str, tuple[Path, str]],
    fallback: bool,
) -> bool:
    """Record skills OpenClaw refused, replacing a refused winner when known or allowed.

    ``load`` is the configured load, before native sources; the result
    replaces ``plan_.configured``. ``replacements`` holds the configured copy
    OpenClaw loads instead of a refused winner, by name; it is already
    passed. Otherwise, with ``fallback``, the next copy in policy order
    replaces the winner in ``extra_dirs``.
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
        replacement = replacements.get(skill.name)
        passed = replacement is not None
        if replacement is None and fallback:
            replacement = _next_copy(
                record, skill.path, plan_.duplicates, plan_.rejected
            )
        duplicates[index] = replace(
            record, selected=replacement[0] if replacement else None
        )
        if replacement is None:
            continue
        path, field = replacement
        if not any(item.name == skill.name and item.path == path for item in selected):
            selected.append(
                contract_skills.FoundSkill(
                    skill.name,
                    path,
                    field,
                    "skill" if kinds[field] == "skill" else "folder",
                )
            )
        if passed:
            continue
        old = plan_.reached.get((skill.field, skill.path), str(skill.path))
        new = plan_.reached.get((field, path), str(path))
        plan_.extra_dirs = [
            new if folder == old else folder for folder in plan_.extra_dirs
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
    # A folder can load under several names, so losers are kept by name.
    losers: set[tuple[str, Path]] = set()
    for record in load.duplicates:
        last = max(order.get(field, -1) for _, _, field in record.copies)
        in_last = [copy for copy in record.copies if order.get(copy[2], -1) == last]
        if record.outcome != "ambiguous" or len(in_last) != 1:
            records.append(record)
            continue
        winner = in_last[0][1]
        losers.update(
            (record.name, path) for _, path, _ in record.copies if path != winner
        )
        records.append(
            contract_skills.DuplicateRecord(
                record.name, record.copies, "selected", winner
            )
        )
    return contract_skills.DirectoryLoad(
        selected=[
            skill for skill in load.selected if (skill.name, skill.path) not in losers
        ],
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


def _refused_native(plan_: SkillPlan, listed: Mapping[str, list[Any]]) -> set[Path]:
    """Return the native copies OpenClaw's list shows it does not load.

    The copy that loads a name outranks every copy it hides, so a copy of a
    higher source was refused. That copy is the located one when ``openclaw
    skills info`` located it, or else one of the source the name is listed
    with. A name that is not listed lost every copy. Where the listed source
    is not one Fabric models, no copy is known to be refused.
    """

    refused: set[Path] = set()
    for name, copies in plan_.native.items():
        rows = listed.get(name)
        if not rows:
            refused.update(copy.path for copy in copies)
            continue
        rank = _SOURCE_RANKS.get(str(rows[0].get("source")))
        where = plan_.located.get(name)
        if where is not None:
            # A configured copy that is no native copy has the extraDirs rank.
            rank = next(
                (copy.rank for copy in copies if copy.path == where.path),
                _SOURCE_RANKS["openclaw-extra"]
                if where.source == "openclaw-extra"
                else rank,
            )
        if rank is not None:
            refused.update(copy.path for copy in copies if copy.rank > rank)
    return refused


def _with_native_sources(
    load: contract_skills.DirectoryLoad,
    native: Mapping[str, Sequence[NativeCopy]],
    refused: set[Path] | None,
    located: Mapping[str, LocatedCopy],
) -> contract_skills.DirectoryLoad:
    """Record every name that more than one source OpenClaw loads at start provides.

    OpenClaw keeps the copy from the highest source (``mergeSkillRecords``):
    the workspace's ``skills/`` over its ``.agents/skills/``, both over the
    shipped skills, and those over ``skills.load.extraDirs``. A located
    plugin skill ranks between the shipped skills and ``extraDirs``. Between
    shipped skills with one name, the Custodian copy wins. Native copies in
    ``refused`` hide nothing.

    A configured folder that is also a native copy, such as a skill in the
    workspace's ``skills/``, is one copy with the native source's rank, so it
    can win the name. A configured copy that a higher source hides is
    ``shadowed``; when a configured copy wins, the outcome is ``selected``.
    A clash with no configured copy is ``selected`` when one copy wins, or
    ``ambiguous`` when the winning source holds several copies, unless
    ``located`` names the one that loads. Clashes only
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
        # A plugin's skill ranks below zero too, but it is no configured copy.
        if len(ranked) < 2 or (not others and all(rank < 0 for rank, _ in ranked)):
            continue
        top_rank = max(rank for rank, _ in ranked)
        top = [copy for rank, copy in ranked if rank == top_rank]
        where = located.get(name)
        if len(top) > 1 and where is not None:
            # OpenClaw's scan order picked the located copy.
            top = [copy for copy in top if copy[1] == where.path] or top
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


def _passed_entries(
    load: contract_skills.DirectoryLoad,
    base_dir: Path,
    directories: Sequence[contract.SkillDirectory],
) -> list[str]:
    """Return the entry paths to pass without a policy, as configured.

    A ``skill`` entry whose folder an earlier entry already supplies is
    not passed again: OpenClaw names a skill without a front matter name
    after the path it reaches it through, so a second alias of one folder
    would load it twice under two names, while the folder loads once.
    """

    first: dict[Path, str] = {}
    for skill in load.selected:
        first.setdefault(skill.path, skill.field)
    passed: list[str] = []
    for (field, kind, resolved), entry in zip(load.entries, directories, strict=True):
        if kind == "skill" and first.get(resolved, field) != field:
            continue
        passed.append(str(_passed_path(base_dir, entry)))
    return _unique(passed)


def _passed_path(base_dir: Path, entry: contract.SkillDirectory) -> Path:
    """Return the path to pass OpenClaw for an entry, reaching the folder Fabric loads.

    OpenClaw normalizes ``..`` lexically, while Fabric follows it through the
    file system, so after a symbolic link the two would reach different
    folders. The parent is resolved through the file system; the last
    component is kept, as OpenClaw names a skill without a front matter name
    after it.
    """

    path = Path(os.getcwd(), base_dir, entry.path)
    if path.name in ("", ".."):
        return Path(os.path.realpath(path))
    return Path(os.path.realpath(path.parent), path.name)


def _discovered_paths(
    directories: Sequence[contract.SkillDirectory], base_dir: Path
) -> list[tuple[str, list[tuple[Path, Path]]]]:
    """Return each entry's field and the ``(reached, resolved)`` skill folders OpenClaw finds."""

    found: list[tuple[str, list[tuple[Path, Path]]]] = []
    for index, entry in enumerate(directories):
        root = _passed_path(base_dir, entry)
        if entry.folder is not None:
            candidates = _openclaw_candidates(root)
        else:
            resolved = _real(root)
            candidates = [(root, resolved)] if resolved is not None else []
        found.append((f"skills.directories[{index}]", candidates))
    return found


def _openclaw_candidates(root: Path) -> list[tuple[Path, Path]]:
    """Return the skill folders OpenClaw discovers in one ``extraDirs`` root.

    Each is ``(reached, resolved)``: the path discovery reaches it through,
    which names a skill without a front matter name, and its real folder.
    They come in OpenClaw's scan order, breadth first by name
    (``discoverSkillCandidates``). Unlike Fabric's discovery, a folder
    reached through several paths, such as through a symbolic link beside
    its target, is listed once per path, as OpenClaw loads it once per path.
    """

    root = Path(os.path.abspath(root))
    root_real = _real(root)
    base = _scanned_folder(root)
    base_real = _real(base)
    if root_real is None or base_real is None or not _inside(base_real, root_real):
        return []
    if _has_skill_file(base):
        return [(base, base_real)]
    nested = base / "skills"
    grouped = base == root / "skills" or base.name == "skills"
    queue = [
        (base / name, 0 if name == "skills" and not _has_skill_file(base / name) else 1)
        for name in _child_dirs(base)
    ]
    found: list[tuple[Path, Path]] = []
    for path, depth in queue:
        real = _real(path)
        if real is None or not _inside(real, base_real):
            continue
        if _has_skill_file(path):
            found.append((path, real))
            continue
        deep = grouped or path == nested or path.is_relative_to(nested)
        if depth < (SKILLS_ROOT_DEPTH if deep else EXTRA_ROOT_DEPTH):
            queue.extend((path / name, depth + 1) for name in _child_dirs(path))
    return found


def _child_dirs(directory: Path) -> list[str]:
    """Return the folders OpenClaw searches below ``directory``, by name (``listChildDirectories``)."""

    try:
        with os.scandir(directory) as listing:
            children = list(listing)
    except OSError:
        return []
    names: list[str] = []
    for child in children:
        if child.name.startswith(".") or child.name == "node_modules":
            continue
        try:
            if child.is_dir():
                names.append(child.name)
        except OSError:
            continue
    return sorted(names)


def _with_alias_copies(
    load: contract_skills.DirectoryLoad,
    discovered: Sequence[tuple[str, Sequence[tuple[Path, Path]]]],
    reader: contract_skills.CandidateReader,
) -> contract_skills.DirectoryLoad:
    """Add the copies OpenClaw loads again through another path of a ``folder`` entry.

    Without a policy each ``folder`` entry is passed whole, and OpenClaw
    loads a skill folder once per path it reaches it through. A skill
    without a front matter name then loads once per alias name, such as
    beside a symbolic link to it; Fabric's discovery loads the folder once,
    so each other name is added as a copy of that folder. Copies with one
    name are one skill to OpenClaw, which keeps one per name.
    """

    names = {(skill.path, skill.name) for skill in load.selected}
    loaded = {path for path, _ in names}
    kinds = {field: kind for field, kind, _ in load.entries}
    added: list[contract_skills.FoundSkill] = []
    for field, candidates in discovered:
        if kinds.get(field) != "folder":
            continue
        reached = [(path, real) for path, real in candidates if real in loaded]
        if not reached:
            continue
        infos = reader(list(dict.fromkeys(real for _, real in reached)))
        for path, real in reached:
            info = infos[real]
            name = info.name or path.name.strip()
            if info.valid is False or not name or (real, name) in names:
                continue
            names.add((real, name))
            added.append(
                contract_skills.FoundSkill(
                    name,
                    real,
                    field,
                    "folder",
                    name_uncertain=info.name_uncertain,
                    validity_uncertain=info.valid is None,
                )
            )
    if not added:
        return load
    order = {field: index for index, (field, _, _) in enumerate(load.entries)}
    selected = sorted([*load.selected, *added], key=lambda skill: order[skill.field])
    by_name: dict[str, list[contract_skills.FoundSkill]] = {}
    for skill in selected:
        by_name.setdefault(skill.name, []).append(skill)
    return contract_skills.DirectoryLoad(
        selected=selected,
        entries=load.entries,
        empty_entries=load.empty_entries,
        skipped=load.skipped,
        duplicates=[
            contract_skills.DuplicateRecord(
                name,
                tuple(("external", skill.path, skill.field) for skill in copies),
                "ambiguous",
                None,
            )
            for name, copies in by_name.items()
            if len(copies) > 1
        ],
    )


def _refusing_hardlinks(
    reader: contract_skills.CandidateReader, *, nix_mode: bool
) -> contract_skills.CandidateReader:
    """Return ``reader`` refusing a hard-linked ``SKILL.md``, as OpenClaw does.

    OpenClaw opens every ``SKILL.md`` outside its shipped skills with hard
    links refused (``shouldRejectHardlinkedPluginFiles``): a regular file with
    more than one link, after following symbolic links, does not load. In
    Nix mode (``OPENCLAW_NIX_MODE=1``) a root in the Nix store is exempt;
    the skill's real folder stands for its root here, which only differs for
    a root of ``/`` or ``/nix``.
    """

    def read(folders: Sequence[Path]) -> Mapping[Path, CandidateInfo]:
        infos = dict(reader(folders))
        for folder in folders:
            if infos[folder].valid is False or (
                nix_mode
                and (
                    str(folder) == NIX_STORE or str(folder).startswith(NIX_STORE + "/")
                )
            ):
                continue
            try:
                status = os.stat(folder / SKILL_FILE)
            except OSError:
                continue
            if stat.S_ISREG(status.st_mode) and status.st_nlink > 1:
                infos[folder] = CandidateInfo(None, valid=False, reason=HARDLINKED)
        return infos

    return read


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
    """Return ``path`` resolved as far as it exists, as the loader's folders are.

    A missing path still comes back absolute, with its drive on Windows, so it
    compares equal to the same folder resolved elsewhere.
    """

    try:
        return Path(os.path.realpath(path))
    except (OSError, ValueError):
        return Path(os.path.abspath(path))


def _real(path: Path) -> Path | None:
    """Return the real path of an existing ``path``, or ``None``."""

    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError):
        return None


def _inside(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


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
