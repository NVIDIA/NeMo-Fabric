# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermes plugin settings: the native section, mounted plugins, and the start checks.

``harness.settings.plugins`` names plugins by where they come from:

``bundled.entries.<id>``
    Plugins that ship with Hermes, turned on or off.
``external.paths`` and ``external.entries.<id>``
    Plugin folders that something else, such as a mounted volume, provides;
    each is linked into ``$HERMES_HOME/plugins`` before discovery.

Each entry takes ``enabled``, ``config`` and ``capabilities``. The adapter
translates them to Hermes's native ``plugins`` section: ``enabled`` true or
false joins ``plugins.enabled`` or ``plugins.disabled``, ``config`` becomes
``plugins.entries.<id>.settings``, and ``capabilities`` becomes
``plugins.entries.<id>.granted_capabilities``. Hermes's own plugin controls,
such as ``isolation`` and the timeouts, pass through. The older
``plugins_enabled`` joins ``plugins.enabled``. After Hermes discovers its
plugins, :func:`check_enabled` confirms that every plugin the configuration
turns on has loaded.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nemo_fabric_adapters.common.plugins as common_plugins
import nemo_fabric_adapters.common.utils as common_utils
from nemo_fabric_adapters.common.lifecycle import LifecycleError


LOGGER = logging.getLogger(__name__)
SETTINGS_FIELD = "harness.settings"
PLUGINS_FIELD = f"{SETTINGS_FIELD}.plugins"
HARNESS = "hermes"
LABEL = "Hermes"
MANIFEST_FILES = ("plugin.yaml", "plugin.yml")
SECTIONS = ("bundled", "external")
# Hermes's own plugin controls, written to ``plugins`` as they are.
CONTROLS = (
    "load_timeout_seconds",
    "hook_callback_timeout",
    "clone_timeout_seconds",
    "auto_update_check_hours",
    "auto_apply",
    "isolation",
    "host",
)
BUNDLED_SOURCE = "bundled"


@dataclass(frozen=True)
class PluginEntry:
    """One ``bundled`` or ``external`` entry from ``harness.settings.plugins``."""

    section: str
    plugin: str
    entry: Mapping[str, Any]

    @property
    def field(self) -> str:
        return f"{PLUGINS_FIELD}.{self.section}.entries.{self.plugin}"

    @property
    def enabled(self) -> bool | None:
        value = self.entry.get("enabled")
        return value if isinstance(value, bool) else None


def _authored(settings: Mapping[str, Any]) -> Mapping[str, Any]:
    value = settings.get("plugins")
    return value if isinstance(value, Mapping) else {}


def _section(settings: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    value = _authored(settings).get(section)
    return value if isinstance(value, Mapping) else {}


def plugin_entries(settings: Mapping[str, Any]) -> list[PluginEntry]:
    """Every ``bundled`` and ``external`` entry, in that order.

    A plugin id may appear in only one section; otherwise fails with
    ``hermes_invalid_config``.
    """

    found: list[PluginEntry] = []
    sections: dict[str, str] = {}
    for section in SECTIONS:
        entries = _section(settings, section).get("entries")
        for plugin_id, entry in (
            entries if isinstance(entries, Mapping) else {}
        ).items():
            plugin = PluginEntry(
                section, plugin_id, entry if isinstance(entry, Mapping) else {}
            )
            if plugin_id in sections:
                raise LifecycleError(
                    "hermes_invalid_config",
                    f"Hermes plugin {plugin_id!r} is listed as both "
                    f"{sections[plugin_id]} and {section}",
                    metadata={"field": plugin.field, "plugin": plugin_id},
                )
            sections[plugin_id] = section
            found.append(plugin)
    return found


def external_paths(settings: Mapping[str, Any]) -> list[str]:
    """``plugins.external.paths``."""

    return common_utils.normalize_list(_section(settings, "external").get("paths"))


def enabled_ids(settings: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``(field, id)`` for each plugin the configuration turns on.

    Reads the ``bundled`` and ``external`` entries with ``enabled: true``,
    then the older ``plugins_enabled``, without repeats.
    """

    ids = [
        (plugin.field, plugin.plugin)
        for plugin in plugin_entries(settings)
        if plugin.enabled is True
    ]
    seen = {plugin_id for _, plugin_id in ids}
    for index, plugin_id in enumerate(
        common_utils.normalize_list(settings.get("plugins_enabled"))
    ):
        if plugin_id not in seen:
            seen.add(plugin_id)
            ids.append((f"{SETTINGS_FIELD}.plugins_enabled[{index}]", plugin_id))
    return ids


def disabled_ids(settings: Mapping[str, Any]) -> list[str]:
    """Plugin ids whose entry sets ``enabled: false``."""

    return [
        plugin.plugin for plugin in plugin_entries(settings) if plugin.enabled is False
    ]


def native_section(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The ``plugins`` section to write to ``config.yaml``.

    Empty lists and entries are left out, so a configuration without plugin
    settings writes no section.
    """

    authored = _authored(settings)
    section: dict[str, Any] = {
        name: copy.deepcopy(authored[name]) for name in CONTROLS if name in authored
    }
    enabled = [plugin_id for _, plugin_id in enabled_ids(settings)]
    disabled = disabled_ids(settings)
    if enabled:
        section["enabled"] = enabled
    if disabled:
        section["disabled"] = disabled
    entries: dict[str, Any] = {}
    for plugin in plugin_entries(settings):
        native: dict[str, Any] = {}
        if isinstance(plugin.entry.get("config"), Mapping):
            native["settings"] = copy.deepcopy(dict(plugin.entry["config"]))
        if plugin.entry.get("capabilities"):
            native["granted_capabilities"] = list(plugin.entry["capabilities"])
        if native:
            entries[plugin.plugin] = native
    if entries:
        section["entries"] = entries
    return section


def configured(settings: Mapping[str, Any]) -> bool:
    """Whether the configuration names any plugin the start checks cover."""

    return bool(enabled_ids(settings) or plugin_entries(settings))


def _matches(
    plugin_id: str, listed: Iterable[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    """Hermes's entries for ``plugin_id``.

    ``plugins.enabled`` selects a plugin by its registry key (``disk-cleanup``,
    ``image_gen/openai``) or its bare manifest name, so both are matched.
    """

    by_key = [entry for entry in listed if entry.get("key") == plugin_id]
    if by_key:
        return by_key
    return [entry for entry in listed if entry.get("name") == plugin_id]


def check_enabled(
    enabled: list[tuple[str, str]],
    listed: Iterable[Mapping[str, Any]],
    disabled: Iterable[str] = (),
) -> None:
    """Fail unless every ``(field, id)`` in ``enabled`` is present, enabled, and
    loaded, and not also turned off.

    ``listed`` is Hermes's ``get_plugin_manager().list_plugins()`` after
    discovery. Hermes silently ignores an unknown ``plugins.enabled`` id and
    records a load failure only as the entry's ``error``, so both would
    otherwise go unnoticed until the agent misses the capability.
    """

    listed = list(listed)
    disabled = set(disabled)

    def lookup(plugin_id: str) -> str | None:
        if plugin_id in disabled:
            return "its entry also sets enabled: false"
        matches = _matches(plugin_id, listed)
        if not matches:
            return None
        if any(entry.get("enabled") and not entry.get("error") for entry in matches):
            return ""
        return next(
            (str(entry["error"]) for entry in matches if entry.get("error")),
            "not enabled",
        )

    common_plugins.check_requested(enabled, lookup, harness=HARNESS, label=LABEL)


def check_bundled(
    entries: Iterable[PluginEntry], listed: Iterable[Mapping[str, Any]]
) -> None:
    """Fail when a ``bundled`` entry names a plugin Hermes doesn't ship."""

    listed = list(listed)
    for plugin in entries:
        if plugin.section != "bundled":
            continue
        matches = _matches(plugin.plugin, listed)
        if not matches:
            raise LifecycleError(
                common_plugins.not_found_code(HARNESS),
                f"{LABEL} plugin {plugin.plugin!r} was not found",
                metadata={"field": plugin.field, "plugin": plugin.plugin},
            )
        sources = {str(entry.get("source")) for entry in matches}
        if BUNDLED_SOURCE not in sources:
            raise LifecycleError(
                "hermes_invalid_config",
                f"Hermes plugin {plugin.plugin!r} doesn't ship with Hermes; "
                "list it under plugins.external",
                metadata={
                    "field": plugin.field,
                    "plugin": plugin.plugin,
                    "source": sorted(sources),
                },
            )


def check_loaded_plugins(settings: Mapping[str, Any]) -> None:
    """Run the start checks against Hermes's discovered plugins."""

    enabled = enabled_ids(settings)
    entries = plugin_entries(settings)
    if not enabled and not any(plugin.section == "bundled" for plugin in entries):
        return
    from hermes_cli.plugins import get_plugin_manager

    listed = get_plugin_manager().list_plugins()
    check_bundled(entries, listed)
    check_enabled(enabled, listed, disabled_ids(settings))


# --- Agent-installed plugins -----------------------------------------------

AGENT_INSTALLS = "installs"


def agent_environments(hermes_home: Path) -> list[Path]:
    """Plugin environments Hermes built in the home, ``installs/<key>/environments/<id>``.

    Hermes's own launcher also writes bootstrap records under ``installs``,
    so only a built environment counts.
    """

    installs = hermes_home / AGENT_INSTALLS
    if not installs.is_dir():
        return []
    return sorted(
        environment
        for environments in installs.glob("*/environments")
        if environments.is_dir()
        for environment in environments.iterdir()
    )


def check_no_agent_installs(hermes_home: Path) -> None:
    """Fail when the Hermes home holds a plugin environment an agent created.

    Importing Hermes switches the process to that environment, dropping NeMo
    Fabric's packages, so start fails before the import. The folder is left
    for the user to remove.
    """

    environments = agent_environments(hermes_home)
    if not environments:
        return
    installs = hermes_home / AGENT_INSTALLS
    raise LifecycleError(
        "hermes_plugin_agent_installed_unsupported",
        f"The Hermes home {str(hermes_home)!r} holds a plugin environment "
        f"({AGENT_INSTALLS}/) created by an agent plugin install. NeMo Fabric "
        "doesn't support agent-installed Hermes plugins. Delete "
        f"{str(installs)!r} and the plugin's folder under plugins/, and provide "
        "the plugin through plugins.external instead.",
        metadata={
            "path": str(installs),
            "hermes_home": str(hermes_home),
            "environments": [str(path) for path in environments],
        },
    )


# --- External plugins -------------------------------------------------------


@dataclass(frozen=True)
class MountedPlugin:
    """One ``plugins.external.paths`` entry: the plugin's manifest name and its folder."""

    index: int
    name: str
    path: Path

    @property
    def field(self) -> str:
        return f"{PLUGINS_FIELD}.external.paths[{self.index}]"


def _manifest_name(folder: Path, field: str) -> str:
    """The plugin's name as Hermes reads it: ``name`` in the manifest, else the
    folder's name."""

    # Parse the manifest as Hermes does: its YAML helper, not PyYAML, which
    # a Hermes install doesn't ship.
    import hermes_yaml as yaml

    manifest = next(
        (folder / name for name in MANIFEST_FILES if (folder / name).is_file()), None
    )
    if manifest is None:
        raise LifecycleError(
            "hermes_plugin_invalid",
            f"Hermes plugin folder {str(folder)!r} has no plugin.yaml",
            metadata={
                "field": field,
                "path": str(folder),
                "reason": "missing_manifest",
            },
        )
    try:
        data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise LifecycleError(
            "hermes_plugin_invalid",
            f"Hermes plugin manifest {str(manifest)!r} can't be read: {error}",
            metadata={
                "field": field,
                "path": str(manifest),
                "reason": "invalid_manifest",
            },
        ) from error
    if not isinstance(data, dict):
        raise LifecycleError(
            "hermes_plugin_invalid",
            f"Hermes plugin manifest {str(manifest)!r} is not a mapping",
            metadata={
                "field": field,
                "path": str(manifest),
                "reason": "invalid_manifest",
            },
        )
    name = data.get("name", folder.name)
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or name.startswith((".", "__"))
    ):
        raise LifecycleError(
            "hermes_plugin_invalid",
            f"Hermes plugin manifest {str(manifest)!r} has an invalid name {name!r}",
            metadata={"field": field, "path": str(manifest), "reason": "invalid_name"},
        )
    return name


def plan_mounted(
    settings: Mapping[str, Any], base_dir: str | Path
) -> list[MountedPlugin]:
    """Check ``plugins.external.paths`` and read each plugin's name.

    A relative entry resolves against ``base_dir``. Fails with
    ``hermes_plugin_not_found`` for a missing folder or an ``external`` entry
    no folder provides, ``hermes_plugin_invalid`` for a folder without a
    readable manifest, and ``hermes_plugin_duplicate`` when two folders share
    a name.
    """

    mounted: list[MountedPlugin] = []
    names: dict[str, MountedPlugin] = {}
    for index, entry in enumerate(external_paths(settings)):
        field = f"{PLUGINS_FIELD}.external.paths[{index}]"
        resolved = common_plugins.resolve_plugin_path(
            entry, base_dir, harness=HARNESS, label=LABEL, field=field
        )
        path = Path(entry).expanduser()
        if not path.is_absolute():
            path = Path(base_dir) / path
        if not resolved.is_dir():
            raise LifecycleError(
                "hermes_plugin_invalid",
                f"Hermes plugin path {str(path)!r} is not a folder",
                metadata={"field": field, "path": str(path), "reason": "not_a_folder"},
            )
        plugin = MountedPlugin(index, _manifest_name(path, field), resolved)
        if (earlier := names.get(plugin.name)) is not None:
            raise LifecycleError(
                "hermes_plugin_duplicate",
                f"Hermes plugin {plugin.name!r} is in {earlier.field} and {field}",
                metadata={
                    "field": field,
                    "plugin": plugin.name,
                    "paths": [str(earlier.path), str(plugin.path)],
                },
            )
        names[plugin.name] = plugin
        mounted.append(plugin)
    for plugin in plugin_entries(settings):
        if plugin.section == "external" and plugin.plugin not in names:
            raise LifecycleError(
                common_plugins.not_found_code(HARNESS),
                f"Hermes plugin {plugin.plugin!r} is not in any "
                "plugins.external.paths folder",
                metadata={"field": plugin.field, "plugin": plugin.plugin},
            )
    return mounted


def stage_mounted(
    mounted: Iterable[MountedPlugin],
    hermes_home: Path,
    links: list[Path] | None = None,
) -> list[Path]:
    """Link each mounted plugin into ``$HERMES_HOME/plugins/<name>``.

    Replaces a symlink of that name, which an earlier start left. A real
    folder of that name is a plugin the agent installed, so start fails with
    ``hermes_plugin_duplicate`` rather than shadow it. Returns the links made,
    for :func:`remove_mounted`, appending each to ``links`` as it's made when
    given; links made before a failure are removed.
    """

    plugins_dir = hermes_home / "plugins"
    links = [] if links is None else links
    try:
        for plugin in mounted:
            plugins_dir.mkdir(parents=True, exist_ok=True)
            link = plugins_dir / plugin.name
            if link.is_symlink():
                link.unlink()
            elif link.exists():
                raise LifecycleError(
                    "hermes_plugin_duplicate",
                    f"Hermes plugin {plugin.name!r} from {plugin.field} has the "
                    "name of a plugin installed in the Hermes home",
                    metadata={
                        "field": plugin.field,
                        "plugin": plugin.name,
                        "paths": [str(link), str(plugin.path)],
                    },
                )
            link.symlink_to(plugin.path, target_is_directory=True)
            links.append(link)
    except BaseException:
        remove_mounted(links)
        raise
    return links


def remove_mounted(links: Iterable[Path]) -> None:
    """Remove the links :func:`stage_mounted` made; the plugin folders stay."""

    for link in links:
        try:
            if link.is_symlink():
                link.unlink()
        except OSError:
            LOGGER.warning(
                "Hermes plugin link %s could not be removed", link, exc_info=True
            )
