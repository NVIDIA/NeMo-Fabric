# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retained OpenClaw plugin settings, merged with the agent's own at each start."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common import local_interfaces


STATE_FOLDER = "fabric"
CONFIG_FILE = "openclaw.json"
PLUGINS_FILE = "plugins.json"
OWNED_FILE = "plugins.owned.json"
INCLUDE = {"$include": f"./{PLUGINS_FILE}"}
SHARED_CONFIG_PLUGIN = "fabric-shared-config"
# List fields merged item by item, so the agent's own items are kept.
LIST_FIELDS: tuple[tuple[str, ...], ...] = (("allow",), ("deny",), ("load", "paths"))
# Copies OpenClaw writes next to a writable configuration.
GENERATED_COPY = re.compile(r"\.(bak(\.\d+)?|last-good|clobbered\..+)")

Leaf = tuple[str, ...]


def _leaves(value: dict[str, Any], prefix: Leaf = ()) -> Iterator[tuple[Leaf, Any]]:
    """Yield every leaf path and value, leaving out the list fields."""
    for key, item in value.items():
        path = (*prefix, key)
        if path in LIST_FIELDS:
            continue
        if isinstance(item, dict) and item:
            yield from _leaves(item, path)
        else:
            yield path, item


_MISSING = object()


def _get(value: dict[str, Any], path: Leaf, default: Any = None) -> Any:
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _same(left: Any, right: Any) -> bool:
    """JSON equality: ``0``, ``False`` and ``0.0`` stay distinct."""
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _set(value: dict[str, Any], path: Leaf, item: Any) -> None:
    for key in path[:-1]:
        if not isinstance(value.get(key), dict):
            value[key] = {}
        value = value[key]
    if item == {} and isinstance(value.get(path[-1]), dict):
        return  # an empty object adds no setting of its own
    value[path[-1]] = copy.deepcopy(item)


def _delete(value: dict[str, Any], path: Leaf) -> None:
    """Delete one leaf, then any object the deletion left empty."""
    parents = []
    for key in path[:-1]:
        child = value.get(key)
        if not isinstance(child, dict):
            return
        parents.append((value, key))
        value = child
    if path[-1] not in value:
        return
    del value[path[-1]]
    while parents and not value:
        parent, key = parents.pop()
        del parent[key]
        value = parent


def _owned_items(owned: dict[str, Any], path: Leaf) -> list[str]:
    items = owned.get("items", {}).get(".".join(path), [])
    return [item for item in items if isinstance(item, str)]


def merge_plugins(
    agent: dict[str, Any], fabric: dict[str, Any], owned: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge NeMo Fabric's plugin section into the agent's.

    NeMo Fabric's leaves win, leaves it set last time and no longer sets are
    deleted, and list fields keep the agent's items. Returns the merged
    section and the new ownership record.
    """
    merged = copy.deepcopy(agent)
    fabric_leaves = dict(_leaves(fabric))
    previous = {
        tuple(path)
        for path in owned.get("leaves", [])
        if isinstance(path, list) and path and all(isinstance(k, str) for k in path)
    }
    for path in sorted(previous - fabric_leaves.keys()):
        _delete(merged, path)
    applied = []
    for path, value in fabric_leaves.items():
        if value == {} and isinstance(_get(merged, path), dict):
            continue  # an empty object adds no setting of its own
        current = _get(merged, path, _MISSING)
        if path not in previous and current is not _MISSING and _same(current, value):
            continue  # the agent's own value; NeMo Fabric doesn't claim it
        _set(merged, path, value)
        applied.append(path)

    owned_items: dict[str, list[str]] = {}
    for path in LIST_FIELDS:
        fabric_items = list(_get(fabric, path) or [])
        if path == ("allow",) and (fabric_items or _get(merged, path)):
            # An existing allowlist must name every plugin NeMo Fabric enables.
            for plugin_id, entry in fabric.get("entries", {}).items():
                enabled = isinstance(entry, dict) and entry.get("enabled") is True
                if enabled and plugin_id not in fabric_items:
                    fabric_items.append(plugin_id)
        previous_items = _owned_items(owned, path)
        stale = set(previous_items) - set(fabric_items)
        before = _get(agent, path)
        before = before if isinstance(before, list) else []
        current = _get(merged, path)
        kept = [
            item
            for item in (current if isinstance(current, list) else [])
            if item not in stale
        ]
        combined = list(dict.fromkeys([*kept, *fabric_items]))
        if combined:
            _set(merged, path, combined)
        else:
            _delete(merged, path)
        # Only items NeMo Fabric added, never ones the agent already had.
        added = [i for i in fabric_items if i not in before or i in previous_items]
        if added:
            owned_items[".".join(path)] = added
    record = {"leaves": [list(path) for path in applied], "items": owned_items}
    return merged, record


def _read_object(path: Path, *, kind: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError) as error:
        raise _state_invalid(path, kind, str(error)) from error
    try:
        value = json.loads(text)
    except ValueError as error:
        raise _state_invalid(path, kind, str(error)) from error
    if not isinstance(value, dict):
        raise _state_invalid(path, kind, "the file must hold a JSON object")
    return value


def _state_invalid(path: Path, kind: str, detail: str) -> lifecycle.LifecycleError:
    return lifecycle.LifecycleError(
        "openclaw_plugin_state_invalid",
        f"OpenClaw {kind} in the retained state folder can't be read; "
        "fix or remove the file",
        metadata={"path": str(path), "detail": detail[-2000:]},
    )


def state_folder(state_dir: Path) -> Path:
    """Return ``<state_dir>/fabric``, created owner-only."""
    folder = state_dir / STATE_FOLDER
    if folder.is_symlink() or (folder.exists() and not folder.is_dir()):
        raise lifecycle.LifecycleError(
            "session_incompatible",
            "Retained OpenClaw configuration folder is invalid",
            metadata={"path": str(folder)},
        )
    folder.mkdir(mode=0o700, exist_ok=True)
    folder.chmod(0o700)
    return folder


def _valid_record(owned: dict[str, Any]) -> bool:
    leaves = owned.get("leaves", [])
    items = owned.get("items", {})
    return (
        isinstance(leaves, list)
        and isinstance(items, dict)
        and all(isinstance(value, list) for value in items.values())
    )


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def prepare(folder: Path, fabric: dict[str, Any]) -> dict[str, Any]:
    """Merge NeMo Fabric's plugin settings into ``plugins.json`` and return them."""
    policy_path = folder / PLUGINS_FILE
    owned_path = folder / OWNED_FILE
    agent = _read_object(policy_path, kind="plugin settings")
    owned = _read_object(owned_path, kind="plugin ownership record")
    pending = owned.pop("pending", None)
    if isinstance(pending, dict):
        # A start that failed between writes: its record counts only if
        # plugins.json holds exactly what that start wrote.
        text = policy_path.read_text(encoding="utf-8") if policy_path.exists() else ""
        if pending.get("sha256") == _digest(text):
            owned = {k: v for k, v in pending.items() if k != "sha256"}
    if not _valid_record(owned):
        raise _state_invalid(
            owned_path, "plugin ownership record", "unexpected record shape"
        )
    merged, record = merge_plugins(agent, fabric, owned)
    text = json.dumps(merged, indent=2) + "\n"
    staged = {**owned, "pending": {**record, "sha256": _digest(text)}}
    for path, content in (
        (owned_path, json.dumps(staged, indent=2) + "\n"),
        (policy_path, text),
        (owned_path, json.dumps(record, indent=2) + "\n"),
    ):
        local_interfaces.write_private_file(path, content)
    return merged


def remove_config(path: Path) -> None:
    """Remove the generated ``openclaw.json`` and OpenClaw's copies of it."""
    path.unlink(missing_ok=True)
    for copy_path in path.parent.glob(f"{path.name}.*"):
        if GENERATED_COPY.fullmatch(copy_path.name[len(path.name) :]):
            copy_path.unlink(missing_ok=True)
