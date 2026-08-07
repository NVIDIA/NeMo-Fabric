# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readiness checks for artifacts written asynchronously by NeMo Relay."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from nemo_fabric_adapters.common import utils as common_utils

ATIF_FINALIZATION_TIMEOUT_SECONDS = 5.0
ATIF_POLL_INTERVAL_SECONDS = 0.05
AtifFileFingerprint = tuple[int, int, int, int]
AtifSnapshot = dict[Path, AtifFileFingerprint]


def expects_local_atif(plugin_config: dict[str, Any]) -> bool:
    """Return whether Relay is configured to write ATIF to the local runtime.

    Relay treats a non-empty storage list as remote-only, so there is no local
    artifact for an adapter to await in that configuration.
    """

    for component in plugin_config.get("components", []):
        if (
            not isinstance(component, dict)
            or component.get("kind") != "observability"
            or component.get("enabled", True) is False
        ):
            continue
        config = component.get("config")
        if not isinstance(config, dict):
            continue
        atif = config.get("atif")
        if isinstance(atif, dict) and atif.get("enabled") and not atif.get("storage"):
            return True
    return False


def _atif_fingerprint(path: Path) -> AtifFileFingerprint | None:
    try:
        status = path.stat()
    except (OSError, RuntimeError, ValueError):
        return None
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)


def snapshot_atif_files(plugin_config: dict[str, Any]) -> AtifSnapshot:
    """Capture ATIF file metadata before an adapter invocation.

    Relay requires ``{session_id}`` in the ATIF filename template and creates a
    new session scope for each turn, so a new path is the normal case. Metadata
    fingerprints also detect a writer that rewrites an existing path without
    reading or hashing artifacts from prior turns.
    """

    snapshot: AtifSnapshot = {}
    for artifact in common_utils.collect_relay_artifacts(plugin_config):
        if artifact.get("kind") != "atif":
            continue
        path = Path(artifact["path"])
        fingerprint = _atif_fingerprint(path)
        if fingerprint is not None:
            snapshot[path] = fingerprint
    return snapshot


def _finalized_atif_path(
    plugin_config: dict[str, Any], before: AtifSnapshot
) -> Path | None:
    """Find a new or changed ATIF path containing a complete JSON object."""

    current = snapshot_atif_files(plugin_config)
    for path in sorted(current):
        if before.get(path) == current[path]:
            continue
        try:
            # Relay writes directly to the final path, so existence alone does
            # not prove that the JSON payload has been written completely.
            document = json.loads(path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(document, dict):
            return path
    return None


async def wait_for_finalized_atif(
    plugin_config: dict[str, Any],
    before: AtifSnapshot,
    *,
    timeout_seconds: float = ATIF_FINALIZATION_TIMEOUT_SECONDS,
    poll_interval_seconds: float = ATIF_POLL_INTERVAL_SECONDS,
) -> Path | None:
    """Wait for one new or changed, complete ATIF file until a deadline."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    while True:
        if path := _finalized_atif_path(plugin_config, before):
            return path
        remaining = deadline - loop.time()
        if remaining <= 0:
            return None
        await asyncio.sleep(min(poll_interval_seconds, remaining))
