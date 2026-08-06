# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readiness checks for artifacts written asynchronously by NeMo Relay."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Collection
from pathlib import Path
from typing import Any

from nemo_fabric_adapters.common import utils as common_utils

ATIF_FINALIZATION_TIMEOUT_SECONDS = 5.0
ATIF_POLL_INTERVAL_SECONDS = 0.05


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


def snapshot_atif_paths(plugin_config: dict[str, Any]) -> frozenset[Path]:
    """Capture ATIF paths that existed before an adapter invocation.

    Relay requires ``{session_id}`` in the ATIF filename template and creates a
    new session scope for each turn. Existing paths therefore cannot satisfy a
    later invocation, even if another process modifies them.
    """

    return frozenset(
        Path(artifact["path"])
        for artifact in common_utils.collect_relay_artifacts(plugin_config)
        if artifact.get("kind") == "atif"
    )


def _finalized_atif_path(
    plugin_config: dict[str, Any], before: Collection[Path]
) -> Path | None:
    """Find a newly created ATIF path containing a complete JSON object."""

    current = snapshot_atif_paths(plugin_config)
    for path in sorted(current.difference(before)):
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
    before: Collection[Path],
    *,
    timeout_seconds: float = ATIF_FINALIZATION_TIMEOUT_SECONDS,
    poll_interval_seconds: float = ATIF_POLL_INTERVAL_SECONDS,
) -> Path | None:
    """Wait for one new, complete ATIF file until a monotonic deadline."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    while True:
        if path := _finalized_atif_path(plugin_config, before):
            return path
        remaining = deadline - loop.time()
        if remaining <= 0:
            return None
        await asyncio.sleep(min(poll_interval_seconds, remaining))
