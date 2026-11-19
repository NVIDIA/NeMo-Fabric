# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin start checks shared by adapters whose harness has native plugins.

Each adapter passes its harness's own plugin settings through and checks, at
start, that every plugin NeMo Fabric turns on is present and loaded. The
errors share one shape across harnesses:

``<harness>_plugin_not_found``
    A configured plugin folder doesn't exist, or the harness has no plugin
    with an enabled identifier. Metadata: ``field``, and ``plugin`` or
    ``path``.
``<harness>_plugin_unavailable``
    The harness has the plugin but didn't load it. Metadata: ``field``,
    ``plugin``, and ``reason``, the harness's own explanation.

``field`` names the setting that asked for the plugin, such as
``harness.settings.plugins.bundled.entries.<id>``.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from pathlib import Path

from nemo_fabric_adapters.common.lifecycle import LifecycleError

# A lookup answers for one plugin identifier: ``None`` when the harness has no
# such plugin, ``""`` when it loaded, or the reason it didn't.
PluginLookup = Callable[[str], str | None]

MAX_REASON = 2000


def not_found_code(harness: str) -> str:
    return f"{harness}_plugin_not_found"


def unavailable_code(harness: str) -> str:
    return f"{harness}_plugin_unavailable"


def resolve_plugin_path(
    entry: str | Path, base_dir: str | Path, *, harness: str, label: str, field: str
) -> Path:
    """Resolve a configured plugin path against ``base_dir``; it must exist.

    Fails with ``<harness>_plugin_not_found`` and metadata ``field`` and
    ``path``. The returned path is absolute with symlinks resolved.
    """

    path = Path(entry)
    try:
        path = path.expanduser()
        if not path.is_absolute():
            path = Path(base_dir) / path
        return path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise LifecycleError(
            not_found_code(harness),
            f"{label} plugin path {str(path)!r} does not exist",
            metadata={"field": field, "path": str(path)},
        ) from None


def check_requested(
    requested: Iterable[tuple[str, str]],
    lookup: PluginLookup,
    *,
    harness: str,
    label: str,
) -> None:
    """Fail start unless the harness loaded every requested plugin.

    ``requested`` holds ``(field, plugin)`` pairs, in the order the settings
    list them. ``lookup`` reads the harness's own plugin listing.
    """

    for field, plugin in requested:
        reason = lookup(plugin)
        if reason is None:
            raise LifecycleError(
                not_found_code(harness),
                f"{label} plugin {plugin!r} was not found",
                metadata={"field": field, "plugin": plugin},
            )
        if reason:
            reason = reason[-MAX_REASON:]
            raise LifecycleError(
                unavailable_code(harness),
                f"{label} plugin {plugin!r} did not load: {reason}",
                metadata={"field": field, "plugin": plugin, "reason": reason},
            )
