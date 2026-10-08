# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect bundled adapter metadata without loading executable adapters."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

__all__ = ["get_adapter_descriptor", "get_target_descriptor"]


def _descriptor(kind: str, record_id: str) -> dict[str, Any]:
    catalog = json.loads(
        files(__package__).joinpath("catalog.json").read_text(encoding="utf-8")
    )
    if catalog["format_version"] != 1:
        raise ValueError(
            f"Unsupported adapter catalog format: {catalog['format_version']!r}"
        )
    # Reloading gives each caller an independent object and avoids mutable global state.
    return catalog[kind][record_id]["descriptor"]


def get_adapter_descriptor(adapter_id: str) -> dict[str, Any]:
    """Return a bundled adapter's unmodified descriptor; raise KeyError for an unknown ID.

    This is a release snapshot, not a capability check against an installed harness.
    Lookup neither imports adapter code nor registers an execution runner.
    """
    return _descriptor("adapters", adapter_id)


def get_target_descriptor(target_id: str) -> dict[str, Any]:
    """Return a bundled target's unmodified descriptor; raise KeyError for an unknown ID."""
    return _descriptor("targets", target_id)
