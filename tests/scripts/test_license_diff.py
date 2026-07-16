# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest

LICENSING_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "licensing"
sys.path.insert(0, str(LICENSING_SCRIPTS))

import license_diff  # noqa: E402
from attributions_lockfile_md import LicenseInventoryEntry  # noqa: E402


def _entry(package: str, version: str, license_name: str) -> LicenseInventoryEntry:
    return LicenseInventoryEntry(package=package, version=version, license=license_name)


def test_compare_inventories_classifies_dependency_changes() -> None:
    unchanged = _entry("unchanged", "1.0.0", "MIT")
    base = {
        "rust": [
            _entry("changed", "1.0.0", "BSD-3-Clause"),
            _entry("removed", "1.0.0", "Apache-2.0"),
            unchanged,
        ]
    }
    current = {
        "rust": [
            _entry("added", "1.0.0", "ISC"),
            _entry("changed", "2.0.0", "Apache-2.0"),
            unchanged,
        ]
    }

    diff = license_diff.compare_inventories(base, current)["rust"]

    assert diff["added"] == [_entry("added", "1.0.0", "ISC")]
    assert diff["removed"] == [_entry("removed", "1.0.0", "Apache-2.0")]
    assert diff["updated_changed"] == [
        {
            "package": "changed",
            "before": [_entry("changed", "1.0.0", "BSD-3-Clause")],
            "after": [_entry("changed", "2.0.0", "Apache-2.0")],
            "removed": [_entry("changed", "1.0.0", "BSD-3-Clause")],
            "added": [_entry("changed", "2.0.0", "Apache-2.0")],
        }
    ]


def test_parse_languages_rejects_unsupported_ecosystems() -> None:
    with pytest.raises(ValueError, match="Unsupported language.*node"):
        license_diff._parse_languages(["node"])
