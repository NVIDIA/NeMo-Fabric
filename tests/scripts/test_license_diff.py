# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import call, patch

import pytest

LICENSING_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "licensing"
sys.path.insert(0, str(LICENSING_SCRIPTS))

import license_diff  # noqa: E402
from attributions_lockfile_md import LicenseInventoryEntry  # noqa: E402


def _entry(package: str, version: str, license_name: str) -> LicenseInventoryEntry:
    return LicenseInventoryEntry(package=package, version=version, license=license_name)


def test_compare_inventories_classifies_dependency_changes():
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


def test_parse_languages_rejects_unsupported_ecosystems():
    with pytest.raises(ValueError, match=r"Unsupported language.*node"):
        license_diff._parse_languages(["node"])


def test_worktree_inventory_cleans_up_when_checkout_fails(tmp_path: Path):
    root = tmp_path / "root"
    worktree = tmp_path / "repo"
    checkout_error = subprocess.CalledProcessError(1, ["git", "worktree", "add"])
    cleanup_result = subprocess.CompletedProcess(
        args=["git", "worktree", "remove"], returncode=0
    )

    with (
        patch.object(license_diff.tempfile, "mkdtemp", return_value=str(tmp_path)),
        patch.object(
            license_diff.subprocess,
            "run",
            side_effect=[checkout_error, cleanup_result],
        ) as mock_run,
        patch.object(license_diff.shutil, "rmtree") as mock_rmtree,
    ):
        with pytest.raises(subprocess.CalledProcessError) as error:
            license_diff._worktree_inventory(root, "base-ref", ["rust"])

    assert error.value is checkout_error
    assert mock_run.call_args_list[-1] == call(
        ["git", "-C", str(root), "worktree", "remove", "--force", str(worktree)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    mock_rmtree.assert_called_once_with(tmp_path, ignore_errors=True)


def test_worktree_inventory_cleans_up_when_generation_fails(tmp_path: Path):
    root = tmp_path / "root"
    worktree = tmp_path / "repo"
    command_result = subprocess.CompletedProcess(args=["git", "worktree"], returncode=0)
    inventory_error = RuntimeError("inventory failed")

    with (
        patch.object(license_diff.tempfile, "mkdtemp", return_value=str(tmp_path)),
        patch.object(
            license_diff.subprocess,
            "run",
            side_effect=[command_result, command_result],
        ) as mock_run,
        patch.object(
            license_diff,
            "generate_inventory",
            side_effect=inventory_error,
        ) as mock_generate_inventory,
        patch.object(license_diff.shutil, "rmtree") as mock_rmtree,
    ):
        with pytest.raises(RuntimeError, match="inventory failed") as error:
            license_diff._worktree_inventory(root, "base-ref", ["python"])

    assert error.value is inventory_error
    mock_generate_inventory.assert_called_once_with(worktree, ["python"], label="base")
    assert mock_run.call_args_list[-1] == call(
        ["git", "-C", str(root), "worktree", "remove", "--force", str(worktree)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    mock_rmtree.assert_called_once_with(tmp_path, ignore_errors=True)
