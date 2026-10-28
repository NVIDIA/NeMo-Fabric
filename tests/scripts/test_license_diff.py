# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
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


@pytest.mark.parametrize("root_version", [None, "12.3.0", "12.2.0"])
def test_python_image_package_is_attributed_and_inventoried(
    tmp_path: Path, root_version
):
    collector = license_diff.attributions_lockfile_md
    manifest = tmp_path / "scripts" / "licensing" / "python-image-packages.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('[[package]]\nname = "pillow"\nversion = "12.3.0"\n')
    root_packages = (
        {"pillow": {"name": "Pillow", "version": root_version}} if root_version else {}
    )

    def render(pkg):
        return collector.RenderedPythonPackage(
            name=pkg["name"],
            version=pkg["version"],
            license_name="MIT-CMU",
            license_texts=[("LICENSE", "notice")],
        )

    with (
        patch.object(collector, "ROOT", tmp_path),
        patch.object(
            collector, "_lockfile_registry_packages", return_value=root_packages
        ),
        patch.object(collector, "_lockfile_only_python_package", side_effect=render),
    ):
        collector.cmd_python()
        inventory = collector._python_license_inventory()

    text = (tmp_path / "ATTRIBUTIONS-Python.md").read_text().lower()
    assert text.count("## pillow (12.3.0)") == 1
    assert text.count("license: `mit-cmu`") == (2 if root_version == "12.2.0" else 1)
    assert sum(row["version"] == "12.3.0" for row in inventory) == 1
    assert all(row["license"] == "MIT-CMU" for row in inventory)


def test_python_inventory_supports_refs_without_image_manifest(tmp_path: Path):
    collector = license_diff.attributions_lockfile_md
    with (
        patch.object(collector, "ROOT", tmp_path),
        patch.object(collector, "_lockfile_registry_packages", return_value={}),
    ):
        assert collector._python_license_inventory() == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain license text", "```"),
        ("an embedded ``` block", "````"),
        ("an embedded ````` block", "``````"),
    ],
)
def test_markdown_code_fence_is_longer_than_embedded_fences(text, expected):
    assert license_diff.attributions_lockfile_md._markdown_code_fence(text) == expected


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
    with pytest.raises(ValueError, match=r"Unsupported language.*javascript"):
        license_diff._parse_languages(["javascript"])


def test_configure_root_points_node_collector_at_typescript_package(
    tmp_path: Path,
):
    with (
        patch.object(license_diff.attributions_lockfile_md, "ROOT"),
        patch.object(license_diff.attributions_lockfile_md, "NODE"),
    ):
        license_diff._configure_root(tmp_path)

        assert license_diff.attributions_lockfile_md.ROOT == tmp_path.resolve()
        assert license_diff.attributions_lockfile_md.NODE == (
            tmp_path.resolve() / "adapter-contract" / "typescript"
        )


def test_configure_root_supports_comparison_with_previous_layout(tmp_path: Path):
    previous_package = tmp_path / "typescript" / "adapter-contract"
    previous_package.mkdir(parents=True)

    with (
        patch.object(license_diff.attributions_lockfile_md, "ROOT"),
        patch.object(license_diff.attributions_lockfile_md, "NODE"),
    ):
        license_diff._configure_root(tmp_path)

        assert license_diff.attributions_lockfile_md.NODE == previous_package.resolve()


def test_node_inventory_includes_locked_dev_dependencies(tmp_path: Path):
    package_dir = tmp_path / "adapter-contract" / "typescript"
    package_dir.mkdir(parents=True)
    lock = {
        "name": "local-package",
        "version": "0.2.0",
        "packages": {
            "": {"name": "local-package", "version": "0.2.0"},
            "node_modules/build-tool": {
                "version": "1.2.3",
                "license": "MIT",
                "dev": True,
            },
            "node_modules/ignored": {
                "version": "4.5.6",
                "license": "ISC",
                "extraneous": True,
            },
        },
    }
    (package_dir / "package-lock.json").write_text(json.dumps(lock))

    with patch.object(
        license_diff.attributions_lockfile_md,
        "NODE",
        package_dir,
    ):
        inventory = license_diff.attributions_lockfile_md._node_license_inventory()

    assert inventory == [_entry("build-tool", "1.2.3", "MIT")]


def test_node_inventory_is_empty_before_package_exists(tmp_path: Path):
    with patch.object(
        license_diff.attributions_lockfile_md,
        "NODE",
        tmp_path / "missing-package",
    ):
        inventory = license_diff.attributions_lockfile_md._node_license_inventory()

    assert inventory == []


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
