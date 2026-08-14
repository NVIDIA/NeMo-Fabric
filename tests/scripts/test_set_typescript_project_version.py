# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


CI_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "ci"
sys.path.insert(0, str(CI_SCRIPTS))

import set_typescript_project_version  # noqa: E402


@pytest.fixture(name="package_files")
def package_files_fixture(tmp_path: Path) -> tuple[Path, Path]:
    package_directory = tmp_path / "adapter-contract" / "typescript"
    package_directory.mkdir(parents=True)
    package_path = package_directory / "package.json"
    lock_path = package_directory / "package-lock.json"
    package_path.write_text(
        json.dumps(
            {
                "name": "nemo-fabric-adapter-contract",
                "version": "0.2.0",
                "devDependencies": {"typescript": "5.9.3"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    lock_path.write_text(
        json.dumps(
            {
                "name": "nemo-fabric-adapter-contract",
                "version": "0.2.0",
                "lockfileVersion": 3,
                "requires": True,
                "packages": {
                    "": {
                        "name": "nemo-fabric-adapter-contract",
                        "version": "0.2.0",
                        "devDependencies": {"typescript": "5.9.3"},
                    },
                    "node_modules/typescript": {
                        "version": "5.9.3",
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return package_path, lock_path


@pytest.mark.parametrize(
    "version",
    [
        "0.3.0-rc.2",
        "0.3.0-dev.1",
        "0.3.0-x.7.z.92",
        "0.3.0+nightly.20260810",
    ],
)
def test_set_typescript_project_version_updates_manifest_and_lockfile(
    package_files: tuple[Path, Path],
    version: str,
):
    package_path, lock_path = package_files
    root = package_path.parents[2]

    set_typescript_project_version.set_typescript_project_version(root, version)

    package = json.loads(package_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert package["version"] == version
    assert lock["version"] == version
    assert lock["packages"][""]["version"] == version
    assert lock["packages"]["node_modules/typescript"]["version"] == "5.9.3"


@pytest.mark.parametrize(
    "version",
    [
        "v0.3.0",
        "0.3",
        "01.3.0",
        "0.03.0",
        "0.3.00",
        "0.3.0-dev.01",
        "0.3.0-",
        "0.3.0+",
    ],
)
def test_set_typescript_project_version_rejects_unsupported_versions(
    package_files: tuple[Path, Path],
    version: str,
):
    package_path, _ = package_files
    root = package_path.parents[2]

    with pytest.raises(SystemExit, match="Unsupported TypeScript package version"):
        set_typescript_project_version.set_typescript_project_version(root, version)


def test_set_typescript_project_version_rejects_lockfile_name_drift(
    package_files: tuple[Path, Path],
):
    package_path, lock_path = package_files
    root = package_path.parents[2]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["packages"][""]["name"] = "wrong-package"
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(SystemExit, match=r"Package names .* are not synchronized"):
        set_typescript_project_version.set_typescript_project_version(root, "0.3.0")
