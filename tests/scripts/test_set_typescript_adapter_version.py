# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "ci"
sys.path.insert(0, str(SCRIPTS))

import set_typescript_adapter_version  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize(
    ("package_name", "workspace", "version"),
    [
        ("nemo-fabric-adapters-common", "common", "0.4.0"),
        ("nemo-fabric-adapters-pi", "pi", "0.2.0-rc.1"),
    ],
)
def test_updates_only_the_selected_adapter_package(
    tmp_path: Path,
    package_name: str,
    workspace: str,
    version: str,
):
    _write_json(
        tmp_path / f"adapters/typescript/{workspace}/package.json",
        {
            "name": package_name,
            "version": "0.1.0",
            "dependencies": (
                {"nemo-fabric-adapters-common": "0.1.0"}
                if workspace == "pi"
                else {}
            ),
        },
    )
    other_workspace = "pi" if workspace == "common" else "common"
    other_name = (
        "nemo-fabric-adapters-pi"
        if other_workspace == "pi"
        else "nemo-fabric-adapters-common"
    )
    _write_json(
        tmp_path / f"adapters/typescript/{other_workspace}/package.json",
        {
            "name": other_name,
            "version": "0.1.0",
            "dependencies": (
                {"nemo-fabric-adapters-common": "0.1.0"}
                if other_workspace == "pi"
                else {}
            ),
        },
    )
    _write_json(
        tmp_path / "adapters/typescript/package-lock.json",
        {
            "packages": {
                workspace: {
                    "name": package_name,
                    "version": "0.1.0",
                    "dependencies": (
                        {"nemo-fabric-adapters-common": "0.1.0"}
                        if workspace == "pi"
                        else {}
                    ),
                },
                other_workspace: {
                    "name": other_name,
                    "version": "0.1.0",
                    "dependencies": (
                        {"nemo-fabric-adapters-common": "0.1.0"}
                        if other_workspace == "pi"
                        else {}
                    ),
                },
            }
        },
    )

    set_typescript_adapter_version.set_typescript_adapter_version(
        tmp_path, package_name, version
    )

    package = json.loads(
        (tmp_path / f"adapters/typescript/{workspace}/package.json").read_text()
    )
    lock = json.loads(
        (tmp_path / "adapters/typescript/package-lock.json").read_text()
    )
    assert package["version"] == version
    assert lock["packages"][workspace]["version"] == version
    assert lock["packages"][other_workspace]["version"] == "0.1.0"
    pi_package = json.loads(
        (tmp_path / "adapters/typescript/pi/package.json").read_text()
    )
    expected_common = version if package_name == "nemo-fabric-adapters-common" else "0.1.0"
    assert pi_package["dependencies"]["nemo-fabric-adapters-common"] == expected_common
    assert (
        lock["packages"]["pi"]["dependencies"]["nemo-fabric-adapters-common"]
        == expected_common
    )


def test_rejects_unknown_adapter_package(tmp_path: Path):
    with pytest.raises(SystemExit, match="Unsupported TypeScript adapter package"):
        set_typescript_adapter_version.set_typescript_adapter_version(
            tmp_path, "unknown", "0.1.0"
        )


def test_rejects_unsupported_version(tmp_path: Path):
    with pytest.raises(SystemExit, match="Unsupported TypeScript package version"):
        set_typescript_adapter_version.set_typescript_adapter_version(
            tmp_path, "nemo-fabric-adapters-common", "not-a-version"
        )


def test_rejects_desynchronized_lock_entry(tmp_path: Path):
    _write_json(
        tmp_path / "adapters/typescript/common/package.json",
        {"name": "nemo-fabric-adapters-common", "version": "0.1.0"},
    )
    _write_json(
        tmp_path / "adapters/typescript/package-lock.json",
        {"packages": {"common": {"name": "other", "version": "0.1.0"}}},
    )

    with pytest.raises(SystemExit, match="Expected synchronized"):
        set_typescript_adapter_version.set_typescript_adapter_version(
            tmp_path, "nemo-fabric-adapters-common", "0.4.0"
        )
