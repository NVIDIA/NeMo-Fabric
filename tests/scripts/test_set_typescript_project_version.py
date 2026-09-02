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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


@pytest.fixture(name="typescript_project")
def typescript_project_fixture(tmp_path: Path) -> Path:
    _write_json(
        tmp_path / "adapter-contract/typescript/package.json",
        {
            "name": "nemo-fabric-adapter-contract",
            "version": "0.2.0",
            "devDependencies": {"typescript": "5.9.3"},
        },
    )
    _write_json(
        tmp_path / "adapter-contract/typescript/package-lock.json",
        {
            "name": "nemo-fabric-adapter-contract",
            "version": "0.2.0",
            "packages": {
                "": {
                    "name": "nemo-fabric-adapter-contract",
                    "version": "0.2.0",
                },
                "node_modules/typescript": {"version": "5.9.3"},
            },
        },
    )
    _write_json(
        tmp_path / "adapters/typescript/package.json",
        {
            "name": "nemo-fabric-typescript-adapters",
            "version": "0.2.0",
            "private": True,
        },
    )
    _write_json(
        tmp_path / "adapters/typescript/common/package.json",
        {
            "name": "nemo-fabric-adapters-common",
            "version": "0.2.0",
            "dependencies": {"nemo-fabric-adapter-contract": "0.2.0"},
        },
    )
    _write_json(
        tmp_path / "adapters/typescript/pi/package.json",
        {
            "name": "nemo-fabric-adapters-pi",
            "version": "0.2.0",
            "dependencies": {
                "nemo-fabric-adapter-contract": "0.2.0",
                "nemo-fabric-adapters-common": "0.2.0",
            },
        },
    )
    _write_json(
        tmp_path / "adapters/typescript/package-lock.json",
        {
            "name": "nemo-fabric-typescript-adapters",
            "version": "0.2.0",
            "packages": {
                "": {
                    "name": "nemo-fabric-typescript-adapters",
                    "version": "0.2.0",
                },
                "../../adapter-contract/typescript": {
                    "name": "nemo-fabric-adapter-contract",
                    "version": "0.2.0",
                },
                "common": {
                    "name": "nemo-fabric-adapters-common",
                    "version": "0.2.0",
                    "dependencies": {"nemo-fabric-adapter-contract": "0.2.0"},
                },
                "pi": {
                    "name": "nemo-fabric-adapters-pi",
                    "version": "0.2.0",
                    "dependencies": {
                        "nemo-fabric-adapter-contract": "0.2.0",
                        "nemo-fabric-adapters-common": "0.2.0",
                    },
                },
                "node_modules/ajv": {"version": "8.20.0"},
            },
        },
    )
    return tmp_path


@pytest.mark.parametrize(
    "version",
    [
        "0.3.0-rc.2",
        "0.3.0-dev.1",
        "0.3.0-x.7.z.92",
        "0.3.0+nightly.20260810",
    ],
)
def test_updates_the_complete_typescript_package_graph(
    typescript_project: Path, version: str
):
    set_typescript_project_version.set_typescript_project_version(
        typescript_project, version
    )

    contract = json.loads(
        (typescript_project / "adapter-contract/typescript/package.json").read_text()
    )
    contract_lock = json.loads(
        (typescript_project / "adapter-contract/typescript/package-lock.json").read_text()
    )
    adapters = json.loads(
        (typescript_project / "adapters/typescript/package.json").read_text()
    )
    adapters_lock = json.loads(
        (typescript_project / "adapters/typescript/package-lock.json").read_text()
    )
    common = json.loads(
        (typescript_project / "adapters/typescript/common/package.json").read_text()
    )
    pi = json.loads(
        (typescript_project / "adapters/typescript/pi/package.json").read_text()
    )

    assert contract["version"] == version
    assert contract_lock["version"] == version
    assert contract_lock["packages"][""]["version"] == version
    assert adapters["version"] == version
    assert adapters_lock["version"] == version
    for workspace in ("", "../../adapter-contract/typescript", "common", "pi"):
        assert adapters_lock["packages"][workspace]["version"] == version
    assert common["version"] == version
    assert common["dependencies"]["nemo-fabric-adapter-contract"] == version
    assert pi["version"] == version
    assert pi["dependencies"]["nemo-fabric-adapter-contract"] == version
    assert pi["dependencies"]["nemo-fabric-adapters-common"] == version
    assert adapters_lock["packages"]["common"]["dependencies"][
        "nemo-fabric-adapter-contract"
    ] == version
    assert adapters_lock["packages"]["pi"]["dependencies"] == {
        "nemo-fabric-adapter-contract": version,
        "nemo-fabric-adapters-common": version,
    }
    assert contract_lock["packages"]["node_modules/typescript"]["version"] == "5.9.3"
    assert adapters_lock["packages"]["node_modules/ajv"]["version"] == "8.20.0"


@pytest.mark.parametrize(
    "version",
    ["v0.3.0", "0.3", "01.3.0", "0.03.0", "0.3.00", "0.3.0-dev.01"],
)
def test_rejects_unsupported_versions(typescript_project: Path, version: str):
    with pytest.raises(SystemExit, match="Unsupported TypeScript package version"):
        set_typescript_project_version.set_typescript_project_version(
            typescript_project, version
        )


def test_rejects_package_metadata_drift(typescript_project: Path):
    lock_path = typescript_project / "adapters/typescript/package-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["packages"]["common"]["name"] = "wrong-package"
    _write_json(lock_path, lock)

    with pytest.raises(
        SystemExit, match="Expected synchronized nemo-fabric-adapters-common"
    ):
        set_typescript_project_version.set_typescript_project_version(
            typescript_project, "0.3.0"
        )


def test_rejects_missing_internal_dependency(typescript_project: Path):
    pi_path = typescript_project / "adapters/typescript/pi/package.json"
    pi = json.loads(pi_path.read_text(encoding="utf-8"))
    del pi["dependencies"]["nemo-fabric-adapters-common"]
    _write_json(pi_path, pi)

    with pytest.raises(SystemExit, match="Expected nemo-fabric-adapters-common"):
        set_typescript_project_version.set_typescript_project_version(
            typescript_project, "0.3.0"
        )
