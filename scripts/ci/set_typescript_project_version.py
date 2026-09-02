# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


NUMERIC_IDENTIFIER = r"(?:0|[1-9]\d*)"
PRERELEASE_IDENTIFIER = rf"(?:{NUMERIC_IDENTIFIER}|\d*[A-Za-z-][0-9A-Za-z-]*)"
BUILD_IDENTIFIER = r"[0-9A-Za-z-]+"
SEMVER_PATTERN = re.compile(
    rf"{NUMERIC_IDENTIFIER}\.{NUMERIC_IDENTIFIER}\.{NUMERIC_IDENTIFIER}"
    rf"(?:-{PRERELEASE_IDENTIFIER}(?:\.{PRERELEASE_IDENTIFIER})*)?"
    rf"(?:\+{BUILD_IDENTIFIER}(?:\.{BUILD_IDENTIFIER})*)?"
)
CONTRACT_DIRECTORY = Path("adapter-contract/typescript")
ADAPTERS_DIRECTORY = Path("adapters/typescript")
CONTRACT_PACKAGE = "nemo-fabric-adapter-contract"
COMMON_PACKAGE = "nemo-fabric-adapters-common"
PI_PACKAGE = "nemo-fabric-adapters-pi"


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a JSON object in {path}")
    return value


def _write_json_object(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _require_object(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemExit(f"Expected {description}")
    return value


def _require_named_package(
    package: dict[str, Any], expected_name: str, description: str
) -> None:
    if package.get("name") != expected_name or "version" not in package:
        raise SystemExit(
            f"Expected synchronized {expected_name} package metadata in {description}"
        )


def _set_exact_dependency(
    package: dict[str, Any], dependency: str, version: str, description: str
) -> None:
    dependencies = _require_object(
        package.get("dependencies"), f"a dependencies object in {description}"
    )
    if dependency not in dependencies:
        raise SystemExit(f"Expected {dependency} dependency in {description}")
    dependencies[dependency] = version


def set_typescript_project_version(root: Path, version: str) -> None:
    if SEMVER_PATTERN.fullmatch(version) is None:
        raise SystemExit(f"Unsupported TypeScript package version: {version}")

    contract_package_path = root / CONTRACT_DIRECTORY / "package.json"
    contract_lock_path = root / CONTRACT_DIRECTORY / "package-lock.json"
    adapters_package_path = root / ADAPTERS_DIRECTORY / "package.json"
    adapters_lock_path = root / ADAPTERS_DIRECTORY / "package-lock.json"
    common_package_path = root / ADAPTERS_DIRECTORY / "common" / "package.json"
    pi_package_path = root / ADAPTERS_DIRECTORY / "pi" / "package.json"

    contract_package = _read_json_object(contract_package_path)
    contract_lock = _read_json_object(contract_lock_path)
    adapters_package = _read_json_object(adapters_package_path)
    adapters_lock = _read_json_object(adapters_lock_path)
    common_package = _read_json_object(common_package_path)
    pi_package = _read_json_object(pi_package_path)

    contract_lock_packages = _require_object(
        contract_lock.get("packages"), f"a packages object in {contract_lock_path}"
    )
    contract_lock_root = _require_object(
        contract_lock_packages.get(""), f"a root package entry in {contract_lock_path}"
    )
    adapters_lock_packages = _require_object(
        adapters_lock.get("packages"), f"a packages object in {adapters_lock_path}"
    )
    adapters_lock_root = _require_object(
        adapters_lock_packages.get(""), f"a root package entry in {adapters_lock_path}"
    )
    adapters_lock_contract = _require_object(
        adapters_lock_packages.get("../../adapter-contract/typescript"),
        f"local {CONTRACT_PACKAGE} metadata in {adapters_lock_path}",
    )
    adapters_lock_common = _require_object(
        adapters_lock_packages.get("common"),
        f"{COMMON_PACKAGE} metadata in {adapters_lock_path}",
    )
    adapters_lock_pi = _require_object(
        adapters_lock_packages.get("pi"),
        f"{PI_PACKAGE} metadata in {adapters_lock_path}",
    )

    _require_named_package(contract_package, CONTRACT_PACKAGE, str(contract_package_path))
    _require_named_package(contract_lock, CONTRACT_PACKAGE, str(contract_lock_path))
    _require_named_package(
        contract_lock_root, CONTRACT_PACKAGE, f"root entry of {contract_lock_path}"
    )
    adapters_name = adapters_package.get("name")
    if not isinstance(adapters_name, str) or not adapters_name:
        raise SystemExit(f"Expected a package name in {adapters_package_path}")
    _require_named_package(adapters_package, adapters_name, str(adapters_package_path))
    _require_named_package(adapters_lock, adapters_name, str(adapters_lock_path))
    _require_named_package(
        adapters_lock_root, adapters_name, f"root entry of {adapters_lock_path}"
    )
    _require_named_package(
        adapters_lock_contract,
        CONTRACT_PACKAGE,
        f"local contract entry of {adapters_lock_path}",
    )
    _require_named_package(common_package, COMMON_PACKAGE, str(common_package_path))
    _require_named_package(
        adapters_lock_common, COMMON_PACKAGE, f"common entry of {adapters_lock_path}"
    )
    _require_named_package(pi_package, PI_PACKAGE, str(pi_package_path))
    _require_named_package(
        adapters_lock_pi, PI_PACKAGE, f"pi entry of {adapters_lock_path}"
    )

    source_values = {
        path: json.dumps(value, sort_keys=True)
        for path, value in (
            (contract_package_path, contract_package),
            (contract_lock_path, contract_lock),
            (adapters_package_path, adapters_package),
            (adapters_lock_path, adapters_lock),
            (common_package_path, common_package),
            (pi_package_path, pi_package),
        )
    }

    for package in (
        contract_package,
        contract_lock,
        contract_lock_root,
        adapters_package,
        adapters_lock,
        adapters_lock_root,
        adapters_lock_contract,
        common_package,
        adapters_lock_common,
        pi_package,
        adapters_lock_pi,
    ):
        package["version"] = version

    for package, description in (
        (common_package, str(common_package_path)),
        (adapters_lock_common, f"common entry of {adapters_lock_path}"),
        (pi_package, str(pi_package_path)),
        (adapters_lock_pi, f"pi entry of {adapters_lock_path}"),
    ):
        _set_exact_dependency(package, CONTRACT_PACKAGE, version, description)
    for package, description in (
        (pi_package, str(pi_package_path)),
        (adapters_lock_pi, f"pi entry of {adapters_lock_path}"),
    ):
        _set_exact_dependency(package, COMMON_PACKAGE, version, description)

    values = {
        contract_package_path: contract_package,
        contract_lock_path: contract_lock,
        adapters_package_path: adapters_package,
        adapters_lock_path: adapters_lock,
        common_package_path: common_package,
        pi_package_path: pi_package,
    }
    changed_paths = [
        path
        for path, value in values.items()
        if source_values[path] != json.dumps(value, sort_keys=True)
    ]

    if changed_paths:
        for path in changed_paths:
            _write_json_object(path, values[path])
        print(f"TypeScript package graph updated to {version}")
    else:
        print(f"TypeScript package graph already set to {version}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: set_typescript_project_version.py <version>")
    set_typescript_project_version(Path.cwd(), sys.argv[1])
