# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from set_typescript_project_version import SEMVER_PATTERN


PACKAGE_DIRECTORIES = {
    "nemo-fabric-adapters-common": Path("adapters/typescript/common"),
    "nemo-fabric-adapters-pi": Path("adapters/typescript/pi"),
}
LOCK_PATH = Path("adapters/typescript/package-lock.json")


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a JSON object in {path}")
    return value


def _write_json_object(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def set_typescript_adapter_version(root: Path, package_name: str, version: str) -> None:
    if SEMVER_PATTERN.fullmatch(version) is None:
        raise SystemExit(f"Unsupported TypeScript package version: {version}")
    package_directory = PACKAGE_DIRECTORIES.get(package_name)
    if package_directory is None:
        raise SystemExit(f"Unsupported TypeScript adapter package: {package_name}")

    package_path = root / package_directory / "package.json"
    lock_path = root / LOCK_PATH
    package = _read_json_object(package_path)
    lock = _read_json_object(lock_path)
    if package.get("name") != package_name:
        raise SystemExit(f"Package name in {package_path} does not match {package_name}")

    packages = lock.get("packages")
    if not isinstance(packages, dict):
        raise SystemExit(f"Expected a packages object in {lock_path}")
    workspace_key = package_directory.relative_to("adapters/typescript").as_posix()
    lock_package = packages.get(workspace_key)
    if not isinstance(lock_package, dict) or lock_package.get("name") != package_name:
        raise SystemExit(f"Expected synchronized {package_name} metadata in {lock_path}")

    package["version"] = version
    lock_package["version"] = version
    if package_name == "nemo-fabric-adapters-common":
        pi_package_path = root / PACKAGE_DIRECTORIES["nemo-fabric-adapters-pi"] / "package.json"
        pi_package = _read_json_object(pi_package_path)
        pi_lock_package = packages.get("pi")
        if (
            pi_package.get("name") != "nemo-fabric-adapters-pi"
            or not isinstance(pi_package.get("dependencies"), dict)
            or not isinstance(pi_lock_package, dict)
            or not isinstance(pi_lock_package.get("dependencies"), dict)
        ):
            raise SystemExit("Expected synchronized Pi dependency metadata")
        pi_package["dependencies"][package_name] = version
        pi_lock_package["dependencies"][package_name] = version
        _write_json_object(pi_package_path, pi_package)
    _write_json_object(package_path, package)
    _write_json_object(lock_path, lock)
    print(f"{package_directory} version updated to {version}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: set_typescript_adapter_version.py <package-name> <version>"
        )
    set_typescript_adapter_version(Path.cwd(), sys.argv[1], sys.argv[2])
