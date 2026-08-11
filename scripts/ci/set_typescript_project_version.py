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
PACKAGE_DIRECTORY = Path("typescript/adapter-contract")


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a JSON object in {path}")
    return value


def _write_json_object(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def set_typescript_project_version(root: Path, version: str) -> None:
    if SEMVER_PATTERN.fullmatch(version) is None:
        raise SystemExit(f"Unsupported TypeScript package version: {version}")

    package_path = root / PACKAGE_DIRECTORY / "package.json"
    lock_path = root / PACKAGE_DIRECTORY / "package-lock.json"
    package = _read_json_object(package_path)
    lock = _read_json_object(lock_path)

    package_name = package.get("name")
    lock_packages = lock.get("packages")
    if not isinstance(package_name, str) or not package_name:
        raise SystemExit(f"Expected a non-empty package name in {package_path}")
    if "version" not in package:
        raise SystemExit(f"Expected a package version in {package_path}")
    if not isinstance(lock_packages, dict):
        raise SystemExit(f"Expected a packages object in {lock_path}")

    lock_root = lock_packages.get("")
    if not isinstance(lock_root, dict):
        raise SystemExit(f"Expected a root package entry in {lock_path}")
    if lock.get("name") != package_name or lock_root.get("name") != package_name:
        raise SystemExit(
            f"Package names in {package_path} and {lock_path} are not synchronized"
        )
    if "version" not in lock or "version" not in lock_root:
        raise SystemExit(f"Expected root package version fields in {lock_path}")

    changed = (
        package.get("version") != version
        or lock.get("version") != version
        or lock_root.get("version") != version
    )
    package["version"] = version
    lock["version"] = version
    lock_root["version"] = version

    if changed:
        _write_json_object(package_path, package)
        _write_json_object(lock_path, lock)
        print(f"{PACKAGE_DIRECTORY} version updated to {version}")
    else:
        print(f"{PACKAGE_DIRECTORY} already set to {version}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: set_typescript_project_version.py <version>")
    set_typescript_project_version(Path.cwd(), sys.argv[1])
