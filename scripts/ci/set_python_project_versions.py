# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path


PROJECT_VERSION_PATTERN = re.compile(r'^version\s*=\s*"[^"]+"$', flags=re.MULTILINE)
INTERNAL_PIN_PATTERN = re.compile(
    r"(?P<prefix>nemo-fabric-[a-z0-9-]+(?:\[[^\]]+\])?\s*==\s*)"
    r'(?P<version>[^"\s,;]+)'
)


def set_python_project_versions(root: Path, version: str) -> None:
    project_paths = (
        root / "sdk" / "python" / "nemo-fabric" / "pyproject.toml",
        root / "adapter-contract" / "python" / "pyproject.toml",
        *sorted((root / "adapters").glob("**/pyproject.toml")),
    )

    for path in project_paths:
        text = path.read_text(encoding="utf-8")
        updated, count = PROJECT_VERSION_PATTERN.subn(
            f'version = "{version}"',
            text,
            count=1,
        )
        if count != 1:
            raise SystemExit(f"Failed to find exactly one project version in {path}")
        updated = INTERNAL_PIN_PATTERN.sub(
            lambda match: f"{match.group('prefix')}{version}",
            updated,
        )
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            print(
                f"{path.relative_to(root)} version and internal pins updated to {version}"
            )
        else:
            print(f"{path.relative_to(root)} already set to {version}")

    runtime_path = (
        root / "sdk" / "python" / "nemo-fabric-runtime" / "pyproject.toml"
    )
    runtime = tomllib.loads(runtime_path.read_text(encoding="utf-8"))
    project = runtime.get("project", {})
    if "version" in project or "version" not in project.get("dynamic", []):
        raise SystemExit(
            "sdk/python/nemo-fabric-runtime/pyproject.toml must keep a dynamic "
            "version derived from Cargo.toml"
        )

    mismatched_pins = []
    for path in project_paths:
        for match in INTERNAL_PIN_PATTERN.finditer(path.read_text(encoding="utf-8")):
            if match.group("version") != version:
                mismatched_pins.append(f"{path.relative_to(root)}: {match.group(0)}")
    if mismatched_pins:
        raise SystemExit(
            "Internal Python dependency pins are not synchronized: "
            + ", ".join(mismatched_pins)
        )
    print(
        "sdk/python/nemo-fabric-runtime/pyproject.toml continues to derive its "
        "version from Cargo.toml"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: set_python_project_versions.py <version>")
    set_python_project_versions(Path.cwd(), sys.argv[1])
