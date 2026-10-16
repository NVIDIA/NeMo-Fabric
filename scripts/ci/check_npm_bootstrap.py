# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover public npm packages and check registry existence without credentials."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


REGISTRY = "https://registry.npmjs.org"


def package_directories(root: Path) -> list[Path]:
    """List public manifest packages with internal runtime dependencies first."""
    workspace = root / "adapters/typescript"
    patterns = json.loads((workspace / "package.json").read_text())["workspaces"]
    candidates = [root / "adapter-contract/typescript"]
    for pattern in patterns:
        matches = sorted(
            directory
            for directory in workspace.glob(pattern)
            if (directory / "package.json").is_file()
        )
        if not matches:
            raise ValueError(f"Workspace pattern has no matches: {pattern}")
        candidates.extend(matches)
    packages = {}
    for directory in candidates:
        directory = directory.resolve()
        directory.relative_to(root.resolve())
        manifest = json.loads((directory / "package.json").read_text())
        if manifest.get("private") is True:
            continue
        name = manifest["name"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"Invalid package name in {directory}")
        if name in packages:
            if packages[name][0] == directory:
                continue
            raise ValueError(f"Duplicate public package name: {name}")
        packages[name] = (directory, manifest)

    ordered = []
    pending = dict(packages)
    while pending:
        ready = [
            name
            for name, (_, manifest) in pending.items()
            if not (
                (
                    set(manifest.get("dependencies", {}))
                    | set(manifest.get("peerDependencies", {}))
                    | set(manifest.get("optionalDependencies", {}))
                )
                & pending.keys()
            )
        ]
        if not ready:
            raise ValueError(f"Public npm dependency cycle: {', '.join(pending)}")
        for name in ready:
            ordered.append(pending.pop(name)[0])
    if not ordered:
        raise ValueError("No public npm packages discovered")
    return ordered


def check_packages(directories: list[Path], *, open_url=urlopen) -> None:
    """Check every package, distinguishing absence from inconclusive lookups."""
    errors = []
    for directory in directories:
        name = json.loads((directory / "package.json").read_text())["name"]
        url = f"{REGISTRY}/{quote(name, safe='')}"
        request = Request(url, headers={"Accept": "application/json"})
        try:
            with open_url(request, timeout=15) as response:
                if response.status != 200:
                    raise ValueError(f"Unexpected registry HTTP {response.status}")
            print(f"Registry package exists: {name}")
        except HTTPError as error:
            if error.code == 404:
                errors.append(
                    f"{name}: package metadata HTTP 404; an npm maintainer must "
                    "bootstrap this package and configure trusted publishing. "
                    "See RELEASING.md#bootstrap-npm-trusted-publishing."
                )
            else:
                errors.append(
                    f"{name}: registry HTTP {error.code}; existence could not be verified"
                )
        except (URLError, TimeoutError, OSError, ValueError) as error:
            errors.append(
                f"{name}: registry lookup failed ({error}); existence could not be verified"
            )
    if errors:
        raise ValueError("\n".join(errors))
    print(
        "Public existence verified; trusted-publisher settings require npm maintainer readback."
    )


def main() -> None:
    """List discovered packages or verify their public registry existence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-directories", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    directories = package_directories(root)
    if args.list_directories:
        for directory in directories:
            print(directory.relative_to(root))
    else:
        check_packages(directories)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        sys.exit(str(error))
