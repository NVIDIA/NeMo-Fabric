#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that every built wheel contains the canonical repository license."""

from __future__ import annotations

from pathlib import Path
from pathlib import PurePosixPath
from zipfile import BadZipFile
from zipfile import ZipFile


class WheelLicenseError(RuntimeError):
    """A wheel is missing the canonical license text."""


def _normalized_text(content: bytes) -> str:
    return content.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def _license_members(archive: ZipFile) -> list[str]:
    members = []
    for name in archive.namelist():
        parts = PurePosixPath(name).parts
        if (
            len(parts) >= 3
            and parts[-3].endswith(".dist-info")
            and parts[-2:] == ("licenses", "LICENSE")
        ):
            members.append(name)
    return members


def validate_wheel_licenses(repository: Path, wheel_directory: Path) -> int:
    """Validate all wheels in a directory and return the number checked."""
    expected = _normalized_text((repository / "LICENSE").read_bytes())
    wheels = sorted(wheel_directory.glob("*.whl"))
    if not wheels:
        raise WheelLicenseError(f"no wheels found in {wheel_directory}")

    errors = []
    for wheel in wheels:
        try:
            with ZipFile(wheel) as archive:
                members = _license_members(archive)
                if len(members) != 1:
                    errors.append(
                        f"{wheel.name} contains {len(members)} canonical license files; "
                        "expected exactly one"
                    )
                    continue
                actual = _normalized_text(archive.read(members[0]))
        except BadZipFile:
            errors.append(f"{wheel.name} is not a valid wheel archive")
            continue
        if actual != expected:
            errors.append(f"{wheel.name} license does not match the repository LICENSE")

    if errors:
        raise WheelLicenseError("; ".join(errors))
    return len(wheels)


def main() -> None:
    """Validate licenses in the repository wheel output directory."""
    repository = Path(__file__).resolve().parents[2]
    try:
        checked = validate_wheel_licenses(repository, repository / "dist")
    except (OSError, UnicodeDecodeError, WheelLicenseError) as error:
        raise SystemExit(f"Wheel license validation failed: {error}") from error
    print(f"Validated canonical license text in {checked} wheel(s)")


if __name__ == "__main__":
    main()
