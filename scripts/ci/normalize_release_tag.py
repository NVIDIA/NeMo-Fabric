# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import sys


RELEASE_TAG_PATTERN = re.compile(
    r"^v?(?P<release>\d+\.\d+\.\d+)"
    r"(?:(?P<prerelease>-(?:alpha|beta|rc)(?:\.\d+)?)"
    r"|-rc(?P<compact_rc>\d+))?"
    r"(?P<build>\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


def normalize_release_tag(tag: str) -> str:
    match = RELEASE_TAG_PATTERN.fullmatch(tag)
    if not match:
        raise ValueError(
            f"Unsupported release tag '{tag}'; use v0.1.0, v0.1.0-rc1, "
            "or raw SemVer such as 0.1.0-rc.1"
        )

    prerelease = match.group("prerelease") or ""
    compact_rc = match.group("compact_rc")
    if compact_rc is not None:
        prerelease = f"-rc.{compact_rc}"

    return f"{match.group('release')}{prerelease}{match.group('build') or ''}"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: normalize_release_tag.py <tag>")
    try:
        print(normalize_release_tag(sys.argv[1]))
    except ValueError as error:
        raise SystemExit(str(error)) from error
