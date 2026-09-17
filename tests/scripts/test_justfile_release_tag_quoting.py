# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "arguments",
    [
        ("normalize-release-tag",),
        ("release-tag-to-py-version",),
        ("set-cargo-version",),
        ("set-version",),
        ("--set", "ref_name={payload}", "set-cargo-version"),
        ("--set", "ref_name={payload}", "set-version"),
    ],
)
def test_release_tag_interpolation_does_not_execute_command_substitution(
    tmp_path: Path,
    arguments: tuple[str, ...],
):
    marker = tmp_path / "interpolation-executed"
    payload = f"$(touch {marker})"
    command = [
        "just",
        *(argument.format(payload=payload) for argument in arguments),
    ]
    if not any("{payload}" in argument for argument in arguments):
        command.append(payload)

    environment = os.environ.copy()
    environment["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert not marker.exists()


@pytest.mark.parametrize(
    ("release_tag", "expected_version"),
    [
        ("v1.2.3", "1.2.3"),
        ("v1.2.3-rc.1", "1.2.3rc1"),
    ],
)
def test_release_tag_to_py_version(release_tag: str, expected_version: str):
    result = subprocess.run(
        ["just", "release-tag-to-py-version", release_tag],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == f"{expected_version}\n"
