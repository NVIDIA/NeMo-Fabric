# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual one-shot reader used by the provider."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Unix task reader")
READER = (
    Path(__file__).resolve().parents[2]
    / "environment-providers/openshell/src/read_artifacts.py"
)


def read(root, paths):
    process = subprocess.run(
        [sys.executable, str(READER)],
        input=json.dumps({"root": str(root), "paths": paths}),
        text=True,
        capture_output=True,
        timeout=5,
    )
    return process.returncode, json.loads(process.stdout), process.stderr


def test_export_preserves_complete_binary_files(tmp_path):
    (tmp_path / "nested").mkdir()
    contents = bytes(range(256)) * 1024
    (tmp_path / "nested/review.gz").write_bytes(contents)
    code, result, stderr = read(tmp_path, ["nested/review.gz"])
    assert code == 0
    assert stderr == ""
    assert bytes(result["files"][0]["content"]) == contents


@pytest.mark.parametrize(
    "path",
    [
        "../PRIVATE",
        "/PRIVATE",
        "nested/../PRIVATE",
        "nested//file",
        "nested\\file",
        "file\0",
        "",
    ],
)
def test_export_rejects_invalid_paths_without_disclosing_data(tmp_path, path):
    code, result, stderr = read(tmp_path, [path])
    assert code == 1
    assert result == {"error": "artifact_read_failed"}
    assert stderr == ""


@pytest.mark.parametrize(
    "kind",
    [
        "leaf_symlink",
        "parent_symlink",
        "root_symlink",
        "fifo",
        "directory",
        "oversized",
    ],
)
def test_export_rejects_nonregular_or_escaping_files(tmp_path, kind):
    root = tmp_path / "artifacts"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "PRIVATE").write_text("PRIVATE")
    path = "review"
    if kind == "leaf_symlink":
        (root / path).symlink_to(outside / "PRIVATE")
    elif kind == "parent_symlink":
        (root / "nested").symlink_to(outside, target_is_directory=True)
        path = "nested/PRIVATE"
    elif kind == "root_symlink":
        root.rmdir()
        root.symlink_to(outside, target_is_directory=True)
        path = "PRIVATE"
    elif kind == "fifo":
        os.mkfifo(root / path)
    elif kind == "directory":
        (root / path).mkdir()
    else:
        (root / path).write_bytes(b"x" * (256 * 1024 + 1))
    code, result, stderr = read(root, [path])
    assert code == 1
    assert result == {"error": "artifact_read_failed"}
    assert stderr == ""


@pytest.mark.parametrize("case", ["file_count", "total_bytes"])
def test_export_rejects_limits_without_partial_file_results(tmp_path, case):
    size = 256 * 1024 if case == "total_bytes" else 1
    count = 3 if case == "total_bytes" else 17
    paths = [f"file-{i}" for i in range(count)]
    for path in paths:
        (tmp_path / path).write_bytes(b"x" * size)
    code, result, stderr = read(tmp_path, paths)
    assert code == 1
    assert result == {"error": "artifact_read_failed"}
    assert stderr == ""
