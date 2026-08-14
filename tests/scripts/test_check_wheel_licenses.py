# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZipFile

import pytest


CI_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "ci"
sys.path.insert(0, str(CI_SCRIPTS))

import check_wheel_licenses  # noqa: E402


def _write_wheel(path: Path, license_content: bytes | None) -> None:
    path.parent.mkdir(parents=True)
    with ZipFile(path, "w") as archive:
        archive.writestr("demo.py", "")
        if license_content is not None:
            archive.writestr("demo-1.0.dist-info/licenses/LICENSE", license_content)


def test_validate_wheel_licenses_accepts_platform_line_endings(tmp_path: Path):
    (tmp_path / "LICENSE").write_text("canonical\nlicense\n", encoding="utf-8")
    wheel_directory = tmp_path / "dist"
    _write_wheel(wheel_directory / "demo.whl", b"canonical\r\nlicense\r\n")

    assert check_wheel_licenses.validate_wheel_licenses(tmp_path, wheel_directory) == 1


@pytest.mark.parametrize(
    ("license_content", "message"),
    [
        (None, "expected exactly one"),
        (b"../../../LICENSE", "does not match"),
    ],
)
def test_validate_wheel_licenses_rejects_invalid_license_content(
    tmp_path: Path,
    license_content: bytes | None,
    message: str,
):
    (tmp_path / "LICENSE").write_text("canonical license", encoding="utf-8")
    wheel_directory = tmp_path / "dist"
    _write_wheel(wheel_directory / "demo.whl", license_content)

    with pytest.raises(check_wheel_licenses.WheelLicenseError, match=message):
        check_wheel_licenses.validate_wheel_licenses(tmp_path, wheel_directory)
