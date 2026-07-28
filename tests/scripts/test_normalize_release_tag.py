# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CI_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "ci"
sys.path.insert(0, str(CI_SCRIPTS))

import normalize_release_tag  # noqa: E402


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("v0.1.0", "0.1.0"),
        ("v0.1.0-rc1", "0.1.0-rc.1"),
        ("0.1.0-rc1", "0.1.0-rc.1"),
        ("0.1.0-rc.1", "0.1.0-rc.1"),
        ("0.1.0-alpha.20260727", "0.1.0-alpha.20260727"),
    ],
)
def test_normalize_release_tag(tag: str, expected: str):
    assert normalize_release_tag.normalize_release_tag(tag) == expected


@pytest.mark.parametrize("tag", ["v0.1", "v0.1.0-dev", "v0.1.0-beta1"])
def test_normalize_release_tag_rejects_unsupported_tags(tag: str):
    with pytest.raises(ValueError, match="Unsupported release tag"):
        normalize_release_tag.normalize_release_tag(tag)
