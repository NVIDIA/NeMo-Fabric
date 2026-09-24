# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native interface credentials remain private and stable across runtime starts."""

import os
import sys

import pytest
from nemo_fabric_adapters.common.credentials import interface_token

if sys.platform == "win32":
    pytest.skip("interface credentials require a POSIX host", allow_module_level=True)


def test_interface_token_is_retained_and_rejects_public_permissions(tmp_path):
    first = interface_token(tmp_path, create=True)
    assert len(first) == 64
    assert interface_token(tmp_path, create=True) == first
    assert interface_token(tmp_path) == first
    path = tmp_path / "interface-token"
    assert path.stat().st_mode & 0o777 == 0o600
    assert sorted(item.name for item in tmp_path.iterdir()) == ["interface-token"]
    path.chmod(0o644)
    with pytest.raises(RuntimeError, match="invalid retained"):
        interface_token(tmp_path)


def test_interface_token_never_replaces_or_follows_existing_entries(tmp_path):
    outside = tmp_path / "other"
    outside.write_text("a" * 64)
    outside.chmod(0o600)
    (tmp_path / "interface-token").symlink_to(outside)
    with pytest.raises(OSError):
        interface_token(tmp_path, create=True)
    assert outside.read_text() == "a" * 64


def test_interface_token_rejects_non_regular_files_without_blocking(tmp_path):
    os.mkfifo(tmp_path / "interface-token", 0o600)
    with pytest.raises(RuntimeError, match="invalid retained"):
        interface_token(tmp_path)
