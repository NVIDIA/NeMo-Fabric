# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native interface secrets remain private and stable across runtime starts."""
import pytest
from nemo_fabric_adapters.common.credentials import token


def test_interface_token_is_retained_and_rejects_public_permissions(tmp_path):
    first = token(tmp_path, create=True)
    assert len(first) == 64
    assert token(tmp_path, create=True) == first
    path = tmp_path / "interface-token"
    assert path.stat().st_mode & 0o777 == 0o600
    path.chmod(0o644)
    with pytest.raises(RuntimeError, match="invalid retained"):
        token(tmp_path)


def test_interface_token_never_follows_symlinks(tmp_path):
    outside = tmp_path / "other"
    outside.write_text("a" * 64)
    outside.chmod(0o600)
    (tmp_path / "interface-token").symlink_to(outside)
    with pytest.raises(OSError):
        token(tmp_path, create=True)
