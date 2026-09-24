# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained credentials for native interfaces that an adapter exposes to users.

A native API or dashboard whose state outlives one runtime needs a credential
that its operator can read from the adapter's retained state directory. The
credential is created once, readable only by its owner, and never replaced.
"""

from __future__ import annotations

import os
import re
import secrets
import stat
import tempfile
from pathlib import Path

TOKEN_FILE = "interface-token"
_TOKEN = re.compile(r"[a-f0-9]{64}")


def interface_token(directory: str | os.PathLike[str], *, create: bool = False) -> str:
    """Return the retained interface token in *directory*.

    With ``create``, a missing token is written completely to a private
    temporary file and then linked into place, so readers never observe a
    partial token and a concurrently published token wins. Reading rejects
    symlinks, non-regular files, group or world permissions, and files owned
    by another user.
    """

    path = Path(directory) / TOKEN_FILE
    if create:
        _publish(path)
    return _read(path)


def _publish(path: Path) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{TOKEN_FILE}-")
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as output:
            output.write(secrets.token_hex(32))
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            pass
    finally:
        os.unlink(temporary)


def _read(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, encoding="ascii") as source:
        info = os.fstat(source.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_mode & 0o077
            or info.st_uid != os.geteuid()
        ):
            raise RuntimeError("invalid retained interface credential")
        value = source.read(65)
    if not _TOKEN.fullmatch(value):
        raise RuntimeError("invalid retained interface credential")
    return value
