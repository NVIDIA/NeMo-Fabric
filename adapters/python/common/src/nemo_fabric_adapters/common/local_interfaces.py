# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for native interfaces that an adapter serves on loopback."""

from __future__ import annotations

import os
import socket
import tempfile
from pathlib import Path


def port_available(port: int) -> bool:
    """Return whether one loopback TCP port is available."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            # On Windows, SO_REUSEADDR lets a bind succeed on a port in use.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
        return True
    except (OSError, OverflowError):
        return False
    finally:
        sock.close()


def write_private_file(path: Path, payload: str) -> None:
    """Atomically write an owner-only file in an owner-only directory.

    On Windows, chmod sets only the read-only flag, so the file inherits the
    directory's permissions.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
