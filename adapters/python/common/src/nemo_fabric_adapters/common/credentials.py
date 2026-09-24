# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Protected persistent local interface credentials owned by native adapters."""

import os
import re
import secrets
import stat
from pathlib import Path

TOKEN_NAME = "interface-token"


def token(home, create=False):
    path = Path(home) / TOKEN_NAME
    if create:
        try:
            fd = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "w") as output:
                output.write(secrets.token_hex(32))
                output.flush()
                os.fsync(output.fileno())
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd) as source:
        info = os.fstat(source.fileno())
        value = source.read(65)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_mode & 0o077
            or info.st_uid != os.geteuid()
            or not re.fullmatch("[a-f0-9]{64}", value)
        ):
            raise RuntimeError("invalid retained interface credential")
    return value
