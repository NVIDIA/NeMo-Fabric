# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-shot, bounded reader executed with the task interpreter; no runtime server."""

import json
import os
import stat
import sys

FILE_LIMIT = 256 * 1024
TOTAL_LIMIT = 512 * 1024
COUNT_LIMIT = 16


def directory(parent, name):
    return os.open(
        name,
        getattr(os, "O_PATH", os.O_RDONLY) | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=parent,
    )


def collect(request):
    root = request["root"]
    paths = request["paths"]
    if not isinstance(root, str) or not root.startswith("/") or ".." in root.split("/"):
        raise ValueError("invalid root")
    if not isinstance(paths, list) or len(paths) > COUNT_LIMIT:
        raise ValueError("invalid file count")
    root_descriptor = directory(None, "/")
    try:
        for name in root.split("/"):
            if name in ("", "."):
                continue
            child = directory(root_descriptor, name)
            os.close(root_descriptor)
            root_descriptor = child
        files = []
        total = 0
        for path in paths:
            if not isinstance(path, str) or not path or "\\" in path or "\0" in path:
                raise ValueError("invalid path")
            parts = path.split("/")
            if any(part in ("", ".", "..") for part in parts):
                raise ValueError("invalid path")
            parent = os.dup(root_descriptor)
            try:
                for name in parts[:-1]:
                    child = directory(parent, name)
                    os.close(parent)
                    parent = child
                descriptor = os.open(
                    parts[-1],
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                    dir_fd=parent,
                )
                with os.fdopen(descriptor, "rb") as stream:
                    before = os.fstat(stream.fileno())
                    if not stat.S_ISREG(before.st_mode) or before.st_size > FILE_LIMIT:
                        raise ValueError("not a bounded regular file")
                    content = stream.read(FILE_LIMIT + 1)
                    after = os.fstat(stream.fileno())
                    if len(content) > FILE_LIMIT or (
                        before.st_size,
                        before.st_mtime_ns,
                    ) != (after.st_size, after.st_mtime_ns):
                        raise ValueError("file changed or exceeded limit")
                    total += len(content)
                    if total > TOTAL_LIMIT:
                        raise ValueError("total exceeded limit")
                    files.append({"path": path, "content": list(content)})
            finally:
                os.close(parent)
        return files
    finally:
        os.close(root_descriptor)


try:
    request = json.loads(sys.stdin.buffer.read(16 * 1024 + 1))
    print(json.dumps({"files": collect(request)}, separators=(",", ":")))
except Exception:
    # Never emit submitted paths, file data, or raw OS errors in diagnostics.
    print('{"error":"artifact_read_failed"}')
    sys.exit(1)
