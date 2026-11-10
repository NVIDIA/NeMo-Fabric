# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the native TUI against the managed Gateway without secrets in argv."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import httpx


def _read_private(path: Path) -> str:
    """Read a private owned file without following its final symlink."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor) as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
        ):
            raise ValueError("Attachment files must be owner-only regular files")
        return stream.read()


def attachment(
    connection: Path, session: str | None = None
) -> tuple[list[str], dict[str, str], str]:
    """Validate discovery and authenticated health, then build a native client launch."""
    if os.name != "posix":
        raise ValueError("Managed terminal attachment currently requires POSIX")
    parent = connection.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.getuid()
        or parent.st_mode & 0o077
    ):
        raise ValueError(
            "Attachment directory must be private and owned by the current user"
        )
    data = json.loads(_read_private(connection))
    parsed = urlsplit(data["http_url"])
    if (
        data["version"] != 1
        or parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or not parsed.port
        or parsed.username
        or parsed.password
        or parsed.path
        or parsed.query
        or parsed.fragment
        or data["ws_url"] != f"ws://127.0.0.1:{parsed.port}"
        or data["token_file"] not in {"dashboard.token", "gateway.token"}
    ):
        raise ValueError("Invalid managed Gateway descriptor")
    if (
        _read_private(connection.with_name("terminal.running")).strip()
        != data["runtime_id"]
    ):
        raise ValueError("Managed terminal attachment is no longer available")
    token = _read_private(connection.with_name(data["token_file"])).strip()
    with httpx.Client(trust_env=False, timeout=3) as client:
        response = client.get(
            data["http_url"] + "/v1/models",
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
    # Validate the client config as private even though it contains only a secret reference.
    _read_private(connection.with_name("client.json"))
    command = Path(data["command"])
    workspace = Path(data["workspace"])
    if (
        not command.is_absolute()
        or not os.access(command, os.X_OK)
        or not workspace.is_absolute()
        or not workspace.is_dir()
    ):
        raise ValueError("Managed executable or workspace is unavailable")
    session = f"fabric-terminal-{uuid.uuid4().hex}" if session is None else session
    if (
        ":" in session
        or not session
        or any(character.isspace() for character in session)
    ):
        raise ValueError("Use a nonempty session name without colons or whitespace")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("OPENCLAW_")
    }
    environment.update(
        {
            "OPENCLAW_CONFIG_PATH": str(connection.with_name("client.json").absolute()),
            "OPENCLAW_STATE_DIR": str(connection.parent / "client-state"),
            "OPENCLAW_CONFIG_READONLY": "1",
            "DO_NOT_TRACK": "1",
            "FABRIC_OPENCLAW_CLIENT_TOKEN": token,
        }
    )
    return (
        [str(command), "tui", "--session", f"agent:{data['agent_id']}:{session}"],
        environment,
        str(workspace),
    )


def main() -> None:
    """Attach to one selected runtime; the client never starts a Gateway."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection", type=Path, required=True)
    parser.add_argument(
        "--session", help="Native session name; omit for a new conversation"
    )
    args = parser.parse_args()
    try:
        command, environment, workspace = attachment(args.connection, args.session)
    except (OSError, ValueError, KeyError, httpx.HTTPError):
        parser.exit(
            1,
            "Managed OpenClaw attachment is unavailable; check the runtime and private connection files.\n",
        )
    os.chdir(workspace)
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
