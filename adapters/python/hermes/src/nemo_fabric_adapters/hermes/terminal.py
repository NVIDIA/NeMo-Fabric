# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attach the native terminal renderer to a NeMo Fabric-owned TUI service."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import stat
import shutil
import sys
from pathlib import Path
from urllib.parse import urlencode

from nemo_fabric_adapters.hermes.interfaces import tui_ready


def _read_private(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor) as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
        ):
            raise ValueError("TUI attachment files must be owner-only regular files")
        return stream.read()


def frontend_command() -> list[str]:
    """Use installed assets directly; never install/build another Hermes runtime."""
    import hermes_cli

    cli = Path(hermes_cli.__file__).parent
    configured = os.environ.get("HERMES_TUI_DIR")
    candidates = (
        [Path(configured) / "dist/entry.js"]
        if configured
        else [cli / "tui_dist/entry.js", cli.parent / "ui-tui/dist/entry.js"]
    )
    entry = next((path for path in candidates if path.is_file()), None)
    node = os.environ.get("HERMES_NODE") or shutil.which("node")
    if entry is None or not node or not os.access(node, os.X_OK):
        raise ValueError(
            "Install the pinned Hermes terminal bundle and Node runtime before attaching"
        )
    return [str(node), "--expose-gc", str(entry.resolve())]


def attachment_environment(connection: Path) -> dict[str, str]:
    """Validate the private descriptor and refuse stale/unauthenticated backends."""
    payload = json.loads(_read_private(connection))
    port = payload["port"]
    if (
        payload["version"] != 1
        or payload["mode"] not in {"dashboard", "headless"}
        or type(port) is not int
        or not 1 <= port <= 65535
        or payload["token_file"] != "tui.token"
    ):
        raise ValueError("Invalid TUI attachment descriptor")
    token = _read_private(connection.with_name("tui.token")).strip()
    if not token or not asyncio.run(tui_ready(port, token)):
        raise ValueError("Managed TUI backend is unavailable or authentication failed")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {"HERMES_TUI_SIDECAR_URL", "HERMES_DESKTOP", "HERMES_SERVE_HEADLESS"}
    }
    environment["HERMES_PYTHON"] = sys.executable
    environment["HERMES_TUI_CWD"] = payload["workspace"]
    environment["HERMES_CWD"] = payload["workspace"]
    environment["HERMES_HOME"] = str(connection.resolve().parent.parent)
    environment["HERMES_TUI_GATEWAY_URL"] = (
        f"ws://127.0.0.1:{port}/api/ws?{urlencode({'token': token})}"
    )
    return environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connection",
        type=Path,
        required=True,
        help="Runtime's private interfaces/tui.json file",
    )
    options = parser.parse_args()
    if os.name != "posix":
        parser.exit(1, "Terminal attachment requires a POSIX host\n")
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        parser.exit(1, "Run terminal attachment from an interactive terminal\n")
    try:
        environment = attachment_environment(options.connection)
    except (OSError, ValueError, KeyError):
        parser.exit(
            1,
            "Cannot attach to the managed TUI backend; check the runtime and private attachment files\n",
        )
    try:
        arguments = frontend_command()
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    # The explicit URL makes the native renderer an attaching client, never a
    # backend owner. Credentials stay in its environment, not process arguments.
    os.execvpe(arguments[0], arguments, environment)


if __name__ == "__main__":
    main()
