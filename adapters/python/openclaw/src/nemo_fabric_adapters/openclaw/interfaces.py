# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private attachment discovery for clients of one managed OpenClaw Gateway."""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import sys
from pathlib import Path
from urllib.parse import urlsplit

from nemo_fabric_adapters.common import lifecycle, local_interfaces

logger = logging.getLogger(__name__)
FILES = ("terminal.running", "terminal", "gateway.json", "client.json", "gateway.token")


def enabled(settings: dict, name: str) -> bool:
    """A present interface is enabled unless explicitly disabled."""
    item = settings.get("interfaces", {}).get(name)
    return item is not None and item.get("enabled", True)


def public_origins(settings: dict) -> list[str]:
    """Accept HTTP origins without credentials, paths, queries, or fragments."""
    origins = (
        settings.get("interfaces", {}).get("dashboard", {}).get("public_origins", [])
    )
    for origin in origins:
        try:
            parsed = urlsplit(origin)
            valid = (
                parsed.scheme in {"http", "https"}
                and parsed.hostname
                and parsed.username is None
                and parsed.password is None
                and not parsed.path
                and not parsed.query
                and not parsed.fragment
                and parsed.port != 0
            )
        except ValueError:
            valid = False
        if not valid:
            raise lifecycle.LifecycleError(
                "openclaw_invalid_origin",
                "Dashboard public_origins must contain HTTP(S) origins",
                metadata={
                    "field": "harness.settings.interfaces.dashboard.public_origins"
                },
            )
    return origins


def publish(
    directory: Path,
    *,
    runtime_id: str,
    port: int,
    token: str,
    command: Path,
    workspace: str,
    agent_id: str,
    dashboard: bool,
    terminal: bool,
) -> None:
    """Publish connection material after Gateway readiness and the marker last."""
    token_file = "dashboard.token" if dashboard else "gateway.token"
    if not dashboard:
        local_interfaces.write_private_file(directory / token_file, token)
    connection = {
        "version": 1,
        "runtime_id": runtime_id,
        "http_url": f"http://127.0.0.1:{port}",
        "ws_url": f"ws://127.0.0.1:{port}",
        "agent_id": agent_id,
        "token_file": token_file,
        "command": str(command),
        "workspace": workspace,
    }
    local_interfaces.write_private_file(
        directory / "gateway.json", json.dumps(connection) + "\n"
    )
    if not terminal:
        return
    client = {
        "gateway": {
            "mode": "remote",
            "remote": {
                "url": connection["ws_url"],
                "transport": "direct",
                "token": {
                    "source": "env",
                    "provider": "default",
                    "id": "FABRIC_OPENCLAW_CLIENT_TOKEN",
                },
            },
        },
    }
    local_interfaces.write_private_file(
        directory / "client.json", json.dumps(client) + "\n"
    )
    launcher = (
        "#!/bin/sh\nexec "
        + " ".join(
            shlex.quote(arg)
            for arg in (
                sys.executable,
                "-m",
                "nemo_fabric_adapters.openclaw.terminal",
                "--connection",
                str(directory / "gateway.json"),
            )
        )
        + ' "$@"\n'
    )
    local_interfaces.write_private_file(directory / "terminal", launcher)
    os.chmod(directory / "terminal", 0o700)
    local_interfaces.write_private_file(
        directory / "terminal.running", runtime_id + "\n"
    )


def remove(directory: Path) -> bool:
    """Revoke the marker first; retain failed paths for a later stop retry."""
    removed = True
    for name in FILES:
        try:
            (directory / name).unlink(missing_ok=True)
        except OSError:
            removed = False
            logger.error(
                "OpenClaw could not remove interface file %s",
                directory / name,
                exc_info=True,
            )
    try:
        shutil.rmtree(directory / "client-state")
    except FileNotFoundError:
        pass
    except OSError:
        removed = False
        logger.error("OpenClaw could not remove private client state", exc_info=True)
    return removed
