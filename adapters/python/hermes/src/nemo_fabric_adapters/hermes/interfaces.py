# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve Hermes's API server and dashboard as children of a managed runtime."""

from __future__ import annotations

import asyncio
import http.client
import json
import logging
import os
import secrets
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common import local_interfaces
from nemo_fabric_adapters.common import utils as common_utils

HEALTH_PATHS = {"api": "/health", "dashboard": "/api/status"}
LOGGER = logging.getLogger(__name__)
# Children run in their own sessions and stop by process group.
POSIX = os.name != "nt"
DEFAULT_STARTUP_TIMEOUT_SECONDS = 60.0
# Fabric gives the adapter host 10 s to stop, including this wait.
STOP_GRACE_SECONDS = 5.0
# Hermes's gateway waits this long for in-flight API runs and cron jobs on
# SIGTERM, so it exits within STOP_GRACE_SECONDS.
GATEWAY_DRAIN_TIMEOUT_SECONDS = 3


def interface_ports(settings: dict[str, Any]) -> dict[str, int]:
    """Return the enabled interfaces' loopback ports, keyed by interface name.

    The descriptor's schema has already checked the settings.
    """
    return {
        name: interface["port"]
        for name, interface in settings.get("interfaces", {}).items()
        if interface.get("enabled", True)
    }


def startup_timeout(settings: dict[str, Any]) -> float:
    """Return the deadline for every enabled interface to pass its health check."""
    return float(
        settings.get("startup_timeout_seconds", DEFAULT_STARTUP_TIMEOUT_SECONDS)
    )


def require_supported_platform() -> None:
    """Refuse interfaces on a host without POSIX process groups."""
    if not POSIX:
        raise lifecycle.LifecycleError(
            "hermes_interfaces_unsupported_platform",
            "Hermes interfaces require a POSIX host",
            metadata={"field": "harness.settings.interfaces"},
        )


@dataclass
class _Child:
    name: str
    port: int
    process: asyncio.subprocess.Process


class InterfaceChildren:
    """Start, track and stop one runtime's interface processes."""

    def __init__(self) -> None:
        self._children: list[_Child] = []
        self._token_path: Path | None = None
        self._agent_config_path: Path | None = None

    async def start(
        self,
        ports: dict[str, int],
        *,
        hermes_home: Path,
        token_path: Path,
        timeout: float,
        environment: dict[str, str],
        agent_config: AgentConfig,
    ) -> None:
        """Start the enabled interfaces and write the API token once all pass health."""
        require_supported_platform()
        for name, port in ports.items():
            if not local_interfaces.port_available(port):
                raise lifecycle.LifecycleError(
                    "hermes_interface_port_unavailable",
                    f"Hermes {name} port {port} is already in use",
                    metadata={
                        "field": f"harness.settings.interfaces.{name}.port",
                        "port": port,
                    },
                )
        web_dist = _web_dist() if "dashboard" in ports else None
        token = secrets.token_urlsafe(32)
        log_dir = hermes_home / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        deadline = asyncio.get_running_loop().time() + timeout
        try:
            config_path = hermes_home / "interfaces" / "agent-config.json"
            local_interfaces.write_private_file(
                config_path, json.dumps(agent_config.to_mapping())
            )
            self._agent_config_path = config_path
            for name, port in ports.items():
                arguments, child_environment = _child_command(
                    name, port, token=token, web_dist=web_dist, environment=environment
                )
                with open(log_dir / f"fabric-interface-{name}.log", "ab") as log:
                    process = await asyncio.create_subprocess_exec(
                        sys.executable,
                        "-m",
                        "nemo_fabric_adapters.hermes.native",
                        "--fabric-config",
                        str(config_path),
                        *arguments,
                        env=child_environment,
                        stdin=asyncio.subprocess.DEVNULL,
                        stdout=log,
                        stderr=asyncio.subprocess.STDOUT,
                        start_new_session=True,
                    )
                self._children.append(_Child(name, port, process))
            for child in self._children:
                await _wait_healthy(child, deadline)
            if "api" in ports:
                local_interfaces.write_private_file(token_path, token)
                self._token_path = token_path
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        """Remove the API token, then stop every interface process group."""
        self._remove_private_file("_token_path")
        running = [
            child for child in self._children if child.process.returncode is None
        ]
        pending: set[asyncio.Future[int]] = set()
        try:
            for child in running:
                common_utils.signal_process_group(child.process.pid, signal.SIGTERM)
            if running:
                waits = [
                    asyncio.ensure_future(child.process.wait()) for child in running
                ]
                _, pending = await asyncio.wait(waits, timeout=STOP_GRACE_SECONDS)
        finally:
            # Runs even when stop is cancelled, so no child outlives the runtime.
            for child in running:
                common_utils.signal_process_group(child.process.pid, signal.SIGKILL)
            self._children = []
            self._remove_private_file("_agent_config_path")
        if pending:
            await asyncio.wait(pending)

    def _remove_private_file(self, attribute: str) -> None:
        path = getattr(self, attribute)
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                # Retain the path for retry; cleanup must not prevent child stop.
                LOGGER.exception("Could not remove Hermes interface file %s", path)
            else:
                setattr(self, attribute, None)


def _web_dist() -> Path:
    configured = os.environ.get("HERMES_WEB_DIST")
    if configured:
        dist = Path(configured)
    else:
        import hermes_cli

        dist = Path(hermes_cli.__file__).parent / "web_dist"
    if not (dist / "index.html").is_file():
        raise lifecycle.LifecycleError(
            "hermes_dashboard_frontend_missing",
            f"Hermes's dashboard needs a built frontend at {dist}; set "
            "HERMES_WEB_DIST to one",
            metadata={"field": "harness.settings.interfaces.dashboard"},
        )
    return dist


async def _wait_healthy(child: _Child, deadline: float) -> None:
    loop = asyncio.get_running_loop()
    while loop.time() < deadline:
        if child.process.returncode is not None:
            raise lifecycle.LifecycleError(
                "hermes_interface_start_failed",
                f"Hermes {child.name} exited before passing its health check",
                metadata={"interface": child.name},
            )
        if await _http_status(child.port, HEALTH_PATHS[child.name]) == 200:
            return
        await asyncio.sleep(0.1)
    raise lifecycle.LifecycleError(
        "hermes_interface_start_timeout",
        f"Hermes {child.name} didn't pass its health check before the startup timeout",
        retryable=True,
        metadata={"interface": child.name},
    )


def _child_command(
    name: str,
    port: int,
    *,
    token: str,
    web_dist: Path | None,
    environment: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    """Return one interface's Hermes CLI arguments and environment."""
    child_environment = {
        key: value
        for key, value in environment.items()
        if not key.startswith("API_SERVER_")
    }
    if name == "api":
        child_environment.update(
            API_SERVER_KEY=token,
            API_SERVER_HOST="127.0.0.1",
            API_SERVER_PORT=str(port),
        )
        return ["gateway", "run"], child_environment
    child_environment["HERMES_WEB_DIST"] = str(web_dist)
    # --isolated keeps one dashboard per runtime instead of attaching to another
    # runtime's dashboard on this host.
    return [
        "dashboard",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-open",
        "--skip-build",
        "--isolated",
    ], child_environment


async def _http_status(port: int, path: str) -> int | None:
    """Return a loopback GET's status, or None when nothing answers.

    ``http.client`` neither follows redirects nor uses a proxy, which can't
    reach this process's loopback anyway.
    """

    def get() -> int | None:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1.0)
        try:
            connection.request("GET", path)
            return connection.getresponse().status
        except (OSError, http.client.HTTPException):
            return None
        finally:
            connection.close()

    return await asyncio.to_thread(get)
