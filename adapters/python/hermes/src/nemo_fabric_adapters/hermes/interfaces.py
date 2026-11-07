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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common import local_interfaces
from nemo_fabric_adapters.common import utils as common_utils

HEALTH_PATHS = {"api": "/health", "dashboard": "/api/status", "terminal": "/api/status"}
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
    ports = {
        name: interface["port"]
        for name, interface in settings.get("interfaces", {}).items()
        if interface.get("enabled", True)
    }
    if "dashboard" in ports and "terminal" in ports:
        if ports["dashboard"] != ports["terminal"]:
            raise lifecycle.LifecycleError(
                "hermes_tui_port_conflict",
                "Dashboard and terminal must use the same TUI backend port",
            )
        del ports["terminal"]
    if len(set(ports.values())) != len(ports):
        raise lifecycle.LifecycleError(
            "hermes_interface_port_conflict", "API and TUI backends need distinct ports"
        )
    return ports


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
    tui_token: str | None = field(default=None, repr=False)


class InterfaceChildren:
    """Start, track and stop one runtime's interface processes."""

    def __init__(self) -> None:
        self._children: list[_Child] = []
        self._token_path: Path | None = None
        self._agent_config_path: Path | None = None
        self._tui_token_path: Path | None = None
        self._tui_descriptor_path: Path | None = None

    async def start(
        self,
        ports: dict[str, int],
        *,
        hermes_home: Path,
        token_path: Path,
        timeout: float,
        environment: dict[str, str],
        agent_config: AgentConfig,
        terminal_enabled: bool = False,
        workspace: str = ".",
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
        tui_token = secrets.token_urlsafe(32)
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
                    name,
                    port,
                    token=token if name == "api" else tui_token,
                    web_dist=web_dist,
                    environment=environment,
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
                self._children.append(
                    _Child(name, port, process, tui_token if name != "api" else None)
                )
            for child in self._children:
                await _wait_healthy(child, deadline)
            if terminal_enabled:
                tui_name = "dashboard" if "dashboard" in ports else "terminal"
                self._tui_token_path = token_path.with_name("tui.token")
                self._tui_descriptor_path = token_path.with_name("tui.json")
                local_interfaces.write_private_file(self._tui_token_path, tui_token)
                local_interfaces.write_private_file(
                    self._tui_descriptor_path,
                    json.dumps(
                        {
                            "version": 1,
                            "mode": "dashboard"
                            if tui_name == "dashboard"
                            else "headless",
                            "port": ports[tui_name],
                            "token_file": "tui.token",
                            "workspace": str(Path(workspace).resolve()),
                        }
                    ),
                )
            if "api" in ports:
                local_interfaces.write_private_file(token_path, token)
                self._token_path = token_path
        except BaseException:
            await self.stop()
            raise

    async def check(self, level: str) -> str | None:
        """Return a contract health reason for owned processes and endpoints."""
        children = tuple(self._children)
        if not children:
            return "runtime_unavailable"
        for child in children:
            if child.process.returncode is not None:
                return "agent_exited"
            if (
                level != "live"
                and await _http_status(child.port, HEALTH_PATHS[child.name]) != 200
            ):
                return "endpoint_unavailable"
            if (
                level != "live"
                and child.tui_token
                and not await tui_ready(child.port, child.tui_token)
            ):
                return "endpoint_unavailable"
        if children != tuple(self._children):
            return "observation_changed"
        return None

    async def stop(self) -> None:
        """Remove the API token, then stop every interface process group."""
        self._remove_private_file("_token_path")
        self._remove_private_file("_tui_descriptor_path")
        self._remove_private_file("_tui_token_path")
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
            if not child.tui_token or await tui_ready(child.port, child.tui_token):
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
        and key
        not in {
            "HERMES_DASHBOARD_SESSION_TOKEN",
            "HERMES_TUI_GATEWAY_URL",
            "HERMES_TUI_SIDECAR_URL",
            "HERMES_DESKTOP",
            "HERMES_SERVE_HEADLESS",
        }
    }
    if name == "api":
        child_environment.update(
            API_SERVER_KEY=token,
            API_SERVER_HOST="127.0.0.1",
            API_SERVER_PORT=str(port),
        )
        return ["gateway", "run"], child_environment
    child_environment["HERMES_DASHBOARD_SESSION_TOKEN"] = token
    if name == "terminal":
        child_environment.pop("HERMES_WEB_DIST", None)
        return [
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--isolated",
        ], child_environment
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


async def tui_ready(port: int, token: str) -> bool:
    """Check authenticated TUI RPC, without creating a conversation or model call."""
    import aiohttp

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=2)
        ) as client:
            async with client.ws_connect(
                f"http://127.0.0.1:{port}/api/ws", params={"token": token}
            ) as ws:
                await ws.send_json(
                    {
                        "jsonrpc": "2.0",
                        "id": "fabric-health",
                        "method": "session.list",
                        "params": {},
                    }
                )
                async for message in ws:
                    if message.type != aiohttp.WSMsgType.TEXT:
                        return False
                    reply = json.loads(message.data)
                    if reply.get("id") == "fabric-health":
                        return "result" in reply
    except (aiohttp.ClientError, TimeoutError, OSError, ValueError):
        return False
    return False
