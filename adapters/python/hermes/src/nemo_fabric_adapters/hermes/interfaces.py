# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own Hermes API and shared browser/terminal TUI backend processes."""

from __future__ import annotations

import asyncio
import http.client
import json
import logging
import math
import os
import secrets
import signal
import stat
import sys
import time
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
# Leave time for gateway draining within the adapter host stop deadline.
DASHBOARD_STOP_GRACE_SECONDS = 2.0
_INTERFACE_OWNER_ENV = "FABRIC_HERMES_INTERFACE_OWNER"
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


class _OwnershipScanChanged(Exception):
    """A fork or exit made an empty process snapshot inconclusive."""


@dataclass
class _Child:
    name: str
    port: int
    process: asyncio.subprocess.Process
    tui_token: str | None = field(default=None, repr=False)
    owner: str | None = field(default=None, repr=False)
    created_at: float = 0.0


class InterfaceChildren:
    """Start, track and stop one runtime's interface processes."""

    def __init__(self) -> None:
        self._children: list[_Child] = []
        self._token_path: Path | None = None
        self._agent_config_path: Path | None = None
        self._tui_token_path: Path | None = None
        self._tui_descriptor_path: Path | None = None
        self._ownership_path: Path | None = None

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
        """Start each backend once and publish credentials after all are ready."""
        require_supported_platform()
        require_released_home(hermes_home)
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
        inherited_lock: tuple[int, ...] = ()
        if "FABRIC_STATE_LOCK_FD" in environment:
            try:
                descriptor = int(environment["FABRIC_STATE_LOCK_FD"])
                if descriptor < 3:
                    raise ValueError("invalid writer lock")
                os.fstat(descriptor)
                inherited_lock = (descriptor,)
            except (ValueError, OSError) as error:
                raise lifecycle.LifecycleError(
                    "session_lock_invalid", "Inherited state lock is unavailable"
                ) from error
        owners = {name: secrets.token_urlsafe(32) for name in ports}
        created_at = time.time()
        if owners:
            ownership_path = hermes_home / "interfaces" / "process-owners.json"
            local_interfaces.write_private_file(
                ownership_path,
                json.dumps({owner: created_at for owner in owners.values()}),
            )
            self._ownership_path = ownership_path
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
                owner = owners[name]
                child_environment[_INTERFACE_OWNER_ENV] = owner
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
                        pass_fds=inherited_lock,
                    )
                self._children.append(
                    _Child(
                        name,
                        port,
                        process,
                        tui_token=tui_token if name != "api" else None,
                        owner=owner,
                        created_at=created_at,
                    )
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
        # Finish chat recovery before stopping the API gateway.
        phases = [
            [child for child in self._children if child.name != "api"],
            [child for child in self._children if child.name == "api"],
        ]
        try:
            for phase in phases:
                if phase:
                    grace = (
                        DASHBOARD_STOP_GRACE_SECONDS
                        if phase[0].name != "api" and phases[1]
                        else STOP_GRACE_SECONDS
                    )
                    await _stop_children(phase, grace)
        finally:
            # Cancellation must also stop detached chat and recovery processes.
            cleanup_error = None
            for phase in phases:
                try:
                    _signal_owned(phase, signal.SIGKILL)
                except Exception as error:
                    cleanup_error = cleanup_error or error
                for child in phase:
                    try:
                        _signal_child(child, signal.SIGKILL)
                    except OSError as error:
                        cleanup_error = cleanup_error or error
            self._remove_private_file("_agent_config_path")
            if cleanup_error is not None:
                raise lifecycle.LifecycleError(
                    "hermes_interface_cleanup_unconfirmed",
                    "Hermes interface cleanup could not be confirmed",
                ) from cleanup_error
        if self._ownership_path is not None:
            self._ownership_path.unlink(missing_ok=True)
            self._ownership_path = None
        self._children = []

    def _remove_private_file(self, attribute: str) -> None:
        """Remove a private file, retaining its path for retry after a failure."""
        path = getattr(self, attribute)
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                # Retain the path for retry; cleanup must not prevent child stop.
                LOGGER.exception("Could not remove Hermes interface file %s", path)
            else:
                setattr(self, attribute, None)


def require_released_home(hermes_home: Path) -> None:
    """Refuse native state access while an earlier interface still owns work."""
    path = hermes_home / "interfaces" / "process-owners.json"
    try:
        if path.parent.is_symlink():
            raise OSError("invalid interface ownership directory")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
    except FileNotFoundError:
        return
    except OSError as error:
        raise lifecycle.LifecycleError(
            "hermes_interface_cleanup_unconfirmed",
            "Earlier Hermes interface ownership cannot be inspected",
        ) from error
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_size > 4096
                or info.st_nlink != 1
            ):
                raise ValueError("invalid ownership record")
            owners = json.loads(stream.read(4097))
        if (
            not isinstance(owners, dict)
            or not 1 <= len(owners) <= 2
            or any(
                not isinstance(owner, str)
                or len(owner) != 43
                or type(issued) not in (int, float)
                or not math.isfinite(issued)
                or issued <= 0
                or issued > time.time() + 1
                for owner, issued in owners.items()
            )
        ):
            raise ValueError("invalid interface owners")
        deadline = time.monotonic() + 0.5
        while True:
            try:
                owned = _owned_processes(owners)
                break
            except _OwnershipScanChanged:
                if time.monotonic() >= deadline:
                    raise lifecycle.LifecycleError(
                        "hermes_interface_cleanup_unconfirmed",
                        "Earlier Hermes interface ownership did not stabilize",
                    )
                time.sleep(0.025)
        if owned:
            raise lifecycle.LifecycleError(
                "hermes_interface_cleanup_unconfirmed",
                "Earlier Hermes interface processes still own work",
            )
        path.unlink()
    except lifecycle.LifecycleError:
        raise
    except (OSError, ValueError, TypeError, OverflowError) as error:
        raise lifecycle.LifecycleError(
            "hermes_interface_cleanup_unconfirmed",
            "Earlier Hermes interface cleanup cannot be confirmed",
        ) from error


def _signal_child(child: _Child, signum: signal.Signals) -> None:
    # A reaped leader's PID can be reused. Owned descendants are signaled
    # separately with their recorded process identity.
    if child.process.returncode is None:
        common_utils.signal_process_group(child.process.pid, signum)


def _owned_processes(owners: dict[str, float]) -> list[Any]:
    # Hermes installs psutil for native process management. The private marker
    # follows detached bootstrap children after their parents have exited.
    if not owners:
        return []
    import psutil

    owned = []
    snapshot = set(psutil.pids())
    earliest = min(owners.values()) - 1
    for process in psutil.process_iter(
        ["create_time", "uids", "status"], ad_value=None
    ):
        user = process.info["uids"]
        created = process.info["create_time"]
        if process.info["status"] == psutil.STATUS_ZOMBIE or (
            user is not None and user.real != os.getuid()
        ):
            continue
        for attempt in range(3):
            try:
                if process.environ().get(_INTERFACE_OWNER_ENV) in owners:
                    owned.append(process)
                break
            except psutil.NoSuchProcess:
                break
            except psutil.AccessDenied as error:
                try:
                    if not process.is_running() or process.status() in {
                        psutil.STATUS_ZOMBIE,
                        psutil.STATUS_DEAD,
                    }:
                        break
                except psutil.NoSuchProcess:
                    break
                if created is not None and created < earliest:
                    break
                if attempt < 2:
                    # An exiting or execing process can briefly hide its environment.
                    time.sleep(0.025)
                    continue
                raise lifecycle.LifecycleError(
                    "hermes_interface_cleanup_unconfirmed",
                    "Hermes interface process ownership cannot be inspected",
                ) from error
    if not owned and snapshot != set(psutil.pids()):
        raise _OwnershipScanChanged
    return owned


def _signal_owned(children: list[_Child], signum: signal.Signals) -> None:
    owners = {child.owner: child.created_at for child in children if child.owner}
    if not owners:
        return
    import psutil

    try:
        processes = _owned_processes(owners)
    except _OwnershipScanChanged:
        return
    for process in processes:
        try:
            # psutil checks the recorded creation time before signaling a PID.
            process.send_signal(signum)
        except psutil.NoSuchProcess:
            pass


async def _stop_children(children: list[_Child], grace: float) -> None:
    for child in children:
        _signal_child(child, signal.SIGTERM)
    _signal_owned(children, signal.SIGTERM)
    if await _wait_stopped(children, grace):
        return
    _signal_owned(children, signal.SIGKILL)
    for child in children:
        _signal_child(child, signal.SIGKILL)
    if not await _wait_stopped(children, 0.5):
        raise lifecycle.LifecycleError(
            "hermes_interface_cleanup_unconfirmed",
            "Hermes interface processes did not exit",
        )


async def _wait_stopped(children: list[_Child], timeout: float) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        try:
            owned = _owned_processes(
                {child.owner: child.created_at for child in children if child.owner}
            )
        except _OwnershipScanChanged:
            owned = None
        if owned == [] and all(
            child.process.returncode is not None for child in children
        ):
            return True
        await asyncio.sleep(0.025)
    return False


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


def _web_dist() -> Path:
    """Resolve built browser assets without installing or building a frontend."""
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
    """Wait for native health and TUI authentication within the shared budget."""
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
        async with (
            asyncio.timeout(2),
            aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as client,
        ):
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
