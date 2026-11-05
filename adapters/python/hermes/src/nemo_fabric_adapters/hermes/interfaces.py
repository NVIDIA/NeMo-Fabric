# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve Hermes's API server and dashboard as children of a managed runtime."""

from __future__ import annotations

import asyncio
import http.client
import json
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

from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common import local_interfaces
from nemo_fabric_adapters.common import utils as common_utils

HEALTH_PATHS = {"api": "/health", "dashboard": "/api/status"}
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


class _OwnershipScanChanged(Exception):
    """A fork or exit made an empty process snapshot inconclusive."""


@dataclass
class _Child:
    name: str
    port: int
    process: asyncio.subprocess.Process
    owner: str | None = field(default=None, repr=False)
    created_at: float = 0.0


class InterfaceChildren:
    """Start, track and stop one runtime's interface processes."""

    def __init__(self) -> None:
        self._children: list[_Child] = []
        self._token_path: Path | None = None
        self._ownership_path: Path | None = None

    async def start(
        self,
        ports: dict[str, int],
        *,
        hermes_home: Path,
        token_path: Path,
        timeout: float,
        environment: dict[str, str],
    ) -> None:
        """Start the enabled interfaces and write the API token once all pass health."""
        if not POSIX:
            raise lifecycle.LifecycleError(
                "hermes_interfaces_unsupported_platform",
                "Hermes interfaces require a POSIX host",
                metadata={"field": "harness.settings.interfaces"},
            )
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
            for name, port in ports.items():
                arguments, child_environment = _child_command(
                    name, port, token=token, web_dist=web_dist, environment=environment
                )
                owner = owners[name]
                child_environment[_INTERFACE_OWNER_ENV] = owner
                with open(log_dir / f"fabric-interface-{name}.log", "ab") as log:
                    process = await asyncio.create_subprocess_exec(
                        sys.executable,
                        "-m",
                        "hermes_cli.main",
                        *arguments,
                        env=child_environment,
                        stdin=asyncio.subprocess.DEVNULL,
                        stdout=log,
                        stderr=asyncio.subprocess.STDOUT,
                        start_new_session=True,
                        pass_fds=inherited_lock,
                    )
                self._children.append(_Child(name, port, process, owner, created_at))
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
        token_path, self._token_path = self._token_path, None
        if token_path is not None:
            token_path.unlink(missing_ok=True)
        # Chat clients can recover a gateway as soon as its connection closes.
        # Finish dashboard-owned work while the API gateway is still available.
        phases = [
            [child for child in self._children if child.name == name]
            for name in ("dashboard", "api")
        ]
        try:
            for phase in phases:
                if phase:
                    grace = (
                        DASHBOARD_STOP_GRACE_SECONDS
                        if phase[0].name == "dashboard" and phases[1]
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
            if cleanup_error is not None:
                raise lifecycle.LifecycleError(
                    "hermes_interface_cleanup_unconfirmed",
                    "Hermes interface cleanup could not be confirmed",
                ) from cleanup_error
        if self._ownership_path is not None:
            self._ownership_path.unlink(missing_ok=True)
            self._ownership_path = None
        self._children = []


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
