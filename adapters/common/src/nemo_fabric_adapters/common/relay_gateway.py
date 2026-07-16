# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supervise the NeMo Relay CLI gateway used by coding-agent adapters."""

from __future__ import annotations

import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RELAY_HEALTH_TIMEOUT_SECONDS = 10.0
RELAY_STOP_TIMEOUT_SECONDS = 5.0
RELAY_VERSION_TIMEOUT_SECONDS = 5.0


class RelayGatewayError(RuntimeError):
    """NeMo Relay gateway lifecycle failure."""


@dataclass(frozen=True)
class RelayGatewayLaunch:
    """Complete invocation-scoped inputs for launching a Relay gateway."""

    executable: Path
    config_path: Path
    bind: str
    url: str
    log_path: Path


def resolve_relay_command(config_root: Path, value: str | Path) -> Path:
    """Resolve the configured Relay CLI to one absolute executable path."""

    command = Path(value)
    if len(command.parts) == 1:
        resolved = shutil.which(str(command))
    else:
        candidate = command if command.is_absolute() else config_root / command
        resolved = shutil.which(str(candidate.resolve()))
    if resolved is None:
        raise FileNotFoundError("NeMo Relay CLI executable was not found")
    return Path(resolved).resolve()


def find_available_tcp_port(host: str = "127.0.0.1") -> int:
    """Return an available loopback TCP port for an imminent gateway launch."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((host, 0))
        return int(listener.getsockname()[1])


def relay_cli_observability_version(executable: Path) -> int:
    """Require a Relay CLI compatible with Fabric's observability v2 contract."""

    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=RELAY_VERSION_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RelayGatewayError(
            "NeMo Relay CLI version could not be determined"
        ) from error
    match = re.search(r"\b(\d+)\.(\d+)\.(\d+)", completed.stdout)
    if completed.returncode != 0 or match is None:
        raise RelayGatewayError("NeMo Relay CLI version could not be determined")
    major, minor, _ = (int(value) for value in match.groups())
    if (major, minor) < (0, 6):
        raise RelayGatewayError(
            "NeMo Relay 0.6 or newer is required; observability version 1 is unsupported"
        )
    return 2


def wait_for_relay_gateway(
    process: subprocess.Popen[Any],
    health_url: str,
    *,
    timeout: float = RELAY_HEALTH_TIMEOUT_SECONDS,
) -> None:
    """Wait until the Relay health endpoint succeeds or startup fails."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise RelayGatewayError(
                f"NeMo Relay gateway exited with status {returncode} before becoming ready"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=1) as response:
                if 200 <= response.status < 300:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.1)
    raise RelayGatewayError(f"NeMo Relay gateway did not become ready at {health_url}")


def stop_relay_gateway(process: subprocess.Popen[Any]) -> None:
    """Stop a Relay gateway idempotently, escalating when it does not exit."""

    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=RELAY_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=RELAY_STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as error:
            raise RelayGatewayError(
                "NeMo Relay gateway did not stop after kill"
            ) from error


def start_relay_gateway(
    *,
    launch: RelayGatewayLaunch,
    cwd: Path,
) -> subprocess.Popen[Any]:
    """Launch and health-check one invocation-scoped Relay gateway."""

    if not launch.config_path.is_file():
        raise RelayGatewayError("NeMo Relay gateway configuration was not generated")
    launch.log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with launch.log_path.open("wb") as log_stream:
            process = subprocess.Popen(
                [
                    str(launch.executable),
                    "--config",
                    str(launch.config_path),
                    "--bind",
                    launch.bind,
                ],
                cwd=cwd,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except OSError as error:
        raise RelayGatewayError(
            f"NeMo Relay gateway could not start; see {launch.log_path}"
        ) from error

    try:
        wait_for_relay_gateway(process, f"{launch.url.rstrip('/')}/healthz")
    except Exception as error:
        try:
            stop_relay_gateway(process)
        except Exception as stop_error:
            raise RelayGatewayError(
                "NeMo Relay gateway failed to become ready and could not be stopped; "
                f"see {launch.log_path}"
            ) from ExceptionGroup(
                "NeMo Relay gateway startup and cleanup failed",
                [error, stop_error],
            )
        raise RelayGatewayError(
            f"NeMo Relay gateway failed to become ready; see {launch.log_path}"
        ) from error
    return process
