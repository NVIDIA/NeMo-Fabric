# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supervise the OpenClaw Gateway across abrupt adapter termination."""

from __future__ import annotations

import argparse
import ctypes
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence


_PR_SET_PDEATHSIG = 1
_WINDOWS_START_BYTE = b"\x01"


def _linux_parent_death_signal(expected_parent_pid: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.restype = ctypes.c_int
    if prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    # The parent can exit before prctl() is called. Checking after registration
    # closes that race without starting the Gateway as an orphan.
    if os.getppid() != expected_parent_pid:
        raise RuntimeError("OpenClaw adapter exited before supervision was ready")


def _exit_code(returncode: int) -> int:
    return returncode if returncode >= 0 else 128 - returncode


def _run_linux(
    command: Sequence[str], *, parent_pid: int, shutdown_timeout: float
) -> int:
    _linux_parent_death_signal(parent_pid)
    shutdown_signal: int | None = None
    shutdown_deadline: float | None = None
    forwarded = False

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal shutdown_deadline, shutdown_signal
        if shutdown_signal is None:
            shutdown_signal = signum
            shutdown_deadline = time.monotonic() + shutdown_timeout

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    child = subprocess.Popen(command)
    while True:
        returncode = child.poll()
        if returncode is not None:
            return _exit_code(returncode)
        if shutdown_signal is not None and not forwarded:
            forwarded = True
            signal.signal(shutdown_signal, signal.SIG_IGN)
            os.killpg(os.getpgrp(), shutdown_signal)
        if shutdown_deadline is not None and time.monotonic() >= shutdown_deadline:
            os.killpg(os.getpgrp(), signal.SIGKILL)
        time.sleep(0.05)


def _run_windows(command: Sequence[str]) -> int:
    # The adapter assigns this process to its Job Object before releasing the
    # gate. OpenClaw then inherits membership in that job when it is spawned.
    if sys.stdin.buffer.read(1) != _WINDOWS_START_BYTE:
        return 1
    return _exit_code(subprocess.call(command))


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--shutdown-timeout", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)
    if arguments.command[:1] == ["--"]:
        arguments.command = arguments.command[1:]
    if not arguments.command:
        parser.error("an OpenClaw command is required")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    if sys.platform == "linux":
        return _run_linux(
            arguments.command,
            parent_pid=arguments.parent_pid,
            shutdown_timeout=arguments.shutdown_timeout,
        )
    if os.name == "nt":
        return _run_windows(arguments.command)
    raise RuntimeError("OpenClaw process supervision is unsupported on this platform")


if __name__ == "__main__":
    raise SystemExit(main())
