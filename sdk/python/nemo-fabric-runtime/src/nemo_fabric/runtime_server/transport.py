# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, single-request Unix socket connections and exclusive host ownership."""

from __future__ import annotations

import asyncio
import os
import signal
import stat
import threading
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

from nemo_fabric.runtime_server.protocol import (
    CALL_SECONDS,
    LIFECYCLE_OPERATIONS,
    READ_SECONDS,
    REQUEST_LIMIT,
    RESPONSE_LIMIT,
    SHUTDOWN_SECONDS,
    SOCKET_PATH,
    ProtocolError,
    decode_object,
    encode_request,
    encode_response,
    envelope,
    validate_response,
)
from nemo_fabric.runtime_server.state import RuntimeServer

if sys.platform != "win32":
    import fcntl
else:
    fcntl = None


def _sandbox_lifecycle_id() -> str | None:
    try:
        boot_id = UUID(Path("/proc/sys/kernel/random/boot_id").read_text().strip())
        # The parenthesized process name can itself contain spaces and parentheses.
        fields = Path("/proc/1/stat").read_text().rsplit(")", 1)[1].split()
        start_time = int(fields[19])  # /proc/PID/stat field 22, after name and PID.
        if start_time >= 0:
            return f"{boot_id}:{start_time}"
    except (OSError, ValueError, IndexError):
        pass
    return None


async def read_message(reader: asyncio.StreamReader, limit: int) -> dict[str, Any]:
    try:
        encoded = await reader.readline()
        if len(encoded) > limit or not encoded.endswith(b"\n"):
            raise ValueError("incomplete or oversized message")
        return decode_object(encoded)
    except (ValueError, RecursionError) as error:
        raise ProtocolError("invalid_response", "transport") from error


async def client(
    request: dict[str, Any],
    *,
    socket_path: Path = Path(SOCKET_PATH),
    timeout: float = CALL_SECONDS,
) -> dict[str, Any]:
    operation = request["operation"]
    if fcntl is None:
        return envelope(
            operation,
            error=ProtocolError("platform_unsupported", "transport", unsupported=True),
        )
    sent = False
    writer = None
    try:
        encoded = encode_request(request)
        # One deadline covers connection, write, and the complete response.
        async with asyncio.timeout(timeout):
            reader, writer = await asyncio.open_unix_connection(
                socket_path, limit=RESPONSE_LIMIT
            )
            writer.write(encoded)
            sent = True
            await writer.drain()
            response = await read_message(reader, RESPONSE_LIMIT)
            validate_response(response, operation)
            return response
    except (OSError, ValueError, TypeError, KeyError, TimeoutError) as error:
        if not isinstance(error, ProtocolError):
            error = ProtocolError(
                "invalid_response" if sent else "host_unavailable", "transport"
            )
        uncertain = sent and operation in (*LIFECYCLE_OPERATIONS, "invoke")
        return envelope(
            operation,
            None,
            changed=None if uncertain else False,
            error=error,
            effects="unknown" if uncertain else "none",
        )
    finally:
        if writer is not None:
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), 1)
            except (OSError, TimeoutError):
                pass


async def serve(
    host: RuntimeServer,
    *,
    socket_path: Path = Path(SOCKET_PATH),
    stop_event: asyncio.Event | None = None,
    ready_event: asyncio.Event | None = None,
) -> None:
    if fcntl is None:
        raise ProtocolError("platform_unsupported", "transport", unsupported=True)
    # Never unlink this persistent lock file: every host must lock the same inode.
    lock = os.open(
        socket_path.with_suffix(".lock"), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
    )
    owned_socket = None
    marker = socket_path.with_suffix(".unclean")
    owned_marker = None
    unused_lifecycle = False
    shutdown_completed = False
    connections_closing = False
    tasks: set[asyncio.Task] = set()
    writers: set[asyncio.StreamWriter] = set()
    loop = asyncio.get_running_loop()
    watchdog = None
    signals_installed = False
    stop = stop_event if stop_event is not None else asyncio.Event()

    def request_stop() -> None:
        nonlocal watchdog
        host.shutting_down = True
        stop.set()
        if watchdog is None:
            # Native worker threads can outlive asyncio cancellation. Bound process exit;
            # the sandbox supervisor still owns descendant cleanup after forced exit.
            watchdog = threading.Timer(SHUTDOWN_SECONDS, os._exit, args=(1,))
            watchdog.daemon = True
            watchdog.start()

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        tasks.add(task)
        writers.add(writer)
        try:
            # A late accepted callback must not keep final connection cleanup open.
            # During runtime shutdown, requests still receive host_stopping below.
            if connections_closing:
                return
            try:
                async with asyncio.timeout(READ_SECONDS):
                    request = await read_message(reader, REQUEST_LIMIT)
                response = await host.handle(request)
            except (ValueError, TimeoutError):
                response = envelope("unknown", error=ProtocolError())
            # A disconnected bridge does not cancel or replay runtime work.
            writer.write(encode_response(response))
            await asyncio.wait_for(writer.drain(), READ_SECONDS)
        except (OSError, TimeoutError):
            pass
        finally:
            writer.close()
            tasks.discard(task)
            writers.discard(writer)

    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.fchmod(lock, 0o600)
        if not stat.S_ISREG(os.fstat(lock).st_mode):
            raise OSError("host lock is not a regular file")
        lifecycle_id = _sandbox_lifecycle_id()
        if marker.exists() or marker.is_symlink():
            try:
                marker_fd = os.open(marker, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(marker_fd, encoding="ascii") as previous:
                    if not stat.S_ISREG(os.fstat(previous.fileno()).st_mode):
                        raise ValueError("ownership marker is not a regular file")
                    previous_id = previous.read(65)
                boot_id, start_time = previous_id.split(":")
                if (
                    lifecycle_id is None
                    or len(previous_id) > 64
                    or not start_time.isdecimal()
                    or previous_id != f"{UUID(boot_id)}:{int(start_time)}"
                    or previous_id == lifecycle_id
                ):
                    raise ValueError("previous sandbox cleanup is unconfirmed")
            except (OSError, ValueError) as error:
                raise ProtocolError("cleanup_unconfirmed", "transport") from error
            # The marker directory belongs exclusively to this sandbox. A new
            # boot or namespace init process means its previous processes exited.
            marker.unlink()
        for relative in ("tmp", "workspace", "artifacts"):
            (host.directory / relative).mkdir(mode=0o700, parents=True, exist_ok=True)
        if socket_path.exists() or socket_path.is_symlink():
            if not stat.S_ISSOCK(socket_path.lstat().st_mode):
                raise OSError("socket path is occupied by a non-socket file")
            socket_path.unlink()
        # Bind with private permissions from the first instant the socket exists.
        previous_umask = os.umask(0o177)
        try:
            server = await asyncio.start_unix_server(
                handle, path=socket_path, limit=REQUEST_LIMIT
            )
        finally:
            os.umask(previous_umask)
        owned_socket = socket_path.stat()
        marker_fd = os.open(
            marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(marker_fd, "w", encoding="ascii") as ownership:
            ownership.write(lifecycle_id or "")
            ownership.flush()
            owned_marker = os.fstat(ownership.fileno())
        if stop_event is None:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, request_stop)
            signals_installed = True
        if ready_event is not None:
            ready_event.set()
        await stop.wait()
        host.shutting_down = True
        server.close()
        async with asyncio.timeout(SHUTDOWN_SECONDS):
            try:
                await host.shutdown()
            except ProtocolError as error:
                # Emit the fixed diagnostic before cancellation waits for
                # a native worker and the watchdog terminates the server.
                print(str(error), file=sys.stderr, flush=True)
                raise
            connections_closing = True
            for writer in tuple(writers):
                writer.close()
            await server.wait_closed()
        # Retain ownership after any runtime lifecycle. SDK stop alone
        # does not prove that harness descendants exited.
        shutdown_completed = True
        unused_lifecycle = not host.has_used_runtime
    finally:
        if owned_socket is not None:
            host.shutting_down = True
            connections_closing = True
            server.close()
            for writer in tuple(writers):
                writer.close()
            try:
                current = socket_path.lstat()
                if (current.st_dev, current.st_ino) == (
                    owned_socket.st_dev,
                    owned_socket.st_ino,
                ):
                    socket_path.unlink()
            except FileNotFoundError:
                pass
            for task in tuple(tasks):
                task.cancel()
            if tasks:
                await asyncio.gather(*tuple(tasks), return_exceptions=True)
        if signals_installed:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.remove_signal_handler(sig)
        if unused_lifecycle:
            try:
                current = marker.lstat()
                # Inode numbers can be reused after the owned marker is unlinked.
                if (
                    current.st_dev,
                    current.st_ino,
                    current.st_ctime_ns,
                    current.st_size,
                ) == (
                    owned_marker.st_dev,
                    owned_marker.st_ino,
                    owned_marker.st_ctime_ns,
                    owned_marker.st_size,
                ):
                    marker.unlink()
            except FileNotFoundError:
                pass
        if shutdown_completed and watchdog is not None:
            watchdog.cancel()
        os.close(lock)
