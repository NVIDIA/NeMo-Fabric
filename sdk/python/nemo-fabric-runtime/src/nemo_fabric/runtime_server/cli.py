# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""fabric-agent-ctl: strict named commands and one JSON response per command."""

from __future__ import annotations

import asyncio
import io
import os
import select
import stat
import sys
import time
from pathlib import Path
from typing import Any

from nemo_fabric.runtime_server import transport
from nemo_fabric.runtime_server.protocol import (
    CONFIG_OPERATIONS,
    INTERFACE_VERSION,
    OPERATIONS,
    READ_SECONDS,
    REQUEST_LIMIT,
    SOCKET_PATH,
    ProtocolError,
    decode_json,
    decode_object,
    encode_request,
    encode_response,
    envelope,
)
from nemo_fabric.runtime_server.state import RuntimeServer


def read_input(path: str) -> Any:
    try:
        if path == "-":
            if sys.stdin is None or sys.stdin.isatty():
                raise ValueError("stdin is unavailable or a terminal")
            try:
                descriptor = sys.stdin.fileno()
            except io.UnsupportedOperation:
                encoded = sys.stdin.buffer.read(REQUEST_LIMIT + 1)
            else:
                encoded = bytearray()
                deadline = time.monotonic() + READ_SECONDS
                while len(encoded) <= REQUEST_LIMIT:
                    remaining = deadline - time.monotonic()
                    if (
                        remaining <= 0
                        or not select.select([descriptor], [], [], remaining)[0]
                    ):
                        raise ValueError("stdin deadline expired")
                    chunk = os.read(
                        descriptor, min(65536, REQUEST_LIMIT + 1 - len(encoded))
                    )
                    if not chunk:
                        break
                    encoded.extend(chunk)
                encoded = bytes(encoded)
        else:
            descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("expected regular input file")
                encoded = stream.read(REQUEST_LIMIT + 1)
        if len(encoded) > REQUEST_LIMIT:
            raise ValueError("input exceeds limit")
        return decode_json(encoded)
    except (OSError, ValueError, RecursionError) as error:
        raise ProtocolError("invalid_input") from error


def parse_command(arguments: list[str]) -> dict[str, Any]:
    if not arguments or arguments[0] not in OPERATIONS:
        raise ProtocolError()
    operation = arguments[0]
    flags: dict[str, Any] = {}
    index = 1
    while index < len(arguments):
        flag = arguments[index]
        if flag in flags:
            raise ProtocolError()
        if flag in (
            "--agent",
            "--runtime-id",
            "--config",
            "--input",
            "--request",
            "--timeout-seconds",
        ):
            if index + 1 == len(arguments) or arguments[index + 1].startswith("--"):
                raise ProtocolError()
            flags[flag] = arguments[index + 1]
            index += 2
        else:
            raise ProtocolError()
    required = set()
    if operation in ("start", "invoke", "stop"):
        required.add("--agent")
    if operation in ("invoke", "stop"):
        required.add("--runtime-id")
    if operation in CONFIG_OPERATIONS:
        required.add("--config")
    allowed = required.copy()
    if operation == "check":
        allowed.add("--timeout-seconds")
    if operation == "invoke":
        payload_flags = flags.keys() & {"--input", "--request"}
        if len(payload_flags) != 1:
            raise ProtocolError()
        allowed.update(payload_flags)
    if not required <= flags.keys() or not flags.keys() <= allowed:
        raise ProtocolError()
    request = {
        "interface_version": INTERFACE_VERSION,
        "operation": operation,
    }
    if "--timeout-seconds" in flags:
        try:
            request["timeout_seconds"] = float(flags["--timeout-seconds"])
        except ValueError as error:
            raise ProtocolError() from error
    for flag in (
        "--agent",
        "--runtime-id",
    ):
        if flag in flags:
            request[flag[2:].replace("-", "_")] = flags[flag]
    if operation in CONFIG_OPERATIONS:
        request["config"] = read_input(flags["--config"])
    if operation == "invoke":
        flag = next(iter(payload_flags))
        request[flag[2:]] = read_input(flags[flag])
    encode_request(request)
    return request


def main(arguments: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if arguments is None else arguments
    operation = arguments[0] if arguments and arguments[0] in OPERATIONS else "unknown"
    try:
        request = parse_command(arguments)
        if transport.fcntl is None:
            raise ProtocolError("platform_unsupported", "transport", unsupported=True)
        socket_path = Path(os.environ.get("FABRIC_HOST_SOCKET", SOCKET_PATH))
        if operation == "serve":
            os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
            host = RuntimeServer(directory=socket_path.parent)
            asyncio.run(transport.serve(host, socket_path=socket_path))
            return 0
        response = asyncio.run(transport.client(request, socket_path=socket_path))
    except Exception as error:
        if operation == "serve":
            print(
                str(error)
                if isinstance(error, ProtocolError)
                and error.code
                in (
                    "platform_unsupported",
                    "cleanup_unconfirmed",
                    "runtime_busy",
                )
                else "Fabric host could not start or shut down cleanly.",
                file=sys.stderr,
            )
            return 1
        if not isinstance(error, ProtocolError):
            error = ProtocolError()
        response = envelope(operation, error=error)
    encoded = encode_response(response)
    sys.stdout.write(encoded.decode("utf-8"))
    return 0 if decode_object(encoded)["status"] == "succeeded" else 1
