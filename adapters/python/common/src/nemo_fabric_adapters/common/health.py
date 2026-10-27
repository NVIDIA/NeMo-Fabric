# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticated loopback transport for adapter health probes."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from nemo_fabric_adapter_contract.models import AdapterCheckRequest, AdapterCheckResult

_LIMIT = 16 * 1024


def unobserved_health(reason: str) -> dict[str, Any]:
    """Return an unknown health result with the given reason code."""
    return {
        "outcome": "unknown",
        "reason_code": reason,
        "activity": "unknown",
        "source": "fabric.adapter",
        "observed_at": None,
    }


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate health field")
        value[key] = item
    return value


class _HealthServer:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.server: asyncio.Server | None = None
        self.writers: set[asyncio.StreamWriter] = set()
        self.tasks: set[asyncio.Task] = set()
        self.closed = False
        self.owns_endpoint = False

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.owns_endpoint:
                with suppress(FileNotFoundError):
                    self.path.unlink()
        finally:
            if self.server is not None:
                self.server.close()
            for writer in tuple(self.writers):
                writer.close()
            tasks = tuple(self.tasks)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.server is not None:
                await self.server.wait_closed()


async def serve_health(
    runtime_id: str,
    directory: str | Path,
    probe: Callable[[str], Awaitable[dict[str, Any]]],
) -> _HealthServer:
    """Publish an authenticated loopback endpoint for one adapter host.

    Requests and responses use HMAC-SHA256 signatures over the direction,
    nonce, and payload. Each deadline applies to its observation coroutine.
    """
    token = secrets.token_hex(32)
    state = _HealthServer(Path(directory) / "health.endpoint.json")

    def proof(direction: str, nonce: str, payload: str) -> str:
        return hmac.new(
            token.encode(), f"{direction}\n{nonce}\n{payload}".encode(), hashlib.sha256
        ).hexdigest()

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        state.tasks.add(task)
        try:
            if state.closed or len(state.writers) >= 64:
                return
            state.writers.add(writer)
            try:
                async with asyncio.timeout(3):
                    line = await reader.readline()
                if not line.endswith(b"\n") or len(line) > _LIMIT:
                    return
                envelope = json.loads(line, object_pairs_hook=_unique)
                if not isinstance(envelope, dict) or set(envelope) != {
                    "nonce",
                    "payload",
                    "proof",
                }:
                    return
                nonce, payload, signature = (
                    envelope[key] for key in ("nonce", "payload", "proof")
                )
                if (
                    not isinstance(nonce, str)
                    or not re.fullmatch(r"[a-f0-9]{64}", nonce)
                    or not isinstance(payload, str)
                    or not isinstance(signature, str)
                    or not re.fullmatch(r"[a-f0-9]{64}", signature)
                    or not hmac.compare_digest(
                        proof("request", nonce, payload), signature
                    )
                ):
                    return
                request = AdapterCheckRequest.from_mapping(
                    json.loads(payload, object_pairs_hook=_unique)
                )
            except Exception:
                return
            if request.runtime_id != runtime_id:
                result = unobserved_health("runtime_unavailable")
            else:
                deadline = asyncio.get_running_loop().time() + request.timeout_seconds
                try:
                    async with asyncio.timeout(request.timeout_seconds):
                        result = AdapterCheckResult.from_mapping(
                            await probe(request.level)
                        ).to_mapping()
                    if asyncio.get_running_loop().time() >= deadline:
                        result = unobserved_health("probe_timeout")
                except TimeoutError:
                    result = unobserved_health("probe_timeout")
                except Exception:
                    result = unobserved_health("probe_error")
            payload = json.dumps(result, separators=(",", ":"), allow_nan=False)
            response = {"payload": payload, "proof": proof("response", nonce, payload)}
            writer.write(json.dumps(response).encode() + b"\n")
            await writer.drain()
        except (OSError, ValueError):
            pass
        finally:
            state.writers.discard(writer)
            writer.close()
            with suppress(OSError):
                await writer.wait_closed()
            state.tasks.discard(task)

    state.server = await asyncio.start_server(handle, "127.0.0.1", 0, limit=_LIMIT)
    try:
        port = state.server.sockets[0].getsockname()[1]
        descriptor = {
            "protocol_version": "fabric.check/v1alpha1",
            "runtime_id": runtime_id,
            "port": port,
            "token": token,
        }
        fd = os.open(state.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        state.owns_endpoint = True
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(descriptor, stream)
    except BaseException:
        await state.close()
        raise
    return state
