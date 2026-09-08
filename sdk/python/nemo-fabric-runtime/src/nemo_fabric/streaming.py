# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA NeMo Relay streaming support for the NVIDIA NeMo Fabric Python SDK."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from contextlib import suppress
from typing import Any

from nemo_fabric._collector_client import _AtofCollectorClient
from nemo_fabric.errors import FabricConfigError
from nemo_fabric.models import (
    FabricConfig,
    RelayAtofConfig,
    RelayAtofFileSinkConfig,
    RelayAtofStreamSinkConfig,
    RelayConfig,
    RelayObservabilityConfig,
)
from nemo_fabric.types import RunResult

_STREAM_SINK_NAME = "nemo-fabric-stream"


class InvokeStream:
    """Async iterator of raw ATOF records for one runtime invocation.

    Consume the final normalized result separately with :meth:`result`. If
    iteration stops early, call :meth:`aclose` before starting another turn.
    """

    def __init__(
        self,
        invoke: Coroutine[Any, Any, RunResult],
        collector_client: _AtofCollectorClient,
        *,
        request_id: str,
        registration_ready: asyncio.Event,
        on_finalize: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """lazydocs: ignore"""

        self._collector_client = collector_client
        self._request_id = request_id
        self._registration_ready = registration_ready
        self._records: AsyncGenerator[dict[str, Any], None] | None = None
        self._next_record_task: asyncio.Task[dict[str, Any]] | None = None
        self._closed = False
        self._finalized = False
        self._on_finalize = on_finalize
        self._finalize_lock = asyncio.Lock()
        try:
            self._task = asyncio.create_task(invoke)
        except BaseException:
            invoke.close()
            raise

    def __aiter__(self) -> InvokeStream:
        """Return this stream as its asynchronous iterator."""

        if not self._finalized:
            self._records_iterator()
        return self

    async def __anext__(self) -> dict[str, Any]:
        """Return the next raw ATOF record."""

        if self._closed or self._finalized:
            await self._finalize()
            raise StopAsyncIteration
        if not await self._wait_for_registration():
            await self._finalize()
            raise StopAsyncIteration
        try:
            return await self._next_record()
        except StopAsyncIteration:
            await self._finalize()
            raise
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            try:
                await self._finish_stream()
            except Exception as cleanup_error:
                error.add_note(
                    f"ATOF collector stream cleanup failed: {cleanup_error}"
                )
            raise

    async def result(self) -> RunResult:
        """Return the terminal normalized result without adding it to the stream."""

        return await asyncio.shield(self._task)

    async def aclose(self) -> None:
        """Stop iteration and drain this turn without cancelling the invocation."""

        self._closed = True
        await self._finalize()

    async def _finalize(self) -> None:
        async with self._finalize_lock:
            if self._finalized:
                return
            stream_error: Exception | None = None
            if await self._wait_for_registration():
                try:
                    while True:
                        await self._next_record()
                except StopAsyncIteration:
                    pass
                except Exception as error:
                    stream_error = error

            if stream_error is not None:
                try:
                    await self._finish_stream()
                except Exception as cleanup_error:
                    stream_error.add_note(
                        f"ATOF collector stream cleanup failed: {cleanup_error}"
                    )
            try:
                await asyncio.shield(self._task)
            except asyncio.CancelledError:
                if not self._task.cancelled():
                    raise
            except Exception:
                pass
            if not self._finalized:
                await self._finish_stream()
            if stream_error is not None:
                raise stream_error

    def _records_iterator(self) -> AsyncGenerator[dict[str, Any], None]:
        if self._records is None:
            self._records = self._collector_client.stream(self._request_id)
        return self._records

    async def _next_record(self) -> dict[str, Any]:
        if self._next_record_task is None:
            self._next_record_task = asyncio.create_task(
                anext(self._records_iterator())
            )
        task = self._next_record_task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                self._next_record_task = None

    async def _wait_for_registration(self) -> bool:
        if self._registration_ready.is_set():
            return True
        waiter = asyncio.create_task(self._registration_ready.wait())
        try:
            await asyncio.wait(
                {waiter, self._task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            return self._registration_ready.is_set()
        finally:
            if not waiter.done():
                waiter.cancel()
            with suppress(asyncio.CancelledError):
                await waiter

    async def _finish_stream(self) -> None:
        if self._finalized:
            return
        try:
            if self._next_record_task is not None:
                self._next_record_task.cancel()
                with suppress(asyncio.CancelledError, StopAsyncIteration):
                    await self._next_record_task
                self._next_record_task = None
            if self._records is not None:
                await self._records.aclose()
                self._records = None
            if self._on_finalize is not None:
                await self._on_finalize()
        finally:
            self._finalized = True


def _relay_enabled(config: FabricConfig) -> bool:
    telemetry = config.telemetry
    return telemetry is not None and "relay" in telemetry.providers


def _sink_name(
    sink: RelayAtofFileSinkConfig | RelayAtofStreamSinkConfig | dict[str, Any],
) -> str | None:
    name = sink.get("name") if isinstance(sink, dict) else getattr(sink, "name", None)
    return name if isinstance(name, str) else None


def _configured_stream_sink(
    config: FabricConfig,
) -> RelayAtofStreamSinkConfig | None:
    relay = config.relay
    observability = relay.observability if isinstance(relay, RelayConfig) else None
    atof = (
        observability.atof
        if isinstance(observability, RelayObservabilityConfig)
        else None
    )
    if not isinstance(atof, RelayAtofConfig) or not atof.enabled:
        return None
    matches = [
        sink for sink in atof.sinks or () if _sink_name(sink) == _STREAM_SINK_NAME
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise FabricConfigError(
            "Relay streaming requires exactly one nemo-fabric-stream sink"
        )
    sink = matches[0]
    if not isinstance(sink, RelayAtofStreamSinkConfig):
        raise FabricConfigError(
            "Relay sink nemo-fabric-stream must be a stream sink"
        )
    if sink.transport not in {"http_post", "ndjson"}:
        raise FabricConfigError(
            "Relay sink nemo-fabric-stream must use http_post or ndjson"
        )
    return sink


def _with_stream_sink(config: FabricConfig, url: str) -> FabricConfig:
    copied = config.model_copy(deep=True)
    if copied.relay is None:
        relay = RelayConfig()
    elif isinstance(copied.relay, RelayConfig):
        relay = copied.relay
    else:
        relay = RelayConfig.model_validate(copied.relay)

    if relay.observability is None:
        observability = RelayObservabilityConfig()
    elif isinstance(relay.observability, RelayObservabilityConfig):
        observability = relay.observability
    else:
        observability = RelayObservabilityConfig.model_validate(relay.observability)

    if observability.atof is None:
        atof = RelayAtofConfig()
    elif isinstance(observability.atof, RelayAtofConfig):
        atof = observability.atof
    else:
        atof = RelayAtofConfig.model_validate(observability.atof)

    if atof.enabled:
        sinks = [
            sink for sink in atof.sinks or () if _sink_name(sink) != _STREAM_SINK_NAME
        ]
    else:
        atof = RelayAtofConfig(enabled=True)
        sinks = []
    sinks.append(
        RelayAtofStreamSinkConfig(
            name=_STREAM_SINK_NAME,
            url=url,
            transport="ndjson",
        )
    )
    atof.sinks = sinks
    observability.atof = atof
    relay.observability = observability
    copied.relay = relay
    return copied
