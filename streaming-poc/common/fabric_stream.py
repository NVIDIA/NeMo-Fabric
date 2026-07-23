# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`Runtime.invoke_stream()` prototype — sugar over `Runtime.invoke()`.

v0.1 contract (per the streaming decision):
    runtime = await start_streaming_runtime(fabric, config)   # relay enabled
    stream  = runtime.invoke_stream(input="...")
    async for record in stream:      # RAW canonical ATOF record (dict)
        ...
    result = await stream.result()   # RunResult, out of band

Relay's config (incl. the ATOF endpoint) is fixed when the adapter subprocess is
spawned at start_runtime, so the loopback endpoint is injected there; `invoke_stream`
is then pure sugar over `invoke`. No Fabric-specific event normalization in v0.1 —
consumers read raw Relay-generated ATOF.

In the shipped SDK this injection would live inside `Fabric.start_runtime`; here a
factory models it without modifying the SDK.
"""

from __future__ import annotations

import asyncio
from typing import Any

from nemo_fabric.errors import FabricCapabilityError
from nemo_fabric.models import (
    FabricConfig,
    RelayAtofConfig,
    RelayAtofEndpointConfig,
    RelayObservabilityConfig,
    TelemetryConfig,
)

from atof_stream import AtofStreamListener

_DRAIN = 0.25  # seconds to drain trailing socket data after a turn completes


def _relay_enabled(config: FabricConfig) -> bool:
    """Streaming availability signal = Relay in telemetry providers.

    NOTE: distinct from ``RuntimeCapabilities.streaming`` (that flag is native
    adapter progressive output, always False for Relay-based streaming).
    """
    tel = config.telemetry
    if tel is None:
        return False
    if not isinstance(tel, TelemetryConfig):
        tel = TelemetryConfig.model_validate(tel)
    return "relay" in (tel.providers or {})


def _coerce(value: Any, model: type) -> Any:
    if value is None:
        return model()
    return value if isinstance(value, model) else model.model_validate(value)


def with_atof_endpoint(config: FabricConfig, url: str) -> FabricConfig:
    """Deep copy with a loopback ndjson ATOF endpoint appended (preserves sinks)."""
    cfg = config.model_copy(deep=True)
    obs = _coerce(
        cfg.relay.observability if cfg.relay else None, RelayObservabilityConfig
    )
    atof = _coerce(obs.atof, RelayAtofConfig)
    atof.enabled = True
    atof.endpoints = list(atof.endpoints or []) + [
        RelayAtofEndpointConfig(url=url, transport="ndjson")
    ]
    obs.atof = atof
    cfg.enable_relay(observability=obs)
    return cfg


class InvokeStream:
    """Async-iterable of raw ATOF records for one invocation; ``result()`` = RunResult.

    Early-exit contract: ``aclose()`` (or breaking the ``async for``) detaches the
    consumer — iteration stops and buffered records are discarded — but does NOT
    interrupt the turn. ``invoke`` is a blocking native call on a worker thread and
    runs to completion; ``result()`` stays awaitable after detaching. There is **no**
    in-flight cancellation in v0.1: ``runtime.stop()`` raises ``FabricStateError``
    while a turn is active, so a turn can only be torn down after it finishes.
    """

    def __init__(
        self, runtime: Any, listener: AtofStreamListener, *, input: Any, request: Any
    ):
        self._listener = listener
        self._task = asyncio.ensure_future(runtime.invoke(input=input, request=request))
        self._closed = False

    def __aiter__(self) -> "InvokeStream":
        return self

    async def __anext__(self) -> dict[str, Any]:
        q = self._listener.records
        while True:
            if self._closed:
                raise StopAsyncIteration
            if not q.empty():
                return q.get_nowait()
            if self._task.done():
                try:  # let trailing records land after the turn finishes
                    return await asyncio.wait_for(q.get(), _DRAIN)
                except asyncio.TimeoutError:
                    raise StopAsyncIteration
            getter = asyncio.ensure_future(q.get())
            await asyncio.wait(
                {getter, self._task}, return_when=asyncio.FIRST_COMPLETED
            )
            if getter.done() and not getter.cancelled():
                return getter.result()
            getter.cancel()

    async def result(self) -> Any:
        return await self._task

    async def aclose(self) -> None:
        """Detach the consumer: stop iteration, leave the turn running.

        Does NOT cancel the invoke task — the blocking native turn runs to
        completion and ``result()`` stays awaitable. (Cancelling the future would
        not stop the native call anyway; it would only break ``result()``.)
        """
        self._closed = True


class StreamingRuntime:
    """A relay-enabled Runtime that also exposes ``invoke_stream()``."""

    def __init__(self, runtime: Any, listener: AtofStreamListener):
        self._runtime = runtime
        self._listener = listener

    def invoke_stream(self, *, input: Any = None, request: Any = None) -> InvokeStream:
        return InvokeStream(self._runtime, self._listener, input=input, request=request)

    async def aclose(self) -> None:
        try:
            await self._runtime.stop()
        finally:
            await self._listener.close()

    async def __aenter__(self) -> "StreamingRuntime":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()


async def start_streaming_runtime(
    fabric: Any, config: FabricConfig, *, base_dir: str | None = None
) -> StreamingRuntime:
    """Sugar over ``Fabric.start_runtime``: inject the loopback endpoint, start, wrap."""
    if not _relay_enabled(config):
        raise FabricCapabilityError(
            "streaming requires relay telemetry enabled", capability="streaming"
        )
    listener = await AtofStreamListener().start()
    cfg = with_atof_endpoint(config, listener.url)
    runtime = await fabric.start_runtime(cfg, base_dir=base_dir)
    return StreamingRuntime(runtime, listener)
