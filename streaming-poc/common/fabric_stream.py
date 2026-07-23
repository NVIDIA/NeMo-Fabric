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

from nemo_fabric.errors import FabricCapabilityError, FabricStateError
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

    The listener queue is shared across turns, so a turn MUST be **finalized** before
    the next one starts or its unread records leak into the next turn. Finalizing
    (full consumption, ``aclose()``, or the next ``invoke_stream``) waits for the
    turn to complete — there is no in-flight cancellation in v0.1 (the blocking
    native call runs to completion; ``runtime.stop()`` refuses while a turn is
    active) — then drains and discards every record still queued for this turn.

    Early-exit contract: ``aclose()`` (or breaking the ``async for`` and then
    ``await stream.aclose()``) detaches the consumer and finalizes; ``result()``
    stays awaitable afterward. Breaking the loop **without** ``aclose()`` leaves the
    turn un-finalized — the next ``invoke_stream`` raises until you finalize it.
    """

    def __init__(
        self, runtime: Any, listener: AtofStreamListener, *, input: Any, request: Any
    ):
        self._listener = listener
        self._task = asyncio.ensure_future(runtime.invoke(input=input, request=request))
        self._closed = False
        self._finalized = False

    def __aiter__(self) -> "InvokeStream":
        return self

    async def __anext__(self) -> dict[str, Any]:
        q = self._listener.records
        while True:
            if self._closed:
                await self._finalize()
                raise StopAsyncIteration
            if not q.empty():
                return q.get_nowait()
            if self._task.done():
                try:  # let trailing records land after the turn finishes
                    return await asyncio.wait_for(q.get(), _DRAIN)
                except asyncio.TimeoutError:
                    await self._finalize()
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
        """Detach the consumer and finalize (run the turn to completion, then drain
        and discard any unread records so none leak into the next turn)."""
        self._closed = True
        await self._finalize()

    async def _finalize(self) -> None:
        if self._finalized:
            return
        # Wait for the turn to finish so no more records can arrive, then drain and
        # discard everything still queued for it. shield() keeps the invoke task
        # running if *this* coroutine is cancelled, and we re-raise CancelledError
        # rather than swallow it — so a cancelled aclose() propagates and
        # ``result()`` stays valid instead of silently becoming cancelled. Only the
        # invoke's *own* error is suppressed here (it remains retrievable via
        # ``result()``).
        try:
            await asyncio.shield(self._task)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        q = self._listener.records
        while True:
            while not q.empty():
                q.get_nowait()
            # Settle window to catch trailing Relay flushes. NOTE: this is a timing
            # heuristic, not a guarantee — a record arriving after _DRAIN can still
            # leak into the next turn. Production needs a positive turn boundary
            # (per-record turn attribution or a terminal delivery ack), not a timer.
            try:
                await asyncio.wait_for(q.get(), _DRAIN)
            except asyncio.TimeoutError:
                break
        self._finalized = True


class StreamingRuntime:
    """A relay-enabled Runtime that also exposes ``invoke_stream()``.

    Enforces **one active invocation at a time** on its single loopback listener:
    starting a turn while the previous one is not finalized raises ``FabricStateError``
    rather than letting the previous turn's records leak into the new stream.
    """

    def __init__(self, runtime: Any, listener: AtofStreamListener):
        self._runtime = runtime
        self._listener = listener
        self._current: InvokeStream | None = None

    def invoke_stream(self, *, input: Any = None, request: Any = None) -> InvokeStream:
        if self._current is not None and not self._current._finalized:
            raise FabricStateError(
                "an invocation is already active on this runtime; fully consume it "
                "or call `await stream.aclose()` before starting the next turn"
            )
        stream = InvokeStream(self._runtime, self._listener, input=input, request=request)
        self._current = stream
        return stream

    async def aclose(self) -> None:
        # Finalize any dangling turn (drain-through-completion) before stopping —
        # runtime.stop() refuses while a turn is in flight.
        if self._current is not None and not self._current._finalized:
            await self._current.aclose()
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
