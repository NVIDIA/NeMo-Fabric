# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime lifecycle support for the NVIDIA NeMo Fabric Python SDK."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from copy import deepcopy
from enum import Enum
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import ValidationError

from nemo_fabric._collector_client import _AtofCollectorClient
from nemo_fabric.errors import (
    FabricCapabilityError,
    FabricConfigError,
    FabricError,
    FabricRuntimeError,
    FabricStateError,
)
from nemo_fabric.models import RunRequest
from nemo_fabric.openai_streaming import OpenAIInvokeStream
from nemo_fabric.streaming import InvokeStream
from nemo_fabric.types import RunPlan, RunResult, RuntimeHandle


logger = logging.getLogger(__name__)

_PI_ADAPTER_ID = "nvidia.fabric.pi"
_PI_STREAM_CORRELATION_MODE = "pi_turn_window"


class RuntimeStatus(str, Enum):
    """Lifecycle state of a runtime.

    ``ACTIVE`` accepts invocations, ``STOPPED`` has released its runtime, and
    ``FAILED`` records a lifecycle failure that prevents further invocations
    but still permits cleanup.
    """

    ACTIVE = "active"
    STOPPED = "stopped"
    FAILED = "failed"


class _RuntimeStream(Protocol):
    """Internal stream state shared by Relay and native OpenAI streams."""

    _finalized: bool
    _task: asyncio.Task[Any]

    async def aclose(self) -> None: ...


class Runtime:
    """One logical, stateful harness execution.

    Create runtimes with ``Fabric.start_runtime()`` rather than calling the
    constructor. A runtime serializes invocations and preserves adapter-owned
    harness state across turns. Use it as an asynchronous context manager to
    stop the runtime on exit.

    Runtime-scoped overrides are recursively merged with invocation overrides;
    invocation values win.
    """

    def __init__(
        self,
        *,
        client: Any,
        plan: RunPlan | Mapping[str, Any],
        runtime: RuntimeHandle | Mapping[str, Any],
        overrides: Mapping[str, Any] | None = None,
        collector: AsyncExitStack | None = None,
        collector_client: _AtofCollectorClient | None = None,
    ) -> None:
        """lazydocs: ignore"""

        self._plan = plan if isinstance(plan, RunPlan) else RunPlan.from_mapping(plan)
        self._runtime = (
            runtime if isinstance(runtime, RuntimeHandle) else RuntimeHandle.from_mapping(runtime)
        )
        self._client = client
        self._overrides = _json_mapping(overrides, "runtime overrides")
        self._messages: list[Any] = []
        self._invocations: list[dict[str, Any]] = []
        self._status = RuntimeStatus.ACTIVE
        self._current_task: asyncio.Task[Any] | None = None
        self._current_stream: _RuntimeStream | None = None
        self._collector = collector
        self._collector_client = collector_client
        self._registered_requests: set[str] = set()
        self._registration_tokens: dict[str, str] = {}
        self._stream_outcomes_finished: set[str] = set()
        self._stream_finalizers_finished: set[str] = set()
        self._stream_cleanup_failed: set[str] = set()
        self._closing = False

    @property
    def status(self) -> RuntimeStatus:
        """Return the current ``ACTIVE``, ``STOPPED``, or ``FAILED`` state."""

        return self._status

    @property
    def messages(self) -> list[Any]:
        """Return a deep copy of the latest harness-provided message history."""

        return deepcopy(self._messages)

    @property
    def invocations(self) -> list[dict[str, Any]]:
        """Return copied request, runtime, and invocation IDs for completed turns."""

        return deepcopy(self._invocations)

    @property
    def handle(self) -> RuntimeHandle:
        """Return a detached snapshot of the runtime handle."""

        return RuntimeHandle.from_mapping(self._runtime.to_mapping())

    @property
    def runtime_id(self) -> str:
        """Return the unique identifier for this started runtime lifecycle."""

        return self._runtime.runtime_id

    @property
    def supports_streaming(self) -> bool:
        """Return whether NVIDIA NeMo Relay ATOF streaming is enabled."""

        return self._collector_client is not None

    @property
    def supports_openai_streaming(self) -> bool:
        """Return whether the selected adapter implements native OpenAI streaming."""

        adapter_descriptor = self._plan.get("adapter_descriptor")
        descriptor = (
            adapter_descriptor.get("descriptor")
            if isinstance(adapter_descriptor, Mapping)
            else None
        )
        descriptor_capabilities = (
            descriptor.get("capabilities") if isinstance(descriptor, Mapping) else None
        )
        return (
            self._plan.capabilities.streaming
            and isinstance(descriptor_capabilities, Mapping)
            and descriptor_capabilities.get("streaming") is True
        )

    async def invoke(
        self,
        *,
        input: Any = None,
        request: RunRequest | None = None,
    ) -> RunResult:
        """Run one turn on this runtime.

        ``input`` and ``request`` are mutually exclusive. Runtime overrides are
        merged below invocation overrides from ``RunRequest``. Concurrent turns
        on the same runtime are rejected.

        Args:
            input: JSON-compatible turn input.
            request: Complete validated ``RunRequest``.

        Returns:
            The normalized ``RunResult`` for this turn.

        Raises:
            FabricConfigError: If request fields conflict or are not
                JSON-compatible.
            FabricStateError: If the runtime is not active, is stopping, or is
                already running a turn.
            FabricNativeUnavailableError: If the native extension is missing.
            FabricRuntimeError: If native invocation fails before returning a
                normalized result.
        """

        self._ensure_no_active_stream()
        self._ensure_invocable()
        payload = _run_request_payload(input=input, request=request)
        if self._uses_pi_stream_correlation():
            self._reserve_request_registration(payload["request_id"])
            current_task = asyncio.current_task()
            self._current_task = current_task
            try:
                return await self._invoke_registered_payload(
                    payload,
                    registration_ready=None,
                    capture_records=False,
                    invocation_claimed=True,
                    registration_reserved=True,
                )
            finally:
                if self._current_task is current_task:
                    self._current_task = None
        return await self._invoke_payload(payload)

    async def _invoke_payload(
        self,
        payload: dict[str, Any],
        *,
        openai_stream_transport: Mapping[str, Any] | None = None,
        absorb_result: bool = True,
        invocation_claimed: bool = False,
    ) -> RunResult:
        if not invocation_claimed:
            self._ensure_invocable()
            self._current_task = asyncio.current_task()
        try:
            merged = _merge_overrides(self._overrides, payload.get("overrides"))
            if merged:
                payload["overrides"] = merged
            else:
                payload.pop("overrides", None)
            native_result: dict[str, Any] | None = None
            try:
                native = self._client._require_native_module("invoke")

                def invoke() -> dict[str, Any]:
                    nonlocal native_result
                    plan_json = json.dumps(self._plan.to_mapping())
                    runtime_json = json.dumps(self._runtime.to_mapping())
                    request_json = json.dumps(payload)
                    if openai_stream_transport is None:
                        encoded = native.invoke_runtime(
                            plan_json,
                            runtime_json,
                            request_json,
                        )
                    else:
                        encoded = native.invoke_openai_stream(
                            plan_json,
                            runtime_json,
                            request_json,
                            json.dumps(dict(openai_stream_transport)),
                        )
                    native_result = json.loads(encoded)
                    return native_result

                result = await _call_blocking(invoke)
                typed_result = RunResult.from_mapping(result)
            except asyncio.CancelledError:
                if absorb_result and native_result is not None:
                    try:
                        self._absorb(RunResult.from_mapping(native_result))
                    except Exception:
                        pass
                stopped = False

                def stop_after_cancel() -> Any:
                    nonlocal stopped
                    result = json.loads(
                        native.stop_runtime(
                            json.dumps(self._plan.to_mapping()),
                            json.dumps(self._runtime.to_mapping()),
                        )
                    )
                    stopped = True
                    return result

                try:
                    await _call_blocking(stop_after_cancel)
                except asyncio.CancelledError:
                    self._status = RuntimeStatus.STOPPED if stopped else RuntimeStatus.FAILED
                    raise
                except Exception:
                    self._status = RuntimeStatus.FAILED
                else:
                    self._status = RuntimeStatus.STOPPED
                raise
            except FabricError:
                self._status = RuntimeStatus.FAILED
                raise
            except Exception as error:
                self._status = RuntimeStatus.FAILED
                raise FabricRuntimeError(str(error), stage="invoke") from error
            if absorb_result:
                self._absorb(typed_result)
            return typed_result
        except FabricError:
            raise
        except Exception as error:
            raise FabricRuntimeError(str(error), stage="invoke") from error
        finally:
            if not invocation_claimed:
                self._current_task = None

    def invoke_stream(
        self,
        *,
        input: Any = None,
        request: RunRequest | None = None,
    ) -> InvokeStream:
        """Start one turn and stream raw NeMo Relay ATOF records as they arrive.

        ``input`` and ``request`` are mutually exclusive. The returned
        ``InvokeStream`` yields raw ATOF dictionaries. Await
        ``stream.result()`` for the terminal normalized ``RunResult``.

        Raises:
            FabricCapabilityError: If the runtime was not started with NeMo Relay
                enabled and ``streaming=True``.
            FabricConfigError: If request fields conflict or are not
                JSON-compatible.
            FabricStateError: If another turn or stream is active.
        """

        if self._collector_client is None:
            raise FabricCapabilityError(
                "streaming requires a configured standalone ATOF collector and "
                "start_runtime(..., streaming=True)",
                stage="invoke",
                code="streaming_unavailable",
                details={"capability": "streaming"},
            )
        self._ensure_no_active_stream()
        self._ensure_invocable()
        payload = _run_request_payload(input=input, request=request)
        request_id = payload["request_id"]
        registration_ready = asyncio.Event()
        self._reserve_request_registration(request_id)
        try:
            stream = InvokeStream(
                self._invoke_registered_payload(
                    payload,
                    registration_ready,
                    capture_records=True,
                    registration_reserved=True,
                ),
                self._collector_client,
                request_id=request_id,
                registration_ready=registration_ready,
                registration_token=self._registration_tokens.get(request_id),
                on_finalize=lambda: self._finish_registered_request(
                    request_id,
                    remove_queue=True,
                    pi_boundary=(
                        "preserve" if self._uses_pi_stream_correlation() else None
                    ),
                    stream_phase="finalizer",
                ),
            )
        except BaseException:
            self._discard_request_registration(request_id)
            raise
        self._current_stream = stream
        return stream

    async def _invoke_registered_payload(
        self,
        payload: dict[str, Any],
        registration_ready: asyncio.Event | None,
        *,
        capture_records: bool,
        invocation_claimed: bool = False,
        registration_reserved: bool = False,
    ) -> RunResult:
        request_id = payload["request_id"]
        if not registration_reserved:
            self._reserve_request_registration(request_id)
        try:
            await self._finish_request_registration(
                request_id,
                capture_records=capture_records,
            )
        except BaseException as error:
            try:
                await self._finish_registered_request(
                    request_id,
                    remove_queue=True,
                    pi_boundary="release",
                    stream_phase="outcome" if capture_records else None,
                )
            except Exception as cleanup_error:
                error.add_note(
                    f"ATOF collector registration cleanup failed: {cleanup_error}"
                )
            raise
        if registration_ready is not None:
            registration_ready.set()
        try:
            result = await self._invoke_payload(
                payload,
                invocation_claimed=invocation_claimed,
            )
        except BaseException as error:
            try:
                await self._finish_registered_request(
                    request_id,
                    remove_queue=not capture_records,
                    pi_boundary="release",
                    stream_phase="outcome" if capture_records else None,
                )
            except Exception as cleanup_error:
                error.add_note(f"ATOF collector deregistration failed: {cleanup_error}")
            raise
        pi_boundary, pi_turn_count = self._pi_result_boundary(result)
        await self._finish_registered_request(
            request_id,
            remove_queue=not capture_records,
            pi_boundary=pi_boundary,
            pi_turn_count=pi_turn_count,
            stream_phase="outcome" if capture_records else None,
        )
        return result

    async def _finish_request_registration(
        self,
        request_id: str,
        *,
        capture_records: bool,
    ) -> None:
        registration = asyncio.create_task(
            self._register_request(
                request_id,
                capture_records=capture_records,
            )
        )
        try:
            await asyncio.shield(registration)
        except asyncio.CancelledError as error:
            # A cancelled HTTP client does not guarantee that the collector
            # stopped handling its POST. Wait for the bounded request to settle
            # before cleanup so a late registration cannot follow its DELETE.
            while not registration.done():
                try:
                    await asyncio.shield(registration)
                except asyncio.CancelledError:
                    continue
            try:
                registration.result()
            except Exception as registration_error:
                error.add_note(
                    f"ATOF collector registration failed: {registration_error}"
                )
            raise

    async def _register_request(
        self,
        request_id: str,
        *,
        capture_records: bool = True,
    ) -> None:
        if self._collector_client is None:
            return
        if self._plan.adapter.adapter_id == _PI_ADAPTER_ID:
            await self._collector_client.register(
                request_id,
                correlation_mode=_PI_STREAM_CORRELATION_MODE,
                capture_records=capture_records,
                registration_token=self._registration_tokens[request_id],
            )
        else:
            await self._collector_client.register(request_id)

    async def _deregister_request(
        self,
        request_id: str,
        *,
        remove_queue: bool,
        pi_boundary: str | None = None,
        pi_turn_count: int | None = None,
        finalize_registration: bool = False,
    ) -> None:
        if (
            self._collector_client is None
            or request_id not in self._registered_requests
        ):
            return
        if self._plan.adapter.adapter_id == _PI_ADAPTER_ID:
            pi_options: dict[str, Any] = {
                "pi_boundary": pi_boundary,
                "registration_token": self._registration_tokens.get(request_id),
            }
            if pi_turn_count is not None:
                pi_options["pi_turn_count"] = pi_turn_count
            await self._collector_client.deregister(
                request_id,
                remove_queue=remove_queue,
                **pi_options,
            )
        else:
            await self._collector_client.deregister(
                request_id,
                remove_queue=remove_queue,
            )
        if finalize_registration:
            self._discard_request_registration(request_id)

    async def _finish_registered_request(
        self,
        request_id: str,
        *,
        remove_queue: bool,
        pi_boundary: str | None,
        pi_turn_count: int | None = None,
        stream_phase: Literal["outcome", "finalizer"] | None = None,
    ) -> None:
        cleanup_succeeded = False
        cleanup_error: BaseException | None = None
        deregistration = asyncio.create_task(
            self._deregister_request(
                request_id,
                remove_queue=remove_queue,
                pi_boundary=pi_boundary,
                pi_turn_count=pi_turn_count,
                finalize_registration=stream_phase is None,
            )
        )
        try:
            await asyncio.shield(deregistration)
            cleanup_succeeded = True
        except asyncio.CancelledError as error:
            # Retain cleanup ownership until the bounded collector request
            # finishes so cancellation cannot reopen a correlation race.
            while not deregistration.done():
                try:
                    await asyncio.shield(deregistration)
                except asyncio.CancelledError:
                    continue
            try:
                deregistration.result()
            except BaseException as deregistration_error:
                error.add_note(
                    f"ATOF collector deregistration failed: {deregistration_error}"
                )
            else:
                cleanup_succeeded = True
            cleanup_error = error
        except BaseException as error:
            cleanup_error = error

        compensate = stream_phase is not None and self._finish_stream_cleanup_phase(
            request_id,
            stream_phase,
            cleanup_succeeded=cleanup_succeeded,
        )
        if compensate:
            try:
                await self._finish_registered_request(
                    request_id,
                    remove_queue=True,
                    pi_boundary=(
                        "quarantine" if self._uses_pi_stream_correlation() else None
                    ),
                )
            except BaseException as compensation_error:
                if isinstance(compensation_error, asyncio.CancelledError):
                    if cleanup_error is not None:
                        compensation_error.add_note(
                            "ATOF collector deregistration failed before "
                            f"compensating cleanup: {cleanup_error}"
                        )
                    raise
                if cleanup_error is None:
                    raise
                cleanup_error.add_note(
                    "ATOF collector compensating deregistration failed: "
                    f"{compensation_error}"
                )
        if cleanup_error is not None:
            raise cleanup_error

    def _finish_stream_cleanup_phase(
        self,
        request_id: str,
        phase: Literal["outcome", "finalizer"],
        *,
        cleanup_succeeded: bool,
    ) -> bool:
        if request_id not in self._registered_requests:
            return False
        finished = (
            self._stream_outcomes_finished
            if phase == "outcome"
            else self._stream_finalizers_finished
        )
        finished.add(request_id)
        if not cleanup_succeeded:
            self._stream_cleanup_failed.add(request_id)
        if (
            request_id in self._stream_outcomes_finished
            and request_id in self._stream_finalizers_finished
        ):
            self._stream_outcomes_finished.discard(request_id)
            self._stream_finalizers_finished.discard(request_id)
            if request_id in self._stream_cleanup_failed:
                return True
            self._discard_request_registration(request_id)
        return False

    def _reserve_request_registration(self, request_id: str) -> None:
        if self._collector_client is None:
            return
        if request_id in self._registered_requests:
            raise FabricStateError(
                f"collector cleanup is still pending for request_id {request_id!r}"
            )
        # Reserve synchronously so a streamed invocation owns all subsequent
        # registration and cleanup callbacks before its background task starts.
        self._registered_requests.add(request_id)
        if self._plan.adapter.adapter_id == _PI_ADAPTER_ID:
            self._registration_tokens[request_id] = uuid4().hex

    def _discard_request_registration(self, request_id: str) -> None:
        self._registered_requests.discard(request_id)
        self._registration_tokens.pop(request_id, None)
        self._stream_outcomes_finished.discard(request_id)
        self._stream_finalizers_finished.discard(request_id)
        self._stream_cleanup_failed.discard(request_id)

    def _uses_pi_stream_correlation(self) -> bool:
        return (
            self._collector_client is not None
            and self._plan.adapter.adapter_id == _PI_ADAPTER_ID
        )

    def _pi_result_boundary(
        self,
        result: RunResult,
    ) -> tuple[str | None, int | None]:
        if not self._uses_pi_stream_correlation():
            return None, None
        turn_started = self._pi_result_started_turn(result)
        if turn_started is True:
            return "wait", self._pi_result_turn_count(result)
        if turn_started is False:
            return "release", None
        return "quarantine", None

    def _pi_result_started_turn(self, result: RunResult) -> bool | None:
        if not self._uses_pi_stream_correlation():
            return None
        adapter_metadata = result.metadata.get("adapter")
        if isinstance(adapter_metadata, Mapping):
            turn_started = adapter_metadata.get("pi_turn_started")
            if isinstance(turn_started, bool):
                return turn_started
        return None

    def _pi_result_turn_count(self, result: RunResult) -> int | None:
        adapter_metadata = result.metadata.get("adapter")
        if isinstance(adapter_metadata, Mapping):
            turn_count = adapter_metadata.get("pi_turn_count")
            if (
                isinstance(turn_count, int)
                and not isinstance(turn_count, bool)
                and turn_count >= 0
            ):
                return turn_count
        return None

    def invoke_openai_stream(
        self,
        *,
        input: Any = None,
        request: RunRequest | None = None,
    ) -> OpenAIInvokeStream:
        """Start one turn and stream native OpenAI chat-completion chunks.

        The returned stream yields ``chat.completion.chunk`` mappings. Await
        ``stream.result()`` for the separate normalized terminal result.

        Raises:
            FabricCapabilityError: If the selected adapter does not advertise
                native OpenAI streaming.
            FabricConfigError: If request fields conflict or are not
                JSON-compatible.
            FabricStateError: If another turn or stream is active.
        """

        if not self.supports_openai_streaming:
            raise FabricCapabilityError(
                "the selected adapter does not support native OpenAI streaming",
                stage="invoke",
                code="openai_streaming_unavailable",
                details={"capability": "streaming"},
            )
        self._ensure_no_active_stream()
        self._ensure_invocable()
        payload = _run_request_payload(input=input, request=request)
        stream = OpenAIInvokeStream(
            lambda transport: self._invoke_payload(
                payload,
                openai_stream_transport=transport,
                absorb_result=False,
            ),
            runtime_id=self.runtime_id,
            request_id=payload["request_id"],
            on_result=self._absorb,
            on_protocol_failure=self._mark_failed,
        )
        self._current_stream = stream
        return stream

    def _mark_failed(self) -> None:
        self._status = RuntimeStatus.FAILED

    def _ensure_no_active_stream(self) -> None:
        if self._current_stream is not None and (
            not self._current_stream._finalized or not self._current_stream._task.done()
        ):
            raise FabricStateError(
                "a streaming invocation is active; fully consume it or call "
                "`await stream.aclose()` before starting another turn"
            )

    def _ensure_invocable(self) -> None:
        if self._status is not RuntimeStatus.ACTIVE:
            raise FabricStateError(f"cannot invoke a {self._status.value} runtime")
        if self._closing:
            raise FabricStateError(
                "cannot invoke while runtime shutdown is in progress"
            )
        if self._current_task is not None:
            raise FabricStateError("runtime is already running an invocation")

    async def stop(self) -> None:
        """Destroy an idle runtime exactly once.

        Repeated calls after a successful stop are no-ops. A failed runtime may
        still be stopped so its resources are released.

        Raises:
            FabricStateError: If the runtime is already stopping or has an
                invocation in flight.
            FabricNativeUnavailableError: If the native extension is missing.
            FabricRuntimeError: If native runtime shutdown fails.
        """

        if self._status is RuntimeStatus.STOPPED:
            await self._close_streaming_resources()
            return
        if self._current_stream is not None and not self._current_stream._task.done():
            raise FabricStateError(
                "cannot stop while a streaming invocation is active; consume "
                "the stream or await `stream.result()` or `stream.aclose()`"
            )
        if self._current_stream is not None and not self._current_stream._finalized:
            await self._current_stream.aclose()
        if self._current_task is not None:
            raise FabricStateError("cannot stop while a turn is in flight")
        if self._closing:
            raise FabricStateError("runtime shutdown is already in progress")
        self._closing = True
        stopped = False
        stop_error: BaseException | None = None
        try:
            native = self._client._require_native_module("stop")

            def stop() -> Any:
                nonlocal stopped
                result = json.loads(
                    native.stop_runtime(
                        json.dumps(self._plan.to_mapping()),
                        json.dumps(self._runtime.to_mapping()),
                    )
                )
                stopped = True
                return result

            await _call_blocking(stop)
        except asyncio.CancelledError as error:
            self._status = RuntimeStatus.STOPPED if stopped else RuntimeStatus.FAILED
            stop_error = error
            raise
        except FabricError as error:
            self._status = RuntimeStatus.FAILED
            stop_error = error
            raise
        except Exception as error:
            self._status = RuntimeStatus.FAILED
            stop_error = FabricRuntimeError(str(error), stage="stop")
            raise stop_error from error
        else:
            self._status = RuntimeStatus.STOPPED
        finally:
            self._closing = False
            try:
                await self._deregister_requests()
            except Exception as cleanup_error:
                if stop_error is not None:
                    stop_error.add_note(f"runtime cleanup failed: {cleanup_error}")
                else:
                    logger.warning(
                        "ATOF collector deregistration failed during runtime shutdown: %s",
                        cleanup_error,
                        exc_info=cleanup_error,
                    )
            finally:
                await self._close_streaming_resources()

    async def _deregister_requests(self) -> None:
        errors: list[Exception] = []
        for request_id in tuple(self._registered_requests):
            try:
                await self._deregister_request(
                    request_id,
                    remove_queue=True,
                    pi_boundary="release",
                    finalize_registration=True,
                )
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("ATOF collector deregistration failed", errors)

    async def _close_streaming_resources(self) -> None:
        try:
            if self._collector_client is not None:
                await self._collector_client.aclose()
        finally:
            if self._collector is not None:
                await self._collector.aclose()

    def _absorb(self, result: RunResult) -> None:
        self._invocations.append(
            {
                "request_id": result.request_id,
                "runtime_id": result.runtime_id,
                "invocation_id": result.invocation_id,
            }
        )
        output = result.output
        messages = output.get("messages") if isinstance(output, Mapping) else None
        if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
            self._messages = deepcopy(list(messages))

    async def __aenter__(self) -> "Runtime":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        try:
            if self._current_stream is not None and (
                not self._current_stream._finalized
                or not self._current_stream._task.done()
            ):
                await self._current_stream.aclose()
            await self.stop()
        except Exception as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"runtime cleanup failed: {cleanup_error}")
        finally:
            await self._close_streaming_resources()


def _json_mapping(value: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise FabricConfigError(f"{name} must be a JSON object")
    pending: list[Any] = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if isinstance(item, (Mapping, list, tuple)):
            identity = id(item)
            if identity in seen:
                continue
            seen.add(identity)
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise FabricConfigError(f"{name} keys must be strings")
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise FabricConfigError(f"{name} must contain JSON-compatible values") from error


def _merge_overrides(
    base: Mapping[str, Any] | None,
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = _json_mapping(base, "request overrides")
    for key, value in _json_mapping(extra, "request overrides").items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _merge_overrides(current, value)
        else:
            result[key] = value
    return result


def _run_request_payload(
    *,
    input: Any,
    request: RunRequest | None,
) -> dict[str, Any]:
    if input is not None and request is not None:
        raise FabricConfigError("input and request are mutually exclusive")
    try:
        if request is not None:
            if not isinstance(request, RunRequest):
                raise FabricConfigError("request must be a RunRequest")
            payload = request.to_mapping()
        else:
            payload = RunRequest(input=input).to_mapping()
    except ValidationError as error:
        raise FabricConfigError(str(error)) from error
    return payload


async def _run_native_lifecycle(
    native: Any,
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        plan_json = json.dumps(dict(plan))
        runtime = json.loads(native.start_runtime(plan_json))
        runtime_json = json.dumps(runtime)
        result: dict[str, Any] | None = None
        invoke_error: Exception | None = None
        try:
            try:
                result = json.loads(
                    native.invoke_runtime(plan_json, runtime_json, json.dumps(dict(request)))
                )
            except Exception as error:
                invoke_error = error
                raise
            return result
        finally:
            try:
                stop_events = json.loads(native.stop_runtime(plan_json, runtime_json))
            except Exception as error:
                if invoke_error is None:
                    if result is None:
                        raise
                    if result.get("status") == "succeeded":
                        result["status"] = "failed"
                        result["error"] = {
                            "stage": "stop",
                            "code": "runtime_stop_failed",
                            "message": str(error) or "runtime shutdown failed",
                            "retryable": False,
                        }
                stop_events = []
            if result is not None and isinstance(stop_events, list):
                result.setdefault("events", []).extend(stop_events)

    try:
        return await _call_blocking(run)
    except FabricError:
        raise
    except Exception as error:
        raise FabricRuntimeError(str(error), stage="run") from error


async def _call_blocking(func: Any) -> Any:
    worker = asyncio.create_task(asyncio.to_thread(func))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError as cancelled:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
        try:
            worker.result()
        except BaseException:
            pass
        raise cancelled
