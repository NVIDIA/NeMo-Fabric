# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the ``nemo_fabric.versions.v1`` report an adapter returns with each result.

NeMo Fabric core reads the report from ``AgentRunResult.extensions`` when the
adapter's descriptor names the key under its ``run_result`` extension schema,
and moves its facts into ``RunResult.what_ran``. Every fact is reported by the
adapter, so each carries ``source`` ``reported`` and ``collected_by``
``adapter``. A fact holds ``value``, or ``state`` and ``reason``, never both.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import sys
import threading
from collections.abc import Callable
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from typing import TypeVar

VERSIONS_REPORT_KEY = "nemo_fabric.versions.v1"
PROBE_TIMEOUT_SECONDS = 5.0

METHOD_ADAPTER_REPORT = "adapter_report"
METHOD_PACKAGE_METADATA = "package_metadata"

_MAX_VALUE_CHARS = 256
_TRUNCATED_PREFIX_CHARS = 240
_TRUNCATED_DIGEST_HEX_DIGITS = 15
_MAX_MODEL_ROLES = 16
_MAX_KEY_CHARS = 128

T = TypeVar("T")


def known_fact(value: str, method: str) -> dict[str, Any]:
    """A reported fact with a value, bounded by the string length rule.

    A lone surrogate, which JSON text from a harness can carry, becomes ``?``
    first, because core rejects a result that holds one.
    """

    sanitized = value.encode("utf-8", "replace").decode("utf-8")
    return {
        "value": bounded_text(sanitized),
        "source": "reported",
        "collected_by": "adapter",
        "method": method,
    }


def missing_fact(
    state: str, reason: str, method: str = METHOD_ADAPTER_REPORT
) -> dict[str, Any]:
    """A reported fact without a value."""

    return {
        "state": state,
        "reason": reason,
        "source": "reported",
        "collected_by": "adapter",
        "method": method,
    }


def no_source_fact() -> dict[str, Any]:
    """A fact this adapter has no source for."""

    return missing_fact("unsupported", "no_source")


def bounded_text(text: str) -> str:
    """Apply NeMo Fabric core's string length rule.

    A string of at most 256 characters is kept whole. A longer one keeps its
    first 240 characters, then ``~``, then the first 15 hex digits of the
    SHA-256 of its UTF-8 bytes, as core does.
    """

    if len(text) <= _MAX_VALUE_CHARS:
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (
        f"{text[:_TRUNCATED_PREFIX_CHARS]}~{digest[:_TRUNCATED_DIGEST_HEX_DIGITS]}"
    )


def package_version_fact(distribution: str) -> dict[str, Any]:
    """The installed version of ``distribution``, read from its package metadata."""

    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return missing_fact("unavailable", "not_collected", METHOD_PACKAGE_METADATA)
    if not version:
        return missing_fact("unavailable", "empty_response", METHOD_PACKAGE_METADATA)
    return known_fact(version, METHOD_PACKAGE_METADATA)


def loaded_module_version_fact(distribution: str, module_name: str) -> dict[str, Any]:
    """The version of ``distribution`` when it installed the loaded ``module_name``.

    Another copy of the module can shadow the installed package, for example a
    source checkout earlier on ``sys.path``. The version is reported only when
    the distribution's file list holds the file the module was loaded from;
    otherwise it reads ``unavailable`` with reason ``not_attributed``. A module
    that is not loaded reads ``not_collected``.
    """

    module = sys.modules.get(module_name)
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        return missing_fact("unavailable", "not_collected", METHOD_PACKAGE_METADATA)
    try:
        installed = importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        return missing_fact("unavailable", "not_collected", METHOD_PACKAGE_METADATA)
    loaded = Path(module_file).resolve()
    if not any(
        Path(str(installed.locate_file(file))).resolve() == loaded
        for file in installed.files or ()
        if file.name == loaded.name
    ):
        return missing_fact("unavailable", "not_attributed", METHOD_PACKAGE_METADATA)
    version = installed.version
    if not version:
        return missing_fact("unavailable", "empty_response", METHOD_PACKAGE_METADATA)
    return known_fact(version, METHOD_PACKAGE_METADATA)


def versions_report(
    *,
    package_version: Mapping[str, Any],
    harness_name: Mapping[str, Any],
    harness_version: Mapping[str, Any],
    served_models: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """The ``nemo_fabric.versions.v1`` report object.

    ``served_models`` maps each plan role to its served-model fact. Like core,
    the report keeps at most 16 roles in sorted order and skips a role longer
    than 128 characters, so the object stays far below core's 64 KiB limit
    for one report key.
    """

    roles = sorted(role for role in served_models if len(role) <= _MAX_KEY_CHARS)
    return {
        "adapter": {"package_version": dict(package_version)},
        "harness": {"name": dict(harness_name), "version": dict(harness_version)},
        "models": {
            role: {"served": dict(served_models[role])}
            for role in roles[:_MAX_MODEL_ROLES]
        },
    }


class BlockingProbe:
    """Run a runtime's blocking version calls off the event loop, one at a time.

    A blocking call cannot be cancelled, so it runs on a daemon thread and the
    caller waits at most ``timeout`` seconds for it. While an earlier call
    still runs, no new thread starts and the call times out at once, so a
    runtime never has more than one outstanding probe. Stopping the runtime
    never waits for the thread, and a daemon thread never delays process exit.
    """

    def __init__(self) -> None:
        self._running = threading.Event()

    async def call(
        self, function: Callable[[], T], *, timeout: float = PROBE_TIMEOUT_SECONDS
    ) -> T:
        """Return ``function()``, or raise ``TimeoutError`` past ``timeout``.

        An exception that ``function`` raises reaches the caller.
        """

        if self._running.is_set():
            raise TimeoutError("an earlier version probe is still running")
        loop = asyncio.get_running_loop()
        outcome: asyncio.Future[T] = loop.create_future()

        def deliver(result: T | None, error: Exception | None) -> None:
            if outcome.done():
                return
            if error is not None:
                outcome.set_exception(error)
            else:
                outcome.set_result(result)  # type: ignore[arg-type]

        def run() -> None:
            result: T | None = None
            error: Exception | None = None
            try:
                result = function()
            except Exception as raised:
                error = raised
            finally:
                self._running.clear()
            try:
                loop.call_soon_threadsafe(deliver, result, error)
            except RuntimeError:
                # The event loop closed while the probe ran; nobody waits for it.
                return

        self._running.set()
        threading.Thread(target=run, name="nemo-fabric-version-probe", daemon=True).start()
        # On timeout, wait_for cancels only this future. The thread keeps
        # running, and its late outcome is dropped.
        return await asyncio.wait_for(outcome, timeout)
