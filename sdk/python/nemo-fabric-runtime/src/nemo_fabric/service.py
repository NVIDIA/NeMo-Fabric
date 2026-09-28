# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Long-lived service lifecycle support for the NeMo Fabric Python SDK."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from collections.abc import Mapping
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any

from nemo_fabric.errors import FabricError
from nemo_fabric.errors import FabricRuntimeError
from nemo_fabric.errors import FabricStateError
from nemo_fabric.runtime import _call_blocking
from nemo_fabric.types import RunPlan
from nemo_fabric.types import ServiceHandle


class ServiceStatus(str, Enum):
    """Lifecycle state of a prepared or attached service."""

    ACTIVE = "active"
    RELEASING = "releasing"
    RELEASED = "released"
    FAILED = "failed"


class Service:
    """One prepared or attached long-lived adapter service.

    Create services with ``Fabric.prepare_service()`` or
    ``Fabric.attach_service()``. Use the object as an asynchronous context
    manager to guarantee release or detach.
    """

    def __init__(
        self,
        *,
        client: Any,
        plan: RunPlan | Mapping[str, Any],
        service: ServiceHandle | Mapping[str, Any],
    ) -> None:
        """lazydocs: ignore"""

        self._client = client
        self._plan = plan if isinstance(plan, RunPlan) else RunPlan.from_mapping(plan)
        self._service = (
            service
            if isinstance(service, ServiceHandle)
            else ServiceHandle.from_mapping(service)
        )
        self._status = ServiceStatus.ACTIVE
        self._lifecycle_lock = asyncio.Lock()

    @property
    def status(self) -> ServiceStatus:
        """Return the current service lifecycle state."""

        return self._status

    @property
    def service_id(self) -> str:
        """Return the unique identifier for this process-local service."""

        return self._service.service_id

    @property
    def handle(self) -> ServiceHandle:
        """Return a detached snapshot of the service handle."""

        return ServiceHandle.from_mapping(self._service.to_mapping())

    @asynccontextmanager
    async def _runtime_start(self) -> AsyncIterator[None]:
        """Serialize one runtime start against service release."""

        async with self._lifecycle_lock:
            yield

    async def release(self) -> None:
        """Stop an owned service or detach from a caller-owned service.

        Release fails while active runtimes still reference the service.
        Repeated calls after a successful release are no-ops.
        """

        if self._status is ServiceStatus.RELEASED:
            return
        if self._status is ServiceStatus.RELEASING:
            raise FabricStateError("service release is already in progress")
        self._status = ServiceStatus.RELEASING
        released = False
        release_started = False
        try:
            async with self._lifecycle_lock:
                native = self._client._require_native_module("release_service")

                def release() -> Any:
                    nonlocal released
                    result = json.loads(
                        native.release_service(
                            json.dumps(self._plan.to_mapping()),
                            json.dumps(self._service.to_mapping()),
                        )
                    )
                    released = True
                    return result

                release_started = True
                await _call_blocking(release)
        except asyncio.CancelledError:
            if released:
                self._status = ServiceStatus.RELEASED
            elif release_started:
                self._status = ServiceStatus.FAILED
            else:
                self._status = ServiceStatus.ACTIVE
            raise
        except FabricError:
            self._status = ServiceStatus.ACTIVE
            raise
        except Exception as error:
            if isinstance(error, getattr(native, "ServiceInUseError", ())):
                self._status = ServiceStatus.ACTIVE
            else:
                self._status = ServiceStatus.FAILED
            raise FabricRuntimeError(str(error), stage="stop") from error
        else:
            self._status = ServiceStatus.RELEASED

    async def __aenter__(self) -> "Service":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        try:
            await self.release()
        except Exception as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"service cleanup failed: {cleanup_error}")
