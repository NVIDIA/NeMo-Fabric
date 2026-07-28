# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public exception hierarchy for the NeMo Fabric Python SDK."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


class FabricError(RuntimeError):
    """Base class for structured SDK-level NeMo Fabric errors.

    Catch this type to handle any SDK failure while preserving machine-readable
    stage, code, retryability, and detail fields.

    Attributes:
        stage: Lifecycle stage that failed, when known.
        code: Stable machine-readable error code, when available.
        retryable: Whether retrying may succeed without changing the request.
        details: Detached structured error details.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        code: str | None = None,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize a structured NeMo Fabric exception.

        Args:
            message: Human-readable failure description.
            stage: Optional lifecycle stage that failed.
            code: Optional stable machine-readable error code.
            retryable: Whether callers may safely retry unchanged input.
            details: Optional structured diagnostic context. The exception
                stores a deep copy.
        """

        super().__init__(message)
        self.stage = stage
        self.code = code
        self.retryable = retryable
        self.details = deepcopy(dict(details or {}))


class FabricConfigError(FabricError):
    """Invalid SDK input, request shape, factory, or resolved config."""


class FabricRuntimeError(FabricError):
    """Failure while starting, invoking, stopping, or otherwise driving a runtime."""


class FabricStateError(FabricRuntimeError):
    """Operation rejected because a local runtime is in the wrong state."""


class FabricCapabilityError(FabricRuntimeError):
    """Operation rejected by resolved runtime capabilities or implementation status."""


class FabricNativeUnavailableError(FabricRuntimeError):
    """SDK call requires the PyO3 extension, but it is not installed or importable."""
