# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public exception hierarchy for the NeMo Fabric Python SDK."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


class FabricError(RuntimeError):
    """Base class for SDK-level Fabric errors."""

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        code: str | None = None,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.retryable = retryable
        self.details = deepcopy(dict(details or {}))


class FabricConfigError(FabricError):
    """Raised when SDK input or resolved config is invalid for the requested API."""


class FabricRuntimeError(FabricError):
    """Raised when a runtime lifecycle call fails."""


class FabricStateError(FabricRuntimeError):
    """Raised when a local SDK handle is used in an invalid lifecycle state."""


class FabricCapabilityError(FabricRuntimeError):
    """Raised when the resolved runtime does not support the requested operation."""


class FabricNativeUnavailableError(FabricRuntimeError):
    """Raised when an SDK method requires the native extension."""
