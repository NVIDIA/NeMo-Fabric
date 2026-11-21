# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Existing request interface; implement it without changing AppSettings."""

from typing import Any

from settings import AppSettings


async def handle_request(
    settings: AppSettings, text: str, request_id: str
) -> dict[str, Any]:
    """Return status/output/error/artifacts/telemetry and the three correlation IDs.

    The application supplies the credential variable name, never its value.
    SDK failures that prevent a normalized result should propagate to the caller.
    No live harness or model credentials are part of this evaluation fixture.
    """
    raise NotImplementedError("Implement the consumer request path")
