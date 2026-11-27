# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Existing batch interface for the repeated-invocation exercise."""

from typing import Any

from settings import AppSettings


async def run_batch(
    settings: AppSettings, prompts: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    """Run ordered (request_id, text) pairs and return the service result shape.

    Conversation state should last for the batch. Shutdown failures remain
    observable to the caller; the application owns retries.
    """
    raise NotImplementedError("Implement the consumer batch path")
