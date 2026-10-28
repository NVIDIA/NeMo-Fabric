# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supported NeMo Relay versions for Python adapters."""

from __future__ import annotations

RELAY_MINIMUM_VERSION = (0, 11, 0)
RELAY_MAXIMUM_VERSION = (0, 12, 0)


def is_supported_relay_version(version: tuple[int, int, int]) -> bool:
    """Check the range of an already parsed stable Relay release."""
    return RELAY_MINIMUM_VERSION <= version < RELAY_MAXIMUM_VERSION
