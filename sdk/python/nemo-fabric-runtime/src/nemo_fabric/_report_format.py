# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readable units for evidence summaries; stored quantities remain unchanged."""

from __future__ import annotations


def format_money(value: int | None) -> str:
    """Format nanoUSD with cents or three significant sub-dollar digits.

    Integer rounding preserves very small positive amounts without presenting
    them as free execution. JSON retains the original nanoUSD quantity.
    """
    if value is None:
        return "Unavailable"
    decimals = 2 if value == 0 else min(9, max(2, 12 - len(str(value))))
    quantum = 10 ** (9 - decimals)
    rounded = (value + quantum // 2) // quantum
    scale = 10**decimals
    return f"${rounded // scale}.{rounded % scale:0{decimals}d}"


def format_duration_ms(value: int | float) -> str:
    """Format a measured duration without changing its measurement scope."""
    return f"{value / 1000:.3f} s"


def format_bytes(value: int) -> str:
    """Format bytes using explicitly binary units."""
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"):
        if amount < 1024 or unit == "EiB":
            return f"{value} B" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable byte unit")
