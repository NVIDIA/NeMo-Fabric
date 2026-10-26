# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_fabric._report_format import format_bytes, format_duration_ms, format_money


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "Unavailable"),
        (0, "$0.00"),
        (1, "$0.000000001"),
        (280000, "$0.000280"),
        (1234567890, "$1.23"),
        (999999999, "$1.000"),
        (18446744073709551615, "$18446744073.71"),
    ],
)
def test_money_preserves_positive_spend_with_readable_precision(value, expected):
    assert format_money(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0.000 s"),
        (0.25, "0.000 s"),
        (6000, "6.000 s"),
        (61000, "61.000 s"),
        (3661000, "3661.000 s"),
    ],
)
def test_duration_uses_explicit_readable_units(value, expected):
    assert format_duration_ms(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, "0 B"), (1024, "1.0 KiB"), (21086208, "20.1 MiB")],
)
def test_memory_uses_binary_units(value, expected):
    assert format_bytes(value) == expected
