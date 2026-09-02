# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test double for the already unit-tested OO Agents middleware installer."""

from __future__ import annotations

from typing import Any


def install_nemo_relay(_event_manager: Any):
    return lambda: None
