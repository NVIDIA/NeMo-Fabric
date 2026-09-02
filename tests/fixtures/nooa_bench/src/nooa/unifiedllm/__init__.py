# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic model client used by the BenchAgent subprocess fixture."""

from __future__ import annotations

from typing import Any


class FixtureModel:
    def __init__(self, name: str, **settings: Any) -> None:
        self.name = name
        self.settings = settings

    async def aclose(self) -> None:
        return None


def get_llm_client(
    name: str,
    *,
    client_type: str | None = None,
    **overrides: Any,
) -> FixtureModel:
    return FixtureModel(name, client_type=client_type, **overrides)
