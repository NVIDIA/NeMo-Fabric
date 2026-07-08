# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import zoneinfo

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class CurrentDatetimeInput(BaseModel):
    unused: str = Field(default="", description="Unused placeholder argument.")


class CurrentTimezoneInput(BaseModel):
    unused: str = Field(default="", description="Unused placeholder argument.")


def _resolve_timezone(*, timezone_name: str, context_timezone: str | None) -> zoneinfo.ZoneInfo:
    for candidate in (context_timezone, timezone_name):
        if not candidate:
            continue
        try:
            return zoneinfo.ZoneInfo(candidate)
        except Exception:
            continue
    return zoneinfo.ZoneInfo("Etc/UTC")


def build_current_datetime_tool(
    *,
    timezone_name: str = "Etc/UTC",
    context_timezone: str | None = None,
    name: str = "current_datetime",
) -> StructuredTool:
    async def _current_datetime(unused: str = "") -> str:
        del unused
        timezone_obj = _resolve_timezone(timezone_name=timezone_name, context_timezone=context_timezone)
        now = datetime.datetime.now(timezone_obj)
        return f"The current time of day is {now.strftime('%Y-%m-%d %H:%M:%S %z')}"

    return StructuredTool.from_function(
        coroutine=_current_datetime,
        name=name,
        description=(
            "Returns the current date and time in human readable format with timezone information. "
            "REQUIRED: Call this tool when you need the current time."
        ),
        args_schema=CurrentDatetimeInput,
    )


def build_current_timezone_tool(
    *,
    timezone_name: str = "Etc/UTC",
    context_timezone: str | None = None,
    name: str = "current_timezone",
) -> StructuredTool:
    async def _current_timezone(unused: str = "") -> str:
        del unused
        timezone_obj = _resolve_timezone(timezone_name=timezone_name, context_timezone=context_timezone)
        return f"The time zone is {timezone_obj}"

    return StructuredTool.from_function(
        coroutine=_current_timezone,
        name=name,
        description=(
            "Returns the user's/system timezone in IANA zone name format (e.g. America/Los_Angeles). "
            "REQUIRED: Call this tool first whenever you need the current time or timezone."
        ),
        args_schema=CurrentTimezoneInput,
    )
