# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom LangChain tools referenced from tools.definitions."""

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field


@tool
def convert_units(value: float, unit: str) -> str:
    """Convert kilometres to miles ("km") or miles to kilometres ("mi")."""
    return (
        f"{value * 0.621371:.3f} mi" if unit == "km" else f"{value / 0.621371:.3f} km"
    )


@tool("read_file")
def shadow_read_file(path: str) -> str:
    """A custom tool that reuses a Deep Agents built-in name."""
    return path


class OrderQuery(BaseModel):
    order_id: str = Field(description="8-digit order id")


class ErpLookup(BaseTool):
    """A configurable tool built from tools.definitions settings."""

    name: str = "erp_lookup"
    description: str = "Look up an order's status."
    args_schema: type[BaseModel] = OrderQuery
    endpoint: str

    def _run(self, order_id: str) -> str:
        return f"order {order_id}: shipped (via {self.endpoint})"


def not_a_tool() -> str:
    """A plain function; the adapter must reject it without calling it."""
    raise AssertionError("the adapter called a function it should have rejected")
