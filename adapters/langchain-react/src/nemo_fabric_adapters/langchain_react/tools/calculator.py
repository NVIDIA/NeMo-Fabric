# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class NumbersInput(BaseModel):
    numbers: list[float] = Field(description="List of numbers to operate on.")


def build_calculator_tools(include: list[str] | None = None) -> list[StructuredTool]:
    allowed = include or ["add", "subtract", "multiply", "divide", "compare"]
    tools: list[StructuredTool] = []

    async def _add(numbers: list[float]) -> float:
        if len(numbers) < 2:
            raise ValueError("This tool only supports addition between two or more numbers.")
        return float(sum(numbers))

    async def _subtract(numbers: list[float]) -> float:
        if len(numbers) != 2:
            raise ValueError("This tool only supports subtraction between two numbers.")
        return float(numbers[0] - numbers[1])

    async def _multiply(numbers: list[float]) -> float:
        if len(numbers) < 2:
            raise ValueError("This tool only supports multiplication between two or more numbers.")
        return float(math.prod(numbers))

    async def _divide(numbers: list[float]) -> float:
        if len(numbers) != 2:
            raise ValueError("This tool only supports division between two numbers.")
        if numbers[1] == 0:
            raise ValueError("Cannot divide by zero.")
        return float(numbers[0] / numbers[1])

    async def _compare(numbers: list[float]) -> str:
        if len(numbers) != 2:
            raise ValueError("This tool only supports comparison between two numbers.")
        a, b = numbers
        if a > b:
            return f"{a} is greater than {b}"
        if a < b:
            return f"{a} is less than {b}"
        return f"{a} is equal to {b}"

    specs: dict[str, tuple[Any, str]] = {
        "add": (_add, "Add two or more numbers together."),
        "subtract": (_subtract, "Subtract one number from another."),
        "multiply": (_multiply, "Multiply two or more numbers together."),
        "divide": (_divide, "Divide one number by another."),
        "compare": (_compare, "Compare two numbers."),
    }
    for tool_name in allowed:
        fn, description = specs[tool_name]
        tools.append(
            StructuredTool.from_function(
                coroutine=fn,
                name=tool_name,
                description=description,
                args_schema=NumbersInput,
            )
        )
    return tools
