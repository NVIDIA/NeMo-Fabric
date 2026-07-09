# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CodeGenerationInput(BaseModel):
    question: str = Field(description="The code generation request.")


def build_code_generation_tool(
    *,
    llm: BaseChatModel,
    name: str = "code_generation",
    programming_language: str = "Python",
    verbose: bool = False,
    description: str | None = None,
) -> StructuredTool:
    system_prompt = """
You are a helpful code assistant that can teach a junior developer how to code. Your language of
choice is {programming_language}. Don't explain the code, just generate the code block itself.
"""
    user_prompt = """
{question}
"""
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", user_prompt)])
    prompt = prompt.partial(programming_language=programming_language)

    async def _code_generation(question: str) -> str:
        logger.info("Running code generation tool")
        prompt_value = await prompt.ainvoke({"question": question})
        response = await llm.ainvoke(prompt_value)
        content = getattr(response, "content", response)
        if verbose:
            logger.debug("Tool input was: %s\nTool output is:\n%s", question, content)
        return str(content)

    return StructuredTool.from_function(
        coroutine=_code_generation,
        name=name,
        description=description
        or (
            "Useful to generate Python code. For any questions about code generation, "
            "you must only use this tool!"
        ),
        args_schema=CodeGenerationInput,
    )


def build_code_generation_tool_from_spec(
    spec: dict[str, Any],
    *,
    name: str,
    build_llm: Callable[[str], BaseChatModel],
) -> StructuredTool:
    llm_name = str(spec.get("llm_name") or "default")
    return build_code_generation_tool(
        llm=build_llm(llm_name),
        name=name,
        programming_language=str(spec.get("programming_language", "Python")),
        verbose=bool(spec.get("verbose", False)),
        description=spec.get("description"),
    )
