# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import os
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ExaSearchInput(BaseModel):
    question: str = Field(description="The question to search the web for.")


def build_exa_internet_search_tool(
    *,
    name: str = "exa_internet_search",
    max_results: int = 5,
    api_key: str | None = None,
    api_key_env: str = "EXA_API_KEY",
    max_retries: int = 3,
    search_type: Literal["auto", "fast", "deep", "neural", "instant"] = "auto",
    livecrawl: Literal["always", "fallback", "never"] = "fallback",
    max_query_length: int = 2000,
    highlights: bool = True,
    max_content_length: int | None = 10000,
) -> StructuredTool:
    resolved_api_key = api_key or os.environ.get(api_key_env, "")

    async def _exa_internet_search(question: str) -> str:
        if not resolved_api_key:
            return "Web search is unavailable: `EXA_API_KEY` is not configured."

        from langchain_exa import ExaSearchResults

        exa_search = ExaSearchResults(exa_api_key=resolved_api_key)
        if len(question) > max_query_length:
            logger.warning("Exa query truncated from %d to %d characters", len(question), max_query_length)
            question = question[: max_query_length - 3] + "..."

        for attempt in range(max_retries):
            try:
                search_response = await exa_search._arun(
                    question,
                    num_results=max_results,
                    type=search_type,
                    livecrawl=livecrawl,
                    text_contents_options=({"max_characters": max_content_length} if max_content_length else None),
                    highlights=highlights or None,
                )
                if isinstance(search_response, str):
                    return f"No web search results found for: {question}"
                if not search_response.results:
                    return f"No web search results found for: {question}"
                web_search_results = "\n\n---\n\n".join(
                    f'<Document href="{doc.url}"/>\n{doc.text}\n</Document>'
                    for doc in search_response.results
                    if doc.text
                )
                return web_search_results or f"No web search results found for: {question}"
            except Exception:
                logger.exception("Exa search attempt %d of %d failed", attempt + 1, max_retries)
                if attempt == max_retries - 1:
                    return f"Web search failed after {max_retries} attempts for: {question}"
                await asyncio.sleep(2**attempt)
        return f"Web search failed after {max_retries} attempts for: {question}"

    return StructuredTool.from_function(
        coroutine=_exa_internet_search,
        name=name,
        description=(
            "This tool retrieves relevant contexts from web search (using Exa) for the given question. "
            "Args: question (str)."
        ),
        args_schema=ExaSearchInput,
    )
