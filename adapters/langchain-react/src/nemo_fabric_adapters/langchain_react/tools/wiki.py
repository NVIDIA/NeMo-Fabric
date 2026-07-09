# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class WikiSearchInput(BaseModel):
    question: str = Field(description="The question to search Wikipedia for.")


def build_wiki_search_tool(*, max_results: int = 2, name: str = "wiki_search") -> StructuredTool:
    async def _wiki_search(question: str) -> str:
        from langchain_community.document_loaders import WikipediaLoader

        search_docs = await WikipediaLoader(query=question, load_max_docs=max_results).aload()
        if not search_docs:
            return "No Wikipedia results found."
        return "\n\n---\n\n".join(
            f'<Document source="{doc.metadata.get("source", "")}" '
            f'page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        )

    return StructuredTool.from_function(
        coroutine=_wiki_search,
        name=name,
        description=(
            "This tool retrieves relevant contexts from wikipedia search for the given question. "
            "Args: question (str)."
        ),
        args_schema=WikiSearchInput,
    )
