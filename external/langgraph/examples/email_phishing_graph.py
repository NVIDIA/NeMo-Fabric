# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application-owned graph factory for the email-phishing example."""

from __future__ import annotations

import os
from typing import Any
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph


class EmailState(TypedDict):
    """State for one email analysis."""

    input: str
    assessment: str


class TextInputGraph:
    """Adapt a raw text invocation to the graph's application state."""

    def __init__(self, graph: StateGraph[EmailState]) -> None:
        self._graph = graph

    def compile(self) -> "CompiledTextInputGraph":
        """Compile the application graph and retain its text-input adapter."""

        return CompiledTextInputGraph(self._graph.compile())


class CompiledTextInputGraph:
    """Invoke a compiled graph with raw text supplied by the application."""

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    async def ainvoke(self, input_value: str) -> dict[str, str]:
        """Map a text input into the state expected by the compiled graph."""

        return await self._graph.ainvoke({"input": input_value})


def build_graph(
    model: str,
    base_url: str,
    api_key_env: str,
) -> TextInputGraph:
    """Return an uncompiled graph that uses an OpenAI-compatible model endpoint."""

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"Environment variable {api_key_env!r} is required")
    llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=0)

    async def classify(state: EmailState) -> dict[str, str]:
        response = await llm.ainvoke(
            [
                (
                    "system",
                    "Classify the email as phishing or benign. Explain the evidence briefly.",
                ),
                ("user", state["input"]),
            ]
        )
        return {"assessment": str(response.content)}

    graph = StateGraph(EmailState)
    graph.add_node("classify", classify)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", END)
    return TextInputGraph(graph)
