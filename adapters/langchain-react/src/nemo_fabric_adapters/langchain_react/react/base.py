# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared LangGraph agent primitives for the LangChain ReAct adapter."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import AIMessageChunk
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages.utils import convert_to_openai_messages
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import DEFAULT_RUNTIME
from pydantic import BaseModel

logger = logging.getLogger(__name__)

AGENT_LOG_PREFIX = "[AGENT]"
INPUT_SCHEMA_MESSAGE = ". Arguments must be provided as a valid JSON object following this format: {schema}"
TOOL_NOT_FOUND_ERROR_MESSAGE = "There is no tool named {tool_name}. Tool must be one of {tools}."
NO_INPUT_ERROR_MESSAGE = "No human input received to the agent, Please ask a valid question."


class AgentDecision(Enum):
    TOOL = "tool"
    END = "finished"


def _extract_reasoning_content(message: BaseMessage) -> str:
    reasoning = message.additional_kwargs.get("reasoning_content", "")
    return reasoning if isinstance(reasoning, str) else str(reasoning)


def _format_agent_thoughts_for_log(message: BaseMessage) -> str:
    content = message.content
    content_text = content if isinstance(content, str) else str(content)
    if content_text.strip():
        return content_text
    return _extract_reasoning_content(message)


def _chunk_to_message(chunk: AIMessageChunk) -> AIMessage:
    additional_kwargs = dict(chunk.additional_kwargs)
    if chunk.tool_calls and not additional_kwargs.get("tool_calls"):
        openai_msg = convert_to_openai_messages([chunk])[0]
        if "tool_calls" in openai_msg:
            additional_kwargs["tool_calls"] = openai_msg["tool_calls"]
    return AIMessage(
        content=chunk.content,
        additional_kwargs=additional_kwargs,
        response_metadata=chunk.response_metadata,
        id=chunk.id,
        usage_metadata=chunk.usage_metadata,
    )


class BaseAgent(ABC):
    def __init__(
        self,
        *,
        llm: BaseChatModel,
        tools: list[BaseTool],
        callbacks: list[AsyncCallbackHandler] | None = None,
        detailed_logs: bool = False,
        log_response_max_chars: int = 1000,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.callbacks = callbacks or []
        self.detailed_logs = detailed_logs
        self.log_response_max_chars = log_response_max_chars
        self.graph = None
        self._runnable_config = RunnableConfig(
            callbacks=self.callbacks,
            configurable={"__pregel_runtime": DEFAULT_RUNTIME},
        )

    async def _stream_llm(
        self,
        runnable: Any,
        inputs: dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> AIMessage:
        effective_config = merge_configs(self._runnable_config, config) if config is not None else self._runnable_config
        chunks: list[AIMessageChunk] = []
        async for chunk in runnable.astream(inputs, config=effective_config):
            chunks.append(chunk)
        if not chunks:
            return AIMessage(content="")
        accumulated = chunks[0]
        for chunk in chunks[1:]:
            accumulated = accumulated + chunk
        return _chunk_to_message(accumulated)

    async def _call_tool(
        self,
        tool: BaseTool,
        tool_input: dict[str, Any] | str,
        max_retries: int = 3,
    ) -> ToolMessage:
        last_exception: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                response = await tool.ainvoke(tool_input, config=self._runnable_config)
                if response is None or (isinstance(response, str) and response == ""):
                    return ToolMessage(
                        name=tool.name,
                        tool_call_id=tool.name,
                        content=f"The tool {tool.name} provided an empty response.",
                    )
                if isinstance(response, dict):
                    response = [response]
                return ToolMessage(name=tool.name, tool_call_id=tool.name, content=response)
            except Exception as exc:
                last_exception = exc
                if attempt == max_retries:
                    break
                await asyncio.sleep(2**attempt)
        error_content = f"Tool call failed after all retry attempts. Last error: {str(last_exception)}"
        logger.error("%s %s", AGENT_LOG_PREFIX, error_content, exc_info=True)
        return ToolMessage(
            name=tool.name,
            tool_call_id=tool.name,
            content=error_content,
            status="error",
        )

    def _log_tool_response(self, tool_name: str, tool_input: Any, tool_response: str) -> None:
        if not self.detailed_logs:
            return
        display_response = (
            tool_response[: self.log_response_max_chars] + "...(truncated)"
            if len(tool_response) > self.log_response_max_chars
            else tool_response
        )
        logger.info("%s tool=%s input=%s response=%s", AGENT_LOG_PREFIX, tool_name, tool_input, display_response)

    def _get_chat_history(self, messages: list[BaseMessage]) -> str:
        return "\n".join(f"{message.type}: {message.content}" for message in messages[:-1])

    @abstractmethod
    async def _build_graph(self, state_schema: type) -> CompiledStateGraph:
        pass


class DualNodeAgent(BaseAgent):
    @abstractmethod
    async def agent_node(self, state: BaseModel, config: RunnableConfig | None = None) -> BaseModel:
        pass

    @abstractmethod
    async def tool_node(self, state: BaseModel) -> BaseModel:
        pass

    @abstractmethod
    async def conditional_edge(self, state: BaseModel) -> str:
        pass

    async def _build_graph(self, state_schema: type) -> CompiledStateGraph:
        graph = StateGraph(state_schema)
        graph.add_node("agent", self.agent_node)
        graph.add_node("tool", self.tool_node)
        graph.add_edge("tool", "agent")
        graph.add_conditional_edges(
            "agent",
            self.conditional_edge,
            {AgentDecision.TOOL: "tool", AgentDecision.END: "__end__"},
        )
        graph.set_entry_point("agent")
        self.graph = graph.compile()
        return self.graph
