# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangGraph ReAct agent graph (NAT-free port)."""

from __future__ import annotations

import ast
import json
import logging
import re
import typing
from json import JSONDecodeError

from langchain_core.agents import AgentAction
from langchain_core.agents import AgentFinish
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from pydantic import Field

from nemo_fabric_adapters.langchain_react.react.base import AGENT_LOG_PREFIX
from nemo_fabric_adapters.langchain_react.react.base import INPUT_SCHEMA_MESSAGE
from nemo_fabric_adapters.langchain_react.react.base import NO_INPUT_ERROR_MESSAGE
from nemo_fabric_adapters.langchain_react.react.base import TOOL_NOT_FOUND_ERROR_MESSAGE
from nemo_fabric_adapters.langchain_react.react.base import AgentDecision
from nemo_fabric_adapters.langchain_react.react.base import DualNodeAgent
from nemo_fabric_adapters.langchain_react.react.base import _format_agent_thoughts_for_log
from nemo_fabric_adapters.langchain_react.react.output_parser import FINAL_ANSWER_PATTERN
from nemo_fabric_adapters.langchain_react.react.output_parser import ReActAgentParsingFailedError
from nemo_fabric_adapters.langchain_react.react.output_parser import ReActOutputParser
from nemo_fabric_adapters.langchain_react.react.output_parser import ReActOutputParserException
from nemo_fabric_adapters.langchain_react.react.prompt import SYSTEM_PROMPT
from nemo_fabric_adapters.langchain_react.react.prompt import USER_PROMPT
from nemo_fabric_adapters.langchain_react.react.text import remove_r1_think_tags

logger = logging.getLogger(__name__)


class WorkflowSettings(BaseModel):
    tool_names: list[str] = Field(default_factory=list)
    llm_name: str = "default"
    verbose: bool = False
    parse_agent_response_max_retries: int = 1
    max_tool_calls: int = 15
    use_native_tool_calling: bool = False
    retry_agent_response_parsing_errors: bool = True
    tool_call_max_retries: int = 1
    pass_tool_call_errors_to_agent: bool = True
    normalize_tool_input_quotes: bool = True
    raise_on_parsing_failure: bool = True
    include_tool_input_schema_in_tool_description: bool = True
    max_history: int = 15
    system_prompt: str | None = None
    additional_instructions: str | None = None
    log_response_max_chars: int = 1000


class ReActGraphState(BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    agent_scratchpad: list[AgentAction] = Field(default_factory=list)
    tool_responses: list[BaseMessage] = Field(default_factory=list)
    final_answer: str | None = Field(default=None)


def _tool_schema_text(tool: BaseTool) -> str:
    schema = getattr(tool, "args_schema", None) or getattr(tool, "input_schema", None)
    if schema is None:
        return "{}"
    model_fields = getattr(schema, "model_fields", None)
    if model_fields is not None:
        return str(model_fields)
    return str(schema)


class ReActAgentGraph(DualNodeAgent):
    def __init__(
        self,
        *,
        llm: BaseChatModel,
        prompt: ChatPromptTemplate,
        tools: list[BaseTool],
        config: WorkflowSettings,
        callbacks: list[typing.Any] | None = None,
    ):
        super().__init__(
            llm=llm,
            tools=tools,
            callbacks=callbacks,
            detailed_logs=config.verbose,
            log_response_max_chars=config.log_response_max_chars,
        )
        self.config = config
        self.parse_agent_response_max_retries = (
            config.parse_agent_response_max_retries if config.retry_agent_response_parsing_errors else 1
        )
        self.tool_call_max_retries = config.tool_call_max_retries
        self.pass_tool_call_errors_to_agent = config.pass_tool_call_errors_to_agent
        self.normalize_tool_input_quotes = config.normalize_tool_input_quotes
        self.raise_on_parsing_failure = config.raise_on_parsing_failure
        self.use_native_tool_calling = config.use_native_tool_calling

        tool_names = ",".join(tool.name for tool in tools)
        if not config.include_tool_input_schema_in_tool_description:
            descriptions = "\n".join(f"{tool.name}: {tool.description}" for tool in tools)
        else:
            descriptions = "\n".join(
                f"{tool.name}: {tool.description}. "
                f"{INPUT_SCHEMA_MESSAGE.format(schema=_tool_schema_text(tool))}"
                for tool in tools
            )
        prompt = prompt.partial(tools=descriptions, tool_names=tool_names)
        self.agent = prompt | self._maybe_bind_llm_and_yield(tools if config.use_native_tool_calling else None)
        self.tools_dict = {tool.name: tool for tool in tools}

    def _maybe_bind_llm_and_yield(
        self,
        tools: list[BaseTool] | None = None,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        if tools is not None:
            try:
                return self.llm.bind_tools(tools)
            except NotImplementedError:
                logger.warning("%s LLM does not support bind_tools, falling back to text parsing", AGENT_LOG_PREFIX)
                self.use_native_tool_calling = False
        smart_models = re.compile(r"gpt-?5", re.IGNORECASE)
        if smart_models.search(str(getattr(self.llm, "model", ""))):
            return self.llm
        return self.llm.bind(stop=["Observation:"])

    def _get_tool(self, tool_name: str) -> BaseTool | None:
        return self.tools_dict.get(tool_name)

    def _parse_tool_input(self, tool_input_str: str) -> tuple[typing.Any, bool]:
        if tool_input_str == "None":
            return tool_input_str, True
        try:
            return json.loads(tool_input_str), True
        except JSONDecodeError:
            pass
        if not self.normalize_tool_input_quotes:
            return tool_input_str, False
        normalized_str = tool_input_str.replace("'", '"')
        try:
            return json.loads(normalized_str), True
        except JSONDecodeError:
            pass
        has_python_none = any(marker in tool_input_str for marker in (": None", "[None", ", None"))
        if has_python_none:
            try:
                parsed_literal = ast.literal_eval(tool_input_str)
                if parsed_literal is None or isinstance(parsed_literal, (dict, list)):
                    return parsed_literal, True
            except (ValueError, SyntaxError):
                pass
        return tool_input_str, False

    async def agent_node(self, state: ReActGraphState, config: RunnableConfig) -> ReActGraphState:
        working_state: list[BaseMessage] = []
        for attempt in range(1, self.parse_agent_response_max_retries + 1):
            if len(state.agent_scratchpad) == 0 and len(working_state) == 0:
                if not state.messages:
                    raise RuntimeError('No input received in state: "messages"')
                content = str(state.messages[-1].content)
                if not content.strip():
                    state.messages += [AIMessage(content=NO_INPUT_ERROR_MESSAGE)]
                    return state
                question = content
                chat_history = self._get_chat_history(state.messages)
                inputs = {"question": question, "chat_history": chat_history}
                output_message = await self._stream_llm(self.agent, inputs, config=config)
            else:
                agent_scratchpad: list[BaseMessage] = []
                for index, intermediate_step in enumerate(state.agent_scratchpad):
                    agent_scratchpad.append(AIMessage(content=intermediate_step.log))
                    agent_scratchpad.append(HumanMessage(content=str(state.tool_responses[index].content)))
                agent_scratchpad += working_state
                chat_history = self._get_chat_history(state.messages)
                question = str(state.messages[-1].content)
                inputs = {
                    "question": question,
                    "agent_scratchpad": agent_scratchpad,
                    "chat_history": chat_history,
                }
                output_message = await self._stream_llm(self.agent, inputs, config=config)

            if isinstance(output_message.content, str):
                output_message.content = remove_r1_think_tags(output_message.content)
            agent_thoughts = _format_agent_thoughts_for_log(output_message)

            try:
                if (
                    self.use_native_tool_calling
                    and hasattr(output_message, "tool_calls")
                    and output_message.tool_calls
                ):
                    tool_call = output_message.tool_calls[0]
                    tool_name = tool_call.get("name", "").strip()
                    tool_args = tool_call.get("args", {})
                    tool_input_str = json.dumps(tool_args) if isinstance(tool_args, dict) else str(tool_args)
                    agent_output = AgentAction(
                        tool=tool_name,
                        tool_input=tool_input_str,
                        log=agent_thoughts or f"Calling {tool_name}",
                    )
                    state.agent_scratchpad += [agent_output]
                    return state

                agent_output = await ReActOutputParser().aparse(str(output_message.content))
                if isinstance(agent_output, AgentFinish):
                    final_answer = agent_output.return_values.get("output", output_message.content)
                    state.messages += [AIMessage(content=str(final_answer))]
                    state.final_answer = str(final_answer)
                else:
                    agent_output.log = str(output_message.content)
                    state.agent_scratchpad += [agent_output]
                return state
            except ReActOutputParserException as ex:
                content_str = str(output_message.content).strip()
                if (
                    ex.missing_action
                    and content_str
                    and not re.match(
                        r"\s*(thought\s*:?|question\s*:|previous\s+conversation)",
                        content_str,
                        re.IGNORECASE,
                    )
                ):
                    state.messages += [AIMessage(content=content_str)]
                    state.final_answer = content_str
                    return state
                if attempt == self.parse_agent_response_max_retries:
                    if self.raise_on_parsing_failure:
                        raise ReActAgentParsingFailedError(
                            observation=str(ex.observation),
                            llm_output=str(output_message.content),
                            attempts=attempt,
                        ) from ex
                    combined_content = f"{ex.observation}\n{output_message.content}"
                    state.messages += [AIMessage(content=combined_content)]
                    return state
                if output_message.content and str(output_message.content).strip():
                    working_state.append(output_message)
                    working_state.append(HumanMessage(content=str(ex.observation)))
                else:
                    working_state.append(
                        HumanMessage(
                            content=str(ex.observation)
                            + " If the available tools cannot answer the question, respond with:\n"
                            "Thought: <reasoning>\nFinal Answer: <answer>"
                        )
                    )
        return state

    async def conditional_edge(self, state: ReActGraphState) -> str:
        if state.final_answer:
            return AgentDecision.END
        if not state.agent_scratchpad:
            return AgentDecision.END
        return AgentDecision.TOOL

    async def tool_node(self, state: ReActGraphState) -> ReActGraphState:
        if not state.agent_scratchpad:
            raise RuntimeError('No tool input received in state: "agent_scratchpad"')
        agent_thoughts = state.agent_scratchpad[-1]
        requested_tool = self._get_tool(agent_thoughts.tool)
        if requested_tool is None:
            configured_tool_names = list(self.tools_dict.keys())
            tool_response = ToolMessage(
                name="agent_error",
                tool_call_id="agent_error",
                content=TOOL_NOT_FOUND_ERROR_MESSAGE.format(
                    tool_name=agent_thoughts.tool,
                    tools=configured_tool_names,
                ),
            )
            state.tool_responses += [tool_response]
            return state

        tool_input, _parsed = self._parse_tool_input(str(agent_thoughts.tool_input).strip())
        tool_response = await self._call_tool(
            requested_tool,
            tool_input,
            max_retries=self.tool_call_max_retries,
        )
        if self.detailed_logs:
            self._log_tool_response(requested_tool.name, tool_input, str(tool_response.content))
        if not self.pass_tool_call_errors_to_agent and tool_response.status == "error":
            raise RuntimeError(f"Tool call failed: {tool_response.content}")
        state.tool_responses += [tool_response]
        return state

    async def build_graph(self):
        return await self._build_graph(state_schema=ReActGraphState)


def create_react_agent_prompt(config: WorkflowSettings) -> ChatPromptTemplate:
    prompt_str = config.system_prompt or SYSTEM_PROMPT
    if config.additional_instructions:
        prompt_str += f" {config.additional_instructions}"
    for variable_name in ("{tools}", "{tool_names}"):
        if variable_name not in prompt_str:
            raise ValueError(f"Invalid system_prompt: missing {variable_name}")
    return ChatPromptTemplate(
        [
            ("system", prompt_str),
            ("user", USER_PROMPT),
            MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
        ]
    )
