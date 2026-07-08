# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

from langchain_core.agents import AgentAction
from langchain_core.agents import AgentFinish
from langchain_core.exceptions import LangChainException

from nemo_fabric_adapters.langchain_react.react.prompt import SYSTEM_PROMPT

FINAL_ANSWER_ACTION = "Final Answer:"
FINAL_ANSWER_PATTERN = re.compile(r"final\s+answer\s*:", re.IGNORECASE)
MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE = "Invalid Format: Missing 'Action:' after 'Thought:'"
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action Input:' after 'Action:'"
)
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = (
    "Parsing LLM output produced both a final answer and a parse-able action:"
)


class ReActAgentParsingFailedError(RuntimeError):
    def __init__(self, observation: str, llm_output: str, attempts: int):
        self.observation = observation
        self.llm_output = llm_output if len(llm_output) <= 200 else llm_output[:200] + "..."
        self.attempts = attempts
        super().__init__(
            "ReActAgentParsingFailedError: "
            f"Failed to parse agent output after {self.attempts} attempts. "
            f"Error: {self.observation}. LLM output: '{self.llm_output}'"
        )


class ReActOutputParserException(ValueError, LangChainException):
    def __init__(
        self,
        observation: str | None = None,
        *,
        missing_action: bool = False,
        missing_action_input: bool = False,
        final_answer_and_action: bool = False,
        llm_output: str | None = None,
    ):
        self.observation = observation
        self.missing_action = missing_action
        self.missing_action_input = missing_action_input
        self.final_answer_and_action = final_answer_and_action
        self.llm_output = llm_output
        super().__init__(f"ReActOutputParserException: observation={self.observation}")


class ReActOutputParser:
    def get_format_instructions(self) -> str:
        return SYSTEM_PROMPT

    async def aparse(self, text: str) -> AgentAction | AgentFinish:
        return self.parse(text)

    def parse(self, text: str) -> AgentAction | AgentFinish:
        includes_answer = bool(FINAL_ANSWER_PATTERN.search(text))
        regex_primary = (
            r"action\s*\d*\s*:\s*(.*?)\s*"
            r"(?:action\s*\d*\s*)?input\s*\d*\s*:\s*"
            r"(.*?)(?=\s*[\n|\s]\s*observation\b|$)"
        )
        action_match = re.search(regex_primary, text, re.DOTALL | re.IGNORECASE)
        if action_match:
            if includes_answer:
                raise ReActOutputParserException(
                    observation=FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE,
                    final_answer_and_action=True,
                    llm_output=text,
                )
            action = action_match.group(1).strip()
            action_input = action_match.group(2).strip(" ").strip('"')
            return AgentAction(action, action_input, text)

        if includes_answer:
            final_answer_match = FINAL_ANSWER_PATTERN.search(text)
            if final_answer_match:
                answer_text = text[final_answer_match.end() :].strip()
                return AgentFinish({"output": answer_text}, text)
            return AgentFinish({"output": text.rsplit(FINAL_ANSWER_ACTION, maxsplit=1)[-1].strip()}, text)

        if not re.search(r"action\s*\d*\s*:\s*(.*?)", text, re.DOTALL | re.IGNORECASE):
            raise ReActOutputParserException(
                observation=MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
                missing_action=True,
                llm_output=text,
            )
        if not re.search(r"[\s]*(?:action\s*\d*\s*)?input\s*\d*\s*:\s*(.*)", text, re.DOTALL | re.IGNORECASE):
            raise ReActOutputParserException(
                observation=MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
                missing_action_input=True,
                llm_output=text,
            )
        raise ReActOutputParserException("Could not parse LLM output", llm_output=text)
