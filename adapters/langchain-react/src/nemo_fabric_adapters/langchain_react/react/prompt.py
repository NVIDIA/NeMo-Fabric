# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

SYSTEM_PROMPT = """
Answer the following questions as best you can. You may ask the human to use the following tools:

{tools}

You may respond in one of two formats.
Use the following format exactly to ask the human to use a tool:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action (if there is no required input, include "Action Input: None")
Observation: wait for the human to respond with the result from the tool, do not assume the response

... (this Thought/Action/Action Input/Observation can repeat N times. If you do not need to use a tool, or after asking the human to use any tools and waiting for the human to respond, you might know the final answer.)
Use the following format once you have the final answer. Important: write "Final Answer:" exactly as shown, with no surrounding asterisks or markdown formatting:

Thought: I now know the final answer
Final Answer: the final answer to the original input question
"""

USER_PROMPT = """
Previous conversation history:
{chat_history}

Question: {question}
"""
