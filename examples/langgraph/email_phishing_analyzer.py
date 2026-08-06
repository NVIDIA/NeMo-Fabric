# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A LangGraph workflow that returns a structured phishing assessment."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any
from typing import TypedDict

from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from pydantic import BaseModel
from pydantic import Field

from examples.langgraph.config import LangGraphExampleConfig
from examples.langgraph.config import build_nim_chat_model
from examples.langgraph.config import load_config

DEFAULT_PROMPT = """Examine the email below for signs of phishing.

Look for suspicious requests, urgency, generic greetings, grammar mistakes,
unusual payment requests, and emotional manipulation. Return a structured,
evidence-based assessment.

Email content:
{body}
"""


class PhishingAssessment(BaseModel):
    """The JSON-safe result projected from the email-analysis graph."""

    is_likely_phishing: bool
    explanation: str = Field(min_length=1)


class EmailPhishingState(TypedDict):
    """Application-owned graph state for one email analysis."""

    body: str
    assessment: dict[str, Any]


def build_email_phishing_analyzer(
    config: LangGraphExampleConfig, *, model: Any | None = None
) -> Any:
    """Build the phishing graph selected by the example workflow configuration."""

    if config.workflow.entrypoint != "langgraph:email_phishing_analyzer":
        raise ValueError("phishing example requires langgraph:email_phishing_analyzer")
    prompt = str(config.workflow.settings.get("prompt", DEFAULT_PROMPT))
    chat_model = model or build_nim_chat_model(config.selected_model())
    structured_model = chat_model.with_structured_output(
        PhishingAssessment, method="function_calling"
    )

    async def analyze_email(state: EmailPhishingState) -> dict[str, Any]:
        assessment = await structured_model.ainvoke(prompt.format(body=state["body"]))
        if not isinstance(assessment, PhishingAssessment):
            assessment = PhishingAssessment.model_validate(assessment)
        return {"assessment": assessment.model_dump()}

    graph = StateGraph(EmailPhishingState)
    graph.add_node("analyze_email", analyze_email)
    graph.add_edge(START, "analyze_email")
    graph.add_edge("analyze_email", END)
    return graph.compile(name="email_phishing_analyzer")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="examples/langgraph/configs/email_phishing_analyzer.yaml"
    )
    parser.add_argument("--input", required=True, help="Email body to analyze.")
    return parser.parse_args()


async def _run() -> None:
    args = _parse_args()
    graph = build_email_phishing_analyzer(load_config(args.config))
    result = await graph.ainvoke({"body": args.input})
    print(json.dumps(result["assessment"], indent=2))


def main() -> None:
    """Run the phishing analyzer example from the command line."""

    asyncio.run(_run())


if __name__ == "__main__":
    main()
