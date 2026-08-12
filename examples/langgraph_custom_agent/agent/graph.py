# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application-defined LangGraph for the email-phishing example."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any
from typing import Literal
from typing import NotRequired
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

RiskClassification = Literal["benign", "phishing"]
URL_PATTERN = re.compile(r"https?://[^\s<>()]+")


class LinkInspection(TypedDict):
    """Validated output from the optional URL inspection tool."""

    url: str
    hostname: str
    indicators: list[str]


class EmailAnalysisState(TypedDict):
    """State accumulated while analyzing one email."""

    email: str
    signals: NotRequired[list[str]]
    link_inspections: NotRequired[list[LinkInspection]]
    classification: NotRequired[RiskClassification]
    explanation: NotRequired[str]


def extract_signals(state: EmailAnalysisState) -> dict[str, list[str]]:
    """Extract a small deterministic set of common phishing signals."""

    email = state["email"].casefold()
    signals: list[str] = []
    if any(term in email for term in ("urgent", "immediately", "act now")):
        signals.append("urgency")
    if any(
        term in email
        for term in ("password", "sign in", "login", "verify your account")
    ):
        signals.append("credential_request")
    if "http://" in email or "https://" in email:
        signals.append("external_link")
    if any(
        term in email
        for term in (
            "account locked",
            "account is locked",
            "account suspended",
            "account is suspended",
        )
    ):
        signals.append("account_threat")
    return {"signals": signals}


def _tool_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        text = [
            item["text"]
            for item in value
            if isinstance(item, Mapping) and isinstance(item.get("text"), str)
        ]
        if text:
            return "\n".join(text)
    raise TypeError("the URL inspection tool must return text content")


def _inspection(url: str, value: Any) -> LinkInspection:
    result = json.loads(_tool_text(value))
    if not isinstance(result, dict):
        raise TypeError("the URL inspection tool result must be an object")
    hostname = result.get("hostname")
    indicators = result.get("indicators")
    if not isinstance(hostname, str) or not isinstance(indicators, list):
        raise TypeError("the URL inspection tool returned an invalid result")
    if any(not isinstance(indicator, str) for indicator in indicators):
        raise TypeError("the URL inspection indicators must be strings")
    return {"url": url, "hostname": hostname, "indicators": indicators}


def classify_risk(
    state: EmailAnalysisState,
) -> dict[str, RiskClassification]:
    """Apply the example's stable, intentionally simple risk policy."""

    classification: RiskClassification = (
        "phishing" if len(state["signals"]) >= 2 else "benign"
    )
    return {"classification": classification}


def build_email_phishing_graph(
    model: BaseChatModel,
    system_instruction: str,
    url_inspector: BaseTool | None = None,
) -> CompiledStateGraph:
    """Build the custom agent from native LangChain dependencies."""

    async def inspect_links(
        state: EmailAnalysisState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        if url_inspector is None:  # pragma: no cover - node is conditionally added
            return {}
        urls = [
            match.group().rstrip(".,;:!?)]}")
            for match in URL_PATTERN.finditer(state["email"])
        ]
        inspections = [
            _inspection(
                url,
                await url_inspector.ainvoke({"url": url}, config=config),
            )
            for url in urls
        ]
        signals = list(state["signals"])
        if any(inspection["indicators"] for inspection in inspections):
            signals.append("suspicious_link")
        return {"link_inspections": inspections, "signals": signals}

    async def explain_assessment(
        state: EmailAnalysisState,
        config: RunnableConfig,
    ) -> dict[str, str]:
        signals = ", ".join(state["signals"]) or "none"
        link_inspections = state.get("link_inspections", [])
        response = await model.ainvoke(
            [
                ("system", system_instruction),
                (
                    "user",
                    "Explain this fixed email-risk assessment concisely.\n"
                    f"Classification: {state['classification']}\n"
                    f"Signals: {signals}\n"
                    f"Link inspections: {json.dumps(link_inspections)}\n"
                    f"Email:\n{state['email']}",
                ),
            ],
            config=config,
        )
        if not isinstance(response.content, str):
            raise TypeError("the explanation model must return text content")
        return {"explanation": response.content}

    builder = StateGraph(EmailAnalysisState)
    builder.add_node("extract_signals", extract_signals)
    builder.add_node("classify_risk", classify_risk)
    builder.add_node("explain_assessment", explain_assessment)
    builder.add_edge(START, "extract_signals")
    if url_inspector is None:
        builder.add_edge("extract_signals", "classify_risk")
    else:
        builder.add_node("inspect_links", inspect_links)
        builder.add_edge("extract_signals", "inspect_links")
        builder.add_edge("inspect_links", "classify_risk")
    builder.add_edge("classify_risk", "explain_assessment")
    builder.add_edge("explain_assessment", END)
    return builder.compile()
