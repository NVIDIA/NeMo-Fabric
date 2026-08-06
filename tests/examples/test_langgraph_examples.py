# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline coverage for the native LangGraph examples."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from examples.langgraph.calculator_mcp import PerUserReactAgent
from examples.langgraph.calculator_mcp import current_timezone
from examples.langgraph.config import load_config
from examples.langgraph.email_phishing_analyzer import PhishingAssessment
from examples.langgraph.email_phishing_analyzer import build_email_phishing_analyzer

ROOT_DIR = Path(__file__).resolve().parents[2]
CALCULATOR_CONFIG = ROOT_DIR / "examples/langgraph/configs/calculator_mcp.yaml"
PHISHING_CONFIG = ROOT_DIR / "examples/langgraph/configs/email_phishing_analyzer.yaml"


def test_calculator_config_preserves_requested_mcp_and_workflow_shape():
    config = load_config(CALCULATOR_CONFIG)

    assert config.selected_model().model == "meta/llama-3.1-70b-instruct"
    assert config.mcp is not None
    assert config.mcp.servers["mcp_math"].transport == "streamable-http"
    assert config.workflow.entrypoint == "langgraph:per_user_react_agent"
    assert config.workflow.settings["tool_names"] == ["current_timezone", "mcp_math"]


async def test_calculator_creates_isolated_graph_and_mcp_client_per_user():
    config = load_config(CALCULATOR_CONFIG)
    mock_tools = []
    for name in config.mcp.servers["mcp_math"].include:  # validated by the example
        mock_tool = MagicMock()
        mock_tool.name = name
        mock_tools.append(mock_tool)
    mock_client_factory = MagicMock()
    mock_client_factory.side_effect = [
        MagicMock(get_tools=AsyncMock(return_value=mock_tools))
        for _ in range(2)
    ]
    mock_graph_factory = MagicMock(side_effect=[MagicMock(), MagicMock()])
    mock_model_factory = MagicMock(side_effect=[MagicMock(), MagicMock()])

    agent = PerUserReactAgent(
        config,
        model_factory=mock_model_factory,
        mcp_client_factory=mock_client_factory,
        graph_factory=mock_graph_factory,
    )
    alice_first = await agent.graph_for("alice")
    alice_second = await agent.graph_for("alice")
    hatter = await agent.graph_for("hatter")

    assert alice_first is alice_second
    assert alice_first is not hatter
    assert mock_model_factory.call_count == 2
    assert mock_client_factory.call_count == 2
    connection = mock_client_factory.call_args.args[0]["mcp_math"]
    assert connection == {
        "transport": "streamable_http",
        "url": "http://localhost:9901/mcp",
    }
    for call in mock_graph_factory.call_args_list:
        assert call.kwargs["checkpointer"] is not None
        assert call.kwargs["name"] == "per_user_calculator"


def test_current_timezone_uses_the_explicit_server_configuration(restore_environ):
    restore_environ["TZ"] = "America/Los_Angeles"

    assert current_timezone.invoke({}) == "America/Los_Angeles"


async def test_phishing_graph_projects_a_json_safe_structured_assessment():
    config = load_config(PHISHING_CONFIG)
    mock_structured_model = MagicMock()
    mock_structured_model.ainvoke = AsyncMock(
        return_value=PhishingAssessment(
            is_likely_phishing=True,
            explanation="It asks for banking information to complete a refund.",
        )
    )
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value = mock_structured_model

    graph = build_email_phishing_analyzer(config, model=mock_model)
    result = await graph.ainvoke(
        {"body": "Provide your routing number so we can issue a refund."}
    )

    assert result["assessment"] == {
        "is_likely_phishing": True,
        "explanation": "It asks for banking information to complete a refund.",
    }
    mock_model.with_structured_output.assert_called_once_with(
        PhishingAssessment, method="function_calling"
    )
