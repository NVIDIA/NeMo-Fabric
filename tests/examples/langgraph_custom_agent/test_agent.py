# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Fabric-independent email-phishing graph."""

from __future__ import annotations

import asyncio
import ast
import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import tool

from examples.langgraph_custom_agent.agent import graph as graph_module
from examples.langgraph_custom_agent.agent.graph import _inspection
from examples.langgraph_custom_agent.agent.graph import build_email_phishing_graph


def test_application_graph_does_not_depend_on_fabric_or_relay():
    source_path = Path(graph_module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    assert not any(
        module == "nemo_fabric" or module.startswith("nemo_fabric.")
        for module in imported_modules
    )
    assert not any(
        module == "nemo_relay" or module.startswith("nemo_relay.")
        for module in imported_modules
    )


def test_graph_keeps_classification_deterministic_and_uses_model_for_explanation():
    graph = build_email_phishing_graph(
        FakeListChatModel(responses=["The email combines several phishing signals."]),
        "Explain the fixed assessment.",
    )

    result = asyncio.run(
        graph.ainvoke(
            {
                "email": (
                    "Urgent: your account is locked. Verify your password at "
                    "https://example.invalid."
                )
            }
        )
    )

    assert result["classification"] == "phishing"
    assert result["signals"] == [
        "urgency",
        "credential_request",
        "external_link",
        "account_threat",
    ]
    assert result["explanation"] == (
        "The email combines several phishing signals."
    )


def test_graph_uses_an_optional_native_url_inspection_tool():
    @tool
    async def inspect_url(url: str) -> list[dict[str, str]]:
        """Inspect one URL."""

        return [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "hostname": "example.invalid",
                        "indicators": ["reserved_test_domain"],
                    }
                ),
            }
        ]

    graph = build_email_phishing_graph(
        FakeListChatModel(responses=["The link adds another risk signal."]),
        "Explain the fixed assessment.",
        inspect_url,
    )

    result = asyncio.run(
        graph.ainvoke({"email": "Review https://example.invalid/login."})
    )

    assert result["classification"] == "phishing"
    assert result["signals"] == [
        "credential_request",
        "external_link",
        "suspicious_link",
    ]
    assert result["link_inspections"] == [
        {
            "url": "https://example.invalid/login",
            "hostname": "example.invalid",
            "indicators": ["reserved_test_domain"],
        }
    ]


@pytest.mark.parametrize(
    ("tool_result", "error_type", "message"),
    [
        ("not JSON", json.JSONDecodeError, None),
        (json.dumps([]), TypeError, "must be an object"),
        (
            json.dumps({"hostname": 7, "indicators": []}),
            TypeError,
            "returned an invalid result",
        ),
        (
            json.dumps({"hostname": "example.invalid", "indicators": [7]}),
            TypeError,
            "indicators must be strings",
        ),
        ([], TypeError, "must return text content"),
    ],
)
def test_url_inspection_rejects_malformed_tool_results(
    tool_result,
    error_type,
    message,
):
    with pytest.raises(error_type, match=message):
        _inspection("https://example.invalid", tool_result)
