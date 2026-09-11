# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import RuntimeContext

from examples.langgraph_openshell.adapter import runtime as runtime_module
from examples.langgraph_openshell.agent.graph import build_courier_graph


def test_graph_retains_the_fallback_route_for_the_delivery_turn() -> None:
    outcomes = iter(["http_403", "http_200"])
    graph = build_courier_graph(lambda _: next(outcomes))
    config = {"configurable": {"thread_id": "runtime-1"}}

    routed = graph.invoke({"command": "route", "attempts": []}, config=config)
    delivered = graph.invoke({"command": "deliver", "attempts": []}, config=config)

    assert routed["selected_route"] == "https://example.com/"
    assert [attempt["outcome"] for attempt in routed["attempts"]] == [
        "http_403",
        "http_200",
    ]
    assert delivered["selected_route"] == routed["selected_route"]
    assert delivered["delivery_status"] == "delivered"


@pytest.mark.asyncio
async def test_adapter_returns_a_receipt_from_the_runtime_artifact_root(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        runtime_module,
        "build_courier_graph",
        lambda: build_courier_graph(
            lambda url: "http_403" if url.endswith("priority-lane") else "http_200"
        ),
    )
    context = RuntimeContext.from_mapping(
        {
            "runtime_id": "runtime-1",
            "invocation_id": "invocation-1",
            "request_id": "request-1",
            "environment": {
                "environment_id": "environment-1",
                "provider": "openshell",
                "control_location": "in_env_control",
                "workspace": "/sandbox",
                "artifacts": str(tmp_path),
                "ownership": "fabric_owned",
            },
            "artifacts": {"root": str(tmp_path)},
        }
    )
    runtime = runtime_module.PortableCourierRuntime()
    await runtime.start(
        {
            "config": AgentConfig(),
            "runtime_context": context.to_mapping(),
        }
    )

    await runtime.invoke(AgentRunRequest(input="route"), context)
    result = await runtime.invoke(AgentRunRequest(input="deliver"), context)

    assert result.output["delivery_status"] == "delivered"
    assert result.artifacts[0].path == "delivery-receipt.json"
    receipt = json.loads((tmp_path / "delivery-receipt.json").read_text())
    assert receipt["selected_route"] == "https://example.com/"
    await runtime.stop()
