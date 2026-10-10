# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential-free stateful LangGraph for the OpenShell example."""

from __future__ import annotations

import operator
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Annotated
from typing import Literal
from typing import NotRequired
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

CourierCommand = Literal["route", "deliver"]


class RouteAttempt(TypedDict):
    """One observable route attempt made by the graph."""

    route: str
    outcome: str


class CourierState(TypedDict):
    """State retained by LangGraph across one Fabric runtime session."""

    command: CourierCommand
    attempts: Annotated[list[RouteAttempt], operator.add]
    selected_route: NotRequired[str]
    delivery_status: NotRequired[str]


def http_probe(url: str) -> str:
    """Probe one harmless public endpoint and return a compact outcome."""

    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            return f"http_{response.status}"
    except urllib.error.HTTPError as error:
        return f"http_{error.code}"
    except (OSError, urllib.error.URLError) as error:
        return f"unavailable_{type(error).__name__}"


def build_courier_graph(
    probe: Callable[[str], str] = http_probe,
) -> CompiledStateGraph:
    """Build a graph whose checkpoint is scoped by the caller's thread id."""

    def choose_operation(state: CourierState) -> str:
        return state["command"]

    def route(_: CourierState) -> dict[str, object]:
        preferred = "https://example.com/priority-lane"
        fallback = "https://example.com/"
        preferred_outcome = probe(preferred)
        attempts: list[RouteAttempt] = [
            {"route": preferred, "outcome": preferred_outcome}
        ]
        if preferred_outcome == "http_200":
            return {"attempts": attempts, "selected_route": preferred}

        fallback_outcome = probe(fallback)
        attempts.append({"route": fallback, "outcome": fallback_outcome})
        selected = (
            fallback if fallback_outcome == "http_200" else "local://offline-ledger"
        )
        return {"attempts": attempts, "selected_route": selected}

    def deliver(state: CourierState) -> dict[str, str]:
        if not state.get("selected_route"):
            return {"delivery_status": "route_required"}
        return {"delivery_status": "delivered"}

    builder = StateGraph(CourierState)
    builder.add_node("route", route)
    builder.add_node("deliver", deliver)
    builder.add_conditional_edges(
        START, choose_operation, {"route": "route", "deliver": "deliver"}
    )
    builder.add_edge("route", END)
    builder.add_edge("deliver", END)
    return builder.compile(checkpointer=InMemorySaver())
