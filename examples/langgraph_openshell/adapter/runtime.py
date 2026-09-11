# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent NeMo Fabric lifecycle for the Portable Courier graph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from nemo_fabric_adapter_contract.models import AgentArtifact
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunResult
from nemo_fabric_adapter_contract.models import AgentRunStatus
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common import lifecycle

from examples.langgraph_openshell.agent.graph import build_courier_graph


def main() -> None:
    """Serve one persistent graph runtime."""

    lifecycle.serve(PortableCourierRuntime, config_loader=AgentConfig.from_mapping)


class PortableCourierRuntime:
    """One compiled graph and checkpoint namespace per Fabric runtime."""

    def __init__(self) -> None:
        self._runtime_id: str | None = None
        self._artifact_root: Path | None = None
        self._graph: CompiledStateGraph | None = None

    async def start(self, payload: dict[str, Any]) -> None:
        config = payload.get("config")
        if not isinstance(config, AgentConfig):
            raise lifecycle.LifecycleError(
                "portable_courier_invalid_config",
                "Portable Courier requires a validated AgentConfig",
            )
        try:
            context = RuntimeContext.from_mapping(payload.get("runtime_context"))
        except Exception as error:
            raise lifecycle.LifecycleError(
                "portable_courier_invalid_runtime_context",
                "Portable Courier requires a valid RuntimeContext",
            ) from error
        if context.environment.artifacts is None:
            raise lifecycle.LifecycleError(
                "portable_courier_artifact_root_required",
                "Portable Courier requires an environment artifact root",
            )
        self._runtime_id = context.runtime_id
        self._artifact_root = Path(context.environment.artifacts)
        self._graph = build_courier_graph()

    async def invoke(
        self,
        request: AgentRunRequest,
        context: RuntimeContext,
    ) -> AgentRunResult:
        if (
            self._graph is None
            or self._runtime_id is None
            or self._artifact_root is None
        ):
            raise lifecycle.LifecycleError(
                "portable_courier_runtime_not_started",
                "Portable Courier is not started",
            )
        if context.runtime_id != self._runtime_id:
            raise lifecycle.LifecycleError(
                "portable_courier_runtime_mismatch",
                "The invocation does not match the active Portable Courier runtime",
            )
        if not isinstance(request.input, str) or request.input not in {
            "route",
            "deliver",
        }:
            raise lifecycle.LifecycleError(
                "portable_courier_invalid_request",
                "Portable Courier accepts only `route` or `deliver`",
            )

        result = self._graph.invoke(
            {"command": request.input, "attempts": []},
            config={"configurable": {"thread_id": self._runtime_id}},
        )
        output = {
            "command": request.input,
            "attempts": result.get("attempts", []),
            "selected_route": result.get("selected_route"),
            "delivery_status": result.get("delivery_status", "routed"),
        }
        artifacts: list[AgentArtifact] = []
        if request.input == "deliver" and result.get("delivery_status") == "delivered":
            self._artifact_root.mkdir(parents=True, exist_ok=True)
            receipt = {
                "runtime_id": self._runtime_id,
                "delivery_status": "delivered",
                "selected_route": result["selected_route"],
                "attempts": result.get("attempts", []),
            }
            (self._artifact_root / "delivery-receipt.json").write_text(
                json.dumps(receipt, indent=2) + "\n",
                encoding="utf-8",
            )
            artifacts.append(
                AgentArtifact(
                    name="delivery-receipt",
                    kind="receipt",
                    path="delivery-receipt.json",
                    media_type="application/json",
                )
            )
        return AgentRunResult(
            status=AgentRunStatus.SUCCEEDED,
            output=output,
            artifacts=artifacts,
        )

    async def stop(self) -> None:
        self._runtime_id = None
        self._artifact_root = None
        self._graph = None


if __name__ == "__main__":
    main()
