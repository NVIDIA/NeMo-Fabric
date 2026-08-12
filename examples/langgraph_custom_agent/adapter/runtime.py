# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimum NeMo Fabric lifecycle for the email-phishing custom agent."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common import lifecycle

from examples.langgraph_custom_agent.adapter.configuration import (
    resolve_agent_dependencies,
)
from examples.langgraph_custom_agent.adapter.mcp import resolve_url_inspector
from examples.langgraph_custom_agent.adapter.telemetry import observe_invocation
from examples.langgraph_custom_agent.agent.graph import build_email_phishing_graph

LOGGER = logging.getLogger(__name__)


def _invocation_failure() -> dict[str, Any]:
    """Return a safe terminal failure without invalidating the runtime."""

    return {
        "response": None,
        "completed": False,
        "failed": True,
        "error": {
            "code": "email_phishing_invoke_failed",
            "message": "The email-phishing agent invocation failed",
            "retryable": False,
        },
    }


def main() -> None:
    """Serve one persistent custom-agent runtime."""

    lifecycle.serve(EmailPhishingRuntime, config_loader=AgentConfig.from_mapping)


def _runtime_context(payload: dict[str, Any]) -> RuntimeContext:
    """Decode the runtime context supplied with a lifecycle operation."""

    try:
        return RuntimeContext.from_mapping(payload.get("runtime_context"))
    except Exception as error:
        raise lifecycle.LifecycleError(
            "email_phishing_invalid_runtime_context",
            "The email-phishing adapter requires a valid RuntimeContext",
        ) from error


class EmailPhishingRuntime:
    """One compiled email-phishing graph owned by one NeMo Fabric runtime."""

    def __init__(self) -> None:
        """Initialize an unstarted custom-agent runtime."""

        self._runtime_id: str | None = None
        self._base_dir: Path | None = None
        self._agent_name: str | None = None
        self._model_name: str | None = None
        self._graph: CompiledStateGraph | None = None

    async def start(self, payload: dict[str, Any]) -> None:
        """Resolve native dependencies and retain one compiled graph."""

        if self._graph is not None:
            raise lifecycle.LifecycleError(
                "email_phishing_runtime_already_started",
                "The email-phishing runtime is already started",
            )
        agent_config = payload.get("config")
        if not isinstance(agent_config, AgentConfig):
            raise lifecycle.LifecycleError(
                "email_phishing_invalid_config",
                "The email-phishing adapter requires a validated AgentConfig",
            )

        context = _runtime_context(payload)
        dependencies = resolve_agent_dependencies(agent_config)
        url_inspector = await resolve_url_inspector(agent_config)
        graph = build_email_phishing_graph(
            dependencies.model,
            dependencies.system_instruction,
            url_inspector,
        )
        self._runtime_id = context.runtime_id
        self._base_dir = Path(payload.get("base_dir") or ".").resolve()
        self._agent_name = str(payload.get("agent_name") or "email-phishing-agent")
        self._model_name = agent_config.models["default"].model
        self._graph = graph

    async def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run one request against the retained graph and return terminal output."""

        if (
            self._graph is None
            or self._runtime_id is None
            or self._base_dir is None
            or self._agent_name is None
            or self._model_name is None
        ):
            raise lifecycle.LifecycleError(
                "email_phishing_runtime_not_started",
                "The email-phishing runtime is not started",
            )
        context = _runtime_context(payload)
        if context.runtime_id != self._runtime_id:
            raise lifecycle.LifecycleError(
                "email_phishing_runtime_mismatch",
                "The invocation does not match the active email-phishing runtime",
            )
        request = payload.get("request")
        email = request.get("input") if isinstance(request, dict) else None
        if not isinstance(email, str) or not email.strip():
            raise lifecycle.LifecycleError(
                "email_phishing_invalid_request",
                "The email-phishing adapter requires a non-empty text input",
            )

        try:
            async with observe_invocation(
                context,
                base_dir=self._base_dir,
                agent_name=self._agent_name,
                model_name=self._model_name,
            ) as telemetry:
                result = await self._graph.ainvoke(
                    {"email": email},
                    config=telemetry.runnable_config,
                )
        except Exception as error:
            LOGGER.error(
                "Email-phishing invocation failed (error_type=%s)",
                type(error).__name__,
            )
            return _invocation_failure()
        output = {
            "response": result["explanation"],
            "classification": result["classification"],
            "signals": result["signals"],
        }
        if "link_inspections" in result:
            output["link_inspections"] = result["link_inspections"]
        relay_artifacts = telemetry.artifacts()
        if relay_artifacts:
            output["relay_artifacts"] = relay_artifacts
        return output

    async def stop(self) -> None:
        """Release the compiled graph and all runtime-owned references."""

        self._runtime_id = None
        self._base_dir = None
        self._agent_name = None
        self._model_name = None
        self._graph = None


if __name__ == "__main__":
    main()
