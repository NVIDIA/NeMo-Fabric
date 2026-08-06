#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a user-owned graph factory through the NVIDIA NeMo Fabric lifecycle host."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nemo_fabric_adapters.common import lifecycle
import nemo_fabric_adapters.common.utils as common_utils


FACTORY_KIND = "langgraph_factory"


class _WorkflowError(lifecycle.LifecycleError):
    """A malformed graph-factory workflow configuration."""

    def __init__(self, message: str) -> None:
        super().__init__("langgraph_invalid_workflow", message)


def main() -> None:
    """Serve the persistent local-host lifecycle protocol."""

    lifecycle.serve(LangGraphRuntime)


def _workflow(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    config = common_utils.fabric_config(payload)
    workflow = config.get("workflow")
    if not isinstance(workflow, Mapping):
        raise _WorkflowError("workflow must be an object")

    entrypoint = workflow.get("entrypoint")
    if not isinstance(entrypoint, Mapping):
        raise _WorkflowError("workflow.entrypoint must be an object")
    if entrypoint.get("kind") != FACTORY_KIND:
        raise _WorkflowError(f"workflow.entrypoint.kind must equal {FACTORY_KIND!r}")

    ref = entrypoint.get("ref")
    if not isinstance(ref, str) or not _is_factory_ref(ref):
        raise _WorkflowError(
            "workflow.entrypoint.ref must use module:factory syntax with Python identifiers"
        )

    settings = workflow.get("settings", {})
    if not isinstance(settings, Mapping):
        raise _WorkflowError("workflow.settings must be an object")
    return ref, dict(settings)


def _is_factory_ref(value: str) -> bool:
    module, separator, attribute = value.partition(":")
    if not separator or not module or not attribute or ":" in attribute:
        return False
    return all(part.isidentifier() for part in module.split(".")) and all(
        part.isidentifier() for part in attribute.split(".")
    )


def _factory_from_ref(ref: str, base_dir: str) -> Any:
    module_name, attribute_path = ref.split(":", maxsplit=1)
    resolved_base_dir = Path(base_dir).resolve()
    if not resolved_base_dir.is_dir():
        raise _WorkflowError("base_dir must name an existing directory")
    base_dir_text = str(resolved_base_dir)
    if base_dir_text not in sys.path:
        sys.path.insert(0, base_dir_text)
    importlib.invalidate_caches()

    try:
        target: Any = importlib.import_module(module_name)
    except Exception as error:
        raise lifecycle.LifecycleError(
            "langgraph_factory_import_failed",
            f"Could not import the configured graph factory module {module_name!r}",
        ) from error

    for attribute in attribute_path.split("."):
        try:
            target = getattr(target, attribute)
        except AttributeError as error:
            raise lifecycle.LifecycleError(
                "langgraph_factory_import_failed",
                f"Configured graph factory {ref!r} was not found",
            ) from error
    if not callable(target):
        raise lifecycle.LifecycleError(
            "langgraph_factory_invalid",
            f"Configured graph factory {ref!r} is not callable",
        )
    return target


async def _await_if_needed(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _json_value(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise lifecycle.LifecycleError(
            "langgraph_result_not_json_serializable",
            "Graph output must be JSON-serializable",
        ) from error


class LangGraphRuntime:
    """Own one compiled graph for the complete NVIDIA NeMo Fabric runtime."""

    def __init__(self) -> None:
        self._graph: Any | None = None

    async def start(self, payload: dict[str, Any]) -> None:
        """Load the configured factory and compile its graph once."""

        if self._graph is not None:
            raise lifecycle.LifecycleError(
                "langgraph_runtime_already_started",
                "LangGraph runtime is already started",
            )

        ref, settings = _workflow(payload)
        factory = _factory_from_ref(ref, common_utils.base_dir(payload))
        try:
            graph = await _await_if_needed(factory(**settings))
        except Exception as error:
            raise lifecycle.LifecycleError(
                "langgraph_factory_failed",
                f"Configured graph factory {ref!r} failed during runtime start",
            ) from error

        compile_graph = getattr(graph, "compile", None)
        if not callable(compile_graph):
            raise lifecycle.LifecycleError(
                "langgraph_graph_not_compilable",
                f"Configured graph factory {ref!r} must return an uncompiled graph with compile()",
            )
        try:
            compiled_graph = await _await_if_needed(compile_graph())
        except Exception as error:
            raise lifecycle.LifecycleError(
                "langgraph_graph_compile_failed",
                f"Configured graph from {ref!r} failed to compile",
            ) from error

        if not callable(getattr(compiled_graph, "ainvoke", None)) and not callable(
            getattr(compiled_graph, "invoke", None)
        ):
            raise lifecycle.LifecycleError(
                "langgraph_graph_not_invokable",
                f"Compiled graph from {ref!r} must provide ainvoke(input) or invoke(input)",
            )
        self._graph = compiled_graph

    async def invoke(self, payload: dict[str, Any]) -> Any:
        """Invoke the compiled graph with the raw NVIDIA NeMo Fabric input."""

        graph = self._graph
        if graph is None:
            raise lifecycle.LifecycleError(
                "langgraph_runtime_not_started",
                "LangGraph runtime has not started a graph",
            )
        request = common_utils.request_payload(payload)
        if not isinstance(request, Mapping):
            raise lifecycle.LifecycleError(
                "langgraph_invalid_request",
                "Graph invocation request must be an object",
            )
        graph_input = request.get("input")
        try:
            ainvoke = getattr(graph, "ainvoke", None)
            if callable(ainvoke):
                result = await _await_if_needed(ainvoke(graph_input))
            else:
                result = await asyncio.to_thread(getattr(graph, "invoke"), graph_input)
                result = await _await_if_needed(result)
        except Exception as error:
            raise lifecycle.LifecycleError(
                "langgraph_graph_invoke_failed",
                "Compiled graph failed during invocation",
            ) from error
        return _json_value(result)

    async def stop(self) -> None:
        """Release the adapter-owned graph reference."""

        self._graph = None


if __name__ == "__main__":
    main()
