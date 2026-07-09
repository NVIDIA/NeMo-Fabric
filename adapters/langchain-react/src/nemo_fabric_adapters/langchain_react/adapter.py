#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fabric adapter entrypoint for LangChain ReAct agents."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from langgraph.errors import GraphRecursionError

CUR_DIR = Path(__file__).parent
ADAPTERS_DIR = CUR_DIR.parent.parent.parent.parent
LANGCHAIN_REACT_SRC = (ADAPTERS_DIR / "langchain-react" / "src").resolve().as_posix()
COMMON_DIR = (ADAPTERS_DIR / "common" / "src").resolve().as_posix()
for path in (LANGCHAIN_REACT_SRC, COMMON_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import nemo_fabric_adapters.common.utils as common_utils  # noqa: E402

from nemo_fabric_adapters.langchain_react.config import harness_settings  # noqa: E402
from nemo_fabric_adapters.langchain_react.config import models_payload  # noqa: E402
from nemo_fabric_adapters.langchain_react.config import request_messages  # noqa: E402
from nemo_fabric_adapters.langchain_react.config import request_overrides  # noqa: E402
from nemo_fabric_adapters.langchain_react.config import request_timezone  # noqa: E402
from nemo_fabric_adapters.langchain_react.config import tools_config  # noqa: E402
from nemo_fabric_adapters.langchain_react.config import workflow_settings  # noqa: E402
from nemo_fabric_adapters.langchain_react.llm import build_chat_model  # noqa: E402
from nemo_fabric_adapters.langchain_react.llm import resolve_model_config  # noqa: E402
from nemo_fabric_adapters.langchain_react.react.graph import ReActAgentGraph  # noqa: E402
from nemo_fabric_adapters.langchain_react.react.graph import ReActGraphState  # noqa: E402
from nemo_fabric_adapters.langchain_react.react.graph import create_react_agent_prompt  # noqa: E402
from nemo_fabric_adapters.langchain_react.tools import ToolResolutionContext  # noqa: E402
from nemo_fabric_adapters.langchain_react.tools import resolve_tools  # noqa: E402


def main() -> None:
    payload = common_utils.load_payload()
    output = run(payload)
    print(json.dumps(output, sort_keys=True))
    if output.get("failed"):
        raise SystemExit(2)


def run(payload: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(run_langchain_react(payload))


async def run_langchain_react(payload: dict[str, Any]) -> dict[str, Any]:
    workflow = workflow_settings(payload)
    settings = harness_settings(payload)
    models = models_payload(payload)
    overrides = request_overrides(payload)
    model_config = resolve_model_config(models, workflow.llm_name, settings, overrides)

    relay_enabled = os.environ.get("FABRIC_RELAY_ENABLED", "").strip().lower() == "true"
    relay_plugin_config = None
    if relay_enabled:
        relay_plugin_config = common_utils.load_relay_plugin_config(payload)

    if relay_plugin_config is not None:
        relay_api_config = common_utils.relay_api_plugin_config(relay_plugin_config)
        from nemo_relay import plugin

        async with plugin.plugin(relay_api_config):
            output = await _invoke_graph(
                payload,
                workflow=workflow,
                models=models,
                settings=settings,
                overrides=overrides,
                model_config=model_config,
                relay_active=True,
            )
        return _attach_relay_output(output, relay_plugin_config)

    return await _invoke_graph(
        payload,
        workflow=workflow,
        models=models,
        settings=settings,
        overrides=overrides,
        model_config=model_config,
        relay_active=False,
    )


def _attach_relay_output(output: dict[str, Any], relay_plugin_config: dict[str, Any]) -> dict[str, Any]:
    # ATIF files are finalized when the relay plugin context exits; collect after shutdown.
    output["relay_runtime"] = {
        "enabled": True,
        "mode": os.environ.get("FABRIC_RELAY_MODE"),
        "config_path": os.environ.get("FABRIC_RELAY_CONFIG_PATH"),
        "emitter": "nemo-relay",
    }
    output["relay_artifacts"] = common_utils.collect_relay_artifacts(relay_plugin_config)
    return output


def _build_tool_llm(
    models: dict[str, Any],
    settings: dict[str, Any],
    overrides: Any | None,
) -> Any:
    def build_llm(llm_name: str):
        tool_model_config = resolve_model_config(models, llm_name, settings, overrides)
        return build_chat_model(tool_model_config)

    return build_llm


def _relay_callbacks() -> list[Any]:
    from nemo_relay.integrations.langgraph.callbacks import NemoRelayCallbackHandler

    return [NemoRelayCallbackHandler()]


async def _invoke_graph(
    payload: dict[str, Any],
    *,
    workflow,
    models: dict[str, Any],
    settings: dict[str, Any],
    overrides: Any | None,
    model_config: dict[str, Any],
    relay_active: bool,
) -> dict[str, Any]:
    error: str | None = None
    response: str | None = None
    messages: list[dict[str, Any]] = []
    recursion_limit_exceeded = False
    try:
        llm = build_chat_model(model_config)
        tool_context = ToolResolutionContext(
            timezone=request_timezone(payload),
            build_llm=_build_tool_llm(models, settings, overrides),
        )
        tools = resolve_tools(workflow.tool_names, tools_config(payload), context=tool_context)
        prompt = create_react_agent_prompt(workflow)
        relay_callbacks = _relay_callbacks() if relay_active else None
        graph_builder = ReActAgentGraph(
            llm=llm,
            prompt=prompt,
            tools=tools,
            config=workflow,
            callbacks=relay_callbacks,
        )
        graph = await graph_builder.build_graph()
        state = ReActGraphState(
            messages=request_messages(payload, max_history=workflow.max_history),
        )
        recursion_limit = (workflow.max_tool_calls + 1) * 2
        result_state = await graph.ainvoke(state, config={"recursion_limit": recursion_limit})
        final_state = ReActGraphState.model_validate(result_state)
        if final_state.final_answer:
            response = final_state.final_answer
        elif final_state.messages:
            response = str(final_state.messages[-1].content)
        else:
            response = ""
        messages = [
            {"role": message.type, "content": str(message.content)}
            for message in final_state.messages
        ]
    except GraphRecursionError:
        recursion_limit_exceeded = True
        error = (
            f"The react agent could not produce a final answer within {workflow.max_tool_calls} "
            "iterations. The agent repeatedly called tools without converging on a response."
        )
        response = error
    except Exception as exc:
        error = str(exc)

    output: dict[str, Any] = {
        "harness": "langchain-react",
        "adapter": "python",
        "mode": "react_agent",
        "model": model_config.get("model"),
        "base_url": model_config.get("base_url"),
        "temperature": model_config.get("temperature"),
        "top_p": model_config.get("top_p"),
        "response": response,
        "completed": error is None,
        "failed": error is not None,
        "messages": messages,
        "error": error,
        "tool_names": workflow.tool_names,
        "use_native_tool_calling": workflow.use_native_tool_calling,
        "recursion_limit_exceeded": recursion_limit_exceeded,
    }

    request = payload.get("request") or {}
    if request.get("request_id"):
        output["request_id"] = request["request_id"]

    return output


if __name__ == "__main__":
    main()
