# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relay 0.9 compatibility for NOOA 0.0.10 tool middleware."""

from __future__ import annotations

from typing import Any

import nemo_relay
from nooa.events import _NO_RETURN
from nooa.nemo_relay_middleware import nemo_relay_agent_call_middleware
from nooa.nemo_relay_middleware import nemo_relay_llm_middleware
from nooa.runtime.middleware import MIDDLEWARE_AGENT_CALL
from nooa.runtime.middleware import MIDDLEWARE_EXECUTE_PYTHON
from nooa.runtime.middleware import MIDDLEWARE_LLM_CALL


async def _tool_middleware(ctx: Any, nxt: Any) -> Any:
    """Use NOOA's tool behavior with Relay 0.9's typed callback result."""
    args = {
        "code": ctx.code,
        **{
            key: value
            for key, value in ctx.params.items()
            if key in {"tool_call_id", "timeout"}
        },
    }
    codec = nemo_relay.typed.BestEffortAnyCodec()
    captured_ctx = None

    async def execute(inner_args: Any) -> Any:
        nonlocal captured_ctx
        if isinstance(inner_args, dict):
            if "code" in inner_args:
                ctx.code = inner_args["code"]
            for key in ("tool_call_id", "timeout"):
                if key in inner_args:
                    ctx.params[key] = inner_args[key]
        captured_ctx = await nxt(ctx)
        result = captured_ctx.result
        if result is None:
            value = None
        else:
            value = getattr(result, "returned_value", _NO_RETURN)
            if value is _NO_RETURN:
                signal = getattr(result, "signal", None)
                if signal is not None:
                    signal_data = getattr(signal, "result", None)
                    value = (
                        signal_data["result"]
                        if isinstance(signal_data, dict) and "result" in signal_data
                        else None
                    )
                else:
                    value = getattr(result, "stdout", None) or None
        return nemo_relay.ToolExecutionResult(codec.to_json(value))

    relay_result = await nemo_relay.tools.execute(
        "execute_python", args, execute, tool_call_id=ctx.params.get("tool_call_id")
    )
    if captured_ctx is not None:
        return captured_ctx
    if isinstance(relay_result, nemo_relay.ToolExecutionResult):
        from nooa.events import ExecutionResult

        ctx.result = ExecutionResult(returned_value=relay_result.result)
        return ctx
    raise RuntimeError(
        "NeMo Relay guardrail blocked code execution — the request was rejected "
        "before running. Check your NeMo Relay guardrail configuration."
    )


def install_nemo_relay_compat(event_manager: Any) -> Any:
    """Install NOOA middleware with a Relay 0.9 compatible tool callback."""
    unsub_agent = event_manager.intercept(
        MIDDLEWARE_AGENT_CALL, nemo_relay_agent_call_middleware
    )
    unsub_llm = event_manager.intercept(MIDDLEWARE_LLM_CALL, nemo_relay_llm_middleware)
    unsub_exec = event_manager.intercept(MIDDLEWARE_EXECUTE_PYTHON, _tool_middleware)

    def uninstall() -> None:
        unsub_agent()
        unsub_llm()
        unsub_exec()

    return uninstall
