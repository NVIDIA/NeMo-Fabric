# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct coverage for the temporary NOOA Relay 0.9 middleware."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import call

import pytest

if not ((3, 12) <= sys.version_info[:2] < (3, 14)):
    pytest.skip("NOOA supports Python 3.12 and 3.13", allow_module_level=True)

pytest.importorskip("nooa")

from nooa.events import ExecutionResult
from nooa.events import _NO_RETURN
from nooa.runtime.middleware import ExecutePythonContext
from nooa.runtime.middleware import MIDDLEWARE_AGENT_CALL
from nooa.runtime.middleware import MIDDLEWARE_EXECUTE_PYTHON
from nooa.runtime.middleware import MIDDLEWARE_LLM_CALL
from nemo_fabric_adapters.nooa import relay_compat


@pytest.mark.parametrize(
    ("tool_result", "expected"),
    [
        pytest.param(None, None, id="none"),
        pytest.param(
            SimpleNamespace(returned_value=["fixture"]), ["fixture"], id="list"
        ),
        pytest.param(
            SimpleNamespace(
                returned_value=_NO_RETURN,
                signal=SimpleNamespace(result={"result": {"answer": 42}}),
            ),
            {"answer": 42},
            id="signal",
        ),
        pytest.param(
            SimpleNamespace(returned_value=_NO_RETURN, signal=None, stdout="output"),
            "output",
            id="stdout",
        ),
    ],
)
async def test_tool_callback_wraps_extracted_result(
    monkeypatch: pytest.MonkeyPatch, tool_result: Any, expected: Any
):
    ctx = SimpleNamespace(code="original", params={"timeout": 1}, result=None)

    async def next_middleware(context: Any) -> Any:
        context.result = tool_result
        return context

    async def relay_execute(
        name: str, args: dict[str, Any], callback: Any, *, tool_call_id: str | None
    ) -> Any:
        assert name == "execute_python"
        assert tool_call_id is None
        assert args == {"code": "original", "timeout": 1}
        wrapped = await callback(args)
        assert isinstance(wrapped, relay_compat.nemo_relay.ToolExecutionResult)
        assert wrapped.result == expected
        return wrapped

    monkeypatch.setattr(relay_compat.nemo_relay.tools, "execute", relay_execute)
    assert await relay_compat._tool_middleware(ctx, next_middleware) is ctx


async def test_tool_callback_applies_intercepted_code_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    ctx = SimpleNamespace(
        code="original",
        params={"timeout": 1, "tool_call_id": "original-id", "other": "private"},
        result=None,
    )

    async def next_middleware(context: Any) -> Any:
        assert context.code == "rewritten"
        assert context.params == {
            "timeout": 9,
            "tool_call_id": "rewritten-id",
            "other": "private",
        }
        context.result = SimpleNamespace(returned_value="done")
        return context

    async def relay_execute(
        name: str, args: dict[str, Any], callback: Any, *, tool_call_id: str | None
    ) -> Any:
        assert name == "execute_python"
        assert tool_call_id == "original-id"
        assert args == {
            "code": "original",
            "timeout": 1,
            "tool_call_id": "original-id",
        }
        return await callback(
            {"code": "rewritten", "timeout": 9, "tool_call_id": "rewritten-id"}
        )

    monkeypatch.setattr(relay_compat.nemo_relay.tools, "execute", relay_execute)
    assert await relay_compat._tool_middleware(ctx, next_middleware) is ctx


async def test_tool_callback_rejection_skips_execution(monkeypatch: pytest.MonkeyPatch):
    ctx = SimpleNamespace(code="original", params={}, result=None)
    next_middleware = MagicMock()

    async def relay_execute(
        _name: str, _args: dict[str, Any], _callback: Any, *, tool_call_id: str | None
    ) -> None:
        return None

    monkeypatch.setattr(relay_compat.nemo_relay.tools, "execute", relay_execute)
    with pytest.raises(RuntimeError, match="guardrail blocked code execution"):
        await relay_compat._tool_middleware(ctx, next_middleware)
    next_middleware.assert_not_called()


async def test_tool_callback_propagates_execution_error(
    monkeypatch: pytest.MonkeyPatch,
):
    ctx = SimpleNamespace(code="original", params={}, result=None)

    async def next_middleware(_context: Any) -> Any:
        raise ValueError("execution failed")

    async def relay_execute(
        _name: str, args: dict[str, Any], callback: Any, *, tool_call_id: str | None
    ) -> Any:
        return await callback(args)

    monkeypatch.setattr(relay_compat.nemo_relay.tools, "execute", relay_execute)
    with pytest.raises(ValueError, match="execution failed"):
        await relay_compat._tool_middleware(ctx, next_middleware)


async def test_tool_execution_intercept_can_short_circuit(
    monkeypatch: pytest.MonkeyPatch,
):
    ctx = ExecutePythonContext(code="original", params={"tool_call_id": "call-1"})
    next_middleware = MagicMock()

    async def relay_execute(
        name: str, args: dict[str, Any], callback: Any, *, tool_call_id: str | None
    ) -> Any:
        assert name == "execute_python"
        assert args == {"code": "original", "tool_call_id": "call-1"}
        assert tool_call_id == "call-1"
        return relay_compat.nemo_relay.ToolExecutionResult({"answer": 42})

    monkeypatch.setattr(relay_compat.nemo_relay.tools, "execute", relay_execute)
    returned = await relay_compat._tool_middleware(ctx, next_middleware)
    assert returned is ctx
    assert isinstance(ctx.result, ExecutionResult)
    assert ctx.result.returned_value == {"answer": 42}
    next_middleware.assert_not_called()


async def test_real_relay_execution_intercept_can_short_circuit():
    ctx = ExecutePythonContext(code="original", params={"tool_call_id": "call-1"})
    next_middleware = MagicMock()

    async def short_circuit(context: Any, _next_call: Any) -> Any:
        assert context.tool_call_id == "call-1"
        return relay_compat.nemo_relay.ToolExecutionInterceptOutcome({"answer": 42})

    name = "nooa-compat-short-circuit-test"
    relay_compat.nemo_relay.intercepts.register_tool_execution(name, 0, short_circuit)
    try:
        with relay_compat.nemo_relay.scope.scope(
            "nooa-compat-test", relay_compat.nemo_relay.ScopeType.Agent
        ):
            returned = await relay_compat._tool_middleware(ctx, next_middleware)
    finally:
        relay_compat.nemo_relay.intercepts.deregister_tool_execution(name)

    assert returned is ctx
    assert isinstance(ctx.result, ExecutionResult)
    assert ctx.result.returned_value == {"answer": 42}
    next_middleware.assert_not_called()


def test_installer_removes_all_three_handlers():
    event_manager = MagicMock()
    unsubscribers = [MagicMock() for _ in range(3)]
    event_manager.intercept.side_effect = unsubscribers

    uninstall = relay_compat.install_nemo_relay_compat(event_manager)
    assert event_manager.intercept.call_args_list == [
        call(MIDDLEWARE_AGENT_CALL, relay_compat.nemo_relay_agent_call_middleware),
        call(MIDDLEWARE_LLM_CALL, relay_compat.nemo_relay_llm_middleware),
        call(MIDDLEWARE_EXECUTE_PYTHON, relay_compat._tool_middleware),
    ]

    uninstall()
    for unsubscribe in unsubscribers:
        unsubscribe.assert_called_once_with()


def test_installer_unsubscribes_fixture_event_manager(monkeypatch: pytest.MonkeyPatch):
    fixture_source = Path(__file__).parents[1] / "fixtures" / "nooa" / "src"
    monkeypatch.syspath_prepend(str(fixture_source))
    from fabric_nooa_test_target import _EventManager

    event_manager = _EventManager()
    uninstall = relay_compat.install_nemo_relay_compat(event_manager)
    assert event_manager._handlers == {
        MIDDLEWARE_AGENT_CALL: [relay_compat.nemo_relay_agent_call_middleware],
        MIDDLEWARE_LLM_CALL: [relay_compat.nemo_relay_llm_middleware],
        MIDDLEWARE_EXECUTE_PYTHON: [relay_compat._tool_middleware],
    }

    uninstall()
    assert all(not handlers for handlers in event_manager._handlers.values())
    with pytest.raises(ValueError, match="not in list"):
        uninstall()
