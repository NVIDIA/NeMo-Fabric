# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Deep Agents adapter's Fabric runtime mapping.

These tests stub the ``deepagents``/``langchain``/``langgraph`` SDKs so they run
without the real harness installed; they assert the normalized Fabric result and
the live runtime's thread continuity. The real SDK is exercised by the opt-in
integration test in ``tests/e2e/test_deepagents.py``.
"""

from __future__ import annotations

import importlib.machinery
import os
import sys
import types
import uuid
from collections.abc import AsyncIterator
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from nemo_fabric_adapter_contract.codec import ContractValidationError
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentMcpServerConfig
from nemo_fabric_adapter_contract.models import McpOAuth2Config
from nemo_fabric_adapter_contract.models import McpServiceAccountConfig
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.deepagents import adapter  # noqa: E402


def lifecycle_start_payload(payload: dict[str, Any]) -> dict[str, Any]:
    start = {key: value for key, value in payload.items() if key != "request"}
    start["config"] = AgentConfig.from_mapping(start["config"])
    return start


def lifecycle_invocation(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "runtime_context": payload["runtime_context"],
        "request": payload["request"],
    }


async def invoke_once(payload: dict[str, Any]) -> dict[str, Any]:
    runtime = adapter.DeepAgentsRuntime()
    await runtime.start(lifecycle_start_payload(payload))
    try:
        return await runtime.invoke(lifecycle_invocation(payload))
    finally:
        await runtime.stop()


async def invoke_twice(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Two ordered invocations on one runtime, which is how Relay state outlives a turn."""

    runtime = adapter.DeepAgentsRuntime()
    await runtime.start(lifecycle_start_payload(payload))
    try:
        first = await runtime.invoke(lifecycle_invocation(payload))
        second = await runtime.invoke(lifecycle_invocation(payload))
        return first, second
    finally:
        await runtime.stop()


@pytest.fixture(name="fake_sdks", autouse=True)
def fake_sdks_fixture(monkeypatch):
    """Stub the deepagents/langchain/langgraph SDKs with mocks.

    Returns a recorder capturing the ``create_deep_agent`` kwargs, the streamed
    ``config``, and the checkpointer close count. ``chat_openai``/``fs_backend``
    expose the mocked classes so tests can assert their construction kwargs.
    """

    recorder: dict[str, Any] = {"saver_exits": 0}

    mock_chat_openai = MagicMock()
    mock_fs_backend = MagicMock()
    recorder["chat_openai"] = mock_chat_openai
    recorder["fs_backend"] = mock_fs_backend

    def build_agent(**kwargs):
        recorder["create_kwargs"] = kwargs

        async def astream(inputs, config=None, *, stream_mode=None, subgraphs=False):
            recorder["config"] = config
            recorder["subgraphs"] = subgraphs
            recorder["checkpointer"] = kwargs.get("checkpointer")
            user = inputs["messages"][-1]["content"]
            ai = {
                "role": "ai",
                "content": f"reply to {user}",
                "usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
            }
            # subgraphs=True yields 3-tuples ``(namespace, mode, chunk)``; the main
            # graph has an empty namespace. ``updates`` carries the message produced
            # this turn; ``values`` is the full replayed state.
            yield ((), "updates", {"agent": {"messages": [ai]}})
            yield ((), "values", {"messages": [{"role": "user", "content": user}, ai]})

        agent = MagicMock()
        agent.astream = astream
        return agent

    deepagents_mod = types.ModuleType("deepagents")
    deepagents_mod.__spec__ = importlib.machinery.ModuleSpec("deepagents", loader=None)
    deepagents_mod.create_deep_agent = MagicMock(side_effect=build_agent)
    backends_mod = types.ModuleType("deepagents.backends")
    backends_mod.FilesystemBackend = mock_fs_backend
    middleware_mod = types.ModuleType("deepagents.middleware")
    subagents_mod = types.ModuleType("deepagents.middleware.subagents")
    subagents_mod.GENERAL_PURPOSE_SUBAGENT = {
        "name": "general-purpose",
        "description": "General-purpose delegated agent.",
        "system_prompt": "Handle the delegated task.",
    }
    deepagents_mod.backends = backends_mod
    deepagents_mod.middleware = middleware_mod
    middleware_mod.subagents = subagents_mod
    monkeypatch.setitem(sys.modules, "deepagents", deepagents_mod)
    monkeypatch.setitem(sys.modules, "deepagents.backends", backends_mod)
    monkeypatch.setitem(sys.modules, "deepagents.middleware", middleware_mod)
    monkeypatch.setitem(sys.modules, "deepagents.middleware.subagents", subagents_mod)

    langchain_openai_mod = types.ModuleType("langchain_openai")
    langchain_openai_mod.ChatOpenAI = mock_chat_openai
    monkeypatch.setitem(sys.modules, "langchain_openai", langchain_openai_mod)

    def open_saver(_conn):
        async def aexit(*_exc):
            # Record cleanup so tests can assert the connection is always closed.
            recorder["saver_exits"] += 1
            return False

        saver_cm = MagicMock()
        saver_cm.__aenter__ = AsyncMock(return_value=MagicMock())
        saver_cm.__aexit__ = AsyncMock(side_effect=aexit)
        return saver_cm

    mock_saver = MagicMock()
    mock_saver.from_conn_string = MagicMock(side_effect=open_saver)

    langgraph_mod = types.ModuleType("langgraph")
    checkpoint_mod = types.ModuleType("langgraph.checkpoint")
    sqlite_mod = types.ModuleType("langgraph.checkpoint.sqlite")
    aio_mod = types.ModuleType("langgraph.checkpoint.sqlite.aio")
    aio_mod.AsyncSqliteSaver = mock_saver
    sqlite_mod.aio = aio_mod
    checkpoint_mod.sqlite = sqlite_mod
    langgraph_mod.checkpoint = checkpoint_mod
    for name, mod in (
        ("langgraph", langgraph_mod),
        ("langgraph.checkpoint", checkpoint_mod),
        ("langgraph.checkpoint.sqlite", sqlite_mod),
        ("langgraph.checkpoint.sqlite.aio", aio_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)

    monkeypatch.setenv("NVIDIA_API_KEY", "test123")
    return recorder


@pytest.fixture(name="make_payload")
def make_payload_fixture():
    """Return a factory that builds an adapter invocation payload."""

    def make(tmp_path: Path, *, runtime_id: str = "run-1") -> dict[str, Any]:
        return {
            "base_dir": str(tmp_path),
            "config": {
                "harness": {"settings": {}},
                "instructions": {
                    "system": {"content": "be concise", "mode": "replace"}
                },
                "models": {
                    "default": {
                        "provider": "nvidia",
                        "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
                        "api_key_env": "NVIDIA_API_KEY",
                        "base_url": "https://integrate.api.nvidia.com/v1",
                    }
                },
            },
            "runtime_context": {
                "runtime_id": runtime_id,
                "invocation_id": "inv-1",
                "request_id": "request-1",
                "environment": {
                    "environment_id": "test-environment",
                    "provider": "test",
                    "control_location": "in_env_control",
                    "workspace": str(tmp_path),
                    "ownership": "caller_owned",
                },
                "artifacts": {},
            },
            "request": {"input": "hello", "request_id": "request-1"},
        }

    return make


@pytest.fixture(name="fake_relay")
def fake_relay_fixture(monkeypatch):
    """Stub nemo_relay's plugin + deepagents integration; return a calls recorder."""

    import contextlib

    calls: dict[str, Any] = {}

    def add_nemo_relay_integration(kwargs, **_):
        merged = dict(kwargs)
        merged["middleware"] = [*(merged.get("middleware") or []), "relay-mw"]
        calls["wrapped"] = True
        calls["integration_adds"] = calls.get("integration_adds", 0) + 1
        return merged

    @contextlib.asynccontextmanager
    async def plugin_ctx(config: object) -> AsyncIterator[None]:
        calls["plugin_open"] = True
        calls["plugin_enters"] = calls.get("plugin_enters", 0) + 1
        calls.setdefault("plugin_configs", []).append(config)
        try:
            yield
        finally:
            calls["plugin_exits"] = calls.get("plugin_exits", 0) + 1

    class ScopeType:
        Agent = "agent"

    class _Handle:
        """Stand-in for nemo_relay ScopeHandle; the adapter compares ``uuid``."""

        def __init__(self, name: str) -> None:
            self.name = name
            self.uuid = str(uuid.uuid4())

    # Relay keeps one LIFO scope stack that outlives an invocation, so the fake keeps
    # one too: a scope that fails to unwind stays current for later turns, which is the
    # state the adapter has to detect.
    stack: list[_Handle] = [_Handle("root")]
    calls["stack"] = stack
    calls["handle_type"] = _Handle

    @contextlib.contextmanager
    def scope_ctx(
        name: str,
        scope_type: object,
        **kwargs: object,
    ) -> Iterator[None]:
        # Record every scope entered so tests can assert the top-level
        # ``deepagents-request`` Agent scope wraps the invocation.
        calls.setdefault("scopes", []).append((name, scope_type))
        calls.setdefault("scope_metadata", []).append(kwargs.get("metadata"))
        stack.append(_Handle(name))
        try:
            yield
        finally:
            stack.pop()

    def get_handle() -> _Handle:
        return stack[-1]

    class NemoRelayDeepAgentsCallbackHandler:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            calls["callback_handler"] = self

    relay_root = types.ModuleType("nemo_relay")
    # A real spec so the adapter preflight's importlib.util.find_spec("nemo_relay")
    # check sees Relay as installed.
    relay_root.__spec__ = importlib.machinery.ModuleSpec("nemo_relay", loader=None)
    plugin_mod = types.ModuleType("nemo_relay.plugin")
    plugin_mod.plugin = plugin_ctx
    scope_mod = types.ModuleType("nemo_relay.scope")
    scope_mod.scope = scope_ctx
    scope_mod.get_handle = get_handle
    relay_root.plugin = plugin_mod
    relay_root.scope = scope_mod
    relay_root.ScopeType = ScopeType
    integrations_pkg = types.ModuleType("nemo_relay.integrations")
    da_integ = types.ModuleType("nemo_relay.integrations.deepagents")
    da_integ.add_nemo_relay_integration = add_nemo_relay_integration
    da_integ.NemoRelayDeepAgentsCallbackHandler = NemoRelayDeepAgentsCallbackHandler
    for name, mod in (
        ("nemo_relay", relay_root),
        ("nemo_relay.plugin", plugin_mod),
        ("nemo_relay.scope", scope_mod),
        ("nemo_relay.integrations", integrations_pkg),
        ("nemo_relay.integrations.deepagents", da_integ),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


@pytest.fixture(name="use_real_langgraph")
def use_real_langgraph_fixture(fake_sdks, monkeypatch):
    """Drop the fake langgraph stubs so the real langchain/langgraph packages resolve."""

    for name in (
        "langgraph",
        "langgraph.checkpoint",
        "langgraph.checkpoint.sqlite",
        "langgraph.checkpoint.sqlite.aio",
        "langgraph.graph",
        "langgraph.graph.message",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)


async def test_single_invocation_normalizes_response_usage_and_thread(
    tmp_path, make_payload, fake_sdks
):
    output = await invoke_once(make_payload(tmp_path))

    assert output["harness"] == "deepagents"
    assert output["mode"] == "deepagents"
    assert output["model"] == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    assert output["response"] == "reply to hello"
    assert output["message_count"] == 2
    assert output["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 7,
        "total_tokens": 12,
    }
    # streamed events are buffered
    assert output["events"] == [{"nodes": ["agent"]}]
    assert output["event_count"] == 1
    assert output["runtime_id"] == "run-1"
    # The first invocation receives a newly assigned LangGraph thread id.
    assert output["thread_id"]
    assert output["resumed"] is False
    assert output["completed"] is True
    assert output["failed"] is False
    assert output["error"] is None
    # system prompt must reach deepagents under the real param name (not ``instructions``)
    assert fake_sdks["create_kwargs"]["system_prompt"] == "be concise"
    assert "instructions" not in fake_sdks["create_kwargs"]


async def test_missing_api_key_fails_runtime_start(tmp_path, make_payload):
    os.environ.pop("NVIDIA_API_KEY", None)

    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        await adapter.DeepAgentsRuntime().start(
            lifecycle_start_payload(make_payload(tmp_path))
        )


async def test_missing_deepagents_package_fails_runtime_start(
    tmp_path, make_payload, monkeypatch
):
    # Force find_spec("deepagents") -> None so the test holds whether or not the
    # real package is installed in the environment.
    import importlib.util as importlib_util

    real_find_spec = importlib_util.find_spec

    def fake_find_spec(
        name: str, *args: object, **kwargs: object
    ) -> importlib.machinery.ModuleSpec | None:
        if name == "deepagents":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib_util, "find_spec", fake_find_spec)
    with pytest.raises(RuntimeError, match="deepagents"):
        await adapter.DeepAgentsRuntime().start(
            lifecycle_start_payload(make_payload(tmp_path))
        )


async def test_agent_creation_error_fails_runtime_start(
    tmp_path, make_payload, monkeypatch
):
    import deepagents

    def boom(**_kwargs):
        raise RuntimeError("agent exploded")

    monkeypatch.setattr(deepagents, "create_deep_agent", boom)
    with pytest.raises(RuntimeError, match="agent exploded"):
        await adapter.DeepAgentsRuntime().start(
            lifecycle_start_payload(make_payload(tmp_path))
        )


async def test_runtime_start_requires_validated_agent_config(tmp_path, make_payload):
    payload = make_payload(tmp_path)
    start = {key: value for key, value in payload.items() if key != "request"}

    with pytest.raises(adapter.lifecycle.LifecycleError, match="validated AgentConfig"):
        await adapter.DeepAgentsRuntime().start(start)


async def test_relay_telemetry_wraps_agent_and_reports_artifacts(
    tmp_path, make_payload, monkeypatch, fake_sdks, fake_relay
):
    artifacts = [{"kind": "atof", "path": str(tmp_path / "events.atof.jsonl")}]
    plugin_config = {"version": 1, "components": []}
    load_relay_plugin_config = MagicMock(return_value=plugin_config)
    monkeypatch.setattr(
        adapter.common_utils,
        "load_relay_plugin_config",
        load_relay_plugin_config,
    )
    monkeypatch.setattr(
        adapter.common_utils, "collect_relay_artifacts", lambda _c: artifacts
    )
    payload = make_payload(tmp_path)
    payload["runtime_context"]["telemetry"] = {
        "relay_enabled": True,
        "metadata": {"telemetry_providers": ["relay"]},
    }

    output = await invoke_once(payload)

    assert fake_relay["wrapped"]
    assert fake_relay["plugin_open"]
    assert fake_relay["plugin_configs"] == [plugin_config]
    assert load_relay_plugin_config.call_args.kwargs == {
        "base_dir": str(tmp_path),
        "runtime_id": "run-1",
        "agent_name": "fabric-agent",
        "model_name": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    }
    assert output["telemetry"] == {
        "enabled": True,
        "provider": "relay",
        "emitter": "deepagents.observability/nemo_relay",
    }
    assert output["relay_artifacts"] == artifacts
    # the relay middleware reached the deepagents kwargs
    assert "relay-mw" in fake_sdks["create_kwargs"]["middleware"]
    # the top-level invocation is wrapped in the deepagents-request Agent scope
    # ("agent" is the fake ScopeType.Agent sentinel from the fake_relay fixture)
    assert fake_relay["scopes"] == [("deepagents-request", "agent")]
    assert fake_relay["scope_metadata"] == [{"nemo_fabric_request_id": "request-1"}]
    # the Deep Agents callback handler is added to the LangGraph run config so
    # LangGraph scopes and human-in-the-loop interrupt/resume marks are captured
    assert fake_relay["callback_handler"] in (fake_sdks["config"] or {}).get(
        "callbacks", []
    )


@pytest.fixture(name="relay_payload")
def relay_payload_fixture(make_payload, monkeypatch):
    """Build a payload with Relay telemetry enabled and its plugin config stubbed."""

    monkeypatch.setattr(
        adapter.common_utils,
        "load_relay_plugin_config",
        lambda *, base_dir, runtime_id, agent_name, model_name: {
            "version": 1,
            "components": [],
        },
    )

    def build(tmp_path) -> dict[str, Any]:
        payload = make_payload(tmp_path)
        payload["runtime_context"]["telemetry"] = {
            "relay_enabled": True,
            "metadata": {"telemetry_providers": ["relay"]},
        }
        return payload

    return build


async def test_relay_scope_teardown_failure_keeps_the_invocation_completed(
    tmp_path, relay_payload, monkeypatch, fake_sdks, fake_relay
):
    """A telemetry teardown fault must not rewrite a completed turn into a failure.

    Relay closes scopes in LIFO order, but LangGraph runs concurrent chain callbacks
    that can finish out of that order, so closing the outer ``deepagents-request``
    scope can raise after the agent has already produced its final response.
    """

    import contextlib

    @contextlib.contextmanager
    def exploding_scope(name: str, scope_type: object, **kwargs: object):
        yield
        raise RuntimeError(
            "invalid argument: scope handle is not at the top of the stack"
        )

    monkeypatch.setattr(sys.modules["nemo_relay.scope"], "scope", exploding_scope)

    output = await invoke_once(relay_payload(tmp_path))

    # The functional outcome is preserved in full.
    assert output["completed"] is True
    assert output["failed"] is False
    assert output["error"] is None
    assert output["response"] == "reply to hello"
    assert output["message_count"] == 2
    # ...and the telemetry fault is still reported, so a consumer can tell that
    # observability degraded and the referenced trajectory may be truncated.
    assert output["telemetry"]["degraded"] is True
    assert "scope handle is not at the top of the stack" in output["telemetry"]["error"]


async def test_relay_plugin_teardown_failure_keeps_the_invocation_completed(
    tmp_path, relay_payload, monkeypatch, fake_sdks, fake_relay
):
    """The plugin context manager is the other half of the telemetry lifecycle.

    A failed export flush on the way out is as much an observability fault as a failed
    scope pop, and must be treated the same way.
    """

    import contextlib

    @contextlib.asynccontextmanager
    async def exploding_plugin(config: object):
        yield
        raise RuntimeError("relay plugin flush failed")

    monkeypatch.setattr(sys.modules["nemo_relay.plugin"], "plugin", exploding_plugin)

    output = await invoke_once(relay_payload(tmp_path))

    assert output["completed"] is True
    assert output["failed"] is False
    assert output["error"] is None
    assert output["response"] == "reply to hello"
    assert output["telemetry"]["degraded"] is True
    assert output["telemetry"]["error"] == "RuntimeError: relay plugin flush failed"


async def test_a_dirty_scope_stack_quarantines_telemetry_for_later_turns(
    tmp_path, relay_payload, monkeypatch, fake_sdks, fake_relay
):
    """Relay's scope stack outlives an invocation, so a fault poisons the runtime.

    Preserving the completed turn is not enough on its own: the scope left current by
    turn 1 is still current for turn 2, which would otherwise report itself clean while
    nesting its trajectory under a stale scope.
    """

    import contextlib

    entered: list[str] = []

    @contextlib.contextmanager
    def leaking_scope(name: str, scope_type: object, **kwargs: object):
        # Mirrors the real failure: the child scope is never popped, so the outer close
        # raises and the scope pushed here stays current for every later turn.
        entered.append(name)
        fake_relay["stack"].append(fake_relay["handle_type"](name))
        yield
        raise RuntimeError(
            "invalid argument: scope handle is not at the top of the stack"
        )

    monkeypatch.setattr(sys.modules["nemo_relay.scope"], "scope", leaking_scope)
    # A recognisable artifact so the faulting turn's publication can be checked by value
    # rather than by the key merely existing: real collection over an empty evidence dir
    # returns [], which would pass a presence-only assertion either way.
    artifacts = [{"kind": "atif", "path": str(tmp_path / "trajectory.atif.json")}]
    monkeypatch.setattr(
        adapter.common_utils, "collect_relay_artifacts", lambda _config: artifacts
    )

    first, second = await invoke_twice(relay_payload(tmp_path))

    assert first["completed"] is True
    assert first["telemetry"]["degraded"] is True

    # Turn 2 is functionally fine and must stay completed, but it ran on a dirty stack
    # and must not claim clean telemetry.
    assert second["completed"] is True
    assert second["error"] is None
    assert second["response"] == "reply to hello"
    assert second["telemetry"]["degraded"] is True
    assert "unreliable" in second["telemetry"]["error"]
    # Only turn 1 opened a request scope; turn 2 must not push onto the dirty stack.
    assert entered == ["deepagents-request"]
    # Turn 1 opened a scope and produced its own (partial) artifacts, so it still
    # publishes them; turn 2 has none of its own and must not claim turn 1's.
    assert first["relay_artifacts"] == artifacts
    assert "relay_artifacts" not in second

    # Turn 1 owns the fault, so it reports it verbatim and needs no separate cause.
    assert "not at the top of the stack" in first["telemetry"]["error"]
    assert "quarantine_cause" not in first["telemetry"]
    # Turn 2 did not fail this way, so the fault appears only as provenance. Repeating
    # it in ``error`` would make a consumer count one fault per turn and blame the wrong
    # turn for it.
    assert "not at the top of the stack" not in second["telemetry"]["error"]
    assert "not at the top of the stack" in second["telemetry"]["quarantine_cause"]


async def test_an_agent_failure_on_a_quarantined_turn_keeps_both_domains(
    tmp_path, relay_payload, monkeypatch, fake_sdks, fake_relay
):
    """Quarantine must not swallow, or be swallowed by, a real agent failure."""

    import contextlib

    @contextlib.contextmanager
    def leaking_scope(name: str, scope_type: object, **kwargs: object):
        fake_relay["stack"].append(fake_relay["handle_type"](name))
        yield
        raise RuntimeError(
            "invalid argument: scope handle is not at the top of the stack"
        )

    monkeypatch.setattr(sys.modules["nemo_relay.scope"], "scope", leaking_scope)
    payload = relay_payload(tmp_path)

    runtime = adapter.DeepAgentsRuntime()
    await runtime.start(lifecycle_start_payload(payload))
    try:
        await runtime.invoke(lifecycle_invocation(payload))

        async def boom(*_args: object, **_kwargs: object):
            raise RuntimeError("model call failed")

        monkeypatch.setattr(adapter, "invoke_compiled_agent", boom)
        quarantined = await runtime.invoke(lifecycle_invocation(payload))
    finally:
        await runtime.stop()

    assert quarantined["completed"] is False
    assert quarantined["failed"] is True
    assert quarantined["error"] == "RuntimeError: model call failed"
    assert quarantined["telemetry"]["degraded"] is True
    assert "unreliable" in quarantined["telemetry"]["error"]


async def test_quarantine_survives_a_stop_and_restart(
    tmp_path, relay_payload, monkeypatch, fake_sdks, fake_relay
):
    """The dirty stack lives in the process, so a restart inherits it.

    Clearing the quarantine in ``stop()`` alongside the rest of the runtime state would
    hand the next runtime a clean bill of health on a stack that is still corrupt.
    """

    import contextlib

    entries = {"count": 0}

    @contextlib.contextmanager
    def leaks_once(name: str, scope_type: object, **kwargs: object):
        # Only the first turn leaks. A later turn would unwind cleanly and report itself
        # clean, so this test fails unless the quarantine itself carried across the
        # restart — the stack is still dirty even though nothing new damages it.
        entries["count"] += 1
        handle = fake_relay["handle_type"](name)
        fake_relay["stack"].append(handle)
        if entries["count"] == 1:
            yield
            raise RuntimeError(
                "invalid argument: scope handle is not at the top of the stack"
            )
        try:
            yield
        finally:
            fake_relay["stack"].remove(handle)

    monkeypatch.setattr(sys.modules["nemo_relay.scope"], "scope", leaks_once)
    payload = relay_payload(tmp_path)

    runtime = adapter.DeepAgentsRuntime()
    await runtime.start(lifecycle_start_payload(payload))
    first = await runtime.invoke(lifecycle_invocation(payload))
    await runtime.stop()

    await runtime.start(lifecycle_start_payload(payload))
    try:
        after_restart = await runtime.invoke(lifecycle_invocation(payload))
    finally:
        await runtime.stop()

    assert first["telemetry"]["degraded"] is True
    assert after_restart["completed"] is True
    assert after_restart["telemetry"]["degraded"] is True
    assert "unreliable" in after_restart["telemetry"]["error"]
    # The restarted runtime never opened a scope of its own; it inherited the verdict.
    assert entries["count"] == 1


async def test_an_unreadable_scope_handle_counts_as_dirty(monkeypatch):
    """A safety check must fail closed when it cannot read the state it guards.

    Comparing missing attributes with a ``None`` default made two unreadable handles
    look equal, which silently disabled the quarantine.
    """

    class Bare:
        """A handle carrying no identity, as a future Relay rename would produce."""

    monkeypatch.setattr(adapter, "_current_scope_handle", lambda: Bare())

    assert adapter._scope_top_unchanged(Bare()) is False


async def test_a_fault_that_unwinds_cleanly_does_not_quarantine_later_turns(
    tmp_path, relay_payload, monkeypatch, fake_sdks, fake_relay
):
    """Quarantine is for a dirty stack, not for any telemetry fault.

    A flush that fails after the scope unwound leaves nothing stale behind, so the next
    turn is genuinely clean and must be reported that way.
    """

    import contextlib

    flushes = {"count": 0}

    @contextlib.asynccontextmanager
    async def flaky_plugin(config: object):
        yield
        flushes["count"] += 1
        if flushes["count"] == 1:
            raise RuntimeError("relay plugin flush failed")

    monkeypatch.setattr(sys.modules["nemo_relay.plugin"], "plugin", flaky_plugin)

    first, second = await invoke_twice(relay_payload(tmp_path))

    assert first["telemetry"]["degraded"] is True
    assert "degraded" not in second["telemetry"]
    assert "error" not in second["telemetry"]


async def test_scope_and_plugin_teardown_faults_are_both_reported(
    tmp_path, relay_payload, monkeypatch, fake_sdks, fake_relay
):
    """A scope fault crossing the plugin's ``__aexit__`` is replaced by the plugin's.

    Both stages must survive that, or one lifecycle fault silently disappears.
    """

    import contextlib

    @contextlib.contextmanager
    def exploding_scope(name: str, scope_type: object, **kwargs: object):
        yield
        raise RuntimeError("scope handle is not at the top of the stack")

    @contextlib.asynccontextmanager
    async def exploding_plugin(config: object):
        yield
        raise RuntimeError("relay plugin flush failed")

    monkeypatch.setattr(sys.modules["nemo_relay.scope"], "scope", exploding_scope)
    monkeypatch.setattr(sys.modules["nemo_relay.plugin"], "plugin", exploding_plugin)

    output = await invoke_once(relay_payload(tmp_path))

    assert output["completed"] is True
    assert output["error"] is None
    assert output["telemetry"]["degraded"] is True
    assert "scope handle is not at the top of the stack" in output["telemetry"]["error"]
    assert "relay plugin flush failed" in output["telemetry"]["error"]


async def test_callback_handler_construction_failure_is_a_normalized_failure(
    tmp_path, relay_payload, monkeypatch, fake_sdks, fake_relay
):
    """Building the callback handler is telemetry setup, so it must normalize too.

    Constructing it outside the guarded region would let the error escape ``invoke``
    instead of returning the required invocation-failure response.
    """

    def exploding_handler(*_args: object, **_kwargs: object):
        raise RuntimeError("callback handler construction failed")

    monkeypatch.setattr(
        sys.modules["nemo_relay.integrations.deepagents"],
        "NemoRelayDeepAgentsCallbackHandler",
        exploding_handler,
    )
    invoke_agent = AsyncMock()
    monkeypatch.setattr(adapter, "invoke_compiled_agent", invoke_agent)

    output = await invoke_once(relay_payload(tmp_path))

    invoke_agent.assert_not_awaited()
    assert output["completed"] is False
    assert output["failed"] is True
    assert output["error"] == "RuntimeError: callback handler construction failed"
    # The same fault is reported in both domains, as it is for any other setup failure.
    assert output["telemetry"]["degraded"] is True
    assert (
        output["telemetry"]["error"]
        == "RuntimeError: callback handler construction failed"
    )


async def test_artifact_collection_failure_does_not_discard_a_completed_turn(
    tmp_path, relay_payload, monkeypatch, fake_sdks, fake_relay
):
    """Collecting artifact references walks the filesystem, so it can fail on its own.

    It runs after the turn is over, so letting it raise would throw away a completed
    invocation — the same failure mode the invocation/telemetry split exists to stop.
    """

    def boom(_config: object) -> list[dict[str, str]]:
        raise OSError("artifact directory disappeared")

    monkeypatch.setattr(adapter.common_utils, "collect_relay_artifacts", boom)

    output = await invoke_once(relay_payload(tmp_path))

    assert output["completed"] is True
    assert output["failed"] is False
    assert output["error"] is None
    assert output["telemetry"]["degraded"] is True
    assert output["telemetry"]["error"] == "OSError: artifact directory disappeared"


async def test_teardown_and_artifact_faults_are_both_reported(
    tmp_path, relay_payload, monkeypatch, fake_sdks, fake_relay
):
    """Two telemetry faults in one turn: neither may silently swallow the other."""

    import contextlib

    @contextlib.contextmanager
    def exploding_scope(name: str, scope_type: object, **kwargs: object):
        yield
        raise RuntimeError("scope handle is not at the top of the stack")

    def boom(_config: object) -> list[dict[str, str]]:
        raise OSError("artifact directory disappeared")

    monkeypatch.setattr(sys.modules["nemo_relay.scope"], "scope", exploding_scope)
    monkeypatch.setattr(adapter.common_utils, "collect_relay_artifacts", boom)

    output = await invoke_once(relay_payload(tmp_path))

    assert output["completed"] is True
    assert output["telemetry"]["degraded"] is True
    assert "scope handle is not at the top of the stack" in output["telemetry"]["error"]
    assert "artifact directory disappeared" in output["telemetry"]["error"]


async def test_relay_setup_failure_before_the_agent_runs_stays_an_invocation_failure(
    tmp_path, relay_payload, monkeypatch, fake_sdks, fake_relay
):
    """Telemetry that fails on the way *in* leaves no functional outcome to preserve."""

    import contextlib

    @contextlib.contextmanager
    def failing_scope(name: str, scope_type: object, **kwargs: object):
        raise RuntimeError("relay scope push failed")
        yield  # pragma: no cover - unreachable, keeps this a generator

    monkeypatch.setattr(sys.modules["nemo_relay.scope"], "scope", failing_scope)
    invoke_agent = AsyncMock()
    monkeypatch.setattr(adapter, "invoke_compiled_agent", invoke_agent)

    output = await invoke_once(relay_payload(tmp_path))

    # The setup fault must short-circuit before any agent request is consumed.
    invoke_agent.assert_not_awaited()
    assert output["completed"] is False
    assert output["failed"] is True
    assert output["error"] == "RuntimeError: relay scope push failed"
    # The same fault is also a telemetry fault, so it is reported in both domains.
    assert output["telemetry"]["degraded"] is True
    assert output["telemetry"]["error"] == "RuntimeError: relay scope push failed"


async def test_agent_failure_under_relay_is_still_an_invocation_failure(
    tmp_path, relay_payload, monkeypatch, fake_sdks, fake_relay
):
    """Guard against the split widening: a real agent failure must still fail."""

    async def boom(*_args: object, **_kwargs: object):
        raise RuntimeError("model call failed")

    monkeypatch.setattr(adapter, "invoke_compiled_agent", boom)

    output = await invoke_once(relay_payload(tmp_path))

    assert output["completed"] is False
    assert output["failed"] is True
    assert output["error"] == "RuntimeError: model call failed"
    # A clean telemetry lifecycle leaves the telemetry block untouched.
    assert "error" not in output["telemetry"]
    assert "degraded" not in output["telemetry"]


async def test_agent_failure_and_teardown_failure_keep_their_own_domains(
    tmp_path, relay_payload, monkeypatch, fake_sdks, fake_relay
):
    """When both fail, the invocation error stays the agent's, not the telemetry one."""

    import contextlib

    async def boom(*_args: object, **_kwargs: object):
        raise RuntimeError("model call failed")

    @contextlib.contextmanager
    def exploding_scope(name: str, scope_type: object, **kwargs: object):
        yield
        raise RuntimeError("scope handle is not at the top of the stack")

    monkeypatch.setattr(adapter, "invoke_compiled_agent", boom)
    monkeypatch.setattr(sys.modules["nemo_relay.scope"], "scope", exploding_scope)

    output = await invoke_once(relay_payload(tmp_path))

    assert output["completed"] is False
    assert output["failed"] is True
    assert output["error"] == "RuntimeError: model call failed"
    assert output["telemetry"]["degraded"] is True
    assert "scope handle is not at the top of the stack" in output["telemetry"]["error"]


async def test_native_telemetry_exports_without_artifacts(
    tmp_path, make_payload, monkeypatch, fake_sdks, fake_relay
):
    payload = make_payload(tmp_path)
    payload["runtime_context"]["telemetry"] = {
        "relay_enabled": False,
        "metadata": {
            "telemetry_providers": ["native"],
            "native_config": {
                "version": 1,
                "components": [
                    {
                        "kind": "observability",
                        "enabled": True,
                        "config": {
                            "version": 1,
                            "opentelemetry": {
                                "enabled": True,
                                "endpoint": "http://localhost:4318/v1/traces",
                            },
                        },
                    }
                ],
            },
        },
    }

    output = await invoke_once(payload)

    assert fake_relay["wrapped"]
    assert fake_relay["plugin_open"]
    assert fake_relay["plugin_configs"] == [
        payload["runtime_context"]["telemetry"]["metadata"]["native_config"]
    ]
    assert output["telemetry"] == {
        "enabled": True,
        "provider": "native",
        "emitter": "deepagents.observability/native",
    }
    # native telemetry exports directly; no ATOF/ATIF relay artifacts are written
    assert "relay_artifacts" not in output
    assert "relay-mw" in fake_sdks["create_kwargs"]["middleware"]
    # the scope + callback handler apply to any observability-enabled run, native included
    assert fake_relay["scopes"] == [("deepagents-request", "agent")]
    assert fake_relay["scope_metadata"] == [{"nemo_fabric_request_id": "request-1"}]
    assert fake_relay["callback_handler"] in (fake_sdks["config"] or {}).get(
        "callbacks", []
    )


async def test_relay_disabled_adds_no_scope_or_callbacks(
    tmp_path, make_payload, fake_sdks
):
    # With telemetry disabled the invocation runs without a Relay scope, callback
    # handler, or middleware, preserving the Relay-neutral default behavior.
    output = await invoke_once(make_payload(tmp_path))

    assert output["completed"] is True
    assert "telemetry" not in output
    assert (fake_sdks["config"] or {}).get("callbacks") is None
    assert "relay-mw" not in (fake_sdks["create_kwargs"].get("middleware") or [])


async def test_missing_nemo_relay_with_native_telemetry_fails_runtime_start(
    tmp_path, make_payload, monkeypatch
):
    # Native telemetry also runs through the nemo_relay plugin, so a core-only
    # install configured with native telemetry must fail with the actionable
    # extra-install message rather than a raw ModuleNotFoundError -- even though
    # relay itself is not enabled. Force find_spec("nemo_relay") -> None (no
    # fake_relay module) so the guard fires regardless of the environment.
    import importlib.util as importlib_util

    real_find_spec = importlib_util.find_spec

    def fake_find_spec(
        name: str, *args: object, **kwargs: object
    ) -> importlib.machinery.ModuleSpec | None:
        if name == "nemo_relay":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib_util, "find_spec", fake_find_spec)

    payload = make_payload(tmp_path)
    payload["runtime_context"]["telemetry"] = {
        "relay_enabled": False,
        "metadata": {
            "telemetry_providers": ["native"],
            "native_config": {
                "version": 1,
                "components": [
                    {
                        "kind": "observability",
                        "enabled": True,
                        "config": {"version": 1},
                    }
                ],
            },
        },
    }

    with pytest.raises(RuntimeError, match="nemo-relay.*\\[relay\\]"):
        await adapter.DeepAgentsRuntime().start(lifecycle_start_payload(payload))


@pytest.mark.usefixtures("fake_relay")
async def test_incomplete_nemo_relay_install_fails_runtime_start(
    tmp_path, make_payload, monkeypatch
):
    monkeypatch.delitem(sys.modules, "nemo_relay.integrations.deepagents")
    payload = make_payload(tmp_path)
    payload["runtime_context"]["telemetry"] = {
        "relay_enabled": False,
        "metadata": {
            "telemetry_providers": ["native"],
            "native_config": {
                "version": 1,
                "components": [
                    {
                        "kind": "observability",
                        "enabled": True,
                        "config": {"version": 1},
                    }
                ],
            },
        },
    }

    with pytest.raises(RuntimeError, match="compatible 'nemo-relay'.*\\[relay\\]"):
        await adapter.DeepAgentsRuntime().start(lifecycle_start_payload(payload))


def test_apply_callbacks_preserves_existing_ahead_of_new():
    # A consumer-provided callback already on the run config must be kept, with the
    # Relay callback appended after it rather than replacing it.
    config = {"configurable": {"thread_id": "t"}, "callbacks": ["consumer-cb"]}
    result = adapter._apply_callbacks(config, ["relay-cb"])

    assert result["callbacks"] == ["consumer-cb", "relay-cb"]
    assert result["configurable"] == {"thread_id": "t"}


def test_apply_callbacks_without_callbacks_leaves_config_untouched():
    config = {"configurable": {"thread_id": "t"}}
    assert adapter._apply_callbacks(config, None) == {
        "configurable": {"thread_id": "t"}
    }


def test_state_dir_uses_runtime_artifact_context(tmp_path, make_payload, monkeypatch):
    context = RuntimeContext.from_mapping(make_payload(tmp_path)["runtime_context"])
    monkeypatch.setenv("FABRIC_ARTIFACTS", str(tmp_path / "ignored"))

    assert adapter.state_dir(context, str(tmp_path)) == (
        tmp_path / "artifacts" / "deepagents" / ".fabric"
    )


async def test_invoke_compiled_agent_wires_callbacks_into_run_config(fake_sdks):
    from deepagents import create_deep_agent

    agent = create_deep_agent(model=object())
    await adapter.invoke_compiled_agent(
        agent, "hello", "thread-1", callbacks=["cb-a", "cb-b"]
    )

    config = fake_sdks["config"]
    assert config["configurable"]["thread_id"] == "thread-1"
    assert config["callbacks"] == ["cb-a", "cb-b"]


async def test_invoke_compiled_agent_without_callbacks_sets_no_callbacks_key(fake_sdks):
    from deepagents import create_deep_agent

    agent = create_deep_agent(model=object())
    await adapter.invoke_compiled_agent(agent, "hello", None, callbacks=None)

    # No thread and no callbacks means the agent is streamed without a config.
    assert fake_sdks["config"] is None


async def test_workspace_roots_filesystem_backend(tmp_path, make_payload, fake_sdks):
    await invoke_once(make_payload(tmp_path))
    backend_kwargs = fake_sdks["fs_backend"].call_args.kwargs
    assert backend_kwargs["root_dir"] == str(tmp_path)
    # virtual_mode=True confines the agent to root_dir: absolute paths and ``..``
    # cannot escape the workspace (and it does not rely on the deprecated default).
    assert backend_kwargs["virtual_mode"] is True


async def test_checkpointer_closed_on_success_and_failure(
    tmp_path, make_payload, monkeypatch, fake_sdks
):
    # The async checkpointer must be closed on both the success and error paths.
    await invoke_once(make_payload(tmp_path))
    assert fake_sdks["saver_exits"] == 1

    import deepagents

    def boom(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(deepagents, "create_deep_agent", boom)
    with pytest.raises(RuntimeError, match="boom"):
        await adapter.DeepAgentsRuntime().start(
            lifecycle_start_payload(make_payload(tmp_path))
        )
    assert fake_sdks["saver_exits"] == 2


async def test_mcp_servers_become_adapter_tools(
    tmp_path, make_payload, monkeypatch, fake_sdks
):
    tool_read = MagicMock()
    tool_read.name = "read_file"
    tool_write = MagicMock()
    tool_write.name = "write_file"
    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[tool_read, tool_write])
    mock_client_cls = MagicMock(return_value=mock_client)

    client_mod = types.ModuleType("langchain_mcp_adapters.client")
    client_mod.MultiServerMCPClient = mock_client_cls
    monkeypatch.setitem(
        sys.modules,
        "langchain_mcp_adapters",
        types.ModuleType("langchain_mcp_adapters"),
    )
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters.client", client_mod)
    payload = make_payload(tmp_path)
    payload["config"]["mcp"] = {
        "servers": {
            "fs": {
                "transport": "streamable-http",
                "url": "http://localhost:9/mcp",
                "custom_headers": {"X-Tenant": "${FABRIC_TEST_MCP_HEADER}"},
            },
            "local": {
                "transport": "stdio",
                "url": "my-server",
                "args": ["--flag", "--config", "repo config.json"],
                "env": {"REPO_MCP_MODE": "test"},
            },
        }
    }
    monkeypatch.setenv("FABRIC_TEST_MCP_HEADER", "fabric")

    output = await invoke_once(payload)

    assert output["failed"] is False
    assert mock_client_cls.call_args.args[0] == {
        "fs": {
            "transport": "streamable_http",
            "url": "http://localhost:9/mcp",
            "headers": {"X-Tenant": "fabric"},
        },
        "local": {
            "transport": "stdio",
            "command": "my-server",
            "args": ["--flag", "--config", "repo config.json"],
            "env": {"REPO_MCP_MODE": "test"},
        },
    }
    tool_names = [tool.name for tool in fake_sdks["create_kwargs"]["tools"]]
    assert tool_names == ["read_file", "write_file"]


@pytest.mark.parametrize(
    "authentication",
    [
        McpOAuth2Config(type="oauth2"),
        McpServiceAccountConfig(
            type="service_account",
            client_id="client",
            client_secret_env="CLIENT_SECRET",
            token_url="https://auth.example.test/token",
        ),
    ],
)
def test_deepagents_rejects_mcp_authentication(authentication):
    with pytest.raises(
        adapter.AdapterConfigError, match="not supported by Deep Agents"
    ):
        adapter._mcp_connection(
            "automation",
            AgentMcpServerConfig(
                transport="streamable-http",
                url="https://mcp.example.test/mcp",
                authentication=authentication,
            ),
        )


@pytest.mark.usefixtures("use_real_langgraph")
async def test_tool_policy_middleware_enforces_enabled_and_blocked_tools():
    pytest.importorskip("langchain.agents.middleware")
    from langchain_core.messages import ToolMessage

    middleware = adapter.tool_policy_middleware({"read_file"}, {"write_file"})

    async def handler(_request: types.SimpleNamespace) -> str:
        return "executed"

    def request(name: str) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            tool_call={"name": name, "id": "call-1", "args": {}}
        )

    blocked = await middleware.awrap_tool_call(request("write_file"), handler)
    assert isinstance(blocked, ToolMessage)
    assert blocked.status == "error"

    allowed = await middleware.awrap_tool_call(request("read_file"), handler)
    assert allowed == "executed"

    unselected = await middleware.awrap_tool_call(request("search"), handler)
    assert isinstance(unselected, ToolMessage)
    assert unselected.status == "error"


@pytest.mark.usefixtures("use_real_langgraph")
async def test_real_langgraph_async_checkpointer(tmp_path, make_payload, monkeypatch):
    # Regression: driving astream with the sync SqliteSaver raises NotImplementedError.
    # Exercise the adapter against a real compiled LangGraph graph + AsyncSqliteSaver.
    pytest.importorskip("langgraph.graph")
    pytest.importorskip("langgraph.checkpoint.sqlite.aio")

    import deepagents
    from langchain_core.messages import AIMessage
    from langgraph.graph import END
    from langgraph.graph import START
    from langgraph.graph import MessagesState
    from langgraph.graph import StateGraph

    def respond(_state):
        return {"messages": [AIMessage(content="ok")]}

    def build(**kwargs):
        graph = StateGraph(MessagesState)
        graph.add_node("respond", respond)
        graph.add_edge(START, "respond")
        graph.add_edge("respond", END)
        checkpointer = kwargs["checkpointer"]
        assert checkpointer is not None
        return graph.compile(checkpointer=checkpointer)

    monkeypatch.setattr(deepagents, "create_deep_agent", build)

    output = await invoke_once(make_payload(tmp_path))

    assert output["failed"] is False, output["error"]
    assert output["response"] == "ok"
    assert output["thread_id"]


async def test_openai_provider_keeps_openai_endpoint(
    tmp_path, make_payload, monkeypatch, fake_sdks
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    payload = make_payload(tmp_path)
    payload["config"]["models"]["default"] = {
        "provider": "openai",
        "model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
    }

    output = await invoke_once(payload)

    # openai must NOT be redirected to NVIDIA's endpoint
    assert output["base_url"] is None
    assert "base_url" not in fake_sdks["chat_openai"].call_args.kwargs


async def test_skill_paths_map_to_skills(tmp_path, make_payload, fake_sdks):
    payload = make_payload(tmp_path)
    payload["config"]["skills"] = {"paths": ["/skills/a", "/skills/b"]}

    await invoke_once(payload)

    assert fake_sdks["create_kwargs"]["skills"] == ["/skills/a", "/skills/b"]


async def test_cost_is_extracted_from_response_metadata(
    tmp_path, make_payload, monkeypatch
):
    import deepagents

    message = {
        "role": "ai",
        "content": "done",
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        "response_metadata": {"cost": 0.0025},
    }

    async def astream(inputs, config=None, *, stream_mode=None, subgraphs=False):
        yield ((), "updates", {"agent": {"messages": [message]}})
        yield ((), "values", {"messages": [message]})

    agent = MagicMock()
    agent.astream = astream
    monkeypatch.setattr(deepagents, "create_deep_agent", MagicMock(return_value=agent))
    output = await invoke_once(make_payload(tmp_path))

    assert output["usage"]["cost"] == 0.0025


async def test_replayed_state_usage_counts_current_turn_only(
    tmp_path, make_payload, monkeypatch
):
    # On a later turn the final state replays prior messages; usage and
    # cost must reflect only the message emitted this turn, not the replayed one.
    import deepagents

    prior = {
        "role": "ai",
        "content": "prior",
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        "response_metadata": {"cost": 0.001},
    }
    current = {
        "role": "ai",
        "content": "now",
        "usage": {"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
        "response_metadata": {"cost": 0.002},
    }

    async def astream(inputs, config=None, *, stream_mode=None, subgraphs=False):
        # Only the current turn's message is emitted as an update...
        yield ((), "updates", {"agent": {"messages": [current]}})
        # ...but the final state also replays the prior turn.
        yield ((), "values", {"messages": [prior, current]})

    agent = MagicMock()
    agent.astream = astream
    monkeypatch.setattr(deepagents, "create_deep_agent", MagicMock(return_value=agent))

    output = await invoke_once(make_payload(tmp_path))

    assert output["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "total_tokens": 4,
        "cost": 0.002,
    }
    # the full transcript is still returned
    assert output["message_count"] == 2


async def test_persistent_runtime_reuses_compiled_agent_and_checkpointer(
    tmp_path, make_payload, fake_sdks
):
    import deepagents

    payload = make_payload(tmp_path, runtime_id="run-persistent")
    runtime = adapter.DeepAgentsRuntime()

    await runtime.start(lifecycle_start_payload(payload))
    first = await runtime.invoke(lifecycle_invocation(payload))
    payload["runtime_context"]["invocation_id"] = "inv-2"
    payload["request"]["input"] = "continue"
    second = await runtime.invoke(lifecycle_invocation(payload))

    assert first["resumed"] is False
    assert second["resumed"] is True
    assert first["thread_id"] == second["thread_id"]
    assert deepagents.create_deep_agent.call_count == 1
    assert fake_sdks["saver_exits"] == 0
    checkpointer = fake_sdks["create_kwargs"]["checkpointer"]
    assert checkpointer is fake_sdks["checkpointer"]

    await runtime.stop()

    assert fake_sdks["saver_exits"] == 1


async def test_persistent_runtime_scopes_relay_per_invocation(
    tmp_path, make_payload, monkeypatch, fake_sdks, fake_relay
):
    artifacts = [{"kind": "atif", "path": str(tmp_path / "trajectory.json")}]
    plugin_config = {"version": 1, "components": []}
    monkeypatch.setattr(
        adapter.common_utils,
        "load_relay_plugin_config",
        lambda *, base_dir, runtime_id, agent_name, model_name: plugin_config,
    )
    monkeypatch.setattr(
        adapter.common_utils,
        "collect_relay_artifacts",
        lambda _config: artifacts,
    )
    payload = make_payload(tmp_path, runtime_id="run-relay-persistent")
    payload["runtime_context"]["telemetry"] = {
        "relay_enabled": True,
        "metadata": {"telemetry_providers": ["relay"], "adapter_outputs": ["atif"]},
    }
    runtime = adapter.DeepAgentsRuntime()

    await runtime.start(lifecycle_start_payload(payload))
    first = await runtime.invoke(lifecycle_invocation(payload))
    payload["runtime_context"]["invocation_id"] = "inv-2"
    payload["request"]["input"] = "continue"
    second = await runtime.invoke(lifecycle_invocation(payload))
    await runtime.stop()

    assert fake_relay["integration_adds"] == 1
    assert fake_relay["plugin_enters"] == 2
    assert fake_relay["plugin_exits"] == 2
    assert fake_relay["plugin_configs"] == [plugin_config, plugin_config]
    assert fake_relay["scopes"] == [
        ("deepagents-request", "agent"),
        ("deepagents-request", "agent"),
    ]
    assert first["thread_id"] == second["thread_id"]
    assert first["relay_artifacts"] == second["relay_artifacts"] == artifacts
    assert fake_sdks["saver_exits"] == 1


async def test_stream_requests_subgraphs(tmp_path, make_payload, fake_sdks):
    # Streaming must opt into subgraphs so delegated (subagent) steps are visible
    # for usage aggregation.
    await invoke_once(make_payload(tmp_path))
    assert fake_sdks["subgraphs"] is True


@pytest.mark.usefixtures("use_real_langgraph")
async def test_subagents_are_gated_by_blocked_tools(tmp_path, make_payload):
    pytest.importorskip("langchain.agents.middleware")
    from langchain_core.messages import ToolMessage

    payload = make_payload(tmp_path)
    payload["config"]["tools"] = {"blocked": ["write_file"]}
    payload["config"]["harness"]["settings"]["deepagents"] = {
        "subagents": [
            {
                "name": "researcher",
                "description": "Researches the workspace.",
                "system_prompt": "Research.",
            }
        ]
    }

    agent_config = AgentConfig.from_mapping(payload["config"])
    create_kwargs = await adapter.build_agent_kwargs(
        agent_config,
        RuntimeContext.from_mapping(payload["runtime_context"]),
        payload["base_dir"],
        MagicMock(),
        agent_config.harness.settings,
    )
    assert create_kwargs["middleware"], (
        "main agent blocked-tools middleware not attached"
    )
    subagents = create_kwargs["subagents"]
    assert [subagent["name"] for subagent in subagents] == [
        "general-purpose",
        "researcher",
    ]
    assert all(subagent["middleware"] for subagent in subagents)

    async def handler(_request: types.SimpleNamespace) -> str:
        return "executed"

    def request(name: str) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            tool_call={"name": name, "id": "call-1", "args": {}}
        )

    gates = [create_kwargs["middleware"][-1]]
    gates.extend(subagent["middleware"][-1] for subagent in subagents)
    assert all(type(gate) is adapter.ToolGateMiddleware for gate in gates)
    for middleware in gates:
        blocked = await middleware.awrap_tool_call(request("write_file"), handler)
        assert isinstance(blocked, ToolMessage)
        assert blocked.status == "error"
        assert (
            await middleware.awrap_tool_call(request("read_file"), handler)
            == "executed"
        )


@pytest.mark.usefixtures("use_real_langgraph")
async def test_default_subagent_is_gated_by_blocked_tools(tmp_path, make_payload):
    payload = make_payload(tmp_path)
    payload["config"]["tools"] = {"blocked": ["write_file"]}

    agent_config = AgentConfig.from_mapping(payload["config"])
    create_kwargs = await adapter.build_agent_kwargs(
        agent_config,
        RuntimeContext.from_mapping(payload["runtime_context"]),
        payload["base_dir"],
        MagicMock(),
        agent_config.harness.settings,
    )

    assert [subagent["name"] for subagent in create_kwargs["subagents"]] == [
        "general-purpose"
    ]
    assert create_kwargs["subagents"][0]["middleware"]


@pytest.mark.parametrize(
    "unsupported", [{"graph_id": "remote"}, {"runnable": "compiled"}]
)
async def test_blocked_tools_reject_unenforceable_subagents(
    tmp_path, make_payload, unsupported
):
    payload = make_payload(tmp_path)
    payload["config"]["tools"] = {"blocked": ["write_file"]}
    payload["config"]["harness"]["settings"]["deepagents"] = {
        "subagents": [{"name": "worker", **unsupported}]
    }

    agent_config = AgentConfig.from_mapping(payload["config"])
    with pytest.raises(adapter.AdapterConfigError, match="cannot be enforced"):
        await adapter.build_agent_kwargs(
            agent_config,
            RuntimeContext.from_mapping(payload["runtime_context"]),
            payload["base_dir"],
            MagicMock(),
            agent_config.harness.settings,
        )


@pytest.mark.parametrize(
    ("subagents", "message"),
    [
        (
            {"name": "researcher"},
            "harness.settings.deepagents.subagents must be a list when a tools policy is configured.",
        ),
        (
            [{"name": "researcher"}, "invalid"],
            "Deep Agents subagents must be mappings when a tools policy is configured.",
        ),
    ],
)
def test_gated_subagents_reject_invalid_configuration(subagents, message):
    with pytest.raises(adapter.AdapterConfigError) as error:
        adapter._gated_subagents(subagents, None, {"write_file"})

    assert str(error.value) == message


async def test_deepagents_passthrough_forwards_supported_options(
    tmp_path, make_payload, fake_sdks
):
    # Documented JSON-serializable options reach create_deep_agent unchanged.
    payload = make_payload(tmp_path)
    payload["config"]["harness"]["settings"]["deepagents"] = {
        "interrupt_on": {"write_file": True}
    }

    await invoke_once(payload)

    assert fake_sdks["create_kwargs"]["interrupt_on"] == {"write_file": True}


async def test_deepagents_passthrough_cannot_override_fabric_owned_keys(
    tmp_path, make_payload
):
    # Overriding a Fabric-owned key (here backend) would defeat workspace confinement;
    # it must fail loudly rather than silently replacing the derived value.
    payload = make_payload(tmp_path)
    payload["config"]["harness"]["settings"]["deepagents"] = {
        "backend": {"root_dir": "/etc"}
    }

    with pytest.raises(adapter.AdapterConfigError, match="backend"):
        await adapter.DeepAgentsRuntime().start(lifecycle_start_payload(payload))


async def test_deepagents_passthrough_rejects_unknown_option(tmp_path, make_payload):
    # A typo or unsupported option must fail clearly instead of being silently dropped.
    payload = make_payload(tmp_path)
    payload["config"]["harness"]["settings"]["deepagents"] = {
        "interupt_on": {}  # note the typo
    }

    with pytest.raises(adapter.AdapterConfigError, match="interupt_on"):
        await adapter.DeepAgentsRuntime().start(lifecycle_start_payload(payload))


async def test_subagent_usage_folded_from_subgraph(tmp_path, make_payload, monkeypatch):
    # Usage from a delegated subagent is emitted under a subgraph namespace; folding
    # it into this turn keeps usage/cost accurate. Duplicate ids are counted once.
    import deepagents

    sub_ai = {
        "role": "ai",
        "content": "subagent work",
        "id": "sub-1",
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
    }
    main_ai = {
        "role": "ai",
        "content": "final",
        "id": "main-1",
        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    }

    async def astream(inputs, config=None, *, stream_mode=None, subgraphs=False):
        assert subgraphs is True
        # subagent step under a subgraph namespace, emitted twice (dedup by id)
        yield (("task:researcher",), "updates", {"agent": {"messages": [sub_ai]}})
        yield (("task:researcher",), "updates", {"agent": {"messages": [sub_ai]}})
        # main graph step + replayed final state
        yield ((), "updates", {"agent": {"messages": [main_ai]}})
        yield ((), "values", {"messages": [main_ai]})

    agent = MagicMock()
    agent.astream = astream
    monkeypatch.setattr(deepagents, "create_deep_agent", MagicMock(return_value=agent))

    output = await invoke_once(make_payload(tmp_path))

    # subagent (10/20/30) + main (2/3/5), the duplicate subagent message counted once
    assert output["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 23,
        "total_tokens": 35,
    }
    # the subgraph step is recorded with its namespace label
    assert any(evt.get("subgraph") == "task:researcher" for evt in output["events"])


async def test_bad_mcp_transport_fails_runtime_start(tmp_path, make_payload):
    # A misconfigured MCP server must fail loudly, not be silently dropped.
    payload = make_payload(tmp_path)
    payload["config"]["mcp"] = {
        "servers": {"bad": {"transport": "carrier-pigeon", "url": "http://x/mcp"}}
    }

    with pytest.raises(adapter.AdapterConfigError, match="transport"):
        await adapter.DeepAgentsRuntime().start(lifecycle_start_payload(payload))


async def test_empty_mcp_url_fails_runtime_start(tmp_path, make_payload):
    payload = make_payload(tmp_path)
    payload["config"]["mcp"] = {
        "servers": {"bad": {"transport": "streamable_http", "url": ""}}
    }

    with pytest.raises(ContractValidationError, match="url"):
        await adapter.DeepAgentsRuntime().start(lifecycle_start_payload(payload))


async def test_unknown_provider_requires_api_key_env(
    tmp_path, make_payload, monkeypatch
):
    # An unknown provider with no explicit api_key_env must fail loudly rather than
    # defaulting to NVIDIA_API_KEY and sending the wrong key to the endpoint.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    payload = make_payload(tmp_path)
    payload["config"]["models"]["default"] = {
        "provider": "anthropic",
        "model": "claude-x",
    }

    with pytest.raises(adapter.AdapterConfigError, match="api_key_env"):
        await adapter.DeepAgentsRuntime().start(lifecycle_start_payload(payload))


async def test_openai_provider_defaults_to_openai_key(
    tmp_path, make_payload, monkeypatch, fake_sdks
):
    # provider openai with no explicit api_key_env defaults to OPENAI_API_KEY, never
    # NVIDIA_API_KEY, and keeps ChatOpenAI's own endpoint.
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    payload = make_payload(tmp_path)
    payload["config"]["models"]["default"] = {
        "provider": "openai",
        "model": "gpt-4o",
    }

    output = await invoke_once(payload)

    assert output["failed"] is False, output["error"]
    assert output["base_url"] is None
    assert "base_url" not in fake_sdks["chat_openai"].call_args.kwargs


async def test_openai_compatible_provider_requires_api_key_env(tmp_path, make_payload):
    # openai-compatible uses ChatOpenAI but has no default credential var, so it must
    # set api_key_env explicitly rather than silently falling back to NVIDIA_API_KEY.
    payload = make_payload(tmp_path)
    payload["config"]["models"]["default"] = {
        "provider": "openai-compatible",
        "model": "some/model",
    }

    with pytest.raises(adapter.AdapterConfigError, match="api_key_env"):
        await adapter.DeepAgentsRuntime().start(lifecycle_start_payload(payload))


async def test_openai_compatible_provider_requires_base_url(tmp_path, make_payload):
    os.environ["CUSTOM_API_KEY"] = "sk-test"
    payload = make_payload(tmp_path)
    payload["config"]["models"]["default"] = {
        "provider": "openai-compatible",
        "model": "some/model",
        "api_key_env": "CUSTOM_API_KEY",
    }

    with pytest.raises(adapter.AdapterConfigError, match="base_url"):
        await adapter.DeepAgentsRuntime().start(lifecycle_start_payload(payload))


def test_main_serves_persistent_runtime(monkeypatch):
    serve = MagicMock()
    monkeypatch.setattr(adapter.lifecycle, "serve", serve)

    adapter.main()

    serve.assert_called_once_with(
        adapter.DeepAgentsRuntime, config_loader=AgentConfig.from_mapping
    )
