# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Deep Agents adapter's Fabric runtime mapping.

These tests stub the ``deepagents``/``langchain``/``langgraph`` SDKs so they run
without the real harness installed; they assert the normalized Fabric result and
the session/resume thread-id handling. The real SDK is exercised by the opt-in
integration test in ``tests/e2e/test_deepagents.py``.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from nemo_fabric_adapters.deepagents import adapter  # noqa: E402


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
            # this turn; ``values`` is the full (on resume, replayed) state.
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
                    "harness": {"settings": {"system_prompt": "be concise"}},
                    "models": {
                        "default": {
                            "provider": "nvidia",
                            "model": "nvidia/nemotron-3-nano-30b-a3b",
                            "api_key_env": "NVIDIA_API_KEY",
                        }
                    },
            },
            "runtime_context": {
                "runtime_id": runtime_id,
                "invocation_id": "inv-1",
                "environment": {"workspace": str(tmp_path)},
            },
            "request": {"input": "hello"},
            "capability_plan": {},
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
    async def plugin_ctx(_config):
        calls["plugin_open"] = True
        calls["plugin_enters"] = calls.get("plugin_enters", 0) + 1
        try:
            yield
        finally:
            calls["plugin_exits"] = calls.get("plugin_exits", 0) + 1

    class ScopeType:
        Agent = "agent"

    @contextlib.contextmanager
    def scope_ctx(name: str, scope_type: object, **_: object) -> Iterator[None]:
        # Record every scope entered so tests can assert the top-level
        # ``deepagents-request`` Agent scope wraps the invocation.
        calls.setdefault("scopes", []).append((name, scope_type))
        yield

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


async def test_oneshot_normalizes_response_usage_and_thread(
    tmp_path, make_payload, fake_sdks
):
    output = await adapter.run_deepagents(make_payload(tmp_path))

    assert output["harness"] == "deepagents"
    assert output["mode"] == "deepagents"
    assert output["model"] == "nvidia/nemotron-3-nano-30b-a3b"
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
    # a LangGraph thread id is assigned and reported; a fresh runtime is not a resume
    assert output["thread_id"]
    assert output["resumed"] is False
    assert output["completed"] is True
    assert output["failed"] is False
    assert output["error"] is None
    # system prompt must reach deepagents under the real param name (not ``instructions``)
    assert fake_sdks["create_kwargs"]["system_prompt"] == "be concise"
    assert "instructions" not in fake_sdks["create_kwargs"]


async def test_missing_api_key_is_normalized(tmp_path, make_payload, monkeypatch):
    # Missing model-provider auth is caught by the adapter preflight. Because the
    # preflight runs inside the guarded scope, the failure is normalized into the
    # Fabric result rather than raising a raw traceback.
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    output = await adapter.run_deepagents(make_payload(tmp_path))

    assert output["failed"] is True
    assert output["completed"] is False
    assert "NVIDIA_API_KEY" in output["error"]


async def test_missing_deepagents_package_is_normalized(
    tmp_path, make_payload, monkeypatch
):
    # Preflight reports a clear, normalized error when the deepagents package is
    # absent. Force find_spec("deepagents") -> None so the test holds whether or
    # not the real package is installed in the environment.
    import importlib.util as importlib_util

    real_find_spec = importlib_util.find_spec

    def fake_find_spec(
        name: str, *args: object, **kwargs: object
    ) -> importlib.machinery.ModuleSpec | None:
        if name == "deepagents":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib_util, "find_spec", fake_find_spec)
    output = await adapter.run_deepagents(make_payload(tmp_path))

    assert output["failed"] is True
    assert "deepagents" in output["error"]


async def test_invocation_error_is_normalized(tmp_path, make_payload, monkeypatch):
    # Errors raised during the agent run are normalized into the Fabric result.
    import deepagents

    def boom(**_kwargs):
        raise RuntimeError("agent exploded")

    monkeypatch.setattr(deepagents, "create_deep_agent", boom)
    output = await adapter.run_deepagents(make_payload(tmp_path))

    assert output["failed"] is True
    assert output["completed"] is False
    assert "agent exploded" in output["error"]


async def test_relay_telemetry_wraps_agent_and_reports_artifacts(
    tmp_path, make_payload, monkeypatch, fake_sdks, fake_relay
):
    artifacts = [{"kind": "atof", "path": str(tmp_path / "events.atof.jsonl")}]
    monkeypatch.setattr(
        adapter.common_utils,
        "load_relay_plugin_config",
        lambda _p: {"version": 1, "components": []},
    )
    monkeypatch.setattr(
        adapter.common_utils, "relay_api_plugin_config", lambda _c: object()
    )
    monkeypatch.setattr(
        adapter.common_utils, "collect_relay_artifacts", lambda _c: artifacts
    )
    payload = make_payload(tmp_path)
    payload["telemetry_plan"] = {
        "providers": ["relay"],
        "relay_enabled": True,
        "relay_project": None,
        "relay_output_dir": None,
        "relay_config": {},
        "native_config": None,
        "adapter_outputs": [],
    }

    output = await adapter.run_deepagents(payload)

    assert fake_relay["wrapped"]
    assert fake_relay["plugin_open"]
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
    # the Deep Agents callback handler is added to the LangGraph run config so
    # LangGraph scopes and human-in-the-loop interrupt/resume marks are captured
    assert fake_relay["callback_handler"] in (fake_sdks["config"] or {}).get(
        "callbacks", []
    )


async def test_native_telemetry_exports_without_artifacts(
    tmp_path, make_payload, monkeypatch, fake_sdks, fake_relay
):
    monkeypatch.setattr(
        adapter.common_utils, "relay_api_plugin_config", lambda _c: object()
    )

    payload = make_payload(tmp_path)
    payload["telemetry_plan"] = {
        "providers": ["native"],
        "relay_enabled": False,
        "relay_project": None,
        "relay_output_dir": None,
        "relay_config": None,
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
        "adapter_outputs": [],
    }

    output = await adapter.run_deepagents(payload)

    assert fake_relay["wrapped"]
    assert fake_relay["plugin_open"]
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
    assert fake_relay["callback_handler"] in (fake_sdks["config"] or {}).get(
        "callbacks", []
    )


async def test_relay_disabled_adds_no_scope_or_callbacks(
    tmp_path, make_payload, fake_sdks
):
    # With telemetry disabled the invocation runs without a Relay scope, callback
    # handler, or middleware, preserving the Relay-neutral default behavior.
    output = await adapter.run_deepagents(make_payload(tmp_path))

    assert output["completed"] is True
    assert "telemetry" not in output
    assert (fake_sdks["config"] or {}).get("callbacks") is None
    assert "relay-mw" not in (fake_sdks["create_kwargs"].get("middleware") or [])


async def test_missing_nemo_relay_with_native_telemetry_is_normalized(
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
    payload["telemetry_plan"] = {
        "providers": ["native"],
        "relay_enabled": False,
        "native_config": {
            "version": 1,
            "components": [
                {"kind": "observability", "enabled": True, "config": {"version": 1}}
            ],
        },
        "adapter_outputs": [],
    }

    output = await adapter.run_deepagents(payload)

    assert output["failed"] is True
    assert "nemo-relay" in output["error"]
    assert "[relay]" in output["error"]


@pytest.mark.usefixtures("fake_relay")
async def test_incomplete_nemo_relay_install_is_normalized(
    tmp_path, make_payload, monkeypatch
):
    monkeypatch.delitem(sys.modules, "nemo_relay.integrations.deepagents")
    payload = make_payload(tmp_path)
    payload["telemetry_plan"] = {
        "providers": ["native"],
        "relay_enabled": False,
        "native_config": {
            "version": 1,
            "components": [
                {"kind": "observability", "enabled": True, "config": {"version": 1}}
            ],
        },
        "adapter_outputs": [],
    }

    output = await adapter.run_deepagents(payload)

    assert output["failed"] is True
    assert "compatible 'nemo-relay' package" in output["error"]
    assert "[relay]" in output["error"]


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


async def test_invoke_agent_wires_callbacks_into_run_config(fake_sdks):
    # invoke_agent threads the supplied callbacks into the LangGraph run config.
    agent_kwargs = {"model": object()}
    await adapter.invoke_agent(
        agent_kwargs, "hello", "thread-1", callbacks=["cb-a", "cb-b"]
    )

    config = fake_sdks["config"]
    assert config["configurable"]["thread_id"] == "thread-1"
    assert config["callbacks"] == ["cb-a", "cb-b"]


async def test_invoke_agent_without_callbacks_sets_no_callbacks_key(fake_sdks):
    await adapter.invoke_agent({"model": object()}, "hello", None, callbacks=None)

    # No thread and no callbacks means the agent is streamed without a config.
    assert fake_sdks["config"] is None


async def test_workspace_roots_filesystem_backend(tmp_path, make_payload, fake_sdks):
    await adapter.run_deepagents(make_payload(tmp_path))
    backend_kwargs = fake_sdks["fs_backend"].call_args.kwargs
    assert backend_kwargs["root_dir"] == str(tmp_path)
    # virtual_mode=True confines the agent to root_dir: absolute paths and ``..``
    # cannot escape the workspace (and it does not rely on the deprecated default).
    assert backend_kwargs["virtual_mode"] is True


async def test_checkpointer_closed_on_success_and_failure(
    tmp_path, make_payload, monkeypatch, fake_sdks
):
    # The async checkpointer must be closed on both the success and error paths.
    await adapter.run_deepagents(make_payload(tmp_path))
    assert fake_sdks["saver_exits"] == 1

    import deepagents

    def boom(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(deepagents, "create_deep_agent", boom)
    output = await adapter.run_deepagents(make_payload(tmp_path))
    assert output["failed"] is True
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
    # McpServerPlan carries the URL/command in ``url``.
    payload["capability_plan"] = {
        "native": {
            "mcp_servers": {
                "fs": {"transport": "streamable-http", "url": "http://localhost:9/mcp"},
                "local": {"transport": "stdio", "url": "my-server --flag"},
            }
        }
    }

    output = await adapter.run_deepagents(payload)

    assert output["failed"] is False
    # Fabric MCP transport is normalized; stdio command/args come from ``url``
    assert mock_client_cls.call_args.args[0] == {
        "fs": {"transport": "streamable_http", "url": "http://localhost:9/mcp"},
        "local": {"transport": "stdio", "command": "my-server", "args": ["--flag"]},
    }
    tool_names = [tool.name for tool in fake_sdks["create_kwargs"]["tools"]]
    assert tool_names == ["read_file", "write_file"]


@pytest.mark.usefixtures("use_real_langgraph")
async def test_blocked_tools_middleware_blocks_configured_tools():
    pytest.importorskip("langchain.agents.middleware")
    from langchain_core.messages import ToolMessage

    middleware = adapter.blocked_tools_middleware({"write_file"})

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

    output = await adapter.run_deepagents(make_payload(tmp_path))

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

    output = await adapter.run_deepagents(payload)

    # openai must NOT be redirected to NVIDIA's endpoint
    assert output["base_url"] is None
    assert "base_url" not in fake_sdks["chat_openai"].call_args.kwargs


async def test_skill_paths_map_to_skills(tmp_path, make_payload, fake_sdks):
    payload = make_payload(tmp_path)
    payload["capability_plan"] = {"native": {"skill_paths": ["/skills/a", "/skills/b"]}}

    await adapter.run_deepagents(payload)

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
    output = await adapter.run_deepagents(make_payload(tmp_path))

    assert output["usage"]["cost"] == 0.0025


async def test_resumed_usage_counts_current_turn_only(
    tmp_path, make_payload, monkeypatch
):
    # On a resumed run the final state replays the prior turn's messages; usage and
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
        # ...but the resumed final state also replays the prior turn.
        yield ((), "values", {"messages": [prior, current]})

    agent = MagicMock()
    agent.astream = astream
    monkeypatch.setattr(deepagents, "create_deep_agent", MagicMock(return_value=agent))

    output = await adapter.run_deepagents(make_payload(tmp_path))

    assert output["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "total_tokens": 4,
        "cost": 0.002,
    }
    # the full transcript is still returned
    assert output["message_count"] == 2


async def test_runtime_resume_reuses_thread_id(tmp_path, make_payload, fake_sdks):
    # Two invocations of the same runtime_id (a started runtime) resume the same
    # LangGraph thread; a one-shot run gets a fresh runtime_id and never resumes.
    payload = make_payload(tmp_path, runtime_id="run-42")

    first = await adapter.run_deepagents(payload)
    assert first["resumed"] is False
    thread_id = first["thread_id"]
    assert thread_id
    # thread id was threaded into the LangGraph config on the invocation
    assert fake_sdks["config"] == {"configurable": {"thread_id": thread_id}}

    second = await adapter.run_deepagents(payload)
    assert second["resumed"] is True
    assert second["thread_id"] == thread_id


async def test_persistent_runtime_reuses_compiled_agent_and_checkpointer(
    tmp_path, make_payload, fake_sdks
):
    import deepagents

    payload = make_payload(tmp_path, runtime_id="run-persistent")
    start_payload = dict(payload)
    start_payload.pop("request")
    runtime = adapter.DeepAgentsRuntime()

    await runtime.start(start_payload)
    first = await runtime.invoke(payload)
    payload["runtime_context"]["invocation_id"] = "inv-2"
    payload["request"]["input"] = "continue"
    second = await runtime.invoke(payload)

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
    monkeypatch.setattr(
        adapter.common_utils,
        "load_relay_plugin_config",
        lambda _payload: {"version": 1, "components": []},
    )
    monkeypatch.setattr(
        adapter.common_utils, "relay_api_plugin_config", lambda _config: object()
    )
    monkeypatch.setattr(
        adapter.common_utils,
        "collect_relay_artifacts",
        lambda _config: artifacts,
    )
    payload = make_payload(tmp_path, runtime_id="run-relay-persistent")
    payload["telemetry_plan"] = {
        "providers": ["relay"],
        "relay_enabled": True,
        "relay_project": None,
        "relay_output_dir": None,
        "relay_config": {},
        "native_config": None,
        "adapter_outputs": ["atif"],
    }
    start_payload = dict(payload)
    start_payload.pop("request")
    runtime = adapter.DeepAgentsRuntime()

    await runtime.start(start_payload)
    first = await runtime.invoke(payload)
    payload["runtime_context"]["invocation_id"] = "inv-2"
    payload["request"]["input"] = "continue"
    second = await runtime.invoke(payload)
    await runtime.stop()

    assert fake_relay["integration_adds"] == 1
    assert fake_relay["plugin_enters"] == 2
    assert fake_relay["plugin_exits"] == 2
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
    await adapter.run_deepagents(make_payload(tmp_path))
    assert fake_sdks["subgraphs"] is True


@pytest.mark.usefixtures("use_real_langgraph")
async def test_subagents_are_gated_by_blocked_tools(tmp_path, make_payload):
    pytest.importorskip("langchain.agents.middleware")
    from langchain_core.messages import ToolMessage

    payload = make_payload(tmp_path)
    payload["config"]["tools"] = {"blocked": ["write_file"]}
    payload["config"]["harness"]["settings"]["deepagents"] = {
        "subagents": [{"name": "researcher", "prompt": "research"}]
    }

    settings = payload["config"]["harness"]["settings"]
    create_kwargs = await adapter.build_agent_kwargs(payload, MagicMock(), settings)
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

    settings = payload["config"]["harness"]["settings"]
    create_kwargs = await adapter.build_agent_kwargs(payload, MagicMock(), settings)

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

    settings = payload["config"]["harness"]["settings"]
    with pytest.raises(adapter.AdapterConfigError, match="cannot be enforced"):
        await adapter.build_agent_kwargs(payload, MagicMock(), settings)


@pytest.mark.parametrize(
    ("subagents", "message"),
    [
        (
            {"name": "researcher"},
            "harness.settings.deepagents.subagents must be a list when tools.blocked is configured.",
        ),
        (
            [{"name": "researcher"}, "invalid"],
            "Deep Agents subagents must be mappings when tools.blocked is configured.",
        ),
    ],
)
def test_gated_subagents_reject_invalid_configuration(subagents, message):
    with pytest.raises(adapter.AdapterConfigError) as error:
        adapter._gated_subagents(subagents, {"write_file"})

    assert str(error.value) == message


async def test_deepagents_passthrough_forwards_supported_options(
    tmp_path, make_payload, fake_sdks
):
    # Documented JSON-serializable options reach create_deep_agent unchanged.
    payload = make_payload(tmp_path)
    payload["config"]["harness"]["settings"]["deepagents"] = {
        "interrupt_on": {"write_file": True}
    }

    await adapter.run_deepagents(payload)

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

    output = await adapter.run_deepagents(payload)

    assert output["failed"] is True
    assert "backend" in output["error"]


async def test_deepagents_passthrough_rejects_unknown_option(tmp_path, make_payload):
    # A typo or unsupported option must fail clearly instead of being silently dropped.
    payload = make_payload(tmp_path)
    payload["config"]["harness"]["settings"]["deepagents"] = {
        "interupt_on": {}  # note the typo
    }

    output = await adapter.run_deepagents(payload)

    assert output["failed"] is True
    assert "interupt_on" in output["error"]


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

    output = await adapter.run_deepagents(make_payload(tmp_path))

    # subagent (10/20/30) + main (2/3/5), the duplicate subagent message counted once
    assert output["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 23,
        "total_tokens": 35,
    }
    # the subgraph step is recorded with its namespace label
    assert any(evt.get("subgraph") == "task:researcher" for evt in output["events"])


async def test_bad_mcp_transport_is_normalized_failure(tmp_path, make_payload):
    # A misconfigured MCP server must fail loudly, not be silently dropped.
    payload = make_payload(tmp_path)
    payload["capability_plan"] = {
        "native": {
            "mcp_servers": {
                "bad": {"transport": "carrier-pigeon", "url": "http://x/mcp"}
            }
        }
    }

    output = await adapter.run_deepagents(payload)

    assert output["failed"] is True
    assert "transport" in output["error"]


async def test_empty_mcp_url_is_normalized_failure(tmp_path, make_payload):
    payload = make_payload(tmp_path)
    payload["capability_plan"] = {
        "native": {"mcp_servers": {"bad": {"transport": "streamable_http", "url": ""}}}
    }

    output = await adapter.run_deepagents(payload)

    assert output["failed"] is True
    assert "url" in output["error"]


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

    output = await adapter.run_deepagents(payload)

    assert output["failed"] is True
    assert "api_key_env" in output["error"]


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

    output = await adapter.run_deepagents(payload)

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

    output = await adapter.run_deepagents(payload)

    assert output["failed"] is True
    assert "api_key_env" in output["error"]
