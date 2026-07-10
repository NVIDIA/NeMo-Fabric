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
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _src in ("adapters/common/src", "adapters/deepagents/src"):
    _path = str(ROOT / _src)
    if _path not in sys.path:
        sys.path.insert(0, _path)

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

        async def astream(inputs, config=None, *, stream_mode=None):
            recorder["config"] = config
            recorder["checkpointer"] = kwargs.get("checkpointer")
            user = inputs["messages"][-1]["content"]
            ai = {
                "role": "ai",
                "content": f"reply to {user}",
                "usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
            }
            # ``updates`` carries the message produced this turn; ``values`` is the
            # full (on resume, replayed) state.
            yield ("updates", {"agent": {"messages": [ai]}})
            yield ("values", {"messages": [{"role": "user", "content": user}, ai]})

        agent = MagicMock()
        agent.astream = astream
        return agent

    deepagents_mod = types.ModuleType("deepagents")
    deepagents_mod.__spec__ = importlib.machinery.ModuleSpec("deepagents", loader=None)
    deepagents_mod.create_deep_agent = MagicMock(side_effect=build_agent)
    backends_mod = types.ModuleType("deepagents.backends")
    backends_mod.FilesystemBackend = mock_fs_backend
    deepagents_mod.backends = backends_mod
    monkeypatch.setitem(sys.modules, "deepagents", deepagents_mod)
    monkeypatch.setitem(sys.modules, "deepagents.backends", backends_mod)

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
            "effective_config": {
                "config_root": str(tmp_path),
                "config": {
                    "harness": {"settings": {"system_prompt": "be concise"}},
                    "models": {
                        "default": {
                            "provider": "nvidia",
                            "model": "nvidia/nemotron-3-nano-30b-a3b",
                            "api_key_env": "NVIDIA_API_KEY",
                        }
                    },
                    "telemetry": {"enabled": False},
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
        return merged

    @contextlib.asynccontextmanager
    async def plugin_ctx(_config):
        calls["plugin_open"] = True
        yield

    relay_root = types.ModuleType("nemo_relay")
    plugin_mod = types.ModuleType("nemo_relay.plugin")
    plugin_mod.plugin = plugin_ctx
    relay_root.plugin = plugin_mod
    integrations_pkg = types.ModuleType("nemo_relay.integrations")
    da_integ = types.ModuleType("nemo_relay.integrations.deepagents")
    da_integ.add_nemo_relay_integration = add_nemo_relay_integration
    for name, mod in (
        ("nemo_relay", relay_root),
        ("nemo_relay.plugin", plugin_mod),
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


async def test_oneshot_normalizes_response_usage_and_thread(tmp_path, make_payload, fake_sdks):
    output = await adapter.run_deepagents(make_payload(tmp_path))

    assert output["harness"] == "deepagents"
    assert output["mode"] == "deepagents"
    assert output["model"] == "nvidia/nemotron-3-nano-30b-a3b"
    assert output["response"] == "reply to hello"
    assert output["message_count"] == 2
    assert output["usage"] == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
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


async def test_missing_api_key_raises(tmp_path, make_payload, monkeypatch):
    # Missing model-provider auth is caught by the adapter preflight. The
    # descriptor declares no static env requirement because the credential is
    # provider-specific.
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        await adapter.run_deepagents(make_payload(tmp_path))


async def test_missing_deepagents_package_raises(tmp_path, make_payload, monkeypatch):
    # Preflight reports a clear error when the deepagents package is absent.
    # Force find_spec("deepagents") -> None so the test holds whether or not the
    # real package is installed in the environment.
    import importlib.util as importlib_util

    real_find_spec = importlib_util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "deepagents":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib_util, "find_spec", fake_find_spec)
    with pytest.raises(RuntimeError, match="deepagents"):
        await adapter.run_deepagents(make_payload(tmp_path))


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
        adapter.common_utils, "load_relay_plugin_config", lambda _p: {"version": 1, "components": []}
    )
    monkeypatch.setattr(adapter.common_utils, "relay_api_plugin_config", lambda _c: object())
    monkeypatch.setattr(adapter.common_utils, "collect_relay_artifacts", lambda _c: artifacts)
    monkeypatch.setenv("FABRIC_RELAY_ENABLED", "true")

    output = await adapter.run_deepagents(make_payload(tmp_path))

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


async def test_native_telemetry_exports_without_artifacts(
    tmp_path, make_payload, monkeypatch, fake_sdks, fake_relay
):
    monkeypatch.setattr(adapter.common_utils, "relay_api_plugin_config", lambda _c: object())

    payload = make_payload(tmp_path)
    payload["effective_config"]["config"]["telemetry"] = {
        "enabled": True,
        "provider": "native",
        "config": {
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


async def test_workspace_roots_filesystem_backend(tmp_path, make_payload, fake_sdks):
    await adapter.run_deepagents(make_payload(tmp_path))
    assert fake_sdks["fs_backend"].call_args.kwargs["root_dir"] == str(tmp_path)


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


async def test_mcp_servers_become_tools_filtered_by_allowed(
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
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters", types.ModuleType("langchain_mcp_adapters"))
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
async def test_allowed_tools_recorded_as_middleware(tmp_path, make_payload, fake_sdks):
    # An allow-list is enforced by a gating middleware over the full tool surface,
    # not by filtering the passed tools list.
    pytest.importorskip("langchain.agents.middleware")
    pytest.importorskip("langgraph.checkpoint.sqlite.aio")
    payload = make_payload(tmp_path)
    payload["effective_config"]["config"]["tools"] = ["read_file"]

    await adapter.run_deepagents(payload)

    assert fake_sdks["create_kwargs"]["middleware"], "gating middleware not attached"


@pytest.mark.usefixtures("use_real_langgraph")
async def test_allowed_tools_middleware_blocks_disallowed_tools():
    pytest.importorskip("langchain.agents.middleware")
    from langchain_core.messages import ToolMessage

    middleware = adapter.allowed_tools_middleware({"read_file"})

    async def handler(_request):
        return "executed"

    def request(name):
        return types.SimpleNamespace(tool_call={"name": name, "id": "call-1", "args": {}})

    blocked = await middleware.awrap_tool_call(request("write_file"), handler)
    assert isinstance(blocked, ToolMessage)
    assert blocked.status == "error"

    allowed = await middleware.awrap_tool_call(request("read_file"), handler)
    assert allowed == "executed"

    # An explicitly empty allow-list denies every tool.
    deny_all = adapter.allowed_tools_middleware(set())
    denied = await deny_all.awrap_tool_call(request("read_file"), handler)
    assert isinstance(denied, ToolMessage)
    assert denied.status == "error"


def test_empty_tools_is_deny_all_not_none():
    # An explicitly empty tools list is a deny-all allow-list, not "no allow-list".
    assert adapter._allowed_tool_names({"effective_config": {"config": {"tools": []}}}) == set()
    assert adapter._allowed_tool_names({"effective_config": {"config": {}}}) is None


@pytest.mark.usefixtures("use_real_langgraph")
async def test_real_langgraph_async_checkpointer(tmp_path, make_payload, monkeypatch):
    # Regression: driving astream with the sync SqliteSaver raises NotImplementedError.
    # Exercise the adapter against a real compiled LangGraph graph + AsyncSqliteSaver.
    pytest.importorskip("langgraph.graph")
    pytest.importorskip("langgraph.checkpoint.sqlite.aio")

    import deepagents
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START, MessagesState, StateGraph

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


async def test_openai_provider_keeps_openai_endpoint(tmp_path, make_payload, monkeypatch, fake_sdks):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    payload = make_payload(tmp_path)
    payload["effective_config"]["config"]["models"]["default"] = {
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


async def test_cost_is_extracted_from_response_metadata(tmp_path, make_payload, monkeypatch):
    import deepagents

    message = {
        "role": "ai",
        "content": "done",
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        "response_metadata": {"cost": 0.0025},
    }

    async def astream(inputs, config=None, *, stream_mode=None):
        yield ("updates", {"agent": {"messages": [message]}})
        yield ("values", {"messages": [message]})

    agent = MagicMock()
    agent.astream = astream
    monkeypatch.setattr(deepagents, "create_deep_agent", MagicMock(return_value=agent))
    output = await adapter.run_deepagents(make_payload(tmp_path))

    assert output["usage"]["cost"] == 0.0025


async def test_resumed_usage_counts_current_turn_only(tmp_path, make_payload, monkeypatch):
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

    async def astream(inputs, config=None, *, stream_mode=None):
        # Only the current turn's message is emitted as an update...
        yield ("updates", {"agent": {"messages": [current]}})
        # ...but the resumed final state also replays the prior turn.
        yield ("values", {"messages": [prior, current]})

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
