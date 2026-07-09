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

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _src in ("adapters/common/src", "adapters/deepagents/src"):
    _path = str(ROOT / _src)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from nemo_fabric_adapters.deepagents import adapter  # noqa: E402


class _FakeChatOpenAI:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeFilesystemBackend:
    def __init__(self, root_dir: str | None = None, **_kwargs: Any) -> None:
        self.root_dir = root_dir


class _FakeAgent:
    def __init__(self, kwargs: dict[str, Any], recorder: dict[str, Any]) -> None:
        self.kwargs = kwargs
        self._recorder = recorder

    async def ainvoke(self, inputs: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        self._recorder["config"] = config
        self._recorder["checkpointer"] = self.kwargs.get("checkpointer")
        user = inputs["messages"][-1]["content"]
        return {
            "messages": [
                {"role": "user", "content": user},
                {
                    "role": "ai",
                    "content": f"reply to {user}",
                    "usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
                },
            ]
        }


@pytest.fixture(name="fake_sdks", autouse=True)
def fake_sdks_fixture(monkeypatch) -> dict[str, Any]:
    recorder: dict[str, Any] = {}

    def create_deep_agent(**kwargs: Any) -> _FakeAgent:
        recorder["create_kwargs"] = kwargs
        return _FakeAgent(kwargs, recorder)

    deepagents_mod = types.ModuleType("deepagents")
    deepagents_mod.__spec__ = importlib.machinery.ModuleSpec("deepagents", loader=None)
    deepagents_mod.create_deep_agent = create_deep_agent
    backends_mod = types.ModuleType("deepagents.backends")
    backends_mod.FilesystemBackend = _FakeFilesystemBackend
    deepagents_mod.backends = backends_mod
    monkeypatch.setitem(sys.modules, "deepagents", deepagents_mod)
    monkeypatch.setitem(sys.modules, "deepagents.backends", backends_mod)

    langchain_openai_mod = types.ModuleType("langchain_openai")
    langchain_openai_mod.ChatOpenAI = _FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", langchain_openai_mod)

    _install_fake_langgraph(monkeypatch)

    monkeypatch.setenv("NVIDIA_API_KEY", "test123")
    return recorder


class _FakeSaverCM:
    def __enter__(self) -> "_FakeSaver":
        return _FakeSaver()

    def __exit__(self, *_exc: Any) -> bool:
        return False


class _FakeSaver:
    """Minimal stand-in for langgraph's SqliteSaver checkpointer."""


class _FakeSqliteSaver:
    @classmethod
    def from_conn_string(cls, _conn: str) -> _FakeSaverCM:
        return _FakeSaverCM()


def _install_fake_langgraph(monkeypatch) -> None:
    langgraph_mod = types.ModuleType("langgraph")
    checkpoint_mod = types.ModuleType("langgraph.checkpoint")
    sqlite_mod = types.ModuleType("langgraph.checkpoint.sqlite")
    sqlite_mod.SqliteSaver = _FakeSqliteSaver
    checkpoint_mod.sqlite = sqlite_mod
    langgraph_mod.checkpoint = checkpoint_mod
    monkeypatch.setitem(sys.modules, "langgraph", langgraph_mod)
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint", checkpoint_mod)
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.sqlite", sqlite_mod)


def _payload(tmp_path: Path, *, runtime_id: str = "run-1") -> dict[str, Any]:
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


async def test_oneshot_normalizes_response_usage_and_thread(
    tmp_path: Path, fake_sdks: dict[str, Any]
) -> None:
    output = await adapter.run_deepagents(_payload(tmp_path))

    assert output["harness"] == "deepagents"
    assert output["mode"] == "deepagents"
    assert output["model"] == "nvidia/nemotron-3-nano-30b-a3b"
    assert output["response"] == "reply to hello"
    assert output["message_count"] == 2
    assert output["usage"] == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
    assert output["runtime_id"] == "run-1"
    # a LangGraph thread id is assigned and reported; a fresh runtime is not a resume
    assert output["thread_id"]
    assert output["resumed"] is False
    assert output["completed"] is True
    assert output["failed"] is False
    assert output["error"] is None
    # system prompt must reach deepagents under the real param name (not ``instructions``)
    assert fake_sdks["create_kwargs"].get("system_prompt") == "be concise"
    assert "instructions" not in fake_sdks["create_kwargs"]


async def test_missing_api_key_raises(tmp_path: Path, monkeypatch) -> None:
    # Missing model-provider auth is caught by the adapter preflight (doctor's
    # requirement.env check is the up-front guard for `fabric doctor`).
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        await adapter.run_deepagents(_payload(tmp_path))


async def test_missing_deepagents_package_raises(tmp_path: Path, monkeypatch) -> None:
    # Preflight reports a clear error when the deepagents package is absent.
    # Force find_spec("deepagents") -> None so the test holds whether or not the
    # real package is installed in the environment.
    import importlib.util as importlib_util

    real_find_spec = importlib_util.find_spec

    def fake_find_spec(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "deepagents":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib_util, "find_spec", fake_find_spec)
    with pytest.raises(RuntimeError, match="deepagents"):
        await adapter.run_deepagents(_payload(tmp_path))


async def test_invocation_error_is_normalized(tmp_path: Path, monkeypatch) -> None:
    # Errors raised during the agent run are normalized into the Fabric result.
    import deepagents

    def boom(**_kwargs: Any) -> Any:
        raise RuntimeError("agent exploded")

    monkeypatch.setattr(deepagents, "create_deep_agent", boom)
    output = await adapter.run_deepagents(_payload(tmp_path))

    assert output["failed"] is True
    assert output["completed"] is False
    assert "agent exploded" in output["error"]


def _install_fake_relay(monkeypatch, calls: dict[str, Any]) -> None:
    import contextlib

    def add_nemo_relay_integration(kwargs: dict[str, Any], **_: Any) -> dict[str, Any]:
        merged = dict(kwargs)
        merged["middleware"] = [*(merged.get("middleware") or []), "relay-mw"]
        calls["wrapped"] = True
        return merged

    @contextlib.asynccontextmanager
    async def plugin_ctx(_config: Any):
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


async def test_relay_telemetry_wraps_agent_and_reports_artifacts(
    tmp_path: Path, monkeypatch, fake_sdks: dict[str, Any]
) -> None:
    calls: dict[str, Any] = {}
    _install_fake_relay(monkeypatch, calls)
    artifacts = [{"kind": "atof", "path": str(tmp_path / "events.atof.jsonl")}]
    monkeypatch.setattr(adapter.common_utils, "load_relay_plugin_config", lambda _p: {"version": 1, "components": []})
    monkeypatch.setattr(adapter.common_utils, "relay_api_plugin_config", lambda _c: object())
    monkeypatch.setattr(adapter.common_utils, "collect_relay_artifacts", lambda _c: artifacts)
    monkeypatch.setenv("FABRIC_RELAY_ENABLED", "true")

    output = await adapter.run_deepagents(_payload(tmp_path))

    assert calls.get("wrapped") and calls.get("plugin_open")
    assert output["telemetry"] == {
        "enabled": True,
        "provider": "relay",
        "emitter": "deepagents.observability/nemo_relay",
    }
    assert output["relay_artifacts"] == artifacts
    # the relay middleware reached the deepagents kwargs
    assert "relay-mw" in fake_sdks["create_kwargs"].get("middleware", [])


async def test_native_telemetry_exports_without_artifacts(
    tmp_path: Path, monkeypatch, fake_sdks: dict[str, Any]
) -> None:
    calls: dict[str, Any] = {}
    _install_fake_relay(monkeypatch, calls)
    monkeypatch.setattr(adapter.common_utils, "relay_api_plugin_config", lambda _c: object())

    payload = _payload(tmp_path)
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

    assert calls.get("wrapped") and calls.get("plugin_open")
    assert output["telemetry"] == {
        "enabled": True,
        "provider": "native",
        "emitter": "deepagents.observability/native",
    }
    # native telemetry exports directly; no ATOF/ATIF relay artifacts are written
    assert "relay_artifacts" not in output
    assert "relay-mw" in fake_sdks["create_kwargs"].get("middleware", [])


async def test_workspace_roots_filesystem_backend(tmp_path: Path, fake_sdks: dict[str, Any]) -> None:
    await adapter.run_deepagents(_payload(tmp_path))
    backend = fake_sdks["create_kwargs"]["backend"]
    assert backend.root_dir == str(tmp_path)


async def test_mcp_servers_become_tools_filtered_by_allowed(
    tmp_path: Path, monkeypatch, fake_sdks: dict[str, Any]
) -> None:
    class _Tool:
        def __init__(self, name: str) -> None:
            self.name = name

    class _FakeMCPClient:
        connections: dict[str, Any] = {}

        def __init__(self, connections: dict[str, Any]) -> None:
            _FakeMCPClient.connections = connections

        async def get_tools(self) -> list[_Tool]:
            return [_Tool("read_file"), _Tool("write_file")]

    client_mod = types.ModuleType("langchain_mcp_adapters.client")
    client_mod.MultiServerMCPClient = _FakeMCPClient
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters", types.ModuleType("langchain_mcp_adapters"))
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters.client", client_mod)

    payload = _payload(tmp_path)
    payload["capability_plan"] = {
        "native": {
            "mcp_servers": {"fs": {"transport": "streamable-http", "url": "http://localhost:9/mcp"}},
            "tools": ["read_file"],  # allow-list filters out write_file
        }
    }

    output = await adapter.run_deepagents(payload)

    assert output["failed"] is False
    # Fabric MCP transport is normalized for langchain-mcp-adapters
    assert _FakeMCPClient.connections == {
        "fs": {"transport": "streamable_http", "url": "http://localhost:9/mcp"}
    }
    tool_names = [tool.name for tool in fake_sdks["create_kwargs"]["tools"]]
    assert tool_names == ["read_file"]


async def test_runtime_resume_reuses_thread_id(tmp_path: Path, fake_sdks: dict[str, Any]) -> None:
    # Two invocations of the same runtime_id (a started runtime) resume the same
    # LangGraph thread; a one-shot run gets a fresh runtime_id and never resumes.
    payload = _payload(tmp_path, runtime_id="run-42")

    first = await adapter.run_deepagents(payload)
    assert first["resumed"] is False
    thread_id = first["thread_id"]
    assert thread_id
    # thread id was threaded into the LangGraph config on the invocation
    assert fake_sdks["config"] == {"configurable": {"thread_id": thread_id}}

    second = await adapter.run_deepagents(payload)
    assert second["resumed"] is True
    assert second["thread_id"] == thread_id
