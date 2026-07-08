# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Hermes SDK adapter's Fabric runtime mapping."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
HERMES_SDK_SRC = ROOT / "adapters" / "hermes-sdk" / "src"
if str(HERMES_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(HERMES_SDK_SRC))

from nemo_fabric_adapters.hermes_sdk import adapter  # noqa: E402


async def test_hermes_sdk_rejects_native_telemetry():
    payload = {
        "effective_config": {
            "config": {"telemetry": {"enabled": True, "provider": "native"}}
        }
    }

    with pytest.raises(ValueError, match="only relay telemetry is supported for Hermes"):
        await adapter.run_hermes_sdk(payload)


async def test_fabric_runtime_id_drives_hermes_session_id_and_db_history(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    db_history = [{"role": "user", "content": "from hermes db"}]

    class FakeSessionDB:
        def get_session(self, session_id: str) -> dict[str, str] | None:
            captured.setdefault("db_get_session", []).append(session_id)
            if session_id == "runtime-resolved-456":
                return {"id": session_id}
            return None

        def resolve_resume_session_id(self, session_id: str) -> str:
            captured["db_resolve_session"] = session_id
            return "runtime-resolved-456"

        def get_messages_as_conversation(self, session_id: str) -> list[dict[str, str]]:
            captured["db_get_messages"] = session_id
            return list(db_history)

    class FakeAIAgent:
        def __init__(
            self,
            *,
            base_url: str | None = None,
            api_key: str | None = None,
            provider: str | None = None,
            model: str = "",
            max_iterations: int = 1,
            enabled_toolsets: list[str] | None = None,
            quiet_mode: bool = True,
            skip_context_files: bool = True,
            skip_memory: bool = True,
            save_trajectories: bool = False,
            max_tokens: int = 512,
            temperature: float = 0.0,
            reasoning_config: dict[str, Any] | None = None,
            insert_reasoning: bool = False,
            platform: str | None = None,
            session_id: str | None = None,
            session_db: Any | None = None,
        ) -> None:
            captured["init"] = {
                "session_id": session_id,
                "session_db": session_db,
                "platform": platform,
                "model": model,
                "provider": provider,
            }
            self.session_id = session_id or "generated-session"
            self.model = model
            self.platform = platform

        def run_conversation(
            self,
            user_message: str,
            *,
            system_message: str | None = None,
            conversation_history: list[dict[str, str]] | None = None,
            sync_honcho: bool = False,
            dont_review: bool = True,
        ) -> dict[str, Any]:
            captured["conversation"] = {
                "user_message": user_message,
                "system_message": system_message,
                "conversation_history": conversation_history,
                "sync_honcho": sync_honcho,
                "dont_review": dont_review,
            }
            return {
                "response": "ok",
                "completed": True,
                "failed": False,
                "messages": [{"role": "assistant", "content": "ok"}],
            }

    hermes_cli = ModuleType("hermes_cli")
    hermes_config = ModuleType("hermes_cli.config")
    hermes_config.load_config = lambda: {}  # type: ignore[attr-defined]
    hermes_plugins = ModuleType("hermes_cli.plugins")
    hermes_plugins.discover_plugins = lambda force=False: None  # type: ignore[attr-defined]
    hermes_plugins.invoke_hook = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    hermes_state = ModuleType("hermes_state")
    hermes_state.SessionDB = FakeSessionDB  # type: ignore[attr-defined]
    run_agent = ModuleType("run_agent")
    run_agent.AIAgent = FakeAIAgent  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", hermes_config)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", hermes_plugins)
    monkeypatch.setitem(sys.modules, "hermes_state", hermes_state)
    monkeypatch.setitem(sys.modules, "run_agent", run_agent)
    os.environ["TEST_API_KEY"] = "secret"

    payload = {
        "effective_config": {
            "agent_name": "demo",
            "config_root": str(tmp_path),
            "config": {
                "harness": {
                    "settings": {
                        "hermes_home": "./hermes-home",
                        "enabled_toolsets": [],
                        "system_prompt": "system",
                    }
                },
                "models": {
                    "default": {
                        "provider": "test-provider",
                        "model": "test-model",
                        "api_key_env": "TEST_API_KEY",
                    }
                },
            },
        },
            "runtime_context": {
                "runtime_id": "runtime-fabric-123",
                "environment": {"workspace": str(tmp_path)},
        },
        "request": {
            "input": "hello",
            "context": {"history": [{"role": "user", "content": "stale"}]},
        },
        "capability_plan": {"native": {}},
    }

    output = await adapter.run_hermes_sdk(payload)

    assert captured["db_resolve_session"] == "runtime-fabric-123"
    assert captured["db_get_session"] == ["runtime-resolved-456"]
    assert captured["db_get_messages"] == "runtime-resolved-456"
    assert captured["init"]["session_id"] == "runtime-fabric-123"
    assert isinstance(captured["init"]["session_db"], FakeSessionDB)
    assert captured["init"]["platform"] == "fabric"
    assert captured["conversation"]["conversation_history"] == db_history
    assert "session_id" not in output
    assert Path(output["hermes_home"]) == (
        tmp_path / "hermes-home" / "runtimes" / "runtime-fabric-123"
    )
