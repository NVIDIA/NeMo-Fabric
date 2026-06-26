# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Hermes SDK adapter's Fabric runtime mapping."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from _utils.utils import update_base_url
from nemo_fabric import FabricClient

ROOT = Path(__file__).resolve().parents[1]
HERMES_SDK_SRC = ROOT / "adapters" / "hermes-sdk" / "src"
if str(HERMES_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(HERMES_SDK_SRC))

from nemo_fabric_adapters.hermes_sdk import adapter  # noqa: E402


async def test_runtime_id_drives_hermes_session_id_and_hermes_db_history(
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
    monkeypatch.setenv("TEST_API_KEY", "secret")

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


class TestHermesSdkE2E:
    """
    E2E Hermes SDK tests, which communicate with a mock API server not requiring an API key.
    """

    @pytest.fixture(autouse=True)
    async def run_hermes_sdk_relay(
        self,
        nemo_relay: ModuleType,
        mock_nvidia_api_key: str,
        code_review_agent_dir: Path,
        api_server: str,
    ):
        assert nemo_relay is not None
        pytest.importorskip("run_agent", reason="hermes extra is required")

        # Use the current Python executable for Hermes to ensure that the subprocess uses the same environment.
        os.environ["HERMES_PYTHON"] = sys.executable
        
        self.code_review_agent_dir = code_review_agent_dir
        self.api_server = api_server
        update_base_url(
            code_review_agent_dir / "profiles" / "hermes-relay.yaml",
            api_server,
        )

        async with FabricClient() as client:
            self.result = await client.run(
                code_review_agent_dir,
                profile="hermes_relay",
                input_text="Reply with exactly: relay ok",
            )

        self.output = self.result["output"]
        self.artifacts = self.result["artifacts"]
        self.artifact_root = Path(self.artifacts["root"]).resolve()
        self.relay_artifacts = self.output["relay_artifacts"]

    async def test_artifacts(self):
        assert self.result["status"] == "succeeded"
        assert self.result["adapter_kind"] == "python"
        assert self.result["metadata"]["adapter_runner"] == "python"
        assert self.result["telemetry"]["relay_enabled"] is True
        assert self.result["telemetry"]["metadata"]["relay_mode"] == "sdk"

        output = self.output
        assert output["adapter"] == "python"
        assert output["harness"] == "hermes"
        assert output["mode"] == "hermes_sdk"
        assert output["base_url"] == f"{self.api_server}/v1"
        assert output["failed"] is False
        assert output["error"] is None
        assert "echo user_count=" in output["response"]
        assert output["relay_runtime"]["enabled"] is True
        assert output["relay_runtime"]["mode"] == "sdk"
        assert output["relay_runtime"]["emitter"] == "hermes.observability/nemo_relay"

        hermes_home = Path(output["hermes_home"]).resolve()
        hermes_config_path = Path(output["hermes_config_path"]).resolve()
        assert hermes_home.is_dir()
        assert hermes_home.is_relative_to(self.code_review_agent_dir)
        assert hermes_config_path.is_file()
        assert hermes_config_path.is_relative_to(self.code_review_agent_dir)

        hermes_config = yaml.safe_load(hermes_config_path.read_text(encoding="utf-8"))
        assert hermes_config["model"]["provider"] == "nvidia"
        assert hermes_config["model"]["default"] == "nvidia/nemotron-3-nano-30b-a3b"
        assert hermes_config["model"]["base_url"] == f"{self.api_server}/v1"
        assert hermes_config["plugins"]["enabled"] == ["observability/nemo_relay"]
        assert output["hermes_native_config"]["plugins"] == ["observability/nemo_relay"]

        expected_artifact_root = (
            self.code_review_agent_dir / "artifacts" / "hermes-relay"
        ).resolve()
        assert self.artifact_root == expected_artifact_root
        assert self.artifact_root.is_dir()

        artifact_by_name = {
            artifact["name"]: artifact
            for artifact in self.artifacts["artifacts"]
        }
        assert "relay_config" in artifact_by_name
        assert "stdout" in artifact_by_name

        relay_config_path = Path(artifact_by_name["relay_config"]["path"]).resolve()
        assert relay_config_path.is_file()
        assert relay_config_path.is_relative_to(self.artifact_root)
        assert not relay_config_path.with_name("relay-plugins.toml").exists()
        relay_config = json.loads(relay_config_path.read_text(encoding="utf-8"))
        assert relay_config["schema_version"] == "fabric.relay/v1alpha1"
        assert relay_config["relay"]["enabled"] is True
        assert relay_config["fabric"]["profile"] == "hermes_relay"

    async def test_atof_artifacts(self):
        kinds = {artifact["kind"] for artifact in self.relay_artifacts}
        assert "atof" in kinds

        atof_paths = [
            Path(artifact["path"]).resolve()
            for artifact in self.relay_artifacts
            if artifact["kind"] == "atof"
        ]
        assert atof_paths
        assert all(path.exists() for path in atof_paths)
        assert all(path.is_relative_to(self.artifact_root) for path in atof_paths)

        atof_records = [
            json.loads(line)
            for line in atof_paths[0].read_text().strip().splitlines()
        ]
        expected_atof_fields = {
            "atof_version",
            "attributes",
            "category",
            "data",
            "kind",
            "metadata",
            "name",
            "parent_uuid",
            "scope_category",
            "timestamp",
            "uuid",
        }
        actual_atof_fields = set().union(*(record.keys() for record in atof_records))
        assert len(atof_records) >= 3
        assert actual_atof_fields.issuperset(expected_atof_fields)
        assert any(record["name"] == "hermes.session.end" for record in atof_records)
        assert any(record.get("scope_category") == "end" for record in atof_records)
        assert all(
            record["metadata"]["model"] == "nvidia/nemotron-3-nano-30b-a3b"
            and record["metadata"]["platform"] == "fabric"
            for record in atof_records
            if record.get("metadata", {}).get("model")
        )

    async def test_atif_artifacts(self):
        kinds = {artifact["kind"] for artifact in self.relay_artifacts}
        assert "atif" in kinds

        atif_paths = [
            Path(artifact["path"]).resolve()
            for artifact in self.relay_artifacts
            if artifact["kind"] == "atif"
        ]
        assert atif_paths
        assert all(path.exists() for path in atif_paths)
        assert all(path.is_relative_to(self.artifact_root) for path in atif_paths)

        trajectory = json.loads(atif_paths[0].read_text())
        assert trajectory["agent"]["name"] in {"code-review-agent", "Hermes Agent"}
        steps = trajectory["steps"]
        assert steps

        first_step = steps[0]
        assert first_step["message"] == "hermes.turn.start"
        assert first_step["extra"]["event_payload"]["is_first_turn"] is True

        last_step = steps[-1]
        assert last_step["message"] == "hermes.session.end"
        assert last_step["extra"]["invocation"]["framework"] == "nemo_relay"
        assert last_step["extra"]["invocation"]["status"] == "completed"
