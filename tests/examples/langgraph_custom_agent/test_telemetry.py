# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the example's optional Relay boundary."""

from __future__ import annotations

import asyncio
import builtins
import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from nemo_fabric_adapter_contract.models import RuntimeContext

from examples.langgraph_custom_agent.adapter.telemetry import _load_plugin_config
from examples.langgraph_custom_agent.adapter.telemetry import observe_invocation
from examples.langgraph_custom_agent.agent.graph import build_email_phishing_graph


def _context(
    config_path: Path | None = None,
    *,
    invocation_id: str = "invocation-1",
) -> RuntimeContext:
    telemetry = None
    if config_path is not None:
        telemetry = {
            "relay_enabled": True,
            "config_path": str(config_path),
            "metadata": {"telemetry_providers": ["relay"]},
        }
    return RuntimeContext.from_mapping(
        {
            "runtime_id": "runtime-1",
            "invocation_id": invocation_id,
            "request_id": f"request-{invocation_id}",
            "environment": {
                "environment_id": "environment-1",
                "provider": "local",
                "control_location": "in_env_control",
                "ownership": "caller_owned",
            },
            "artifacts": {},
            "telemetry": telemetry,
        }
    )


def test_relay_disabled_path_does_not_import_relay(tmp_path, monkeypatch):
    imported = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "nemo_relay" or name.startswith("nemo_relay."):
            imported.append(name)
            raise AssertionError("Relay must remain optional")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    async def run() -> None:
        async with observe_invocation(
            _context(),
            base_dir=tmp_path,
            agent_name="email-phishing",
            model_name="test-model",
        ) as telemetry:
            assert telemetry.runnable_config is None
            assert telemetry.artifacts() == []

    asyncio.run(run())
    assert imported == []


def test_partial_relay_config_defaults_to_observability_version_3(tmp_path):
    config_path = tmp_path / "relay-config.json"
    config_path.write_text('{"relay":{"config":{}}}', encoding="utf-8")

    plugin_config = _load_plugin_config(
        _context(config_path),
        base_dir=tmp_path,
        agent_name="email-phishing",
        model_name="test-model",
    )

    assert plugin_config == {
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": True,
                "config": {"version": 3},
            }
        ],
    }


def test_relay_observes_graph_and_model_backed_node(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "relay-config.json"
    config_path.write_text(
        json.dumps(
            {
                "relay": {
                    "config": {
                        "version": 1,
                        "components": [
                            {
                                "kind": "observability",
                                "enabled": True,
                                "config": {
                                    "version": 3,
                                    "atof": {
                                        "enabled": True,
                                        "sinks": [
                                            {
                                                "type": "file",
                                                "output_directory": "relay",
                                                "filename": "events.atof.jsonl",
                                                "mode": "append",
                                            }
                                        ],
                                    },
                                    "atif": {
                                        "enabled": True,
                                        "output_directory": "relay",
                                        "filename_template": (
                                            "trajectory-{session_id}.atif.json"
                                        ),
                                    },
                                },
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    graph = build_email_phishing_graph(
        FakeListChatModel(
            responses=[
                "The first fixed assessment is phishing.",
                "The second fixed assessment is phishing.",
            ]
        ),
        "Explain the fixed assessment.",
    )

    async def run(invocation_id: str) -> list[dict[str, str]]:
        async with observe_invocation(
            _context(config_path, invocation_id=invocation_id),
            base_dir=tmp_path,
            agent_name="email-phishing",
            model_name="test-model",
        ) as telemetry:
            result = await graph.ainvoke(
                {"email": ("Urgent: verify your password at https://example.invalid.")},
                config=telemetry.runnable_config,
            )
        assert result["classification"] == "phishing"
        assert (
            telemetry.plugin_config["components"][0]["config"]["atif"]["model_name"]
            == "test-model"
        )
        assert telemetry.plugin_config["version"] == 1
        assert telemetry.plugin_config["components"][0]["config"]["version"] == 3
        return telemetry.artifacts()

    artifacts = asyncio.run(run("invocation-1"))
    asyncio.run(run("invocation-2"))

    assert {artifact["kind"] for artifact in artifacts} == {"atof", "atif"}
    atof_path = Path(
        next(artifact["path"] for artifact in artifacts if artifact["kind"] == "atof")
    )
    events = [json.loads(line) for line in atof_path.read_text().splitlines()]
    assert any(event.get("name") == "email-phishing-invocation" for event in events)
    assert any(
        event.get("metadata", {}).get("langgraph_node") == "explain_assessment"
        for event in events
    )
    invocation_ids = {
        event.get("metadata", {}).get("nemo_fabric_invocation_id") for event in events
    }
    assert {"invocation-1", "invocation-2"}.issubset(invocation_ids)


def test_relay_rejects_observability_config_version_2(tmp_path):
    config_path = tmp_path / "relay-config.json"
    config_path.write_text(
        json.dumps(
            {
                "relay": {
                    "config": {
                        "version": 1,
                        "components": [
                            {
                                "kind": "observability",
                                "enabled": True,
                                "config": {"version": 2},
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="unsupported NeMo Relay observability config version 2; expected version 3",
    ):
        _load_plugin_config(
            _context(config_path),
            base_dir=tmp_path,
            agent_name="email-phishing",
            model_name="test-model",
        )
