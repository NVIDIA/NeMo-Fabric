# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from claude_agent_sdk import AssistantMessage
from claude_agent_sdk import ClaudeSDKError
from claude_agent_sdk import CLIConnectionError
from claude_agent_sdk import CLIJSONDecodeError
from claude_agent_sdk import CLINotFoundError
from claude_agent_sdk import ProcessError
from claude_agent_sdk import ResultMessage
from claude_agent_sdk import SystemMessage
from claude_agent_sdk import TextBlock
from claude_agent_sdk._errors import MessageParseError

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = ROOT / "adapters" / "claude" / "src" / "nemo_fabric_adapters" / "claude" / "adapter.py"


def load_claude_adapter():
    spec = importlib.util.spec_from_file_location("fabric_claude_adapter", ADAPTER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load_claude_adapter()


def test_claude_descriptor_is_narrow_and_versioned():
    descriptor_path = ROOT / "adapters" / "claude" / "fabric-adapter.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))

    assert descriptor == {
        "contract_version": "fabric.adapter/v1alpha1",
        "adapter_id": "nvidia.fabric.claude",
        "harness": "claude",
        "adapter_kind": "python",
        "runner": {
            "module": "nemo_fabric_adapters.claude.adapter",
            "callable": "run",
        },
        "config": {"accepts": ["models", "tools", "mcp", "skills"]},
    }


@pytest.fixture(name="claude_payload")
def claude_payload_fixture(tmp_path) -> dict[str, Any]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skill_path = tmp_path / "skills" / "review"
    skill_path.mkdir(parents=True)
    (skill_path / "SKILL.md").write_text("# Review\n", encoding="utf-8")
    return {
        "effective_config": {
            "agent_name": "claude-test",
            "config_root": str(tmp_path),
            "config": {
                "harness": {
                    "adapter_id": "nvidia.fabric.claude",
                    "settings": {
                        "system_prompt": "Review carefully.",
                        "allowed_tools": ["Read"],
                        "disallowed_tools": ["WebFetch"],
                        "permission_mode": "dontAsk",
                        "max_turns": 4,
                        "max_budget_usd": 1.5,
                        "setting_sources": [],
                        "timeout_seconds": 30,
                        "env": {"ANTHROPIC_API_KEY": "configured-secret"},
                    },
                },
                "models": {
                    "default": {
                        "provider": "anthropic",
                        "model": "anthropic/claude-test-model",
                        "api_key_env": "ANTHROPIC_API_KEY",
                    }
                },
                "tools": ["Read", "Glob", "Grep"],
            },
        },
        "runtime_context": {
            "runtime_id": "runtime-claude-1",
            "invocation_id": "invocation-1",
            "environment": {"workspace": str(workspace)},
            "artifacts": {"root": str(tmp_path / "artifacts"), "artifacts": []},
        },
        "request": {"request_id": "request-1", "input": "Inspect the patch"},
        "capability_plan": {
            "native": {
                "tools_configured": True,
                "mcp_servers": {
                    "repo": {
                        "transport": "stdio",
                        "url": "repo-mcp --root .",
                        "exposure": "harness_native",
                    },
                    "docs": {
                        "transport": "streamable-http",
                        "url": "https://mcp.example.test",
                        "exposure": "harness_native",
                    },
                },
                "skill_paths": [str(skill_path)],
            }
        },
    }


def test_build_options_maps_normalized_capabilities_and_claude_settings(claude_payload):
    options = adapter.build_options(claude_payload, resume="claude-session")

    assert options.resume == "claude-session"
    assert options.cwd == Path(claude_payload["runtime_context"]["environment"]["workspace"])
    assert options.model == "claude-test-model"
    assert options.system_prompt == "Review carefully."
    assert options.tools == ["Read", "Glob", "Grep", "Skill"]
    assert options.allowed_tools == ["Read"]
    assert options.disallowed_tools == ["WebFetch"]
    assert options.permission_mode == "dontAsk"
    assert options.max_turns == 4
    assert options.max_budget_usd == 1.5
    assert options.setting_sources == []
    assert options.skills == "all"
    assert len(options.plugins) == 1
    plugin_path = Path(options.plugins[0]["path"])
    assert options.plugins[0]["type"] == "local"
    assert json.loads((plugin_path / ".claude-plugin" / "plugin.json").read_text()) == {
        "description": "Skills provided by NeMo Fabric",
        "name": "nemo-fabric-skills",
        "version": "1.0.0",
    }
    assert (plugin_path / "skills" / "review" / "SKILL.md").read_text() == "# Review\n"
    assert options.strict_mcp_config is True
    assert options.mcp_servers == {
        "docs": {"type": "http", "url": "https://mcp.example.test"},
        "repo": {"type": "stdio", "command": "repo-mcp", "args": ["--root", "."]},
    }


def test_build_options_maps_blocked_tools_to_disallowed_tools(claude_payload):
    claude_payload["effective_config"]["config"]["tools"] = {"blocked": ["Bash", "WebFetch"]}

    options = adapter.build_options(claude_payload, resume=None)

    assert options.tools is None
    assert options.disallowed_tools == ["Bash", "WebFetch"]


@pytest.mark.parametrize(
    ("name", "normalized_field"),
    [
        ("model_name", "FabricConfig.models"),
        ("cwd", "FabricConfig.environment.workspace"),
        ("tools", "FabricConfig.tools"),
        ("mcp_servers", "FabricConfig.mcp"),
        ("skills", "FabricConfig.skills"),
    ],
)
def test_build_options_rejects_normalized_capabilities_in_harness_settings(claude_payload, name, normalized_field):
    claude_payload["effective_config"]["config"]["harness"]["settings"][name] = []

    with pytest.raises(adapter.AdapterConfigError, match=normalized_field.replace(".", r"\.")):
        adapter.build_options(claude_payload, resume=None)


def test_build_options_rejects_skill_path_without_skill_manifest(claude_payload):
    skill_path = Path(claude_payload["capability_plan"]["native"]["skill_paths"][0])
    (skill_path / "SKILL.md").unlink()

    with pytest.raises(adapter.AdapterConfigError, match="SKILL.md"):
        adapter.build_options(claude_payload, resume=None)


def test_build_options_rejects_unknown_normalized_tool_preset(claude_payload):
    claude_payload["effective_config"]["config"]["tools"] = {
        "type": "preset",
        "preset": "unknown",
    }

    with pytest.raises(adapter.AdapterConfigError, match="tools preset"):
        adapter.build_options(claude_payload, resume=None)


def test_selected_model_rejects_unsupported_provider(claude_payload):
    model = claude_payload["effective_config"]["config"]["models"]["default"]
    model["provider"] = "nvidia"

    with pytest.raises(adapter.AdapterConfigError, match="provider must be anthropic"):
        adapter.selected_model(claude_payload)


def test_build_options_ignores_tools_not_routed_to_adapter(claude_payload):
    claude_payload["capability_plan"]["native"]["tools_configured"] = False

    options = adapter.build_options(claude_payload, resume=None)

    assert options.tools is None


def test_state_round_trip_is_keyed_by_fabric_runtime(claude_payload):
    runtime_id = adapter.runtime_id(claude_payload)
    adapter.save_claude_session_id(claude_payload, runtime_id, "claude-session")

    assert adapter.load_claude_session_id(claude_payload, runtime_id) == "claude-session"
    state_path = adapter.runtime_state_path(claude_payload, runtime_id)
    assert state_path.parent.name == "runtimes"
    assert runtime_id not in state_path.name


def test_state_loader_rejects_non_object_json(claude_payload):
    runtime_id = adapter.runtime_id(claude_payload)
    state_path = adapter.runtime_state_path(claude_payload, runtime_id)
    state_path.parent.mkdir(parents=True)
    state_path.write_text("[]", encoding="utf-8")

    with pytest.raises(adapter.AdapterStateError, match="runtime state is invalid"):
        adapter.load_claude_session_id(claude_payload, runtime_id)


def test_normalize_result_exposes_session_usage_cost_and_buffered_events(claude_payload):
    messages = [
        SystemMessage(subtype="init", data={"session_id": "claude-session"}),
        AssistantMessage(
            content=[TextBlock(text="done")],
            model="claude-test-model",
            usage={"input_tokens": 10, "output_tokens": 3},
            session_id="claude-session",
        ),
    ]
    result = ResultMessage(
        subtype="success",
        duration_ms=100,
        duration_api_ms=80,
        is_error=False,
        num_turns=1,
        session_id="claude-session",
        total_cost_usd=0.02,
        usage={"input_tokens": 10, "output_tokens": 3},
        result="done",
    )

    output = adapter.normalize_result(claude_payload, messages, result)

    assert output["response"] == "done"
    assert output["session_id"] == "claude-session"
    assert output["usage"] == {"input_tokens": 10, "output_tokens": 3}
    assert output["cost_usd"] == 0.02
    assert output["duration_ms"] == 100
    assert [event["type"] for event in output["events"]] == [
        "SystemMessage",
        "AssistantMessage",
    ]


async def test_run_claude_resumes_and_persists_session(claude_payload, monkeypatch):
    captured = []

    async def query_result(*, prompt, options):
        captured.append((prompt, options.resume, dict(options.env), dict(os.environ)))
        yield AssistantMessage(content=[TextBlock(text="done")], model="claude-test-model")
        yield ResultMessage(
            subtype="success",
            duration_ms=100,
            duration_api_ms=80,
            is_error=False,
            num_turns=1,
            session_id="claude-session",
            total_cost_usd=0.02,
            usage={"input_tokens": 1, "output_tokens": 1},
            result="done",
        )

    mock_query = MagicMock(side_effect=query_result)
    monkeypatch.setattr(adapter, "query", mock_query)
    monkeypatch.setenv("FABRIC_UNRELATED_SECRET", "do-not-forward")

    first = await adapter.run_claude(claude_payload)
    claude_payload["runtime_context"]["invocation_id"] = "invocation-2"
    second = await adapter.run_claude(claude_payload)

    assert first["failed"] is False
    assert second["failed"] is False
    assert [entry[0] for entry in captured] == ["Inspect the patch", "Inspect the patch"]
    assert [entry[1] for entry in captured] == [None, "claude-session"]
    assert all(entry[2]["FABRIC_UNRELATED_SECRET"] == "" for entry in captured)
    assert all(entry[2]["ANTHROPIC_API_KEY"] == "configured-secret" for entry in captured)
    assert all(entry[3]["FABRIC_UNRELATED_SECRET"] == "do-not-forward" for entry in captured)
    assert os.environ["FABRIC_UNRELATED_SECRET"] == "do-not-forward"


def test_build_options_forwards_default_anthropic_api_key(claude_payload, monkeypatch):
    model = claude_payload["effective_config"]["config"]["models"]["default"]
    model.pop("api_key_env")
    settings = claude_payload["effective_config"]["config"]["harness"]["settings"]
    settings.pop("env")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "default-secret")

    options = adapter.build_options(claude_payload, resume=None)

    assert options.env["ANTHROPIC_API_KEY"] == "default-secret"


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (CLINotFoundError("raw path", "/secret/claude"), "claude_cli_not_found"),
        (CLIConnectionError("raw connection"), "claude_connection_failed"),
        (ProcessError("raw process", exit_code=9, stderr="secret"), "claude_process_failed"),
        (CLIJSONDecodeError("secret-json", ValueError("bad")), "claude_invalid_json"),
        (MessageParseError("raw parse", data={"secret": "value"}), "claude_message_parse_failed"),
        (ClaudeSDKError("raw sdk"), "claude_failed"),
    ],
)
def test_sdk_errors_are_structured_without_raw_provider_data(error, code):
    output = adapter.sdk_failure(error)
    serialized = json.dumps(output)

    assert output["error"]["code"] == code
    assert "secret" not in serialized
    assert "raw " not in serialized


def test_error_result_is_normalized_as_failure(claude_payload):
    result = ResultMessage(
        subtype="error_max_turns",
        duration_ms=100,
        duration_api_ms=80,
        is_error=True,
        num_turns=4,
        session_id="claude-session",
        errors=["provider-specific failure"],
    )

    output = adapter.normalize_result(claude_payload, [], result)

    assert output["failed"] is True
    assert output["error"] == {
        "code": "claude_result_failed",
        "message": "Claude returned an error result",
        "retryable": False,
        "metadata": {"subtype": "error_max_turns"},
    }


def test_error_subtype_is_failure_when_sdk_flag_is_false(claude_payload):
    result = ResultMessage(
        subtype="error_max_budget_usd",
        duration_ms=100,
        duration_api_ms=80,
        is_error=False,
        num_turns=4,
        session_id="claude-session",
    )

    output = adapter.normalize_result(claude_payload, [], result)

    assert output["completed"] is False
    assert output["failed"] is True
    assert output["error"]["metadata"] == {"subtype": "error_max_budget_usd"}


def test_run_normalizes_unexpected_exception(claude_payload, monkeypatch):
    monkeypatch.setattr(
        adapter,
        "run_claude",
        MagicMock(side_effect=RuntimeError("secret")),
    )

    output = adapter.run(claude_payload)

    assert output["error"]["code"] == "claude_adapter_internal_error"
    assert "secret" not in json.dumps(output)


def test_main_normalizes_payload_load_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        adapter.common_utils,
        "load_payload",
        MagicMock(side_effect=ValueError("secret payload")),
    )

    with pytest.raises(SystemExit, match="2"):
        adapter.main()

    output = json.loads(capsys.readouterr().out)
    assert output["error"]["code"] == "claude_adapter_internal_error"
    assert "secret" not in json.dumps(output)
