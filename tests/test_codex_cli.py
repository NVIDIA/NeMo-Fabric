# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_fabric import FabricClient, FabricConfig

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = (
    ROOT
    / "adapters"
    / "codex-cli"
    / "src"
    / "nemo_fabric_adapters"
    / "codex_cli"
    / "adapter.py"
)


def load_codex_adapter():
    spec = importlib.util.spec_from_file_location("fabric_codex_adapter", ADAPTER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="codex_payload")
def codex_payload_fixture(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return {
        "effective_config": {
            "agent_name": "codex-test",
            "config_root": str(tmp_path),
            "config": {
                "harness": {
                    "adapter_id": "nvidia.fabric.codex.cli",
                    "settings": {
                        "sandbox": "read-only",
                        "codex_profile": "team",
                        "config_overrides": {
                            "features.web_search": False,
                            "model_reasoning_effort": "high",
                        },
                    },
                },
                "models": {
                    "default": {
                        "provider": "openai",
                        "model": "openai/gpt-5.4",
                    }
                },
                "runtime": {"mode": "oneshot"},
            },
        },
        "runtime_context": {
            "runtime_id": "runtime-1",
            "invocation_id": "invocation-1",
            "request_id": "request-1",
            "environment": {"workspace": str(workspace)},
            "artifacts": {"root": str(tmp_path / "artifacts")},
        },
        "request": {"input": "Inspect the change."},
    }


def codex_jsonl(thread_id, response, *, usage=None):
    events = [
        {"type": "thread.started", "thread_id": thread_id},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {"id": "item-1", "type": "agent_message", "text": response},
        },
        {
            "type": "turn.completed",
            "usage": usage
            or {
                "input_tokens": 10,
                "cached_input_tokens": 2,
                "output_tokens": 3,
            },
        },
    ]
    return "\n".join(json.dumps(event) for event in events) + "\n"


def write_mock_codex(path):
    path.write_text(
        """#!/usr/bin/env python3
import json
import sys

args = sys.argv[1:]
thread_id = args[args.index("resume") + 1] if "resume" in args else "thread-fake"
prompt = sys.stdin.read().strip()
events = [
    {"type": "thread.started", "thread_id": thread_id},
    {"type": "turn.started"},
    {
        "type": "item.completed",
        "item": {
            "id": "item-1",
            "type": "agent_message",
            "text": f"{thread_id}:{prompt}",
        },
    },
    {
        "type": "turn.completed",
        "usage": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1},
    },
]
for event in events:
    print(json.dumps(event))
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def fabric_config(tmp_path, mock_codex, *, mode):
    return FabricConfig.from_mapping(
        {
            "schema_version": "fabric.agent/v1alpha1",
            "metadata": {"name": "codex-runtime-test"},
            "harness": {
                "adapter_id": "nvidia.fabric.codex.cli",
                "resolution": "preinstalled",
                "settings": {
                    "codex_command": str(mock_codex),
                    "sandbox": "read-only",
                    "skip_git_repo_check": True,
                },
            },
            "runtime": {
                "mode": mode,
                "transport": "cli",
                "artifacts": str(tmp_path / "artifacts"),
            },
            "environment": {
                "provider": "local",
                "workspace": str(tmp_path),
                "artifacts": str(tmp_path / "artifacts"),
            },
        }
    )


def test_oneshot_command_uses_fabric_overrides_and_codex_owned_auth(codex_payload):
    adapter = load_codex_adapter()

    command = adapter.build_command(codex_payload)

    assert command[:4] == ["codex", "exec", "--json", "--ephemeral"]
    assert ["--sandbox", "read-only"] == command[4:6]
    assert ["--profile", "team"] == command[6:8]
    assert ["--model", "gpt-5.4"] == command[-3:-1]
    assert 'features.web_search=false' in command
    assert 'model_reasoning_effort="high"' in command
    assert command[-1] == "-"


def test_relative_codex_command_resolves_from_config_root(codex_payload):
    adapter = load_codex_adapter()
    settings = codex_payload["effective_config"]["config"]["harness"]["settings"]
    settings["codex_command"] = "./tools/codex"

    command = adapter.build_command(codex_payload)

    config_root = Path(codex_payload["effective_config"]["config_root"])
    assert command[0] == str(config_root / "tools" / "codex")


def test_reported_command_redacts_secret_config_overrides():
    adapter = load_codex_adapter()

    command = ["codex", "exec", "--config", 'provider.api_key="secret"', "-"]

    assert adapter.redact_command(command)[-2] == "<redacted>"
    assert command[-2] == 'provider.api_key="secret"'


def test_config_override_values_use_tomli_writer():
    adapter = load_codex_adapter()

    assert adapter.toml_value("café") == '"café"'
    assert adapter.toml_value([1, "two"]) == '[\n    1,\n    "two",\n]'
    with pytest.raises(ValueError, match="scalar or array"):
        adapter.toml_value({"nested": True})


def test_session_reuses_codex_thread_across_invocations(codex_payload, monkeypatch):
    adapter = load_codex_adapter()
    codex_payload["effective_config"]["config"]["runtime"]["mode"] = "session"
    codex_payload["runtime_context"]["session_id"] = "review-session"
    mock_run = MagicMock(
        side_effect=[
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=codex_jsonl("thread-123", "first response"),
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=codex_jsonl("thread-123", "second response"),
                stderr="",
            ),
        ]
    )
    monkeypatch.setattr(adapter.subprocess, "run", mock_run)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CODEX_HOME", "/tmp/codex-home")
    monkeypatch.setenv("FABRIC_UNRELATED_SECRET", "do-not-forward")
    codex_payload["effective_config"]["config"]["harness"]["settings"]["env"] = {
        "CODEX_EXPLICIT": "forward-me"
    }

    first = adapter.run_codex(codex_payload)
    codex_payload["runtime_context"]["invocation_id"] = "invocation-2"
    codex_payload["request"]["input"] = "Continue."
    second = adapter.run_codex(codex_payload)

    first_command = mock_run.call_args_list[0].args[0]
    second_command = mock_run.call_args_list[1].args[0]
    assert "--ephemeral" not in first_command
    assert "resume" not in first_command
    assert second_command[-3:] == ["resume", "thread-123", "-"]
    assert first["response"] == "first response"
    assert second["response"] == "second response"
    assert second["session_id"] == "review-session"
    assert second["thread_id"] == "thread-123"
    assert second["usage"]["cached_input_tokens"] == 2
    child_env = mock_run.call_args_list[0].kwargs["env"]
    assert child_env["CODEX_HOME"] == "/tmp/codex-home"
    assert child_env["CODEX_EXPLICIT"] == "forward-me"
    assert "OPENAI_API_KEY" not in child_env
    assert "FABRIC_UNRELATED_SECRET" not in child_env
    assert mock_run.call_args_list[0].kwargs["timeout"] == 1800

    state_path = adapter.session_state_path(codex_payload, "review-session")
    assert json.loads(state_path.read_text(encoding="utf-8")) == {
        "session_id": "review-session",
        "thread_id": "thread-123",
    }


def test_oneshot_does_not_persist_codex_thread(codex_payload, monkeypatch):
    adapter = load_codex_adapter()
    mock_run = MagicMock(
        return_value=subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=codex_jsonl("thread-ephemeral", "done"),
            stderr="",
        )
    )
    monkeypatch.setattr(adapter.subprocess, "run", mock_run)

    output = adapter.run_codex(codex_payload)

    assert output["thread_id"] == "thread-ephemeral"
    assert output["response"] == "done"
    assert "events" not in output
    assert "stdout" not in output
    assert "stderr" not in output
    assert not (Path(output["state_dir"]) / "sessions").exists()


def test_adapter_rejects_structured_input_until_chat_is_supported(codex_payload):
    adapter = load_codex_adapter()
    codex_payload["request"]["input"] = {
        "messages": [{"role": "user", "content": "Inspect the change."}]
    }

    with pytest.raises(ValueError, match="requires text input"):
        adapter.request_to_prompt(codex_payload)


@pytest.mark.parametrize("env", [[], "CODEX_FLAG=1"])
def test_adapter_rejects_non_mapping_env(codex_payload, env):
    adapter = load_codex_adapter()
    settings = codex_payload["effective_config"]["config"]["harness"]["settings"]
    settings["env"] = env

    with pytest.raises(ValueError, match="env must be a mapping"):
        adapter.build_env(codex_payload)


@pytest.mark.parametrize(
    ("error", "message", "returncode"),
    [
        (FileNotFoundError("codex not found"), "codex not found", 127),
        (
            subprocess.TimeoutExpired(["codex"], 1800),
            "timed out after 1800 seconds",
            124,
        ),
    ],
)
def test_process_launch_failures_return_structured_results(
    codex_payload, monkeypatch, error, message, returncode
):
    adapter = load_codex_adapter()
    monkeypatch.setattr(adapter.subprocess, "run", MagicMock(side_effect=error))

    output = adapter.run_codex(codex_payload)

    assert output["failed"] is True
    assert output["returncode"] == returncode
    assert message in output["error"]
    assert "stdout" not in output
    assert "stderr" not in output


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), "30"])
def test_adapter_rejects_invalid_timeout(codex_payload, timeout):
    adapter = load_codex_adapter()
    settings = codex_payload["effective_config"]["config"]["harness"]["settings"]
    settings["timeout_seconds"] = timeout

    with pytest.raises(ValueError, match="timeout_seconds"):
        adapter.run_codex(codex_payload)


def test_adapter_rejects_unsupported_runtime_mode(codex_payload):
    adapter = load_codex_adapter()
    codex_payload["effective_config"]["config"]["runtime"]["mode"] = "service"

    with pytest.raises(ValueError, match="supports only oneshot and session"):
        adapter.run_codex(codex_payload)


def test_session_fails_if_codex_does_not_return_thread_identity(
    codex_payload, monkeypatch
):
    adapter = load_codex_adapter()
    codex_payload["effective_config"]["config"]["runtime"]["mode"] = "session"
    mock_run = MagicMock(
        return_value=subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "untracked"},
                }
            ),
            stderr="",
        )
    )
    monkeypatch.setattr(adapter.subprocess, "run", mock_run)

    output = adapter.run_codex(codex_payload)

    assert output["failed"] is True
    assert "thread identity" in output["error"]


def test_successful_process_without_final_response_is_failed(codex_payload, monkeypatch):
    adapter = load_codex_adapter()
    mock_run = MagicMock(
        return_value=subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="\n".join(
                [
                    json.dumps({"type": "thread.started", "thread_id": "thread-123"}),
                    json.dumps({"type": "turn.completed", "usage": {}}),
                ]
            ),
            stderr="",
        )
    )
    monkeypatch.setattr(adapter.subprocess, "run", mock_run)

    output = adapter.run_codex(codex_payload)

    assert output["failed"] is True
    assert "final agent message" in output["error"]


async def test_fabric_session_invokes_codex_then_resumes(tmp_path):
    mock_codex = tmp_path / "codex"
    write_mock_codex(mock_codex)
    config = fabric_config(tmp_path, mock_codex, mode="session")

    async with await FabricClient().start_session(
        config,
        base_dir=tmp_path,
        session_id="fabric-session",
    ) as session:
        assert session.session_id == "fabric-session"
        first = await session.invoke(input="first")
        second = await session.invoke(input="second")

    assert first.runtime_id == second.runtime_id
    assert first.output["response"] == "thread-fake:first"
    assert second.output["response"] == "thread-fake:second"
    assert first.output["thread_id"] == second.output["thread_id"] == "thread-fake"
    assert "resume" not in first.output["command"]
    assert second.output["command"][-3:] == ["resume", "thread-fake", "-"]


async def test_fabric_oneshot_is_ephemeral_and_uses_cached_codex_auth(
    tmp_path, monkeypatch
):
    mock_codex = tmp_path / "codex"
    write_mock_codex(mock_codex)
    config = fabric_config(tmp_path, mock_codex, mode="oneshot")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async with FabricClient() as client:
        report = await client.doctor(config, base_dir=tmp_path)
        result = await client.run(
            config,
            base_dir=tmp_path,
            input="inspect",
        )

    assert report.status == "pass"
    assert any(
        check.name == "requirement.binary" and "codex_command" in check.message
        for check in report.checks
    )
    assert not any(check.name == "requirement.env" for check in report.checks)
    assert result.output["response"] == "thread-fake:inspect"
    assert "--ephemeral" in result.output["command"]
    assert result.output["session_id"] is None


@pytest.mark.parametrize(
    ("profile", "mode", "session_capability"),
    [
        ("codex_cli", "oneshot", False),
        ("codex_cli_session", "session", True),
    ],
)
def test_codex_profiles_resolve_runtime_mode(profile, mode, session_capability):
    plan = FabricClient().plan(
        ROOT / "examples" / "code-review-agent",
        profiles=[profile],
    )

    assert plan.adapter.adapter_id == "nvidia.fabric.codex.cli"
    assert plan.adapter.harness == "codex"
    assert plan.effective_config.config.runtime.mode == mode
    assert plan.effective_config.config.runtime.input_schema == "text"
    assert plan.capabilities.session is session_capability
    settings = plan.effective_config.config.harness.settings
    assert settings["config_overrides"]["model_reasoning_effort"] == "high"
    unsupported = plan["capability_plan"]["unsupported"]
    assert not unsupported.get("skill_paths")
    assert not unsupported.get("mcp_servers")
