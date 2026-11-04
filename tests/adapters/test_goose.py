# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle and configuration tests for the Goose adapter."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.goose import adapter


@pytest.fixture(name="mock_goose")
def mock_goose_fixture(repo_root: Path, tmp_path: Path) -> Path:
    os.environ["FAKE_GOOSE_CAPTURE"] = str(tmp_path / "capture.jsonl")
    os.environ["FAKE_GOOSE_CHILD_PID"] = str(tmp_path / "child.pid")
    return repo_root / "tests/_utils/mock_goose.py"


def _captures(tmp_path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (tmp_path / "capture.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def _context(workspace: Path, deadline_millis: int | None = None) -> RuntimeContext:
    payload: dict[str, Any] = {
        "runtime_id": "goose-runtime",
        "invocation_id": "goose-invocation",
        "request_id": "goose-request",
        "environment": {
            "environment_id": "goose-environment",
            "provider": "local",
            "control_location": "in_env_control",
            "workspace": str(workspace),
            "env": {},
            "ownership": "caller_owned",
        },
        "artifacts": {},
    }
    if deadline_millis is not None:
        payload["deadline_millis"] = deadline_millis
    return RuntimeContext.from_mapping(payload)


def _config(command: Path, **overrides: Any) -> AgentConfig:
    mapping: dict[str, Any] = {
        "harness": {"settings": {"goose_command": str(command)}},
        "models": {
            "default": {
                "provider": "openai",
                "model": "served-model",
                "base_url": "http://models.example.test/v1",
            }
        },
    }
    mapping.update(overrides)
    return AgentConfig.from_mapping(mapping)


async def _started(config: AgentConfig, tmp_path: Path) -> adapter.GooseRuntime:
    runtime = adapter.GooseRuntime()
    await runtime.start(
        {
            "config": config,
            "runtime_context": _context(tmp_path).to_mapping(),
            "base_dir": str(tmp_path),
        }
    )
    return runtime


async def _invoke(
    runtime: adapter.GooseRuntime, tmp_path: Path, text: str, **context: Any
):
    return await runtime.invoke(
        AgentRunRequest(input=text), _context(tmp_path, **context)
    )


def test_goose_descriptor_and_module_entrypoint(repo_root: Path):
    descriptor = json.loads(
        (repo_root / "adapters/python/goose/goose.fabric-adapter.json").read_text(
            encoding="utf-8"
        )
    )
    result = subprocess.run(
        [sys.executable, "-m", "nemo_fabric_adapters.goose.adapter"],
        input="",
        text=True,
        capture_output=True,
        check=False,
    )

    assert descriptor["adapter_id"] == "nvidia.fabric.goose"
    assert descriptor["runner"] == {"module": "nemo_fabric_adapters.goose.adapter"}
    assert descriptor["requirements"] == {}
    assert descriptor["model_schema"]["else"] == {"required": ["base_url"]}
    assert set(descriptor["settings_schema"]["properties"]) == {"goose_command"}
    assert descriptor["config"] == {
        "accepts": [
            "models",
            "models.base_url",
            "models.temperature",
            "instructions.system",
            "runtime.max_turns",
            "mcp",
        ],
        "system_instruction_modes": ["replace", "append"],
    }
    assert descriptor["capabilities"] == {
        "service": False,
        "streaming": False,
        "updates": False,
        "cancellation": False,
    }
    assert "telemetry" not in descriptor
    assert result.returncode == 0, result.stderr


async def test_goose_maps_configuration_into_one_isolated_run(
    mock_goose: Path, tmp_path: Path
):
    os.environ.update(
        {
            "FABRIC_TEST_MODEL_KEY": "secret-value",
            "MCP_TOKEN": "token-value",
            "OPENAI_HOST": "https://ambient.example.test",
        }
    )
    config = _config(
        mock_goose,
        models={
            "default": {
                "provider": "openai",
                "model": "served-model",
                "base_url": "http://models.example.test/v1",
                "api_key_env": "FABRIC_TEST_MODEL_KEY",
                "temperature": 0.2,
            }
        },
        instructions={"system": {"content": "-Be brief.", "mode": "append"}},
        runtime={"max_turns": 7},
        mcp={
            "servers": {
                "local": {
                    "transport": "stdio",
                    "url": "python",
                    "args": ["server.py"],
                    "env": {"MODE": "test"},
                },
                "docs": {
                    "transport": "streamable-http",
                    "url": "https://mcp.example.test/mcp",
                    "custom_headers": {"Authorization": "Bearer ${MCP_TOKEN}"},
                },
            }
        },
    )
    runtime = await _started(config, tmp_path)
    try:
        result = await _invoke(runtime, tmp_path, "Hello.")
    finally:
        await runtime.stop()

    assert result.status == "succeeded"
    assert result.output == {"response": "turn 1: Hello."}
    [capture] = _captures(tmp_path)
    assert capture["task"] == "Hello."
    assert capture["cwd"] == str(tmp_path)
    assert capture["args"] == [
        "run",
        "--name",
        "nemo-fabric",
        "--output-format",
        "stream-json",
        "--max-turns",
        "7",
        "--system=-Be brief.",
        "--instructions",
        "-",
    ]
    env = capture["env"]
    root = env.pop("GOOSE_PATH_ROOT")
    assert env == {
        "GOOSE_PROVIDER": "openai",
        "GOOSE_MODEL": "served-model",
        "GOOSE_MODE": "auto",
        "GOOSE_DISABLE_KEYRING": "1",
        "GOOSE_TELEMETRY_OFF": "1",
        "GOOSE_TEMPERATURE": "0.2",
        "OPENAI_API_KEY": "secret-value",
        "OPENAI_BASE_URL": "http://models.example.test/v1",
    }
    assert not Path(root).exists()
    assert capture["config"] == {
        "extensions": {
            "local": {
                "enabled": True,
                "name": "local",
                "description": "",
                "timeout": 300,
                "type": "stdio",
                "cmd": "python",
                "args": ["server.py"],
                "envs": {"MODE": "test"},
            },
            "docs": {
                "enabled": True,
                "name": "docs",
                "description": "",
                "timeout": 300,
                "type": "streamable_http",
                "uri": "https://mcp.example.test/mcp",
                "headers": {"Authorization": "Bearer token-value"},
            },
        }
    }


async def test_goose_replace_instruction_uses_a_system_prompt_file(
    mock_goose: Path, tmp_path: Path
):
    os.environ["GOOSE_SYSTEM_PROMPT_FILE_PATH"] = "/ambient/prompt.md"
    runtime = await _started(
        _config(mock_goose, instructions={"system": {"content": "Only answer."}}),
        tmp_path,
    )
    prompt = Path(runtime._env["GOOSE_SYSTEM_PROMPT_FILE_PATH"])
    try:
        assert prompt.read_text(encoding="utf-8") == "Only answer."
        result = await _invoke(runtime, tmp_path, "Hello.")
    finally:
        await runtime.stop()

    assert result.status == "succeeded"
    [capture] = _captures(tmp_path)
    assert not any(arg.startswith("--system") for arg in capture["args"])
    assert capture["env"]["GOOSE_SYSTEM_PROMPT_FILE_PATH"] == str(prompt)
    assert not prompt.exists()


async def test_goose_continues_one_session_and_reports_turn_usage(
    mock_goose: Path, tmp_path: Path
):
    runtime = await _started(_config(mock_goose), tmp_path)
    try:
        first = await _invoke(runtime, tmp_path, "first")
        second = await _invoke(runtime, tmp_path, "second")
    finally:
        await runtime.stop()

    assert first.output == {"response": "turn 1: first"}
    assert second.output == {"response": "turn 2: second"}
    first_run, second_run = _captures(tmp_path)
    assert "--resume" not in first_run["args"]
    assert "--resume" in second_run["args"]
    assert second_run["history"] == ["first"]
    for result in (first, second):
        assert result.usage is not None
        assert result.usage.to_mapping() == {
            "input_tokens": 10,
            "cached_input_tokens": 3,
            "input_tokens_include_cache": True,
            "output_tokens": 2,
            "total_tokens": 12,
        }


async def test_goose_runtimes_do_not_share_sessions(mock_goose: Path, tmp_path: Path):
    first = await _started(_config(mock_goose), tmp_path)
    second = await _started(_config(mock_goose), tmp_path)
    try:
        await _invoke(first, tmp_path, "first runtime")
        result = await _invoke(second, tmp_path, "second runtime")
    finally:
        await first.stop()
        await second.stop()

    assert result.output == {"response": "turn 1: second runtime"}
    first_run, second_run = _captures(tmp_path)
    assert first_run["env"]["GOOSE_PATH_ROOT"] != second_run["env"]["GOOSE_PATH_ROOT"]
    assert "--resume" not in second_run["args"]


@pytest.mark.parametrize(
    ("task", "code"),
    [
        pytest.param("MAX_TURNS", "goose_max_turns_reached", id="max-turns"),
        pytest.param("MODEL_ERROR", "goose_model_error", id="model-error"),
        pytest.param("SILENT", "goose_no_response", id="no-response"),
        pytest.param("EXTENSION_FAILURE", "goose_extension_failed", id="extension"),
    ],
)
async def test_goose_unfinished_runs_fail_with_usage(
    mock_goose: Path, tmp_path: Path, task: str, code: str
):
    runtime = await _started(_config(mock_goose), tmp_path)
    try:
        result = await _invoke(runtime, tmp_path, task)
    finally:
        await runtime.stop()

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.code == code
    assert result.usage is not None
    assert result.usage.total_tokens == 12


async def test_goose_exit_failure_reports_the_error_event(
    mock_goose: Path, tmp_path: Path
):
    runtime = await _started(_config(mock_goose), tmp_path)
    try:
        failed = await _invoke(runtime, tmp_path, "EXIT_FAILURE")
        after = await _invoke(runtime, tmp_path, "after")
    finally:
        await runtime.stop()

    assert failed.status == "failed"
    assert failed.error is not None
    assert failed.error.code == "goose_exit_failure"
    assert failed.error.message == (
        "Goose exited with code 1: provider rejected the request"
    )
    assert failed.usage is None
    # The failed run's tokens are unknown, so the next turn cannot use it as a baseline.
    assert after.status == "succeeded"
    assert after.usage is None


def _gone(pid: int) -> bool:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


@pytest.mark.skipif(sys.platform != "linux", reason="Finds tool commands by session")
async def test_goose_timeout_stops_goose_and_its_tool_commands(
    mock_goose: Path, tmp_path: Path
):
    runtime = await _started(_config(mock_goose), tmp_path)
    try:
        started = time.monotonic()
        result = await _invoke(
            runtime,
            tmp_path,
            "SLEEP",
            deadline_millis=int((time.time() + 2) * 1000),
        )
        elapsed = time.monotonic() - started
        tool_stopped = _gone(int((tmp_path / "child.pid").read_text(encoding="utf-8")))
        after = await _invoke(runtime, tmp_path, "after")
    finally:
        await runtime.stop()

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.code == "goose_timeout"
    assert elapsed < 10
    assert tool_stopped
    assert after.status == "succeeded"
    assert after.usage is None


@pytest.mark.skipif(sys.platform != "linux", reason="Finds tool commands by session")
async def test_goose_stop_ends_tool_commands_left_running(
    mock_goose: Path, tmp_path: Path
):
    runtime = await _started(_config(mock_goose), tmp_path)
    try:
        result = await _invoke(runtime, tmp_path, "BACKGROUND")
        child = int((tmp_path / "child.pid").read_text(encoding="utf-8"))
        os.kill(child, 0)
    finally:
        await runtime.stop()

    assert result.status == "succeeded"
    assert _gone(child)


async def test_goose_start_strips_ambient_model_overrides(
    mock_goose: Path, tmp_path: Path
):
    os.environ.update(
        {
            "OPENAI_BASE_URL": "https://ambient.example.test/v1",
            "OPENAI_BASE_PATH": "v2/chat",
            "GOOSE_TEMPERATURE": "1.5",
        }
    )
    config = _config(
        mock_goose, models={"default": {"provider": "openai", "model": "gpt-test"}}
    )
    runtime = await _started(config, tmp_path)
    try:
        await _invoke(runtime, tmp_path, "Hello.")
    finally:
        await runtime.stop()

    [capture] = _captures(tmp_path)
    assert "OPENAI_BASE_URL" not in capture["env"]
    assert "OPENAI_BASE_PATH" not in capture["env"]
    assert "GOOSE_TEMPERATURE" not in capture["env"]


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        pytest.param(
            {"harness": {"settings": {"goose_command": "missing-goose"}}},
            "goose_command_not_found",
            id="command",
        ),
        pytest.param(
            {"models": {"default": {"provider": "nvidia", "model": "served-model"}}},
            "goose_base_url_required",
            id="base-url",
        ),
        pytest.param(
            {
                "models": {
                    "default": {
                        "provider": "openai",
                        "model": "gpt-test",
                        "api_key_env": "FABRIC_TEST_MISSING_KEY",
                    }
                }
            },
            "goose_credential_missing",
            id="credential",
        ),
        pytest.param(
            {"mcp": {"servers": {"old": {"transport": "sse", "url": "http://x/sse"}}}},
            "goose_mcp_configuration_unsupported",
            id="sse",
        ),
        pytest.param(
            {
                "mcp": {
                    "servers": {
                        "docs": {
                            "transport": "stdio",
                            "url": "python",
                            "allowed_tools": ["search"],
                        }
                    }
                }
            },
            "goose_mcp_configuration_unsupported",
            id="tool-policy",
        ),
    ],
)
async def test_goose_start_rejects_unsupported_configuration(
    mock_goose: Path, tmp_path: Path, overrides: dict[str, Any], code: str
):
    mapping: dict[str, Any] = {
        "harness": {"settings": {"goose_command": str(mock_goose)}},
        "models": {"default": {"provider": "openai", "model": "gpt-test"}},
    }
    mapping.update(overrides)
    config = AgentConfig.from_mapping(mapping)
    runtime = adapter.GooseRuntime()

    with pytest.raises(lifecycle.LifecycleError) as caught:
        await runtime.start(
            {
                "config": config,
                "runtime_context": _context(tmp_path).to_mapping(),
                "base_dir": str(tmp_path),
            }
        )

    assert caught.value.code == code
    assert runtime._root is None


async def test_goose_stop_is_idempotent(mock_goose: Path, tmp_path: Path):
    runtime = await _started(_config(mock_goose), tmp_path)
    root = Path(runtime._env["GOOSE_PATH_ROOT"])

    await runtime.stop()
    await runtime.stop()

    assert not root.exists()
    with pytest.raises(lifecycle.LifecycleError, match="not started"):
        await _invoke(runtime, tmp_path, "Hello.")


async def test_goose_plan_doctor_and_run_without_credentials(
    mock_goose: Path, tmp_path: Path
):
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "goose-test"},
            "harness": {
                "adapter_id": "nvidia.fabric.goose",
                "resolution": "preinstalled",
                "settings": {"goose_command": str(mock_goose)},
            },
            "models": {
                "default": {
                    "provider": "openai",
                    "model": "served-model",
                    "base_url": "http://models.example.test/v1",
                }
            },
            "instructions": {"system": {"content": "Be brief.", "mode": "append"}},
            "runtime": {"max_turns": 3},
            "environment": {"provider": "local", "workspace": "."},
        }
    )
    fabric = Fabric()

    plan = fabric.plan(config, base_dir=tmp_path)
    report = await fabric.doctor(config, base_dir=tmp_path)
    result = await fabric.run(config, input="Hello.", base_dir=tmp_path)

    assert plan.adapter.adapter_id == "nvidia.fabric.goose"
    assert report.status == "pass"
    assert result.status == "succeeded"
    assert result.output == {"response": "turn 1: Hello."}
