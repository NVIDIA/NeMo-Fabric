# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in live check that tier-0 harnesses start with unavailable MCP servers.

Each runtime gets three MCP servers: a reachable stdio time server, an HTTP URL
with nothing listening, and a stdio command that does not exist. The runtime
must start, the agent must call the time tool, and every result must report
the two unavailable servers under ``output.mcp.unavailable_servers``.

The model endpoint must serve Chat Completions, Responses (Codex) and Anthropic
Messages (Claude Code) under one OpenAI-compatible ``/v1`` base URL::

    RUN_FABRIC_LIVE_MCP_E2E=1 \\
    FABRIC_LIVE_MODEL_BASE_URL=http://model-host:8000/v1 \\
    FABRIC_LIVE_MODEL=nvidia/nemotron-3-ultra-550b-a55b \\
        pytest tests/e2e/test_live_partial_mcp.py

Harness prerequisites:

- Hermes runs only under the Hermes environment: run pytest with
  ``.venv-hermes/bin/python`` and ``ADAPTER_PYTHON`` set to it. The time server
  needs ``mcp_server_time``, which that environment lacks, so set
  ``FABRIC_LIVE_MCP_PYTHON`` to an interpreter that has it (for example
  ``.venv/bin/python``).
- OpenClaw needs ``FABRIC_OPENCLAW_TEST_COMMAND`` pointing at the pinned
  ``openclaw`` executable and a Node version that OpenClaw accepts.
- Claude Code reads ``ANTHROPIC_*`` and ``CLAUDE_*`` variables from the
  environment; unset them so the configured endpoint is used.
- Relay variants for Codex need an https or loopback model URL.
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_FABRIC_LIVE_MCP_E2E") != "1",
    reason="set RUN_FABRIC_LIVE_MCP_E2E=1 to run live partial MCP checks",
)

BASE_URL = os.environ.get("FABRIC_LIVE_MODEL_BASE_URL", "")
MODEL = os.environ.get("FABRIC_LIVE_MODEL", "")
MCP_PYTHON = os.environ.get("FABRIC_LIVE_MCP_PYTHON", sys.executable)
TIME_SERVER = Path(__file__).resolve().parents[1] / "fixtures" / "mcp_time_logged.py"
KEY_ENV = "FABRIC_LIVE_MODEL_API_KEY"
TIME_PROMPT = (
    "Use the get_current_time tool from the time MCP server to get the current "
    "time in America/Los_Angeles. Find it with tool search if needed and do not "
    "run shell commands. Reply with just that time."
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _model_host() -> str:
    return urlparse(BASE_URL).hostname or ""


def _loopback_or_https() -> bool:
    parsed = urlparse(BASE_URL)
    if parsed.scheme == "https":
        return True
    try:
        return ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        return parsed.hostname == "localhost"


def _model(provider: str):
    from nemo_fabric import ModelConfig

    return ModelConfig(
        provider=provider, model=MODEL, api_key_env=KEY_ENV, base_url=BASE_URL
    )


def _deepagents():
    from examples.code_review_agent import deepagents_config

    config = deepagents_config()
    config.models = {"default": _model("openai-compatible")}
    return config, "chat"


def _nooa():
    from examples.code_review_agent import nooa_config

    config = nooa_config()
    config.skills = None
    config.models = {"default": _model("openai")}
    return config, "text"


def _codex():
    from examples.code_review_agent import codex_config

    config = codex_config()
    config.harness.settings = {"sandbox": "read-only"}
    config.models = {"default": _model("nemotron")}
    return config, "text"


def _claude():
    from examples.code_review_agent import claude_config
    from nemo_fabric import ToolsConfig

    config = claude_config()
    if urlparse(BASE_URL).scheme == "http" and not _loopback_or_https():
        config.harness.settings["insecure_http_model_hosts"] = [_model_host()]
    # Under ``dontAsk`` only enabled tools run; enable the time server's tools.
    config.tools = ToolsConfig(
        enabled=["mcp__time__get_current_time", "mcp__time__convert_time"]
    )
    config.models = {"default": _model("nvidia")}
    return config, "chat"


def _hermes():
    pytest.importorskip("hermes_cli", reason="run under the Hermes environment")
    from examples.code_review_agent import hermes_config
    from nemo_fabric import ToolsConfig

    config = hermes_config()
    config.models = {"default": _model("custom")}
    config.tools = ToolsConfig(enabled=["mcp-time", "mcp-offline", "mcp-missing"])
    return config, "chat"


def _openclaw():
    command = os.environ.get("FABRIC_OPENCLAW_TEST_COMMAND")
    if not command:
        pytest.skip("set FABRIC_OPENCLAW_TEST_COMMAND to the openclaw executable")
    from examples.code_review_agent import openclaw_config

    config = openclaw_config()
    config.harness.settings = {
        "openclaw_command": command,
        "startup_timeout_seconds": 75,
    }
    config.tools = None
    config.skills = None
    config.models = {"default": _model("local-vllm")}
    return config, "text"


HARNESSES: dict[str, tuple[Callable[[], tuple[object, str]], bool]] = {
    # name: (config builder, supports Relay)
    "deepagents": (_deepagents, True),
    "nooa": (_nooa, True),
    "codex": (_codex, True),
    "claude": (_claude, True),
    "hermes": (_hermes, True),
    "openclaw": (_openclaw, False),
}
CASES = [
    pytest.param(name, relay, id=f"{name}-{'relay' if relay else 'native'}")
    for name, (_, supports_relay) in HARNESSES.items()
    for relay in ((False, True) if supports_relay else (False,))
]


@pytest.mark.parametrize(("harness", "relay"), CASES)
async def test_harness_starts_with_unavailable_mcp_servers(
    harness, relay, tmp_path, monkeypatch
):
    """Start, call the reachable tool, and report the unavailable servers."""
    if not BASE_URL or not MODEL:
        pytest.skip("set FABRIC_LIVE_MODEL_BASE_URL and FABRIC_LIVE_MODEL")
    if relay and harness == "codex" and not _loopback_or_https():
        pytest.skip("Codex Relay needs an https or loopback model URL")
    probe = subprocess.run(
        [MCP_PYTHON, "-c", "import mcp_server_time"], capture_output=True
    )
    if probe.returncode != 0:
        pytest.skip(f"{MCP_PYTHON} cannot import mcp_server_time")

    from examples.code_review_agent import with_relay
    from nemo_fabric import (
        EnvironmentConfig,
        Fabric,
        InstructionConfig,
        InstructionsConfig,
        RuntimeConfig,
    )

    monkeypatch.setenv(KEY_ENV, os.environ.get(KEY_ENV, "unused"))
    config, input_schema = HARNESSES[harness][0]()
    if relay:
        config = with_relay(config)
    config.instructions = InstructionsConfig(
        system=InstructionConfig(content="You are a helpful assistant. Use tools.")
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config.environment = EnvironmentConfig(
        provider="local", workspace=workspace, artifacts=tmp_path / "artifacts"
    )
    config.environment.env[KEY_ENV] = os.environ[KEY_ENV]
    config.runtime = RuntimeConfig(
        input_schema=input_schema,
        output_schema="message",
        artifacts=tmp_path / "artifacts",
        max_turns=8 if harness == "hermes" else None,
    )
    time_log = tmp_path / "time-calls.jsonl"
    config.add_mcp_server(
        "time",
        transport="stdio",
        url=MCP_PYTHON,
        args=[str(TIME_SERVER)],
        env={"TIME_MCP_LOG": str(time_log)},
    )
    config.add_mcp_server(
        "offline",
        transport="streamable-http",
        url=f"http://127.0.0.1:{_free_port()}/mcp",
    )
    config.add_mcp_server(
        "missing", transport="stdio", url="nemo-fabric-missing-mcp-server-command"
    )

    async with await Fabric().start_runtime(
        config, base_dir=tmp_path, streaming=relay
    ) as runtime:
        first = await runtime.invoke(input=TIME_PROMPT)
        second = await runtime.invoke(input="Which tools do you have? One line.")

    results = (first.to_mapping(), second.to_mapping())
    assert first["status"] == second["status"] == "succeeded", results
    lines = time_log.read_text().splitlines() if time_log.exists() else []
    calls = [json.loads(line) for line in lines]
    assert {"tool": "get_current_time", "timezone": "America/Los_Angeles"} in calls
    for turn in (first, second):
        unavailable = turn["output"]["mcp"]["unavailable_servers"]
        assert {server["name"] for server in unavailable} == {"offline", "missing"}
        assert {server["reason"] for server in unavailable} <= {"failed", "timed_out"}
    if relay:
        assert first.telemetry[0].provider == "relay", results[0]
