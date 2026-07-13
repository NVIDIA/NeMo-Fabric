# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude adapter boundary and opt-in Claude Agent SDK integration tests."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from nemo_fabric import EnvironmentConfig
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import HarnessConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import ModelConfig
from nemo_fabric import RuntimeConfig

ROOT = Path(__file__).resolve().parents[2]
MOCK_CLAUDE_CLI = ROOT / "tests" / "fixtures" / "claude" / "mock-claude-cli.py"
SESSION_ID = "11111111-1111-4111-8111-111111111111"


def fabric_config(tmp_path, *, cli_path=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    settings = {
        "python": sys.executable,
        "setting_sources": [],
        "permission_mode": "dontAsk",
    }
    if cli_path is not None:
        settings.update(
            {
                "cli_path": str(cli_path),
                "env": {
                    "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK": "1",
                    "MOCK_CLAUDE_CLI_LOG": str(tmp_path / "claude-args.jsonl"),
                },
            }
        )
    config = FabricConfig(
        metadata=MetadataConfig(name="claude-runtime-test"),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.claude",
            resolution="preinstalled",
            settings=settings,
        ),
        models={
            "default": ModelConfig(provider="anthropic", model="claude-test-model")
        },
        runtime=RuntimeConfig(artifacts=tmp_path / "artifacts"),
        environment=EnvironmentConfig(
            provider="local",
            workspace=tmp_path,
            artifacts=tmp_path / "artifacts",
        ),
    )
    if cli_path is not None:
        skill_path = tmp_path / "skills" / "review"
        skill_path.mkdir(parents=True)
        (skill_path / "SKILL.md").write_text("# Review\n", encoding="utf-8")
        config.add_skill_path(skill_path)
        config.add_mcp_server(
            "docs",
            transport="streamable-http",
            url="https://mcp.example.test",
        )
    return config


async def test_fabric_session_launches_fresh_processes_and_resumes(tmp_path):
    config = fabric_config(tmp_path, cli_path=MOCK_CLAUDE_CLI)

    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        first = await runtime.invoke(input="first")
        second = await runtime.invoke(input="second")

    assert first.status == second.status == "succeeded"
    assert first.runtime_id == second.runtime_id
    assert first.output["session_id"] == second.output["session_id"] == SESSION_ID
    assert first.output["response"] == second.output["response"] == "mock Claude response"
    assert first.output["usage"] == {"input_tokens": 1, "output_tokens": 2}
    assert first.output["cost_usd"] == 0.001
    assert [event["type"] for event in first.output["events"]] == ["AssistantMessage"]
    arguments = [
        json.loads(line)
        for line in (tmp_path / "claude-args.jsonl").read_text().splitlines()
    ]
    assert len(arguments) == 2
    assert "--resume" not in arguments[0]
    assert arguments[1][arguments[1].index("--resume") + 1] == SESSION_ID
    assert all("--mcp-config" in args for args in arguments)
    assert all("--plugin-dir" in args for args in arguments)
    plugin_paths = [args[args.index("--plugin-dir") + 1] for args in arguments]
    assert plugin_paths[0] == plugin_paths[1]
    assert not any(artifact.kind == "stderr" for artifact in second.artifacts.artifacts)


@pytest.mark.skipif(
    os.environ.get("RUN_FABRIC_CLAUDE_INTEGRATION") != "1",
    reason="set RUN_FABRIC_CLAUDE_INTEGRATION=1 to run Claude Agent SDK integration",
)
async def test_live_claude_one_shot_and_session(tmp_path):
    fabric = Fabric()
    one_shot = await fabric.run(
        fabric_config(tmp_path / "oneshot"),
        base_dir=tmp_path / "oneshot",
        input="Reply only with: FABRIC_CLAUDE_OK",
    )
    assert one_shot.status == "succeeded"

    session_root = tmp_path / "session"
    async with await fabric.start_runtime(
        fabric_config(session_root), base_dir=session_root
    ) as session:
        first = await session.invoke(input="Remember token FABRIC-CONTINUITY-7")
        second = await session.invoke(input="Reply only with the token I asked you to remember")
    assert first.status == second.status == "succeeded"
    assert first.output["session_id"] == second.output["session_id"]
    assert "FABRIC-CONTINUITY-7" in second.output["response"]
