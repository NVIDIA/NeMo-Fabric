# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude adapter boundary and opt-in Claude Agent SDK integration tests."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from _utils.utils import assert_semantic_relay_artifacts
from nemo_fabric import (
    EnvironmentConfig,
    Fabric,
    FabricConfig,
    HarnessConfig,
    MetadataConfig,
    ModelConfig,
    RelayAtifConfig,
    RelayAtofConfig,
    RelayObservabilityConfig,
    RuntimeConfig,
)

ROOT = Path(__file__).resolve().parents[2]
MOCK_CLAUDE_CLI = ROOT / "tests" / "fixtures" / "claude" / "mock-claude-cli.py"
SESSION_ID = "11111111-1111-4111-8111-111111111111"


def write_mock_relay_gateway(path: Path, log_path: Path) -> None:
    path.write_text(
        f"""#!{sys.executable}
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

args = sys.argv[1:]
if args == ["--version"]:
    print("nemo-relay 0.6.0")
    raise SystemExit(0)
Path({str(log_path)!r}).write_text(json.dumps(args), encoding="utf-8")
bind = args[args.index("--bind") + 1]
host, port = bind.rsplit(":", 1)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/healthz" else 404)
        self.end_headers()

    def log_message(self, format, *args):
        pass

HTTPServer((host, int(port)), Handler).serve_forever()
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def fabric_config(
    tmp_path,
    *,
    cli_path=None,
    relay=False,
    nemo_relay_command=None,
):
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
                    "MOCK_CLAUDE_CLI_ENV_LOG": str(tmp_path / "claude-env.jsonl"),
                },
            }
        )
    if nemo_relay_command is not None:
        settings["nemo_relay_command"] = str(nemo_relay_command)
    config = FabricConfig(
        metadata=MetadataConfig(name="claude-runtime-test"),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.claude",
            resolution="preinstalled",
            settings=settings,
        ),
        models={
            "default": ModelConfig(
                provider="anthropic",
                model=os.environ.get(
                    "FABRIC_TEST_CLAUDE_MODEL",
                    "claude-sonnet-4-5",
                ),
            )
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
    if relay:
        config.enable_relay(
            observability=RelayObservabilityConfig(
                atof=RelayAtofConfig(enabled=True),
                atif=RelayAtifConfig(enabled=True),
            )
        )
    return config


async def test_fabric_session_reuses_persistent_claude_runtime(tmp_path):
    config = fabric_config(tmp_path, cli_path=MOCK_CLAUDE_CLI)

    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        first = await runtime.invoke(input="first")
        second = await runtime.invoke(input="second")

    assert first.status == second.status == "succeeded"
    assert first.runtime_id == second.runtime_id
    assert first.output["session_id"] == second.output["session_id"] == SESSION_ID
    assert (
        first.output["response"] == second.output["response"] == "mock Claude response"
    )
    assert first.output["usage"] == {"input_tokens": 1, "output_tokens": 2}
    assert first.output["cost_usd"] == 0.001
    assert [event["type"] for event in first.output["events"]] == ["AssistantMessage"]
    assert first.metadata["adapter_runner"] == "persistent_local_host"
    assert first.metadata["host_pid"] == second.metadata["host_pid"]
    arguments = [
        json.loads(line)
        for line in (tmp_path / "claude-args.jsonl").read_text().splitlines()
    ]
    assert len(arguments) == 1
    assert "--resume" not in arguments[0]
    assert all("--mcp-config" in args for args in arguments)
    assert all("--plugin-dir" in args for args in arguments)
    plugin_paths = [args[args.index("--plugin-dir") + 1] for args in arguments]
    assert len(plugin_paths) == 1
    assert not any(artifact.kind == "stderr" for artifact in second.artifacts.artifacts)


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="mock Relay gateway cannot pass its loopback health check on macOS",
)
async def test_fabric_claude_relay_supervises_gateway_and_injects_plugin(tmp_path):
    mock_relay = tmp_path / "nemo-relay"
    relay_args_path = tmp_path / "relay-args.json"
    write_mock_relay_gateway(mock_relay, relay_args_path)
    config = fabric_config(
        tmp_path,
        cli_path=MOCK_CLAUDE_CLI,
        relay=True,
        nemo_relay_command=mock_relay,
    )

    result = await Fabric().run(config, base_dir=tmp_path, input="inspect")

    assert result.status == "succeeded"
    assert result.telemetry[0].provider == "relay"
    relay_runtime = result.output["relay_runtime"]
    assert relay_runtime["enabled"] is True
    assert relay_runtime["emitter"] == "claude-agent-sdk/nemo-relay"
    assert Path(relay_runtime["gateway_log_path"]).is_file()
    assert Path(relay_runtime["gateway_config_path"]).is_file()
    assert result.output["relay_artifacts"] == []

    relay_args = json.loads(relay_args_path.read_text(encoding="utf-8"))
    assert relay_args[0] == "--config"
    assert relay_args[2] == "--bind"
    assert relay_args[3] in relay_runtime["gateway_url"]
    claude_args = json.loads((tmp_path / "claude-args.jsonl").read_text())
    assert claude_args.count("--plugin-dir") == 2
    plugin_paths = [
        Path(claude_args[index + 1])
        for index, value in enumerate(claude_args)
        if value == "--plugin-dir"
    ]
    relay_plugin_path = next(
        path for path in plugin_paths if path.name == "claude-plugin"
    )
    assert relay_plugin_path.name == "claude-plugin"
    assert not relay_plugin_path.exists()
    claude_env = json.loads((tmp_path / "claude-env.jsonl").read_text())
    assert claude_env == {
        "ANTHROPIC_BASE_URL": relay_runtime["gateway_url"],
        "NEMO_RELAY_GATEWAY_URL": relay_runtime["gateway_url"],
    }


@pytest.mark.skipif(
    not os.environ.get("FABRIC_NEMO_RELAY_COMMAND"),
    reason="set FABRIC_NEMO_RELAY_COMMAND to test an installed NeMo Relay CLI",
)
async def test_fabric_claude_accepts_real_relay_gateway_with_mock_claude(tmp_path):
    config = fabric_config(
        tmp_path,
        cli_path=MOCK_CLAUDE_CLI,
        relay=True,
        nemo_relay_command=os.environ["FABRIC_NEMO_RELAY_COMMAND"],
    )

    result = await Fabric().run(config, base_dir=tmp_path, input="inspect")

    assert result.status == "succeeded"
    assert result.output["relay_runtime"]["enabled"] is True
    gateway_log_path = Path(result.output["relay_runtime"]["gateway_log_path"])
    assert gateway_log_path.is_file()


@pytest.mark.skipif(
    os.environ.get("RUN_FABRIC_CLAUDE_INTEGRATION") != "1",
    reason="set RUN_FABRIC_CLAUDE_INTEGRATION=1 to run Claude Agent SDK integration",
)
async def test_live_claude_single_invocation_and_runtime(tmp_path):
    fabric = Fabric()
    single = await fabric.run(
        fabric_config(tmp_path / "single"),
        base_dir=tmp_path / "single",
        input="Reply only with: FABRIC_CLAUDE_OK",
    )
    assert single.status == "succeeded"

    session_root = tmp_path / "session"
    async with await fabric.start_runtime(
        fabric_config(session_root), base_dir=session_root
    ) as session:
        first = await session.invoke(input="Remember token FABRIC-CONTINUITY-7")
        second = await session.invoke(
            input="Reply only with the token I asked you to remember"
        )
    assert first.status == second.status == "succeeded"
    assert first.output["session_id"] == second.output["session_id"]
    assert "FABRIC-CONTINUITY-7" in second.output["response"]


@pytest.mark.skipif(
    os.environ.get("RUN_FABRIC_CLAUDE_RELAY_INTEGRATION") != "1",
    reason="set RUN_FABRIC_CLAUDE_RELAY_INTEGRATION=1 to run Claude with NeMo Relay",
)
async def test_live_claude_relay_one_shot(tmp_path):
    result = await Fabric().run(
        fabric_config(tmp_path, relay=True),
        base_dir=tmp_path,
        input="Use one simple tool, then reply only with: FABRIC_CLAUDE_RELAY_OK",
    )

    assert result.status == "succeeded"
    assert result.output["relay_runtime"]["enabled"] is True
    assert {artifact["kind"] for artifact in result.output["relay_artifacts"]} == {
        "atof",
        "atif",
    }
    assert_semantic_relay_artifacts(
        result.output,
        "FABRIC_CLAUDE_RELAY_OK",
    )


@pytest.mark.skipif(
    os.environ.get("RUN_FABRIC_CLAUDE_RELAY_INTEGRATION") != "1",
    reason="set RUN_FABRIC_CLAUDE_RELAY_INTEGRATION=1 to run Claude with NeMo Relay",
)
async def test_live_claude_relay_session(tmp_path):
    relay_command = os.environ.get("FABRIC_TEST_NEMO_RELAY_COMMAND")
    config = fabric_config(
        tmp_path,
        relay=True,
        nemo_relay_command=relay_command,
    )

    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        first = await runtime.invoke(input="Remember token FABRIC-CLAUDE-RELAY-7")
        second = await runtime.invoke(
            input="Reply only with the token I asked you to remember"
        )

    results = (first.to_mapping(), second.to_mapping())
    assert first.status == second.status == "succeeded", results
    assert first.output["session_id"] == second.output["session_id"], results
    assert first.metadata["host_pid"] == second.metadata["host_pid"], results
    assert "FABRIC-CLAUDE-RELAY-7" in second.output["response"], results
    for turn in (first, second):
        assert turn.telemetry[0].provider == "relay", turn.to_mapping()
        assert {artifact["kind"] for artifact in turn.output["relay_artifacts"]} == {
            "atof",
            "atif",
        }, turn.to_mapping()
