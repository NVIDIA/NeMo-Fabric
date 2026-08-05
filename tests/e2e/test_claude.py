# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude adapter boundary and opt-in Claude Agent SDK integration tests."""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import pytest
import requests
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
    RelayAtofFileSinkConfig,
    RelayObservabilityConfig,
    RuntimeConfig,
    ToolsConfig,
)

ROOT = Path(__file__).resolve().parents[2]
MOCK_CLAUDE_CLI = ROOT / "tests" / "fixtures" / "claude" / "mock-claude-cli.py"
SESSION_ID = "11111111-1111-4111-8111-111111111111"


@pytest.fixture(name="use_current_python_for_adapter_discovery", autouse=True)
def use_current_python_for_adapter_discovery_fixture(restore_environ) -> None:
    restore_environ["ADAPTER_PYTHON"] = sys.executable


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
        "setting_sources": [],
        "permission_mode": "dontAsk",
    }
    environment_env = (
        {
            "FABRIC_TEST_CLAUDE_CLI_PATH": str(cli_path),
            "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK": "1",
            "MOCK_CLAUDE_CLI_LOG": str(tmp_path / "claude-args.jsonl"),
            "MOCK_CLAUDE_CLI_ENV_LOG": str(tmp_path / "claude-env.jsonl"),
        }
        if cli_path is not None
        else {}
    )
    if nemo_relay_command is not None:
        environment_env["FABRIC_TEST_NEMO_RELAY_COMMAND"] = str(nemo_relay_command)
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
            env=environment_env,
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
                atof=RelayAtofConfig(
                    enabled=True,
                    sinks=[RelayAtofFileSinkConfig()],
                ),
                atif=RelayAtifConfig(enabled=True),
            )
        )
    return config


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="the mock Claude CLI is not yet a Windows executable",
)
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


@pytest.mark.parametrize("enabled", [True, False])
async def test_mcp_stdio_transport(api_server, tmp_path, enabled):
    tool_name = "mcp__mcp_server_time__get_current_time"
    scenario_response = requests.post(
        f"{api_server}/_scenario",
        json={
            "tool_call": {
                "name": tool_name,
                "arguments": {"timezone": "America/Los_Angeles"},
            }
        },
        timeout=5,
    )
    scenario_response.raise_for_status()

    config = fabric_config(tmp_path)
    config.models["default"].provider = "fabric-test"
    config.models["default"].model = "fabric-echo"
    config.models["default"].api_key_env = "FABRIC_TEST_API_KEY"
    config.models["default"].base_url = f"{api_server}/v1"
    config.environment.env["FABRIC_TEST_API_KEY"] = "test"
    config.tools = ToolsConfig(enabled=[tool_name])
    config.add_mcp_server(
        "mcp_server_time",
        transport="stdio",
        url=sys.executable,
        args=["-m", "mcp_server_time"],
        env={"MCP_TIME_TEST": "enabled"},
    )

    if not enabled:
        config.remove_mcp_server("mcp_server_time")

    result = await Fabric().run(
        config,
        base_dir=tmp_path,
        input="Use the time MCP tool to get the current time in America/Los_Angeles.",
    )

    assert result["status"] == "succeeded", result.to_mapping()
    tool_uses = [
        block
        for event in result["output"]["events"]
        if event["type"] == "AssistantMessage"
        for block in event["message"]["content"]
        if block.get("name") == tool_name
    ]
    tool_results = [
        block
        for event in result["output"]["events"]
        if event["type"] == "UserMessage"
        and isinstance(event["message"]["content"], list)
        for block in event["message"]["content"]
        if "tool_use_id" in block
    ]

    if not enabled:
        assert not tool_uses
        assert not tool_results
    else:
        assert tool_uses
        assert tool_results
        assert not tool_results[0]["is_error"]
        assert "America/Los_Angeles" in str(tool_results[0]["content"])


@pytest.mark.skipif(
    sys.platform in {"darwin", "win32"},
    reason="the mock Relay gateway is not supported on macOS or Windows",
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

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        async with await Fabric().start_runtime(
            config,
            base_dir=tmp_path,
            streaming=True,
        ) as runtime:
            first_stream = runtime.invoke_stream(
                input="Remember token FABRIC-CLAUDE-RELAY-7"
            )
            first_records = [record async for record in first_stream]
            first = await first_stream.result()
            second_stream = runtime.invoke_stream(
                input="Reply only with the token I asked you to remember"
            )
            second_records = [record async for record in second_stream]
            second = await second_stream.result()

    results = (first.to_mapping(), second.to_mapping())
    assert first_records
    assert second_records
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
