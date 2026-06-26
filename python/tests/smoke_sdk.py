# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the POC Python SDK."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
from shutil import copytree
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

from nemo_fabric import FabricClient

COMMAND = ("cargo", "run", "-q", "-p", "fabric-cli", "--")


async def main() -> None:
    async with FabricClient(
        command=COMMAND,
        cwd=ROOT,
    ) as client:
        await smoke(client)


async def smoke(client: FabricClient) -> None:
    example_agent = ROOT / "examples" / "code-review-agent"
    fixture_agent = ROOT / "tests" / "fixtures" / "hermes-shim-agent"
    process_fixture_agent = ROOT / "tests" / "fixtures" / "hermes-cli-agent"

    assert client.validate(example_agent).startswith("validated")

    plan = client.plan(example_agent, profile="env_local")
    assert plan["agent_name"] == "code-review-agent"
    assert plan["adapter_descriptor"]["source"] == "repository"
    assert plan["adapter_descriptor"]["descriptor"]["adapter_id"] == "nvidia.fabric.hermes.sdk"
    assert plan["environment_plan"]["provider"] == "local"

    report = await client.doctor(fixture_agent, profile="env_local")
    assert report["agent_name"] == "hermes-shim-agent"
    assert report["checks"]

    multi_plan = client.plan(fixture_agent, profile=("env_local", "mcp_github"))
    assert multi_plan["profiles"] == ["env_local", "mcp_github"]
    assert "profile" not in multi_plan
    assert multi_plan["telemetry_plan"]["relay_enabled"] is True

    with tempfile.TemporaryDirectory(prefix="fabric-python-sdk-") as tmpdir:
        temp_agent = Path(tmpdir) / "hermes-shim-agent-sdk"
        temp_cli_agent = Path(tmpdir) / "hermes-shim-agent-cli"
        temp_process_agent = Path(tmpdir) / "hermes-cli-agent-sdk"
        temp_process_cli_agent = Path(tmpdir) / "hermes-cli-agent-cli"
        copytree(fixture_agent, temp_agent)
        copytree(fixture_agent, temp_cli_agent)
        copytree(process_fixture_agent, temp_process_agent)
        copytree(process_fixture_agent, temp_process_cli_agent)

        hermes_result = await client.run(
            temp_agent,
            profile="env_local",
            input_text="hello hermes",
        )
        hermes_cli_result = call_json(
            "run",
            temp_cli_agent,
            "--profile",
            "env_local",
            "--input",
            "hello hermes",
        )
        structured = await client.run(
            temp_agent,
            profile="env_local",
            request={
                "request_id": "sdk-structured-request",
                "input": "hello structured sdk",
                "context": {"task": {"source": "sdk-smoke"}},
            },
        )
        process_result = await client.run(
            temp_process_agent,
            profile="env_local",
            input_text="hello process adapter",
        )
        process_cli_result = call_json(
            "run",
            temp_process_cli_agent,
            "--profile",
            "env_local",
            "--input",
            "hello process adapter",
        )

        assert_sdk_cli_runresult_parity(
            hermes_cli_result,
            hermes_result,
            adapter_kind="python",
            adapter_id="test.fabric.hermes_shim",
            adapter_runner="python",
            mode="shim",
        )
        assert_sdk_cli_runresult_parity(
            process_cli_result,
            process_result,
            adapter_kind="process",
            adapter_id="nvidia.fabric.hermes.cli",
            adapter_runner="process",
            mode="hermes_cli_oneshot",
        )
        assert_relay_disabled_native_observability(hermes_result)
        assert_relay_disabled_native_observability(process_result)

    assert hermes_result["status"] == "succeeded"
    assert hermes_result["adapter_kind"] == "python"
    assert hermes_result["output"]["harness"] == "hermes"
    assert hermes_result["output"]["received"] == "hello hermes"
    assert hermes_result["output"]["native_skill_paths"]
    assert hermes_result["output"]["native_mcp_servers"] == ["github"]
    assert hermes_result["output"]["managed_skill_paths"] == []
    assert hermes_result["output"]["managed_mcp_servers"] == []

    assert structured["request_id"] == "sdk-structured-request"
    assert structured["output"]["received"] == "hello structured sdk"

    process_response = json.loads(process_result["output"]["response"])
    assert process_response["fake_hermes"] is True
    assert process_response["prompt"] == "hello process adapter"


def assert_sdk_cli_runresult_parity(
    cli_result: dict,
    sdk_result: dict,
    *,
    adapter_kind: str,
    adapter_id: str,
    adapter_runner: str,
    mode: str,
) -> None:
    comparable_fields = [
        "agent_name",
        "profile",
        "harness_type",
        "adapter_kind",
        "adapter_id",
        "status",
    ]
    for field in comparable_fields:
        assert cli_result[field] == sdk_result[field], field

    assert cli_result.get("error") == sdk_result.get("error")
    assert cli_result["adapter_kind"] == adapter_kind
    assert cli_result["adapter_id"] == adapter_id
    assert cli_result["metadata"]["adapter_runner"] == adapter_runner
    assert sdk_result["metadata"]["adapter_runner"] == adapter_runner
    assert cli_result["output"]["harness"] == "hermes"
    assert sdk_result["output"]["harness"] == "hermes"
    assert cli_result["output"]["mode"] == mode
    assert sdk_result["output"]["mode"] == mode

    for result in (cli_result, sdk_result):
        assert result["status"] == "succeeded"
        assert result["runtime_id"].startswith("runtime-")
        assert result["invocation_id"].startswith("invocation-")
        assert result["request_id"].startswith("request-")
        assert isinstance(result["artifacts"]["artifacts"], list)
        assert isinstance(result["events"], list)
        assert result["events"], "RunResult events should not be empty"


def assert_relay_disabled_native_observability(result: dict) -> None:
    artifact_by_name = {
        artifact["name"]: artifact
        for artifact in result["artifacts"]["artifacts"]
    }
    assert "stdout" in artifact_by_name
    assert "relay_config" not in artifact_by_name
    assert not any(name.startswith("relay_") for name in artifact_by_name)

    stdout_path = Path(artifact_by_name["stdout"]["path"])
    assert stdout_path.is_file()
    assert stdout_path.read_text(encoding="utf-8").strip()

    event_kinds = {event["kind"] for event in result["events"]}
    assert {"runtime_start", "invocation_start", "invocation_end"} <= event_kinds

    telemetry = result["telemetry"]
    assert telemetry is not None
    assert telemetry["relay_enabled"] is False


def call_json(*args: object) -> dict:
    completed = subprocess.run(
        [*COMMAND, *(str(arg) for arg in args)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed: {completed.args}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


if __name__ == "__main__":
    asyncio.run(main())
