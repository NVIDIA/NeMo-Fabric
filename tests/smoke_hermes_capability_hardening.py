# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke coverage for Hermes capability hardening."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from shutil import copytree
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "src"))

from nemo_fabric import FabricClient  # noqa: E402

COMMAND = ("cargo", "run", "-q", "-p", "fabric-cli", "--")


async def main() -> None:
    assert_hermes_config_variation_matrix()

    cases = [
        {
            "name": "python_adapter",
            "fixture": ROOT / "tests" / "fixtures" / "hermes-shim-agent",
            "profile": "env_local",
            "adapter_kind": "python",
            "adapter_id": "test.fabric.hermes_shim",
            "adapter_runner": "python",
            "mode": "shim",
        },
        {
            "name": "process_adapter",
            "fixture": ROOT / "tests" / "fixtures" / "hermes-cli-agent",
            "profile": "env_local",
            "adapter_kind": "process",
            "adapter_id": "nvidia.fabric.hermes.cli",
            "adapter_runner": "process",
            "mode": "hermes_cli_oneshot",
        },
    ]

    with tempfile.TemporaryDirectory(prefix="hermes-capability-hardening-") as tmpdir:
        for case in cases:
            cli_result = run_case(case, tmpdir, "cli")
            sdk_result = await run_sdk_case(case, tmpdir)

            assert_normalized_runresult_parity(cli_result, sdk_result, case)
            assert_relay_disabled_native_observability(cli_result, case)
            assert_relay_disabled_native_observability(sdk_result, case)
            assert_hermes_capability_config_visible(cli_result, case)
            assert_hermes_capability_config_visible(sdk_result, case)

    print("smoke_hermes_capability_hardening ok")


def assert_hermes_config_variation_matrix() -> None:
    matrix = [
        {
            "name": "hermes_sdk",
            "agent": ROOT / "examples" / "code-review-agent",
            "profiles": ["hermes_sdk"],
            "adapter_id": "nvidia.fabric.hermes.sdk",
            "adapter_kind": "python",
            "runtime_mode": "oneshot",
            "relay_enabled": False,
        },
        {
            "name": "hermes_cli",
            "agent": ROOT / "examples" / "code-review-agent",
            "profiles": ["hermes_cli"],
            "adapter_id": "nvidia.fabric.hermes.cli",
            "adapter_kind": "process",
            "runtime_mode": "oneshot",
            "relay_enabled": False,
        },
        {
            "name": "hermes_relay",
            "agent": ROOT / "examples" / "code-review-agent",
            "profiles": ["hermes_relay"],
            "adapter_id": "nvidia.fabric.hermes.sdk",
            "adapter_kind": "python",
            "runtime_mode": "oneshot",
            "relay_enabled": True,
        },
        {
            "name": "hermes_cli_relay",
            "agent": ROOT / "examples" / "code-review-agent",
            "profiles": ["hermes_cli_relay"],
            "adapter_id": "nvidia.fabric.hermes.cli",
            "adapter_kind": "process",
            "runtime_mode": "oneshot",
            "relay_enabled": True,
        },
        {
            "name": "stacked_mcp_profile",
            "agent": ROOT / "tests" / "fixtures" / "hermes-shim-agent",
            "profiles": ["env_local", "mcp_github"],
            "adapter_id": "test.fabric.hermes_shim",
            "adapter_kind": "python",
            "runtime_mode": "session",
            "relay_enabled": True,
        },
    ]

    for case in matrix:
        args: list[object] = ["plan", case["agent"]]
        for profile in case["profiles"]:
            args.extend(["--profile", profile])
        plan = call_json(*args)

        assert plan["profiles"] == case["profiles"], case["name"]
        descriptor = plan["adapter_descriptor"]["descriptor"]
        assert descriptor["adapter_id"] == case["adapter_id"], case["name"]
        assert descriptor["adapter_kind"] == case["adapter_kind"], case["name"]

        config = plan["config"]
        assert config["runtime"]["mode"] == case["runtime_mode"], case["name"]
        assert config["models"]["default"]["model"], case["name"]

        environment_plan = plan["environment_plan"]
        assert environment_plan["workspace"], case["name"]
        assert environment_plan["artifacts"], case["name"]

        native_capabilities = plan["capability_plan"]["native"]
        assert native_capabilities["skill_paths"], case["name"]
        assert "github" in native_capabilities["mcp_servers"], case["name"]

        telemetry_plan = plan["telemetry_plan"]
        assert telemetry_plan["relay_enabled"] is case["relay_enabled"], case["name"]
        if case["relay_enabled"]:
            assert telemetry_plan["relay_output_dir"], case["name"]


def run_case(case: dict[str, Any], tmpdir: str, surface: str) -> dict[str, Any]:
    temp_agent = Path(tmpdir) / f"{case['name']}-{surface}"
    copytree(case["fixture"], temp_agent)
    return call_json(
        "run",
        temp_agent,
        "--profile",
        case["profile"],
        "--input",
        f"hello {case['name']}",
    )


async def run_sdk_case(case: dict[str, Any], tmpdir: str) -> dict[str, Any]:
    temp_agent = Path(tmpdir) / f"{case['name']}-sdk"
    copytree(case["fixture"], temp_agent)
    async with FabricClient(command=COMMAND, cwd=ROOT) as client:
        return await client.run(
            temp_agent,
            profile=case["profile"],
            input_text=f"hello {case['name']}",
        )


def assert_normalized_runresult_parity(
    cli_result: dict[str, Any],
    sdk_result: dict[str, Any],
    case: dict[str, Any],
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

    assert cli_result["adapter_kind"] == case["adapter_kind"]
    assert cli_result["adapter_id"] == case["adapter_id"]
    assert cli_result["metadata"]["adapter_runner"] == case["adapter_runner"]
    assert sdk_result["metadata"]["adapter_runner"] == case["adapter_runner"]
    assert cli_result["output"]["harness"] == "hermes"
    assert sdk_result["output"]["harness"] == "hermes"
    assert cli_result["output"]["mode"] == case["mode"]
    assert sdk_result["output"]["mode"] == case["mode"]

    for result in (cli_result, sdk_result):
        assert result["status"] == "succeeded"
        assert result["runtime_id"].startswith("runtime-")
        assert result["invocation_id"].startswith("invocation-")
        assert result["request_id"].startswith("request-")
        assert isinstance(result["artifacts"]["artifacts"], list)
        assert isinstance(result["events"], list)
        assert result["events"], "RunResult events should not be empty"


def assert_relay_disabled_native_observability(
    result: dict[str, Any],
    case: dict[str, Any],
) -> None:
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

    if case["adapter_kind"] == "process":
        output = result["output"]
        assert output["returncode"] == 0
        assert output["stderr"] == ""
        assert Path(output["fabric_invocation"]).is_file()


def assert_hermes_capability_config_visible(
    result: dict[str, Any],
    case: dict[str, Any],
) -> None:
    output = result["output"]
    if case["adapter_kind"] == "python":
        assert output["native_mcp_servers"] == ["github"]
        assert output["native_skill_paths"]
        assert output["managed_mcp_servers"] == []
        assert output["managed_skill_paths"] == []
        return

    hermes_config_path = Path(output["hermes_config_path"])
    assert hermes_config_path.is_file()
    native_config = output["hermes_native_config"]
    assert native_config["mcp_servers"] == ["github"]
    assert native_config["skill_dirs"]
    response = json.loads(output["response"])
    assert response["fake_hermes"] is True
    assert response["prompt"] == "hello process_adapter"


def call_json(*args: object) -> dict[str, Any]:
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
