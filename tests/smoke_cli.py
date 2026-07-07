# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for CLI examples used in the Fabric README/design snippets."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from shutil import copytree

from _utils.utils import assert_relay_disabled_native_observability

ROOT = Path(__file__).resolve().parents[1]
COMMAND = ("cargo", "run", "-q", "-p", "fabric-cli", "--")


def main() -> None:
    example_agent = ROOT / "examples" / "code-review-agent"
    fixture_agent = ROOT / "tests" / "fixtures" / "hermes-shim-agent"
    with tempfile.TemporaryDirectory(prefix="fabric-cli-smoke-") as tmpdir:
        temp_example = Path(tmpdir) / "code-review-agent"
        temp_fixture = Path(tmpdir) / "hermes-shim-agent"
        copytree(example_agent, temp_example)
        copytree(fixture_agent, temp_fixture)

        assert call_text("validate", temp_example).startswith("validated")
        inspected = call_json("inspect", temp_example)
        assert inspected["agent_name"] == "code-review-agent"
        assert inspected.get("profiles", []) == []
        assert inspected["config"]["metadata"]["name"] == "code-review-agent"

        plan = call_json("plan", temp_example, "--profile", "env_local")
        assert plan["agent_name"] == "code-review-agent"
        assert plan["effective_config"]["agent_name"] == "code-review-agent"
        assert plan["effective_config"]["profiles"] == ["env_local"]
        assert plan["adapter_descriptor"]["source"] == "repository"
        assert plan["adapter_descriptor"]["descriptor"]["adapter_id"] == "nvidia.fabric.hermes.sdk"

        agent_schema = call_json("schema", "--name", "agent")
        assert agent_schema["title"] == "FabricConfig"

        schema_dir = Path(tmpdir) / "schemas"
        call_text("schema", "--output-dir", schema_dir)
        assert (schema_dir / "agent.schema.json").is_file()
        assert (schema_dir / "adapter-descriptor.schema.json").is_file()
        assert (schema_dir / "effective-config.schema.json").is_file()
        assert (schema_dir / "adapter-invocation.schema.json").is_file()
        assert (schema_dir / "runtime-context.schema.json").is_file()
        assert (schema_dir / "environment-handle.schema.json").is_file()
        assert (schema_dir / "runtime-handle.schema.json").is_file()
        assert (schema_dir / "invocation-handle.schema.json").is_file()
        assert (schema_dir / "error-info.schema.json").is_file()
        assert (schema_dir / "fabric-event.schema.json").is_file()

        direct_profile = temp_example / "profiles" / "hermes-sdk.yaml"
        direct_plan = call_json("plan", temp_example, "--profile", direct_profile)
        assert direct_plan["profiles"] == [str(direct_profile)]
        assert direct_plan["adapter_descriptor"]["descriptor"]["adapter_id"] == "nvidia.fabric.hermes.sdk"

        profile_plans = [
            (("hermes_sdk",), "nvidia.fabric.hermes.sdk", "python", False),
            (("hermes_cli",), "nvidia.fabric.hermes.cli", "process", False),
            (("hermes_sdk", "relay"), "nvidia.fabric.hermes.sdk", "python", True),
            (("hermes_cli", "relay"), "nvidia.fabric.hermes.cli", "process", True),
        ]
        for profiles, adapter_id, adapter_kind, relay_enabled in profile_plans:
            profile_args = [
                arg for profile in profiles for arg in ("--profile", profile)
            ]
            profile_plan = call_json("plan", temp_example, *profile_args)
            assert profile_plan["profiles"] == list(profiles)
            descriptor = profile_plan["adapter_descriptor"]["descriptor"]
            assert descriptor["adapter_id"] == adapter_id
            assert descriptor["adapter_kind"] == adapter_kind
            assert "mode" not in profile_plan["config"]["runtime"]
            assert profile_plan["capability_plan"]["native"]["skill_paths"]
            assert "github" in profile_plan["capability_plan"]["native"]["mcp_servers"]
            telemetry_plan = profile_plan["telemetry_plan"]
            assert telemetry_plan["relay_enabled"] is relay_enabled
            if relay_enabled:
                assert telemetry_plan["relay_output_dir"]
            else:
                assert not telemetry_plan.get("relay_output_dir")

        multi_plan = call_json(
            "plan",
            temp_fixture,
            "--profile",
            "env_local",
            "--profile",
            "mcp_github",
        )
        assert multi_plan["profiles"] == ["env_local", "mcp_github"]
        assert "profile" not in multi_plan
        assert multi_plan["telemetry_plan"]["relay_enabled"] is True

        doctor = call_json("doctor", temp_fixture, "--profile", "env_local")
        assert doctor["agent_name"] == "hermes-shim-agent"
        assert doctor["checks"]

        hermes = call_json("run", temp_fixture, "--profile", "env_local", "--input", "hello hermes")
        assert hermes["status"] == "succeeded"
        assert hermes["adapter_kind"] == "python"
        assert hermes["output"]["mode"] == "shim"
        assert hermes["output"]["native_skill_paths"]
        assert hermes["output"]["native_mcp_servers"] == ["github"]
        assert hermes["output"]["managed_skill_paths"] == []
        assert hermes["output"]["managed_mcp_servers"] == []
        assert_relay_disabled_native_observability(hermes)

        request = json.dumps(
            {
                "request_id": "cli-structured-request",
                "input": "hello structured hermes",
                "context": {"task": {"source": "smoke"}},
            }
        )
        structured = call_json("run", temp_fixture, "--profile", "env_local", "--request-json", request)
        assert structured["request_id"] == "cli-structured-request"
        assert structured["output"]["received"] == "hello structured hermes"

        chat = run_with_stdin(
            "/help\n/verbose on\nhello chat\n/verbose off\n/clear\n/info\n/exit\n",
            "chat",
            temp_fixture,
            "--profile",
            "env_local",
            "--session-id",
            "cli-session-123",
            "--verbose",
        )
        assert chat.stdout == ""
        assert '"received": "hello chat"' in chat.stderr
        assert '"session_id": "cli-session-123"' in chat.stderr
        assert "NEMO FABRIC" in chat.stderr
        assert "interactive runtime session" in chat.stderr
        assert "agent: hermes-shim-agent" in chat.stderr
        assert "profile: env_local" in chat.stderr
        assert "harness: hermes" in chat.stderr
        assert "adapter: python" in chat.stderr
        assert chat.stderr.count("session_id: cli-session-123 (provided)") >= 2
        assert "you[env_local:cli-session-123]> " in chat.stderr
        assert "you[env_local:cli-session-123]> \nagent> {" in chat.stderr
        assert "agent> {" in chat.stderr
        assert "runtime_id: runtime-" in chat.stderr
        assert "/verbose on|off" in chat.stderr
        assert "/clear" in chat.stderr
        assert "verbose: on" in chat.stderr
        assert "verbose: off" in chat.stderr
        assert "\x1b[2J\x1b[H" in chat.stderr
        assert "\n\n+-- turn 1 metadata" in chat.stderr
        assert "+-- turn 1 metadata" in chat.stderr
        assert "| status: succeeded" in chat.stderr
        assert "| request_id: request-" in chat.stderr
        assert "| invocation_id: invocation-" in chat.stderr
        assert "| artifact_count:" in chat.stderr


def call_text(*args: object) -> str:
    completed = run(*args)
    return completed.stdout.strip()


def call_json(*args: object) -> dict:
    completed = run(*args)
    return json.loads(completed.stdout)


def run(*args: object) -> subprocess.CompletedProcess[str]:
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
    return completed


def run_with_stdin(stdin: str, *args: object) -> subprocess.CompletedProcess[str]:
    completed = run_raw(stdin, *args)
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed: {completed.args}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def run_raw(stdin: str, *args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*COMMAND, *(str(arg) for arg in args)],
        cwd=ROOT,
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )


if __name__ == "__main__":
    main()
