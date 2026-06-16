"""Smoke test for CLI examples used in the Fabric README/design snippets."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from shutil import copytree

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
        assert inspected["kind"] == "fabric_config"

        plan = call_json("plan", temp_example, "--profile", "env_local")
        assert plan["agent_name"] == "code-review-agent"
        assert plan["adapter_descriptor"]["source"] == "repository"
        assert plan["adapter_descriptor"]["descriptor"]["adapter_id"] == "nvidia.fabric.hermes.sdk"

        agent_schema = call_json("schema", "--name", "agent")
        assert agent_schema["title"] == "FabricConfig"

        schema_dir = Path(tmpdir) / "schemas"
        call_text("schema", "--output-dir", schema_dir)
        assert (schema_dir / "agent.schema.json").is_file()
        assert (schema_dir / "adapter-descriptor.schema.json").is_file()

        direct_profile = temp_example / "profiles" / "hermes-sdk.yaml"
        direct_plan = call_json("plan", temp_example, "--profile", direct_profile)
        assert direct_plan["profile"] == str(direct_profile)
        assert direct_plan["adapter_descriptor"]["descriptor"]["adapter_id"] == "nvidia.fabric.hermes.sdk"

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


if __name__ == "__main__":
    main()
