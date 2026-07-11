# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: typed (in-memory) config is first-class, with no agent directory.

The unified SDK methods accept a typed config object and resolve, diagnose, and
run it without an on-disk agent package:

* ``plan`` / ``doctor`` resolve a maintained (repository) adapter
  with ``base_dir=None`` -- zero filesystem layout, no ``agent.yaml``.
* ``run`` drives a real core runtime run using only a local adapter directory
  (still no agent package).
This complements ``test_native_sdk.py``, which exercises ``plan`` with a
``base_dir`` pointed at an agent package.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from shutil import copytree

import yaml
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import FabricProfileConfig
from nemo_fabric import RunRequest
from nemo_fabric import RunResult

ROOT = Path(__file__).resolve().parents[2]
COMMAND = ("cargo", "run", "-q", "-p", "fabric-cli", "--")
# The test adapter (needs only python3, no secrets), shipped as a fixture.
SHIM_AGENT = ROOT / "tests" / "fixtures" / "hermes-shim-agent"
SHIM_ADAPTERS = ROOT / "tests" / "fixtures" / "hermes-shim-agent" / "adapters"


def _repository_adapter_config() -> FabricConfig:
    """Config referencing a maintained adapter resolvable without any package."""

    return FabricConfig.from_mapping(
        {
            "schema_version": "fabric.agent/v1alpha1",
            "metadata": {"name": "typed-only-agent"},
            "harness": {
                "adapter_id": "nvidia.fabric.hermes.sdk",
                "resolution": "preinstalled",
            },
            "models": {
                "default": {
                    "provider": "nvidia",
                    "model": "test-model",
                    "temperature": 0.0,
                }
            },
            "runtime": {
                "input_schema": "chat",
                "output_schema": "message",
                "artifacts": "./artifacts",
            },
            "environment": {
                "provider": "local",
                "workspace": "./ws",
                "artifacts": "./artifacts/local",
            },
            "telemetry": None,
        }
    )


def _shim_adapter_config() -> FabricConfig:
    """Config referencing the test adapter (runs without secrets)."""

    config = _repository_adapter_config().to_mapping()
    config["metadata"] = {"name": "typed-only-run"}
    config["harness"] = {
        "adapter_id": "test.fabric.hermes_shim",
        "resolution": "preinstalled",
        "settings": {"workspace": "./ws"},
    }
    config["models"] = {
        "default": {"provider": "test", "model": "test-model", "temperature": 0.0}
    }
    return FabricConfig.from_mapping(config)


async def resolves_and_diagnoses_without_a_directory(client: Fabric) -> None:
    """plan / doctor resolve a maintained adapter with no package."""

    config = _repository_adapter_config()

    # base_dir=None: the literal "no path at all" call must still resolve.
    plan_no_path = client.plan(config)
    assert plan_no_path["agent_name"] == "typed-only-agent"

    # Point base_dir at an EMPTY directory (not an agent package): resolution can
    # then only succeed via the baked-in repository adapter dir, so the source is
    # deterministic regardless of the process CWD.
    with tempfile.TemporaryDirectory(prefix="typed-no-dir-") as empty:
        plan = client.plan(config, base_dir=empty)
        report = await client.doctor(config, base_dir=empty)

    descriptor = plan["adapter_descriptor"]
    assert descriptor["descriptor"]["adapter_id"] == "nvidia.fabric.hermes.sdk"
    # "repository" (not "local") proves it resolved without any on-disk package.
    assert descriptor["source"] == "repository", descriptor["source"]

    assert report["agent_name"] == "typed-only-agent"
    assert report.checks, "doctor produced no checks"
    assert report["status"] in {"pass", "warn", "fail"}, report["status"]


async def runs_without_an_agent_package(client: Fabric) -> None:
    """run drives a core run with only an adapter dir (no agent.yaml)."""

    config = _shim_adapter_config()
    with tempfile.TemporaryDirectory(prefix="typed-run-") as tmpdir:
        base = Path(tmpdir) / "scratch"
        # Only the adapter lives here -- this is deliberately NOT an agent
        # package (no agent.yaml / profiles / repos / skills).
        copytree(SHIM_ADAPTERS, base / "adapters")
        (base / "ws").mkdir()
        assert not (base / "agent.yaml").exists()
        result = await client.run(
            config,
            base_dir=base,
            request=RunRequest(
                input="hello typed",
                request_id="typed-request-1",
                context={"job_id": "job-1"},
                overrides={"max_iterations": 1},
            ),
        )

    assert isinstance(result, RunResult)
    assert result["status"] == "succeeded", result.get("status")
    assert result.request_id == "typed-request-1"
    assert result["adapter_kind"] == "python"
    assert result["metadata"]["adapter_runner"] == "python"
    assert result["output"]["received"] == "hello typed"


def sdk_and_cli_profile_stacks_match(client: Fabric) -> None:
    """The same config/profile stack plans identically through CLI and SDK."""

    config = FabricConfig.from_mapping(_load_yaml(SHIM_AGENT / "agent.yaml"))
    profiles = [
        FabricProfileConfig.model_validate(
            _load_yaml(SHIM_AGENT / "profiles" / "env-local.yaml")
        ),
        FabricProfileConfig.model_validate(
            _load_yaml(SHIM_AGENT / "profiles" / "mcp-github.yaml")
        ),
    ]

    sdk_plan = client.plan(config, profiles=profiles, base_dir=SHIM_AGENT)
    cli_plan = _cli_plan(SHIM_AGENT, "env_local", "mcp_github")

    assert (
        sdk_plan.profiles == tuple(cli_plan["profiles"]) == ("env_local", "mcp_github")
    )
    assert "profile" not in sdk_plan
    assert "profile" not in cli_plan
    sdk_mapping = sdk_plan.to_mapping()
    assert sdk_mapping["config"] == cli_plan["config"]
    assert (
        sdk_mapping["effective_config"]["config"]
        == cli_plan["effective_config"]["config"]
    )
    assert sdk_mapping["adapter_descriptor"] == cli_plan["adapter_descriptor"]
    assert sdk_mapping["capabilities"] == cli_plan["capabilities"]
    assert sdk_mapping["capability_plan"] == cli_plan["capability_plan"]
    assert sdk_mapping["environment_plan"] == cli_plan["environment_plan"]
    assert sdk_mapping["telemetry_plan"] == cli_plan["telemetry_plan"]
    assert sdk_mapping["resolution"] == cli_plan["resolution"]


async def test_typed_config():
    client = Fabric()
    sdk_and_cli_profile_stacks_match(client)
    await resolves_and_diagnoses_without_a_directory(client)
    await runs_without_an_agent_package(client)


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _cli_plan(agent: Path, *profiles: str) -> dict:
    args = [*COMMAND, "plan", str(agent)]
    for profile in profiles:
        args.extend(["--profile", profile])
    completed = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    return json.loads(completed.stdout)
