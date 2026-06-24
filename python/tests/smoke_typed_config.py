# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: typed (in-memory) config is first-class, with no agent directory.

WS4 guardrail. The SDK's ``*_config`` methods accept a typed config object and
resolve, diagnose, and run it without an on-disk agent package:

* ``plan_config`` / ``doctor_config`` resolve a maintained (repository) adapter
  with ``base_dir=None`` -- zero filesystem layout, no ``agent.yaml``.
* ``run_config`` drives a real core runtime run using only a local adapter directory
  (still no agent package).
* the ``*_config`` methods are native-only; the CLI fallback raises a clear,
  documented error rather than silently degrading.

This complements ``smoke_native_sdk.py``, which exercises ``plan_config`` with a
``base_dir`` pointed at an agent package.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from shutil import copytree

from nemo_fabric import FabricClient, FabricNativeUnavailableError

ROOT = Path(__file__).resolve().parents[2]
# The test adapter (needs only python3, no secrets), shipped as a fixture.
SHIM_ADAPTERS = ROOT / "tests" / "fixtures" / "hermes-shim-agent" / "adapters"


def _repository_adapter_config() -> dict:
    """Config referencing a maintained adapter resolvable without any package."""

    return {
        "schema_version": "fabric.agent/v1alpha1",
        "metadata": {"name": "typed-only-agent"},
        "harness": {
            "adapter_id": "nvidia.fabric.hermes.sdk",
            "resolution": "preinstalled",
        },
        "models": {
            "default": {"provider": "nvidia", "model": "test-model", "temperature": 0.0}
        },
        "runtime": {
            "mode": "oneshot",
            "transport": "library",
            "input_schema": "chat",
            "output_schema": "message",
            "artifacts": "./artifacts",
        },
        "environment": {
            "provider": "local",
            "workspace": "./ws",
            "artifacts": "./artifacts/local",
        },
        "telemetry": {"enabled": False},
    }


def _shim_adapter_config() -> dict:
    """Config referencing the test adapter (runs without secrets)."""

    config = _repository_adapter_config()
    config["metadata"] = {"name": "typed-only-run"}
    config["harness"] = {
        "adapter_id": "test.fabric.hermes_shim",
        "resolution": "preinstalled",
        "settings": {"workspace": "./ws"},
    }
    config["models"] = {
        "default": {"provider": "test", "model": "test-model", "temperature": 0.0}
    }
    return config


async def resolves_and_diagnoses_without_a_directory(client: FabricClient) -> None:
    """plan_config / doctor_config resolve a maintained adapter with no package."""

    config = _repository_adapter_config()

    # base_dir=None: the literal "no path at all" call must still resolve.
    plan_no_path = client.plan_config(config)
    assert plan_no_path["agent_name"] == "typed-only-agent"

    # Point base_dir at an EMPTY directory (not an agent package): resolution can
    # then only succeed via the baked-in repository adapter dir, so the source is
    # deterministic regardless of the process CWD.
    with tempfile.TemporaryDirectory(prefix="typed-no-dir-") as empty:
        plan = client.plan_config(config, base_dir=empty)
        report = await client.doctor_config(config, base_dir=empty)

    descriptor = plan["adapter_descriptor"]
    assert descriptor["descriptor"]["adapter_id"] == "nvidia.fabric.hermes.sdk"
    # "repository" (not "local") proves it resolved without any on-disk package.
    assert descriptor["source"] == "repository", descriptor["source"]

    assert report["agent_name"] == "typed-only-agent"
    assert report["checks"], "doctor_config produced no checks"
    assert report["status"] in {"pass", "warn", "fail"}, report["status"]


async def runs_without_an_agent_package(client: FabricClient) -> None:
    """run_config drives a core run with only an adapter dir (no agent.yaml)."""

    config = _shim_adapter_config()
    with tempfile.TemporaryDirectory(prefix="typed-run-") as tmpdir:
        base = Path(tmpdir) / "scratch"
        # Only the adapter lives here -- this is deliberately NOT an agent
        # package (no agent.yaml / profiles / repos / skills).
        copytree(SHIM_ADAPTERS, base / "adapters")
        (base / "ws").mkdir()
        assert not (base / "agent.yaml").exists()
        result = await client.run_config(config, base_dir=base, input_text="hello typed")

    assert result["status"] == "succeeded", result.get("status")
    assert result["adapter_kind"] == "python"
    assert result["metadata"]["adapter_runner"] == "python"
    assert result["output"]["received"] == "hello typed"


async def typed_config_requires_native() -> None:
    """The CLI fallback surfaces a clear error for every typed-config method."""

    cli_client = FabricClient(command=("fabric",))
    config = _repository_adapter_config()

    # plan_config is sync; doctor_config / run_config are async. All three are
    # native-only, so each must raise rather than silently degrade over the CLI.
    try:
        cli_client.plan_config(config)
    except FabricNativeUnavailableError:
        pass
    else:
        raise AssertionError("plan_config should require the native extension over the CLI path")

    for name, coro in (
        ("doctor_config", cli_client.doctor_config(config)),
        ("run_config", cli_client.run_config(config)),
    ):
        try:
            await coro
        except FabricNativeUnavailableError:
            pass
        else:
            raise AssertionError(f"{name} should require the native extension over the CLI path")


async def main() -> None:
    await typed_config_requires_native()
    async with FabricClient() as client:
        await resolves_and_diagnoses_without_a_directory(client)
        await runs_without_an_agent_package(client)
    print("smoke_typed_config ok")


if __name__ == "__main__":
    asyncio.run(main())
