# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: typed config is the first-class Fabric input.

The SDK methods accept a complete typed config and an optional base directory:

* ``plan`` / ``doctor`` resolve a maintained (repository) adapter
  with ``base_dir=None``.
* ``run`` drives a real core runtime run using only a local adapter directory
  and a typed config.
This complements ``test_native_sdk.py``, which exercises ``plan`` with a
``base_dir`` pointed at an agent package.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from shutil import copytree

import pytest

from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import FabricConfigError
from nemo_fabric import RunRequest
from nemo_fabric import RunResult

ROOT = Path(__file__).resolve().parents[2]
SHIM_ADAPTERS = ROOT / "tests" / "fixtures" / "hermes-shim-agent" / "adapters"


def _repository_adapter_config() -> FabricConfig:
    """Config referencing a maintained adapter."""

    return FabricConfig.from_mapping(
        {
            "schema_version": "fabric.agent/v1alpha1",
            "metadata": {"name": "typed-only-agent"},
            "harness": {
                "adapter_id": "nvidia.fabric.hermes",
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
        "settings": {},
    }
    config["models"] = {
        "default": {"provider": "test", "model": "test-model", "temperature": 0.0}
    }
    return FabricConfig.from_mapping(config)


async def resolves_and_diagnoses_typed_config(client: Fabric) -> None:
    """Plan and doctor resolve a complete typed config."""

    config = _repository_adapter_config()

    plan_no_path = client.plan(config)
    assert plan_no_path["agent_name"] == "typed-only-agent"

    with tempfile.TemporaryDirectory(prefix="typed-no-dir-") as empty:
        plan = client.plan(config, base_dir=empty)
        report = await client.doctor(config, base_dir=empty)

    descriptor = plan["adapter_descriptor"]
    assert descriptor["descriptor"]["adapter_id"] == "nvidia.fabric.hermes"
    assert descriptor["source"] == "repository", descriptor["source"]

    assert report["agent_name"] == "typed-only-agent"
    assert report.checks, "doctor produced no checks"
    assert report["status"] in {"pass", "warn", "fail"}, report["status"]


async def runs_with_typed_config_and_adapter_directory(client: Fabric) -> None:
    """Run with a typed config and local adapter descriptor."""

    config = _shim_adapter_config()
    with tempfile.TemporaryDirectory(prefix="typed-run-") as tmpdir:
        base = Path(tmpdir) / "scratch"
        copytree(SHIM_ADAPTERS, base / "adapters")
        (base / "ws").mkdir()
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
    assert result["metadata"]["adapter_runner"] == "persistent_local_host"
    assert result["output"]["received"] == "hello typed"


async def test_typed_config():
    client = Fabric()
    await resolves_and_diagnoses_typed_config(client)
    await runs_with_typed_config_and_adapter_directory(client)


@pytest.mark.parametrize(
    ("adapter_id", "provider", "supported_provider"),
    [
        ("nvidia.fabric.claude", "openai", "anthropic"),
        ("nvidia.fabric.codex", "anthropic", "openai"),
    ],
)
async def test_plan_and_doctor_share_model_provider_diagnostic(
    adapter_id: str,
    provider: str,
    supported_provider: str,
):
    config = _repository_adapter_config()
    config.harness.adapter_id = adapter_id
    config.models["default"].provider = provider

    with pytest.raises(FabricConfigError) as plan_error:
        Fabric().plan(config)
    with pytest.raises(FabricConfigError) as doctor_error:
        await Fabric().doctor(config)

    plan_diagnostic = str(plan_error.value)
    assert str(doctor_error.value) == plan_diagnostic
    assert "models.default.provider" in plan_diagnostic
    assert supported_provider in plan_diagnostic


def test_plan_rejects_undeclared_model_setting_without_exposing_value():
    config = _repository_adapter_config()
    config.harness.adapter_id = "nvidia.fabric.claude"
    config.models["default"].provider = "anthropic"
    config.models["default"].settings["regionn"] = "secret-setting-value"

    with pytest.raises(FabricConfigError) as caught:
        Fabric().plan(config)

    diagnostic = str(caught.value)
    assert "models.default.settings.regionn" in diagnostic
    assert "secret-setting-value" not in diagnostic
