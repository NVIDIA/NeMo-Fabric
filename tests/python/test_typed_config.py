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
    config["metadata"] = {
        "name": "typed-only-run",
        "caller_annotation": "sdk",
    }
    config["harness"] = {
        "adapter_id": "test.fabric.hermes_shim",
        "resolution": "preinstalled",
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
        plan = client.plan(config, base_dir=base)
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
    assert plan.config.metadata.extra_fields["caller_annotation"] == "sdk"
    assert result["status"] == "succeeded", result["status"]
    assert result.request_id == "typed-request-1"
    assert result["adapter_kind"] == "python"
    assert result["metadata"]["adapter_runner"] == "persistent_local_host"
    assert "caller_annotation" not in result.metadata
    assert result["output"]["received"] == "hello typed"


async def diagnoses_adapter_incompatibility_without_weakening_plan(client: Fabric) -> None:
    """Doctor reports unsupported config that strict planning rejects."""

    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "incompatible-agent"},
            "harness": {"adapter_id": "nvidia.fabric.codex"},
            "runtime": {"max_turns": 3},
            "tools": {"enabled": []},
        }
    )

    with pytest.raises(FabricConfigError, match=r"runtime\.max_turns"):
        client.plan(config, base_dir=ROOT)

    report = await client.doctor(config, base_dir=ROOT)

    assert report.status == "fail"
    assert any(
        check.name == "config.unsupported"
        and check.metadata.get("field") == "runtime.max_turns"
        for check in report.checks
    )
    assert any(
        check.name == "capability.unsupported"
        and "tools.enabled" in check.message
        for check in report.checks
    )


def _model_provider_config(adapter_id: str, provider: str) -> FabricConfig:
    """Build a repository-backed config for model compatibility tests."""

    config = _repository_adapter_config().to_mapping()
    config["harness"]["adapter_id"] = adapter_id
    config["models"]["default"] = {
        "provider": provider,
        "model": "test-model",
    }
    return FabricConfig.from_mapping(config)


@pytest.mark.parametrize(
    ("adapter_id", "provider"),
    [
        ("nvidia.fabric.claude", "openai"),
        ("nvidia.fabric.codex", "anthropic"),
    ],
)
async def test_plan_and_doctor_require_custom_provider_connection(
    adapter_id: str,
    provider: str,
):
    config = _model_provider_config(adapter_id, provider)

    with pytest.raises(FabricConfigError, match=r"models\.default\.base_url"):
        Fabric().plan(config, base_dir=ROOT)

    report = await Fabric().doctor(config, base_dir=ROOT)

    assert report.status == "fail"
    assert any(
        check.name == "config.unsupported"
        and check.metadata.get("field") == "models.default.base_url"
        for check in report.checks
    )
    assert any(
        check.name == "config.unsupported"
        and check.metadata.get("field") == "models.default.api_key_env"
        for check in report.checks
    )


@pytest.mark.parametrize(
    ("adapter_id", "provider"),
    [
        ("nvidia.fabric.claude", "anthropic"),
        ("nvidia.fabric.codex", "openai"),
    ],
)
def test_plan_accepts_native_model_provider(adapter_id: str, provider: str):
    config = _model_provider_config(adapter_id, provider)

    plan = Fabric().plan(config, base_dir=ROOT)

    assert plan.config.models["default"]["provider"] == provider


@pytest.mark.parametrize(
    "adapter_id",
    ["nvidia.fabric.claude", "nvidia.fabric.codex"],
)
def test_plan_accepts_explicit_custom_provider_connection(adapter_id: str):
    config = _model_provider_config(adapter_id, "acme")
    config.models["default"].base_url = "https://models.example/v1"
    config.models["default"].api_key_env = "ACME_API_KEY"

    plan = Fabric().plan(config, base_dir=ROOT)

    assert plan.config.models["default"]["provider"] == "acme"


async def test_plan_and_doctor_reject_undeclared_model_setting():
    config = _model_provider_config("nvidia.fabric.claude", "anthropic")
    config.models["default"].settings["api_timeout"] = 30

    with pytest.raises(
        FabricConfigError,
        match=r"models\.default\.settings\.api_timeout",
    ):
        Fabric().plan(config, base_dir=ROOT)

    report = await Fabric().doctor(config, base_dir=ROOT)

    assert report.status == "fail"
    assert any(
        check.name == "config.unsupported"
        and check.metadata.get("field")
        == "models.default.settings.api_timeout"
        for check in report.checks
    )


async def test_typed_config():
    client = Fabric()
    await resolves_and_diagnoses_typed_config(client)
    await runs_with_typed_config_and_adapter_directory(client)
    await diagnoses_adapter_incompatibility_without_weakening_plan(client)
