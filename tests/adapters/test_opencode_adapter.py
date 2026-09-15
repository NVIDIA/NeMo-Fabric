# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused planning tests for the OpenCode v2 adapter."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from nemo_fabric import DiscoveryConfig
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import FabricConfigError
from nemo_fabric import HarnessConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import ModelConfig

ROOT = Path(__file__).resolve().parents[2]
DESCRIPTOR = ROOT / "adapters/typescript/opencode/opencode.fabric-adapter.json"


def config(*, api_key_env: str | None = "TEST_API_KEY", base_url: str | None = None) -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="opencode-adapter-test"),
        harness=HarnessConfig(adapter_id="nvidia.fabric.opencode"),
        discovery=DiscoveryConfig(local_paths=[DESCRIPTOR]),
        models={
            "default": ModelConfig(
                provider="openai",
                model="gpt-4.1-mini",
                api_key_env=api_key_env,
                base_url=base_url,
            )
        },
    )


def test_opencode_descriptor_declares_the_minimum_supported_surface():
    descriptor = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))

    assert descriptor == {
        "contract_version": "fabric.adapter/v1alpha2",
        "adapter_id": "nvidia.fabric.opencode",
        "adapter_kind": "process",
        "runner": {"command": "bun", "script": "dist/cli.js"},
        "requirements": {"binaries": ["bun"]},
        "config": {"accepts": ["models", "models.base_url"]},
        "model_schema": {
            "type": "object",
            "properties": {
                "provider": {"type": "string", "minLength": 1},
                "model": {"type": "string", "minLength": 1},
                "api_key_env": {"type": "string", "minLength": 1},
                "base_url": {
                    "type": ["string", "null"],
                    "format": "uri",
                    "pattern": "^https?://",
                },
            },
            "required": ["provider", "model", "api_key_env"],
            "additionalProperties": False,
        },
        "capabilities": {
            "streaming": False,
            "cancellation": False,
            "updates": False,
            "service": False,
        },
    }


def test_opencode_descriptor_plans_and_projects_the_selected_model():
    plan = Fabric().plan(config(), base_dir=ROOT)

    assert plan.adapter_descriptor["descriptor"]["adapter_id"] == "nvidia.fabric.opencode"
    assert plan.agent_config == {
        "models": {
            "default": {
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "api_key_env": "TEST_API_KEY",
            }
        }
    }


def test_opencode_model_schema_requires_a_credential_name():
    with pytest.raises(FabricConfigError, match="api_key_env"):
        Fabric().plan(config(api_key_env=None), base_dir=ROOT)


def test_opencode_descriptor_plans_an_openai_compatible_model_endpoint():
    plan = Fabric().plan(config(base_url="https://example.test/v1"), base_dir=ROOT)

    assert plan.agent_config["models"]["default"]["base_url"] == "https://example.test/v1"


def test_opencode_descriptor_rejects_a_non_http_model_endpoint():
    with pytest.raises(FabricConfigError, match="base_url"):
        Fabric().plan(config(base_url="ftp://example.test/v1"), base_dir=ROOT)


@pytest.mark.anyio
async def test_opencode_descriptor_doctor_requires_bun(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    bun = tmp_path / ("bun.exe" if os.name == "nt" else "bun")
    if os.name == "nt":
        bun.write_bytes(b"")
    else:
        bun.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        bun.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")

    report = await Fabric().doctor(config(), base_dir=ROOT)

    assert report.status in {"pass", "warn"}
    assert any(
        check.name == "requirement.binary" and "bun" in check.message and check.status == "pass"
        for check in report.checks
    )


@pytest.mark.anyio
async def test_opencode_descriptor_doctor_reports_a_missing_bun(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("PATH", "")

    report = await Fabric().doctor(config(), base_dir=ROOT)

    assert report.status == "fail"
    assert any(
        check.name == "requirement.binary" and "bun" in check.message and check.status == "fail"
        for check in report.checks
    )
