# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenCode E2E coverage.

Run the credentialed smoke test with:
RUN_FABRIC_OPENCODE_INTEGRATION=1 \\
OPENCODE_LIVE_PROVIDER=openai \\
OPENCODE_LIVE_MODEL=<model> \\
OPENCODE_LIVE_API_KEY_ENV=OPENAI_API_KEY \\
OPENCODE_LIVE_BASE_URL=https://example.com/v1 \\
pytest tests/e2e/test_opencode.py
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import uuid

import pytest
import requests

from nemo_fabric import DiscoveryConfig
from nemo_fabric import EnvironmentConfig
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import HarnessConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import ModelConfig
from nemo_fabric import RuntimeConfig

ROOT = Path(__file__).resolve().parents[2]
DESCRIPTOR = ROOT / "adapters/typescript/opencode/opencode.fabric-adapter.json"

pytestmark = pytest.mark.skipif(
    shutil.which("bun") is None, reason="Bun is required for the OpenCode adapter E2E"
)


def opencode_config(api_server: str, workspace: Path) -> FabricConfig:
    artifacts = workspace / "artifacts"
    return FabricConfig(
        metadata=MetadataConfig(name="opencode-e2e"),
        harness=HarnessConfig(adapter_id="nvidia.fabric.opencode"),
        discovery=DiscoveryConfig(local_paths=[DESCRIPTOR]),
        models={
            "default": ModelConfig(
                provider="fabric-test",
                model="fabric-echo",
                api_key_env="OPENCODE_E2E_KEY",
                base_url=f"{api_server}/v1",
            )
        },
        environment=EnvironmentConfig(
            provider="local",
            workspace=workspace,
            artifacts=artifacts,
            env={"OPENCODE_E2E_KEY": "test-key"},
        ),
        runtime=RuntimeConfig(
            input_schema="chat", output_schema="message", artifacts=artifacts
        ),
    )


def live_opencode_config(
    *,
    workspace: Path,
    provider: str,
    model: str,
    api_key_env: str,
    credential: str,
    base_url: str,
) -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="opencode-live"),
        harness=HarnessConfig(adapter_id="nvidia.fabric.opencode"),
        discovery=DiscoveryConfig(local_paths=[DESCRIPTOR]),
        models={
            "default": ModelConfig(
                provider=provider,
                model=model,
                api_key_env=api_key_env,
                base_url=base_url,
            )
        },
        environment=EnvironmentConfig(
            provider="local", workspace=workspace, env={api_key_env: credential}
        ),
    )


def result_error_summary(result) -> str:
    if result.error is None:
        return "no structured error"
    return f"{result.error.code}: {result.error.message}"


async def test_opencode_runs_against_a_local_openai_compatible_endpoint(
    api_server, tmp_path
):
    config = opencode_config(api_server, tmp_path)
    fabric = Fabric()

    plan = fabric.plan(config, base_dir=tmp_path)
    assert plan.adapter_descriptor["descriptor"]["adapter_id"] == "nvidia.fabric.opencode"

    single = await fabric.run(config, base_dir=tmp_path, input="single")
    assert single["status"] == "succeeded", result_error_summary(single)
    assert single["output"]["response"] == "echo user_count=1 latest=single"

    async with await fabric.start_runtime(config, base_dir=tmp_path) as runtime:
        first = await runtime.invoke(input="first")
        second = await runtime.invoke(input="second")

    async with await fabric.start_runtime(config, base_dir=tmp_path) as isolated_runtime:
        isolated = await isolated_runtime.invoke(input="isolated")

    assert first["status"] == second["status"] == "succeeded"
    assert first["output"]["response"] == "echo user_count=1 latest=first"
    assert second["output"]["response"] == "echo user_count=2 latest=second"
    assert isolated["status"] == "succeeded"
    assert isolated["output"]["response"] == "echo user_count=1 latest=isolated"
    captured = requests.get(f"{api_server}/_requests", timeout=5).json()
    assert captured
    assert all("prompt_cache_key" not in payload for payload in captured)


async def test_opencode_live_provider_runtime(tmp_path):
    if os.environ.get("RUN_FABRIC_OPENCODE_INTEGRATION") != "1":
        pytest.skip("set RUN_FABRIC_OPENCODE_INTEGRATION=1 to run")
    provider = os.environ.get("OPENCODE_LIVE_PROVIDER")
    model = os.environ.get("OPENCODE_LIVE_MODEL")
    api_key_env = os.environ.get("OPENCODE_LIVE_API_KEY_ENV")
    base_url = os.environ.get("OPENCODE_LIVE_BASE_URL")
    if not provider or not model or not api_key_env or not base_url:
        pytest.fail(
            "set OPENCODE_LIVE_PROVIDER, OPENCODE_LIVE_MODEL, "
            "OPENCODE_LIVE_API_KEY_ENV, and OPENCODE_LIVE_BASE_URL"
        )
    credential = os.environ.get(api_key_env)
    if not credential:
        pytest.fail(f"the configured credential {api_key_env!r} is not set")

    workspace = tmp_path
    config = live_opencode_config(
        workspace=workspace,
        provider=provider,
        model=model,
        api_key_env=api_key_env,
        credential=credential,
        base_url=base_url,
    )

    async with await Fabric().start_runtime(config, base_dir=workspace) as runtime:
        nonce = f"fabric-opencode-{uuid.uuid4().hex[:8]}"
        first = await runtime.invoke(input=f"Remember this token exactly: {nonce}")
        second = await runtime.invoke(
            input="Reply with only the token I asked you to remember."
        )

    assert first["status"] == second["status"] == "succeeded", (
        "OpenCode live invocation failed. "
        f"first={result_error_summary(first)}; "
        f"second={result_error_summary(second)}"
    )
    assert first["output"]["response"]
    assert nonce in second["output"]["response"]
