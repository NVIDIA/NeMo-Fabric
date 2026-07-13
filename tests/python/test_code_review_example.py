# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the code-review example."""

import json
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock

from examples.code_review_agent import __main__ as main_module
from examples.code_review_agent import (
    BASE_DIR,
    base_config,
    codex_cli_config,
    hermes_config,
    with_fabric_managed_github_mcp,
    with_native_otel,
    with_opensandbox,
    with_relay,
    with_relay_openinference,
    with_relay_otel,
)
from nemo_fabric import Fabric, FabricConfig, RunOutput


def test_variant_builders_return_independent_complete_configs():
    base = base_config()
    hermes = hermes_config()
    codex = codex_cli_config()

    for config in (base, hermes, codex):
        assert isinstance(config, FabricConfig)
        assert config.metadata.name == "code-review-agent"
        assert config.environment is not None
        assert "default" in config.models

    assert hermes is not base
    assert hermes.harness is not base.harness
    assert codex.harness.adapter_id == "nvidia.fabric.codex.cli"
    assert codex.mcp is None
    assert codex.skills is None
    assert base.mcp is not None
    assert base.skills is not None


def test_capability_and_telemetry_variants_do_not_mutate_their_input():
    base = hermes_config()
    variants = (
        with_fabric_managed_github_mcp(base),
        with_native_otel(base),
        with_opensandbox(base),
        with_relay(base),
        with_relay_openinference(base),
        with_relay_otel(base),
    )

    assert base.telemetry is not None
    assert base.telemetry.enabled is False
    assert base.environment is not None
    assert base.environment.provider == "local"
    assert base.mcp is not None
    assert base.mcp.servers["github"].exposure == "harness_native"
    assert all(variant is not base for variant in variants)
    assert variants[0].mcp is not None
    assert variants[0].mcp.servers["github"].exposure == "fabric_managed"
    assert variants[1].telemetry is not None
    assert variants[1].telemetry.provider == "native"
    assert variants[2].environment is not None
    assert variants[2].environment.provider == "opensandbox"
    assert variants[3].telemetry is not None
    assert variants[3].telemetry.provider == "relay"


def test_variants_plan_without_file_profiles():
    client = Fabric()

    for config in (hermes_config(), codex_cli_config()):
        plan = client.plan(config, base_dir=BASE_DIR)
        assert plan.profiles == ()
        assert plan.agent_name == "code-review-agent"
        assert plan.adapter.adapter_id == config.harness.adapter_id


def test_example_entrypoint_plans_without_starting_a_runtime():
    cases = (
        ([], "nvidia.fabric.hermes", False),
        (["--variant", "codex-cli"], "nvidia.fabric.codex.cli", False),
        (["--relay"], "nvidia.fabric.hermes", True),
    )

    for options, adapter_id, relay_enabled in cases:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "examples.code_review_agent",
                *options,
                "--plan",
            ],
            cwd=BASE_DIR.parents[1],
            text=True,
            capture_output=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        plan = json.loads(completed.stdout)
        assert plan["agent_name"] == "code-review-agent"
        assert plan["profiles"] == []
        assert (
            plan["adapter_descriptor"]["descriptor"]["adapter_id"] == adapter_id
        )
        assert plan["telemetry_plan"]["relay_enabled"] is relay_enabled


async def test_example_entrypoint_shows_response_after_normalized_output(
    monkeypatch,
    capsys,
):
    result = MagicMock()
    result.output = RunOutput.from_mapping({"response": "visible response"})
    result.to_mapping.return_value = {"output": result.output.to_mapping()}
    mock_fabric = MagicMock()
    mock_fabric.run = AsyncMock(return_value=result)
    monkeypatch.setattr(main_module, "Fabric", lambda: mock_fabric)
    monkeypatch.setattr(
        sys,
        "argv",
        ["code_review_agent", "--show-output", "--input", "review this"],
    )

    await main_module.main()

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert captured.err == ""
    assert lines[-1] == "visible response"
    assert json.loads("\n".join(lines[:-1])) == {
        "output": {"response": "visible response"}
    }
