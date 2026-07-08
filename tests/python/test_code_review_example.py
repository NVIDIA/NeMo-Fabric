# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the Pydantic-first code-review example."""

from examples.code_review_agent import (
    BASE_DIR,
    base_config,
    codex_cli_config,
    hermes_cli_config,
    hermes_sdk_config,
    with_fabric_managed_github_mcp,
    with_native_otel,
    with_opensandbox,
    with_relay,
    with_relay_openinference,
    with_relay_otel,
)
from nemo_fabric import Fabric, FabricConfig


def test_variant_builders_return_independent_complete_configs():
    base = base_config()
    sdk = hermes_sdk_config()
    cli = hermes_cli_config()
    codex = codex_cli_config()

    for config in (base, sdk, cli, codex):
        assert isinstance(config, FabricConfig)
        assert config.metadata.name == "code-review-agent"
        assert config.environment is not None
        assert "default" in config.models

    assert sdk is not base
    assert sdk.harness is not base.harness
    assert cli.harness.adapter_id == "nvidia.fabric.hermes.cli"
    assert codex.harness.adapter_id == "nvidia.fabric.codex.cli"
    assert codex.mcp is None
    assert codex.skills is None
    assert base.mcp is not None
    assert base.skills is not None


def test_capability_and_telemetry_variants_do_not_mutate_their_input():
    base = hermes_sdk_config()
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


def test_typed_variants_plan_without_file_profiles():
    client = Fabric()

    for config in (hermes_sdk_config(), hermes_cli_config(), codex_cli_config()):
        plan = client.plan(config, base_dir=BASE_DIR)
        assert plan.profiles == ()
        assert plan.agent_name == "code-review-agent"
        assert plan.adapter.adapter_id == config.harness.adapter_id
