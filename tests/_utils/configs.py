# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete typed configs shared by Fabric tests."""

from __future__ import annotations

from nemo_fabric import EnvironmentConfig, FabricConfig, HarnessConfig, MetadataConfig, ModelConfig, RuntimeConfig


def hermes_shim_config() -> FabricConfig:
    config = FabricConfig(
        metadata=MetadataConfig(name="hermes-shim-agent", description="Test-only Hermes-shaped agent."),
        harness=HarnessConfig(
            adapter_id="test.fabric.hermes_shim",
            resolution="preinstalled",
            settings={},
        ),
        models={"default": ModelConfig(provider="test", model="test-model", temperature=0.0)},
        runtime=RuntimeConfig(input_schema="chat", output_schema="message", artifacts="./artifacts"),
        environment=EnvironmentConfig(
            provider="local",
            workspace="./repos/my-service",
            artifacts="./artifacts/local",
        ),
    )
    config.add_skill_path("./skills/code-review")
    config.add_mcp_server(
        "github",
        transport="streamable-http",
        url="${GITHUB_MCP_URL}",
        exposure="harness_native",
    )
    return config


def swebench_shim_config() -> FabricConfig:
    config = hermes_shim_config()
    config.harness = HarnessConfig(
        adapter_id="test.fabric.hermes_shim",
        resolution="preinstalled",
        settings={
            "mode": "swebench_shim",
            "target_file": "calculator.py",
            "expected_before": "return 41",
            "replacement": "return 42",
            "new_file": "generated/fix-notes.txt",
            "new_file_contents": "patched by Fabric\n",
        },
    )
    config.runtime = RuntimeConfig(
        input_schema="swe_bench_task",
        output_schema="patch_result",
        artifacts="./artifacts/swebench",
    )
    config.environment = EnvironmentConfig(
        provider="local",
        workspace="./repos/my-service",
        artifacts="./artifacts/swebench",
    )
    return config


def harbor_swebench_config() -> FabricConfig:
    config = hermes_shim_config()
    workspace = "./repos/swebench-django-13741"
    config.harness = HarnessConfig(
        adapter_id="test.fabric.hermes_shim",
        resolution="preinstalled",
        settings={
            "mode": "swebench_shim",
            "target_file": "django/contrib/auth/forms.py",
            "expected_before": "        kwargs.setdefault(\"required\", False)\n        super().__init__(*args, **kwargs)",
            "replacement": "        kwargs.setdefault(\"required\", False)\n        kwargs.setdefault('disabled', True)\n        super().__init__(*args, **kwargs)",
        },
    )
    config.runtime = RuntimeConfig(
        input_schema="harbor_swe_bench_task",
        output_schema="patch_result",
        artifacts="./artifacts/harbor-swebench-django-13741",
    )
    config.environment = EnvironmentConfig(
        provider="local",
        workspace=workspace,
        artifacts="./artifacts/harbor-swebench-django-13741",
        metadata={"source": "harbor_swebench", "instance_id": "django__django-13741"},
    )
    return config
