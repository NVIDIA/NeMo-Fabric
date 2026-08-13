# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the installed native Python SDK."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import nemo_fabric._native as native
import pytest
from _utils.configs import hermes_shim_config
from examples.code_review_agent import BASE_DIR
from examples.code_review_agent import base_config
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import FabricConfigError
from nemo_fabric import RunRequest


async def test_native_sdk(hermes_shim_agent_dir: Path):
    assert native.version()

    await smoke(Fabric(), hermes_shim_agent_dir)


async def test_adapter_python_selects_python_adapter_interpreter(
    hermes_shim_agent_dir: Path,
):
    os.environ["ADAPTER_PYTHON"] = sys.executable

    result = await Fabric().run(
        hermes_shim_config(),
        base_dir=hermes_shim_agent_dir,
        input="hello adapter python",
    )

    assert result["status"] == "succeeded"


def test_adapter_python_rejects_invalid_path_during_plan(
    hermes_shim_agent_dir: Path,
):
    invalid_python = hermes_shim_agent_dir / "missing-python"
    os.environ["ADAPTER_PYTHON"] = str(invalid_python)

    with pytest.raises(FabricConfigError, match="ADAPTER_PYTHON"):
        Fabric().plan(
            hermes_shim_config(),
            base_dir=hermes_shim_agent_dir,
        )


def test_native_run_rejects_multiple_request_sources(hermes_shim_agent_dir: Path):
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        native.run_config(
            json.dumps(hermes_shim_config().model_dump(mode="json", exclude_none=True)),
            str(hermes_shim_agent_dir),
            "text",
            None,
            "{}",
            None,
        )


def test_plan_rejects_adapter_incompatible_normalized_tool_policy():
    config = base_config()
    config.harness.adapter_id = "nvidia.fabric.codex"
    config.models["default"].provider = "openai"
    config.models["default"].model = "openai/gpt-5.4"
    config.models["default"].temperature = None
    config.block_tools("Bash")

    with pytest.raises(
        FabricConfigError,
        match=r"nvidia\.fabric\.codex.*tools\.blocked",
    ):
        Fabric().plan(config, base_dir=BASE_DIR)


@pytest.mark.parametrize(
    ("authentication", "capability"),
    [
        ({"type": "oauth2"}, "mcp.auth.oauth2"),
        (
            {
                "type": "service_account",
                "client_id": "fabric-client",
                "client_secret_env": "MCP_CLIENT_SECRET",
                "token_url": "https://auth.example/token",
                "token_endpoint_auth_method": "client_secret_basic",
            },
            "mcp.auth.service_account",
        ),
    ],
)
def test_plan_rejects_adapter_incompatible_mcp_authentication(
    authentication,
    capability,
):
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "unsupported-mcp-authentication"},
            "harness": {"adapter_id": "nvidia.fabric.claude"},
            "mcp": {
                "servers": {
                    "docs": {
                        "transport": "streamable-http",
                        "url": "https://mcp.example/docs",
                        "authentication": authentication,
                    }
                }
            },
        }
    )

    with pytest.raises(FabricConfigError) as exc_info:
        Fabric().plan(config, base_dir=BASE_DIR)

    message = str(exc_info.value)
    assert "mcp.servers.docs.authentication" in message
    assert capability in message


async def smoke(client: Fabric, fixture_agent: Path) -> None:
    example_config = base_config()

    plan = client.plan(example_config, base_dir=BASE_DIR)
    assert plan["agent_name"] == "code-review-agent"
    assert plan.base_dir == BASE_DIR
    assert plan.config.metadata.name == "code-review-agent"
    assert (
        plan["adapter_descriptor"]["descriptor"]["adapter_id"]
        == "nvidia.fabric.hermes"
    )
    assert "mcp_servers" not in plan["capability_plan"]["native"]
    assert plan["capability_plan"]["native"]["skill_paths"]

    minimal = FabricConfig.from_mapping(
        {
            "metadata": {"name": "minimal-typed-agent"},
            "harness": {"adapter_id": "nvidia.fabric.hermes"},
        }
    )
    minimal_plan = client.plan(minimal)
    assert minimal_plan.config.runtime.input_schema == "text"
    assert minimal_plan.config.runtime.output_schema == "text"

    typed_config = FabricConfig.from_mapping(
        {
            "schema_version": "fabric.agent/v1alpha1",
            "metadata": {"name": "typed-hermes-shim-agent"},
            "harness": {
                "adapter_id": "test.fabric.hermes_shim",
                "resolution": "preinstalled",
                "settings": {},
            },
            "models": {
                "default": {
                    "provider": "test",
                    "model": "test-model",
                    "temperature": 0.0,
                }
            },
            "runtime": {
                "input_schema": "chat",
                "output_schema": "message",
                "artifacts": "./artifacts",
                "timeout_seconds": 30,
            },
            "environment": {
                "provider": "local",
                "workspace": "./repos/my-service",
                "artifacts": "./artifacts/local",
            },
            "mcp": {
                "servers": {
                    "github": {
                        "transport": "streamable-http",
                        "url": "${GITHUB_MCP_URL}",
                        "exposure": "harness_native",
                        "custom_headers": {"X-Tenant": "fabric"},
                        "authentication": {
                            "type": "oauth2",
                            "client_id": "fabric-client",
                            "scopes": ["repo"],
                        },
                    }
                }
            },
            "telemetry": {"providers": {"relay": {}}},
            "relay": {"output_dir": "./artifacts/relay"},
            "consumer_extension": {
                "base": True,
                "custom": True,
                "nested": {"first": 1, "second": 2},
            },
        }
    )
    typed_plan = client.plan(
        typed_config,
        base_dir=fixture_agent,
    )
    assert typed_plan["agent_name"] == "typed-hermes-shim-agent"
    assert typed_plan["adapter_descriptor"]["source"] == "local"
    assert typed_plan["telemetry_plan"]["relay_enabled"] is True
    native_mcp = typed_plan["capability_plan"]["native"]["mcp_servers"]["github"]
    assert native_mcp["custom_headers"] == {"X-Tenant": "fabric"}
    assert native_mcp["authentication"] == {
        "type": "oauth2",
        "client_id": "fabric-client",
        "scopes": ["repo"],
    }
    resolved_config = typed_plan.config.to_mapping()
    assert resolved_config["harness"]["adapter_id"] == "test.fabric.hermes_shim"
    assert "settings" not in resolved_config["harness"]
    assert resolved_config["runtime"]["timeout_seconds"] == 30
    assert resolved_config["consumer_extension"] == {
        "base": True,
        "custom": True,
        "nested": {"first": 1, "second": 2},
    }

    result = await client.run(
        hermes_shim_config(),
        base_dir=fixture_agent,
        input="hello native",
    )
    async with await client.start_runtime(
        hermes_shim_config(),
        base_dir=fixture_agent,
    ) as runtime:
        first = await runtime.invoke(input="hello runtime one")
        second = await runtime.invoke(input="hello runtime two")
        openai_stream = runtime.invoke_openai_stream(input="hello streaming")
        openai_chunks = [chunk async for chunk in openai_stream]
        openai_result = await openai_stream.result()
        empty_stream = runtime.invoke_openai_stream(
            request=RunRequest(
                input="hello empty streaming",
                context={"openai_stream_mode": "empty"},
            )
        )
        empty_chunks = [chunk async for chunk in empty_stream]
        empty_result = await empty_stream.result()

    assert result["status"] == "succeeded"
    assert result.harness == "hermes"
    assert result["adapter_kind"] == "python"
    assert result["metadata"]["adapter_runner"] == "persistent_local_host"
    assert result["output"]["received"] == "hello native"
    assert result.output["native_mcp_servers"] == ["github"]
    assert any(artifact.name == "stdout" for artifact in result.artifacts.artifacts)
    assert first["status"] == "succeeded"
    assert second["status"] == "succeeded"
    assert first.harness == "hermes"
    assert first["runtime_id"] == second["runtime_id"]
    assert runtime.handle["runtime_id"] == first["runtime_id"]
    assert runtime.supports_openai_streaming is True
    assert [
        choice["delta"]["content"]
        for chunk in openai_chunks
        for choice in chunk["choices"]
    ] == ["hel", "lo"]
    assert openai_result.status == "succeeded"
    assert openai_result.output["received"] == "hello streaming"
    assert openai_result.output["openai_stream_invocation_count"] == 1
    assert empty_chunks == []
    assert empty_result.status == "succeeded"
    assert empty_result.output["openai_stream_invocation_count"] == 2
