# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenClaw native projection stays within the Fabric adapter."""

import pytest
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapters.openclaw.adapter import native_configuration


def test_public_models_tools_and_native_features_share_one_native_configuration():
    config = AgentConfig.from_mapping(
        {
            "harness": {
                "settings": {
                    "agent_name": "writer",
                    "timeout_seconds": 42,
                    "native_config": {
                        "gateway": {"port": 18888, "controlUi": {"enabled": True}},
                        "plugins": {"allow": ["tavily"]},
                        "agents": {"defaults": {"heartbeat": {"every": "5m"}}},
                    },
                }
            },
            "models": {
                "default": {
                    "provider": "openai",
                    "model": "first",
                    "base_url": "https://first.invalid/v1",
                    "api_key_env": "FIRST_KEY",
                    "settings": {"api": "openai-responses"},
                },
                "fast": {
                    "provider": "openai",
                    "model": "second",
                    "base_url": "https://second.invalid/v1",
                    "api_key_env": "SECOND_KEY",
                    "settings": {"api": "openai-completions"},
                },
            },
            "tools": {"enabled": ["read"]},
        }
    )
    native = native_configuration(config, "/workspace")
    assert native["gateway"]["port"] == 18888
    assert native["plugins"]["allow"] == ["tavily"]
    assert native["agents"]["defaults"]["heartbeat"] == {"every": "5m"}
    assert native["agents"]["defaults"]["timeoutSeconds"] == 42
    assert native["agents"]["entries"]["writer"]["tools"] == {"allow": ["read"]}
    assert native["models"]["providers"]["fabric_default"]["apiKey"] == "${FIRST_KEY}"
    assert (
        native["models"]["providers"]["fabric_fast"]["baseUrl"]
        == "https://second.invalid/v1"
    )
    assert native["models"]["providers"]["fabric_default"]["api"] == "openai-responses"


def test_native_config_cannot_override_public_model_or_tool_constraints():
    config = AgentConfig.from_mapping(
        {
            "harness": {
                "settings": {"native_config": {"models": {"providers": {"other": {}}}}}
            },
            "models": {
                "default": {
                    "provider": "openai",
                    "model": "model",
                    "base_url": "https://model.invalid",
                    "settings": {"api": "openai-completions"},
                }
            },
        }
    )
    with pytest.raises(ValueError, match="models"):
        native_configuration(config, "/workspace")


async def test_public_lifecycle_starts_gateway_and_invokes_without_consumer_translation(
    tmp_path,
):
    import json
    import os
    import sys
    from pathlib import Path
    from nemo_fabric import Fabric, FabricConfig

    root = Path(__file__).resolve().parents[2]
    os.environ["PYTHONPATH"] = str(root / "adapters/python/openclaw/src")
    os.environ["ADAPTER_PYTHON"] = sys.executable
    cli = tmp_path / "openclaw.mjs"
    cli.write_text("""
if (process.argv[3] !== "call") { setInterval(() => {}, 1000); }
else {
  const method = process.argv[4];
  console.log(JSON.stringify(method === "agent" ? {status:"ok", result:{payloads:[{text:"owner response"}]}} : {messages:[]}));
}
""")
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "openclaw-fixture"},
            "harness": {
                "adapter_id": "nvidia.fabric.openclaw",
                "settings": {"cli": str(cli)},
            },
            "models": {
                "default": {
                    "provider": "openai",
                    "model": "model",
                    "base_url": "https://model.invalid",
                    "settings": {"api": "openai-completions"},
                }
            },
        }
    )
    result = await Fabric().run(config, base_dir=tmp_path, input="hello")
    assert result.status == "succeeded", result.to_mapping()
    assert result.output["response"] == "owner response"
    native = json.loads((tmp_path / ".openclaw/openclaw.json").read_text())
    assert native["models"]["providers"]["fabric_default"]["models"][0]["id"] == "model"


def test_enabled_dashboard_requires_protected_adapter_owned_token():
    config = AgentConfig.from_mapping(
        {
            "harness": {
                "settings": {
                    "native_config": {"gateway": {"controlUi": {"enabled": True}}}
                }
            },
            "models": {
                "default": {
                    "provider": "openai",
                    "model": "model",
                    "base_url": "https://model.invalid",
                    "extensions": {"api": "openai-completions"},
                }
            },
        }
    )
    gateway = native_configuration(config, "/workspace")["gateway"]
    assert gateway["auth"] == {"mode": "token", "token": "${FABRIC_INTERFACE_TOKEN}"}
    assert gateway["bind"] == "loopback"
    assert gateway["controlUi"]["allowedOrigins"] == [
        "http://127.0.0.1:18789",
        "http://localhost:18789",
    ]


def test_blocked_tools_do_not_replace_unspecified_allowlist():
    config = AgentConfig.from_mapping(
        {
            "harness": {"settings": {}},
            "models": {
                "default": {
                    "provider": "openai",
                    "model": "test",
                    "base_url": "http://localhost/v1",
                    "extensions": {"api": "openai-completions"},
                }
            },
            "tools": {"blocked": ["exec"]},
        }
    )
    native = native_configuration(config, "/workspace")
    assert native["agents"]["entries"]["main"]["tools"] == {"deny": ["exec"]}
