# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native OpenClaw configuration, model roles, and retained state."""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

import pytest
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric.errors import FabricConfigError
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.openclaw import adapter


def _context(workspace: Path, runtime_id: str = "openclaw-runtime") -> RuntimeContext:
    return RuntimeContext.from_mapping(
        {
            "runtime_id": runtime_id,
            "invocation_id": "openclaw-invocation",
            "request_id": "openclaw-request",
            "environment": {
                "environment_id": "openclaw-environment",
                "provider": "local",
                "control_location": "in_env_control",
                "workspace": str(workspace),
                "env": {},
                "ownership": "caller_owned",
            },
            "artifacts": {},
        }
    )


def _generate(tmp_path: Path, mapping: dict, port: int = 20_000) -> dict:
    return adapter._openclaw_config(
        AgentConfig.from_mapping(mapping),
        _context(tmp_path),
        base_dir=tmp_path,
        port=port,
        token_env="OPENCLAW_GATEWAY_TOKEN",
    )


MODEL = {"provider": "openai", "model": "fabric-model"}


def test_native_config_adds_sections_and_overrides_only_defaults(tmp_path: Path):
    native = {
        "plugins": {"allow": ["brave"], "entries": {"brave": {"enabled": True}}},
        "tools": {"web": {"search": {"enabled": True, "provider": "brave"}}},
        "agents": {"defaults": {"heartbeat": {"every": "30m"}}},
        "gateway": {"controlUi": {"enabled": True}},
        "telemetry": {"enabled": True},
    }
    generated = _generate(
        tmp_path,
        {
            "harness": {"settings": {"native_config": native}},
            "models": {"default": MODEL},
            "tools": {"enabled": ["read"]},
        },
    )

    assert generated["plugins"] == native["plugins"]
    assert generated["tools"] == {"allow": ["read"], "web": native["tools"]["web"]}
    assert generated["agents"]["defaults"]["heartbeat"] == {"every": "30m"}
    assert generated["agents"]["defaults"]["model"]["primary"] == "openai/fabric-model"
    assert generated["telemetry"] == {"enabled": True}, "defaults are overridable"
    assert generated["gateway"]["controlUi"] == {
        "enabled": True,
        "allowedOrigins": ["http://127.0.0.1:20000", "http://localhost:20000"],
    }
    assert generated["gateway"]["auth"]["mode"] == "token"


@pytest.mark.parametrize(
    "native",
    [
        {"tools": {"allow": ["exec"]}},
        {"models": {"providers": {}}},
        {"agents": {"defaults": {"workspace": "/elsewhere"}}},
        {"gateway": {"auth": {"mode": "none"}}},
    ],
)
def test_native_config_cannot_replace_fabric_owned_configuration(
    tmp_path: Path, native: dict
):
    with pytest.raises(lifecycle.LifecycleError, match="native_config"):
        _generate(
            tmp_path,
            {
                "harness": {"settings": {"native_config": native}},
                "models": {"default": MODEL},
                "tools": {"enabled": ["read"]},
            },
        )


def test_every_model_role_is_registered_with_its_protocol_and_metadata(
    tmp_path: Path,
):
    default = {
        "provider": "openai",
        "model": "reasoner",
        "base_url": "https://models.example.test/v1",
        "api_key_env": "MODEL_KEY",
        "api": "openai-responses",
        "max_tokens": 8192,
        "settings": {
            "model_metadata": {"contextWindow": 65536, "reasoning": True},
            "reasoning_effort": "high",
        },
    }
    review = {
        "provider": "anthropic",
        "model": "reviewer",
        "base_url": "https://review.example.test/v1",
        "api": "anthropic-messages",
    }
    generated = _generate(
        tmp_path,
        {"models": {"default": default, "primary": default, "review": review}},
    )

    providers = generated["models"]["providers"]
    assert providers["openai"] == {
        "baseUrl": "https://models.example.test/v1",
        "api": "openai-responses",
        "models": [
            {
                "contextWindow": 65536,
                "reasoning": True,
                "id": "reasoner",
                "name": "reasoner",
                "maxTokens": 8192,
            }
        ],
        "apiKey": {"source": "env", "provider": "default", "id": "MODEL_KEY"},
    }
    assert providers["anthropic-review"]["api"] == "anthropic-messages"
    defaults = generated["agents"]["defaults"]
    assert defaults["model"] == {"primary": "openai/reasoner"}
    assert defaults["thinkingDefault"] == "high"
    assert defaults["models"]["openai/reasoner"]["alias"] == "default"
    assert defaults["models"]["anthropic-review/reviewer"] == {
        "agentRuntime": {"id": "openclaw"},
        "alias": "review",
    }
    assert len(defaults["models"]) == 2, "identical roles share one model"


def test_fixed_port_requires_its_derived_ports(monkeypatch):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        free = sock.getsockname()[1]
    monkeypatch.setattr(adapter, "_port_available", lambda port: port != free + 2)
    with pytest.raises(lifecycle.LifecycleError, match="port"):
        adapter._select_port({"port": free})
    monkeypatch.setattr(adapter, "_port_available", lambda port: True)
    assert adapter._select_port({"port": free}) == free


def _plan(tmp_path: Path, settings: dict) -> None:
    Fabric().plan(
        FabricConfig.from_mapping(
            {
                "metadata": {"name": "openclaw-native"},
                "harness": {"adapter_id": "nvidia.fabric.openclaw", "settings": settings},
                "models": {
                    "default": {
                        **MODEL,
                        "base_url": "http://127.0.0.1:9/v1",
                        "api": "openai-completions",
                    }
                },
            }
        ),
        base_dir=tmp_path,
    )


def test_planning_checks_native_settings_before_startup(tmp_path: Path):
    _plan(tmp_path, {"state_dir": "state", "port": 18800})
    control_ui = {"gateway": {"controlUi": {"enabled": True}}}
    _plan(tmp_path, {"state_dir": "state", "native_config": control_ui})
    with pytest.raises(FabricConfigError, match="state_dir"):
        _plan(tmp_path, {"native_config": control_ui})
    with pytest.raises(FabricConfigError, match="gateway.port"):
        _plan(tmp_path, {"native_config": {"gateway": {"port": 18800}}})
    with pytest.raises(FabricConfigError, match="port"):
        _plan(tmp_path, {"port": 18800, "port_range": {"start": 20000, "end": 20200}})


@pytest.mark.skipif(
    sys.platform in {"darwin", "win32"},
    reason="The mock Gateway lifecycle runs only on Linux CI, as in test_openclaw.py",
)
async def test_state_dir_retains_state_credential_and_log(
    repo_root: Path, tmp_path: Path, monkeypatch
):
    state = tmp_path / "state"
    capture = tmp_path / "config.json"
    monkeypatch.setenv("FAKE_OPENCLAW_CAPTURE", str(capture))
    monkeypatch.setenv("FAKE_OPENCLAW_REQUEST", str(tmp_path / "request.json"))
    config = AgentConfig.from_mapping(
        {
            "harness": {
                "settings": {
                    "openclaw_command": str(repo_root / "tests/_utils/mock_openclaw.py"),
                    "state_dir": str(state),
                }
            },
            "models": {"default": MODEL},
        }
    )
    payload = {
        "config": config,
        "runtime_context": _context(tmp_path).to_mapping(),
        "base_dir": str(tmp_path),
    }

    runtime = adapter.OpenClawRuntime()
    await runtime.start(payload)
    try:
        token = (state / "interface-token").read_text(encoding="ascii")
        assert runtime._token == token
        assert json.loads((state / "openclaw.json").read_text()) == json.loads(
            capture.read_text()
        )
        with pytest.raises(lifecycle.LifecycleError, match="in use"):
            await adapter.OpenClawRuntime().start(payload)
        result = await runtime.invoke(AgentRunRequest(input="Hello."), _context(tmp_path))
        assert result.status == "succeeded"
    finally:
        await runtime.stop()
    assert (state / "gateway.log").exists()
    assert (state / "openclaw.json").stat().st_mode & 0o077 == 0

    restarted = adapter.OpenClawRuntime()
    await restarted.start(payload)
    try:
        assert restarted._token == token, "the credential survives restarts"
    finally:
        await restarted.stop()
    assert os.path.isdir(state)
