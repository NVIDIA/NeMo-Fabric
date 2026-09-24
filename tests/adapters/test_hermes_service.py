# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hermes service mode uses public config and never replays uncertain turns."""

import os
from unittest.mock import AsyncMock, MagicMock
import pytest
from nemo_fabric_adapter_contract.models import (
    AgentConfig,
    AgentRunRequest,
    RuntimeContext,
)
import sys

if sys.version_info >= (3, 14):
    pytest.skip("Hermes adapter supports Python 3.11–3.13", allow_module_level=True)

from nemo_fabric_adapters.hermes import service


async def test_service_mode_persists_native_config_and_quarantines_failed_turn(
    tmp_path, monkeypatch
):
    os.environ["FIXTURE_KEY"] = "fixture-placeholder"
    process = MagicMock(returncode=None)
    process.wait = AsyncMock()
    monkeypatch.setattr(
        service.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)
    )
    monkeypatch.setattr(service.os, "killpg", MagicMock())
    request = MagicMock(return_value={"data": [{"id": "primary"}]})
    monkeypatch.setattr(service, "api_request", request)
    runtime = service.HermesServiceRuntime()
    context = {
        "runtime_id": "fixture",
        "invocation_id": "invoke",
        "request_id": "request",
        "artifacts": {},
        "environment": {
            "provider": "local",
            "environment_id": "local",
            "ownership": "caller_owned",
            "control_location": "external_control",
            "workspace": str(tmp_path),
        },
    }
    config = AgentConfig.from_mapping(
        {
            "harness": {
                "settings": {
                    "mode": "service",
                    "interfaces": {"dashboard": {"enabled": False}},
                    "native_config": {"web": {"backend": "tavily"}},
                }
            },
            "models": {
                "default": {
                    "provider": "openai",
                    "model": "fixture",
                    "base_url": "https://fixture.invalid",
                    "api_key_env": "FIXTURE_KEY",
                    "api": "openai-responses",
                }
            },
        }
    )
    await runtime.start(
        {"config": config, "runtime_context": context, "base_dir": str(tmp_path)}
    )
    assert service.configuration_matches(runtime.native)
    assert runtime.native["model"]["api_mode"] == "codex_responses"
    assert runtime.native["model"]["provider"] == "custom:fabric"
    assert runtime.native["providers"]["fabric"]["key_env"] == "FIXTURE_KEY"
    assert "api_key" not in runtime.native["providers"]["fabric"]
    assert (tmp_path / ".hermes/interface-token").stat().st_mode & 0o777 == 0o600
    request.return_value = {
        "status": "completed",
        "id": "turn-1",
        "output": [{"content": [{"type": "output_text", "text": "hello"}]}],
    }
    result = await runtime.invoke(
        AgentRunRequest(input="hello"), RuntimeContext.from_mapping(context)
    )
    assert result.output["response"] == "hello"
    request.side_effect = OSError("uncertain network result")
    result = await runtime.invoke(
        AgentRunRequest(input="again"), RuntimeContext.from_mapping(context)
    )
    assert result.status == "failed"
    with pytest.raises(RuntimeError, match="no replay"):
        await runtime.invoke(
            AgentRunRequest(input="again"), RuntimeContext.from_mapping(context)
        )
    assert request.call_count == 3
