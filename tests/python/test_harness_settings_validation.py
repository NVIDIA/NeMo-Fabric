# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planning validation for descriptor-owned harness settings schemas."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import nemo_fabric.client as client_module
import pytest
from _utils.configs import minimal_config
from nemo_fabric import Fabric
from nemo_fabric import FabricConfigError


ROOT = Path(__file__).resolve().parents[2]
ADAPTER_DESCRIPTORS = {
    "nvidia.fabric.claude": ROOT / "adapters" / "claude" / "fabric-adapter.json",
    "nvidia.fabric.codex": ROOT / "adapters" / "codex" / "fabric-adapter.json",
    "nvidia.fabric.langchain.deepagents": (
        ROOT / "adapters" / "deepagents" / "fabric-adapter.json"
    ),
    "nvidia.fabric.hermes": ROOT / "adapters" / "hermes" / "fabric-adapter.json",
}


_config = partial(
    minimal_config,
    name="harness-settings-validation",
    adapter_id="nvidia.fabric.claude",
    resolution="preinstalled",
)


@pytest.mark.parametrize(
    ("adapter_id", "settings"),
    [
        pytest.param(
            "nvidia.fabric.claude",
            {
                "setting_sources": ["user", "project", "local"],
                "max_budget_usd": 1.5,
                "permission_mode": "dontAsk",
            },
            id="claude",
        ),
        pytest.param(
            "nvidia.fabric.codex",
            {
                "sandbox": "workspace-write",
                "approval_mode": "deny_all",
                "developer_instructions": "Review the implementation.",
                "personality": "pragmatic",
                "reasoning_effort": "high",
                "service_tier": "priority",
                "output_schema": {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                },
                "config_overrides": {
                    "features.apps": False,
                    "web_search": "disabled",
                },
            },
            id="codex",
        ),
        pytest.param(
            "nvidia.fabric.langchain.deepagents",
            {
                "deepagents": {
                    "interrupt_on": {
                        "write_file": True,
                        "delete_file": {
                            "allowed_decisions": ["approve", "reject"],
                            "description": "Approve destructive changes.",
                            "args_schema": {"type": "object"},
                        },
                    },
                    "subagents": [
                        {
                            "name": "researcher",
                            "description": "Researches the requested topic.",
                            "system_prompt": "Return concise findings.",
                            "model": "openai:gpt-5.5",
                            "interrupt_on": {"write_file": True},
                            "skills": ["/skills/research"],
                            "response_format": {
                                "type": "object",
                                "properties": {"findings": {"type": "string"}},
                            },
                        },
                        {
                            "name": "remote-researcher",
                            "description": "Runs research asynchronously.",
                            "graph_id": "researcher",
                            "url": "https://agents.example.test",
                            "headers": {"X-Test": "value"},
                        },
                    ],
                }
            },
            id="deepagents",
        ),
        pytest.param(
            "nvidia.fabric.hermes",
            {
                "reasoning_config": {"enabled": True, "effort": "medium"},
                "plugins_enabled": ["custom/plugin"],
                "save_trajectories": True,
                "max_tokens": 1024,
                "terminal_timeout": 90,
            },
            id="hermes",
        ),
    ],
)
def test_repository_settings_are_validated_and_preserved(
    tmp_path: Path,
    adapter_id: str,
    settings: dict[str, Any],
):
    plan = Fabric().plan(
        _config(settings, adapter_id=adapter_id),
        base_dir=tmp_path,
    )

    assert plan.config.harness.settings == settings
    assert plan["adapter_descriptor"]["source"] == "repository"
    assert Path(plan["adapter_descriptor"]["path"]).samefile(
        ADAPTER_DESCRIPTORS[adapter_id]
    )


@pytest.mark.parametrize("adapter_id", ADAPTER_DESCRIPTORS)
def test_settings_schema_defaults_are_not_applied(
    tmp_path: Path,
    adapter_id: str,
):
    plan = Fabric().plan(
        _config({}, adapter_id=adapter_id),
        base_dir=tmp_path,
    )

    assert plan.config.harness.settings == {}


@pytest.mark.parametrize(
    ("adapter_id", "settings", "settings_path"),
    [
        pytest.param(
            "nvidia.fabric.claude",
            {"unknown": True},
            "harness.settings.unknown",
            id="claude-unknown",
        ),
        pytest.param(
            "nvidia.fabric.claude",
            {"python": True},
            "harness.settings.python",
            id="claude-legacy-python",
        ),
        pytest.param(
            "nvidia.fabric.claude",
            {"setting_sources": "project"},
            "harness.settings.setting_sources",
            id="claude-setting-sources-type",
        ),
        pytest.param(
            "nvidia.fabric.claude",
            {"setting_sources": ["project", "invalid"]},
            "harness.settings.setting_sources.1",
            id="claude-setting-sources-item",
        ),
        pytest.param(
            "nvidia.fabric.claude",
            {"max_budget_usd": "1"},
            "harness.settings.max_budget_usd",
            id="claude-budget-type",
        ),
        pytest.param(
            "nvidia.fabric.claude",
            {"max_budget_usd": 0},
            "harness.settings.max_budget_usd",
            id="claude-budget-range",
        ),
        pytest.param(
            "nvidia.fabric.claude",
            {"max_budget_usd": True},
            "harness.settings.max_budget_usd",
            id="claude-budget-boolean",
        ),
        pytest.param(
            "nvidia.fabric.claude",
            {"permission_mode": False},
            "harness.settings.permission_mode",
            id="claude-permission-mode-type",
        ),
        pytest.param(
            "nvidia.fabric.claude",
            {"permission_mode": "invalid"},
            "harness.settings.permission_mode",
            id="claude-permission-mode",
        ),
        pytest.param(
            "nvidia.fabric.codex",
            {"base_instructions": "legacy"},
            "harness.settings.base_instructions",
            id="codex-normalized-base-instructions",
        ),
        pytest.param(
            "nvidia.fabric.codex",
            {"sandbox": "invalid"},
            "harness.settings.sandbox",
            id="codex-sandbox",
        ),
        pytest.param(
            "nvidia.fabric.codex",
            {"approval_mode": "ask"},
            "harness.settings.approval_mode",
            id="codex-approval-mode",
        ),
        pytest.param(
            "nvidia.fabric.codex",
            {"developer_instructions": ""},
            "harness.settings.developer_instructions",
            id="codex-developer-instructions",
        ),
        pytest.param(
            "nvidia.fabric.codex",
            {"personality": "terse"},
            "harness.settings.personality",
            id="codex-personality",
        ),
        pytest.param(
            "nvidia.fabric.codex",
            {"reasoning_effort": False},
            "harness.settings.reasoning_effort",
            id="codex-reasoning-effort",
        ),
        pytest.param(
            "nvidia.fabric.codex",
            {"service_tier": ""},
            "harness.settings.service_tier",
            id="codex-service-tier",
        ),
        pytest.param(
            "nvidia.fabric.codex",
            {"output_schema": []},
            "harness.settings.output_schema",
            id="codex-output-schema",
        ),
        pytest.param(
            "nvidia.fabric.codex",
            {"config_overrides": {"features..apps": False}},
            "harness.settings.config_overrides",
            id="codex-config-override-key",
        ),
        pytest.param(
            "nvidia.fabric.langchain.deepagents",
            {"deepagents": []},
            "harness.settings.deepagents",
            id="deepagents-root-type",
        ),
        pytest.param(
            "nvidia.fabric.langchain.deepagents",
            {"deepagents": {"unknown": True}},
            "harness.settings.deepagents.unknown",
            id="deepagents-unknown",
        ),
        pytest.param(
            "nvidia.fabric.langchain.deepagents",
            {"deepagents": {"interrupt_on": {"write_file": "yes"}}},
            "harness.settings.deepagents.interrupt_on.write_file",
            id="deepagents-interrupt-policy",
        ),
        pytest.param(
            "nvidia.fabric.langchain.deepagents",
            {
                "deepagents": {
                    "interrupt_on": {"write_file": {"allowed_decisions": ["allow"]}}
                }
            },
            "harness.settings.deepagents.interrupt_on.write_file",
            id="deepagents-interrupt-decision",
        ),
        pytest.param(
            "nvidia.fabric.langchain.deepagents",
            {"deepagents": {"subagents": [{"name": "researcher"}]}},
            "harness.settings.deepagents.subagents.0",
            id="deepagents-subagent-required-fields",
        ),
        pytest.param(
            "nvidia.fabric.hermes",
            {"max_iterations": 4},
            "harness.settings.max_iterations",
            id="hermes-normalized-max-turns",
        ),
        pytest.param(
            "nvidia.fabric.hermes",
            {"reasoning_config": {"effort": "extreme"}},
            "harness.settings.reasoning_config.effort",
            id="hermes-reasoning-effort",
        ),
        pytest.param(
            "nvidia.fabric.hermes",
            {"reasoning_config": {"budget": 1024}},
            "harness.settings.reasoning_config.budget",
            id="hermes-reasoning-unknown",
        ),
        pytest.param(
            "nvidia.fabric.hermes",
            {"plugins_enabled": [""]},
            "harness.settings.plugins_enabled.0",
            id="hermes-plugin-id",
        ),
        pytest.param(
            "nvidia.fabric.hermes",
            {"save_trajectories": "yes"},
            "harness.settings.save_trajectories",
            id="hermes-save-trajectories",
        ),
        pytest.param(
            "nvidia.fabric.hermes",
            {"max_tokens": 0},
            "harness.settings.max_tokens",
            id="hermes-max-tokens",
        ),
        pytest.param(
            "nvidia.fabric.hermes",
            {"terminal_timeout": False},
            "harness.settings.terminal_timeout",
            id="hermes-terminal-timeout",
        ),
    ],
)
def test_invalid_settings_report_resolved_descriptor_and_path(
    tmp_path: Path,
    adapter_id: str,
    settings: dict[str, Any],
    settings_path: str,
):
    with pytest.raises(FabricConfigError) as caught:
        Fabric().plan(
            _config(settings, adapter_id=adapter_id),
            base_dir=tmp_path,
        )

    message = str(caught.value)
    assert adapter_id in message
    assert str(ADAPTER_DESCRIPTORS[adapter_id].resolve()) in message
    assert settings_path in message


def test_unknown_adapter_fails_before_settings_validation(tmp_path: Path):
    with pytest.raises(FabricConfigError) as caught:
        Fabric().plan(
            _config({"unknown": True}, adapter_id="test.fabric.missing"),
            base_dir=tmp_path,
        )

    message = str(caught.value)
    assert "unknown adapter `test.fabric.missing`" in message
    assert "invalid harness settings" not in message


def test_unknown_adapter_precedes_legacy_python_setting_type(tmp_path: Path):
    with pytest.raises(FabricConfigError) as caught:
        Fabric().plan(
            _config({"python": True}, adapter_id="test.fabric.missing"),
            base_dir=tmp_path,
        )

    message = str(caught.value)
    assert "unknown adapter `test.fabric.missing`" in message
    assert "expected path string" not in message


async def test_invalid_settings_fail_before_runtime_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime_start = MagicMock()
    monkeypatch.setattr(client_module._native, "start_runtime", runtime_start)

    with pytest.raises(FabricConfigError):
        await Fabric().start_runtime(
            _config({"unknown": True}),
            base_dir=tmp_path,
        )

    runtime_start.assert_not_called()
