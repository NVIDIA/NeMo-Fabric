# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planning validation for descriptor-owned harness settings schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import nemo_fabric.client as client_module
import pytest
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import FabricConfigError


ROOT = Path(__file__).resolve().parents[2]
CLAUDE_DESCRIPTOR = ROOT / "adapters" / "claude" / "fabric-adapter.json"


def _config(
    settings: dict[str, Any] | None = None,
    *,
    adapter_id: str = "nvidia.fabric.claude",
) -> FabricConfig:
    return FabricConfig.from_mapping(
        {
            "metadata": {"name": "harness-settings-validation"},
            "harness": {
                "adapter_id": adapter_id,
                "resolution": "preinstalled",
                "settings": settings or {},
            },
        }
    )


def test_repository_claude_settings_are_validated_and_preserved(tmp_path: Path):
    settings = {
        "setting_sources": ["user", "project", "local"],
        "max_budget_usd": 1.5,
        "permission_mode": "dontAsk",
    }

    plan = Fabric().plan(_config(settings), base_dir=tmp_path)

    assert plan.config.harness.settings == settings
    assert plan["adapter_descriptor"]["source"] == "repository"
    assert Path(plan["adapter_descriptor"]["path"]).samefile(CLAUDE_DESCRIPTOR)


def test_claude_settings_schema_defaults_are_not_applied(tmp_path: Path):
    plan = Fabric().plan(
        _config({"permission_mode": "default"}),
        base_dir=tmp_path,
    )

    assert plan.config.harness.settings == {"permission_mode": "default"}
    assert "setting_sources" not in plan.config.harness.settings


@pytest.mark.parametrize(
    ("settings", "settings_path"),
    [
        ({"unknown": True}, "harness.settings.unknown"),
        ({"python": True}, "harness.settings.python"),
        ({"setting_sources": "project"}, "harness.settings.setting_sources"),
        (
            {"setting_sources": ["project", "invalid"]},
            "harness.settings.setting_sources.1",
        ),
        ({"max_budget_usd": "1"}, "harness.settings.max_budget_usd"),
        ({"max_budget_usd": 0}, "harness.settings.max_budget_usd"),
        ({"max_budget_usd": True}, "harness.settings.max_budget_usd"),
        ({"permission_mode": False}, "harness.settings.permission_mode"),
        ({"permission_mode": "invalid"}, "harness.settings.permission_mode"),
    ],
)
def test_invalid_claude_settings_report_resolved_descriptor_and_path(
    tmp_path: Path,
    settings: dict[str, Any],
    settings_path: str,
):
    with pytest.raises(FabricConfigError) as caught:
        Fabric().plan(_config(settings), base_dir=tmp_path)

    message = str(caught.value)
    assert "nvidia.fabric.claude" in message
    assert str(CLAUDE_DESCRIPTOR.resolve()) in message
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
