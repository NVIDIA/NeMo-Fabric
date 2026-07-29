# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free tests for Hermes configuration construction."""

import builtins
import json
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "Hermes adapter requires Python 3.13 or earlier",
        allow_module_level=True,
    )

from nemo_fabric_adapters.hermes import adapter


def test_build_hermes_config_omits_unset_values_without_hermes_agent():
    payload = {
        "config": {
            "harness": {"settings": {}},
            "models": {
                "default": {
                    "provider": "nvidia",
                    "model": "nvidia/test-model",
                }
            },
        }
    }

    config = adapter.build_hermes_config(payload)

    assert config["model"] == {
        "provider": "nvidia",
        "default": "nvidia/test-model",
    }
    assert config["agent"] == {}


def test_write_hermes_config_round_trips_without_pyyaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    real_import = builtins.__import__

    def import_without_yaml(name: str, *args: object, **kwargs: object) -> object:
        if name == "yaml":
            raise ImportError("No module named yaml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_yaml)
    payload = {
        "config": {
            "harness": {"settings": {}},
            "models": {"default": {"provider": "nvidia", "model": "nvidia/test-model"}},
        }
    }

    config_path, config = adapter.write_hermes_config(payload, tmp_path / "hermes-home")

    assert json.loads(config_path.read_text(encoding="utf-8")) == config
