# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free tests for Hermes configuration construction."""

import sys

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
        "base_url": "https://integrate.api.nvidia.com/v1",
    }
    assert config["agent"] == {}
