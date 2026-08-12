# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the LangGraph custom-agent example tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest


@pytest.fixture(name="runtime_context_factory")
def runtime_context_factory_fixture() -> Callable[[str, str], dict[str, Any]]:
    def build(runtime_id: str, invocation_id: str) -> dict[str, Any]:
        return {
            "runtime_id": runtime_id,
            "invocation_id": invocation_id,
            "request_id": f"request-{invocation_id}",
            "environment": {
                "environment_id": "environment-1",
                "provider": "local",
                "control_location": "in_env_control",
                "ownership": "caller_owned",
            },
            "artifacts": {},
        }

    return build


@pytest.fixture(name="agent_config_mapping")
def agent_config_mapping_fixture() -> dict[str, Any]:
    return {
        "models": {
            "default": {
                "provider": "nvidia",
                "model": "nvidia/test-model",
                "api_key_env": "TEST_API_KEY",
                "base_url": "https://example.test/v1",
            }
        }
    }


@pytest.fixture(name="lifecycle_request_factory")
def lifecycle_request_factory_fixture() -> Callable[
    [str, dict[str, Any]], dict[str, Any]
]:
    def build(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"operation": operation, "payload": payload}

    return build
