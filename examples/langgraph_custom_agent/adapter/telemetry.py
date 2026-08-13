# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional Relay boundary for one custom-agent invocation."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import AsyncIterator

from langchain_core.runnables import RunnableConfig
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common import utils as common_utils


@dataclass(slots=True)
class InvocationTelemetry:
    """Runnable config and artifacts for one Relay-enabled invocation."""

    runnable_config: RunnableConfig | None = None
    plugin_config: dict[str, Any] | None = None

    def artifacts(self) -> list[dict[str, str]]:
        if self.plugin_config is None:
            return []
        return common_utils.collect_relay_artifacts(self.plugin_config)


def _telemetry_error(code: str, message: str) -> lifecycle.LifecycleError:
    return lifecycle.LifecycleError(code, message)


def _load_plugin_config(
    context: RuntimeContext,
    *,
    base_dir: Path,
    agent_name: str,
    model_name: str,
) -> dict[str, Any]:
    telemetry = context.telemetry
    if telemetry is None or telemetry.config_path is None:
        raise _telemetry_error(
            "email_phishing_relay_config_missing",
            "Relay is enabled but RuntimeContext has no Relay configuration path",
        )
    try:
        wrapper = json.loads(Path(telemetry.config_path).read_text(encoding="utf-8"))
        plugin_config = wrapper.get("relay", {}).get("config") or {}
        if "components" not in plugin_config:
            plugin_config = {
                "version": 1,
                "components": [
                    {
                        "kind": "observability",
                        "enabled": True,
                        "config": plugin_config or {"version": 3},
                    }
                ],
            }
        plugin_config.setdefault("version", 1)
        plugin_config.setdefault("components", [])
    except (AttributeError, json.JSONDecodeError, OSError, TypeError) as error:
        raise _telemetry_error(
            "email_phishing_relay_config_invalid",
            "Relay configuration could not be loaded",
        ) from error

    common_utils.normalize_relay_output_dirs(
        plugin_config,
        {
            "agent_name": agent_name,
            "base_dir": str(base_dir),
            "config": {"models": {"default": {"model": model_name}}},
            "runtime_context": context.to_mapping(),
        },
    )
    for component in plugin_config["components"]:
        if component.get("kind") != "observability":
            continue
        atif = component.get("config", {}).get("atif")
        if isinstance(atif, dict) and atif.get("enabled"):
            atif["model_name"] = model_name
    return plugin_config


def _relay_api() -> tuple[Any, Any, Any, Any]:
    try:
        from nemo_relay import ScopeType
        from nemo_relay import plugin
        from nemo_relay import scope
        from nemo_relay.integrations.langgraph import NemoRelayCallbackHandler
    except (ImportError, AttributeError) as error:
        raise _telemetry_error(
            "email_phishing_relay_unavailable",
            "Relay telemetry requires a compatible nemo-relay installation",
        ) from error
    return plugin, scope, ScopeType, NemoRelayCallbackHandler


@asynccontextmanager
async def observe_invocation(
    context: RuntimeContext,
    *,
    base_dir: Path,
    agent_name: str,
    model_name: str,
) -> AsyncIterator[InvocationTelemetry]:
    """Activate Relay when NeMo Fabric supplies an enabled telemetry context."""

    telemetry = context.telemetry
    if telemetry is None or not telemetry.relay_enabled:
        yield InvocationTelemetry()
        return

    plugin_config = _load_plugin_config(
        context,
        base_dir=base_dir,
        agent_name=agent_name,
        model_name=model_name,
    )
    common_utils.reject_ambient_relay_plugin_config()
    plugin, scope, scope_type, callback_handler = _relay_api()
    observation = InvocationTelemetry(
        runnable_config={"callbacks": [callback_handler()]},
        plugin_config=plugin_config,
    )
    async with plugin.plugin(plugin_config) as activation_report:
        common_utils.reject_inherited_relay_plugin_config(activation_report)
        with scope.scope(
            "email-phishing-invocation",
            scope_type.Agent,
            metadata={
                "nemo_fabric_runtime_id": context.runtime_id,
                "nemo_fabric_invocation_id": context.invocation_id,
                "nemo_fabric_request_id": context.request_id,
            },
        ):
            yield observation
