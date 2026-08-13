# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare Hermes' optional NeMo Relay integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nemo_fabric_adapter_contract.models import RuntimeContext
import nemo_fabric_adapters.common.utils as common_utils


# Hermes 0.16+ discovers Relay from this TOML path and falls back to direct
# ATIF/ATOF only when TOML initialization fails. Clear only those enable flags.
HERMES_RELAY_ENV_NAMES = (
    "HERMES_NEMO_RELAY_PLUGINS_TOML",
    "HERMES_NEMO_RELAY_ATIF_ENABLED",
    "HERMES_NEMO_RELAY_ATOF_ENABLED",
)


def finalize_hermes_relay_session(session_id: str) -> None:
    """Finalize one Relay session through the installed Hermes lifecycle API."""
    try:
        from hermes_cli.lifecycle import finalize_session
    except ModuleNotFoundError as error:
        if error.name != "hermes_cli.lifecycle":
            raise
        # Hermes 0.19 exposes the same finalization boundary as a plugin hook.
        from hermes_cli.plugins import invoke_hook

        invoke_hook("on_session_finalize", session_id=session_id, platform="fabric")
    else:
        finalize_session(session_id=session_id, platform="fabric")


def validate_hermes_telemetry_provider(runtime_context: RuntimeContext) -> None:
    telemetry = runtime_context.telemetry
    providers = telemetry.metadata.get("telemetry_providers", []) if telemetry else []
    if any(provider != "relay" for provider in providers):
        raise ValueError("only relay telemetry is supported for Hermes")


def write_hermes_relay_plugin_config(
    payload: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Stage Fabric's resolved Relay config for Hermes' bundled integration."""

    common_utils.reject_ambient_relay_plugin_config()
    hermes_plugin_config = common_utils.load_relay_plugin_config(payload)
    for component in hermes_plugin_config.get("components", []):
        if component.get("kind") != "observability":
            continue
        observability = component.get("config")
        if not isinstance(observability, dict):
            continue

        # Fabric finalizes Hermes' Relay session after every invocation. Each
        # finalization reinitializes Relay for the next turn, so a file sink
        # cannot overwrite the runtime-scoped artifact it created previously.
        for sink in (observability.get("atof") or {}).get("sinks") or []:
            if isinstance(sink, dict) and sink.get("type") == "file":
                if sink.get("mode") == "overwrite":
                    sink["mode"] = "append"
    _, plugin_config_path = common_utils.write_relay_configs(
        plugin_config=hermes_plugin_config
    )
    if plugin_config_path is None:
        raise RuntimeError("Hermes Relay plugin configuration was not generated")
    return plugin_config_path, hermes_plugin_config
