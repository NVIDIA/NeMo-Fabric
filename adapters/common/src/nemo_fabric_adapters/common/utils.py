# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared adapter utility helpers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_relay import plugin
    from nemo_relay.observability import (
        AtifConfig,
        AtofConfig,
        HttpStorageConfig,
        OtlpConfig,
        S3StorageConfig,
    )


def effective_config(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("effective_config") or {}


def fabric_config(payload: dict[str, Any]) -> dict[str, Any]:
    return effective_config(payload).get("config") or {}


def config_root(payload: dict[str, Any]) -> str:
    return effective_config(payload).get("config_root") or payload.get("config_root") or "."


def agent_name(payload: dict[str, Any]) -> str:
    return effective_config(payload).get("agent_name") or payload.get("agent_name") or "fabric-agent"


def runtime_context(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("runtime_context") or {}


def runtime_session_id(payload: dict[str, Any]) -> str | None:
    """Return Fabric's session key for adapter-owned harness session mapping."""

    context = runtime_context(payload)
    session_id = context.get("session_id")
    if session_id:
        return str(session_id)
    runtime_id = context.get("runtime_id")
    if runtime_id:
        return str(runtime_id)
    return None


def environment_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return runtime_context(payload).get("environment") or payload.get("environment") or {}


def settings_payload(payload: dict[str, Any]) -> dict[str, Any]:
    harness = fabric_config(payload).get("harness") or {}
    return harness.get("settings") or payload.get("settings") or {}


def models_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return fabric_config(payload).get("models") or payload.get("models") or {}


def capability_plan(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("capability_plan") or payload.get("capabilities") or {}


def normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        value = [value]
    return [str(item) for item in value if str(item)]


def dump_yaml(value: dict[str, Any]) -> str:
    try:
        import yaml
        return yaml.safe_dump(value, sort_keys=False)
    except ImportError:
        return json.dumps(value, indent=2, sort_keys=False) + "\n"


def load_relay_plugin_config(payload: dict[str, Any]) -> dict[str, Any]:
    config_path = os.environ.get("FABRIC_RELAY_CONFIG_PATH")
    if not config_path:
        raise RuntimeError("FABRIC_RELAY_CONFIG_PATH is required when Relay is enabled")

    with Path(config_path).open(encoding="utf-8") as stream:
        wrapper = json.load(stream)

    relay = wrapper.get("relay", {})
    plugin_config = relay.get("config") or {}
    if "components" not in plugin_config:
        plugin_config = {
            "version": 1,
            "components": [
                {
                    "kind": "observability",
                    "enabled": True,
                    "config": plugin_config or {"version": 1},
                }
            ],
        }
    plugin_config.setdefault("version", 1)
    plugin_config.setdefault("components", [])
    normalize_relay_output_dirs(plugin_config, payload)
    return plugin_config


def normalize_relay_output_dirs(plugin_config: dict[str, Any], payload: dict[str, Any]) -> None:
    base = Path(config_root(payload)).resolve()
    for component in plugin_config.get("components", []):
        if component.get("kind") != "observability":
            continue
        config = component.setdefault("config", {})
        config.setdefault("version", 1)
        for section_name in ("atof", "atif"):
            section = config.get(section_name)
            if not isinstance(section, dict) or not section.get("enabled"):
                continue
            output_directory = section.get("output_directory")
            if output_directory:
                path = Path(output_directory)
                if not path.is_absolute():
                    section["output_directory"] = str(base / path)
            else:
                section["output_directory"] = str(base / "artifacts" / "relay")
            
            Path(section["output_directory"]).mkdir(parents=True, exist_ok=True)
            if section_name == "atof":
                section.setdefault("filename", "events.atof.jsonl")
                section.setdefault("mode", "overwrite")
            if section_name == "atif":
                section.setdefault("filename_template", "trajectory-{session_id}.atif.json")
                section.setdefault("agent_name", agent_name(payload))
                section.setdefault("model_name", _relay_model_name(payload))


def relay_api_plugin_config(plugin_config: dict[str, Any]) -> plugin.PluginConfig:
    from nemo_relay import plugin
    from nemo_relay.observability import (
        ComponentSpec,
        ConfigPolicy,
        ObservabilityConfig,
    )

    components: list[Any] = []
    for component in plugin_config.get("components", []):
        if not isinstance(component, dict):
            continue
        enabled = bool(component.get("enabled", True))
        config = component.get("config") or {}
        if component.get("kind") == "observability" and isinstance(config, dict):
            policy = config.get("policy") if isinstance(config.get("policy"), dict) else {}
            components.append(
                ComponentSpec(
                    ObservabilityConfig(
                        version=int(config.get("version", 1)),
                        atof=_relay_api_atof_config(config.get("atof")),
                        atif=_relay_api_atif_config(
                            config.get("atif"),
                        ),
                        opentelemetry=_relay_api_otlp_config(config.get("opentelemetry")),
                        openinference=_relay_api_otlp_config(config.get("openinference")),
                        policy=ConfigPolicy(
                            unknown_component=policy.get("unknown_component", "warn"),
                            unknown_field=policy.get("unknown_field", "warn"),
                            unsupported_value=policy.get("unsupported_value", "error"),
                        ),
                    ),
                    enabled=enabled,
                )
            )
            continue
        components.append(
            plugin.ComponentSpec(
                kind=str(component.get("kind") or ""),
                enabled=enabled,
                config=config if isinstance(config, dict) else {},
            )
        )

    policy = plugin_config.get("policy") if isinstance(plugin_config.get("policy"), dict) else {}
    plugin_config = plugin.PluginConfig(
        version=int(plugin_config.get("version", 1)),
        components=components,
        policy=plugin.ConfigPolicy(
            unknown_component=policy.get("unknown_component", "warn"),
            unknown_field=policy.get("unknown_field", "warn"),
            unsupported_value=policy.get("unsupported_value", "error"),
        ),
    )

    report = plugin.validate(plugin_config)
    if any(diagnostic["level"] == "error" for diagnostic in report["diagnostics"]):
        raise RuntimeError(report["diagnostics"])

    return plugin_config


def _relay_api_atof_config(value: Any) -> AtofConfig | None:
    if not isinstance(value, dict):
        return None
    from nemo_relay.observability import AtofConfig, AtofEndpointConfig

    endpoint_configs = value.get("endpoints")
    endpoints = None
    if isinstance(endpoint_configs, list):
        endpoints = [
            AtofEndpointConfig(
                url=str(endpoint.get("url", "")),
                transport=endpoint.get("transport", "http_post"),
                headers=endpoint.get("headers", {}),
                timeout_millis=int(endpoint.get("timeout_millis", 3000)),
            )
            for endpoint in endpoint_configs
            if isinstance(endpoint, dict)
        ]
    return AtofConfig(
        enabled=bool(value.get("enabled", False)),
        output_directory=value.get("output_directory"),
        filename=value.get("filename"),
        mode=value.get("mode", "append"),
        endpoints=endpoints,
    )


def _relay_api_atif_config(value: Any) -> AtifConfig | None:
    if not isinstance(value, dict):
        return None
    from nemo_relay.observability import AtifConfig

    storage_configs = value.get("storage")
    storage = None
    if isinstance(storage_configs, list):
        storage = [
            _relay_api_storage_config(item)
            for item in storage_configs
            if isinstance(item, dict)
        ]
    return AtifConfig(
        enabled=bool(value.get("enabled", False)),
        agent_name=value.get("agent_name", "NeMo Relay"),
        agent_version=value.get("agent_version"),
        model_name=value.get("model_name", "unknown"),
        tool_definitions=value.get("tool_definitions"),
        extra=value.get("extra"),
        output_directory=value.get("output_directory"),
        filename_template=value.get("filename_template", "nemo-relay-atif-{session_id}.json"),
        storage=storage,
    )


def _relay_api_storage_config(value: dict[str, Any]) -> HttpStorageConfig | S3StorageConfig:
    if value.get("type") == "s3":
        from nemo_relay.observability import S3StorageConfig

        return S3StorageConfig(
            bucket=value.get("bucket", ""),
            key_prefix=value.get("key_prefix"),
            access_key_id=value.get("access_key_id"),
            secret_access_key_var=value.get("secret_access_key_var"),
            session_token_var=value.get("session_token_var"),
            region=value.get("region"),
            endpoint_url=value.get("endpoint_url"),
            allow_http=value.get("allow_http"),
        )
    from nemo_relay.observability import HttpStorageConfig

    return HttpStorageConfig(
        endpoint=value.get("endpoint", ""),
        headers=value.get("headers", {}),
        header_env=value.get("header_env", {}),
        timeout_millis=int(value.get("timeout_millis", 3000)),
    )


def _relay_api_otlp_config(value: Any) -> OtlpConfig | None:
    if not isinstance(value, dict):
        return None
    from nemo_relay.observability import OtlpConfig

    return OtlpConfig(
        enabled=bool(value.get("enabled", False)),
        transport=value.get("transport", "http_binary"),
        endpoint=value.get("endpoint"),
        headers=value.get("headers", {}),
        resource_attributes=value.get("resource_attributes", {}),
        service_name=value.get("service_name", "nemo-relay"),
        service_namespace=value.get("service_namespace"),
        service_version=value.get("service_version"),
        instrumentation_scope=value.get("instrumentation_scope"),
        timeout_millis=int(value.get("timeout_millis", 3000)),
    )


def collect_relay_artifacts(plugin_config: dict[str, Any]) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    for component in plugin_config.get("components", []):
        if component.get("kind") != "observability":
            continue
        config = component.get("config") or {}
        for section_name, pattern in (
            ("atof", "*.jsonl"),
            ("atif", "*.json"),
        ):
            section = config.get(section_name)
            if not isinstance(section, dict) or not section.get("enabled"):
                continue
            directory = Path(section.get("output_directory") or ".")
            if not directory.exists():
                continue
            for path in sorted(directory.glob(pattern)):
                artifacts.append({"kind": section_name, "path": str(path)})
    return artifacts


def write_relay_configs(
    *,
    relay_config: dict[str, Any] | None = None,
    plugin_config: dict[str, Any] | None = None,
) -> tuple[Path | None, Path | None]:
    try:
        import tomli_w

        config_path = os.environ.get("FABRIC_RELAY_CONFIG_PATH")
        if not config_path:
            raise RuntimeError("FABRIC_RELAY_CONFIG_PATH is required when Relay is enabled")

        config_path = Path(config_path)
        config_dir = config_path.parent / "relay-config"
        config_dir.mkdir(parents=True, exist_ok=True)
        relay_config_path = None
        plugin_config_path = None

        if relay_config is not None:
            relay_config_path = config_dir / "config.toml"
            relay_config_path.write_text(tomli_w.dumps(relay_config), encoding="utf-8")

        if plugin_config is not None:
            plugin_config_path = config_dir / "plugins.toml"
            plugin_config_path.write_text(tomli_w.dumps(plugin_config), encoding="utf-8")

        return relay_config_path, plugin_config_path
    except ImportError:
        print("tomli_w is not installed, skipping writing Relay TOML", file=sys.stderr)
        return None, None


def _relay_model_name(payload: dict[str, Any]) -> str:
    settings = settings_payload(payload)
    models = models_payload(payload)
    model_config = models.get(settings.get("model", "default"), {})
    return settings.get("model_name") or model_config.get("model") or "unknown"
