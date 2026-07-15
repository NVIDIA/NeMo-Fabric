# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared adapter utility helpers."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from nemo_relay import plugin
    from nemo_relay.observability import AtifConfig
    from nemo_relay.observability import AtofConfig
    from nemo_relay.observability import AtofFileSinkConfig
    from nemo_relay.observability import AtofStreamSinkConfig
    from nemo_relay.observability import HttpStorageConfig
    from nemo_relay.observability import OtlpConfig
    from nemo_relay.observability import S3StorageConfig


def current_virtualenv() -> Path | None:
    """Return the current virtual environment, if Python is running in one."""

    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        return None
    return Path(sys.prefix)


def virtualenv_subprocess_env() -> dict[str, str]:
    """
    When inside of a virtual environment, return a copy of os.environ with the virtualenv exposed.

    When outside of a virtual environment a copy of os.environ is returned.
    """

    env = os.environ.copy()
    virtualenv = current_virtualenv()
    if virtualenv is None:
        return env

    scripts = virtualenv / ("Scripts" if os.name == "nt" else "bin")
    path = env.get("PATH")
    env["VIRTUAL_ENV"] = str(virtualenv)
    env["PATH"] = os.pathsep.join(part for part in (str(scripts), path) if part)
    env.pop("PYTHONHOME", None)
    return env


def request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("request") or {}


def effective_config(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("effective_config") or {}


def fabric_config(payload: dict[str, Any]) -> dict[str, Any]:
    return effective_config(payload).get("config") or {}


def config_root(payload: dict[str, Any]) -> str:
    return (
        effective_config(payload).get("config_root")
        or payload.get("config_root")
        or "."
    )


def agent_name(payload: dict[str, Any]) -> str:
    return (
        effective_config(payload).get("agent_name")
        or payload.get("agent_name")
        or "fabric-agent"
    )


def load_payload() -> dict[str, Any]:
    invocation_path = os.environ.get("FABRIC_INVOCATION")
    if invocation_path:
        path = Path(invocation_path)
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return json.load(sys.stdin)


def runtime_context(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("runtime_context") or {}


def runtime_id(payload: dict[str, Any]) -> str:
    """Return the Fabric runtime id used to key adapter-owned state."""

    value = runtime_context(payload).get("runtime_id")
    if not value:
        raise ValueError("runtime_context.runtime_id is required")
    return str(value)


def runtime_state_directory(base: str | Path, payload: dict[str, Any]) -> Path:
    """Return a harness-owned state directory isolated to one Fabric runtime."""

    return Path(base).joinpath("runtimes", runtime_id(payload))


def environment_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return (
        runtime_context(payload).get("environment") or payload.get("environment") or {}
    )


def settings_payload(payload: dict[str, Any]) -> dict[str, Any]:
    harness = fabric_config(payload).get("harness") or {}
    return harness.get("settings") or payload.get("settings") or {}


def models_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return fabric_config(payload).get("models") or payload.get("models") or {}


def default_base_url(provider: str | None) -> str | None:
    if provider == "nvidia":
        return "https://integrate.api.nvidia.com/v1"
    return None


def get_base_url(settings: dict[str, Any], model_config: dict[str, Any]) -> str | None:
    return (
        settings.get("base_url")
        or (model_config.get("settings") or {}).get("base_url")
        or default_base_url(model_config.get("provider"))
    )


def selected_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    settings = settings_payload(payload)
    models = models_payload(payload)
    model_config = models.get(settings.get("model", "default"), {})
    if not isinstance(model_config, dict):
        return {}
    return model_config


def telemetry_payload(payload: dict[str, Any]) -> dict[str, Any]:
    telemetry = (
        fabric_config(payload).get("telemetry") or payload.get("telemetry") or {}
    )
    return telemetry if isinstance(telemetry, dict) else {}


def telemetry_plan(payload: dict[str, Any]) -> dict[str, Any]:
    plan = payload.get("telemetry_plan") or {}
    return plan if isinstance(plan, dict) else {}


def telemetry_providers(payload: dict[str, Any]) -> list[str]:
    providers = telemetry_plan(payload).get("providers")
    if isinstance(providers, list):
        return [str(provider) for provider in providers if str(provider)]
    return []


def relay_enabled(payload: dict[str, Any]) -> bool:
    return telemetry_plan(payload).get("relay_enabled") is True


def native_telemetry_config(payload: dict[str, Any]) -> dict[str, Any]:
    config = telemetry_plan(payload).get("native_config") or {}
    return config if isinstance(config, dict) else {}


def capability_plan(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("capability_plan") or payload.get("capabilities") or {}


def tools_config(payload: dict[str, Any]) -> dict[str, Any]:
    tools = fabric_config(payload).get("tools") or {}
    return tools if isinstance(tools, dict) else {}


def blocked_tools(payload: dict[str, Any]) -> list[str]:
    blocked = tools_config(payload).get("blocked")
    return normalize_list(blocked)


def normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        value = [value]
    return [str(item) for item in value if str(item)]


def merge_unique(*values: Any) -> list[str]:
    merged: list[str] = []
    for value in values:
        for item in normalize_list(value):
            if item not in merged:
                merged.append(item)
    return merged


def without_none(mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if value is not None}


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
                    "config": plugin_config or {"version": 2},
                }
            ],
        }
    plugin_config.setdefault("version", 1)
    plugin_config.setdefault("components", [])
    normalize_relay_output_dirs(plugin_config, payload)
    return plugin_config


def load_relay_dynamic_plugins(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Load ordered dynamic plugin specs and resolve invocation-relative paths."""

    config_path = os.environ.get("FABRIC_RELAY_CONFIG_PATH")
    if not config_path:
        raise RuntimeError("FABRIC_RELAY_CONFIG_PATH is required when Relay is enabled")

    with Path(config_path).open(encoding="utf-8") as stream:
        wrapper = json.load(stream)

    specs = (wrapper.get("relay") or {}).get("dynamic_plugins") or []
    if not isinstance(specs, list):
        raise ValueError("relay.dynamic_plugins must be a list")
    root = Path(config_root(payload)).resolve()
    resolved: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise ValueError(f"relay.dynamic_plugins[{index}] must be an object")
        item = copy.deepcopy(spec)
        for field in ("manifest_ref", "environment_ref"):
            value = item.get(field)
            if value is None:
                continue
            path = Path(str(value))
            item[field] = str(path if path.is_absolute() else (root / path).resolve())
        resolved.append(item)
    return resolved


def normalize_relay_output_dirs(
    plugin_config: dict[str, Any], payload: dict[str, Any]
) -> None:
    base = Path(config_root(payload)).resolve()
    runtime_id = runtime_context(payload)["runtime_id"]
    for component in plugin_config.get("components", []):
        if component.get("kind") != "observability":
            continue
        config = component.setdefault("config", {})
        version = int(config.setdefault("version", 2))
        if version != 2:
            raise ValueError("NeMo Relay observability config version 2 is required")

        atof = config.get("atof")
        if isinstance(atof, dict) and atof.get("enabled"):
            sinks = atof.setdefault("sinks", [])
            if not isinstance(sinks, list):
                raise ValueError("Relay ATOF sinks must be a list")
            if not any(
                isinstance(sink, dict) and sink.get("type") == "file" for sink in sinks
            ):
                sinks.append({"type": "file"})
            for sink in sinks:
                if not isinstance(sink, dict) or sink.get("type") != "file":
                    continue
                output_directory = sink.get("output_directory")
                path = (
                    Path(output_directory)
                    if output_directory
                    else base / "artifacts" / "relay"
                )
                if not path.is_absolute():
                    path = base / path
                sink["output_directory"] = str(path / str(runtime_id))
                Path(sink["output_directory"]).mkdir(parents=True, exist_ok=True)
                sink.setdefault("filename", "events.atof.jsonl")
                sink.setdefault("mode", "overwrite")

        atif = config.get("atif")
        if isinstance(atif, dict) and atif.get("enabled"):
            output_directory = atif.get("output_directory")
            path = (
                Path(output_directory)
                if output_directory
                else base / "artifacts" / "relay"
            )
            if not path.is_absolute():
                path = base / path
            atif["output_directory"] = str(path / str(runtime_id))
            Path(atif["output_directory"]).mkdir(parents=True, exist_ok=True)
            atif.setdefault("filename_template", "trajectory-{session_id}.atif.json")
            atif.setdefault("agent_name", agent_name(payload))
            atif.setdefault("model_name", relay_model_name(payload))


def relay_api_plugin_config(plugin_config: dict[str, Any]) -> plugin.PluginConfig:
    from nemo_relay import plugin
    from nemo_relay.observability import ComponentSpec
    from nemo_relay.observability import ConfigPolicy
    from nemo_relay.observability import ObservabilityConfig

    components: list[Any] = []
    for component in plugin_config.get("components", []):
        if not isinstance(component, dict):
            continue
        enabled = bool(component.get("enabled", True))
        config = component.get("config") or {}
        if component.get("kind") == "observability" and isinstance(config, dict):
            policy = (
                config.get("policy") if isinstance(config.get("policy"), dict) else {}
            )
            components.append(
                ComponentSpec(
                    ObservabilityConfig(
                        version=int(config.get("version", 2)),
                        atof=_relay_api_atof_config(config.get("atof")),
                        atif=_relay_api_atif_config(
                            config.get("atif"),
                        ),
                        opentelemetry=_relay_api_otlp_config(
                            config.get("opentelemetry")
                        ),
                        openinference=_relay_api_otlp_config(
                            config.get("openinference")
                        ),
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

    policy = (
        plugin_config.get("policy")
        if isinstance(plugin_config.get("policy"), dict)
        else {}
    )
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


def relay_api_dynamic_plugins(
    specs: list[dict[str, Any]],
) -> list[plugin.DynamicPluginActivationSpec]:
    """Convert Fabric's typed dynamic plugin specs to Relay's owned host API."""

    from nemo_relay import plugin

    return [
        plugin.DynamicPluginActivationSpec(
            plugin_id=str(spec["plugin_id"]),
            kind=spec["kind"],
            manifest_ref=str(spec["manifest_ref"]),
            environment_ref=(
                str(spec["environment_ref"])
                if spec.get("environment_ref") is not None
                else None
            ),
            config=spec.get("config") or {},
        )
        for spec in specs
    ]


def _relay_api_atof_config(value: Any) -> AtofConfig | None:
    if not isinstance(value, dict):
        return None
    from nemo_relay.observability import AtofConfig
    from nemo_relay.observability import AtofFileSinkConfig
    from nemo_relay.observability import AtofStreamSinkConfig

    sinks: list[AtofFileSinkConfig | AtofStreamSinkConfig] = []
    for sink in value.get("sinks") or []:
        if not isinstance(sink, dict):
            continue
        if sink.get("type") == "file":
            sinks.append(
                AtofFileSinkConfig(
                    output_directory=sink.get("output_directory"),
                    filename=sink.get("filename"),
                    mode=sink.get("mode", "append"),
                )
            )
        elif sink.get("type") == "stream":
            sinks.append(
                AtofStreamSinkConfig(
                    name=sink.get("name"),
                    url=str(sink.get("url", "")),
                    transport=sink.get("transport", "http_post"),
                    headers=sink.get("headers", {}),
                    header_env=sink.get("header_env", {}),
                    timeout_millis=int(sink.get("timeout_millis", 3000)),
                    field_name_policy=sink.get("field_name_policy", "preserve"),
                )
            )
    return AtofConfig(
        enabled=bool(value.get("enabled", False)),
        sinks=sinks,
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
        filename_template=value.get(
            "filename_template", "nemo-relay-atif-{session_id}.json"
        ),
        storage=storage,
    )


def _relay_api_storage_config(
    value: dict[str, Any],
) -> HttpStorageConfig | S3StorageConfig:
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
        mark_projection=value.get("mark_projection", "inherit"),
        mark_exclude_names=value.get("mark_exclude_names", ["llm.chunk"]),
        transport=value.get("transport", "http_binary"),
        endpoint=value.get("endpoint"),
        headers=value.get("headers", {}),
        resource_attributes=value.get("resource_attributes", {}),
        service_name=value.get("service_name", "nemo-relay"),
        service_namespace=value.get("service_namespace"),
        service_version=value.get("service_version"),
        instrumentation_scope=value.get("instrumentation_scope"),
        timeout_millis=int(value.get("timeout_millis", 3000)),
        attribute_mappings=value.get("attribute_mappings", []),
    )


def collect_relay_artifacts(plugin_config: dict[str, Any]) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    for component in plugin_config.get("components", []):
        if component.get("kind") != "observability":
            continue
        config = component.get("config") or {}
        atof = config.get("atof")
        if isinstance(atof, dict) and atof.get("enabled"):
            for sink in atof.get("sinks") or []:
                if not isinstance(sink, dict) or sink.get("type") != "file":
                    continue
                directory = Path(sink.get("output_directory") or ".")
                if directory.exists():
                    filename = sink.get("filename")
                    paths = (
                        [directory / str(filename)]
                        if filename
                        else sorted(directory.glob("*.jsonl"))
                    )
                    for path in paths:
                        if not path.is_file():
                            continue
                        artifacts.append({"kind": "atof", "path": str(path)})
        atif = config.get("atif")
        if isinstance(atif, dict) and atif.get("enabled"):
            directory = Path(atif.get("output_directory") or ".")
            if directory.exists():
                for path in sorted(directory.glob("*.json")):
                    artifacts.append({"kind": "atif", "path": str(path)})
    return artifacts


def relay_cli_plugin_config(
    plugin_config: dict[str, Any], *, observability_version: int
) -> dict[str, Any]:
    """Render normalized Relay intent for the current external CLI contract."""

    rendered = copy.deepcopy(plugin_config)
    if observability_version != 2:
        raise ValueError(
            "NeMo Relay 0.6 or newer is required; observability version 1 is unsupported"
        )
    for component in rendered.get("components", []):
        if not isinstance(component, dict) or component.get("kind") != "observability":
            continue
        config = component.get("config")
        if isinstance(config, dict) and int(config.get("version", 2)) != 2:
            raise ValueError("NeMo Relay observability config version 2 is required")
    return rendered


def write_relay_configs(
    *,
    relay_config: dict[str, Any] | None = None,
    plugin_config: dict[str, Any] | None = None,
    observability_version: int = 2,
) -> tuple[Path | None, Path | None]:
    try:
        import tomli_w

        config_path = os.environ.get("FABRIC_RELAY_CONFIG_PATH")
        if not config_path:
            raise RuntimeError(
                "FABRIC_RELAY_CONFIG_PATH is required when Relay is enabled"
            )

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
            plugin_config_path.write_text(
                tomli_w.dumps(
                    relay_cli_plugin_config(
                        plugin_config,
                        observability_version=observability_version,
                    )
                ),
                encoding="utf-8",
            )

        return relay_config_path, plugin_config_path
    except ImportError as e:
        raise RuntimeError("tomli_w is not installed") from e


def provision_relay_dynamic_plugins(
    *,
    executable: Path,
    relay_config_path: Path,
    plugin_config_path: Path,
    specs: list[dict[str, Any]],
    env: dict[str, str],
    cwd: Path,
) -> list[dict[str, Any]]:
    """Provision dynamic plugins through Relay's invocation-scoped CLI lifecycle."""

    receipts: list[dict[str, Any]] = []
    for spec in specs:
        plugin_id = str(spec["plugin_id"])
        manifest_ref = str(spec["manifest_ref"])
        if not _relay_plugin_manifest_registered(plugin_config_path, manifest_ref):
            _run_relay_lifecycle_command(
                executable,
                relay_config_path,
                ["plugins", "add", manifest_ref],
                env=env,
                cwd=cwd,
            )
        _attach_relay_dynamic_plugin_config(
            plugin_config_path,
            manifest_ref,
            spec.get("config") or {},
        )
        _run_relay_lifecycle_command(
            executable,
            relay_config_path,
            ["plugins", "enable", plugin_id],
            env=env,
            cwd=cwd,
        )
        _run_relay_lifecycle_command(
            executable,
            relay_config_path,
            ["plugins", "validate", plugin_id, "--json"],
            env=env,
            cwd=cwd,
        )
        receipts.append(
            {
                "plugin_id": plugin_id,
                "kind": str(spec["kind"]),
                "registered": True,
                "enabled": True,
                "validated": True,
            }
        )
    return receipts


def _run_relay_lifecycle_command(
    executable: Path,
    relay_config_path: Path,
    args: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
) -> None:
    completed = subprocess.run(
        [str(executable), "--config", str(relay_config_path), *args],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        action = " ".join(args[:2])
        raise RuntimeError(
            f"NeMo Relay {action} failed with status {completed.returncode}"
        )


def _relay_plugin_manifest_registered(
    plugin_config_path: Path, manifest_ref: str
) -> bool:
    if not plugin_config_path.is_file():
        return False
    document = tomllib.loads(plugin_config_path.read_text(encoding="utf-8"))
    entries = (document.get("plugins") or {}).get("dynamic") or []
    target = Path(manifest_ref).resolve()
    return any(
        isinstance(entry, dict)
        and Path(str(entry.get("manifest", ""))).resolve() == target
        for entry in entries
    )


def _attach_relay_dynamic_plugin_config(
    plugin_config_path: Path,
    manifest_ref: str,
    config: dict[str, Any],
) -> None:
    import tomli_w

    document = tomllib.loads(plugin_config_path.read_text(encoding="utf-8"))
    entries = (document.get("plugins") or {}).get("dynamic") or []
    target = Path(manifest_ref).resolve()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if Path(str(entry.get("manifest", ""))).resolve() == target:
            entry["config"] = copy.deepcopy(config)
            plugin_config_path.write_text(tomli_w.dumps(document), encoding="utf-8")
            return
    raise RuntimeError(
        "Relay lifecycle did not register the requested dynamic plugin manifest"
    )


def relay_model_name(payload: dict[str, Any]) -> str:
    settings = settings_payload(payload)
    models = models_payload(payload)
    model_config = models.get(settings.get("model", "default"), {})
    return settings.get("model_name") or model_config.get("model") or "unknown"
