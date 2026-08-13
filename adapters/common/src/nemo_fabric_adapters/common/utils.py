# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared adapter utility helpers."""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


_FIELD_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")


def validate_http_header(server_name: str, name: str, value: str) -> None:
    """Validate one HTTP header name and value."""

    if not isinstance(name, str) or not _FIELD_NAME.fullmatch(name):
        raise ValueError(
            f"Invalid HTTP header name {name!r} for MCP server {server_name!r}"
        )

    if not isinstance(value, str):
        raise TypeError(
            f"HTTP header value for {name!r} on MCP server {server_name!r} "
            "must be a string"
        )

    if not value or not value.strip():
        raise ValueError(
            f"HTTP header value for {name!r} on MCP server {server_name!r} "
            "must not be blank"
        )

    try:
        encoded = value.encode("latin-1")
    except UnicodeEncodeError as error:
        raise ValueError(
            f"HTTP header value for {name!r} on MCP server {server_name!r} "
            "is not Latin-1 encodable"
        ) from error

    if value[:1] in (" ", "\t") or value[-1:] in (" ", "\t"):
        raise ValueError(
            f"HTTP header value for {name!r} on MCP server {server_name!r} "
            "has outer whitespace"
        )

    if any((byte < 0x20 and byte != 0x09) or byte == 0x7F for byte in encoded):
        raise ValueError(
            f"HTTP header value for {name!r} on MCP server {server_name!r} "
            "contains a control character"
        )


def validate_http_headers(server_name: str, value: dict[str, str]) -> None:
    """
    Validate an MCP custom-header mapping.

    Use this method for harnesses that support environment variable expansion
    in HTTP headers. For harnesses that don't support environment variable
    expansion, use expand_http_headers instead.
    """

    for name, item in value.items():
        validate_http_header(server_name, name, item)


def expand_http_headers(server_name: str, value: dict[str, str]) -> dict[str, str]:
    """
    Expand environment variables and validate an MCP custom-header mapping.

    Use this method instead of validate_http_headers for harnesses that don't
    support environment variable expansion in HTTP headers.
    """

    expanded: dict[str, str] = {}
    for name, item in value.items():
        item = os.path.expandvars(item)
        validate_http_header(server_name, name, item)
        expanded[name] = item

    return expanded


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


def fabric_config(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("config") or {}


def base_dir(payload: dict[str, Any]) -> str:
    value = payload.get("base_dir")
    if not isinstance(value, str) or not value:
        raise ValueError("base_dir is required")
    if not Path(value).is_absolute():
        raise ValueError("base_dir must be an absolute path")
    return value


def agent_name(payload: dict[str, Any]) -> str:
    return payload.get("agent_name") or "fabric-agent"


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
    """Return the NeMo Fabric runtime id used to key adapter-owned state."""

    value = runtime_context(payload).get("runtime_id")
    if not value:
        raise ValueError("runtime_context.runtime_id is required")
    return str(value)


def environment_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return (
        runtime_context(payload).get("environment")
        or fabric_config(payload).get("environment")
        or payload.get("environment")
        or {}
    )


def settings_payload(payload: dict[str, Any]) -> dict[str, Any]:
    harness = fabric_config(payload).get("harness") or {}
    return harness.get("settings") or payload.get("settings") or {}


def models_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return fabric_config(payload).get("models") or payload.get("models") or {}


def get_base_url(model_config: dict[str, Any]) -> str | None:
    """Return the explicitly configured model endpoint."""

    return model_config.get("base_url")


def selected_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    models = models_payload(payload)
    model_config = models.get("default")
    if model_config is None and len(models) == 1:
        model_config = next(iter(models.values()))
    if not isinstance(model_config, dict):
        return {}
    return model_config


def system_instruction(payload: dict[str, Any]) -> str | None:
    instructions = fabric_config(payload).get("instructions") or {}
    system = instructions.get("system") or {}
    value = system.get("content")
    return value if isinstance(value, str) else None


def max_turns(payload: dict[str, Any]) -> int | None:
    value = (fabric_config(payload).get("runtime") or {}).get("max_turns")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def timeout_seconds(payload: dict[str, Any], *, default: float) -> float:
    value = (fabric_config(payload).get("runtime") or {}).get("timeout_seconds")
    return float(default if value is None else value)


def environment_env(payload: dict[str, Any]) -> dict[str, str]:
    value = environment_payload(payload).get("env") or {}
    if not isinstance(value, dict):
        return {}
    return {
        str(name): str(item)
        for name, item in value.items()
        if isinstance(name, str) and isinstance(item, str)
    }


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


def ambient_relay_plugin_config_paths() -> list[Path]:
    """Return ambient user or project Relay plugin configs visible to Python."""

    user_directory: Path | None = None
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home and Path(xdg_config_home).is_absolute():
        user_directory = Path(xdg_config_home) / "nemo-relay"
    else:
        home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
        if home is not None:
            user_directory = Path(home) / ".config" / "nemo-relay"

    candidates: list[Path] = []
    if user_directory is not None:
        candidates.append(user_directory / "plugins.toml")
    try:
        cwd = Path.cwd()
    except OSError:
        cwd = None
    if cwd is not None:
        project_config = next(
            (
                ancestor / ".nemo-relay" / "plugins.toml"
                for ancestor in (cwd, *cwd.parents)
                if (ancestor / ".nemo-relay" / "plugins.toml").exists()
            ),
            None,
        )
        if project_config is not None:
            candidates.append(project_config)

    visible: list[Path] = []
    for candidate in candidates:
        if candidate.exists() and candidate not in visible:
            visible.append(candidate)
    return visible


def reject_ambient_relay_plugin_config() -> None:
    """Fail before Relay's Python API can merge unmanaged plugin config."""

    paths = ambient_relay_plugin_config_paths()
    if not paths:
        return
    joined = ", ".join(str(path) for path in paths)
    raise RuntimeError(
        "NeMo Fabric cannot isolate Relay's Python plugin runtime from ambient "
        f"user or project configuration: {joined}. Move or remove these files "
        "before starting this Relay-enabled runtime."
    )


def reject_inherited_relay_plugin_config(report: Any) -> None:
    """Reject discovered user or project config while allowing system policy."""

    if not isinstance(report, dict):
        raise RuntimeError("NeMo Relay did not return a plugin activation report")
    diagnostics = report.get("diagnostics")
    if not isinstance(diagnostics, list):
        raise RuntimeError("NeMo Relay returned an invalid plugin activation report")
    inherited = []
    # Relay 0.7.2 exposes the source only in this message. Keep the system-policy
    # allowlist exact and fail closed until Relay provides a structured source path.
    message_prefix = "inherited plugin configuration from discovered file: "
    system_config = Path("/etc/nemo-relay/plugins.toml")
    for diagnostic in diagnostics:
        if (
            not isinstance(diagnostic, dict)
            or diagnostic.get("code") != "plugin.configuration_inherited"
        ):
            continue
        message = diagnostic.get("message")
        if (
            isinstance(message, str)
            and message.startswith(message_prefix)
            and Path(message.removeprefix(message_prefix)) == system_config
        ):
            continue
        inherited.append(diagnostic)
    if not inherited:
        return
    details = "; ".join(
        str(diagnostic.get("message") or "inherited plugin configuration")
        for diagnostic in inherited
    )
    raise RuntimeError(
        "NeMo Fabric refuses Relay plugin configuration inherited from ambient "
        f"user or project files: {details}"
    )


def capability_plan(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("capability_plan") or payload.get("capabilities") or {}


def tools_config(payload: dict[str, Any]) -> dict[str, Any]:
    tools = fabric_config(payload).get("tools") or {}
    return tools if isinstance(tools, dict) else {}


def enabled_tools(payload: dict[str, Any]) -> list[str] | None:
    tools = tools_config(payload)
    if "enabled" not in tools:
        return None
    return normalize_list(tools.get("enabled"))


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


def validate_relay_observability_v3(plugin_config: dict[str, Any]) -> None:
    """Validate Fabric's Relay observability schema boundary without mutation."""

    legacy_flat_otel_fields = {
        "attribute_mappings",
        "capture_content",
        "endpoint",
        "header_env",
        "headers",
        "instrumentation_scope",
        "mark_exclude_names",
        "mark_projection",
        "resource_attributes",
        "semantic_selector",
        "service_name",
        "service_namespace",
        "service_version",
        "timeout_millis",
        "transport",
    }
    for component in plugin_config.get("components") or []:
        if not isinstance(component, dict) or component.get("kind") != "observability":
            continue
        config = component.get("config")
        if not isinstance(config, dict):
            raise ValueError(
                "NeMo Relay observability component config must be an object"
            )
        if "version" in config:
            version = config["version"]
            if isinstance(version, bool) or not isinstance(version, int) or version != 3:
                raise ValueError(
                    "unsupported NeMo Relay observability config version "
                    f"{version!r}; expected version 3"
                )
        if "openinference" in config:
            raise ValueError(
                "NeMo Relay observability config version 3 removed the standalone "
                "openinference section; use an opentelemetry endpoint with type "
                "'openinference'"
            )
        opentelemetry = config.get("opentelemetry")
        if isinstance(opentelemetry, dict):
            legacy_fields = sorted(legacy_flat_otel_fields.intersection(opentelemetry))
            if legacy_fields:
                raise ValueError(
                    "NeMo Relay observability config version 3 requires exporter "
                    "fields inside opentelemetry.endpoints: "
                    + ", ".join(legacy_fields)
                )
            endpoints = opentelemetry.get("endpoints")
            if opentelemetry.get("enabled") is True and (
                not isinstance(endpoints, list) or not endpoints
            ):
                raise ValueError(
                    "enabled NeMo Relay OpenTelemetry requires at least one endpoint"
                )


def load_relay_plugin_config(payload: dict[str, Any]) -> dict[str, Any]:
    config_path = os.environ.get("FABRIC_RELAY_CONFIG_PATH")
    if not config_path:
        raise RuntimeError("FABRIC_RELAY_CONFIG_PATH is required when Relay is enabled")

    with Path(config_path).open(encoding="utf-8") as stream:
        wrapper = json.load(stream)

    relay = wrapper.get("relay", {})
    plugin_config = relay.get("config") or {}
    if "components" not in plugin_config:
        components = (
            [
                {
                    "kind": "observability",
                    "enabled": True,
                    "config": plugin_config,
                }
            ]
            if plugin_config
            else []
        )
        plugin_config = {"version": 1, "components": components}
    plugin_config.setdefault("version", 1)
    plugin_config.setdefault("components", [])
    normalize_relay_output_dirs(plugin_config, payload)
    return plugin_config


def normalize_relay_output_dirs(
    plugin_config: dict[str, Any], payload: dict[str, Any]
) -> None:
    validate_relay_observability_v3(plugin_config)

    base = Path(base_dir(payload)).resolve()
    runtime_id = runtime_context(payload)["runtime_id"]
    for component in plugin_config.get("components", []):
        if not isinstance(component, dict) or component.get("kind") != "observability":
            continue
        config = component["config"]

        atof = config.get("atof")
        if isinstance(atof, dict) and atof.get("enabled"):
            for sink in atof.get("sinks") or []:
                if not isinstance(sink, dict) or sink.get("type") != "file":
                    continue
                output_directory = sink.get("output_directory")
                if output_directory:
                    path = Path(output_directory)
                    if not path.is_absolute():
                        path = base / path
                else:
                    path = base / "artifacts" / "relay"
                sink["output_directory"] = str(path / str(runtime_id))
                Path(sink["output_directory"]).mkdir(parents=True, exist_ok=True)
                sink.setdefault("filename", "events.atof.jsonl")
                sink.setdefault("mode", "overwrite")

        atif = config.get("atif")
        if not isinstance(atif, dict) or not atif.get("enabled"):
            continue
        output_directory = atif.get("output_directory")
        if output_directory:
            path = Path(output_directory)
            if not path.is_absolute():
                path = base / path
        else:
            path = base / "artifacts" / "relay"

        atif["output_directory"] = str(path / str(runtime_id))
        Path(atif["output_directory"]).mkdir(parents=True, exist_ok=True)
        atif.setdefault("filename_template", "trajectory-{session_id}.atif.json")
        atif.setdefault("agent_name", agent_name(payload))
        atif.setdefault("model_name", relay_model_name(payload))


def _artifact_directory(value: Any) -> Path | None:
    if not value:
        return None
    try:
        directory = Path(value).resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    return directory if directory.is_dir() else None


def _artifact_file(value: Any, *, directory: Path) -> Path | None:
    try:
        path = Path(value).resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    if not path.is_file() or not path.is_relative_to(directory):
        return None
    return path


def _artifact_name_is_local(value: str) -> bool:
    try:
        return Path(value).name == value
    except (OSError, ValueError):
        return False


def _artifact_glob(directory: Path, pattern: str) -> list[Path]:
    if not _artifact_name_is_local(pattern):
        return []
    try:
        return sorted(directory.glob(pattern))
    except (OSError, ValueError):
        return []


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
                directory = _artifact_directory(sink.get("output_directory"))
                if directory is not None:
                    filename = sink.get("filename")
                    if isinstance(filename, str) and filename:
                        paths = (
                            [directory / filename]
                            if _artifact_name_is_local(filename)
                            else []
                        )
                    else:
                        paths = _artifact_glob(directory, "*.jsonl")
                    for path in paths:
                        resolved = _artifact_file(path, directory=directory)
                        if resolved is not None:
                            artifacts.append({"kind": "atof", "path": str(resolved)})

        atif = config.get("atif")
        if isinstance(atif, dict) and atif.get("enabled"):
            directory = _artifact_directory(atif.get("output_directory"))
            if directory is not None:
                template = atif.get("filename_template")
                if not isinstance(template, str) or not template:
                    continue
                pattern = glob.escape(template).replace("{session_id}", "*")
                for path in _artifact_glob(directory, pattern):
                    resolved = _artifact_file(path, directory=directory)
                    if resolved is not None:
                        artifacts.append({"kind": "atif", "path": str(resolved)})
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
            raise RuntimeError(
                "FABRIC_RELAY_CONFIG_PATH is required when Relay is enabled"
            )

        config_path = Path(config_path)
        config_dir = config_path.parent / "relay-config"
        if plugin_config is not None:
            validate_relay_observability_v3(plugin_config)
        config_dir.mkdir(parents=True, exist_ok=True)
        relay_config_path = None
        plugin_config_path = None

        if relay_config is not None:
            relay_config_path = config_dir / "config.toml"
            relay_config_path.write_text(tomli_w.dumps(relay_config), encoding="utf-8")

        if plugin_config is not None:
            plugin_config_path = config_dir / "plugins.toml"
            plugin_config_path.write_text(
                tomli_w.dumps(plugin_config),
                encoding="utf-8",
            )

        return relay_config_path, plugin_config_path
    except ImportError as e:
        raise RuntimeError("tomli_w is not installed") from e


def relay_model_name(payload: dict[str, Any]) -> str:
    return selected_model_config(payload).get("model") or "unknown"
