#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Codex through its native Python SDK and the Fabric adapter contract."""

from __future__ import annotations

import asyncio
import json
import math
import os
import shlex
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any

from openai_codex import (
    ApprovalMode,
    AsyncCodex,
    CodexConfig,
    CodexError,
    Sandbox,
    TransportClosedError,
    is_retryable_error,
)
from openai_codex.generated.v2_all import SkillsExtraRootsSetResponse
from openai_codex.types import Personality, ReasoningEffort, TurnStatus

import nemo_fabric_adapters.common.relay_gateway as relay_gateway
import nemo_fabric_adapters.common.relay_hooks as relay_hooks
import nemo_fabric_adapters.common.utils as common_utils


DEFAULT_TIMEOUT_SECONDS = 1800.0
INTERRUPT_TIMEOUT_SECONDS = 5.0
SANDBOXES = {
    "read-only": Sandbox.read_only,
    "workspace-write": Sandbox.workspace_write,
    "danger-full-access": Sandbox.full_access,
}
APPROVAL_MODES = {
    "auto_review": ApprovalMode.auto_review,
    "deny_all": ApprovalMode.deny_all,
}
INHERITED_ENV_NAMES = {
    "APPDATA",
    "CODEX_HOME",
    "CODEX_SQLITE_HOME",
    "COMSPEC",
    "HOME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOCALAPPDATA",
    "NO_PROXY",
    "OPENAI_API_KEY",
    "PATH",
    "PATHEXT",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USERPROFILE",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}
REMOVED_CLI_SETTINGS = {
    "codex_args",
    "codex_command",
    "codex_profile",
    "codex_state_dir",
    "skip_git_repo_check",
}
NORMALIZED_SETTING_FIELDS = {
    "cwd": "FabricConfig.environment.workspace",
    "mcp_servers": "FabricConfig.mcp",
    "model_name": "FabricConfig.models",
    "skills": "FabricConfig.skills",
}


@dataclass(frozen=True)
class CodexRelaySettings:
    """Invocation-scoped Relay state consumed by the Codex SDK adapter."""

    gateway: relay_gateway.RelayGatewayLaunch
    plugin_config: dict[str, Any]


class CodexAdapterError(Exception):
    """Expected adapter error with a stable public code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.metadata = metadata or {}


class AdapterInputError(CodexAdapterError):
    """Invalid Fabric invocation input."""


class AdapterConfigError(CodexAdapterError):
    """Invalid Codex adapter configuration."""


class AdapterStateError(CodexAdapterError):
    """Invalid persisted runtime state."""


class AdapterRelayError(CodexAdapterError):
    """NeMo Relay setup or lifecycle failure."""


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise AdapterConfigError(
            "codex_invalid_configuration", f"{name} must be a mapping"
        )
    return value


def _settings(payload: dict[str, Any]) -> dict[str, Any]:
    return _mapping(common_utils.settings_payload(payload), name="harness.settings")


def _validate_settings_boundary(settings: dict[str, Any]) -> None:
    removed = sorted(REMOVED_CLI_SETTINGS.intersection(settings))
    if removed:
        names = ", ".join(f"harness.settings.{name}" for name in removed)
        raise AdapterConfigError(
            "codex_invalid_configuration",
            f"Codex CLI-only settings are not supported by the SDK adapter: {names}",
        )
    for name, normalized_field in NORMALIZED_SETTING_FIELDS.items():
        if name in settings:
            raise AdapterConfigError(
                "codex_invalid_configuration",
                f"harness.settings.{name} is not supported; use {normalized_field}",
            )


def runtime_id(payload: dict[str, Any]) -> str:
    value = common_utils.runtime_context(payload).get("runtime_id")
    if not isinstance(value, str) or not value:
        raise AdapterInputError(
            "codex_invalid_request", "Fabric runtime ID is required"
        )
    return value


def request_prompt(payload: dict[str, Any]) -> str:
    value = (payload.get("request") or {}).get("input")
    if not isinstance(value, str):
        raise AdapterInputError("codex_invalid_request", "Codex input must be text")
    return value


def _native_capabilities(payload: dict[str, Any]) -> dict[str, Any]:
    plan = _mapping(common_utils.capability_plan(payload), name="capability_plan")
    return _mapping(plan.get("native"), name="capability_plan.native")


def _native_mcp_servers(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    servers = _mapping(
        _native_capabilities(payload).get("mcp_servers"),
        name="native MCP servers",
    )
    result: dict[str, dict[str, Any]] = {}
    for name, raw in sorted(servers.items()):
        if not isinstance(name, str) or not name:
            raise AdapterConfigError(
                "codex_invalid_configuration",
                "MCP server names must be non-empty strings",
            )
        server = _mapping(raw, name=f"MCP server {name}")
        transport = server.get("transport")
        if not isinstance(transport, str) or not transport:
            raise AdapterConfigError(
                "codex_invalid_configuration",
                f"MCP server {name} transport is required",
            )
        target = server.get("url")
        if not isinstance(target, str) or not target:
            raise AdapterConfigError(
                "codex_invalid_configuration",
                f"MCP server {name} URL is required",
            )
        target = os.path.expandvars(target).strip()
        if not target:
            raise AdapterConfigError(
                "codex_invalid_configuration",
                f"MCP server {name} URL is required",
            )
        normalized_transport = transport.strip().lower().replace("_", "-")
        if normalized_transport == "stdio":
            try:
                command = shlex.split(target)
            except ValueError as error:
                raise AdapterConfigError(
                    "codex_invalid_configuration",
                    f"MCP server {name} command is invalid",
                ) from error
            if not command:
                raise AdapterConfigError(
                    "codex_invalid_configuration",
                    f"MCP server {name} command is required",
                )
            result[name] = {"command": command[0], "args": command[1:]}
        elif normalized_transport in {"http", "streamable-http"}:
            result[name] = {"url": target}
        else:
            raise AdapterConfigError(
                "codex_invalid_configuration",
                f"unsupported Codex MCP transport: {transport}",
            )
    return result


def _native_skill_paths(payload: dict[str, Any]) -> list[Path]:
    values = _native_capabilities(payload).get("skill_paths", [])
    if not isinstance(values, list) or any(
        not isinstance(value, (str, Path)) or not str(value) for value in values
    ):
        raise AdapterConfigError(
            "codex_invalid_configuration",
            "native skill_paths must be a list of paths",
        )

    paths: list[Path] = []
    names: set[str] = set()
    config_root = Path(common_utils.base_dir(payload))
    for value in values:
        skill_path = Path(value)
        if not skill_path.is_absolute():
            skill_path = config_root / skill_path
        skill_path = skill_path.resolve()
        skill_file = skill_path / "SKILL.md"
        if not skill_path.is_dir() or not skill_file.is_file():
            raise AdapterConfigError(
                "codex_invalid_configuration",
                "Fabric skill path must be a directory containing SKILL.md: "
                f"{skill_path}",
            )
        name = skill_path.name
        if not name or name in names:
            raise AdapterConfigError(
                "codex_invalid_configuration",
                f"Fabric skill names must be unique: {name}",
            )
        names.add(name)
        paths.append(skill_path)
    return paths


async def _register_skill_roots(codex: AsyncCodex, skill_paths: list[Path]) -> None:
    if not skill_paths:
        return

    # The pinned SDK does not yet wrap the app-server's process-scoped
    # skills/extraRoots/set request. Keep the pinned-SDK compatibility seam
    # here so arbitrary Fabric skill paths become discoverable without
    # modifying the consumer workspace.
    await codex.models()
    client = getattr(codex, "_client", None)
    request = getattr(client, "request", None)
    if not callable(request):
        raise AdapterConfigError(
            "codex_invalid_configuration",
            "Codex SDK does not expose the required skill registration request",
        )
    await request(
        "skills/extraRoots/set",
        {"extraRoots": [str(path) for path in skill_paths]},
        response_model=SkillsExtraRootsSetResponse,
    )


def resolve_cwd(payload: dict[str, Any]) -> Path:
    environment = _mapping(
        common_utils.environment_payload(payload), name="runtime environment"
    )
    value = environment.get("workspace") or common_utils.base_dir(payload)
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(common_utils.base_dir(payload)) / path
    return path.resolve()


def _selected_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    settings = _settings(payload)
    models = _mapping(common_utils.models_payload(payload), name="models")
    selected = models.get(settings.get("model", "default")) or {}
    return _mapping(selected, name="selected model")


def selected_model(payload: dict[str, Any]) -> str | None:
    model_config = _selected_model_config(payload)
    value = model_config.get("model")
    if value is None:
        return None
    provider = model_config.get("provider")
    if provider not in {"openai", "nvidia"}:
        raise AdapterConfigError(
            "codex_invalid_configuration",
            "selected model provider must be openai or nvidia for the Codex adapter",
        )
    if not isinstance(value, str) or not value:
        raise AdapterConfigError(
            "codex_invalid_configuration", "model must be a non-empty string"
        )
    return value.removeprefix("openai/") if provider == "openai" else value


def selected_model_provider(payload: dict[str, Any]) -> str:
    return str(_selected_model_config(payload).get("provider") or "openai")


def nvidia_model_provider_config(payload: dict[str, Any]) -> dict[str, Any]:
    model_config = _selected_model_config(payload)
    if model_config.get("provider") != "nvidia":
        return {}
    api_key_env = model_config.get("api_key_env") or "NVIDIA_API_KEY"
    if not isinstance(api_key_env, str) or not api_key_env:
        raise AdapterConfigError(
            "codex_invalid_configuration",
            "models.default.api_key_env must be a non-empty string",
        )
    if not os.environ.get(api_key_env):
        raise AdapterConfigError(
            "codex_invalid_configuration",
            f"{api_key_env} is required for the NVIDIA model provider",
        )
    model_settings = _mapping(
        model_config.get("settings"), name="selected model settings"
    )
    base_url = (
        model_settings.get("base_url")
        or os.environ.get("NVIDIA_FRONTIER_BASE_URL")
    )
    if not isinstance(base_url, str) or not base_url:
        raise AdapterConfigError(
            "codex_invalid_configuration",
            "models.default.settings.base_url or NVIDIA_FRONTIER_BASE_URL is required "
            "for the NVIDIA model provider",
        )
    return {
        "model_providers": {
            "nvidia": {
                "name": "NVIDIA",
                "base_url": base_url.rstrip("/"),
                "env_key": api_key_env,
                "wire_api": "responses",
            }
        }
    }


def sandbox(payload: dict[str, Any]) -> Sandbox:
    value = _settings(payload).get("sandbox", "read-only")
    try:
        return SANDBOXES[value]
    except (KeyError, TypeError) as error:
        raise AdapterConfigError(
            "codex_invalid_configuration",
            f"sandbox must be one of: {', '.join(sorted(SANDBOXES))}",
        ) from error


def approval_mode(payload: dict[str, Any]) -> ApprovalMode:
    value = _settings(payload).get("approval_mode", "auto_review")
    try:
        return APPROVAL_MODES[value]
    except (KeyError, TypeError) as error:
        raise AdapterConfigError(
            "codex_invalid_configuration",
            f"approval_mode must be one of: {', '.join(sorted(APPROVAL_MODES))}",
        ) from error


def timeout_seconds(payload: dict[str, Any]) -> float:
    value = _settings(payload).get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterConfigError(
            "codex_invalid_configuration", "timeout_seconds must be positive"
        )
    result = float(value)
    if result <= 0 or not math.isfinite(result):
        raise AdapterConfigError(
            "codex_invalid_configuration", "timeout_seconds must be positive"
        )
    return result


def _optional_string(settings: dict[str, Any], name: str) -> str | None:
    value = settings.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise AdapterConfigError(
            "codex_invalid_configuration",
            f"harness.settings.{name} must be a non-empty string",
        )
    return value


def child_environment(
    payload: dict[str, Any], *, relay_gateway_url: str | None = None
) -> dict[str, str]:
    values = dict.fromkeys(os.environ, "")
    values.update(
        {name: os.environ[name] for name in INHERITED_ENV_NAMES if name in os.environ}
    )
    telemetry = common_utils.runtime_context(payload).get("telemetry")
    if telemetry is None:
        telemetry = {}
    if not isinstance(telemetry, dict):
        raise AdapterInputError(
            "codex_invalid_request", "runtime_context.telemetry must be a mapping"
        )
    telemetry_env = telemetry.get("env")
    if telemetry_env is None:
        telemetry_env = {}
    if not isinstance(telemetry_env, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in telemetry_env.items()
    ):
        raise AdapterInputError(
            "codex_invalid_request",
            "runtime_context.telemetry.env must contain strings",
        )
    values.update(telemetry_env)
    model_config = _selected_model_config(payload)
    api_key_env = model_config.get("api_key_env")
    if isinstance(api_key_env, str) and api_key_env in os.environ:
        values[api_key_env] = os.environ[api_key_env]
    configured = _mapping(_settings(payload).get("env"), name="harness.settings.env")
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in configured.items()
    ):
        raise AdapterConfigError(
            "codex_invalid_configuration",
            "harness.settings.env must contain strings",
        )
    values.update(configured)
    if selected_model_provider(payload) == "nvidia":
        codex_home = state_dir(payload) / "nvidia-home"
        values["CODEX_HOME"] = str(codex_home)
    # The SDK overlays this mapping on the parent environment. An empty
    # originator is still treated as an override by Codex and produces invalid
    # initialize metadata ("/<version>"). Use the official SDK client identity
    # without inheriting the identity of a parent Codex process.
    values["CODEX_INTERNAL_ORIGINATOR_OVERRIDE"] = "codex_python_sdk"
    if relay_gateway_url is not None:
        values["NEMO_RELAY_GATEWAY_URL"] = relay_gateway_url
    return values


def _artifact_root(payload: dict[str, Any]) -> Path:
    artifacts = common_utils.runtime_context(payload).get("artifacts") or {}
    root = artifacts.get("root") if isinstance(artifacts, dict) else None
    if root:
        return Path(str(root))
    return Path(common_utils.base_dir(payload)) / "artifacts" / "codex"


def state_dir(payload: dict[str, Any]) -> Path:
    return _artifact_root(payload) / ".fabric" / "codex"


def runtime_state_path(payload: dict[str, Any], fabric_runtime_id: str) -> Path:
    digest = sha256(fabric_runtime_id.encode("utf-8")).hexdigest()
    return state_dir(payload) / "runtimes" / f"{digest}.json"


def load_thread_id(payload: dict[str, Any], fabric_runtime_id: str) -> str | None:
    path = runtime_state_path(payload, fabric_runtime_id)
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError("state must be an object")
        if state.get("runtime_id") != fabric_runtime_id:
            raise ValueError("runtime mismatch")
        thread_id = state.get("codex_thread_id")
        if not isinstance(thread_id, str) or not thread_id:
            raise ValueError("missing Codex thread")
        return thread_id
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise AdapterStateError(
            "codex_invalid_runtime_state", "Codex runtime state is invalid"
        ) from error


def save_thread_id(
    payload: dict[str, Any], fabric_runtime_id: str, codex_thread_id: str
) -> None:
    if not codex_thread_id:
        raise AdapterStateError(
            "codex_invalid_runtime_state", "Codex thread ID is missing"
        )
    path = runtime_state_path(payload, fabric_runtime_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    invocation_id = (
        common_utils.runtime_context(payload).get("invocation_id") or "invocation"
    )
    temporary = path.with_suffix(f".{invocation_id}.tmp")
    temporary.write_text(
        json.dumps(
            {"runtime_id": fabric_runtime_id, "codex_thread_id": codex_thread_id},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _merge_config(target: dict[str, Any], layer: dict[str, Any]) -> None:
    for key, value in layer.items():
        existing = target.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            _merge_config(existing, value)
        else:
            target[key] = value


def _json_value(value: Any, *, name: str) -> Any:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise AdapterConfigError(
            "codex_invalid_configuration", f"{name} must be JSON-compatible"
        ) from error
    return value


def _apply_config_overrides(
    config: dict[str, Any], overrides: dict[str, Any]
) -> None:
    for dotted_key, value in sorted(overrides.items()):
        if not isinstance(dotted_key, str):
            raise AdapterConfigError(
                "codex_invalid_configuration",
                "config_overrides keys must be strings",
            )
        parts = dotted_key.split(".")
        if any(not part for part in parts):
            raise AdapterConfigError(
                "codex_invalid_configuration",
                f"invalid Codex config override key {dotted_key!r}",
            )
        target = config
        for part in parts[:-1]:
            existing = target.setdefault(part, {})
            if not isinstance(existing, dict):
                raise AdapterConfigError(
                    "codex_invalid_configuration",
                    f"Codex config override {dotted_key!r} conflicts with {part!r}",
                )
            target = existing
        target[parts[-1]] = _json_value(
            value, name=f"config_overrides.{dotted_key}"
        )


def native_codex_telemetry_config(payload: dict[str, Any]) -> dict[str, Any]:
    if "native" not in common_utils.telemetry_providers(payload):
        return {}

    telemetry_config = common_utils.native_telemetry_config(payload)
    for component in telemetry_config.get("components") or []:
        if (
            not isinstance(component, dict)
            or component.get("kind") != "observability"
            or not component.get("enabled", True)
        ):
            continue
        component_config = component.get("config") or {}
        opentelemetry = component_config.get("opentelemetry") or {}
        if not isinstance(opentelemetry, dict) or not opentelemetry.get("enabled"):
            continue

        otel: dict[str, Any] = {}
        resource_attributes = opentelemetry.get("resource_attributes") or {}
        environment = resource_attributes.get("deployment.environment")
        if environment is not None:
            otel["environment"] = environment

        endpoint = opentelemetry.get("endpoint")
        if endpoint:
            transport = opentelemetry.get("transport", "http_binary")
            exporters = {
                "http_binary": ("otlp-http", "binary"),
                "grpc": ("otlp-grpc", "grpc"),
                "http_json": ("otlp-http", "json"),
            }
            try:
                exporter, protocol = exporters[transport]
            except (KeyError, TypeError) as error:
                raise AdapterConfigError(
                    "codex_invalid_configuration",
                    f"unsupported Codex native OpenTelemetry transport {transport!r}",
                ) from error
            otel["trace_exporter"] = {
                exporter: {"endpoint": endpoint, "protocol": protocol}
            }
        return {"otel": otel}
    return {}


def prepare_codex_relay(payload: dict[str, Any]) -> CodexRelaySettings | None:
    """Generate invocation-scoped Relay gateway configuration."""

    if not common_utils.relay_enabled(payload):
        return None
    command = _settings(payload).get("nemo_relay_command") or "nemo-relay"
    if not isinstance(command, (str, Path)):
        raise AdapterConfigError(
            "codex_invalid_configuration", "nemo_relay_command must be a path"
        )
    try:
        executable = relay_gateway.resolve_relay_command(
            Path(common_utils.base_dir(payload)).resolve(), command
        )
    except FileNotFoundError as error:
        raise AdapterRelayError(
            "codex_relay_unavailable", "NeMo Relay CLI executable was not found"
        ) from error

    try:
        relay_contract = relay_gateway.relay_cli_contract(executable)
        plugin_config = common_utils.load_relay_plugin_config(payload)
        config_path, plugin_config_path = common_utils.write_relay_configs(
            # The SDK owns Codex execution. Relay needs only gateway defaults and
            # the sibling plugins.toml; configuring an agent command would retain
            # a misleading dependency on the removed Codex CLI launch path.
            relay_config={},
            plugin_config=plugin_config,
            observability_version=relay_contract.observability_version,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        raise AdapterRelayError(
            "codex_relay_configuration_failed",
            "NeMo Relay runtime configuration is unavailable",
        ) from error
    if config_path is None or plugin_config_path is None:
        raise AdapterRelayError(
            "codex_relay_configuration_failed",
            "NeMo Relay runtime configuration is unavailable",
        )

    port = relay_gateway.find_available_tcp_port()
    bind = f"127.0.0.1:{port}"
    return CodexRelaySettings(
        gateway=relay_gateway.RelayGatewayLaunch(
            executable=executable,
            config_path=config_path,
            bind=bind,
            url=f"http://{bind}",
            log_path=config_path.parent / "gateway.log",
        ),
        plugin_config=plugin_config,
    )


def thread_config(
    payload: dict[str, Any], relay: CodexRelaySettings | None
) -> dict[str, Any]:
    """Build request-scoped Codex config without writing a user profile."""

    config = native_codex_telemetry_config(payload)
    _merge_config(config, nvidia_model_provider_config(payload))
    mcp_servers = _native_mcp_servers(payload)
    if mcp_servers:
        config["mcp_servers"] = mcp_servers
    overrides = _mapping(
        _settings(payload).get("config_overrides"),
        name="harness.settings.config_overrides",
    )
    _apply_config_overrides(config, overrides)
    if relay is not None:
        _merge_config(
            config,
            {
                # Keep the SDK-selected built-in provider so Codex retains its
                # native API-key and ChatGPT authentication behavior. Relay is
                # only the transport endpoint for this invocation.
                "openai_base_url": relay.gateway.url,
                "features": {
                    "hooks": True,
                    # Relay disables delegated multi-agent execution because
                    # Codex encrypts delegated task content before it reaches
                    # the gateway, making those spans opaque.
                    "multi_agent_v2": {"enabled": False},
                },
                "hooks": relay_hooks.render_relay_hooks(
                    "codex", relay.gateway.executable
                )["hooks"],
                # This runtime-only request override is the SDK-native equivalent
                # of the former non-interactive CLI flag. Fabric generated and
                # vetted every hook command above.
                "bypass_hook_trust": True,
            },
        )
    return config


def sdk_config(
    payload: dict[str, Any], relay: CodexRelaySettings | None
) -> CodexConfig:
    codex_bin = _optional_string(_settings(payload), "codex_bin")
    if codex_bin is not None:
        path = Path(codex_bin)
        if not path.is_absolute():
            path = (Path(common_utils.base_dir(payload)) / path).resolve()
        codex_bin = str(path)
    return CodexConfig(
        codex_bin=codex_bin,
        cwd=str(resolve_cwd(payload)),
        env=child_environment(
            payload,
            relay_gateway_url=relay.gateway.url if relay is not None else None,
        ),
    )


def _personality(payload: dict[str, Any]) -> Personality | None:
    value = _optional_string(_settings(payload), "personality")
    if value is None:
        return None
    try:
        return Personality(value)
    except ValueError as error:
        raise AdapterConfigError(
            "codex_invalid_configuration", "personality is invalid"
        ) from error


def _reasoning_effort(payload: dict[str, Any]) -> ReasoningEffort | None:
    value = _optional_string(_settings(payload), "reasoning_effort")
    if value is None:
        return None
    try:
        return ReasoningEffort(value)
    except ValueError as error:
        raise AdapterConfigError(
            "codex_invalid_configuration", "reasoning_effort is invalid"
        ) from error


def _output_schema(payload: dict[str, Any]) -> dict[str, Any] | None:
    value = _settings(payload).get("output_schema")
    if value is None:
        return None
    return _mapping(_json_value(value, name="output_schema"), name="output_schema")


def validate_payload(payload: dict[str, Any]) -> str:
    """Validate pure invocation inputs before starting SDK or Relay processes."""

    settings = _settings(payload)
    _validate_settings_boundary(settings)
    request_prompt(payload)
    _native_skill_paths(payload)
    fabric_runtime_id = runtime_id(payload)
    resolve_cwd(payload)
    selected_model(payload)
    sandbox(payload)
    approval_mode(payload)
    timeout_seconds(payload)
    for name in (
        "base_instructions",
        "developer_instructions",
        "service_name",
        "service_tier",
    ):
        _optional_string(settings, name)
    _personality(payload)
    _reasoning_effort(payload)
    _output_schema(payload)
    if (
        common_utils.relay_enabled(payload)
        and selected_model_provider(payload) != "openai"
    ):
        raise AdapterConfigError(
            "codex_invalid_configuration",
            "NeMo Relay requires the built-in openai model provider",
        )
    child_environment(payload)
    thread_config(payload, None)
    return fabric_runtime_id


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json", by_alias=True))
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise AdapterConfigError(
        "codex_invalid_configuration", "Codex SDK result is not JSON-safe"
    )


def _failure(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    **metadata: Any,
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "retryable": retryable,
    }
    if metadata:
        error["metadata"] = metadata
    return {
        "harness": "codex",
        "adapter": "sdk",
        "mode": "codex_sdk_runtime",
        "response": None,
        "completed": False,
        "failed": True,
        "error": error,
        "events": [],
    }


def adapter_failure(error: CodexAdapterError) -> dict[str, Any]:
    return _failure(error.code, error.message, **error.metadata)


def sdk_failure(error: BaseException) -> dict[str, Any]:
    if isinstance(error, TimeoutError):
        return _failure("codex_timed_out", "Codex invocation timed out")
    if isinstance(error, TransportClosedError):
        return _failure(
            "codex_connection_failed", "Codex SDK runtime connection closed"
        )
    if isinstance(error, CodexError):
        return _failure(
            "codex_sdk_failed",
            "Codex SDK request failed",
            retryable=is_retryable_error(error),
            sdk_error=type(error).__name__,
        )
    if isinstance(error, OSError):
        return _failure(
            "codex_runtime_unavailable", "Codex SDK runtime could not start"
        )
    return _failure(
        "codex_turn_failed",
        str(error) or "Codex turn failed",
    )


def normalize_result(
    payload: dict[str, Any], *, thread_id: str, result: Any
) -> dict[str, Any]:
    status = _json_safe(result.status)
    completed = result.status == TurnStatus.completed and result.final_response is not None
    error = None
    if not completed:
        message = (
            result.error.message
            if result.error is not None
            else "Codex invocation did not return a final response"
        )
        error = {
            "code": "codex_turn_incomplete",
            "message": message,
            "retryable": False,
            "metadata": {"status": status},
        }
    return {
        "harness": "codex",
        "adapter": "sdk",
        "mode": "codex_sdk_runtime",
        "cwd": str(resolve_cwd(payload)),
        "model": selected_model(payload),
        "thread_id": thread_id,
        "turn_id": result.id,
        "turn_status": status,
        "response": result.final_response,
        "usage": _json_safe(result.usage),
        "started_at": result.started_at,
        "completed_at": result.completed_at,
        "duration_ms": result.duration_ms,
        "completed": completed,
        "failed": not completed,
        "error": error,
        "events": [_json_safe(item) for item in result.items],
        "state_dir": str(state_dir(payload)),
    }


async def _interrupt_turn(handle: Any) -> None:
    if handle is None:
        return
    try:
        async with asyncio.timeout(INTERRUPT_TIMEOUT_SECONDS):
            await handle.interrupt()
    except (TimeoutError, CodexError, RuntimeError, OSError):
        # The SDK process is closed immediately afterwards, which is the final
        # cancellation boundary if the runtime cannot acknowledge interrupt.
        pass


async def invoke_codex_sdk(
    payload: dict[str, Any],
    *,
    prior_thread_id: str | None,
    relay: CodexRelaySettings | None,
) -> tuple[dict[str, Any], str | None]:
    """Execute one SDK turn and always close the app-server transport."""

    settings = _settings(payload)
    config = thread_config(payload, relay)
    skill_paths = _native_skill_paths(payload)
    prompt = request_prompt(payload)
    client_config = sdk_config(payload, relay)
    codex: AsyncCodex | None = None
    handle = None
    output: dict[str, Any]
    thread_id: str | None = None
    try:
        if selected_model_provider(payload) == "nvidia":
            await asyncio.to_thread(
                Path(client_config.env["CODEX_HOME"]).mkdir,
                parents=True,
                exist_ok=True,
            )
        codex = AsyncCodex(config=client_config)
        async with asyncio.timeout(timeout_seconds(payload)):
            await _register_skill_roots(codex, skill_paths)
            common = {
                "approval_mode": approval_mode(payload),
                "base_instructions": _optional_string(settings, "base_instructions"),
                "config": config or None,
                "cwd": str(resolve_cwd(payload)),
                "developer_instructions": _optional_string(
                    settings, "developer_instructions"
                ),
                "model": selected_model(payload),
                "model_provider": selected_model_provider(payload),
                "personality": _personality(payload),
                "sandbox": sandbox(payload),
                "service_tier": _optional_string(settings, "service_tier"),
            }
            if prior_thread_id is None:
                thread = await codex.thread_start(
                    **common,
                    service_name=_optional_string(settings, "service_name"),
                )
            else:
                thread = await codex.thread_resume(prior_thread_id, **common)
                if thread.id != prior_thread_id:
                    raise AdapterStateError(
                        "codex_thread_mismatch",
                        "Codex thread identity changed during resume",
                    )
            thread_id = thread.id
            handle = await thread.turn(
                prompt,
                effort=_reasoning_effort(payload),
                output_schema=_output_schema(payload),
            )
            result = await handle.run()
            output = normalize_result(payload, thread_id=thread.id, result=result)
    except TimeoutError as error:
        await _interrupt_turn(handle)
        output = sdk_failure(error)
    except CodexAdapterError:
        raise
    except (CodexError, RuntimeError, OSError) as error:
        output = sdk_failure(error)
    finally:
        if codex is not None:
            try:
                await codex.close()
            except Exception:
                output = _failure(
                    "codex_sdk_stop_failed", "Codex SDK runtime failed to stop"
                )
    return output, thread_id


def _relay_output(
    output: dict[str, Any], relay: CodexRelaySettings
) -> dict[str, Any]:
    output["relay_runtime"] = {
        "enabled": True,
        "emitter": "codex-sdk/nemo-relay",
        "config_path": os.environ.get("FABRIC_RELAY_CONFIG_PATH"),
        "gateway_config_path": str(relay.gateway.config_path),
        "gateway_url": relay.gateway.url,
        "gateway_log_path": str(relay.gateway.log_path),
    }
    output["relay_artifacts"] = common_utils.collect_relay_artifacts(
        relay.plugin_config
    )
    return output


async def run_codex(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one Fabric invocation with SDK-owned Codex execution."""

    fabric_runtime_id = validate_payload(payload)
    prior_thread_id = load_thread_id(payload, fabric_runtime_id)
    relay = prepare_codex_relay(payload)
    gateway_process = None
    cleanup_error: AdapterRelayError | None = None
    try:
        if relay is not None:
            try:
                gateway_process = relay_gateway.start_relay_gateway(
                    launch=relay.gateway, cwd=resolve_cwd(payload)
                )
            except relay_gateway.RelayGatewayError as error:
                raise AdapterRelayError(
                    "codex_relay_start_failed",
                    "NeMo Relay gateway failed to start",
                    metadata={"gateway_log_path": str(relay.gateway.log_path)},
                ) from error
        output, thread_id = await invoke_codex_sdk(
            payload, prior_thread_id=prior_thread_id, relay=relay
        )
        if not output["failed"] and thread_id is not None:
            save_thread_id(payload, fabric_runtime_id, thread_id)
    finally:
        if gateway_process is not None:
            try:
                relay_gateway.stop_relay_gateway(gateway_process)
            except relay_gateway.RelayGatewayError:
                cleanup_error = AdapterRelayError(
                    "codex_relay_stop_failed",
                    "NeMo Relay gateway failed to stop",
                    metadata={
                        "gateway_log_path": str(relay.gateway.log_path)
                        if relay is not None
                        else ""
                    },
                )

    if relay is not None:
        output = _relay_output(output, relay)
    if cleanup_error is not None:
        cleanup: dict[str, Any] = {
            "code": cleanup_error.code,
            "message": cleanup_error.message,
            "retryable": False,
        }
        if cleanup_error.metadata:
            cleanup["metadata"] = cleanup_error.metadata
        output["relay_runtime"]["cleanup_error"] = cleanup
        if not output["failed"]:
            output["completed"] = False
            output["failed"] = True
            output["error"] = cleanup
    return output


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one Fabric invocation from the synchronous adapter boundary."""

    try:
        return asyncio.run(run_codex(payload))
    except CodexAdapterError as error:
        return adapter_failure(error)
    except Exception:
        return _failure(
            "codex_adapter_internal_error", "Codex adapter failed unexpectedly"
        )


def main() -> None:
    try:
        payload = common_utils.load_payload()
    except Exception:
        output = _failure(
            "codex_adapter_internal_error", "Codex adapter failed unexpectedly"
        )
    else:
        output = run(payload)
    print(json.dumps(output, sort_keys=True))
    if output.get("failed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
