#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Codex through its native Python SDK and the Fabric adapter contract."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import subprocess
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
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

from nemo_fabric_adapter_contract.codec import ContractValidationError
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentModelConfig
from nemo_fabric_adapter_contract.models import RuntimeContext
import nemo_fabric_adapters.common.relay_gateway as relay_gateway
import nemo_fabric_adapters.common.relay_hooks as relay_hooks
import nemo_fabric_adapters.common.relay_artifacts as relay_artifacts
import nemo_fabric_adapters.common.utils as common_utils
from nemo_fabric_adapters.common import lifecycle


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
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CodexRelaySettings:
    """Runtime-scoped Relay state consumed by the Codex SDK adapter."""

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


def _settings(config: AgentConfig) -> dict[str, Any]:
    return config.harness.settings if config.harness else {}


def request_prompt(payload: dict[str, Any]) -> str:
    value = (payload.get("request") or {}).get("input")
    if not isinstance(value, str):
        raise AdapterInputError("codex_invalid_request", "Codex input must be text")
    return value


def _native_mcp_servers(config: AgentConfig) -> dict[str, dict[str, Any]]:
    servers = config.mcp.servers if config.mcp else {}
    result: dict[str, dict[str, Any]] = {}
    for name, server in sorted(servers.items()):
        transport = server.transport
        target = os.path.expandvars(server.url).strip()
        if not target:
            raise AdapterConfigError(
                "codex_invalid_configuration",
                f"MCP server {name} URL is required",
            )
        normalized_transport = transport.strip().lower().replace("_", "-")
        if normalized_transport == "stdio":
            result[name] = {
                "command": target,
                "args": server.args,
            }
            if env := server.env:
                result[name]["env"] = env
        elif normalized_transport in {"http", "streamable-http"}:
            result[name] = {"url": target}
        else:
            raise AdapterConfigError(
                "codex_invalid_configuration",
                f"unsupported Codex MCP transport: {transport}",
            )
    return result


def _native_skill_paths(config: AgentConfig, base_dir: str) -> list[Path]:
    values = config.skills.paths if config.skills else []

    paths: list[Path] = []
    names: set[str] = set()
    config_root = Path(base_dir)
    for value in values:
        skill_path = Path(value)
        if not skill_path.is_absolute():
            skill_path = config_root / skill_path
        skill_path = skill_path.resolve()
        skill_file = skill_path / "SKILL.md"
        if not skill_path.is_dir() or not skill_file.is_file():
            raise AdapterConfigError(
                "codex_invalid_configuration",
                "NeMo Fabric skill path must be a directory containing SKILL.md: "
                f"{skill_path}",
            )
        name = skill_path.name
        if not name or name in names:
            raise AdapterConfigError(
                "codex_invalid_configuration",
                f"NeMo Fabric skill names must be unique: {name}",
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


def resolve_cwd(context: RuntimeContext, base_dir: str) -> Path:
    path = Path(context.environment.workspace or base_dir)
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve()


def _selected_model_config(config: AgentConfig) -> AgentModelConfig:
    model = config.models.get("default")
    if model is None and len(config.models) == 1:
        model = next(iter(config.models.values()))
    if model is None:
        raise AdapterConfigError(
            "codex_invalid_configuration",
            "Codex requires a default model or exactly one model",
        )
    return model


def selected_model(config: AgentConfig) -> str:
    model = _selected_model_config(config)
    return model.model.removeprefix("openai/") if model.provider == "openai" else model.model


def custom_model_provider_config(
    config: AgentConfig, context: RuntimeContext
) -> dict[str, Any]:
    model_config = _selected_model_config(config)
    provider = model_config.provider
    if provider == "openai":
        return {}
    api_key_env = model_config.api_key_env
    if api_key_env is None:
        raise AdapterConfigError(
            "codex_invalid_configuration",
            "selected model api_key_env is required for a custom "
            "Responses-compatible provider",
        )
    if not (
        context.environment.env.get(api_key_env)
        or os.environ.get(api_key_env)
    ):
        raise AdapterConfigError(
            "codex_invalid_configuration",
            f"{api_key_env} is required for the selected model provider",
        )
    base_url = model_config.base_url
    if not base_url:
        raise AdapterConfigError(
            "codex_invalid_configuration",
            "selected model base_url is required for a custom "
            "Responses-compatible provider",
        )
    return {
        "model_providers": {
            provider: {
                "name": provider,
                "base_url": base_url.rstrip("/"),
                "env_key": api_key_env,
                "wire_api": "responses",
            }
        }
    }


def openai_model_provider_config(config: AgentConfig) -> dict[str, Any]:
    model_config = _selected_model_config(config)
    if model_config.provider != "openai":
        return {}
    base_url = model_config.base_url
    return {"openai_base_url": base_url.rstrip("/")} if base_url else {}


def sandbox(config: AgentConfig) -> Sandbox:
    value = _settings(config).get("sandbox", "read-only")
    try:
        return SANDBOXES[value]
    except (KeyError, TypeError) as error:
        raise AdapterConfigError(
            "codex_invalid_configuration",
            f"sandbox must be one of: {', '.join(sorted(SANDBOXES))}",
        ) from error


def approval_mode(config: AgentConfig) -> ApprovalMode:
    value = _settings(config).get("approval_mode", "auto_review")
    try:
        return APPROVAL_MODES[value]
    except (KeyError, TypeError) as error:
        raise AdapterConfigError(
            "codex_invalid_configuration",
            f"approval_mode must be one of: {', '.join(sorted(APPROVAL_MODES))}",
        ) from error


def timeout_seconds() -> float:
    value = DEFAULT_TIMEOUT_SECONDS
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
    config: AgentConfig,
    context: RuntimeContext,
    base_dir: str,
    *,
    relay_gateway_url: str | None = None,
) -> dict[str, str]:
    values = dict.fromkeys(os.environ, "")
    values.update(
        {name: os.environ[name] for name in INHERITED_ENV_NAMES if name in os.environ}
    )
    telemetry_env = context.telemetry.env if context.telemetry else {}
    values.update(telemetry_env)
    model_config = _selected_model_config(config)
    api_key_env = model_config.api_key_env
    if api_key_env is not None and api_key_env in os.environ:
        values[api_key_env] = os.environ[api_key_env]
    configured = context.environment.env
    values.update(configured)
    if (
        model_config.provider == "openai"
        and api_key_env is not None
        and api_key_env in values
    ):
        values["OPENAI_API_KEY"] = values[api_key_env]
    if model_config.provider != "openai":
        codex_home = state_dir(context, base_dir) / "custom-provider-home"
        values["CODEX_HOME"] = str(codex_home)
    # The SDK overlays this mapping on the parent environment. An empty
    # originator is still treated as an override by Codex and produces invalid
    # initialize metadata ("/<version>"). Use the official SDK client identity
    # without inheriting the identity of a parent Codex process.
    values["CODEX_INTERNAL_ORIGINATOR_OVERRIDE"] = "codex_python_sdk"
    if relay_gateway_url is not None:
        values["NEMO_RELAY_GATEWAY_URL"] = relay_gateway_url
    return values


def _artifact_root(context: RuntimeContext, base_dir: str) -> Path:
    root = context.artifacts.root
    if root:
        return Path(str(root))
    return Path(base_dir) / "artifacts" / "codex"


def state_dir(context: RuntimeContext, base_dir: str) -> Path:
    return _artifact_root(context, base_dir) / ".fabric" / "codex"


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


def _apply_config_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> None:
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
        target[parts[-1]] = _json_value(value, name=f"config_overrides.{dotted_key}")


def native_codex_telemetry_config(context: RuntimeContext) -> dict[str, Any]:
    telemetry = context.telemetry
    if telemetry is None or "native" not in telemetry.metadata.get("telemetry_providers", []):
        return {}

    telemetry_config = telemetry.metadata.get("native_config", {})
    if not isinstance(telemetry_config, dict):
        return {}
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


def prepare_codex_relay(
    payload: dict[str, Any],
    config: AgentConfig,
    context: RuntimeContext,
    base_dir: str,
) -> CodexRelaySettings | None:
    """Generate invocation-scoped Relay gateway configuration."""

    if context.telemetry is None or not context.telemetry.relay_enabled:
        return None
    command = os.environ.get("FABRIC_TEST_NEMO_RELAY_COMMAND", "nemo-relay")
    try:
        executable = relay_gateway.resolve_relay_command(
            Path(base_dir).resolve(), command
        )
    except FileNotFoundError as error:
        raise AdapterRelayError(
            "codex_relay_unavailable", "NeMo Relay CLI executable was not found"
        ) from error

    try:
        relay_contract = relay_gateway.relay_cli_contract(executable)
        plugin_config = common_utils.load_relay_plugin_config(
            payload, model_name=_selected_model_config(config).model
        )
        config_path, plugin_config_path = common_utils.write_relay_configs(
            # Codex execution remains SDK-owned; Relay runs only as a gateway.
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

    base_url = _selected_model_config(config).base_url
    port = relay_gateway.find_available_tcp_port()
    bind = f"127.0.0.1:{port}"
    return CodexRelaySettings(
        gateway=relay_gateway.RelayGatewayLaunch(
            executable=executable,
            config_path=config_path,
            bind=bind,
            url=f"http://{bind}",
            log_path=config_path.parent / "gateway.log",
            openai_base_url=base_url.rstrip("/") if base_url else None,
        ),
        plugin_config=plugin_config,
    )


def thread_config(
    config: AgentConfig,
    context: RuntimeContext,
    relay: CodexRelaySettings | None,
) -> dict[str, Any]:
    """Build request-scoped Codex config without writing a user profile."""

    result = native_codex_telemetry_config(context)
    _merge_config(result, custom_model_provider_config(config, context))
    _merge_config(result, openai_model_provider_config(config))
    mcp_servers = _native_mcp_servers(config)
    if mcp_servers:
        result["mcp_servers"] = mcp_servers
    overrides = _mapping(
        _settings(config).get("config_overrides"),
        name="harness.settings.config_overrides",
    )
    _apply_config_overrides(result, overrides)
    if relay is not None:
        provider = _selected_model_config(config).provider
        transport_config = (
            {"openai_base_url": relay.gateway.url}
            if provider == "openai"
            else {
                "model_providers": {
                    provider: {
                        "base_url": relay.gateway.url,
                    }
                }
            }
        )
        _merge_config(
            result,
            {
                **transport_config,
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
    return result


def sdk_config(
    config: AgentConfig,
    context: RuntimeContext,
    base_dir: str,
    relay: CodexRelaySettings | None,
) -> CodexConfig:
    codex_bin = os.environ.get("FABRIC_TEST_CODEX_BIN")
    if codex_bin:
        path = Path(codex_bin)
        if not path.is_absolute():
            path = (Path(base_dir) / path).resolve()
        codex_bin = str(path)
    return CodexConfig(
        codex_bin=codex_bin,
        cwd=str(resolve_cwd(context, base_dir)),
        env=child_environment(
            config,
            context,
            base_dir,
            relay_gateway_url=relay.gateway.url if relay is not None else None,
        ),
    )


def _personality(config: AgentConfig) -> Personality | None:
    value = _optional_string(_settings(config), "personality")
    if value is None:
        return None
    try:
        return Personality(value)
    except ValueError as error:
        raise AdapterConfigError(
            "codex_invalid_configuration", "personality is invalid"
        ) from error


def _reasoning_effort(config: AgentConfig) -> ReasoningEffort | None:
    value = _optional_string(_settings(config), "reasoning_effort")
    if value is None:
        return None
    try:
        return ReasoningEffort(value)
    except ValueError as error:
        raise AdapterConfigError(
            "codex_invalid_configuration", "reasoning_effort is invalid"
        ) from error


def _output_schema(config: AgentConfig) -> dict[str, Any] | None:
    value = _settings(config).get("output_schema")
    if value is None:
        return None
    return _mapping(_json_value(value, name="output_schema"), name="output_schema")


def validate_runtime_payload(
    config: AgentConfig, context: RuntimeContext, base_dir: str
) -> str:
    """Validate runtime-owned configuration before starting SDK or Relay processes."""

    settings = _settings(config)
    _native_skill_paths(config, base_dir)
    fabric_runtime_id = context.runtime_id
    resolve_cwd(context, base_dir)
    selected_model(config)
    sandbox(config)
    approval_mode(config)
    timeout_seconds()
    for name in (
        "developer_instructions",
        "service_tier",
    ):
        _optional_string(settings, name)
    _personality(config)
    _reasoning_effort(config)
    _output_schema(config)
    child_environment(config, context, base_dir)
    thread_config(config, context, None)
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
    config: AgentConfig,
    context: RuntimeContext,
    base_dir: str,
    *,
    thread_id: str,
    result: Any,
) -> dict[str, Any]:
    status = _json_safe(result.status)
    completed = (
        result.status == TurnStatus.completed and result.final_response is not None
    )
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
        "cwd": str(resolve_cwd(context, base_dir)),
        "model": selected_model(config),
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
        "state_dir": str(state_dir(context, base_dir)),
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


def _thread_options(
    config: AgentConfig,
    context: RuntimeContext,
    base_dir: str,
    relay: CodexRelaySettings | None,
) -> dict[str, Any]:
    settings = _settings(config)
    return {
        "approval_mode": approval_mode(config),
        "base_instructions": (
            config.instructions.system.content
            if config.instructions and config.instructions.system
            else None
        ),
        "config": thread_config(config, context, relay) or None,
        "cwd": str(resolve_cwd(context, base_dir)),
        "developer_instructions": _optional_string(settings, "developer_instructions"),
        "model": selected_model(config),
        "model_provider": _selected_model_config(config).provider,
        "personality": _personality(config),
        "sandbox": sandbox(config),
        "service_tier": _optional_string(settings, "service_tier"),
    }


async def _open_thread(
    codex: AsyncCodex,
    config: AgentConfig,
    context: RuntimeContext,
    base_dir: str,
    *,
    relay: CodexRelaySettings | None,
) -> Any:
    options = _thread_options(config, context, base_dir, relay)
    return await codex.thread_start(**options)


async def _invoke_thread(
    config: AgentConfig,
    context: RuntimeContext,
    base_dir: str,
    invocation: dict[str, Any],
    thread: Any,
) -> tuple[dict[str, Any], bool]:
    """Run one turn and report whether the connected SDK transport remains usable."""

    handle = None
    try:
        async with asyncio.timeout(timeout_seconds()):
            handle = await thread.turn(
                request_prompt(invocation),
                effort=_reasoning_effort(config),
                output_schema=_output_schema(config),
            )
            result = await handle.run()
            return (
                normalize_result(
                    config, context, base_dir, thread_id=thread.id, result=result
                ),
                True,
            )
    except TimeoutError as error:
        await _interrupt_turn(handle)
        return sdk_failure(error), False
    except CodexAdapterError:
        raise
    except (CodexError, RuntimeError, OSError) as error:
        return sdk_failure(error), False


def _relay_output(
    output: dict[str, Any],
    relay: CodexRelaySettings,
    *,
    artifacts: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    output["relay_runtime"] = {
        "enabled": True,
        "emitter": "codex-sdk/nemo-relay",
        "config_path": os.environ.get("FABRIC_RELAY_CONFIG_PATH"),
        "gateway_config_path": str(relay.gateway.config_path),
        "gateway_url": relay.gateway.url,
        "gateway_log_path": str(relay.gateway.log_path),
    }
    output["relay_artifacts"] = (
        common_utils.collect_relay_artifacts(relay.plugin_config)
        if artifacts is None
        else artifacts
    )
    return output


def _start_relay_gateway(
    context: RuntimeContext,
    base_dir: str,
    relay: CodexRelaySettings | None,
) -> subprocess.Popen[Any] | None:
    if relay is None:
        return None
    try:
        return relay_gateway.start_relay_gateway(
            launch=relay.gateway, cwd=resolve_cwd(context, base_dir)
        )
    except relay_gateway.RelayGatewayError as error:
        raise AdapterRelayError(
            "codex_relay_start_failed",
            "NeMo Relay gateway failed to start",
            metadata={"gateway_log_path": str(relay.gateway.log_path)},
        ) from error


def _cleanup_relay(
    relay: CodexRelaySettings | None,
    process: subprocess.Popen[Any] | None,
) -> AdapterRelayError | None:
    if process is None:
        return None
    try:
        relay_gateway.stop_relay_gateway(process)
    except relay_gateway.RelayGatewayError:
        return AdapterRelayError(
            "codex_relay_stop_failed",
            "NeMo Relay gateway failed to stop",
            metadata={
                "gateway_log_path": str(relay.gateway.log_path)
                if relay is not None
                else ""
            },
        )
    return None


def _as_lifecycle_error(error: CodexAdapterError) -> lifecycle.LifecycleError:
    return lifecycle.LifecycleError(
        error.code,
        error.message,
        metadata=error.metadata,
    )


def _runtime_context(payload: dict[str, Any]) -> RuntimeContext:
    try:
        return RuntimeContext.from_mapping(payload.get("runtime_context"))
    except ContractValidationError as error:
        raise lifecycle.LifecycleError(
            "codex_invalid_runtime_context",
            "Codex runtime context is invalid",
        ) from error


class CodexRuntime:
    """One Codex app-server client and thread owned by a Fabric runtime."""

    def __init__(self) -> None:
        self._config: AgentConfig | None = None
        self._context: RuntimeContext | None = None
        self._base_dir: str | None = None
        self._fabric_runtime_id: str | None = None
        self._client: AsyncCodex | None = None
        self._thread: Any = None
        self._relay: CodexRelaySettings | None = None
        self._gateway_process: subprocess.Popen[Any] | None = None
        self._unusable = False

    async def start(self, payload: dict[str, Any]) -> None:
        if self._client is not None:
            raise lifecycle.LifecycleError(
                "codex_runtime_already_started",
                "Codex runtime is already started",
            )

        try:
            agent_config = payload.get("config")
            if not isinstance(agent_config, AgentConfig):
                raise lifecycle.LifecycleError(
                    "codex_invalid_config", "Codex requires a validated AgentConfig"
                )
            context = _runtime_context(payload)
            base_dir = common_utils.base_dir(payload)
            fabric_runtime_id = validate_runtime_payload(agent_config, context, base_dir)
            relay = prepare_codex_relay(payload, agent_config, context, base_dir)
            self._relay = relay
            self._gateway_process = _start_relay_gateway(context, base_dir, relay)
            client_config = sdk_config(agent_config, context, base_dir, relay)
            if _selected_model_config(agent_config).provider != "openai":
                await asyncio.to_thread(
                    Path(client_config.env["CODEX_HOME"]).mkdir,
                    parents=True,
                    exist_ok=True,
                )
            client = AsyncCodex(config=client_config)
            self._client = client
            await _register_skill_roots(client, _native_skill_paths(agent_config, base_dir))
            thread = await _open_thread(
                client,
                agent_config,
                context,
                base_dir,
                relay=relay,
            )
        except CodexAdapterError as error:
            await self._cleanup_failed_start()
            raise _as_lifecycle_error(error) from error
        except (CodexError, RuntimeError, OSError) as error:
            await self._cleanup_failed_start()
            reported = sdk_failure(error)["error"]
            raise lifecycle.LifecycleError(
                reported["code"],
                reported["message"],
                retryable=reported["retryable"],
                metadata=reported.get("metadata"),
            ) from error
        except BaseException:
            await self._cleanup_failed_start()
            raise

        self._config = agent_config
        self._context = context
        self._base_dir = base_dir
        self._fabric_runtime_id = fabric_runtime_id
        self._thread = thread

    async def invoke(self, invocation: dict[str, Any]) -> dict[str, Any]:
        if (
            self._config is None
            or self._context is None
            or self._base_dir is None
            or self._client is None
            or self._thread is None
            or self._fabric_runtime_id is None
        ):
            raise lifecycle.LifecycleError(
                "codex_runtime_not_started",
                "Codex runtime is not started",
            )
        config = self._config
        context = self._context
        base_dir = self._base_dir
        runtime_context = _runtime_context(invocation)
        if runtime_context.runtime_id != self._fabric_runtime_id:
            raise lifecycle.LifecycleError(
                "codex_runtime_mismatch",
                "Codex invocation does not match the connected runtime",
            )
        if self._unusable:
            return _failure(
                "codex_runtime_unavailable",
                "Codex runtime cannot accept another invocation after a runtime failure",
            )

        try:
            request_prompt(invocation)
            timeout_seconds()
            _reasoning_effort(config)
            _output_schema(config)
            relay = self._relay
            atif_before = (
                relay_artifacts.snapshot_atif_files(relay.plugin_config)
                if relay is not None
                and relay_artifacts.expects_local_atif(relay.plugin_config)
                else None
            )
            output, usable = await _invoke_thread(
                config, context, base_dir, invocation, self._thread
            )
            if (
                output.get("completed")
                and relay is not None
                and atif_before is not None
            ):
                finalized = await relay_artifacts.wait_for_finalized_atif(
                    relay.plugin_config, atif_before
                )
                if finalized is None:
                    self._unusable = True
                    return _relay_output(
                        adapter_failure(
                            AdapterRelayError(
                                "codex_relay_atif_timeout",
                                "NeMo Relay did not finalize an ATIF artifact before the deadline",
                                metadata={
                                    "timeout_seconds": relay_artifacts.ATIF_FINALIZATION_TIMEOUT_SECONDS,
                                },
                            )
                        ),
                        relay,
                        artifacts=[],
                    )
        except AdapterRelayError as error:
            output = adapter_failure(error)
            usable = False
        except CodexAdapterError as error:
            output = adapter_failure(error)
            usable = True

        self._unusable = not usable
        if self._relay is not None:
            output = _relay_output(output, self._relay)
        return output

    async def stop(self) -> None:
        client = self._client
        self._client = None
        self._config = None
        self._context = None
        self._base_dir = None
        self._thread = None
        self._fabric_runtime_id = None
        self._unusable = True

        close_error: BaseException | None = None
        try:
            if client is not None:
                await client.close()
        except BaseException as error:
            if not isinstance(error, asyncio.CancelledError):
                LOGGER.exception("Codex SDK client failed to close")
            close_error = error
        finally:
            cleanup_error = _cleanup_relay(self._relay, self._gateway_process)
            self._relay = None
            self._gateway_process = None

        if isinstance(close_error, asyncio.CancelledError):
            raise close_error
        if close_error is not None:
            if cleanup_error is not None:
                LOGGER.error(
                    "Codex Relay cleanup also failed during close: %s",
                    cleanup_error.code,
                )
            raise lifecycle.LifecycleError(
                "codex_sdk_stop_failed",
                "Codex SDK runtime failed to stop",
            ) from close_error
        if cleanup_error is not None:
            raise _as_lifecycle_error(cleanup_error)

    async def _cleanup_failed_start(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            try:
                await client.close()
            except Exception:
                LOGGER.exception("Codex SDK cleanup after start failure also failed")
        cleanup_error = _cleanup_relay(self._relay, self._gateway_process)
        self._relay = None
        self._gateway_process = None
        if cleanup_error is not None:
            LOGGER.error(
                "Codex Relay cleanup after start failure also failed: %s",
                cleanup_error.code,
            )


def main() -> None:
    """Serve the persistent local-host lifecycle protocol."""

    lifecycle.serve(CodexRuntime, config_loader=AgentConfig.from_mapping)


if __name__ == "__main__":
    main()
