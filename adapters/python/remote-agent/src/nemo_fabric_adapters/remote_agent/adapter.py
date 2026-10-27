#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP client adapter for OpenAI and Anthropic-compatible remote agents."""

from __future__ import annotations

import json
import math
import os
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
from nemo_fabric_adapter_contract import models as contract
from nemo_fabric_adapters.common import instructions as common_instructions
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common import provenance
from nemo_fabric_adapters.common import openai_chat
from nemo_fabric_adapters.common import utils as common_utils
from nemo_fabric_adapter_contract.usage import (
    ACCOUNTING_KEY,
    UsageObservation,
    usage_from_observations,
)

_PACKAGE_DISTRIBUTION = "nemo-fabric-adapters-remote-agent"


DEFAULT_API_TYPE = "openai-responses"
DEFAULT_ANTHROPIC_MAX_TOKENS = 4096
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_READ_TIMEOUT_SECONDS = 600.0
API_PATHS = {
    "openai-responses": "/responses",
    "openai-completions": "/chat/completions",
    "anthropic-messages": "/messages",
}
FABRIC_REQUEST_ID_METADATA = "nemo_fabric_request_id"


def _api_url(base_url: str, api_type: str) -> str:
    return f"{base_url.rstrip('/')}{API_PATHS[api_type]}"


def _selected_model(config: contract.AgentConfig) -> contract.AgentModelConfig:
    model = config.models.get("default")
    if model is None and len(config.models) == 1:
        model = next(iter(config.models.values()))
    if model is None:
        raise lifecycle.LifecycleError(
            "remote_agent_missing_model",
            "Remote Agent requires a default model or exactly one model",
        )
    return model


def _timeout_setting(settings: dict[str, Any], name: str, default: float) -> float:
    value = settings.get(name, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise lifecycle.LifecycleError(
            "remote_agent_invalid_configuration",
            f"Remote Agent {name} must be a positive finite number",
            metadata={"field": f"harness.settings.{name}"},
        )
    return float(value)


def _response_text(response: dict[str, Any]) -> str:
    for item in response["output"]:
        if item.get("type") != "message":
            continue
        parts = item.get("content", [])
        text = "".join(
            part["text"] for part in parts if part.get("type") == "output_text"
        )
        if text:
            return text
    raise RuntimeError("remote agent response did not include output text")


async def _sse_events(
    response: httpx.Response,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    event = "message"
    data: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if data:
                yield event, json.loads("\n".join(data))
            event = "message"
            data = []
        elif line.startswith("event:"):
            event = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data.append(line.removeprefix("data:").strip())
    if data:
        yield event, json.loads("\n".join(data))


def _relay_streaming_enabled(
    settings: dict[str, Any],
    context: contract.RuntimeContext,
) -> bool:
    configured = settings.get("relay_streaming", False) is True
    relay_enabled = context.telemetry is not None and context.telemetry.relay_enabled
    if configured != relay_enabled:
        reason = (
            "requires Relay telemetry to be enabled"
            if configured
            else "must be explicitly enabled when Relay telemetry is configured"
        )
        raise lifecycle.LifecycleError(
            "remote_agent_invalid_relay_configuration",
            f"Remote Agent relay_streaming {reason}",
            metadata={"field": "harness.settings.relay_streaming"},
        )
    return configured


class RemoteAgentRuntime:
    """One HTTP client and transcript owned by a NeMo Fabric runtime."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._endpoint: str | None = None
        self._config: contract.AgentConfig | None = None
        self._runtime_id: str | None = None
        self._api_type = DEFAULT_API_TYPE
        self._messages: list[dict[str, str]] = []
        self._relay_streaming = False
        self._model_roles: list[str] = []
        self._package_version = provenance.package_version_fact(_PACKAGE_DISTRIBUTION)

    async def start(self, payload: dict[str, Any]) -> None:
        config: contract.AgentConfig = payload["config"]
        self._model_roles = list(config.models)
        context = contract.RuntimeContext.from_mapping(payload["runtime_context"])
        settings = config.harness.settings if config.harness is not None else {}
        self._api_type = settings.get("api_type", DEFAULT_API_TYPE)
        if self._api_type not in API_PATHS:
            raise lifecycle.LifecycleError(
                "remote_agent_invalid_configuration",
                "Remote Agent api_type is not supported",
                metadata={"field": "harness.settings.api_type"},
            )
        base_url = settings.get("base_url")
        if not isinstance(base_url, str) or not base_url.startswith(
            ("http://", "https://")
        ):
            raise lifecycle.LifecycleError(
                "remote_agent_invalid_configuration",
                "Remote Agent base_url must use HTTP or HTTPS",
                metadata={"field": "harness.settings.base_url"},
            )
        common_instructions.system_instruction(
            config,
            adapter="Remote Agent",
            supported_modes={"replace"},
        )
        self._relay_streaming = _relay_streaming_enabled(settings, context)
        if self._relay_streaming is True and self._api_type == "anthropic-messages":
            # According to a strict reading of https://platform.claude.com/docs/en/api/messages#metadata.user_id
            # It appears that `user_id` is the only metadata field
            raise lifecycle.LifecycleError(
                "remote_agent_invalid_relay_configuration",
                "Remote Agent relay_streaming supports only openai-responses and openai-completions",
                metadata={"field": "harness.settings.api_type"},
            )
        model = _selected_model(config)
        headers: dict[str, str] = {}
        if model.api_key_env is not None:
            try:
                credential = os.environ[model.api_key_env]
            except KeyError as e:
                raise lifecycle.LifecycleError(
                    "remote_agent_missing_api_key",
                    f"Remote agent API key environment variable {model.api_key_env} is not set",
                ) from e
            if self._api_type == "anthropic-messages":
                headers["x-api-key"] = credential
            else:
                headers["authorization"] = f"Bearer {credential}"
        if self._api_type == "anthropic-messages":
            headers["anthropic-version"] = "2023-06-01"

        self._endpoint = _api_url(base_url, self._api_type)
        connect_timeout = _timeout_setting(
            settings,
            "connect_timeout_seconds",
            DEFAULT_CONNECT_TIMEOUT_SECONDS,
        )
        read_timeout = _timeout_setting(
            settings,
            "read_timeout_seconds",
            DEFAULT_READ_TIMEOUT_SECONDS,
        )

        # Attempt to negotiate HTTP/2, this will automatically fall back
        # to HTTP/1.1 if the server does not support it.
        self._client = httpx.AsyncClient(
            headers=headers,
            http2=True,
            timeout=httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=read_timeout,
                pool=connect_timeout,
            ),
        )
        self._config = config
        self._runtime_id = context.runtime_id

    async def invoke(
        self,
        request: contract.AgentRunRequest,
        context: contract.RuntimeContext,
    ) -> contract.AgentRunResult:
        """Run one invocation and report this adapter's package version."""

        result = await self._invoke(request, context)
        result.extensions[provenance.VERSIONS_REPORT_KEY] = (
            provenance.package_version_report(self._package_version, self._model_roles)
        )
        return result

    async def _invoke(
        self,
        request: contract.AgentRunRequest,
        context: contract.RuntimeContext,
    ) -> contract.AgentRunResult:
        if self._client is None or self._config is None:
            raise lifecycle.LifecycleError(
                "remote_agent_not_started", "Remote agent runtime is not started"
            )
        if context.runtime_id != self._runtime_id:
            raise lifecycle.LifecycleError(
                "remote_agent_runtime_mismatch",
                "Remote agent invocation does not match the active runtime",
            )

        user_text = common_utils.normalize_user_input(request.input)
        metadata = (
            {FABRIC_REQUEST_ID_METADATA: context.request_id}
            if self._relay_streaming
            else None
        )
        observed: contract.AgentUsage | None = None

        def remember_usage(value: contract.AgentUsage) -> None:
            nonlocal observed
            observed = value

        def partial_failure(
            code: str, message: str, *, retryable: bool = False
        ) -> contract.AgentRunResult:
            return contract.AgentRunResult(
                status=contract.AgentRunStatus.FAILED,
                output={},
                error=contract.AgentRunError(
                    code=code, message=message, retryable=retryable
                ),
                usage=observed,
            )

        try:
            if self._api_type == "openai-responses":
                text, usage = await self._invoke_responses(
                    user_text, metadata, on_usage=remember_usage
                )
            elif self._api_type == "openai-completions":
                text, usage = await self._invoke_completions(
                    user_text, metadata, on_usage=remember_usage
                )
            else:
                text, usage = await self._invoke_messages(
                    user_text, metadata, on_usage=remember_usage
                )
        except httpx.HTTPStatusError as error:
            return self._result_error(error.response.status_code)
        except httpx.RequestError as error:
            if observed is not None:
                return partial_failure(
                    "remote_agent_transport_failed",
                    "Remote agent request could not be completed",
                    retryable=True,
                )
            raise lifecycle.LifecycleError(
                "remote_agent_transport_failed",
                "Remote agent request could not be completed",
                retryable=True,
            ) from error
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            if observed is not None:
                return partial_failure(
                    "remote_agent_invalid_response",
                    "Remote agent returned an invalid response",
                )
            raise lifecycle.LifecycleError(
                "remote_agent_invalid_response",
                "Remote agent returned an invalid response",
            ) from error
        except RuntimeError as error:
            if observed is not None:
                return partial_failure(
                    "remote_agent_invalid_response",
                    "Remote agent returned an invalid response",
                )
            raise lifecycle.LifecycleError(
                "remote_agent_invalid_response",
                "Remote agent returned an invalid response",
            ) from error

        self._messages.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": text},
            ]
        )
        return contract.AgentRunResult(
            status=contract.AgentRunStatus.SUCCEEDED,
            output={"response": text},
            usage=usage,
        )

    async def stop(self) -> None:
        client, self._client = self._client, None
        self._endpoint = None
        self._config = None
        self._runtime_id = None
        self._relay_streaming = False
        self._messages = []
        if client is not None:
            await client.aclose()

    async def _invoke_responses(
        self,
        user_text: str,
        metadata: dict[str, str] | None,
        *,
        on_usage: Callable[[contract.AgentUsage], None] | None = None,
    ) -> tuple[str, contract.AgentUsage | None]:
        config = self._config
        if config is None:
            raise RuntimeError("remote agent runtime is not configured")
        model = _selected_model(config)
        payload: dict[str, Any] = {
            "model": model.model,
            "input": [*self._messages, {"role": "user", "content": user_text}],
            "stream": True,
        }
        if config.instructions and config.instructions.system:
            payload["instructions"] = config.instructions.system.content
        if model.temperature is not None:
            payload["temperature"] = model.temperature
        if model.top_p is not None:
            payload["top_p"] = model.top_p
        if model.max_tokens is not None:
            payload["max_output_tokens"] = model.max_tokens
        if metadata is not None:
            payload["metadata"] = metadata
        latest_usage: dict[str, Any] = {}
        async with self._client.stream(
            "POST", self._endpoint, json=payload
        ) as response:
            response.raise_for_status()
            async for event, value in _sse_events(response):
                value = _event_object(value, "event")
                report = value.get("response")
                if isinstance(report, dict) and report.get("usage") is not None:
                    raw = _usage_object(report["usage"])
                    candidate = _responses_usage(raw)
                    # Empty or unknown-only objects provide no newer counters.
                    # Reported zero remains usable evidence, not an omission.
                    if candidate.extensions[ACCOUNTING_KEY]["observed"]:
                        latest_usage = raw
                        if on_usage is not None:
                            on_usage(candidate)
                if event == "response.completed":
                    completed = _event_object(value["response"], "response")
                    raw = _usage_object(completed.get("usage"))
                    usage = _responses_usage(raw)
                    if not usage.extensions[ACCOUNTING_KEY]["observed"]:
                        usage = _responses_usage(
                            latest_usage, gap_reason=contract.GapReason.MISSING_USAGE
                        )
                    return _response_text(completed), usage
        raise RuntimeError("remote agent response ended without completion")

    async def _invoke_completions(
        self,
        user_text: str,
        metadata: dict[str, str] | None,
        *,
        on_usage: Callable[[contract.AgentUsage], None] | None = None,
    ) -> tuple[str, contract.AgentUsage | None]:
        config = self._config
        if config is None:
            raise RuntimeError("remote agent runtime is not configured")
        model = _selected_model(config)
        messages = list(self._messages)
        if config.instructions and config.instructions.system:
            messages.insert(
                0, {"role": "system", "content": config.instructions.system.content}
            )
        messages.append({"role": "user", "content": user_text})
        return await openai_chat.invoke(
            self._client,
            self._endpoint,
            model=model.model,
            messages=messages,
            temperature=model.temperature,
            top_p=model.top_p,
            max_tokens=model.max_tokens,
            metadata=metadata,
            on_usage=on_usage,
        )

    async def _invoke_messages(
        self,
        user_text: str,
        metadata: dict[str, str] | None,
        *,
        on_usage: Callable[[contract.AgentUsage], None] | None = None,
    ) -> tuple[str, contract.AgentUsage | None]:
        config = self._config
        if config is None:
            raise RuntimeError("remote agent runtime is not configured")
        model = _selected_model(config)
        payload: dict[str, Any] = {
            "model": model.model,
            "messages": [*self._messages, {"role": "user", "content": user_text}],
            "max_tokens": model.max_tokens
            or model.settings.get("max_tokens", DEFAULT_ANTHROPIC_MAX_TOKENS),
            "stream": True,
        }
        if config.instructions and config.instructions.system:
            payload["system"] = config.instructions.system.content
        if model.temperature is not None:
            payload["temperature"] = model.temperature
        if model.top_p is not None:
            payload["top_p"] = model.top_p
        if metadata is not None:
            payload["metadata"] = metadata
        text = ""
        counts: dict[str, Any] = {}
        async with self._client.stream(
            "POST", self._endpoint, json=payload
        ) as response:
            response.raise_for_status()
            async for event, value in _sse_events(response):
                if event == "message_start":
                    raw = _usage_object(
                        _event_object(value["message"], "message").get("usage")
                    )
                    counts.update(
                        {
                            key: raw[key]
                            for key in (
                                "input_tokens",
                                "output_tokens",
                                "cache_read_input_tokens",
                                "cache_creation_input_tokens",
                            )
                            if key in raw
                        }
                    )
                elif event == "content_block_delta":
                    delta = value["delta"]
                    if delta["type"] == "text_delta":
                        text += delta["text"]
                elif event == "message_delta":
                    # Message deltas are cumulative snapshots, not additive calls.
                    counts.update(_usage_object(value.get("usage")))
                elif event == "message_stop":
                    return text, _usage(
                        counts.get("input_tokens"),
                        counts.get("output_tokens"),
                        None,
                        cached_input_tokens=counts.get("cache_read_input_tokens"),
                        cache_write_input_tokens=counts.get(
                            "cache_creation_input_tokens"
                        ),
                        input_tokens_include_cache=False,
                    )
                if (
                    on_usage is not None
                    and counts
                    and event in ("message_start", "message_delta")
                ):
                    on_usage(
                        _usage(
                            counts.get("input_tokens"),
                            counts.get("output_tokens"),
                            None,
                            cached_input_tokens=counts.get("cache_read_input_tokens"),
                            cache_write_input_tokens=counts.get(
                                "cache_creation_input_tokens"
                            ),
                            input_tokens_include_cache=False,
                        )
                    )
        raise RuntimeError("remote agent response ended without completion")

    def _result_error(self, status_code: int) -> contract.AgentRunResult:
        return contract.AgentRunResult(
            status=contract.AgentRunStatus.FAILED,
            output={},
            error=contract.AgentRunError(
                code="remote_agent_http_error",
                message=f"Remote agent returned HTTP status {status_code}",
                retryable=status_code >= 500 or status_code == 429,
            ),
        )


def _event_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"remote agent {name} must be an object")
    return value


def _usage_object(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("remote agent usage must be an object")
    return value


def _responses_usage(
    raw: dict[str, Any],
    *,
    gap_reason: contract.GapReason = contract.GapReason.SCOPE_UNVERIFIED,
) -> contract.AgentUsage:
    details = _usage_object(raw.get("input_tokens_details"))
    return _usage(
        raw.get("input_tokens"),
        raw.get("output_tokens"),
        raw.get("total_tokens"),
        cached_input_tokens=details.get("cached_tokens"),
        input_tokens_include_cache=True,
        gap_reason=gap_reason,
    )


def _usage(
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
    *,
    cached_input_tokens: int | None = None,
    cache_write_input_tokens: int | None = None,
    input_tokens_include_cache: bool = True,
    gap_reason: contract.GapReason = contract.GapReason.SCOPE_UNVERIFIED,
) -> contract.AgentUsage:
    return usage_from_observations(
        [
            UsageObservation(
                reported=contract.ReportedUsage(
                    input_tokens=input_tokens,
                    cached_input_tokens=cached_input_tokens,
                    cache_write_input_tokens=cache_write_input_tokens,
                    input_tokens_include_cache=input_tokens_include_cache,
                    output_tokens=output_tokens,
                    output_tokens_include_reasoning=True,
                    total_tokens=total_tokens,
                ),
                gap=gap_reason,
                tokens_aligned=True,
            )
        ]
    )


def main() -> None:
    lifecycle.serve(RemoteAgentRuntime, config_loader=contract.AgentConfig.from_mapping)


if __name__ == "__main__":
    main()
