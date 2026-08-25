#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP client adapter for OpenAI and Anthropic-compatible remote agents."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import httpx
from nemo_fabric_adapter_contract import models as contract
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common.relay_gateway import RelaySettings
from nemo_fabric_adapters.common import relay_gateway
from nemo_fabric_adapters.common import utils as common_utils


DEFAULT_API_TYPE = "openai-responses"
DEFAULT_ANTHROPIC_MAX_TOKENS = 4096
API_PATHS = {
    "openai-responses": "/responses",
    "openai-completions": "/chat/completions",
    "anthropic-messages": "/messages",
}


def _base_url(value: str) -> str:
    return f"{value.rstrip('/')}/"


def _relay_base_url(value: str) -> str:
    return _base_url(f"{value.rstrip('/')}/v1")


def _api_url(base_url: str, api_type: str) -> str:
    return f"{base_url.rstrip('/')}{API_PATHS[api_type]}"


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


async def _sse_events(response: httpx.Response):
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


class RemoteAgentRuntime:
    """One HTTP client and transcript owned by a Fabric runtime."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._endpoint: str | None = None
        self._config: contract.AgentConfig | None = None
        self._runtime_id: str | None = None
        self._api_type = DEFAULT_API_TYPE
        self._messages: list[dict[str, str]] = []
        self._relay: RelaySettings | None = None
        self._gateway_process: subprocess.Popen[Any] | None = None

    async def start(self, payload: dict[str, Any]) -> None:
        config: contract.AgentConfig = payload["config"]
        context = contract.RuntimeContext.from_mapping(payload["runtime_context"])
        settings = config.harness.settings
        self._api_type = settings.get("api_type", DEFAULT_API_TYPE)
        base_url = settings["base_url"]
        try:
            self._relay = self._prepare_relay(config, context, payload, base_url)
            if self._relay is not None:
                self._gateway_process = relay_gateway.start_relay_gateway(
                    launch=self._relay.gateway,
                    cwd=Path(common_utils.base_dir(payload)),
                )
                base_url = _relay_base_url(self._relay.gateway.url)

            model = config.models["default"]
            headers: dict[str, str] = {}
            if model.api_key_env is not None:
                credential = os.environ[model.api_key_env]
                if self._api_type == "anthropic-messages":
                    headers["x-api-key"] = credential
                else:
                    headers["authorization"] = f"Bearer {credential}"
            if self._api_type == "anthropic-messages":
                headers["anthropic-version"] = "2023-06-01"

            self._endpoint = _api_url(base_url, self._api_type)
            self._client = httpx.AsyncClient(headers=headers, timeout=None)
            self._config = config
            self._runtime_id = context.runtime_id
        except Exception:
            await self.stop()
            raise

    async def invoke(
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
        try:
            if self._api_type == "openai-responses":
                text, usage = await self._invoke_responses(user_text)
            elif self._api_type == "openai-completions":
                text, usage = await self._invoke_completions(user_text)
            else:
                text, usage = await self._invoke_messages(user_text)
        except httpx.HTTPStatusError as error:
            return self._result_error(error.response.status_code)
        except httpx.RequestError as error:
            raise lifecycle.LifecycleError(
                "remote_agent_transport_failed",
                "Remote agent request could not be completed",
                retryable=True,
            ) from error
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise lifecycle.LifecycleError(
                "remote_agent_invalid_response",
                "Remote agent returned an invalid response",
            ) from error
        except RuntimeError as error:
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
        output: dict[str, Any] = {"response": text}
        if self._relay is not None:
            output["relay_runtime"] = {
                "enabled": True,
                "emitter": "httpx/nemo-relay",
                "gateway_url": self._relay.gateway.url,
                "gateway_log_path": str(self._relay.gateway.log_path),
            }
            output["relay_artifacts"] = common_utils.collect_relay_artifacts(
                self._relay.plugin_config
            )
        return contract.AgentRunResult(
            status=contract.AgentRunStatus.SUCCEEDED,
            output=output,
            usage=usage,
        )

    async def stop(self) -> None:
        client, self._client = self._client, None
        self._endpoint = None
        self._config = None
        self._runtime_id = None
        self._messages = []
        if client is not None:
            await client.aclose()
        process, self._gateway_process = self._gateway_process, None
        relay, self._relay = self._relay, None
        if process is not None:
            try:
                relay_gateway.stop_relay_gateway(process)
            except relay_gateway.RelayGatewayError as error:
                raise lifecycle.LifecycleError(
                    "remote_agent_relay_stop_failed",
                    "NeMo Relay gateway failed to stop",
                    metadata={"gateway_log_path": str(relay.gateway.log_path)},
                ) from error

    async def _invoke_responses(
        self, user_text: str
    ) -> tuple[str, contract.AgentUsage | None]:
        config = self._config
        if config is None:
            raise RuntimeError("remote agent runtime is not configured")
        model = config.models["default"]
        payload: dict[str, Any] = {
            "model": model.model,
            "input": [*self._messages, {"role": "user", "content": user_text}],
            "stream": True,
        }
        if config.instructions and config.instructions.system:
            payload["instructions"] = config.instructions.system.content
        if model.temperature is not None:
            payload["temperature"] = model.temperature
        async with self._client.stream(
            "POST", self._endpoint, json=payload
        ) as response:
            response.raise_for_status()
            async for event, value in _sse_events(response):
                if event == "response.completed":
                    completed = value["response"]
                    usage = completed.get("usage", {})
                    return _response_text(completed), _usage(
                        usage.get("input_tokens"),
                        usage.get("output_tokens"),
                        usage.get("total_tokens"),
                    )
        raise RuntimeError("remote agent response ended without completion")

    async def _invoke_completions(
        self, user_text: str
    ) -> tuple[str, contract.AgentUsage | None]:
        config = self._config
        if config is None:
            raise RuntimeError("remote agent runtime is not configured")
        model = config.models["default"]
        messages = list(self._messages)
        if config.instructions and config.instructions.system:
            messages.insert(
                0, {"role": "system", "content": config.instructions.system.content}
            )
        messages.append({"role": "user", "content": user_text})
        payload: dict[str, Any] = {"model": model.model, "messages": messages}
        if model.temperature is not None:
            payload["temperature"] = model.temperature
        response = await self._client.post(self._endpoint, json=payload)
        response.raise_for_status()
        value = response.json()
        usage = value.get("usage", {})
        return value["choices"][0]["message"]["content"], _usage(
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
        )

    async def _invoke_messages(
        self, user_text: str
    ) -> tuple[str, contract.AgentUsage | None]:
        config = self._config
        if config is None:
            raise RuntimeError("remote agent runtime is not configured")
        model = config.models["default"]
        payload: dict[str, Any] = {
            "model": model.model,
            "messages": [*self._messages, {"role": "user", "content": user_text}],
            "max_tokens": model.settings.get(
                "max_tokens", DEFAULT_ANTHROPIC_MAX_TOKENS
            ),
            "stream": True,
        }
        if config.instructions and config.instructions.system:
            payload["system"] = config.instructions.system.content
        if model.temperature is not None:
            payload["temperature"] = model.temperature
        text = ""
        input_tokens = output_tokens = None
        async with self._client.stream(
            "POST", self._endpoint, json=payload
        ) as response:
            response.raise_for_status()
            async for event, value in _sse_events(response):
                if event == "message_start":
                    input_tokens = value["message"]["usage"].get("input_tokens")
                elif event == "content_block_delta":
                    delta = value["delta"]
                    if delta["type"] == "text_delta":
                        text += delta["text"]
                elif event == "message_delta":
                    output_tokens = value["usage"].get("output_tokens")
                elif event == "message_stop":
                    return text, _usage(input_tokens, output_tokens, None)
        raise RuntimeError("remote agent response ended without completion")

    def _prepare_relay(
        self,
        config: contract.AgentConfig,
        context: contract.RuntimeContext,
        payload: dict[str, Any],
        base_url: str,
    ) -> RelaySettings | None:
        if context.telemetry is None or not context.telemetry.relay_enabled:
            return None
        base_dir = Path(common_utils.base_dir(payload)).resolve()
        command = os.environ.get("FABRIC_TEST_NEMO_RELAY_COMMAND", "nemo-relay")
        try:
            executable = relay_gateway.resolve_relay_command(base_dir, command)
            relay_gateway.relay_cli_contract(executable)
            plugin_config = common_utils.load_relay_plugin_config(
                {
                    "agent_name": common_utils.agent_name(payload),
                    "base_dir": str(base_dir),
                    "config": config.to_mapping(),
                    "runtime_context": context.to_mapping(),
                }
            )
            config_path, _plugin_config_path = common_utils.write_relay_configs(
                relay_config={}, plugin_config=plugin_config
            )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            raise lifecycle.LifecycleError(
                "remote_agent_relay_configuration_failed",
                "NeMo Relay runtime configuration is unavailable",
            ) from error
        if config_path is None:
            raise lifecycle.LifecycleError(
                "remote_agent_relay_configuration_failed",
                "NeMo Relay runtime configuration is unavailable",
            )
        port = relay_gateway.find_available_tcp_port()
        upstream = base_url.rstrip("/")
        return RelaySettings(
            gateway=relay_gateway.RelayGatewayLaunch(
                executable=executable,
                config_path=config_path,
                bind=f"127.0.0.1:{port}",
                url=f"http://127.0.0.1:{port}",
                log_path=config_path.parent / "gateway.log",
                openai_base_url=(
                    upstream if self._api_type != "anthropic-messages" else None
                ),
                anthropic_base_url=(
                    upstream.removesuffix("/v1")
                    if self._api_type == "anthropic-messages"
                    else None
                ),
            ),
            plugin_config=plugin_config,
        )

    def _result_error(self, status_code: int) -> contract.AgentRunResult:
        output: dict[str, Any] = {}
        if self._relay is not None:
            output["relay_runtime"] = {"enabled": True, "emitter": "httpx/nemo-relay"}
        return contract.AgentRunResult(
            status=contract.AgentRunStatus.FAILED,
            output=output,
            error=contract.AgentRunError(
                code="remote_agent_http_error",
                message=f"Remote agent returned HTTP status {status_code}",
                retryable=status_code >= 500 or status_code == 429,
            ),
        )


def _usage(
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
) -> contract.AgentUsage | None:
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    return contract.AgentUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def main() -> None:
    lifecycle.serve(RemoteAgentRuntime, config_loader=contract.AgentConfig.from_mapping)


if __name__ == "__main__":
    main()
