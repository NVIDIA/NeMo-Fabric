# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relay-backed streaming E2E for the Remote Agent adapter."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import pytest
import requests
import uvicorn

from examples.code_review_agent import hermes_config, with_relay
from nemo_fabric import (
    EnvironmentConfig,
    Fabric,
    FabricConfig,
    HarnessConfig,
    MetadataConfig,
    ModelConfig,
    RelayAtofConfig,
    RelayAtofStreamSinkConfig,
    RelayObservabilityConfig,
    RunRequest,
    Runtime,
    RuntimeConfig,
)

pytestmark = pytest.mark.usefixtures("requires_hermes_agent")

_REQUEST_ID_METADATA = "nemo_fabric_request_id"
_STREAM_SINK_NAME = "nemo-fabric-stream"


@contextmanager
def _remote_hermes_server(
    *,
    port: int,
    runtime: Runtime,
    runtime_loop: asyncio.AbstractEventLoop,
    received_request_ids: list[str],
) -> Iterator[str]:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        payload = await request.json()
        request_id = payload["metadata"][_REQUEST_ID_METADATA]
        received_request_ids.append(request_id)
        messages = payload.get("messages") or []
        user_messages = [
            message
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        user_input = user_messages[-1]["content"]
        invocation = asyncio.run_coroutine_threadsafe(
            runtime.invoke(
                request=RunRequest(input=user_input, request_id=request_id)
            ),
            runtime_loop,
        )
        result = await asyncio.wrap_future(invocation)
        assert result.status == "succeeded", result.to_mapping()
        return JSONResponse(
            {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion",
                "created": 0,
                "model": payload.get("model", "remote-hermes"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result.output["response"],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline:
            try:
                response = requests.get(f"{base_url}/health", timeout=1)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError("Remote Hermes test server did not become healthy")
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError("Remote Hermes test server did not stop")


def _stream_sink(url: str) -> RelayAtofStreamSinkConfig:
    return RelayAtofStreamSinkConfig(
        name=_STREAM_SINK_NAME,
        url=url,
        transport="ndjson",
        timeout_millis=10_000,
    )


@pytest.mark.usefixtures("mock_nvidia_api_key", "nemo_relay")
async def test_remote_agent_streams_two_correlated_hermes_invocations(
    api_server: str,
    code_review_agent_dir: Path,
    repo_root: Path,
    tmp_path: Path,
    unused_tcp_port_factory,
):
    os.environ["ADAPTER_PYTHON"] = sys.executable
    collector_url = f"http://127.0.0.1:{unused_tcp_port_factory()}/atof"
    remote_port = unused_tcp_port_factory()

    remote_config = FabricConfig(
        metadata=MetadataConfig(name="remote-agent-streaming-e2e"),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.remote-agent",
            resolution="preinstalled",
            settings={
                "base_url": f"http://127.0.0.1:{remote_port}/v1",
                "api_type": "openai-completions",
                "relay_streaming": True,
            },
        ),
        models={"default": ModelConfig(provider="test", model="fabric-echo")},
        runtime=RuntimeConfig(
            input_schema="text",
            output_schema="message",
            artifacts=tmp_path / "remote-artifacts",
        ),
        environment=EnvironmentConfig(
            provider="local",
            workspace=tmp_path,
            artifacts=tmp_path / "remote-artifacts",
        ),
    ).enable_relay(
        observability=RelayObservabilityConfig(
            atof=RelayAtofConfig(
                enabled=True,
                sinks=[_stream_sink(collector_url)],
            )
        )
    )

    remote_hermes_config = with_relay(hermes_config())
    remote_hermes_config.models["default"].base_url = f"{api_server}/v1"
    assert remote_hermes_config.relay is not None
    assert remote_hermes_config.relay.observability is not None
    assert remote_hermes_config.relay.observability.atof is not None
    assert remote_hermes_config.relay.observability.atof.sinks is not None
    remote_hermes_config.relay.observability.atof.sinks.append(
        _stream_sink(collector_url)
    )

    request_ids = ["remote-request-1", "remote-request-2"]
    received_request_ids: list[str] = []
    records_by_request: dict[str, list[dict[str, object]]] = {}
    results = []
    runtime_loop = asyncio.get_running_loop()

    # The independently deployed service is already running before Fabric binds
    # the ATOF listener, matching the expected production startup order.
    async with await Fabric().start_runtime(
        remote_hermes_config,
        base_dir=code_review_agent_dir,
    ) as hermes_runtime:
        with _remote_hermes_server(
            port=remote_port,
            runtime=hermes_runtime,
            runtime_loop=runtime_loop,
            received_request_ids=received_request_ids,
        ):
            async with await Fabric().start_runtime(
                remote_config,
                base_dir=repo_root,
                streaming=True,
            ) as remote_runtime:
                for index, request_id in enumerate(request_ids, start=1):
                    stream = remote_runtime.invoke_stream(
                        request=RunRequest(
                            input=f"remote turn {index}",
                            request_id=request_id,
                        )
                    )
                    records_by_request[request_id] = [
                        record async for record in stream
                    ]
                    results.append(await stream.result())

    assert received_request_ids == request_ids
    assert all(result.status == "succeeded" for result in results), [
        result.to_mapping() for result in results
    ]
    for request_id, other_request_id in zip(
        request_ids, reversed(request_ids), strict=True
    ):
        records = records_by_request[request_id]
        assert records
        encoded_records = json.dumps(records)
        assert request_id in encoded_records
        assert other_request_id not in encoded_records
