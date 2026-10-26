# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only observations of the OpenClaw Gateway's native health endpoints."""

import asyncio
import gzip
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio

from nemo_fabric_adapters.openclaw.adapter import OpenClawRuntime


@pytest_asyncio.fixture(name="gateway_health", loop_scope="function")
async def gateway_health_fixture():
    requests = []
    responses = {
        "/healthz": (200, {"ok": True, "status": "live"}),
        "/v1/models": (200, {"object": "list", "data": []}),
        "/readyz": (200, {"ready": True}),
    }

    async def respond(request):
        requests.append(request)
        value = responses[request.url.path]
        if isinstance(value, Exception):
            raise value
        if isinstance(value, httpx.Response):
            return value
        status, body = value
        encoded = body if isinstance(body, bytes) else json.dumps(body).encode()
        return httpx.Response(status, stream=httpx.ByteStream(encoded))

    runtime = OpenClawRuntime()
    runtime._gateway_url = "http://gateway.test"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
        headers={"Authorization": "Bearer TEST_ONLY_GATEWAY_TOKEN"},
    ) as client:
        runtime._client = client
        yield runtime, responses, requests


@pytest.mark.parametrize(
    ("level", "path", "source"),
    [
        ("live", "/healthz", "fabric.adapter.liveness"),
        ("active", "/v1/models", "fabric.adapter.interface"),
        ("ready", "/readyz", "fabric.adapter.readiness"),
    ],
)
async def test_gateway_health_uses_native_read_only_evidence(
    gateway_health, level, path, source
):
    runtime, _, requests = gateway_health

    result = await runtime.check(level)

    assert result["outcome"] == "passed"
    assert result["reason_code"] is None
    assert result["source"] == source
    assert result["observed_at"].endswith("Z")
    assert result["activity"] == "unknown"
    assert [(request.method, request.url.path) for request in requests] == [
        ("GET", path)
    ]
    assert requests[0].headers["authorization"] == "Bearer TEST_ONLY_GATEWAY_TOKEN"
    assert "TEST_ONLY_GATEWAY_TOKEN" not in json.dumps(result)


async def test_gateway_readiness_includes_native_channel_failures(gateway_health):
    runtime, responses, requests = gateway_health
    responses["/readyz"] = (
        503,
        {"ready": False, "failing": ["telegram"], "uptimeMs": 1000},
    )

    assert (await runtime.check("live"))["outcome"] == "passed"
    assert (await runtime.check("active"))["outcome"] == "passed"
    result = await runtime.check("ready")

    assert result["outcome"] == "failed"
    assert result["reason_code"] == "health_failed"
    assert result["source"] == "fabric.adapter.readiness"
    assert result["observed_at"] is not None
    assert [request.url.path for request in requests] == [
        "/healthz",
        "/v1/models",
        "/readyz",
    ]


@pytest.mark.parametrize("status", [401, 403, 404, 503])
async def test_gateway_active_rejects_unavailable_authenticated_api(
    gateway_health, status
):
    runtime, responses, _ = gateway_health
    responses["/v1/models"] = (status, {"error": "TEST_ONLY_PRIVATE_DIAGNOSTIC"})

    result = await runtime.check("active")

    assert result["outcome"] == "failed"
    assert result["reason_code"] == "endpoint_unavailable"
    assert "TEST_ONLY_PRIVATE_DIAGNOSTIC" not in json.dumps(result)


@pytest.mark.parametrize(
    ("level", "path", "status", "body"),
    [
        ("live", "/healthz", 200, b"<html>Gateway UI</html>"),
        ("live", "/healthz", 200, {"ok": True, "status": "started"}),
        ("live", "/healthz", 200, {"ok": 1, "status": "live"}),
        ("live", "/healthz", 200, b'{"ok": false, "ok": true, "status": "live"}'),
        ("active", "/v1/models", 200, {"object": "list", "data": {}}),
        ("active", "/v1/models", 200, {"object": "list", "data": [{}]}),
        ("active", "/v1/models", 200, {"object": "model", "data": []}),
        ("ready", "/readyz", 200, {"ready": "true"}),
        ("ready", "/readyz", 200, {"ready": False}),
        ("ready", "/readyz", 200, b'{"ready": false, "ready": true}'),
        ("ready", "/readyz", 503, {"ready": True}),
        ("ready", "/readyz", 503, {"error": "unavailable"}),
        pytest.param("ready", "/readyz", 200, b"x" * (16 * 1024 + 1), id="oversized"),
    ],
)
async def test_gateway_health_never_accepts_invalid_native_evidence(
    gateway_health, level, path, status, body
):
    runtime, responses, _ = gateway_health
    responses[path] = (status, body)

    result = await runtime.check(level)

    assert result["outcome"] == "unknown"
    assert result["reason_code"] == "probe_error"
    assert result["observed_at"] is None


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (httpx.ReadTimeout("TEST_ONLY_PRIVATE_DIAGNOSTIC"), "probe_timeout"),
        (httpx.ReadError("TEST_ONLY_PRIVATE_DIAGNOSTIC"), "probe_error"),
    ],
)
async def test_gateway_probe_errors_are_fixed_and_do_not_poison_runtime(
    gateway_health, error, reason, caplog
):
    runtime, responses, _ = gateway_health
    responses["/healthz"] = error

    result = await runtime.check("live")

    assert result["outcome"] == "unknown"
    assert result["reason_code"] == reason
    assert runtime._gateway_failure is None
    assert not runtime._gateway_failed.is_set()
    assert "TEST_ONLY_PRIVATE_DIAGNOSTIC" not in json.dumps(result) + caplog.text


async def test_gateway_check_does_not_restart_or_stop_owned_gateway(gateway_health):
    runtime, _, requests = gateway_health
    runtime._process = MagicMock(spec=asyncio.subprocess.Process)
    runtime._process.returncode = 1
    runtime.stop = AsyncMock()

    result = await runtime.check("live")

    assert result["outcome"] == "failed"
    assert result["reason_code"] == "agent_exited"
    assert requests == []
    runtime.stop.assert_not_called()
    runtime._process.terminate.assert_not_called()
    runtime._process.kill.assert_not_called()


async def test_gateway_check_rejects_stopping_connection(gateway_health):
    runtime, _, requests = gateway_health
    runtime._stopping = True

    result = await runtime.check("ready")

    assert result["outcome"] == "unknown"
    assert result["reason_code"] == "observation_changed"
    assert requests == []


async def test_gateway_unknown_level_is_unsupported_without_native_requests(
    gateway_health,
):
    runtime, _, requests = gateway_health

    result = await runtime.check("unknown-level")

    assert result["outcome"] == "unsupported"
    assert result["reason_code"] == "unsupported_level"
    assert requests == []


async def test_gateway_health_never_follows_redirects_with_credentials(gateway_health):
    runtime, responses, requests = gateway_health
    runtime._client.follow_redirects = True
    responses["/v1/models"] = httpx.Response(
        302, headers={"location": "https://elsewhere.test/v1/models"}
    )

    result = await runtime.check("active")

    assert result["outcome"] == "failed"
    assert result["reason_code"] == "endpoint_unavailable"
    assert [request.url.host for request in requests] == ["gateway.test"]


async def test_gateway_health_does_not_decompress_unbounded_responses(gateway_health):
    runtime, responses, _ = gateway_health
    responses["/readyz"] = httpx.Response(
        200,
        headers={"Content-Encoding": "gzip"},
        content=gzip.compress(b'{"ready":true}'),
    )
    result = await runtime.check("ready")
    assert result["outcome"] == "unknown"
    assert result["reason_code"] == "probe_error"


async def test_gateway_probe_cancellation_keeps_the_connection_usable(gateway_health):
    runtime, _, _ = gateway_health
    entered, release = asyncio.Event(), asyncio.Event()

    async def hold(request):
        entered.set()
        await release.wait()
        return httpx.Response(200, json={"ready": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(hold)) as client:
        runtime._client = client
        checking = asyncio.create_task(runtime.check("ready"))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            checking.cancel()
            with pytest.raises(asyncio.CancelledError):
                await checking
            assert not client.is_closed
            assert runtime._gateway_failure is None
            assert not runtime._gateway_failed.is_set()
        finally:
            checking.cancel()
            await asyncio.gather(checking, return_exceptions=True)
