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

from nemo_fabric_adapter_contract.models import AgentConfig, RuntimeContext
from nemo_fabric_adapters.common.lifecycle import LifecycleError
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
    runtime._service_host = True
    runtime._gateway_url = "http://gateway.test"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
        headers={"Authorization": "Bearer TEST_ONLY_GATEWAY_TOKEN"},
    ) as client:
        runtime._client = client
        yield runtime, responses, requests


@pytest_asyncio.fixture(name="runtime_health", loop_scope="function")
async def runtime_health_fixture(gateway_health, tmp_path):
    runtime, responses, requests = gateway_health
    runtime._service_host = False
    runtime._config = AgentConfig.from_mapping(
        {"models": {"default": {"provider": "test", "model": "test-model"}}}
    )
    runtime._context = RuntimeContext.from_mapping(
        {
            "runtime_id": "runtime-health-test",
            "invocation_id": "invocation-health-test",
            "request_id": "request-health-test",
            "environment": {
                "environment_id": "environment-health-test",
                "provider": "local",
                "control_location": "external_control",
                "ownership": "caller_owned",
                "workspace": str(tmp_path),
                "env": {},
            },
            "artifacts": {},
        }
    )
    runtime._agent_id = "researcher"
    responses["/startupz"] = (200, {"ok": True, "status": "started"})
    responses["/v1/models"] = (
        200,
        {
            "object": "list",
            "data": [{"id": "openclaw/researcher", "object": "model"}],
        },
    )
    responses["/v1/chat/completions"] = httpx.Response(405, headers={"Allow": "POST"})
    yield runtime, responses, requests


@pytest.mark.parametrize(
    ("level", "paths"),
    [
        ("live", ["/healthz"]),
        ("active", ["/startupz", "/v1/models", "/v1/chat/completions"]),
        ("ready", ["/readyz"]),
    ],
)
async def test_runtime_health_observes_the_selected_agent_without_invoking(
    runtime_health, level, paths
):
    runtime, _, requests = runtime_health
    result = await runtime.check(level)
    assert result["outcome"] == "passed"
    assert result["activity"] == "idle"
    assert result["observed_at"] is not None
    assert [request.url.path for request in requests] == paths
    assert all(request.method == "GET" for request in requests)


@pytest.mark.parametrize("models", [[], [{"id": "openclaw/other", "object": "model"}]])
async def test_runtime_active_requires_the_selected_agent_registration(
    runtime_health, models
):
    runtime, responses, requests = runtime_health
    responses["/v1/models"] = (200, {"object": "list", "data": models})
    result = await runtime.check("active")
    assert result["outcome"] == "failed"
    assert result["reason_code"] == "endpoint_unavailable"
    assert [request.url.path for request in requests] == ["/startupz", "/v1/models"]
    runtime._service_host = True
    assert (await runtime.check("active"))["outcome"] == "passed"


@pytest.mark.parametrize(
    ("status", "body", "outcome"),
    [
        (503, {"ok": False, "status": "starting"}, "failed"),
        (200, {"ok": False, "status": "started"}, "unknown"),
        (200, {"ok": True, "status": "starting"}, "unknown"),
        (200, {"ok": 1, "status": "started"}, "unknown"),
        (200, b'{"ok":false,"ok":true,"status":"started"}', "unknown"),
    ],
)
async def test_runtime_active_stops_when_startup_is_unavailable_or_invalid(
    runtime_health, status, body, outcome
):
    runtime, responses, requests = runtime_health
    responses["/startupz"] = (status, body)
    result = await runtime.check("active")
    assert result["outcome"] == outcome
    assert [request.url.path for request in requests] == ["/startupz"]


@pytest.mark.parametrize(
    ("status", "allow", "outcome"),
    [
        (404, None, "failed"),
        (200, None, "failed"),
        (405, None, "unknown"),
        (405, "GET", "unknown"),
    ],
)
async def test_runtime_active_requires_the_chat_endpoint_even_when_models_are_available(
    runtime_health, status, allow, outcome
):
    runtime, responses, requests = runtime_health
    responses["/v1/chat/completions"] = httpx.Response(
        status, headers={} if allow is None else {"Allow": allow}
    )
    result = await runtime.check("active")
    assert result["outcome"] == outcome
    assert [request.url.path for request in requests] == [
        "/startupz",
        "/v1/models",
        "/v1/chat/completions",
    ]
    assert all(request.method == "GET" for request in requests)


async def test_runtime_active_rejects_an_interface_disabled_after_an_invocation_failure(
    runtime_health,
):
    runtime, _, requests = runtime_health
    runtime._gateway_failure = LifecycleError("test_failure", "TEST_ONLY_PRIVATE")
    assert (await runtime.check("live"))["outcome"] == "passed"
    requests.clear()
    result = await runtime.check("active")
    assert result["outcome"] == "failed"
    assert result["reason_code"] == "endpoint_unavailable"
    assert requests == []
    assert "TEST_ONLY_PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize("field", ["_context", "_config"])
async def test_runtime_health_requires_an_initialized_runtime(runtime_health, field):
    runtime, _, requests = runtime_health
    setattr(runtime, field, None)
    result = await runtime.check("live")
    assert result["outcome"] == "unknown"
    assert result["reason_code"] == "runtime_unavailable"
    assert requests == []


async def test_runtime_health_reports_stopped_activity_after_owned_process_exit(
    runtime_health,
):
    runtime, _, requests = runtime_health
    runtime._process = MagicMock(spec=asyncio.subprocess.Process)
    runtime._process.returncode = 1
    result = await runtime.check("live")
    assert result["outcome"] == "failed"
    assert result["reason_code"] == "agent_exited"
    assert result["activity"] == "stopped"
    assert requests == []


@pytest.mark.parametrize("error", [httpx.RemoteProtocolError, httpx.ReadTimeout])
@pytest.mark.parametrize("changed_context", [False, True])
async def test_runtime_health_preserves_process_exit_during_transport_failure(
    runtime_health, error, changed_context
):
    runtime, _, _ = runtime_health
    runtime._process = MagicMock(spec=asyncio.subprocess.Process)
    runtime._process.returncode = None

    async def exit_during_request(request):
        runtime._process.returncode = 1
        if changed_context:
            runtime._context = None
        raise error("TEST_ONLY_PRIVATE", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(exit_during_request)
    ) as client:
        runtime._client = client
        result = await runtime.check("live")
    if changed_context:
        assert result["outcome"] == "unknown"
        assert result["reason_code"] == "observation_changed"
        assert result["activity"] == "unknown"
        assert result["observed_at"] is None
    else:
        assert result["outcome"] == "failed"
        assert result["reason_code"] == "agent_exited"
        assert result["activity"] == "stopped"
        assert result["observed_at"] is not None
    assert "TEST_ONLY_PRIVATE" not in json.dumps(result)


async def test_runtime_health_invalidates_a_changed_native_session(runtime_health):
    runtime, _, _ = runtime_health

    async def replace_context(request):
        runtime._context = None
        return httpx.Response(
            200, stream=httpx.ByteStream(b'{"ok":true,"status":"live"}')
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(replace_context)
    ) as client:
        runtime._client = client
        result = await runtime.check("live")
    assert result["outcome"] == "unknown"
    assert result["reason_code"] == "observation_changed"


async def test_runtime_health_does_not_pass_if_the_interface_fails_during_the_probe(
    runtime_health,
):
    runtime, responses, _ = runtime_health

    async def fail_interface(request):
        runtime._gateway_failure = LifecycleError("test_failure", "TEST_ONLY_PRIVATE")
        value = responses[request.url.path]
        if isinstance(value, httpx.Response):
            return value
        status, body = value
        return httpx.Response(
            status, stream=httpx.ByteStream(json.dumps(body).encode())
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(fail_interface)
    ) as client:
        runtime._client = client
        result = await runtime.check("active")
    assert result["outcome"] == "failed"
    assert result["reason_code"] == "endpoint_unavailable"


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
