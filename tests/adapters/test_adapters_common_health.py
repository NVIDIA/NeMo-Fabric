# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python hosts authenticate health probes independently of invocation traffic."""

import asyncio
import hashlib
import hmac
import io
import json
import os
import secrets
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from nemo_fabric_adapters.common import health, lifecycle
from nemo_fabric_adapter_contract.models import AgentRunResult, AgentRunStatus


@pytest_asyncio.fixture(name="health_channel", loop_scope="function")
async def health_channel_fixture(tmp_path):
    probe = AsyncMock(
        return_value={
            "outcome": "passed",
            "reason_code": None,
            "activity": "unknown",
            "source": "fabric.adapter.openclaw.gateway",
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    server = await health.serve_health("service-1", tmp_path, probe)
    endpoint_path = tmp_path / "health.endpoint.json"
    endpoint = json.loads(endpoint_path.read_text())
    try:
        yield endpoint, probe, endpoint_path, server
    finally:
        await server.close()


@pytest.fixture(name="health_exchange")
def health_exchange_fixture():
    async def exchange(endpoint, *, changes=None, envelope_changes=None):
        payload = json.dumps(
            {
                "runtime_id": "service-1",
                "level": "live",
                "timeout_seconds": 1,
                **(changes or {}),
            }
        )
        nonce = secrets.token_hex(32)
        proof = hmac.new(
            endpoint["token"].encode(),
            f"request\n{nonce}\n{payload}".encode(),
            hashlib.sha256,
        ).hexdigest()
        envelope = {"nonce": nonce, "payload": payload, "proof": proof}
        envelope.update(envelope_changes or {})
        reader, writer = await asyncio.open_connection("127.0.0.1", endpoint["port"])
        try:
            writer.write(json.dumps(envelope).encode() + b"\n")
            await writer.drain()
            wire = await asyncio.wait_for(reader.readline(), 2)
            if not wire:
                return None
            assert endpoint["token"].encode() not in wire
            response = json.loads(wire)
            assert set(response) == {"payload", "proof"}
            expected = hmac.new(
                endpoint["token"].encode(),
                f"response\n{nonce}\n{response['payload']}".encode(),
                hashlib.sha256,
            ).hexdigest()
            assert hmac.compare_digest(response["proof"], expected)
            return json.loads(response["payload"])
        finally:
            writer.close()
            await writer.wait_closed()

    return exchange


async def test_health_channel_has_private_credentials_and_signed_results(
    health_channel, health_exchange
):
    endpoint, probe, path, _ = health_channel
    assert endpoint["protocol_version"] == "fabric.check/v1alpha1"
    assert endpoint["runtime_id"] == "service-1"
    assert len(endpoint["token"]) == 64
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
    result = await health_exchange(endpoint)
    assert result == probe.return_value
    probe.assert_awaited_once_with("live")


@pytest.mark.parametrize(
    "changes",
    [
        {"proof": "0" * 64},
        {"nonce": "invalid"},
        {"proof": "invalid"},
        {"extra": "TEST_ONLY_PRIVATE"},
    ],
)
async def test_unauthenticated_health_requests_never_dispatch_probes(
    health_channel, health_exchange, changes
):
    endpoint, probe, _, _ = health_channel
    assert await health_exchange(endpoint, envelope_changes=changes) is None
    probe.assert_not_awaited()


@pytest.mark.parametrize(
    ("changes", "outcome", "reason"),
    [
        ({"runtime_id": "other-service"}, "unknown", "runtime_unavailable"),
    ],
)
async def test_health_channel_rejects_other_targets(
    health_channel, health_exchange, changes, outcome, reason
):
    endpoint, probe, _, _ = health_channel
    result = await health_exchange(endpoint, changes=changes)
    assert result["outcome"] == outcome
    assert result["reason_code"] == reason
    probe.assert_not_awaited()


@pytest.mark.parametrize("level", ["operational", "unknown-level"])
async def test_health_channel_rejects_unknown_levels(
    health_channel, health_exchange, level
):
    endpoint, probe, _, _ = health_channel
    assert await health_exchange(endpoint, changes={"level": level}) is None
    probe.assert_not_awaited()


async def test_health_deadline_cancels_only_the_probe(health_channel, health_exchange):
    endpoint, probe, _, _ = health_channel
    cancelled = asyncio.Event()

    async def block(level):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    probe.side_effect = block
    result = await health_exchange(endpoint, changes={"timeout_seconds": 0.02})
    assert result["outcome"] == "unknown"
    assert result["reason_code"] == "probe_timeout"
    await asyncio.wait_for(cancelled.wait(), 1)
    probe.side_effect = None
    assert (await health_exchange(endpoint))["outcome"] == "passed"


@pytest.mark.parametrize(
    "value",
    [
        {"raw": "TEST_ONLY_PRIVATE"},
        {
            "outcome": "passed",
            "reason_code": None,
            "activity": "unknown",
            "source": "fabric.adapter.openclaw.gateway",
            "observed_at": None,
        },
        RuntimeError("TEST_ONLY_PRIVATE"),
    ],
)
async def test_health_channel_never_exposes_invalid_or_exception_diagnostics(
    health_channel, health_exchange, value, capsys
):
    endpoint, probe, _, _ = health_channel
    if isinstance(value, Exception):
        probe.side_effect = value
    else:
        probe.return_value = value
    result = await health_exchange(endpoint)
    assert result["outcome"] == "unknown"
    assert result["reason_code"] == "probe_error"
    captured = capsys.readouterr()
    assert "TEST_ONLY_PRIVATE" not in json.dumps(result) + captured.out + captured.err


async def test_health_server_closes_pending_connections_and_removes_credentials(
    health_channel,
    health_exchange,
):
    endpoint, probe, path, server = health_channel
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def block(level):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    probe.side_effect = block
    checking = asyncio.create_task(health_exchange(endpoint))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await server.close()
        assert not path.exists()
        assert await asyncio.wait_for(checking, 1) is None
        assert cancelled.is_set()
        with pytest.raises(OSError):
            await asyncio.open_connection("127.0.0.1", endpoint["port"])
    finally:
        checking.cancel()
        await asyncio.gather(checking, return_exceptions=True)


async def test_health_server_never_overwrites_an_existing_endpoint(tmp_path):
    path = tmp_path / "health.endpoint.json"
    path.write_text("existing endpoint")
    with pytest.raises(OSError):
        await health.serve_health("service-1", tmp_path, AsyncMock())
    assert path.read_text() == "existing endpoint"


@pytest.fixture(name="health_runtime")
def health_runtime_fixture():
    runtime = MagicMock(spec=lifecycle.AdapterRuntime)
    runtime.start = AsyncMock(return_value=None)
    runtime.stop = AsyncMock()
    runtime.check = AsyncMock(
        return_value={
            "outcome": "passed",
            "reason_code": None,
            "activity": "unknown",
            "source": "fabric.adapter.openclaw.gateway",
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return runtime


async def test_python_host_health_remains_available_during_invocation(
    tmp_path, health_runtime, health_exchange
):
    os.environ["FABRIC_HOME"] = str(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()

    async def invoke(request, context):
        entered.set()
        await release.wait()
        return AgentRunResult(status=AgentRunStatus.SUCCEEDED, output="finished")

    health_runtime.invoke.side_effect = invoke
    state = lifecycle._HostState()
    await lifecycle._handle_start(state, lambda: health_runtime, {}, "service-1", None)
    endpoint = json.loads((tmp_path / "health.endpoint.json").read_text())
    invoking = asyncio.create_task(
        lifecycle._handle_invoke(
            state,
            health_runtime,
            {
                "request": {"input": "hello"},
                "runtime_context": {
                    "runtime_id": "service-1",
                    "invocation_id": "invocation-1",
                    "request_id": "request-1",
                    "environment": {
                        "environment_id": "env-1",
                        "provider": "local",
                        "control_location": "external_control",
                        "ownership": "caller_owned",
                    },
                    "artifacts": {},
                },
            },
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert (await health_exchange(endpoint))["outcome"] == "passed"
        assert not invoking.done()
        release.set()
        assert (await invoking)["outcome"]["status"] == "succeeded"
    finally:
        release.set()
        await asyncio.gather(invoking, return_exceptions=True)
        await lifecycle._handle_stop(state, health_runtime)
        await state.health.close()


async def test_python_host_stop_invalidates_pending_health_observations(
    tmp_path, health_runtime, health_exchange
):
    os.environ["FABRIC_HOME"] = str(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()

    async def probe(level):
        entered.set()
        await release.wait()
        return health_runtime.check.return_value

    health_runtime.check.side_effect = probe
    state = lifecycle._HostState()
    await lifecycle._handle_start(state, lambda: health_runtime, {}, "service-1", None)
    endpoint = json.loads((tmp_path / "health.endpoint.json").read_text())
    checking = asyncio.create_task(health_exchange(endpoint))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await lifecycle._handle_stop(state, health_runtime)
        release.set()
        result = await checking
        assert result["outcome"] == "unknown"
        assert result["reason_code"] == "observation_changed"
    finally:
        release.set()
        await asyncio.gather(checking, return_exceptions=True)
        await state.health.close()


async def test_python_host_cleans_up_when_health_channel_startup_fails(
    tmp_path, health_runtime
):
    os.environ["FABRIC_HOME"] = str(tmp_path)
    path = tmp_path / "health.endpoint.json"
    path.write_text("existing endpoint")
    state = lifecycle._HostState()
    with pytest.raises(lifecycle.LifecycleError):
        await lifecycle._handle_start(
            state, lambda: health_runtime, {}, "service-1", None
        )
    health_runtime.stop.assert_awaited_once_with()
    assert state.runtime is None
    assert path.read_text() == "existing endpoint"


async def test_health_cleanup_failure_does_not_skip_runtime_cleanup(
    tmp_path, health_runtime, monkeypatch
):
    os.environ["FABRIC_HOME"] = str(tmp_path)
    server = MagicMock()
    server.close = AsyncMock(side_effect=OSError("test cleanup failure"))
    monkeypatch.setattr(health, "serve_health", AsyncMock(return_value=server))
    start = {
        "operation": "start",
        "payload": {"runtime_context": {"runtime_id": "service-1"}},
    }
    with pytest.raises(OSError):
        await lifecycle._serve(
            lambda: health_runtime,
            config_loader=None,
            input_stream=io.StringIO(json.dumps(start) + "\n"),
            output_stream=io.StringIO(),
        )
    health_runtime.stop.assert_awaited_once_with()
