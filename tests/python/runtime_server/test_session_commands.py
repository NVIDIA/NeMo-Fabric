# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The managed command bridge carries explicit session selection unchanged."""

import json
import asyncio
import os

import pytest

from nemo_fabric import SessionRef
from nemo_fabric.errors import FabricRuntimeError
from nemo_fabric.runtime_server import cli


def test_start_command_reads_an_explicit_session_selector(tmp_path, config):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    session_path = tmp_path / "session.json"
    selector = {
        "schema_version": "fabric.session_ref/v1alpha1",
        "adapter_id": "nvidia.fabric.hermes",
        "session_id": "original-session",
    }
    session_path.write_text(json.dumps(selector))

    request = cli.parse_command(
        [
            "start",
            "--agent",
            "researcher",
            "--config",
            str(config_path),
            "--session-ref",
            str(session_path),
        ]
    )

    assert request["session_ref"] == selector


def test_start_command_reads_files_without_nonblock_flag(tmp_path, config, monkeypatch):
    monkeypatch.delattr(os, "O_NONBLOCK", raising=False)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    request = cli.parse_command(
        ["start", "--agent", "researcher", "--config", str(path)]
    )
    assert request["config"] == config


def test_start_command_accepts_one_bundled_config_and_session(tmp_path, config):
    payload = {
        "config": config,
        "session_ref": {
            "schema_version": "fabric.session_ref/v1alpha1",
            "adapter_id": "nvidia.fabric.hermes",
            "session_id": "original-session",
        },
    }
    path = tmp_path / "start-request.json"
    path.write_text(json.dumps(payload))
    request = cli.parse_command(
        ["start", "--agent", "researcher", "--request", str(path)]
    )
    assert request["config"] == config
    assert request["session_ref"] == payload["session_ref"]


@pytest.mark.parametrize("extra", [{"operation": "stop"}, {"runtime_id": "other"}])
def test_bundled_start_cannot_override_command_routing(tmp_path, config, extra):
    from nemo_fabric.runtime_server.protocol import ProtocolError

    path = tmp_path / "start-request.json"
    path.write_text(json.dumps({"config": config, **extra}))
    with pytest.raises(ProtocolError):
        cli.parse_command(["start", "--agent", "researcher", "--request", str(path)])


async def test_runtime_server_forwards_explicit_session_selection(
    host, host_request, mock_fabric
):
    selector = {
        "schema_version": "fabric.session_ref/v1alpha1",
        "adapter_id": "nvidia.fabric.hermes",
        "session_id": "original-session",
    }
    response = await host.handle({**host_request("start"), "session_ref": selector})

    assert response["status"] == "succeeded"
    assert mock_fabric.start_runtime.await_args.kwargs[
        "session_ref"
    ] == SessionRef.from_mapping(selector)


@pytest.mark.parametrize(
    "code",
    [
        "state_in_use",
        "session_not_found",
        "session_incompatible",
        "session_adapter_mismatch",
        "session_resume_unsupported",
    ],
)
async def test_known_start_rejection_can_be_corrected(
    host, host_request, mock_fabric, mock_runtime, code
):
    mock_fabric.start_runtime.side_effect = [
        FabricRuntimeError("PRIVATE", code=code),
        mock_runtime,
    ]
    rejected = await host.handle(host_request("start"))
    assert rejected["error"]["code"] == code
    assert "PRIVATE" not in str(rejected)
    assert not host.start_uncertain
    assert (await host.handle(host_request("start")))["status"] == "succeeded"


async def test_busy_stop_returns_promptly_and_retains_ownership(
    host, host_request, mock_runtime
):
    await host.handle(host_request("start"))
    entered, release = asyncio.Event(), asyncio.Event()

    async def invoke(**kwargs):
        entered.set()
        await release.wait()
        return mock_runtime.invoke.return_value

    mock_runtime.invoke.side_effect = invoke
    task = asyncio.create_task(host.handle(host_request("invoke", input={})))
    await entered.wait()
    try:
        response = await asyncio.wait_for(host.handle(host_request("stop")), 0.5)
        assert response["error"]["code"] == "runtime_busy"
        assert host.runtimes["runtime-1"].runtime is mock_runtime
        mock_runtime.stop.assert_not_awaited()
    finally:
        release.set()
        await task
    assert (await host.handle(host_request("stop")))["status"] == "succeeded"


async def test_runtime_server_accepts_an_omitted_session_reference(
    host, host_request, mock_fabric
):
    response = await host.handle(host_request("start"))
    assert response["status"] == "succeeded"
    assert "session_ref" not in mock_fabric.start_runtime.await_args.kwargs


async def test_runtime_server_rejects_mode_wrapper_before_start(
    host, host_request, mock_fabric
):
    response = await host.handle(
        {**host_request("start"), "session_ref": {"mode": "create"}}
    )
    assert response["status"] == "failed"
    mock_fabric.start_runtime.assert_not_awaited()


async def test_unconfirmed_start_cleanup_retains_sandbox_ownership(
    host, host_request, mock_fabric
):
    """A cleanup failure keeps subsequent starts blocked and preserves ownership."""
    mock_fabric.start_runtime.side_effect = FabricRuntimeError(
        "PRIVATE cleanup detail", code="cleanup_unconfirmed"
    )
    failed = await host.handle(host_request("start"))
    assert failed["error"]["effects"] == "unknown"
    assert "PRIVATE" not in str(failed)
    assert host.start_uncertain
    assert host.has_used_runtime
    retry = await host.handle(host_request("start"))
    assert retry["error"]["code"] == "start_uncertain"
    assert mock_fabric.start_runtime.await_count == 1
