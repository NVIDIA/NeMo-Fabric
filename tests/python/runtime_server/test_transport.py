# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host spec: exclusive socket ownership, response framing, and uncertain calls."""

import asyncio
import json
import os
import stat
from pathlib import Path
from unittest.mock import MagicMock

import sys

import pytest

if sys.platform == "win32":
    pytest.skip(
        "The Unix command bridge requires a Unix platform", allow_module_level=True
    )

from nemo_fabric.runtime_server import protocol, transport


async def test_unreachable_host_does_not_claim_state_or_effects(
    socket_path, host_request
):
    response = await transport.client(host_request("start"), socket_path=socket_path)
    assert response["status"] == "failed"
    assert response["result"] is None
    assert response["changed"] is False
    assert response["error"]["effects"] == "none"


@pytest.mark.parametrize("runtime_id", [[], {}, 7, None, ""])
async def test_malformed_start_response_preserves_unknown_dispatched_effects(
    socket_path, host_request, runtime_id
):
    async def peer(reader, writer):
        await reader.readline()
        response = protocol.envelope(
            "start",
            {
                "generation": "new",
                "host_state": "ready",
                "runtimes": {},
                "runtime_id": runtime_id,
            },
            changed=True,
        )
        writer.write(json.dumps(response).encode() + b"\n")
        await writer.drain()
        writer.close()

    server = await asyncio.start_unix_server(peer, path=socket_path)
    async with server:
        response = await transport.client(
            host_request("start"), socket_path=socket_path
        )
    assert response["error"]["code"] == "invalid_response"
    assert response["changed"] is None
    assert response["error"]["effects"] == "unknown"


async def test_serve_has_exclusive_private_socket_and_preserves_data(
    host, host_request, mock_runtime, socket_path
):
    stop, ready = asyncio.Event(), asyncio.Event()
    retained = socket_path.parent / "retained"
    retained.write_text("keep")
    task = asyncio.create_task(
        transport.serve(
            host, socket_path=socket_path, stop_event=stop, ready_event=ready
        )
    )
    try:
        await asyncio.wait_for(ready.wait(), 1)
        for relative in ("tmp", "workspace", "artifacts"):
            assert (host.directory / relative).is_dir()
            assert stat.S_IMODE((host.directory / relative).stat().st_mode) == 0o700
        inode = socket_path.stat().st_ino
        assert stat.S_IMODE(socket_path.stat().st_mode) == 0o600
        response = await transport.client(
            host_request("start"), socket_path=socket_path
        )
        assert response["status"] == "succeeded"
        with pytest.raises(OSError):
            await transport.serve(
                host, socket_path=socket_path, stop_event=asyncio.Event()
            )
        assert socket_path.stat().st_ino == inode
        response = await transport.client(
            host_request("start"), socket_path=socket_path
        )
        assert response["error"]["code"] == "runtime_exists"
        assert len(host.runtimes) == 1
    finally:
        stop.set()
        await asyncio.wait_for(task, 2)
    assert not socket_path.exists()
    assert retained.read_text() == "keep"
    mock_runtime.stop.assert_awaited_once_with()
    assert socket_path.with_suffix(".unclean").exists()


@pytest.mark.parametrize(
    "reply",
    [
        b"",
        b'{"operation":"invoke"}',
        b"{}\n",
        b'{"x":1,"x":2}\n',
        b" " * (protocol.RESPONSE_LIMIT + 1) + b"\n",
    ],
    ids=["missing", "truncated", "malformed", "duplicate", "oversized"],
)
async def test_missing_malformed_and_truncated_responses_are_unknown_and_never_replayed(
    socket_path, host_request, reply
):
    calls = []

    async def peer(reader, writer):
        calls.append(await reader.readline())
        writer.write(reply)
        try:
            await writer.drain()
        except ConnectionError:
            pass
        finally:
            writer.close()

    server = await asyncio.start_unix_server(peer, path=socket_path)
    async with server:
        response = await transport.client(
            host_request("invoke", input={"prompt": "PRIVATE"}),
            socket_path=socket_path,
            timeout=0.2,
        )
    assert response["status"] == "failed"
    assert response["changed"] is None
    assert response["error"]["effects"] == "unknown"
    assert len(calls) == 1
    assert "PRIVATE" not in str(response)


async def test_client_timeout_does_not_cancel_or_replay_host_invocation(
    host, host_request, mock_runtime, socket_path
):
    await host.handle(host_request("start"))
    entered, release, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    native_result = mock_runtime.invoke.return_value

    async def invoke(**kwargs):
        entered.set()
        await release.wait()
        completed.set()
        return native_result

    mock_runtime.invoke.side_effect = invoke
    stop, ready = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(
        transport.serve(
            host, socket_path=socket_path, stop_event=stop, ready_event=ready
        )
    )
    try:
        await asyncio.wait_for(ready.wait(), 1)
        response = await transport.client(
            host_request("invoke", input={"prompt": "hello"}),
            socket_path=socket_path,
            timeout=0.1,
        )
        assert entered.is_set()
        assert response["error"]["effects"] == "unknown"
        refused = await transport.client(host_request("start"), socket_path=socket_path)
        assert refused["error"]["code"] == "runtime_exists"
        assert len(host.runtimes) == 1
        release.set()
        await asyncio.wait_for(completed.wait(), 1)
        mock_runtime.invoke.assert_awaited_once()
    finally:
        release.set()
        stop.set()
        await asyncio.wait_for(task, 2)


async def test_cleanup_removes_only_owned_socket(host, socket_path):
    stop, ready = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(
        transport.serve(
            host, socket_path=socket_path, stop_event=stop, ready_event=ready
        )
    )
    try:
        await asyncio.wait_for(ready.wait(), 1)
        socket_path.unlink()
        socket_path.write_text("replacement")
        marker = socket_path.with_suffix(".unclean")
        marker.unlink()
        marker.write_text("caller replacement marker")
    finally:
        stop.set()
        await asyncio.wait_for(task, 2)
    assert socket_path.read_text() == "replacement"
    assert marker.read_text() == "caller replacement marker"


@pytest.mark.parametrize("changed_field", ["st_ctime_ns", "st_size"])
async def test_cleanup_preserves_replacement_marker_when_inode_is_reused(
    host, socket_path, monkeypatch, changed_field
):
    stop, ready = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(
        transport.serve(
            host, socket_path=socket_path, stop_event=stop, ready_event=ready
        )
    )
    marker = socket_path.with_suffix(".unclean")
    try:
        await asyncio.wait_for(ready.wait(), 1)
        original = marker.lstat()
        marker.unlink()
        marker.write_text("caller replacement marker")
        # Model a filesystem reusing the old inode for the caller's new file.
        replacement = MagicMock(spec=os.stat_result)
        for field in ("st_dev", "st_ino", "st_ctime_ns", "st_size"):
            setattr(replacement, field, getattr(original, field))
        setattr(replacement, changed_field, getattr(original, changed_field) + 1)
        original_lstat = Path.lstat

        def lstat(path, *args, **kwargs):
            if path == marker:
                return replacement
            return original_lstat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "lstat", lstat)
    finally:
        stop.set()
        await asyncio.wait_for(task, 2)
    assert marker.read_text() == "caller replacement marker"


@pytest.mark.parametrize("operation", ["start", "invoke", "stop"])
async def test_connections_accepted_during_shutdown_receive_no_effects_response(
    host, host_request, mock_fabric, mock_runtime, socket_path, operation
):
    stop, ready = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(
        transport.serve(
            host, socket_path=socket_path, stop_event=stop, ready_event=ready
        )
    )
    try:
        await asyncio.wait_for(ready.wait(), 1)
        # Exercise the transport while shutdown is flagged and the socket is open.
        host.shutting_down = True
        request = host_request(operation)
        if operation == "invoke":
            request["input"] = {"prompt": "hello"}
        response = await transport.client(request, socket_path=socket_path)
        assert response["operation"] == operation
        assert response["status"] == "failed"
        assert response["error"]["code"] == "host_stopping"
        assert response["changed"] is False
        assert response["error"]["effects"] == "none"
        mock_fabric.start_runtime.assert_not_awaited()
        mock_runtime.invoke.assert_not_awaited()
        mock_runtime.stop.assert_not_awaited()
    finally:
        stop.set()
        await asyncio.wait_for(task, 2)


async def test_server_refuses_restart_when_previous_cleanup_is_unconfirmed(
    host, socket_path
):
    marker = socket_path.with_suffix(".unclean")
    marker.write_text("previous host may still own adapter processes")
    with pytest.raises(protocol.ProtocolError) as caught:
        await asyncio.wait_for(
            transport.serve(host, socket_path=socket_path, stop_event=asyncio.Event()),
            0.1,
        )
    assert caught.value.code == "cleanup_unconfirmed"
    assert marker.exists()
    assert not socket_path.exists()


async def test_shutdown_closes_idle_connections_before_waiting_for_server_close(
    host, socket_path
):
    stop, ready = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(
        transport.serve(
            host, socket_path=socket_path, stop_event=stop, ready_event=ready
        )
    )
    await ready.wait()
    reader, writer = await asyncio.open_unix_connection(socket_path)
    try:
        stop.set()
        await asyncio.wait_for(task, 0.2)
        assert await asyncio.wait_for(reader.read(), 0.2) == b""
        assert not socket_path.with_suffix(".unclean").exists()
    finally:
        writer.close()
        await writer.wait_closed()


async def test_shutdown_deadline_is_bounded_and_rejects_new_work(
    host, host_request, mock_runtime, socket_path, monkeypatch
):
    await host.handle(host_request("start"))
    entered = asyncio.Event()

    async def stop_runtime():
        entered.set()
        await asyncio.Event().wait()

    mock_runtime.stop.side_effect = stop_runtime
    monkeypatch.setattr(transport, "SHUTDOWN_SECONDS", 0.05)
    stop, ready = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(
        transport.serve(
            host, socket_path=socket_path, stop_event=stop, ready_event=ready
        )
    )
    await asyncio.wait_for(ready.wait(), 1)
    stop.set()
    await asyncio.wait_for(entered.wait(), 1)
    response = await host.handle(host_request("invoke", input={"prompt": "hello"}))
    assert response["error"]["code"] == "host_stopping"
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(task, 1)
    assert not socket_path.exists()
    mock_runtime.invoke.assert_not_awaited()


async def test_startup_preserves_caller_file_at_socket_path(host, socket_path):
    socket_path.write_text("keep")
    with pytest.raises(OSError):
        await transport.serve(host, socket_path=socket_path, stop_event=asyncio.Event())
    assert socket_path.read_text() == "keep"
