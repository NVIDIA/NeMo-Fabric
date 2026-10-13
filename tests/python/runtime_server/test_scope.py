# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_fabric.runtime_server import cli


def test_serve_remains_the_server_bootstrap():
    assert cli.parse_command(["serve"]) == {
        "interface_version": 2,
        "operation": "serve",
    }


async def test_stop_ends_the_sandbox_lifecycle(
    host, host_request, mock_fabric, mock_runtime
):
    assert (await host.handle(host_request("start")))["status"] == "succeeded"
    assert (await host.handle(host_request("invoke", input={"message": "review"})))[
        "status"
    ] == "succeeded"
    assert (await host.handle(host_request("stop")))["status"] == "succeeded"
    rejected = await host.handle(host_request("start"))
    assert rejected["status"] == "failed"
    assert rejected["error"]["effects"] == "none"
    assert mock_fabric.start_runtime.await_count == 1
    mock_runtime.stop.assert_awaited_once_with()
