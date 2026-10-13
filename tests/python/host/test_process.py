# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command processes exercise the real SDK, socket, and graceful signal cleanup."""

import asyncio
import json
import os
import shutil
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

import pytest

if sys.platform == "win32":
    pytest.skip(
        "The Unix command bridge requires a Unix platform", allow_module_level=True
    )
from nemo_fabric import Fabric, FabricConfig
from _utils.configs import hermes_shim_config


async def test_command_process_lifecycle_and_signal_cleanup(
    hermes_shim_agent_dir,
):
    with tempfile.TemporaryDirectory(prefix="fh-process-", dir="/tmp") as directory:
        root = Path(directory)
        shutil.copytree(hermes_shim_agent_dir, root, dirs_exist_ok=True)
        socket = root / "fabric.sock"
        environment = {**os.environ, "FABRIC_HOST_SOCKET": str(socket)}
        command = [str(Path(sys.prefix) / "bin" / "fabric-agent-ctl")]
        config = hermes_shim_config().to_mapping()
        name = config["metadata"]["name"]
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config))
        retained = root / "retained"
        retained.write_text("KEEP")

        async def call(operation, *flags, stdin=None):
            process = await asyncio.create_subprocess_exec(
                *command,
                operation,
                *flags,
                env=environment,
                stdin=asyncio.subprocess.PIPE
                if stdin is not None
                else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(stdin), 10)
            assert len(stdout.splitlines()) == 1
            response = json.loads(stdout)
            assert process.returncode == (
                0 if response["status"] == "succeeded" else 1
            ), stderr
            return response

        # Caller-side planning happens before the host; planning creates no agent.
        Fabric().plan(FabricConfig.model_validate(config), base_dir=root)
        assert not socket.exists()
        host = await asyncio.create_subprocess_exec(
            *command,
            "serve",
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            async with asyncio.timeout(10):
                while not socket.exists():
                    assert host.returncode is None
                    await asyncio.sleep(0.02)
            configured = await call(
                "start",
                "--agent",
                name,
                "--config",
                str(config_path),
            )
            assert configured["status"] == "succeeded", configured
            runtime_id = configured["result"]["runtime_id"]
            invoked = await call(
                "invoke",
                "--agent",
                name,
                "--runtime-id",
                runtime_id,
                "--input",
                "-",
                stdin=b'{"message":"hello"}',
            )
            assert invoked["status"] == "succeeded", invoked
            assert invoked["result"]["fabric_result"]["output"]["received"] == {
                "message": "hello"
            }
            stopped = await call(
                "stop",
                "--agent",
                name,
                "--runtime-id",
                runtime_id,
            )
            assert stopped["status"] == "succeeded", stopped
            assert host.returncode is None
            snapshot = await call(
                "stop",
                "--agent",
                name,
                "--runtime-id",
                runtime_id,
            )
            assert snapshot["changed"] is False
            replacement_start = await call(
                "start", "--agent", name, "--config", str(config_path)
            )
            assert replacement_start["error"]["code"] == "runtime_exists"
            host.terminate()
            stdout, stderr = await asyncio.wait_for(host.communicate(), 6)
            assert host.returncode == 0, stderr
            assert stdout == b""
            assert not socket.exists()
            assert retained.read_text() == "KEEP"
            assert socket.with_suffix(".unclean").is_file()
        finally:
            if host.returncode is None:
                host.kill()
                await host.wait()


@pytest.mark.parametrize(
    "shutdown_mode",
    ["busy", "invoke_timeout"],
)
async def test_busy_shutdown_and_timeout_keep_sandbox_owned(
    blocking_adapter_config, tmp_path, shutdown_mode, process_is_running
):
    with tempfile.TemporaryDirectory(prefix="fh-shutdown-", dir="/tmp") as directory:
        root = Path(directory)
        for name in ("adapter.py", "blocking.fabric-adapter.json"):
            shutil.copyfile(tmp_path / name, root / name)
        (root / "spawn-harness").write_text("spawn")
        if shutdown_mode == "invoke_timeout":
            blocking_adapter_config.runtime.timeout_seconds = 0.1
        socket = root / "fabric.sock"
        environment = {**os.environ, "FABRIC_HOST_SOCKET": str(socket)}
        command = [str(Path(sys.prefix) / "bin" / "fabric-agent-ctl")]
        config_path = root / "config.json"
        config_path.write_text(json.dumps(blocking_adapter_config.to_mapping()))
        retained = root / "retained"
        retained.write_text("KEEP")
        host = await asyncio.create_subprocess_exec(
            *command,
            "serve",
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        invocation = None

        async def call(operation, *flags, expected_exit=0):
            process = await asyncio.create_subprocess_exec(
                *command,
                operation,
                *flags,
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), 10)
            assert process.returncode == expected_exit, stderr
            return json.loads(stdout)

        try:
            async with asyncio.timeout(10):
                while not socket.exists():
                    assert host.returncode is None
                    await asyncio.sleep(0.02)
            started = await call(
                "start",
                "--agent",
                "blocking",
                "--config",
                str(config_path),
            )
            input_path = root / "input.json"
            input_path.write_text('{"message":"slow review"}')
            invocation = await asyncio.create_subprocess_exec(
                *command,
                "invoke",
                "--agent",
                "blocking",
                "--runtime-id",
                started["result"]["runtime_id"],
                "--input",
                str(input_path),
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            async with asyncio.timeout(5):
                while not (root / "entered").exists():
                    await asyncio.sleep(0.01)
            pid = int((root / "adapter.pid").read_text())
            if shutdown_mode == "invoke_timeout":
                invoke_stdout, _ = await asyncio.wait_for(invocation.communicate(), 5)
                assert invocation.returncode == 1
                assert json.loads(invoke_stdout)["error"]["code"] == "invoke_failed"
                with pytest.raises(ProcessLookupError):
                    os.kill(pid, 0)
                # Native timeout teardown now terminates the adapter's tree,
                # including a harness that creates a separate process session.
                assert not process_is_running(int((root / "harness.pid").read_text()))
                stopped = await call(
                    "stop",
                    "--agent",
                    "blocking",
                    "--runtime-id",
                    started["result"]["runtime_id"],
                )
                assert stopped["status"] == "succeeded"
                replacement_runtime = await call(
                    "start",
                    "--agent",
                    "blocking",
                    "--config",
                    str(config_path),
                    expected_exit=1,
                )
                assert replacement_runtime["error"]["code"] == "runtime_exists"
                assert (
                    replacement_runtime["changed"] is False
                    and replacement_runtime["error"]["effects"] == "none"
                )
            begin = asyncio.get_running_loop().time()
            host.terminate()
            stdout, stderr = await asyncio.wait_for(host.communicate(), 6)
            assert host.returncode == (0 if shutdown_mode == "invoke_timeout" else 1), (
                stderr
            )
            if shutdown_mode == "busy":
                assert b"cleanup is unconfirmed" in stderr
            assert asyncio.get_running_loop().time() - begin < 5
            assert stdout == b""
            if shutdown_mode == "busy":
                assert process_is_running(int((root / "harness.pid").read_text()))
            else:
                assert not process_is_running(int((root / "harness.pid").read_text()))
            assert not socket.exists()
            assert socket.with_suffix(".unclean").is_file()
            assert retained.read_text() == "KEEP"
            await asyncio.wait_for(invocation.communicate(), 2)
            replacement = await asyncio.create_subprocess_exec(
                *command,
                "serve",
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            replacement_stdout, replacement_stderr = await asyncio.wait_for(
                replacement.communicate(), 3
            )
            assert replacement.returncode == 1
            assert replacement_stdout == b""
            assert b"cleanup is unconfirmed" in replacement_stderr
            assert socket.with_suffix(".unclean").is_file()
            assert not socket.exists()
        finally:
            if (root / "harness.pid").exists():
                with suppress(ProcessLookupError):
                    os.kill(int((root / "harness.pid").read_text()), 9)
            (root / "release").write_text("release")
            if host.returncode is None:
                host.kill()
                await host.wait()
            if invocation is not None and invocation.returncode is None:
                invocation.kill()
                await invocation.wait()
            # Release the controlled adapter before removing its workspace if
            # this test failed before SDK cleanup could run.
            if (root / "adapter.pid").exists():
                pid = int((root / "adapter.pid").read_text())
                for _ in range(100):
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                    await asyncio.sleep(0.01)
