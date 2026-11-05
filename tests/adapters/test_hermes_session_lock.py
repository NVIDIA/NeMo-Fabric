# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native interface writers retain the inherited lock after their owner exits."""

import asyncio
import os
import sys
from unittest.mock import AsyncMock

import pytest

from nemo_fabric_adapters.hermes import interfaces

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="POSIX inherited writer lock"
)


async def test_interface_child_retains_state_lock_until_process_exit(
    monkeypatch, tmp_path, unused_tcp_port
):
    import fcntl

    guard = (tmp_path / "state.lock").open("w")
    fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    create_process = asyncio.create_subprocess_exec

    async def spawn_writer(*args, **kwargs):
        return await create_process(
            sys.executable, "-c", "import time; time.sleep(60)", **kwargs
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn_writer)
    monkeypatch.setattr(interfaces, "_wait_healthy", AsyncMock())
    children = interfaces.InterfaceChildren()
    try:
        await children.start(
            {"api": unused_tcp_port},
            hermes_home=tmp_path,
            token_path=tmp_path / "api.token",
            timeout=5,
            environment={**os.environ, "FABRIC_STATE_LOCK_FD": str(guard.fileno())},
        )
        guard.close()
        with (tmp_path / "state.lock").open("w") as competing:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competing.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        await children.stop()
        with (tmp_path / "state.lock").open("w") as replacement:
            fcntl.flock(replacement.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        guard.close()
        await children.stop()
