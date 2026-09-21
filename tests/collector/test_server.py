# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_fabric_collector import server as collector_server


async def test_serve_collector_configures_completion_wait_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_application = MagicMock()
    mock_create_app = MagicMock(return_value=mock_application)
    mock_config = MagicMock(backlog=2048)
    mock_listener = MagicMock()
    mock_listener.getsockname.return_value = ("127.0.0.1", 43123)
    mock_server = MagicMock(started=True)
    mock_server.serve = AsyncMock()
    monkeypatch.setattr(collector_server, "create_app", mock_create_app)
    monkeypatch.setattr(
        collector_server.uvicorn,
        "Config",
        MagicMock(return_value=mock_config),
    )
    monkeypatch.setattr(
        collector_server.socket,
        "create_server",
        MagicMock(return_value=mock_listener),
    )
    monkeypatch.setattr(
        collector_server,
        "_EmbeddedServer",
        MagicMock(return_value=mock_server),
    )

    async with collector_server.serve_collector(
        standalone=True,
        completion_wait_timeout=2.5,
    ) as base_url:
        assert base_url == "http://127.0.0.1:43123"

    mock_create_app.assert_called_once_with(
        standalone=True,
        completion_wait_timeout=2.5,
    )
    mock_listener.close.assert_called_once_with()
