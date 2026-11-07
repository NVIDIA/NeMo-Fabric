# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private attachment and managed browser ownership regressions."""

import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_fabric_adapters.hermes import terminal

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Hermes native interfaces require POSIX"
)


@pytest.fixture(name="connection")
def connection_fixture(tmp_path):
    connection = tmp_path / "tui.json"
    connection.write_text(
        json.dumps(
            {
                "version": 1,
                "mode": "headless",
                "port": 19119,
                "token_file": "tui.token",
                "workspace": str(tmp_path),
            }
        )
    )
    connection.chmod(0o600)
    token = tmp_path / "tui.token"
    token.write_text("test-private-token")
    token.chmod(0o600)
    return connection


def test_terminal_refuses_public_credentials(connection, monkeypatch):
    probe = AsyncMock(return_value=True)
    monkeypatch.setattr(terminal, "tui_ready", probe)
    connection.with_name("tui.token").chmod(0o644)
    with pytest.raises(ValueError, match="owner-only"):
        terminal.attachment_environment(connection)
    probe.assert_not_called()


def test_terminal_refuses_symlink_credentials(connection, tmp_path):
    token = connection.with_name("tui.token")
    private = tmp_path / "other-token"
    token.rename(private)
    token.symlink_to(private)
    with pytest.raises(OSError):
        terminal.attachment_environment(connection)


def test_terminal_stale_backend_does_not_spawn(connection, monkeypatch):
    monkeypatch.setattr(terminal, "tui_ready", AsyncMock(return_value=False))
    execute = MagicMock()
    monkeypatch.setattr(terminal.os, "execvpe", execute)
    with pytest.raises(ValueError, match="unavailable"):
        terminal.attachment_environment(connection)
    execute.assert_not_called()


def test_terminal_credentials_stay_in_environment(connection, monkeypatch):
    monkeypatch.setattr(terminal, "tui_ready", AsyncMock(return_value=True))
    os.environ["HERMES_TUI_SIDECAR_URL"] = "ws://unrelated-server"
    environment = terminal.attachment_environment(connection)
    assert "test-private-token" in environment["HERMES_TUI_GATEWAY_URL"]
    assert environment["HERMES_TUI_CWD"] == str(connection.parent)
    assert "HERMES_TUI_SIDECAR_URL" not in environment
    assert "test-private-token" not in connection.read_text()


@pytest.mark.usefixtures("requires_hermes_agent")
def test_browser_default_profile_attaches_to_owned_backend(tmp_path, monkeypatch):
    from hermes_cli import main_tui_launch, web_server_chat
    from nemo_fabric_adapters.hermes import native

    os.environ["HERMES_HOME"] = str(tmp_path)
    command = ["/usr/bin/node", "--expose-gc", str(tmp_path / "dist/entry.js")]
    monkeypatch.setattr(terminal, "frontend_command", lambda: command)
    make_argv = MagicMock(return_value=(command, tmp_path))
    monkeypatch.setattr(main_tui_launch, "_make_tui_argv", make_argv)
    monkeypatch.setattr(
        web_server_chat,
        "_build_gateway_ws_url",
        lambda: "ws://127.0.0.1:19119/api/ws?token=test",
    )
    original = web_server_chat._resolve_chat_argv
    monkeypatch.setattr(web_server_chat, "_resolve_chat_argv", original)
    native.configure_browser_attachment()
    argv, _, environment = web_server_chat._resolve_chat_argv(profile="default")
    assert argv == command
    assert environment["HERMES_TUI_GATEWAY_URL"].startswith("ws://127.0.0.1:19119/")
    with pytest.raises(ValueError, match="runtime's profile"):
        web_server_chat._resolve_chat_argv(profile="other-profile")
    make_argv.assert_called_once()
