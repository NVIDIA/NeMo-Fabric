# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Managed OpenClaw interface ownership and private attachment contracts."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.openclaw import interfaces, terminal


@pytest.fixture(name="connection")
def connection_fixture(tmp_path):
    if os.name != "posix":
        pytest.skip("Managed terminal attachment requires POSIX")
    directory = tmp_path / "interfaces"
    interfaces.publish(
        directory,
        runtime_id="test-runtime",
        port=18789,
        token="test-secret-token",
        command=Path(sys.executable).absolute(),
        workspace=str(tmp_path),
        agent_id="reviewer",
        dashboard=False,
        terminal=True,
    )
    return directory / "gateway.json"


@pytest.mark.parametrize(
    "origin",
    ["https://chat.example.com", "http://localhost:1234", "https://[::1]:8443"],
)
def test_public_origins_accept_http_origins(origin):
    assert interfaces.public_origins(
        {"interfaces": {"dashboard": {"public_origins": [origin]}}}
    ) == [origin]


@pytest.mark.parametrize(
    "origin",
    [
        "https://host/path",
        "https://host/",
        "https://host?key=value",
        "https://user:secret@host",
        "javascript:alert(1)",
        "https://host:0",
        "http://host:invalid",
        "https://host#token",
    ],
)
def test_public_origins_reject_non_origins(origin):
    with pytest.raises(lifecycle.LifecycleError, match="HTTP\\(S\\) origins"):
        interfaces.public_origins(
            {"interfaces": {"dashboard": {"public_origins": [origin]}}}
        )


def test_discovery_has_no_secret_and_cleanup_revokes_marker_first(
    connection, monkeypatch
):
    directory = connection.parent
    assert "test-secret-token" not in connection.read_text()
    assert "test-secret-token" not in (directory / "terminal").read_text()
    assert "test-secret-token" not in (directory / "client.json").read_text()
    assert (directory / "terminal").stat().st_mode & 0o777 == 0o700
    for name in ("gateway.json", "gateway.token", "client.json", "terminal.running"):
        assert (directory / name).stat().st_mode & 0o777 == 0o600
    removed = []
    unlink = Path.unlink

    def record(path, **kwargs):
        removed.append(path.name)
        return unlink(path, **kwargs)

    monkeypatch.setattr(Path, "unlink", record)
    assert interfaces.remove(directory)
    assert removed[0] == "terminal.running"
    assert not any((directory / name).exists() for name in interfaces.FILES)


def test_cleanup_retries_failed_removal(connection, monkeypatch):
    unlink = Path.unlink

    def deny(path, **kwargs):
        if path.name == "gateway.token":
            raise PermissionError("denied")
        return unlink(path, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "unlink", deny)
        assert not interfaces.remove(connection.parent)
    assert interfaces.remove(connection.parent)
    assert not connection.with_name("gateway.token").exists()


def test_terminal_uses_configured_gateway_without_secret_argv(connection, monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr(terminal.httpx, "Client", mock_client)
    os.environ["OPENCLAW_GATEWAY_URL"] = "ws://unmanaged.example"
    os.environ["OPENCLAW_GATEWAY_TOKEN"] = "ambient-wrong-token"
    command, environment, workspace = terminal.attachment(connection, "review")
    assert command[1:] == ["tui", "--session", "agent:reviewer:review"]
    assert all("test-secret-token" not in arg for arg in command)
    assert "OPENCLAW_GATEWAY_URL" not in environment
    assert "OPENCLAW_GATEWAY_TOKEN" not in environment
    assert environment["FABRIC_OPENCLAW_CLIENT_TOKEN"] == "test-secret-token"
    assert workspace == str(connection.parent.parent)
    config = json.loads(Path(environment["OPENCLAW_CONFIG_PATH"]).read_text())
    assert config["gateway"]["mode"] == "remote"
    assert config["gateway"]["remote"]["url"] == "ws://127.0.0.1:18789"
    mock_client.return_value.__enter__.return_value.get.assert_called_once_with(
        "http://127.0.0.1:18789/v1/models",
        headers={"Authorization": "Bearer test-secret-token"},
    )


@pytest.mark.parametrize(
    "name", ["gateway.json", "gateway.token", "client.json", "terminal.running"]
)
def test_terminal_rejects_public_or_symlinked_files(connection, name, monkeypatch):
    monkeypatch.setattr(terminal.httpx, "Client", MagicMock())
    file = connection.with_name(name)
    file.chmod(0o644)
    with pytest.raises(ValueError, match="owner-only"):
        terminal.attachment(connection)
    file.chmod(0o600)
    target = file.with_suffix(".target")
    file.rename(target)
    file.symlink_to(target)
    with pytest.raises(OSError):
        terminal.attachment(connection)


@pytest.mark.parametrize("session", ["agent:other:main", "two words", ""])
def test_terminal_rejects_ambiguous_session(connection, session, monkeypatch):
    monkeypatch.setattr(terminal.httpx, "Client", MagicMock())
    with pytest.raises(ValueError, match="session name"):
        terminal.attachment(connection, session)
