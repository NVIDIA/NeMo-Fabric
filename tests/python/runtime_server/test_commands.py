# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host spec: bounded JSON input, strict flags, and one safe stdout response."""

import io
import json
import os

import sys

import pytest

if sys.platform == "win32":
    pytest.skip(
        "The Unix command bridge requires a Unix platform", allow_module_level=True
    )

from nemo_fabric.runtime_server import cli, protocol


@pytest.mark.parametrize(
    "flags", [["--live"], ["--active"], ["--ready"], ["--operational"]]
)
def test_server_check_rejects_agent_levels_before_transport(monkeypatch, capsys, flags):
    from unittest.mock import AsyncMock

    mock_client = AsyncMock()
    monkeypatch.setattr(cli.transport, "client", mock_client)
    assert cli.main(["check", *flags]) == 1
    response = json.loads(capsys.readouterr().out)
    assert response["operation"] == "check"
    assert response["error"]["code"] == "invalid_request"
    assert response["changed"] is False
    assert response["error"]["effects"] == "none"
    mock_client.assert_not_awaited()


def test_start_bootstraps_without_extra_lifecycle_flags(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    request = cli.parse_command(
        ["start", "--agent", "researcher", "--config", str(path)]
    )
    assert request["config"] == config


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["unknown"],
        ["stop", "--agent", "researcher"],
        ["invoke", "--agent", "researcher", "--agent", "researcher"],
        ["start", "--live"],
        ["invoke", "--surprise"],
        ["stop", "--expected-descriptor-fingerprint", "a" * 64],
        [
            "stop",
            "--agent",
            "researcher",
            "--runtime-id",
            "\u0000",
        ],
        [
            "stop",
            "--agent",
            "\u0000",
            "--runtime-id",
            "r",
        ],
        ["configure", "--agent", "researcher", "--config"],
    ],
)
def test_invalid_flags_emit_one_safe_json_response_and_exit_one(arguments, capsys):
    assert cli.main(arguments) == 1
    captured = capsys.readouterr()
    assert captured.out.endswith("\n")
    assert len(captured.out.splitlines()) == 1
    response = json.loads(captured.out)
    assert response["status"] == "failed"
    assert response["error"]["code"] == "invalid_request"
    assert response["error"]["effects"] == "none"


@pytest.mark.parametrize(
    "arguments",
    [["serve", "--agent", "researcher"], ["serve", "--expected-generation", "x"]],
)
def test_serve_errors_use_stderr_without_command_response(arguments, capsys):
    assert cli.main(arguments) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err


@pytest.mark.parametrize(
    "operation,flags,fields",
    [
        (
            "stop",
            [
                "--agent",
                "researcher",
                "--runtime-id",
                "r",
            ],
            {"agent": "researcher", "runtime_id": "r"},
        ),
        ("serve", [], {}),
    ],
)
def test_named_commands_construct_versioned_requests(operation, flags, fields):
    assert cli.parse_command([operation, *flags]) == {
        "interface_version": 2,
        "operation": operation,
        **fields,
    }


def test_config_and_input_files_are_objects_without_wrappers(tmp_path, config):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(config))
    request = cli.parse_command(
        [
            "start",
            "--agent",
            "researcher",
            "--config",
            str(path),
        ]
    )
    assert request["config"] == config
    path.write_text('{"prompt":"hello"}')
    request = cli.parse_command(
        ["invoke", "--agent", "researcher", "--runtime-id", "r", "--input", str(path)]
    )
    assert request["input"] == {"prompt": "hello"}


@pytest.mark.parametrize(
    "fingerprint", ["", "a" * 63, "a" * 64, "a" * 65, "A" * 64, "g" * 64]
)
def test_removed_fingerprint_flag_is_rejected_before_sending_command(
    tmp_path, config, fingerprint
):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(protocol.ProtocolError):
        cli.parse_command(
            [
                "start",
                "--agent",
                "researcher",
                "--config",
                str(path),
                "--expected-descriptor-fingerprint",
                fingerprint,
            ]
        )


@pytest.mark.parametrize(
    "content", [b"{}{}", b'{"x":NaN}', b'{"x":1,"x":2}', b"\xff", b"{"]
)
def test_invalid_json_files_are_rejected_before_commands(tmp_path, content):
    path = tmp_path / "input.json"
    path.write_bytes(content)
    with pytest.raises(protocol.ProtocolError, match="bounded UTF-8 JSON value"):
        cli.parse_command(
            [
                "invoke",
                "--agent",
                "researcher",
                "--runtime-id",
                "r",
                "--input",
                str(path),
            ]
        )


def test_fifo_and_oversized_file_are_rejected_without_blocking(tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    large = tmp_path / "large.json"
    large.write_bytes(b" " * (protocol.REQUEST_LIMIT + 1))
    for path in (fifo, large):
        with pytest.raises(protocol.ProtocolError):
            cli.parse_command(
                [
                    "invoke",
                    "--agent",
                    "researcher",
                    "--runtime-id",
                    "r",
                    "--input",
                    str(path),
                ]
            )


def test_stdin_is_read_only_when_explicitly_selected(monkeypatch):
    stream = io.TextIOWrapper(io.BytesIO(b'{"prompt":"hello"}'))
    monkeypatch.setattr("sys.stdin", stream)
    request = cli.parse_command(
        ["invoke", "--agent", "researcher", "--runtime-id", "r", "--input", "-"]
    )
    assert request["input"] == {"prompt": "hello"}


def test_combined_socket_request_limit_includes_framing():
    request = {
        "interface_version": 2,
        "operation": "invoke",
        "agent": "researcher",
        "runtime_id": "r",
        "input": {"prompt": "x" * protocol.REQUEST_LIMIT},
    }
    with pytest.raises(protocol.ProtocolError) as error:
        protocol.encode_request(request)
    assert error.value.code == "request_too_large"


def test_oversized_response_reports_unknown_effects():
    response = protocol.envelope(
        "invoke", {"output": "x" * protocol.RESPONSE_LIMIT}, changed=None
    )
    encoded = protocol.encode_response(response)
    assert len(encoded) <= protocol.RESPONSE_LIMIT
    result = json.loads(encoded)
    assert result["status"] == "failed"
    assert result["changed"] is None
    assert result["error"]["effects"] == "unknown"


@pytest.mark.parametrize(
    "response",
    [
        {
            "operation": "invoke",
            "status": "succeeded",
            "changed": None,
            "result": None,
            "error": None,
        },
        {
            "operation": "invoke",
            "status": "succeeded",
            "changed": None,
            "result": {"runtime_id": "r", "fabric_result": {"status": "failed"}},
            "error": None,
        },
        {
            "operation": "stop",
            "status": "succeeded",
            "changed": True,
            "result": {},
            "error": None,
        },
        {
            "operation": "check",
            "status": "succeeded",
            "changed": False,
            "result": {"health": None},
            "error": None,
        },
    ],
)
def test_malformed_success_response_cannot_establish_success(response):
    with pytest.raises(protocol.ProtocolError):
        protocol.validate_response(response, response["operation"])


@pytest.mark.parametrize("source", ["unusable_stdin", "missing_file"])
def test_unusable_input_preserves_existing_bridge_error_code(
    source, tmp_path, monkeypatch, capsys
):
    path = str(tmp_path / "missing.json")
    if source == "unusable_stdin":
        monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(b"PRIVATE")))
        path = "-"
    assert (
        cli.main(
            [
                "start",
                "--agent",
                "researcher",
                "--config",
                path,
            ]
        )
        == 1
    )
    response = json.loads(capsys.readouterr().out)
    assert response["error"]["code"] == "invalid_input"
    assert response["error"]["effects"] == "none"
    assert response["changed"] is False
