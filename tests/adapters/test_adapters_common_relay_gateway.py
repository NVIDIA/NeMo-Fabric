# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import subprocess
from unittest.mock import MagicMock, call

import pytest

import nemo_fabric_adapters.common.relay_gateway as relay_gateway


def test_resolve_relay_command_returns_absolute_executable(monkeypatch, tmp_path):
    executable = tmp_path / "bin" / "nemo-relay"
    executable.parent.mkdir()
    executable.touch()
    monkeypatch.setattr(
        relay_gateway.shutil, "which", MagicMock(return_value=str(executable))
    )

    resolved = relay_gateway.resolve_relay_command(tmp_path, "nemo-relay")

    assert resolved == executable.resolve()


def test_resolve_relay_command_treats_tilde_path_as_config_relative(
    monkeypatch, tmp_path
):
    executable = (tmp_path / "~" / "bin" / "nemo-relay").resolve()
    mock_which = MagicMock(return_value=str(executable))
    monkeypatch.setattr(relay_gateway.shutil, "which", mock_which)

    resolved = relay_gateway.resolve_relay_command(tmp_path, "~/bin/nemo-relay")

    assert resolved == executable
    mock_which.assert_called_once_with(str(executable))


def test_resolve_relay_command_rejects_missing_executable(monkeypatch, tmp_path):
    monkeypatch.setattr(relay_gateway.shutil, "which", MagicMock(return_value=None))

    with pytest.raises(
        FileNotFoundError, match="NeMo Relay CLI executable was not found"
    ):
        relay_gateway.resolve_relay_command(tmp_path, "nemo-relay")


@pytest.mark.parametrize(
    ("output", "expected_version"),
    [
        ("nemo-relay 0.6.0-alpha.20260714\n", (0, 6, 0)),
        ("nemo-relay 0.6.99\n", (0, 6, 99)),
    ],
)
def test_relay_cli_contract_selects_compatible_contract(
    monkeypatch, tmp_path, output, expected_version
):
    monkeypatch.setattr(
        relay_gateway.subprocess,
        "run",
        MagicMock(return_value=subprocess.CompletedProcess([], 0, stdout=output)),
    )

    assert relay_gateway.relay_cli_contract(
        tmp_path / "nemo-relay"
    ) == relay_gateway.RelayCliContract(
        version=expected_version,
        observability_version=2,
    )


@pytest.mark.parametrize("output", ["nemo-relay 0.5.9", "nemo-relay 0.7.0"])
def test_relay_cli_contract_rejects_unsupported_version(monkeypatch, tmp_path, output):
    monkeypatch.setattr(
        relay_gateway.subprocess,
        "run",
        MagicMock(return_value=subprocess.CompletedProcess([], 0, stdout=output)),
    )

    with pytest.raises(
        relay_gateway.RelayGatewayError, match="Fabric requires >=0.6.0,<0.7.0"
    ):
        relay_gateway.relay_cli_contract(tmp_path / "nemo-relay")


def test_relay_cli_contract_rejects_unparseable_output(monkeypatch, tmp_path):
    monkeypatch.setattr(
        relay_gateway.subprocess,
        "run",
        MagicMock(return_value=subprocess.CompletedProcess([], 0, stdout="unknown")),
    )

    with pytest.raises(
        relay_gateway.RelayGatewayError, match="version could not be determined"
    ):
        relay_gateway.relay_cli_contract(tmp_path / "nemo-relay")


def test_start_relay_gateway_captures_logs_and_waits_for_health(monkeypatch, tmp_path):
    executable = tmp_path / "nemo-relay"
    config_path = tmp_path / "relay-config" / "config.toml"
    config_path.parent.mkdir()
    config_path.write_text('[agents.claude]\ncommand = "claude"\n', encoding="utf-8")
    log_path = config_path.parent / "gateway.log"
    process = MagicMock()
    mock_popen = MagicMock(return_value=process)
    mock_wait = MagicMock()
    monkeypatch.setattr(relay_gateway.subprocess, "Popen", mock_popen)
    monkeypatch.setattr(relay_gateway, "wait_for_relay_gateway", mock_wait)
    launch = relay_gateway.RelayGatewayLaunch(
        executable=executable,
        config_path=config_path,
        bind="127.0.0.1:43210",
        url="http://127.0.0.1:43210",
        log_path=log_path,
        openai_base_url="https://openai.example/v1",
        anthropic_base_url="https://anthropic.example",
    )

    started = relay_gateway.start_relay_gateway(
        launch=launch,
        cwd=tmp_path,
    )

    assert started is process
    assert mock_popen.call_args.args[0] == [
        str(executable),
        "--config",
        str(config_path),
        "--bind",
        "127.0.0.1:43210",
        "--openai-base-url",
        "https://openai.example/v1",
        "--anthropic-base-url",
        "https://anthropic.example",
    ]
    assert mock_popen.call_args.kwargs["cwd"] == tmp_path
    assert mock_popen.call_args.kwargs["stderr"] is subprocess.STDOUT
    assert mock_popen.call_args.kwargs["start_new_session"] is True
    assert mock_popen.call_args.kwargs["stdout"].name == str(log_path)
    mock_wait.assert_called_once_with(process, "http://127.0.0.1:43210/healthz")


def test_start_relay_gateway_stops_failed_process_and_preserves_log(
    monkeypatch, tmp_path
):
    executable = tmp_path / "nemo-relay"
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")
    log_path = tmp_path / "gateway.log"
    process = MagicMock()
    monkeypatch.setattr(
        relay_gateway.subprocess, "Popen", MagicMock(return_value=process)
    )
    monkeypatch.setattr(
        relay_gateway,
        "wait_for_relay_gateway",
        MagicMock(side_effect=relay_gateway.RelayGatewayError("not ready")),
    )
    mock_stop = MagicMock()
    monkeypatch.setattr(relay_gateway, "stop_relay_gateway", mock_stop)
    launch = relay_gateway.RelayGatewayLaunch(
        executable=executable,
        config_path=config_path,
        bind="127.0.0.1:43210",
        url="http://127.0.0.1:43210",
        log_path=log_path,
    )

    with pytest.raises(
        relay_gateway.RelayGatewayError,
        match=re.escape(str(log_path)),
    ):
        relay_gateway.start_relay_gateway(
            launch=launch,
            cwd=tmp_path,
        )

    mock_stop.assert_called_once_with(process)
    assert log_path.exists()


def test_start_relay_gateway_reports_readiness_and_stop_failures(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")
    readiness_error = relay_gateway.RelayGatewayError("not ready")
    stop_error = relay_gateway.RelayGatewayError("could not stop")
    process = MagicMock()
    monkeypatch.setattr(
        relay_gateway.subprocess, "Popen", MagicMock(return_value=process)
    )
    monkeypatch.setattr(
        relay_gateway,
        "wait_for_relay_gateway",
        MagicMock(side_effect=readiness_error),
    )
    mock_stop = MagicMock(side_effect=stop_error)
    monkeypatch.setattr(relay_gateway, "stop_relay_gateway", mock_stop)
    launch = relay_gateway.RelayGatewayLaunch(
        executable=tmp_path / "nemo-relay",
        config_path=config_path,
        bind="127.0.0.1:43210",
        url="http://127.0.0.1:43210",
        log_path=tmp_path / "gateway.log",
    )

    with pytest.raises(
        relay_gateway.RelayGatewayError, match="could not be stopped"
    ) as captured:
        relay_gateway.start_relay_gateway(launch=launch, cwd=tmp_path)

    mock_stop.assert_called_once_with(process)
    assert isinstance(captured.value.__cause__, ExceptionGroup)
    assert captured.value.__cause__.exceptions == (readiness_error, stop_error)


def test_wait_for_relay_gateway_reports_early_exit():
    process = MagicMock()
    process.poll.return_value = 17

    with pytest.raises(relay_gateway.RelayGatewayError, match="status 17"):
        relay_gateway.wait_for_relay_gateway(
            process,
            "http://127.0.0.1:43210/healthz",
        )


def test_wait_for_relay_gateway_times_out():
    process = MagicMock()
    process.poll.return_value = None

    with pytest.raises(relay_gateway.RelayGatewayError, match="did not become ready"):
        relay_gateway.wait_for_relay_gateway(
            process,
            "http://127.0.0.1:43210/healthz",
            timeout=0,
        )


def test_stop_relay_gateway_terminates_then_kills_after_timeout():
    process = MagicMock()
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("nemo-relay", 5), None]

    relay_gateway.stop_relay_gateway(process)

    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.wait.call_args_list == [call(timeout=5), call(timeout=5)]


def test_stop_relay_gateway_is_idempotent_for_exited_process():
    process = MagicMock()
    process.poll.return_value = 0

    relay_gateway.stop_relay_gateway(process)

    process.terminate.assert_not_called()
    process.kill.assert_not_called()
