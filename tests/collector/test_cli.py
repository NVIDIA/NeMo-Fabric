# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
from unittest.mock import patch

import pytest

from nemo_fabric_collector import __main__ as collector_cli


@pytest.mark.parametrize(
    ("token", "message"),
    [
        (None, "unset or empty"),
        ("", "unset or empty"),
        ("x" * 31, "at least 32 characters"),
        ("é" * 32, "ASCII token"),
    ],
)
def test_token_from_environment_rejects_invalid_token(
    token: str | None,
    message: str,
    capsys: pytest.CaptureFixture[str],
):
    variable = "NEMO_FABRIC_TEST_TOKEN"
    if token is None:
        os.environ.pop(variable, None)
    else:
        os.environ[variable] = token

    with pytest.raises(SystemExit):
        collector_cli._token_from_environment(
            argparse.ArgumentParser(),
            option="--test-token-env",
            variable=variable,
        )

    assert message in capsys.readouterr().err


def test_token_from_environment_strips_valid_token():
    variable = "NEMO_FABRIC_TEST_TOKEN"
    os.environ[variable] = f"  {'x' * 32}  "

    token = collector_cli._token_from_environment(
        argparse.ArgumentParser(),
        option="--test-token-env",
        variable=variable,
    )

    assert token == "x" * 32


@pytest.mark.parametrize(
    "tls_arguments",
    [
        ["--tls-cert", "certificate.pem"],
        ["--tls-key", "key.pem"],
    ],
)
def test_main_requires_tls_certificate_and_key_together(
    tls_arguments: list[str],
    capsys: pytest.CaptureFixture[str],
):
    with patch.object(sys, "argv", ["nemo-fabric-collector", *tls_arguments]):
        with pytest.raises(SystemExit):
            collector_cli.main()

    assert "must be specified together" in capsys.readouterr().err


def test_main_warns_when_authentication_is_used_without_tls():
    variable = "NEMO_FABRIC_TEST_TOKEN"
    os.environ[variable] = "x" * 32
    arguments = [
        "nemo-fabric-collector",
        "--publish-token-env",
        variable,
    ]

    with (
        patch.object(sys, "argv", arguments),
        patch.object(collector_cli.uvicorn, "run") as mock_run,
        pytest.warns(RuntimeWarning, match="without TLS"),
    ):
        collector_cli.main()

    application = mock_run.call_args.args[0]
    assert application.state.publish_token == "x" * 32
    assert application.state.control_token is None
    assert mock_run.call_args.kwargs["ssl_certfile"] is None
    assert mock_run.call_args.kwargs["ssl_keyfile"] is None


def test_main_passes_tls_files_to_uvicorn():
    arguments = [
        "nemo-fabric-collector",
        "--tls-cert",
        "certificate.pem",
        "--tls-key",
        "key.pem",
    ]

    with (
        patch.object(sys, "argv", arguments),
        patch.object(collector_cli.uvicorn, "run") as mock_run,
    ):
        collector_cli.main()

    assert mock_run.call_args.kwargs["ssl_certfile"] == "certificate.pem"
    assert mock_run.call_args.kwargs["ssl_keyfile"] == "key.pem"
