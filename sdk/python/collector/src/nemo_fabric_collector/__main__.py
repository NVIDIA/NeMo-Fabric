# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for the NeMo Fabric collector."""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import uvicorn

from nemo_fabric_collector.app import create_app

_MIN_TOKEN_LENGTH = 32
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def main() -> None:
    """Run the collector with a single Uvicorn worker."""

    parser = argparse.ArgumentParser(description="Run the NeMo Fabric collector")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--tls-cert",
        type=Path,
        help="path to a TLS certificate file",
    )
    parser.add_argument(
        "--tls-key",
        type=Path,
        help="path to a TLS key file",
    )
    parser.add_argument(
        "--publish-token-env",
        help="environment variable containing the /v1/atof Bearer token",
    )
    parser.add_argument(
        "--control-token-env",
        help="environment variable containing the control API Bearer token",
    )
    args = parser.parse_args()

    if (args.tls_cert is None) != (args.tls_key is None):
        parser.error("--tls-cert and --tls-key must be specified together")
    if args.host not in _LOOPBACK_HOSTS and (
        args.publish_token_env is None or args.control_token_env is None
    ):
        parser.error(
            "--publish-token-env and --control-token-env are required when "
            "--host is not a loopback address"
        )

    publish_token = _token_from_environment(
        parser,
        option="--publish-token-env",
        variable=args.publish_token_env,
    )
    control_token = _token_from_environment(
        parser,
        option="--control-token-env",
        variable=args.control_token_env,
    )
    if args.tls_cert is None and (
        args.publish_token_env is not None or args.control_token_env is not None
    ):
        warnings.warn(
            "Bearer token authentication is configured without TLS; tokens will "
            "be transmitted in cleartext",
            RuntimeWarning,
            stacklevel=1,
        )
    application = create_app(
        publish_token=publish_token,
        control_token=control_token,
    )
    uvicorn.run(
        application,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        ssl_certfile=str(args.tls_cert) if args.tls_cert is not None else None,
        ssl_keyfile=str(args.tls_key) if args.tls_key is not None else None,
    )


def _token_from_environment(
    parser: argparse.ArgumentParser,
    *,
    option: str,
    variable: str | None,
) -> str | None:
    if variable is None:
        return None

    token = os.environ.get(variable)
    if token is not None:
        token = token.strip()

    if not token:
        parser.error(f"{option} names an unset or empty environment variable")
    if not token.isascii():
        parser.error(f"{option} must reference an ASCII token")
    if len(token) < _MIN_TOKEN_LENGTH:
        parser.error(
            f"{option} must reference a token containing at least "
            f"{_MIN_TOKEN_LENGTH} characters"
        )
    return token


if __name__ == "__main__":
    main()
