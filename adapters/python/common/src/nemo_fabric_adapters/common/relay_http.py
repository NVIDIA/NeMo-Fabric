# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit host trust for credential-free Relay model forwarding."""

from __future__ import annotations

import ipaddress
import re
from typing import Any

from .relay_gateway import RelayCliContract

_HOSTNAME = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*\.?\Z"
)


class RelayHttpModelError(ValueError):
    """Invalid or unsupported explicit model-host trust."""


def relay_http_model_config(
    settings: dict[str, Any], contract: RelayCliContract
) -> dict[str, Any]:
    """Translate explicit harness trust into Relay's upstream configuration."""
    key = "insecure_http_model_hosts"
    if key not in settings:
        return {}
    hosts = settings[key]
    if not isinstance(hosts, list):
        raise RelayHttpModelError(
            f"harness.settings.{key} must be a list of bare hostnames or IP addresses"
        )
    normalized: list[str] = []
    for index, host in enumerate(hosts):
        if not isinstance(host, str) or not host or len(host) > 253:
            raise RelayHttpModelError(
                f"harness.settings.{key}[{index}] must be a bare hostname or IP address"
            )
        if ":" in host:
            try:
                if "%" in host:
                    raise ipaddress.AddressValueError(
                        "scoped addresses are unsupported"
                    )
                host = str(ipaddress.IPv6Address(host))
            except ipaddress.AddressValueError as error:
                raise RelayHttpModelError(
                    f"harness.settings.{key}[{index}] must be a bare IPv6 address without brackets or a zone identifier"
                ) from error
        else:
            if not _HOSTNAME.fullmatch(host):
                raise RelayHttpModelError(
                    f"harness.settings.{key}[{index}] must be a bare hostname or IP address without a scheme, port, path, or wildcard"
                )
            host = host.lower().removesuffix(".")
            # Do not reinterpret numeric DNS spellings as alternate IPv4 addresses.
            if re.fullmatch(r"(?:[0-9]+|0x[0-9a-f]*)", host.rsplit(".", 1)[-1]):
                try:
                    host = str(ipaddress.IPv4Address(host))
                except ipaddress.AddressValueError as error:
                    raise RelayHttpModelError(
                        f"harness.settings.{key}[{index}] must use a canonical IPv4 address"
                    ) from error
        if host not in normalized:
            normalized.append(host)
    if contract.version < (0, 11, 0):
        raise RelayHttpModelError(
            f"harness.settings.{key} requires NeMo Relay CLI >=0.11.0; upgrade Relay or remove the setting"
        )
    return {"upstream": {key: normalized}}
