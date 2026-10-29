# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_fabric_adapters.common.relay_gateway import RelayCliContract
from nemo_fabric_adapters.common.relay_http import (
    RelayHttpModelError,
    relay_http_model_config,
)


def test_relay_http_hosts_preserve_absence_and_explicit_revocation():
    old = RelayCliContract(version=(0, 10, 4))
    new = RelayCliContract(version=(0, 11, 0))
    assert relay_http_model_config({}, old) == {}
    assert relay_http_model_config({"insecure_http_model_hosts": []}, new) == {
        "upstream": {"insecure_http_model_hosts": []}
    }
    assert relay_http_model_config(
        {
            "insecure_http_model_hosts": [
                "MODEL.TEST.",
                "model.test",
                "192.168.1.10",
                "2001:0DB8:0:0::1",
            ]
        },
        new,
    ) == {
        "upstream": {
            "insecure_http_model_hosts": ["model.test", "192.168.1.10", "2001:db8::1"]
        }
    }


@pytest.mark.parametrize(
    "hosts",
    [
        None,
        True,
        "model.test",
        [1],
        [""],
        ["http://model.test"],
        ["model.test:80"],
        ["model.test/v1"],
        ["*.test"],
        ["010.0.0.1"],
        ["10.1"],
        ["0x7f000001"],
        ["0x"],
        ["model.0x"],
        [" secret"],
        ["[2001:db8::1]"],
        ["fe80::1%eth0"],
        ["2001:db8::g"],
        ["user@model.test"],
        ["model.test?x=1"],
        ["model.test#frag"],
        ["-model.test"],
        ["model..test"],
        ["a" * 64 + ".test"],
        [".".join(["a" * 63] * 4)],
        ["1.2.3.4.5"],
        ["256.1.1.1"],
    ],
)
def test_relay_http_hosts_reject_invalid_trust(hosts):
    with pytest.raises(RelayHttpModelError, match="insecure_http_model_hosts"):
        relay_http_model_config(
            {"insecure_http_model_hosts": hosts}, RelayCliContract(version=(0, 11, 0))
        )


@pytest.mark.parametrize("hosts", [[], ["model.test"]])
def test_relay_http_hosts_require_supported_relay(hosts):
    with pytest.raises(RelayHttpModelError, match=">=0.11.0"):
        relay_http_model_config(
            {"insecure_http_model_hosts": hosts}, RelayCliContract(version=(0, 10, 4))
        )


@pytest.mark.parametrize(
    "host", ["model.test:80", "http://model.test", "https://model.test"]
)
def test_relay_http_hosts_diagnose_schemes_and_ports(host):
    with pytest.raises(
        RelayHttpModelError, match="without a scheme, port, path, or wildcard"
    ):
        relay_http_model_config(
            {"insecure_http_model_hosts": [host]}, RelayCliContract(version=(0, 11, 0))
        )
