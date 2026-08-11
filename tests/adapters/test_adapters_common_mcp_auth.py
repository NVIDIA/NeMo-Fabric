# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest
from nemo_fabric_adapters.common import mcp_auth


def test_parse_oauth2_config_normalizes_fields():
    config = mcp_auth.parse_oauth2_config(
        "docs",
        {
            "type": "oauth2",
            "client_id": "fabric-client",
            "client_secret_env": "FABRIC_MCP_CLIENT_SECRET",
            "scopes": ["read", "write"],
            "redirect_uri": "http://127.0.0.1:8765/callback",
        },
    )

    assert config == mcp_auth.McpOAuth2Config(
        client_id="fabric-client",
        client_secret_env="FABRIC_MCP_CLIENT_SECRET",
        scopes=("read", "write"),
        redirect_uri="http://127.0.0.1:8765/callback",
    )
    assert config.scope == "read write"


def test_parse_oauth2_config_allows_dynamic_registration_to_supply_client_secret():
    config = mcp_auth.parse_oauth2_config(
        "docs",
        {
            "type": "oauth2",
            "token_endpoint_auth_method": "client_secret_post",
        },
    )

    assert config.client_id is None
    assert config.client_secret_env is None
    assert config.enable_dynamic_registration is True
    assert config.token_endpoint_auth_method == "client_secret_post"


@pytest.mark.parametrize("value", [None, {}, {"type": "bearer"}])
def test_parse_oauth2_config_rejects_unsupported_authentication(value):
    with pytest.raises(mcp_auth.McpAuthConfigError, match="unsupported"):
        mcp_auth.parse_oauth2_config("docs", value)


def test_normalize_custom_headers_requires_mapping():
    with pytest.raises(mcp_auth.McpAuthConfigError, match="must be a mapping"):
        mcp_auth.normalize_custom_headers("docs", ["X-Tenant", "fabric"])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("X-Tenant", False),
        ("X-Tenant\r", True),
        ("X-Tenant\nvalue", True),
        ("X-Tenant\r\nvalue", True),
    ],
)
def test_contains_crlf(value, expected):
    assert mcp_auth.contains_crlf(value) is expected


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("X-Foo\r", "bar"),
        ("X-Foo\n", "bar"),
        ("X-Foo", "bar\r"),
        ("X-Foo", "bar\nX-Evil: injected"),
    ],
)
def test_normalize_custom_headers_rejects_newlines(name, value):
    with pytest.raises(mcp_auth.McpAuthConfigError) as error:
        mcp_auth.normalize_custom_headers("docs", {name: value})

    assert str(error.value) == (
        f"MCP server 'docs' custom_headers contain invalid characters in {name!r}"
    )


def test_normalize_custom_headers_expands_environment_variables():
    os.environ["FABRIC_HEADER_VALUE"] = "fabric"

    assert mcp_auth.normalize_custom_headers(
        "docs", {"X-Tenant": "${FABRIC_HEADER_VALUE}"}
    ) == {"X-Tenant": "fabric"}


def test_normalize_custom_headers_rejects_newlines_after_expansion():
    os.environ["FABRIC_HEADER_VALUE"] = "fabric\r\nX-Evil: injected"

    with pytest.raises(mcp_auth.McpAuthConfigError, match="invalid characters"):
        mcp_auth.normalize_custom_headers(
            "docs", {"X-Tenant": "${FABRIC_HEADER_VALUE}"}
        )
