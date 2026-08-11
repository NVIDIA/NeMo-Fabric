# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared MCP authentication configuration helpers for Fabric adapters."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class McpAuthConfigError(ValueError):
    """Invalid normalized MCP authentication configuration."""


@dataclass(frozen=True)
class McpOAuth2Config:
    """Harness-neutral OAuth2 fields from an MCP server configuration."""

    client_id: str | None
    client_secret_env: str | None
    scopes: tuple[str, ...]
    redirect_uri: str | None
    enable_dynamic_registration: bool = True
    client_name: str | None = None
    token_endpoint_auth_method: str | None = None
    authorization_timeout_seconds: float = 300.0

    @property
    def scope(self) -> str | None:
        value = " ".join(self.scopes)
        return value or None


TOKEN_ENDPOINT_AUTH_METHODS = {
    "none",
    "client_secret_post",
    "client_secret_basic",
}


def _string_tuple(server_name: str, field: str, value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise McpAuthConfigError(
            f"MCP server {server_name!r} authentication.{field} must be a list of strings"
        )
    return tuple(value)


def _positive_timeout(server_name: str, field: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise McpAuthConfigError(
            f"MCP server {server_name!r} authentication.{field} must be greater than zero"
        )
    return float(value)


def _token_endpoint_auth_method(server_name: str, value: Any) -> str | None:
    if value is None:
        return None
    if value not in TOKEN_ENDPOINT_AUTH_METHODS:
        supported = ", ".join(sorted(TOKEN_ENDPOINT_AUTH_METHODS))
        raise McpAuthConfigError(
            f"MCP server {server_name!r} authentication.token_endpoint_auth_method "
            f"must be one of: {supported}"
        )
    return str(value)


def parse_oauth2_config(server_name: str, value: Any) -> McpOAuth2Config:
    """Parse Fabric's normalized OAuth2 mapping without applying harness policy."""

    if not isinstance(value, Mapping) or value.get("type") != "oauth2":
        raise McpAuthConfigError(
            f"MCP server {server_name!r} has unsupported authentication type"
        )
    client_id = (
        str(raw_client_id) if (raw_client_id := value.get("client_id")) else None
    )
    secret_env = (
        str(raw_secret_env)
        if (raw_secret_env := value.get("client_secret_env"))
        else None
    )
    dynamic_registration = value.get("enable_dynamic_registration", True)
    if not isinstance(dynamic_registration, bool):
        raise McpAuthConfigError(
            f"MCP server {server_name!r} authentication.enable_dynamic_registration must be a boolean"
        )
    if secret_env and not client_id:
        raise McpAuthConfigError(
            f"MCP server {server_name!r} authentication.client_secret_env requires client_id"
        )
    if not client_id and not dynamic_registration:
        raise McpAuthConfigError(
            f"MCP server {server_name!r} authentication.client_id is required when "
            "dynamic registration is disabled"
        )
    method = _token_endpoint_auth_method(
        server_name,
        value.get("token_endpoint_auth_method"),
    )
    if (
        method in {"client_secret_basic", "client_secret_post"}
        and client_id
        and not secret_env
    ):
        raise McpAuthConfigError(
            f"MCP server {server_name!r} authentication.token_endpoint_auth_method "
            "requires client_secret_env for a pre-registered client"
        )
    if method == "none" and secret_env:
        raise McpAuthConfigError(
            f"MCP server {server_name!r} authentication.token_endpoint_auth_method "
            "'none' cannot use client_secret_env"
        )
    return McpOAuth2Config(
        client_id=client_id,
        client_secret_env=secret_env,
        scopes=_string_tuple(server_name, "scopes", value.get("scopes")),
        redirect_uri=(
            str(redirect_uri) if (redirect_uri := value.get("redirect_uri")) else None
        ),
        enable_dynamic_registration=dynamic_registration,
        client_name=(
            str(client_name) if (client_name := value.get("client_name")) else None
        ),
        token_endpoint_auth_method=method,
        authorization_timeout_seconds=_positive_timeout(
            server_name,
            "authorization_timeout_seconds",
            value.get("authorization_timeout_seconds", 300),
        ),
    )


def contains_crlf(value: str) -> bool:
    """Return whether a string contains a carriage return or line feed."""

    return "\r" in value or "\n" in value


def normalize_custom_headers(server_name: str, value: dict[str, str]) -> dict[str, str]:
    """Validate and expand an MCP custom-header mapping."""

    if not isinstance(value, Mapping):
        raise McpAuthConfigError(
            f"MCP server {server_name!r} custom_headers must be a mapping"
        )
    results: dict[str, str] = {}
    for name, item in value.items():
        if contains_crlf(name) or contains_crlf(item):
            raise McpAuthConfigError(
                f"MCP server {server_name!r} custom_headers contain invalid characters in {name!r}"
            )
        expanded_item = os.path.expandvars(item)
        if contains_crlf(expanded_item):
            raise McpAuthConfigError(
                f"MCP server {server_name!r} custom_headers contain invalid characters in {name!r}"
            )
        results[name] = expanded_item

    return results
