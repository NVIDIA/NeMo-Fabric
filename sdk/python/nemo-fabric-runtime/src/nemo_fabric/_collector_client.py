# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP control client for the standalone ATOF collector."""

from __future__ import annotations

import os
from collections.abc import Mapping
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

from nemo_fabric.errors import FabricConfigError, FabricRuntimeError
from nemo_fabric.models import RelayAtofStreamSinkConfig


class _AtofCollectorClient:
    """Register and deregister requests to an ATOF collector."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        headers: Mapping[str, str],
    ) -> None:
        self.base_url = base_url
        client_headers = dict(headers)
        client_headers.setdefault("Accept", "application/json")
        self._client = httpx.AsyncClient(
            headers=client_headers,
            timeout=timeout_seconds,
        )

    @classmethod
    def from_sink(cls, sink: RelayAtofStreamSinkConfig) -> _AtofCollectorClient:
        if sink.transport != "ndjson":
            raise FabricConfigError(
                "Relay sink nemo-fabric-stream must use ndjson with the "
                "standalone collector"
            )
        try:
            parsed = urlsplit(sink.url)
            parsed.port  # Trigger port number validation if given
        except ValueError as error:
            raise FabricConfigError(
                "Relay sink nemo-fabric-stream has an invalid URL"
            ) from error
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise FabricConfigError(
                "Relay sink nemo-fabric-stream must use "
                "an http(s) URL without credentials, query, or fragment"
            )

        headers = dict(sink.headers)
        for name, variable in sink.header_env.items():
            value = os.environ.get(variable)
            if value is None:
                raise FabricConfigError(
                    "Relay sink nemo-fabric-stream header_env names an unset "
                    "environment variable"
                )
            headers[name] = value

        return cls(
            base_url=urlunsplit(
                (parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "")
            ),
            timeout_seconds=sink.timeout_millis / 1000,
            headers=headers,
        )

    async def register(self, request_id: str) -> None:
        await self._request(
            "POST",
            "/v1/register",
            expected_status=201,
            json={"request_id": request_id},
        )

    async def deregister(self, request_id: str, *, remove_queue: bool) -> None:
        encoded_request_id = quote(request_id, safe="")
        await self._request(
            "DELETE",
            f"/v1/deregister-request/{encoded_request_id}",
            expected_status=204,
            params={"remove_queue": "true" if remove_queue else "false"},
        )

    async def aclose(self) -> None:
        if not self._client.is_closed:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        expected_status: int,
        json: Mapping[str, object] | None = None,
        params: Mapping[str, str] | None = None,
    ) -> None:
        try:
            response = await self._client.request(
                method,
                f"{self.base_url}{path}",
                json=json,
                params=params,
            )
        except httpx.RequestError as error:
            raise FabricRuntimeError(
                f"ATOF collector request failed: {error}",
                stage="invoke",
                code="collector_request_failed",
            ) from error
        if response.status_code != expected_status:
            raise FabricRuntimeError(
                f"ATOF collector returned HTTP {response.status_code} for "
                f"{method} {path}; expected HTTP {expected_status}",
                stage="invoke",
                code="collector_request_failed",
            )
