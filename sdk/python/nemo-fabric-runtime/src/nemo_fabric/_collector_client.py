# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP control client for the standalone ATOF collector."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from nemo_fabric.errors import FabricConfigError, FabricRuntimeError
from nemo_fabric.models import RelayAtofStreamSinkConfig


@dataclass(frozen=True)
class _AtofCollectorClient:
    """Register and deregister requests to an ATOF collector."""

    base_url: str
    timeout_seconds: float
    headers: Mapping[str, str]

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

    def register(self, request_id: str) -> None:
        payload = json.dumps({"request_id": request_id}).encode()
        self._request(
            "POST",
            "/v1/register",
            expected_status=201,
            body=payload,
            content_type="application/json",
        )

    def deregister(self, request_id: str, *, remove_queue: bool) -> None:
        encoded_request_id = quote(request_id, safe="")
        query_value = "true" if remove_queue else "false"
        self._request(
            "DELETE",
            f"/v1/deregister-request/{encoded_request_id}"
            f"?remove_queue={query_value}",
            expected_status=204,
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        expected_status: int,
        body: bytes | None = None,
        content_type: str | None = None,
    ) -> None:
        headers = dict(self.headers)
        headers.setdefault("Accept", "application/json")
        if content_type is not None:
            headers["Content-Type"] = content_type
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                status = response.status
        except HTTPError as error:
            raise FabricRuntimeError(
                f"ATOF collector returned HTTP {error.code} for {method} {path}",
                stage="invoke",
                code="collector_request_failed",
            ) from error
        except (HTTPException, OSError, URLError) as error:
            raise FabricRuntimeError(
                f"ATOF collector request failed: {error}",
                stage="invoke",
                code="collector_request_failed",
            ) from error
        if status != expected_status:
            raise FabricRuntimeError(
                f"ATOF collector returned HTTP {status} for {method} {path}; "
                f"expected HTTP {expected_status}",
                stage="invoke",
                code="collector_request_failed",
            )
