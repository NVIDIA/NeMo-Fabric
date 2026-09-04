# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest

from nemo_fabric_collector.app import create_app

PUBLISH_TOKEN = "p" * 32
CONTROL_TOKEN = "c" * 32


async def test_healthz_does_not_require_authentication(
    collector_client: httpx.AsyncClient,
):
    response = await collector_client.get("/healthz")

    assert response.status_code == 200
    assert response.text == "ok"


async def test_endpoints_do_not_require_tokens_when_authentication_is_disabled():
    application = create_app()
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://collector.test",
    ) as client:
        registered = await client.post(
            "/v1/register",
            json={"request_id": "request-1"},
        )
        published = await client.post("/v1/atof", content=b"{}\n")
        deregistered = await client.delete(
            "/v1/deregister-request/request-1?remove_queue=true"
        )
    await application.state.collector.close()

    assert registered.status_code == 201
    assert published.status_code == 200
    assert deregistered.status_code == 204


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("POST", "/v1/register", {"json": {"request_id": "request-1"}}),
        ("GET", "/v1/stream/request-1", {}),
        ("DELETE", "/v1/deregister-request/request-1", {}),
    ],
)
async def test_control_endpoints_require_control_token(
    collector_client: httpx.AsyncClient,
    method: str,
    path: str,
    kwargs: dict,
):
    response = await collector_client.request(method, path, **kwargs)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


async def test_atof_endpoint_requires_publish_token(
    collector_client: httpx.AsyncClient,
):
    response = await collector_client.post("/v1/atof", content=b"{}\n")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    ("path", "token"),
    [
        ("/v1/register", PUBLISH_TOKEN),
        ("/v1/atof", CONTROL_TOKEN),
    ],
)
async def test_tokens_are_not_interchangeable(
    collector_client: httpx.AsyncClient,
    path: str,
    token: str,
):
    response = await collector_client.post(
        path,
        headers={"Authorization": f"Bearer {token}"},
        json={"request_id": "request-1"},
    )

    assert response.status_code == 401


async def test_register_rejects_invalid_and_duplicate_request_ids(
    collector_client: httpx.AsyncClient,
):
    headers = {"Authorization": f"Bearer {CONTROL_TOKEN}"}

    invalid = await collector_client.post(
        "/v1/register",
        headers=headers,
        json={"request_id": ""},
    )
    created = await collector_client.post(
        "/v1/register",
        headers=headers,
        json={"request_id": "request-1"},
    )
    duplicate = await collector_client.post(
        "/v1/register",
        headers=headers,
        json={"request_id": "request-1"},
    )

    assert invalid.status_code == 400
    assert created.status_code == 201
    assert created.json() == {"request_id": "request-1", "status": "ready"}
    assert duplicate.status_code == 409


async def test_atof_records_are_routed_and_streamed_as_ndjson(
    collector_client: httpx.AsyncClient,
):
    control_headers = {"Authorization": f"Bearer {CONTROL_TOKEN}"}
    publish_headers = {"Authorization": f"Bearer {PUBLISH_TOKEN}"}
    root = {
        "kind": "scope",
        "scope_category": "start",
        "uuid": "root-1",
        "metadata": {"nemo_fabric_request_id": "request-1"},
    }
    child = {
        "kind": "mark",
        "uuid": "mark-1",
        "parent_uuid": "root-1",
    }
    body = b"not-json\n" + b"\n".join(
        json.dumps(record).encode() for record in (root, child)
    )

    registered = await collector_client.post(
        "/v1/register",
        headers=control_headers,
        json={"request_id": "request-1"},
    )
    published = await collector_client.post(
        "/v1/atof",
        headers=publish_headers,
        content=body,
    )
    deregistered = await collector_client.delete(
        "/v1/deregister-request/request-1",
        headers=control_headers,
    )
    streamed = await collector_client.get(
        "/v1/stream/request-1",
        headers=control_headers,
    )

    assert registered.status_code == 201
    assert published.status_code == 200
    assert deregistered.status_code == 204
    assert streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("application/x-ndjson")
    assert [json.loads(line) for line in streamed.text.splitlines()] == [root, child]

    missing = await collector_client.get(
        "/v1/stream/request-1",
        headers=control_headers,
    )
    assert missing.status_code == 404


async def test_deregister_remove_queue_discards_records(
    collector_client: httpx.AsyncClient,
):
    control_headers = {"Authorization": f"Bearer {CONTROL_TOKEN}"}
    publish_headers = {"Authorization": f"Bearer {PUBLISH_TOKEN}"}
    root = {
        "kind": "scope",
        "scope_category": "start",
        "uuid": "root-1",
        "metadata": {"nemo_fabric_request_id": "request-1"},
    }
    await collector_client.post(
        "/v1/register",
        headers=control_headers,
        json={"request_id": "request-1"},
    )
    await collector_client.post(
        "/v1/atof",
        headers=publish_headers,
        content=json.dumps(root),
    )

    response = await collector_client.delete(
        "/v1/deregister-request/request-1?remove_queue=true",
        headers=control_headers,
    )

    assert response.status_code == 204
    stream = await collector_client.get(
        "/v1/stream/request-1",
        headers=control_headers,
    )
    assert stream.status_code == 404


async def test_deregister_rejects_invalid_remove_queue(
    collector_client: httpx.AsyncClient,
):
    response = await collector_client.delete(
        "/v1/deregister-request/request-1?remove_queue=sometimes",
        headers={"Authorization": f"Bearer {CONTROL_TOKEN}"},
    )

    assert response.status_code == 400


async def test_atof_rejects_oversized_record(collector_client: httpx.AsyncClient):
    response = await collector_client.post(
        "/v1/atof",
        headers={"Authorization": f"Bearer {PUBLISH_TOKEN}"},
        content=b"x" * (1024 * 1024 + 1),
    )

    assert response.status_code == 413
