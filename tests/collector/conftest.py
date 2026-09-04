# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator

import httpx
import pytest
from starlette.applications import Starlette

from nemo_fabric_collector.app import create_app

PUBLISH_TOKEN = "p" * 32
CONTROL_TOKEN = "c" * 32


@pytest.fixture(name="collector_app")
async def collector_app_fixture() -> AsyncIterator[Starlette]:
    application = create_app(
        publish_token=PUBLISH_TOKEN,
        control_token=CONTROL_TOKEN,
    )
    yield application
    await application.state.collector.close()


@pytest.fixture(name="collector_client")
async def collector_client_fixture(
    collector_app: Starlette,
) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=collector_app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://collector.test",
    ) as client:
        yield client
