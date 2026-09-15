# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential-free end-to-end coverage for the packaged Pi adapter."""

from __future__ import annotations

import json
import os
from pathlib import Path

from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
import pytest
import requests


ROOT = Path(__file__).resolve().parents[2]
DESCRIPTOR = ROOT / "adapters/typescript/pi/pi.fabric-adapter.json"
PI_CLI = ROOT / "adapters/typescript/pi/dist/cli.js"
MARKER = "SESSION_MARKER_7a943c"

if os.environ.get("CI") == "true" and not PI_CLI.is_file():
    raise RuntimeError(
        "Python CI must build the packaged Pi adapter before collecting its E2E test"
    )

pytestmark = pytest.mark.skipif(
    not PI_CLI.is_file(),
    reason="build the packaged Pi adapter before running this test",
)


def config(
    tmp_path: Path,
    api_server: str,
    session: dict[str, str] | None,
) -> FabricConfig:
    harness: dict[str, object] = {"adapter_id": "nvidia.fabric.pi"}
    if session is not None:
        harness["settings"] = {"session": session}
    return FabricConfig.model_validate(
        {
            "metadata": {"name": "pi-session-persistence"},
            "discovery": {"local_paths": [str(DESCRIPTOR)]},
            "harness": harness,
            "models": {
                "default": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "base_url": f"{api_server}/v1",
                    "api_key_env": "OPENAI_API_KEY",
                }
            },
            "environment": {"workspace": str(tmp_path / "workspace")},
            "runtime": {"artifacts": str(tmp_path / "artifacts")},
        }
    )


async def test_pi_explicit_session_resumes_without_implicit_workspace_sharing(
    tmp_path: Path,
    api_server: str,
):
    (tmp_path / "workspace").mkdir()
    (tmp_path / "artifacts").mkdir()
    os.environ["OPENAI_API_KEY"] = "fixture-not-a-secret"

    create = config(
        tmp_path,
        api_server,
        {"mode": "create", "id": "persistent-conversation"},
    )
    async with await Fabric().start_runtime(create, base_dir=tmp_path) as runtime:
        await runtime.invoke(input=f"Remember {MARKER}")
        await runtime.invoke(input="What was the marker?")

    resume = config(
        tmp_path,
        api_server,
        {"mode": "resume", "id": "persistent-conversation"},
    )
    async with await Fabric().start_runtime(resume, base_dir=tmp_path) as runtime:
        await runtime.invoke(input="What was the marker?")

    async with await Fabric().start_runtime(
        config(tmp_path, api_server, None), base_dir=tmp_path
    ) as runtime:
        await runtime.invoke(input="What was the marker?")

    response = requests.get(f"{api_server}/_requests", timeout=5)
    response.raise_for_status()
    model_inputs = [json.dumps(request["input"]) for request in response.json()]
    assert [MARKER in model_input for model_input in model_inputs] == [
        True,
        True,
        True,
        False,
    ]
