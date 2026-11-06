# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the public retained-session contract across OpenClaw Gateway processes."""

import json
import os
import shutil
import sys
from uuid import uuid4

import pytest

from nemo_fabric import Fabric, FabricConfig, SessionRef
from nemo_fabric.errors import FabricRuntimeError

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="Mock Gateway uses a POSIX launcher"
)


@pytest.fixture(name="openclaw_resume_config")
def openclaw_resume_config_fixture(tmp_path, repo_root):
    os.environ["ADAPTER_PYTHON"] = sys.executable
    os.environ["FAKE_OPENCLAW_CAPTURE"] = str(tmp_path / "native-config.json")
    os.environ["FAKE_OPENCLAW_REQUEST"] = str(tmp_path / "native-request.json")
    os.environ["FAKE_OPENCLAW_MEMORY_REPLY"] = "1"
    return FabricConfig.from_mapping(
        {
            "metadata": {"name": "assitant"},
            "harness": {
                "adapter_id": "nvidia.fabric.openclaw",
                "resolution": "preinstalled",
                "settings": {
                    "openclaw_command": str(repo_root / "tests/_utils/mock_openclaw.py")
                },
            },
            "models": {"default": {"provider": "test", "model": "echo"}},
            "discovery": {"local_paths": [str(repo_root / "adapters/python/openclaw")]},
            "runtime": {"keep_state": True, "artifacts": str(tmp_path / "artifacts")},
            "environment": {"provider": "local", "workspace": str(tmp_path)},
        }
    )


async def test_openclaw_sdk_resumes_after_storage_move_and_agent_name_correction(
    tmp_path, openclaw_resume_config
):
    nonce = "conversation-only-" + uuid4().hex
    fabric = Fabric()
    async with await fabric.start_runtime(
        openclaw_resume_config, base_dir=tmp_path
    ) as first:
        reference = first.session_ref
        assert isinstance(reference, SessionRef)
        first_id = first.runtime_id
        assert (await first.invoke(input=nonce)).session_ref == reference

    moved = tmp_path / "mounted-again"
    shutil.move(tmp_path / "artifacts", moved)
    current = openclaw_resume_config.model_copy(deep=True)
    current.runtime.artifacts = str(moved)
    current.metadata.name = "assistant"
    async with await fabric.start_runtime(
        current, base_dir=tmp_path, session_ref=reference
    ) as resumed:
        assert resumed.runtime_id != first_id
        assert resumed.session_ref == reference
        result = await resumed.invoke(input="Recall the earlier conversation.")
        assert result.session_ref == reference
        assert nonce in result.output["response"]
    request = json.loads((tmp_path / "native-request.json").read_text())
    assert nonce not in json.dumps(request)


async def test_openclaw_sdk_retention_starts_fresh_without_a_reference(
    tmp_path, openclaw_resume_config
):
    fabric = Fabric()
    async with await fabric.start_runtime(
        openclaw_resume_config, base_dir=tmp_path
    ) as first:
        reference = first.session_ref
        await first.invoke(input="Retained secret amber-1428.")
    async with await fabric.start_runtime(
        openclaw_resume_config, base_dir=tmp_path
    ) as fresh:
        assert fresh.session_ref != reference
        result = await fresh.invoke(input="A separate conversation.")
        assert "amber-1428" not in result.output["response"]
    async with await fabric.start_runtime(
        openclaw_resume_config, base_dir=tmp_path, session_ref=reference
    ) as resumed:
        result = await resumed.invoke(input="Recall the saved secret.")
        assert "amber-1428" in result.output["response"]


async def test_openclaw_sdk_service_keeps_ownership_and_accepts_explicit_resume(
    tmp_path, openclaw_resume_config
):
    fabric = Fabric()
    service = await fabric.prepare_service(openclaw_resume_config, base_dir=tmp_path)
    try:
        async with await fabric.start_runtime(
            openclaw_resume_config, base_dir=tmp_path, service=service
        ) as first:
            reference = first.session_ref
            await first.invoke(input="Service secret cedar-3837.")
            with pytest.raises(FabricRuntimeError, match="in use"):
                await fabric.start_runtime(
                    openclaw_resume_config, base_dir=tmp_path, service=service
                )
        with pytest.raises(FabricRuntimeError, match="in use"):
            await fabric.start_runtime(openclaw_resume_config, base_dir=tmp_path)
        async with await fabric.start_runtime(
            openclaw_resume_config,
            base_dir=tmp_path,
            service=service,
            session_ref=reference,
        ) as resumed:
            assert resumed.session_ref == reference
            result = await resumed.invoke(input="Recall the saved service secret.")
            assert "cedar-3837" in result.output["response"]
    finally:
        await service.release()
    async with await fabric.start_runtime(
        openclaw_resume_config, base_dir=tmp_path, session_ref=reference
    ) as independent:
        assert independent.session_ref == reference
