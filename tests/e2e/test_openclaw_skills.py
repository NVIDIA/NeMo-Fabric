# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fabric configuration through the installed native loader, without inference.

OPENCLAW_TEST_PACKAGE selects the pinned package installed by the native CI step.
The loader hashes the bytes it reads; this does not prove model use or mounts.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest
from nemo_fabric_adapter_contract.models import AgentConfig, RuntimeContext
from nemo_fabric_adapters.openclaw import adapter


@pytest.fixture(name="openclaw_package")
def openclaw_package_fixture() -> Path:
    """Require a supported native package when the dedicated CI step enables it."""
    configured = os.environ.get("OPENCLAW_TEST_PACKAGE")
    if configured is None:
        pytest.skip("OPENCLAW_TEST_PACKAGE must name the installed native package")
    package = Path(configured).resolve()
    manifest = json.loads((package / "package.json").read_text())
    assert manifest["version"] in adapter.SUPPORTED_OPENCLAW_VERSIONS
    return package


@pytest.mark.parametrize("skill", ["default", "alternate", None])
def test_configured_skill_is_loaded_by_native_openclaw(
    tmp_path: Path,
    openclaw_package: Path,
    default_skill: Path,
    alternate_skill: Path,
    skill: str | None,
):
    """Check native selection, source path, and content hash for supplied skills."""
    sources = {"default": default_skill, "alternate": alternate_skill}
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = AgentConfig.from_mapping(
        {
            "models": {"default": {"provider": "test", "model": "unused"}},
            "skills": {"paths": [sources[skill]] if skill else []},
        }
    )
    context = RuntimeContext.from_mapping(
        {
            "runtime_id": "skill-loading",
            "invocation_id": "skill-loading",
            "request_id": "skill-loading",
            "environment": {
                "environment_id": "skill-loading",
                "provider": "local",
                "control_location": "in_env_control",
                "workspace": str(workspace),
                "env": {},
                "ownership": "caller_owned",
            },
            "artifacts": {},
        }
    )
    native = adapter._openclaw_config(
        config, context, base_dir=tmp_path, port=18789, token_env="UNUSED_TEST_TOKEN"
    )
    config_path = tmp_path / "openclaw.json"
    config_path.write_text(json.dumps(native))
    home = tmp_path / "home"
    home.mkdir()
    probe = Path(__file__).parents[1] / "_utils" / "openclaw_skills.mjs"
    result = subprocess.run(
        ["node", str(probe), str(openclaw_package), str(config_path)],
        cwd=workspace,
        env={
            "PATH": os.environ["PATH"],
            "HOME": str(home),
            "OPENCLAW_HOME": str(home),
            "OPENCLAW_STATE_DIR": str(home / ".openclaw"),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    loaded = [entry for entry in json.loads(result.stdout) if entry["name"] in sources]
    if skill is None:
        assert loaded == []
    else:
        source = (sources[skill] / "SKILL.md").resolve()
        assert loaded == [
            {
                "name": skill,
                "source": "openclaw-extra",
                "path": str(source),
                "contentHash": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ]
