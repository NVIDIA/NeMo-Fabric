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
    manifest = json.loads((package / "package.json").read_text(encoding="utf-8"))
    assert manifest["version"] in adapter.SUPPORTED_OPENCLAW_VERSIONS
    return package


def _skill(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Use the {name} test skill.\n---\n\n"
        f"Reply with `{name} loaded`.\n",
        encoding="utf-8",
    )
    return directory


def _native_config(tmp_path: Path, paths: list[Path]) -> dict:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    config = AgentConfig.from_mapping(
        {
            "models": {"default": {"provider": "test", "model": "unused"}},
            "skills": {"paths": paths},
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
    return adapter._openclaw_config(
        config, context, base_dir=tmp_path, port=18789, token_env="UNUSED_TEST_TOKEN"
    )


def _load_natively(
    tmp_path: Path, openclaw_package: Path, paths: list[Path]
) -> list[dict]:
    native = _native_config(tmp_path, paths)
    config_path = tmp_path / "openclaw.json"
    config_path.write_text(json.dumps(native), encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    probe = Path(__file__).parents[1] / "_utils" / "openclaw_skills.mjs"
    result = subprocess.run(
        ["node", str(probe), str(openclaw_package), str(config_path)],
        cwd=tmp_path / "workspace",
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
    return json.loads(result.stdout)


def _loaded(skill_dir: Path, name: str) -> dict:
    source = (skill_dir / "SKILL.md").resolve()
    return {
        "name": name,
        "source": "openclaw-extra",
        "path": str(source),
        "contentHash": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


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
    entries = _load_natively(
        tmp_path, openclaw_package, [sources[skill]] if skill else []
    )
    loaded = [entry for entry in entries if entry["name"] in sources]
    if skill is None:
        assert loaded == []
    else:
        assert loaded == [_loaded(sources[skill], skill)]


def test_skill_folders_are_loaded_by_native_openclaw(
    tmp_path: Path, openclaw_package: Path
):
    """Load each skill directly inside a folder, mixed with a separate skill."""
    shared = tmp_path / "shared"
    one = _skill(shared / "one", "fabric-one")
    two = _skill(shared / "two", "fabric-two")
    (shared / "notes.md").write_text("Not a skill.\n", encoding="utf-8")
    _skill(shared / "group" / "nested", "fabric-nested")
    separate = _skill(tmp_path / "mounted" / "three", "fabric-three")

    entries = _load_natively(tmp_path, openclaw_package, [shared, separate])

    loaded = [entry for entry in entries if entry["name"].startswith("fabric-")]
    assert sorted(loaded, key=lambda entry: entry["name"]) == [
        _loaded(one, "fabric-one"),
        _loaded(separate, "fabric-three"),
        _loaded(two, "fabric-two"),
    ]


def test_duplicate_skill_names_fail_naming_both_paths(tmp_path: Path):
    """Refuse two configured skills that share a name."""
    first = _skill(tmp_path / "shared" / "review", "fabric-review")
    second = _skill(tmp_path / "mounted" / "review", "fabric-review")

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        _native_config(tmp_path, [tmp_path / "shared", second])

    assert caught.value.code == "openclaw_skill_invalid"
    assert caught.value.metadata["field"] == "skills.paths[1]"
    assert str(first.resolve()) in caught.value.message
    assert str(second.resolve()) in caught.value.message
