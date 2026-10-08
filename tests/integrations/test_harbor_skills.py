# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-side Harbor skill translation, independent of adapter selection."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from pathlib import PurePosixPath
from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_fabric import Fabric, FabricConfigError, RunResult
from nemo_fabric.integrations.harbor import runner
from nemo_fabric.integrations.harbor.fabric_agent import FabricAgent
from nemo_fabric.integrations.harbor.models import FabricRunPayload

pytestmark = pytest.mark.usefixtures("requires_harbor")


@pytest.fixture(name="skill_collection")
def skill_collection_fixture(
    tmp_path: Path, default_skill: Path, alternate_skill: Path
):
    root = tmp_path / "uploaded-skills"
    # Deliberately create entries in reverse order to exercise stable ordering.
    shutil.copytree(default_skill, root / "default")
    shutil.copytree(alternate_skill, root / "alternate")
    return root


@pytest.fixture(name="mock_fabric")
def mock_fabric_fixture(monkeypatch: pytest.MonkeyPatch):
    mock_fabric = MagicMock(spec=Fabric)
    mock_fabric.run = AsyncMock(return_value=MagicMock(spec=RunResult))
    monkeypatch.setattr(runner, "Fabric", MagicMock(return_value=mock_fabric))
    monkeypatch.setattr(runner, "publish_telemetry_evidence", MagicMock())
    return mock_fabric


@pytest.fixture(name="skill_payload")
def skill_payload_fixture(
    tmp_path: Path, skill_collection: Path, skill_descriptor: Path
):
    payload = FabricAgent(
        logs_dir=tmp_path / "logs",
        fabric_adapter_id="acme.skills",
        fabric_adapter_descriptor=skill_descriptor,
        skills_dir=str(skill_collection),
    )._build_spec("Use the uploaded skills.")
    # Harbor's host-side task paths are POSIX, even on a Windows host. Simulate
    # task-side filesystem access using this host's native temporary directory.
    payload.config_base_dir = PurePosixPath(tmp_path.as_posix())
    return payload


@pytest.fixture(name="skill_descriptor")
def skill_descriptor_fixture(tmp_path: Path):
    path = tmp_path / "skills.fabric-adapter.json"
    path.write_text(
        json.dumps(
            {
                "contract_version": "fabric.adapter/v1alpha2",
                "adapter_id": "acme.skills",
                "adapter_kind": "python",
                "runner": {"module": "not_installed.skills"},
                "config": {"accepts": ["skills"]},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "adapter_id", ["nvidia.fabric.openhands", "nvidia.fabric.claude"]
)
async def test_harbor_collection_is_accepted_by_individual_path_adapters(
    tmp_path: Path, skill_collection: Path, mock_fabric, adapter_id: str
):
    from nemo_fabric_adapter_contract.models import AgentConfig, RuntimeContext
    from nemo_fabric_adapters.claude import adapter as claude
    from nemo_fabric_adapters.openhands import adapter as openhands

    payload = FabricAgent(
        logs_dir=tmp_path / "logs",
        fabric_adapter_id=adapter_id,
        skills_dir=str(skill_collection),
    )._build_spec("Use both skills.")
    payload.config_base_dir = PurePosixPath(tmp_path.as_posix())

    async def validate(config, *, base_dir, request, expected_descriptor_sha256):
        assert expected_descriptor_sha256 == payload.adapter_descriptor_sha256
        adapter_config = AgentConfig.from_mapping(
            {"skills": config.skills.to_mapping()}
        )
        expected = [skill_collection / "alternate", skill_collection / "default"]
        if adapter_id == "nvidia.fabric.openhands":
            assert openhands._skill_paths(adapter_config, str(base_dir)) == expected
        else:
            plugins = claude._stage_skill_plugin(
                adapter_config,
                RuntimeContext.from_mapping(
                    {
                        "runtime_id": "harbor-skills",
                        "invocation_id": "skill-invocation",
                        "request_id": "skill-request",
                        "environment": {
                            "environment_id": "task",
                            "provider": "local",
                            "control_location": "in_env_control",
                            "workspace": str(tmp_path),
                            "ownership": "caller_owned",
                        },
                        "artifacts": {"root": str(tmp_path)},
                    }
                ),
                str(base_dir),
            )
            staged = Path(plugins[0]["path"]) / "skills"
            assert sorted(path.name for path in staged.iterdir()) == [
                "alternate",
                "default",
            ]
        return MagicMock(spec=RunResult)

    mock_fabric.run.side_effect = validate
    await runner.run(payload)
    mock_fabric.run.assert_awaited_once()


def test_host_transports_task_path_without_reading_host_filesystem(
    tmp_path: Path, skill_descriptor: Path
):
    payload = FabricAgent(
        logs_dir=tmp_path,
        fabric_adapter_id="acme.skills",
        fabric_adapter_descriptor=skill_descriptor,
        skills_dir="/harbor/task-only-skills",
    )._build_spec("Use skills.")
    assert str(payload.skills_dir) == "/harbor/task-only-skills"
    assert payload.config.skills is None


@pytest.mark.parametrize("names", [[], ["default"], ["alternate", "default"]])
async def test_empty_single_multiple_collections(
    skill_payload, skill_collection: Path, mock_fabric, names: list[str]
):
    for child in skill_collection.iterdir():
        if child.name not in names:
            shutil.rmtree(child)
    await runner.run(skill_payload)
    config = mock_fabric.run.call_args.args[0]
    assert (config.skills.paths if config.skills else []) == [
        str(skill_collection / name) for name in names
    ]
    assert skill_payload.config.skills is None


async def test_explicit_skills_are_preserved_without_duplicates(
    skill_payload, skill_collection: Path, mock_fabric, default_skill: Path
):
    skill_payload.config.add_skill_path(default_skill)
    skill_payload.config.add_skill_path(skill_collection / "default")
    original = skill_payload.config.to_mapping()
    await runner.run(skill_payload)
    await runner.run(skill_payload)
    for call in mock_fabric.run.call_args_list:
        assert call.args[0].skills.paths == [
            str(default_skill),
            str(skill_collection / "default"),
            str(skill_collection / "alternate"),
        ]
    assert skill_payload.config.to_mapping() == original


async def test_relative_collection_is_resolved_against_task_base_dir(
    skill_payload, skill_collection: Path, mock_fabric
):
    skill_payload.skills_dir = Path(skill_collection.name)
    transported = FabricRunPayload.model_validate_json(skill_payload.model_dump_json())
    await runner.run(transported)
    assert mock_fabric.run.call_args.args[0].skills.paths == [
        str(skill_collection / "alternate"),
        str(skill_collection / "default"),
    ]


async def test_relative_explicit_skill_overlapping_collection_is_not_added_twice(
    skill_payload, skill_collection: Path, mock_fabric
):
    relative_skill = Path(skill_collection.name) / "default"
    skill_payload.config.add_skill_path(relative_skill)
    skill_payload.skills_dir = Path(skill_collection.name)
    original = skill_payload.config.to_mapping()
    await runner.run(skill_payload)
    config = mock_fabric.run.call_args.args[0]
    assert config.skills.paths == [
        str(relative_skill),
        str(skill_collection / "alternate"),
    ]
    assert skill_payload.config.to_mapping() == original


def test_cli_preserves_malformed_skill_path_diagnostic(
    tmp_path: Path, skill_payload, skill_collection: Path
):
    offending_path = skill_collection / "default"
    (offending_path / "SKILL.md").unlink()
    spec = tmp_path / "spec.json"
    result = tmp_path / "result.json"
    spec.write_text(skill_payload.model_dump_json(), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nemo_fabric.integrations.harbor.runner",
            "--spec",
            str(spec),
            "--result",
            str(result),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 1
    diagnostic = json.loads(result.read_text())["runner_error"]
    assert str(offending_path) in diagnostic["message"]
    assert diagnostic["code"] == "harbor_skills_invalid"


@pytest.mark.parametrize(
    "malformation",
    [
        "missing",
        "file",
        "missing-skill",
        "skill-is-dir",
        "skill-is-symlink",
        "loose-file",
    ],
)
async def test_malformed_collection_fails_before_harness_execution(
    skill_payload, skill_collection: Path, mock_fabric, malformation: str
):
    offending_path = skill_collection
    if malformation == "missing":
        shutil.rmtree(skill_collection)
    elif malformation == "file":
        shutil.rmtree(skill_collection)
        skill_collection.write_text("not a collection", encoding="utf-8")
    elif malformation == "missing-skill":
        (skill_collection / "default" / "SKILL.md").unlink()
        offending_path = skill_collection / "default"
    elif malformation == "skill-is-dir":
        path = skill_collection / "default" / "SKILL.md"
        path.unlink()
        path.mkdir()
        offending_path = skill_collection / "default"
    elif malformation == "skill-is-symlink":
        path = skill_collection / "default" / "SKILL.md"
        path.unlink()
        path.symlink_to(skill_collection / "alternate" / "SKILL.md")
        assert path.is_file()
        offending_path = skill_collection / "default"
    else:
        (skill_collection / "SKILL.md").write_text(
            "not a child skill", encoding="utf-8"
        )
        offending_path = skill_collection / "SKILL.md"
    with pytest.raises(FabricConfigError, match="Harbor skills collection") as error:
        await runner.run(skill_payload)
    assert str(offending_path) in str(error.value)
    mock_fabric.run.assert_not_awaited()


async def test_no_collection_leaves_explicit_fabric_skills_unchanged(
    skill_payload, mock_fabric, default_skill: Path
):
    skill_payload.skills_dir = None
    skill_payload.config.add_skill_path(default_skill)
    await runner.run(skill_payload)
    assert mock_fabric.run.call_args.args[0].skills.paths == [str(default_skill)]


@pytest.mark.skipif(
    sys.platform == "win32", reason="mock Claude CLI requires a POSIX executable"
)
@pytest.mark.parametrize("relative_overlap", [False, True])
def test_task_runner_cli_stages_collection_through_real_claude_sdk(
    tmp_path: Path, skill_collection: Path, repo_root: Path, relative_overlap: bool
):
    logs = tmp_path / "logs"
    artifacts = tmp_path / "artifacts"
    agent = FabricAgent(
        logs_dir=logs,
        fabric_adapter_id="nvidia.fabric.claude",
        model_name="anthropic/claude-test-model",
        fabric_python=sys.executable,
        fabric_workspace=str(tmp_path),
        skills_dir=str(skill_collection),
        fabric_harness_settings={"permission_mode": "dontAsk", "setting_sources": []},
        fabric_environment_env={
            "FABRIC_TEST_CLAUDE_CLI_PATH": str(
                repo_root / "tests/fixtures/claude/mock-claude-cli.py"
            ),
            "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK": "1",
            "MOCK_CLAUDE_CLI_LOG": str(tmp_path / "cli-args.jsonl"),
        },
    )
    payload = agent._build_spec("Use both skills.")
    if relative_overlap:
        payload.config_base_dir = PurePosixPath(tmp_path.as_posix())
        payload.skills_dir = Path(skill_collection.name)
        payload.config.add_skill_path(Path(skill_collection.name) / "default")
    payload.config.runtime.artifacts = artifacts
    payload.config.environment.artifacts = artifacts
    payload.logs_dir = logs
    spec_path = tmp_path / "spec.json"
    result_path = tmp_path / "result.json"
    spec_path.write_text(payload.model_dump_json(), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nemo_fabric.integrations.harbor.runner",
            "--spec",
            str(spec_path),
            "--result",
            str(result_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, **agent._runner_env},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = RunResult.from_mapping(json.loads(result_path.read_text(encoding="utf-8")))
    assert result.status == "succeeded", result.to_mapping()
    arguments = json.loads((tmp_path / "cli-args.jsonl").read_text().splitlines()[0])
    plugin = Path(arguments[arguments.index("--plugin-dir") + 1]) / "skills"
    for name in ("alternate", "default"):
        assert (plugin / name / "SKILL.md").read_bytes() == (
            skill_collection / name / "SKILL.md"
        ).read_bytes()
