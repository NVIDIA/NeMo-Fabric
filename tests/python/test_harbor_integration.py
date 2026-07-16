# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the Harbor consumer integration."""

from __future__ import annotations

import json
import shlex
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.usefixtures("requires_harbor")

CONFIG_FACTORY = "harbor_demo_config:build_hermes"
CONFIG_BASE_DIR = "/opt/fabric-demo"

try:
    from harbor.models.agent.context import AgentContext
    from harbor.models.task.config import MCPServerConfig
    from nemo_fabric.integrations.harbor import FabricAgent
except ImportError:
    pass


@dataclass
class ExecResult:
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""


class FakeHarborEnvironment:
    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.commands: list[str] = []
        self.environments: list[dict[str, str] | None] = []
        self.uploads: list[tuple[Path, str]] = []
        self.directory_uploads: list[tuple[Path, str]] = []
        self.operations: list[tuple[str, str]] = []

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        timeout_sec: int | None = None,
        env: dict[str, str] | None = None,
        **_: Any,
    ) -> ExecResult:
        self.commands.append(command)
        self.operations.append(("exec", command))
        self.environments.append(env)
        if "nemo_fabric.integrations.harbor.runner" in command:
            arguments = shlex.split(command)
            result_path = arguments[arguments.index("--result") + 1]
            self.files[result_path] = json.dumps(
                {
                    "agent_name": "harbor-demo",
                    "profiles": [],
                    "harness": "hermes",
                    "adapter_kind": "python",
                    "adapter_id": "nvidia.fabric.hermes",
                    "status": "succeeded",
                    "runtime_id": "runtime-1",
                    "invocation_id": "invocation-1",
                    "request_id": "request-1",
                    "output": {"response": "done"},
                    "error": None,
                    "artifacts": {
                        "root": "/workspace/agent/artifacts",
                        "artifacts": [
                            {
                                "name": "stdout",
                                "kind": "log",
                                "path": "/workspace/agent/artifacts/stdout.txt",
                                "media_type": "text/plain",
                            },
                            {
                                "name": "workspace_patch",
                                "kind": "patch",
                                "path": "/workspace/agent/artifacts/workspace.patch",
                                "media_type": "text/x-diff",
                            },
                        ],
                    },
                    "telemetry": [],
                    "events": [],
                    "metadata": {},
                }
            )
            return ExecResult()
        return ExecResult()

    async def upload_file(self, source_path: Path, target_path: str) -> None:
        self.uploads.append((source_path, target_path))
        self.files[target_path] = source_path.read_text(encoding="utf-8")

    async def download_file(self, remote_path: str, host_path: Path) -> None:
        host_path.write_text(self.files[remote_path], encoding="utf-8")

    async def upload_dir(self, source_dir: Path, target_dir: str) -> None:
        self.directory_uploads.append((Path(source_dir), target_dir))
        self.operations.append(("upload_dir", target_dir))


async def test_harbor_integration(tmp_path: Path):
    from nemo_fabric import RunRequest

    agent = FabricAgent(
        logs_dir=tmp_path,
        fabric_config_factory=CONFIG_FACTORY,
        fabric_config_base_dir=CONFIG_BASE_DIR,
        model_name="nvidia/test-model",
        skills_dir="/opt/fabric-demo/skills",
        mcp_servers=[
            MCPServerConfig(
                name="github",
                transport="streamable-http",
                url="https://mcp.example.test",
            )
        ],
        extra_env={"NVIDIA_API_KEY": "test-key"},
    )
    environment = FakeHarborEnvironment()
    context = AgentContext()

    assert isinstance(agent._build_request("fix the bug"), RunRequest)

    await agent.setup(environment)  # type: ignore[arg-type]
    await agent.run("fix the bug", environment, context)  # type: ignore[arg-type]

    spec_paths = [path for path in environment.files if path.startswith("/tmp/fabric-run-")]
    assert len(spec_paths) == 1
    assert len(environment.uploads) == 1
    spec = json.loads(environment.files[spec_paths[0]])
    request = spec["request"]
    assert request["input"] == "fix the bug"
    assert request["context"] == {"source": "harbor"}
    assert request["request_id"].startswith("request-")
    assert spec["config_factory"] == CONFIG_FACTORY
    assert spec["config_base_dir"] == CONFIG_BASE_DIR
    assert spec["model_name"] == "nvidia/test-model"
    assert spec["skills_dir"] == "/opt/fabric-demo/skills"
    assert spec["mcp_servers"] == [
        {
            "name": "github",
            "transport": "streamable-http",
            "url": "https://mcp.example.test",
            "command": None,
            "args": [],
        }
    ]

    fabric_commands = [
        command for command in environment.commands if "nemo_fabric.integrations.harbor.runner" in command
    ]
    assert len(fabric_commands) == 1
    assert not any(command.startswith("cat > ") for command in environment.commands)
    assert "python3 -m nemo_fabric.integrations.harbor.runner" in fabric_commands[0]
    assert environment.environments[environment.commands.index(fabric_commands[0])] == {
        "NVIDIA_API_KEY": "test-key",
        "ADAPTER_PYTHON": "python3",
    }
    assert context.metadata
    assert context.metadata["fabric"]["status"] == "succeeded"
    assert "profiles" not in context.metadata["fabric"]
    assert context.metadata["fabric"]["adapter_id"] == "nvidia.fabric.hermes"
    artifacts = context.metadata["fabric"]["artifacts"]["artifacts"]
    assert {artifact["name"] for artifact in artifacts} == {"stdout", "workspace_patch"}


async def test_harbor_exchange_paths_are_unique_per_run(tmp_path: Path):
    agent = FabricAgent(
        logs_dir=tmp_path,
        fabric_config_factory=CONFIG_FACTORY,
        fabric_config_base_dir=CONFIG_BASE_DIR,
    )
    environment = FakeHarborEnvironment()

    await agent.setup(environment)  # type: ignore[arg-type]
    await agent.run("first", environment, AgentContext())  # type: ignore[arg-type]
    await agent.run("second", environment, AgentContext())  # type: ignore[arg-type]

    spec_paths = [path for path in environment.files if path.startswith("/tmp/fabric-run-")]
    assert len(spec_paths) == 2
    assert len(set(spec_paths)) == 2
    result_paths = [path for path in environment.files if path.startswith("/tmp/fabric-result-")]
    assert len(result_paths) == 2
    assert len(set(result_paths)) == 2
    assert len(list(tmp_path.glob("fabric-result-*.json"))) == 2


def test_harbor_rejects_invalid_downloaded_result(tmp_path: Path):
    from nemo_fabric import FabricConfigError
    from nemo_fabric.integrations.harbor.fabric_agent import populate_context_from_result

    result_path = tmp_path / "fabric-result.json"
    result_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FabricConfigError):
        populate_context_from_result(AgentContext(), result_path)


async def test_harbor_uploads_a_portable_config_bundle(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "harbor_config.py").write_text(
        "def build_config(): ...\n",
        encoding="utf-8",
    )
    agent = FabricAgent(
        logs_dir=tmp_path / "logs",
        fabric_config_factory="harbor_config:build_config",
        fabric_config_bundle=bundle,
    )
    environment = FakeHarborEnvironment()

    await agent.setup(environment)  # type: ignore[arg-type]

    assert environment.directory_uploads == [(bundle, "/tmp/nemo-fabric-config")]
    spec = agent._build_spec("fix it")
    assert spec.config_factory == "harbor_config:build_config"
    assert spec.config_base_dir == Path("/tmp/nemo-fabric-config")


async def test_harbor_uploads_bundle_before_package_install(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "harbor_config.py").write_text(
        "def build_config(): ...\n",
        encoding="utf-8",
    )
    agent = FabricAgent(
        logs_dir=tmp_path / "logs",
        fabric_config_factory="harbor_config:build_config",
        fabric_config_bundle=bundle,
        fabric_package="/tmp/nemo-fabric-config/wheelhouse/nemo_fabric.whl",
    )
    environment = FakeHarborEnvironment()

    await agent.setup(environment)  # type: ignore[arg-type]

    upload_index = environment.operations.index(("upload_dir", "/tmp/nemo-fabric-config"))
    install_index = next(
        index
        for index, operation in enumerate(environment.operations)
        if operation[0] == "exec" and "pip install" in operation[1]
    )
    assert upload_index < install_index


def test_harbor_config_requires_an_explicit_base_dir(tmp_path: Path):
    with pytest.raises(ValueError, match="fabric_config_base_dir is required"):
        FabricAgent(
            logs_dir=tmp_path / "logs",
            fabric_config_factory="harbor_config:build_config",
        )


def test_harbor_config_rejects_an_invalid_factory_reference(tmp_path: Path):
    with pytest.raises(ValueError, match="module:callable"):
        FabricAgent(
            logs_dir=tmp_path / "logs",
            fabric_config_factory="not-a-factory-reference",
            fabric_config_base_dir="/opt/fabric",
        )


def test_harbor_config_bundle_rejects_an_explicit_base_dir(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    with pytest.raises(ValueError, match="derived from fabric_config_target"):
        FabricAgent(
            logs_dir=tmp_path / "logs",
            fabric_config_factory="harbor_config:build_config",
            fabric_config_base_dir="/opt/fabric",
            fabric_config_bundle=bundle,
        )


def test_harbor_config_rejects_a_relative_base_dir(tmp_path: Path):
    with pytest.raises(ValueError, match="must be an absolute"):
        FabricAgent(
            logs_dir=tmp_path / "logs",
            fabric_config_factory="harbor_config:build_config",
            fabric_config_base_dir="relative/config",
        )


def test_harbor_propagates_runtime_identity(tmp_path: Path):
    agent = FabricAgent(
        logs_dir=tmp_path,
        fabric_config_factory=CONFIG_FACTORY,
        fabric_config_base_dir=CONFIG_BASE_DIR,
    )
    agent.session_id = "trial__agent"
    agent.context_id = uuid.UUID("594025f3-7d65-4655-8576-4bee95002eae")

    request = agent._build_request("fix it")

    assert request.context == {
        "source": "harbor",
        "harbor_session_id": "trial__agent",
        "harbor_context_id": "594025f3-7d65-4655-8576-4bee95002eae",
    }
    assert agent.SUPPORTS_ATIF is True


async def test_harbor_structured_package_install_is_shell_safe(tmp_path: Path):
    agent = FabricAgent(
        logs_dir=tmp_path,
        fabric_config_factory=CONFIG_FACTORY,
        fabric_config_base_dir=CONFIG_BASE_DIR,
        fabric_package="nemo-fabric[codex,harbor,runtime]==0.1.0",
        extra_env={
            "NVIDIA_API_KEY": "test-key",
            "PIP_INDEX_URL": "https://packages.example/simple",
        },
    )
    environment = FakeHarborEnvironment()

    await agent.setup(environment)  # type: ignore[arg-type]

    assert environment.commands[1] == (
        "python3 -m venv /tmp/nemo-fabric-venv && "
        "/tmp/nemo-fabric-venv/bin/python -m pip install "
        "--disable-pip-version-check 'nemo-fabric[codex,harbor,runtime]==0.1.0'"
    )
    assert environment.environments[1] == {
        "PIP_INDEX_URL": "https://packages.example/simple"
    }

    await agent.run("fix it", environment, AgentContext())  # type: ignore[arg-type]
    runner = next(command for command in environment.commands if "nemo_fabric.integrations.harbor.runner" in command)
    assert runner.startswith("PATH=/tmp/nemo-fabric-venv/bin:$PATH /tmp/nemo-fabric-venv/bin/python -m ")
    runner_index = environment.commands.index(runner)
    assert environment.environments[runner_index] == {
        "NVIDIA_API_KEY": "test-key",
        "PIP_INDEX_URL": "https://packages.example/simple",
        "ADAPTER_PYTHON": "/tmp/nemo-fabric-venv/bin/python",
    }


async def test_harbor_custom_install_uses_explicit_runner_environment(tmp_path: Path):
    with pytest.warns(DeprecationWarning, match="fabric_install_command"):
        agent = FabricAgent(
            logs_dir=tmp_path,
            fabric_config_factory=CONFIG_FACTORY,
            fabric_config_base_dir=CONFIG_BASE_DIR,
            fabric_python="/tmp/custom-fabric/bin/python",
            fabric_install_command="install-fabric-for-test",
            extra_env={"NVIDIA_API_KEY": "test-key", "ADAPTER_PYTHON": "/wrong/python"},
        )
    environment = FakeHarborEnvironment()

    await agent.setup(environment)  # type: ignore[arg-type]
    install_index = environment.commands.index("install-fabric-for-test")
    assert environment.environments[install_index] == {}
    await agent.run("fix it", environment, AgentContext())  # type: ignore[arg-type]

    runner = next(command for command in environment.commands if "nemo_fabric.integrations.harbor.runner" in command)
    assert runner.startswith(
        "PATH=/tmp/custom-fabric/bin:$PATH /tmp/custom-fabric/bin/python -m "
    )
    runner_index = environment.commands.index(runner)
    assert environment.environments[runner_index] == {
        "NVIDIA_API_KEY": "test-key",
        "ADAPTER_PYTHON": "/tmp/custom-fabric/bin/python",
    }


def test_harbor_populates_usage_from_canonical_atif(tmp_path: Path):
    from nemo_fabric.integrations.harbor.fabric_agent import populate_context_from_trajectory

    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "session_id": "runtime-1",
                "agent": {"name": "fabric", "version": "0.1.0"},
                "steps": [{"step_id": 1, "source": "agent", "message": "done"}],
                "final_metrics": {
                    "total_prompt_tokens": 12,
                    "total_cached_tokens": 3,
                    "total_completion_tokens": 4,
                    "total_cost_usd": 0.25,
                }
            }
        ),
        encoding="utf-8",
    )
    context = AgentContext()

    populate_context_from_trajectory(context, trajectory)

    assert context.n_input_tokens == 12
    assert context.n_cache_tokens == 3
    assert context.n_output_tokens == 4
    assert context.cost_usd == 0.25
    assert context.metadata["fabric"]["harbor_atif_validation"] == {
        "status": "succeeded",
        "error": None,
    }


def test_harbor_records_trajectory_read_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from nemo_fabric.integrations.harbor.fabric_agent import populate_context_from_trajectory

    trajectory = tmp_path / "trajectory.json"
    trajectory.touch()
    context = AgentContext()

    def fail_read(*args, **kwargs):
        raise OSError("trajectory unavailable")

    monkeypatch.setattr(Path, "read_text", fail_read)

    populate_context_from_trajectory(context, trajectory)

    assert context.metadata["fabric"]["harbor_atif_validation"] == {
        "status": "failed",
        "error": "trajectory unavailable",
    }


def test_harbor_attaches_telemetry_summary_to_metadata(tmp_path: Path):
    from nemo_fabric.integrations.harbor.fabric_agent import populate_context_from_telemetry_summary

    summary = tmp_path / "telemetry-validation.json"
    summary.write_text(
        json.dumps({"status": "succeeded", "atof": {"records": 7}}),
        encoding="utf-8",
    )
    context = AgentContext()
    context.metadata = {"fabric": {"status": "succeeded"}}

    populate_context_from_telemetry_summary(context, summary)

    assert context.metadata["fabric"]["telemetry_validation"] == {
        "status": "succeeded",
        "atof": {"records": 7},
    }


def test_harbor_records_malformed_telemetry_summary(tmp_path: Path):
    from nemo_fabric.integrations.harbor.fabric_agent import populate_context_from_telemetry_summary

    summary = tmp_path / "telemetry-validation.json"
    summary.write_text("not json", encoding="utf-8")
    context = AgentContext()

    populate_context_from_telemetry_summary(context, summary)

    assert context.metadata["fabric"]["telemetry_validation"] == {
        "status": "failed",
        "error": "telemetry summary could not be loaded",
    }


def test_harbor_records_non_utf8_telemetry_summary(tmp_path: Path):
    from nemo_fabric.integrations.harbor.fabric_agent import populate_context_from_telemetry_summary

    summary = tmp_path / "telemetry-validation.json"
    summary.write_bytes(b"\xff")
    context = AgentContext()

    populate_context_from_telemetry_summary(context, summary)

    assert context.metadata["fabric"]["telemetry_validation"] == {
        "status": "failed",
        "error": "telemetry summary could not be loaded",
    }
