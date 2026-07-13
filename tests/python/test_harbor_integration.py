# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the Harbor consumer integration."""

from __future__ import annotations

import json
import shlex
import sys
import types
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))


def install_harbor_stubs() -> None:
    """Install minimal Harbor stubs for this smoke when Harbor is not present."""

    class BaseAgent:
        def __init__(self, logs_dir: Path, *args: Any, **kwargs: Any) -> None:
            self.logs_dir = logs_dir
            self.model_name = kwargs.get("model_name")
            self.skills_dir = kwargs.get("skills_dir")
            self.mcp_servers = kwargs.get("mcp_servers", [])
            self.extra_env = kwargs.get("extra_env")

    class BaseEnvironment:
        pass

    class AgentContext:
        def __init__(self) -> None:
            self.metadata: dict[str, Any] | None = None

    class MCPServerConfig:
        def __init__(
            self,
            *,
            name: str,
            transport: str,
            url: str | None = None,
            command: str | None = None,
            args: list[str] | None = None,
        ) -> None:
            self.name = name
            self.transport = transport
            self.url = url
            self.command = command
            self.args = args or []

        def model_dump(self, *, mode: str) -> dict[str, Any]:
            assert mode == "python"
            return vars(self)

    modules = {
        "harbor": types.ModuleType("harbor"),
        "harbor.agents": types.ModuleType("harbor.agents"),
        "harbor.agents.base": types.ModuleType("harbor.agents.base"),
        "harbor.environments": types.ModuleType("harbor.environments"),
        "harbor.environments.base": types.ModuleType("harbor.environments.base"),
        "harbor.models": types.ModuleType("harbor.models"),
        "harbor.models.agent": types.ModuleType("harbor.models.agent"),
        "harbor.models.agent.context": types.ModuleType("harbor.models.agent.context"),
        "harbor.models.task": types.ModuleType("harbor.models.task"),
        "harbor.models.task.config": types.ModuleType("harbor.models.task.config"),
    }
    modules["harbor.agents.base"].BaseAgent = BaseAgent
    modules["harbor.environments.base"].BaseEnvironment = BaseEnvironment
    modules["harbor.models.agent.context"].AgentContext = AgentContext
    modules["harbor.models.task.config"].MCPServerConfig = MCPServerConfig
    sys.modules.update(modules)


try:
    from harbor.models.agent.context import AgentContext
    from harbor.models.task.config import MCPServerConfig
    from nemo_fabric.integrations.harbor import FabricAgent
except ImportError:
    install_harbor_stubs()
    from harbor.models.agent.context import AgentContext
    from harbor.models.task.config import MCPServerConfig
    from nemo_fabric.integrations.harbor import FabricAgent


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

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        timeout_sec: int | None = None,
        env: dict[str, str] | None = None,
        **_: Any,
    ) -> ExecResult:
        self.commands.append(command)
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
                    "adapter_id": "nvidia.fabric.hermes.sdk",
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


async def test_harbor_integration(tmp_path: Path):
    from nemo_fabric import RunRequest

    agent = FabricAgent(
        logs_dir=tmp_path,
        fabric_config_path="/opt/fabric-demo/agent.yaml",
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
    assert spec["config_path"] == "/opt/fabric-demo/agent.yaml"
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
    assert environment.environments[environment.commands.index(fabric_commands[0])] == {"NVIDIA_API_KEY": "test-key"}
    assert context.metadata
    assert context.metadata["fabric"]["status"] == "succeeded"
    assert "profiles" not in context.metadata["fabric"]
    assert context.metadata["fabric"]["adapter_id"] == "nvidia.fabric.hermes.sdk"
    artifacts = context.metadata["fabric"]["artifacts"]["artifacts"]
    assert {artifact["name"] for artifact in artifacts} == {"stdout", "workspace_patch"}


async def test_harbor_exchange_paths_are_unique_per_run(tmp_path: Path):
    agent = FabricAgent(
        logs_dir=tmp_path,
        fabric_config_path="/opt/fabric-demo/agent.yaml",
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
    (bundle / "agent.yaml").write_text("harness: {}", encoding="utf-8")
    agent = FabricAgent(
        logs_dir=tmp_path / "logs",
        fabric_config_path="agent.yaml",
        fabric_config_bundle=bundle,
    )
    environment = FakeHarborEnvironment()

    await agent.setup(environment)  # type: ignore[arg-type]

    assert environment.directory_uploads == [(bundle, "/tmp/nemo-fabric-config")]
    assert agent._build_spec("fix it").config_path == Path("/tmp/nemo-fabric-config/agent.yaml")


def test_harbor_config_bundle_rejects_unsafe_entrypoints(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    with pytest.raises(ValueError, match="must be relative"):
        FabricAgent(
            logs_dir=tmp_path / "logs",
            fabric_config_path="../agent.yaml",
            fabric_config_bundle=bundle,
        )


def test_harbor_config_bundle_rejects_escaping_symlink(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text("harness: {}", encoding="utf-8")
    (bundle / "agent.yaml").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink must stay within"):
        FabricAgent(
            logs_dir=tmp_path / "logs",
            fabric_config_path="agent.yaml",
            fabric_config_bundle=bundle,
        )


def test_harbor_propagates_runtime_identity(tmp_path: Path):
    agent = FabricAgent(
        logs_dir=tmp_path,
        fabric_config_path="/opt/fabric/agent.yaml",
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
        fabric_config_path="/opt/fabric/agent.yaml",
        fabric_package="nemo-fabric[codex,harbor,runtime]==0.1.0",
    )
    environment = FakeHarborEnvironment()

    await agent.setup(environment)  # type: ignore[arg-type]

    assert environment.commands[1] == (
        "python3 -m venv /tmp/nemo-fabric-venv && "
        "/tmp/nemo-fabric-venv/bin/python -m pip install "
        "--disable-pip-version-check 'nemo-fabric[codex,harbor,runtime]==0.1.0'"
    )

    await agent.run("fix it", environment, AgentContext())  # type: ignore[arg-type]
    runner = next(command for command in environment.commands if "nemo_fabric.integrations.harbor.runner" in command)
    assert runner.startswith("PATH=/tmp/nemo-fabric-venv/bin:$PATH /tmp/nemo-fabric-venv/bin/python -m ")


def test_harbor_populates_usage_from_canonical_atif(tmp_path: Path):
    from nemo_fabric.integrations.harbor.fabric_agent import populate_context_from_trajectory

    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text(
        json.dumps(
            {
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


def test_harbor_attaches_telemetry_summary_to_metadata(tmp_path: Path):
    from nemo_fabric.integrations.harbor.fabric_agent import populate_context_from_telemetry_summary

    summary = tmp_path / "telemetry-validation.json"
    summary.write_text(
        json.dumps({"status": "succeeded", "atof": {"records": 7}}),
        encoding="utf-8",
    )
    context = AgentContext(metadata={"fabric": {"status": "succeeded"}})

    populate_context_from_telemetry_summary(context, summary)

    assert context.metadata["fabric"]["telemetry_validation"] == {
        "status": "succeeded",
        "atof": {"records": 7},
    }
