# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the Harbor consumer integration."""

from __future__ import annotations

import json
import shlex
import sys
import types
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
    from nemo_fabric.integrations.harbor import FabricAgent
    from harbor.models.agent.context import AgentContext
    from harbor.models.task.config import MCPServerConfig
except ImportError:
    install_harbor_stubs()
    from nemo_fabric.integrations.harbor import FabricAgent
    from harbor.models.agent.context import AgentContext
    from harbor.models.task.config import MCPServerConfig


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

    spec_paths = [
        path for path in environment.files if path.startswith("/tmp/fabric-run-")
    ]
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
        command
        for command in environment.commands
        if "nemo_fabric.integrations.harbor.runner" in command
    ]
    assert len(fabric_commands) == 1
    assert not any(command.startswith("cat > ") for command in environment.commands)
    assert "python3 -m nemo_fabric.integrations.harbor.runner" in fabric_commands[0]
    assert environment.environments[environment.commands.index(fabric_commands[0])] == {
        "NVIDIA_API_KEY": "test-key"
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
        fabric_config_path="/opt/fabric-demo/agent.yaml",
    )
    environment = FakeHarborEnvironment()

    await agent.setup(environment)  # type: ignore[arg-type]
    await agent.run("first", environment, AgentContext())  # type: ignore[arg-type]
    await agent.run("second", environment, AgentContext())  # type: ignore[arg-type]

    spec_paths = [path for path in environment.files if path.startswith("/tmp/fabric-run-")]
    assert len(spec_paths) == 2
    assert len(set(spec_paths)) == 2
    result_paths = [
        path for path in environment.files if path.startswith("/tmp/fabric-result-")
    ]
    assert len(result_paths) == 2
    assert len(set(result_paths)) == 2
    assert len(list(tmp_path.glob("fabric-result-*.json"))) == 2


def test_harbor_rejects_invalid_downloaded_result(tmp_path: Path):
    from nemo_fabric import FabricConfigError
    from nemo_fabric.integrations.harbor.fabric_agent import (
        populate_context_from_result,
    )

    result_path = tmp_path / "fabric-result.json"
    result_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FabricConfigError):
        populate_context_from_result(AgentContext(), result_path)
