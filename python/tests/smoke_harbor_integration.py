# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the Harbor consumer integration."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

    class BaseEnvironment:
        pass

    class AgentContext:
        def __init__(self) -> None:
            self.metadata: dict[str, Any] | None = None

    modules = {
        "harbor": types.ModuleType("harbor"),
        "harbor.agents": types.ModuleType("harbor.agents"),
        "harbor.agents.base": types.ModuleType("harbor.agents.base"),
        "harbor.environments": types.ModuleType("harbor.environments"),
        "harbor.environments.base": types.ModuleType("harbor.environments.base"),
        "harbor.models": types.ModuleType("harbor.models"),
        "harbor.models.agent": types.ModuleType("harbor.models.agent"),
        "harbor.models.agent.context": types.ModuleType("harbor.models.agent.context"),
    }
    modules["harbor.agents.base"].BaseAgent = BaseAgent
    modules["harbor.environments.base"].BaseEnvironment = BaseEnvironment
    modules["harbor.models.agent.context"].AgentContext = AgentContext
    sys.modules.update(modules)


try:
    from nemo_fabric.integrations.harbor import FabricAgent
    from harbor.models.agent.context import AgentContext
except ImportError:
    install_harbor_stubs()
    from nemo_fabric.integrations.harbor import FabricAgent
    from harbor.models.agent.context import AgentContext


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
        if command.startswith("cat > "):
            path, contents = command.split(" <<'FABRIC_JSON'\n", maxsplit=1)
            path = path.removeprefix("cat > ").strip()
            contents = contents.removesuffix("\nFABRIC_JSON")
            self.files[path] = contents
            return ExecResult()
        if "nemo_fabric.integrations.harbor.runner" in command:
            self.files["/logs/agent/fabric-result.json"] = json.dumps(
                {
                    "status": "succeeded",
                    "runtime_id": "runtime-1",
                    "invocation_id": "invocation-1",
                    "request_id": "harbor-request-1",
                    "profiles": ["env_local", "mcp_github"],
                    "harness": "hermes",
                    "adapter_id": "nvidia.fabric.hermes.sdk",
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
                    "telemetry": None,
                }
            )
            return ExecResult()
        return ExecResult()

    async def download_file(self, remote_path: str, host_path: Path) -> None:
        host_path.write_text(self.files[remote_path], encoding="utf-8")


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="fabric-harbor-integration-") as tmpdir:
        agent = FabricAgent(
            logs_dir=Path(tmpdir),
            fabric_config_path="/opt/fabric-demo/agent.yaml",
            fabric_profile_paths=[
                "/opt/fabric-demo/profiles/hermes.yaml",
                "/opt/fabric-demo/profiles/telemetry.yaml",
            ],
            model_name="nvidia/test-model",
            extra_env={"NVIDIA_API_KEY": "test-key"},
        )
        environment = FakeHarborEnvironment()
        context = AgentContext()

        await agent.setup(environment)  # type: ignore[arg-type]
        await agent.run("fix the bug", environment, context)  # type: ignore[arg-type]

    spec = json.loads(environment.files["/tmp/fabric-run.json"])
    request = spec["request"]
    assert request["input"] == "fix the bug"
    assert request["context"]["source"] == "harbor"
    assert request["context"]["model_name"] == "nvidia/test-model"
    assert spec["config_path"] == "/opt/fabric-demo/agent.yaml"
    assert spec["profile_paths"] == [
        "/opt/fabric-demo/profiles/hermes.yaml",
        "/opt/fabric-demo/profiles/telemetry.yaml",
    ]

    fabric_commands = [
        command
        for command in environment.commands
        if "nemo_fabric.integrations.harbor.runner" in command
    ]
    assert len(fabric_commands) == 1
    assert "python3 -m nemo_fabric.integrations.harbor.runner" in fabric_commands[0]
    assert environment.environments[environment.commands.index(fabric_commands[0])] == {
        "NVIDIA_API_KEY": "test-key"
    }
    assert context.metadata
    assert context.metadata["fabric"]["status"] == "succeeded"
    assert context.metadata["fabric"]["profiles"] == ["env_local", "mcp_github"]
    assert context.metadata["fabric"]["adapter_id"] == "nvidia.fabric.hermes.sdk"
    artifacts = context.metadata["fabric"]["artifacts"]["artifacts"]
    assert {artifact["name"] for artifact in artifacts} == {"stdout", "workspace_patch"}


if __name__ == "__main__":
    asyncio.run(main())
