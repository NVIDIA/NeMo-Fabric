"""Smoke test for the Harbor consumer integration."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

try:
    from nemo_fabric.integrations.harbor import FabricAgent
    from harbor.models.agent.context import AgentContext
except ImportError as exc:
    raise SystemExit(
        "Install Harbor before running this smoke, for example: pip install -e ../harbor"
    ) from exc


@dataclass
class ExecResult:
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""


class FakeHarborEnvironment:
    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.commands: list[str] = []

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        timeout_sec: int | None = None,
        **_: Any,
    ) -> ExecResult:
        self.commands.append(command)
        if command.startswith("cat > "):
            path, contents = command.split(" <<'FABRIC_JSON'\n", maxsplit=1)
            path = path.removeprefix("cat > ").strip()
            contents = contents.removesuffix("\nFABRIC_JSON")
            self.files[path] = contents
            return ExecResult()
        if "fabric run" in command and "> /logs/agent/fabric-result.json" in command:
            self.files["/logs/agent/fabric-result.json"] = json.dumps(
                {
                    "status": "succeeded",
                    "runtime_id": "runtime-1",
                    "invocation_id": "invocation-1",
                    "request_id": "harbor-request-1",
                    "profile": "env_local",
                    "harness_type": "hermes",
                    "adapter_id": "nvidia.fabric.hermes.sdk",
                    "artifacts": {"artifacts": []},
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
            fabric_agent_path="/workspace/agent",
            fabric_profiles=["env_local", "mcp_github"],
            fabric_cli="fabric",
            model_name="nvidia/test-model",
        )
        environment = FakeHarborEnvironment()
        context = AgentContext()

        await agent.setup(environment)  # type: ignore[arg-type]
        await agent.run("fix the bug", environment, context)  # type: ignore[arg-type]

    request = json.loads(environment.files["/tmp/fabric-request.json"])
    assert request["input"] == "fix the bug"
    assert request["context"]["source"] == "harbor"
    assert request["context"]["model_name"] == "nvidia/test-model"

    fabric_commands = [command for command in environment.commands if "fabric run" in command]
    assert len(fabric_commands) == 1
    assert "--profile env_local --profile mcp_github" in fabric_commands[0]
    assert context.metadata
    assert context.metadata["fabric"]["status"] == "succeeded"
    assert context.metadata["fabric"]["adapter_id"] == "nvidia.fabric.hermes.sdk"


if __name__ == "__main__":
    asyncio.run(main())
