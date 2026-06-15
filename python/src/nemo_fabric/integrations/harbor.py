"""Harbor consumer integration for NeMo Fabric."""

from __future__ import annotations

import json
import shlex
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class FabricAgent(BaseAgent):
    """Harbor agent wrapper that delegates harness execution to Fabric.

    Harbor owns task materialization, environment lifecycle, verification, and
    reward calculation. Fabric owns the selected agent harness invocation.
    """

    def __init__(
        self,
        logs_dir: Path,
        fabric_agent_path: str,
        fabric_profiles: str | Sequence[str] | None = None,
        fabric_cli: str = "fabric",
        fabric_request_path: str = "/tmp/fabric-request.json",
        fabric_result_path: str = "/logs/agent/fabric-result.json",
        fabric_install_command: str | None = None,
        fabric_cwd: str | None = None,
        fabric_timeout_sec: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, *args, **kwargs)
        self.fabric_agent_path = fabric_agent_path
        self.fabric_profiles = normalize_profiles(fabric_profiles)
        self.fabric_cli = fabric_cli
        self.fabric_request_path = fabric_request_path
        self.fabric_result_path = fabric_result_path
        self.fabric_install_command = fabric_install_command
        self.fabric_cwd = fabric_cwd
        self.fabric_timeout_sec = fabric_timeout_sec

    @staticmethod
    def name() -> str:
        return "fabric"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        result = await environment.exec("mkdir -p /logs/agent /tmp", timeout_sec=30)
        ensure_success("Fabric setup failed", result)
        if self.fabric_install_command:
            result = await environment.exec(
                self.fabric_install_command,
                cwd=self.fabric_cwd,
                timeout_sec=self.fabric_timeout_sec,
            )
            ensure_success("Fabric install command failed", result)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        request = self._build_request(instruction)
        result = await environment.exec(
            write_json_command(self.fabric_request_path, request),
            cwd=self.fabric_cwd,
            timeout_sec=30,
        )
        ensure_success("Fabric request write failed", result)

        result = await environment.exec(
            fabric_run_command(
                fabric_cli=self.fabric_cli,
                fabric_agent_path=self.fabric_agent_path,
                fabric_profiles=self.fabric_profiles,
                request_path=self.fabric_request_path,
                result_path=self.fabric_result_path,
            ),
            cwd=self.fabric_cwd,
            timeout_sec=self.fabric_timeout_sec,
        )
        ensure_success("Fabric run failed", result)

        host_result_path = self.logs_dir / "fabric-result.json"
        await environment.download_file(self.fabric_result_path, host_result_path)
        populate_context_from_result(context, host_result_path)

    def _build_request(self, instruction: str) -> dict[str, Any]:
        return {
            "request_id": f"harbor-{uuid.uuid4()}",
            "input": instruction,
            "context": {
                "source": "harbor",
                "model_name": self.model_name,
                "skills_dir": self.skills_dir,
                "mcp_servers": [dump_mcp_server(server) for server in self.mcp_servers],
            },
        }


def normalize_profiles(profiles: str | Sequence[str] | None) -> list[str]:
    if profiles is None:
        return []
    if isinstance(profiles, str):
        return [profiles]
    return [profile for profile in profiles if profile]


def write_json_command(path: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, indent=2)
    return f"cat > {shlex.quote(path)} <<'FABRIC_JSON'\n{encoded}\nFABRIC_JSON"


def fabric_run_command(
    *,
    fabric_cli: str,
    fabric_agent_path: str,
    fabric_profiles: Sequence[str],
    request_path: str,
    result_path: str,
) -> str:
    parts = [
        shlex.quote(fabric_cli),
        "run",
        shlex.quote(fabric_agent_path),
        "--request-file",
        shlex.quote(request_path),
    ]
    for profile in fabric_profiles:
        parts.extend(["--profile", shlex.quote(profile)])
    return f"{' '.join(parts)} > {shlex.quote(result_path)}"


def ensure_success(message: str, result: Any) -> None:
    if getattr(result, "return_code", 1) == 0:
        return
    stdout = getattr(result, "stdout", "")
    stderr = getattr(result, "stderr", "")
    raise RuntimeError(f"{message} (exit {result.return_code}): {stderr or stdout}")


def dump_mcp_server(server: Any) -> dict[str, Any]:
    if hasattr(server, "model_dump"):
        return server.model_dump(mode="json")
    if hasattr(server, "dict"):
        return server.dict()
    return dict(server)


def populate_context_from_result(context: AgentContext, path: Path) -> None:
    result = json.loads(path.read_text(encoding="utf-8"))
    if context.metadata is None:
        context.metadata = {}
    context.metadata["fabric"] = {
        "status": result.get("status"),
        "runtime_id": result.get("runtime_id"),
        "invocation_id": result.get("invocation_id"),
        "request_id": result.get("request_id"),
        "profile": result.get("profile"),
        "harness_type": result.get("harness_type"),
        "adapter_id": result.get("adapter_id"),
        "artifacts": result.get("artifacts", {}),
        "telemetry": result.get("telemetry"),
        "error": result.get("error"),
    }
