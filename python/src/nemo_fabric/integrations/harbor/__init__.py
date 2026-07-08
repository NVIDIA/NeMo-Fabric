# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor consumer integration for NeMo Fabric."""

from __future__ import annotations

import importlib.metadata
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
        fabric_config_path: str,
        fabric_profile_paths: str | Sequence[str] | None = None,
        fabric_python: str = "python3",
        fabric_spec_path: str = "/tmp/fabric-run.json",
        fabric_result_path: str = "/logs/agent/fabric-result.json",
        fabric_install_command: str | None = None,
        fabric_cwd: str | None = None,
        fabric_timeout_sec: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, *args, **kwargs)
        self.fabric_config_path = fabric_config_path
        self.fabric_profile_paths = normalize_paths(fabric_profile_paths)
        self.fabric_python = fabric_python
        self.fabric_spec_path = fabric_spec_path
        self.fabric_result_path = fabric_result_path
        self.fabric_install_command = fabric_install_command
        self.fabric_cwd = fabric_cwd
        self.fabric_timeout_sec = fabric_timeout_sec

    @staticmethod
    def name() -> str:
        return "fabric"

    def version(self) -> str | None:
        try:
            return importlib.metadata.version("nemo-fabric-runtime")
        except importlib.metadata.PackageNotFoundError:
            return None

    async def setup(self, environment: BaseEnvironment) -> None:
        result = await environment.exec("mkdir -p /logs/agent /tmp", timeout_sec=30)
        ensure_success("Fabric setup failed", result)
        if self.fabric_install_command:
            result = await environment.exec(
                self.fabric_install_command,
                cwd=self.fabric_cwd,
                env=self.extra_env,
                timeout_sec=self.fabric_timeout_sec,
            )
            ensure_success("Fabric install command failed", result)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        spec = {
            "config_path": self.fabric_config_path,
            "profile_paths": self.fabric_profile_paths,
            "request": self._build_request(instruction),
        }
        result = await environment.exec(
            write_json_command(self.fabric_spec_path, spec),
            cwd=self.fabric_cwd,
            timeout_sec=30,
        )
        ensure_success("Fabric run specification write failed", result)

        result = await environment.exec(
            fabric_runner_command(
                fabric_python=self.fabric_python,
                spec_path=self.fabric_spec_path,
                result_path=self.fabric_result_path,
            ),
            cwd=self.fabric_cwd,
            env=self.extra_env,
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


def normalize_paths(paths: str | Sequence[str] | None) -> list[str]:
    if paths is None:
        return []
    if isinstance(paths, str):
        return [paths]
    return [path for path in paths if path]


def write_json_command(path: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, indent=2)
    return f"cat > {shlex.quote(path)} <<'FABRIC_JSON'\n{encoded}\nFABRIC_JSON"


def fabric_runner_command(
    *,
    fabric_python: str,
    spec_path: str,
    result_path: str,
) -> str:
    parts = [
        shlex.quote(fabric_python),
        "-m",
        "nemo_fabric.integrations.harbor.runner",
        "--spec",
        shlex.quote(spec_path),
        "--result",
        shlex.quote(result_path),
    ]
    return " ".join(parts)


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
        "profiles": result.get("profiles", []),
        "harness": result.get("harness"),
        "adapter_id": result.get("adapter_id"),
        "artifacts": result.get("artifacts", {}),
        "telemetry": result.get("telemetry"),
        "error": result.get("error"),
    }
