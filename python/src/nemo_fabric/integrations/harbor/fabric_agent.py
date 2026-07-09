# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor agent implementation backed by the Fabric Python SDK."""

from __future__ import annotations

import importlib.metadata
import json
import shlex
import uuid
from pathlib import Path
from typing import Any

from nemo_fabric import RunRequest, RunResult
from nemo_fabric.integrations.harbor.models import HarborMcpServer, HarborRunSpec

try:
    from harbor.agents.base import BaseAgent
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
except ModuleNotFoundError as error:  # pragma: no cover - exercised without harbor extra
    _HARBOR_IMPORT_ERROR = error
else:
    _HARBOR_IMPORT_ERROR = None


if _HARBOR_IMPORT_ERROR is not None:

    class FabricAgent:
        """Placeholder that reports the missing Harbor optional dependency."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ModuleNotFoundError(
                "nemo_fabric.integrations.harbor requires the Harbor optional "
                "dependency; install nemo-fabric with the harbor extra"
            ) from _HARBOR_IMPORT_ERROR

        @staticmethod
        def name() -> str:
            return "fabric"

else:

    class FabricAgent(BaseAgent):
        """Harbor agent wrapper that delegates harness execution to Fabric.

        Harbor owns task materialization, environment lifecycle, verification, and
        reward calculation. Fabric owns the selected agent harness invocation.
        """

        def __init__(
            self,
            logs_dir: Path,
            fabric_config_path: str,
            fabric_python: str = "python3",
            fabric_install_command: str | None = None,
            fabric_cwd: str | None = None,
            fabric_timeout_sec: int | None = None,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            super().__init__(logs_dir=logs_dir, *args, **kwargs)
            self.fabric_config_path = fabric_config_path
            self.fabric_python = fabric_python
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
            token = uuid.uuid4().hex
            spec = self._build_spec(instruction)
            host_spec_path = self.logs_dir / f"fabric-run-{token}.json"
            remote_spec_path = f"/tmp/fabric-run-{token}.json"
            remote_result_path = f"/tmp/fabric-result-{token}.json"
            host_result_path = self.logs_dir / f"fabric-result-{token}.json"
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            host_spec_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
            await environment.upload_file(host_spec_path, remote_spec_path)

            result = await environment.exec(
                fabric_runner_command(
                    fabric_python=self.fabric_python,
                    spec_path=remote_spec_path,
                    result_path=remote_result_path,
                ),
                cwd=self.fabric_cwd,
                env=self.extra_env,
                timeout_sec=self.fabric_timeout_sec,
            )
            ensure_success("Fabric run failed", result)

            await environment.download_file(remote_result_path, host_result_path)
            populate_context_from_result(context, host_result_path)

        def _build_request(self, instruction: str) -> RunRequest:
            return RunRequest(input=instruction, context={"source": "harbor"})

        def _build_spec(self, instruction: str) -> HarborRunSpec:
            return HarborRunSpec(
                config_path=self.fabric_config_path,
                request=self._build_request(instruction),
                model_name=self.model_name,
                skills_dir=self.skills_dir,
                mcp_servers=tuple(
                    HarborMcpServer.model_validate(server.model_dump(mode="python"))
                    for server in self.mcp_servers
                ),
            )


def fabric_runner_command(
    *,
    fabric_python: str,
    spec_path: str,
    result_path: str,
) -> str:
    """Build the sandbox-local runner command."""

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
    """Raise when a Harbor environment command fails."""

    if getattr(result, "return_code", 1) == 0:
        return
    stdout = getattr(result, "stdout", "")
    stderr = getattr(result, "stderr", "")
    raise RuntimeError(f"{message} (exit {result.return_code}): {stderr or stdout}")


def populate_context_from_result(context: AgentContext, path: Path) -> RunResult:
    """Validate a downloaded result and copy its summary into Harbor metadata."""

    result = RunResult.from_mapping(json.loads(path.read_text(encoding="utf-8")))
    mapping = result.to_mapping()
    if context.metadata is None:
        context.metadata = {}
    context.metadata["fabric"] = {
        "status": mapping["status"],
        "runtime_id": mapping["runtime_id"],
        "invocation_id": mapping["invocation_id"],
        "request_id": mapping["request_id"],
        "harness": mapping["harness"],
        "adapter_id": mapping.get("adapter_id"),
        "artifacts": mapping["artifacts"],
        "telemetry": mapping["telemetry"],
        "error": mapping.get("error"),
    }
    return result
