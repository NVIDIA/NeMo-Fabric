# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor agent implementation backed by the Fabric Python SDK."""

from __future__ import annotations

import importlib.metadata
import json
import shlex
import uuid
import warnings
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any
from typing import Literal
from typing import cast

from nemo_fabric import EnvironmentConfig
from nemo_fabric import FabricConfig
from nemo_fabric import HarnessConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import ModelConfig
from nemo_fabric import RelayAtifConfig
from nemo_fabric import RelayAtofConfig
from nemo_fabric import RelayObservabilityConfig
from nemo_fabric import RunRequest
from nemo_fabric import RunResult
from nemo_fabric import RuntimeConfig
from nemo_fabric import ToolsConfig
from nemo_fabric.integrations.harbor.models import FabricRunPayload
from nemo_fabric.integrations.harbor.models import HarborMcpServer

INSTALL_ENV_NAMES = {
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "PIP_CERT",
    "PIP_CLIENT_CERT",
    "PIP_EXTRA_INDEX_URL",
    "PIP_FIND_LINKS",
    "PIP_INDEX_URL",
    "PIP_TRUSTED_HOST",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}
HARBOR_ARTIFACT_ROOT = "/logs/agent/fabric-artifacts"
HARBOR_DEFAULT_WORKSPACE = "/testbed"

try:
    from harbor.agents.base import BaseAgent
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
except (
    ModuleNotFoundError
) as error:  # pragma: no cover - exercised without harbor extra
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

        SUPPORTS_ATIF = True

        def __init__(
            self,
            logs_dir: Path,
            fabric_adapter_id: str,
            fabric_config_base_dir: str | None = None,
            fabric_config_bundle: Path | None = None,
            fabric_config_target: str = "/tmp/nemo-fabric-config",
            fabric_workspace: str = HARBOR_DEFAULT_WORKSPACE,
            fabric_harness_settings: dict[str, Any] | None = None,
            fabric_blocked_tools: list[str] | None = None,
            fabric_telemetry: Literal["none", "relay"] = "none",
            fabric_python: str = "python3",
            fabric_package: str | None = None,
            fabric_venv_path: str = "/tmp/nemo-fabric-venv",
            fabric_install_command: str | None = None,
            fabric_cwd: str | None = None,
            fabric_timeout_sec: int | None = None,
            extra_env: dict[str, str] | None = None,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            super().__init__(logs_dir=logs_dir, extra_env=extra_env, *args, **kwargs)
            # Harbor passes agent-scoped environment variables to custom agents,
            # while newer BaseAgent versions intentionally ignore unknown kwargs.
            # Retain the mapping here for both old and new Harbor releases.
            self._extra_env = dict(extra_env or {})
            if not fabric_adapter_id.strip():
                raise ValueError("fabric_adapter_id must not be empty")
            workspace = PurePosixPath(fabric_workspace)
            if not workspace.is_absolute() or ".." in workspace.parts:
                raise ValueError(
                    "fabric_workspace must be an absolute task-environment path"
                )
            blocked_tools = list(fabric_blocked_tools or [])
            if any(
                not isinstance(tool, str) or not tool.strip() for tool in blocked_tools
            ):
                raise ValueError("fabric_blocked_tools must contain non-empty strings")
            if fabric_telemetry not in {"none", "relay"}:
                raise ValueError("fabric_telemetry must be 'none' or 'relay'")
            self.fabric_adapter_id = fabric_adapter_id
            self.fabric_config_base_dir = fabric_config_base_dir
            self.fabric_config_bundle = fabric_config_bundle
            self.fabric_config_target = fabric_config_target
            self.fabric_workspace = str(workspace)
            self.fabric_harness_settings = dict(fabric_harness_settings or {})
            self.fabric_blocked_tools = blocked_tools
            self.fabric_telemetry = fabric_telemetry
            self.fabric_python = fabric_python
            self.fabric_package = fabric_package
            self.fabric_venv_path = fabric_venv_path
            self.fabric_install_command = fabric_install_command
            self.fabric_cwd = fabric_cwd
            self.fabric_timeout_sec = fabric_timeout_sec
            if fabric_package and fabric_install_command:
                raise ValueError(
                    "fabric_package and fabric_install_command are mutually exclusive"
                )
            if fabric_install_command:
                warnings.warn(
                    "fabric_install_command is deprecated; use fabric_package",
                    DeprecationWarning,
                    stacklevel=2,
                )
            self._environment_config_base_dir = (
                self._resolve_environment_config_base_dir()
            )
            self._result_path: Path | None = None

        @staticmethod
        def name() -> str:
            return "fabric"

        def version(self) -> str | None:
            try:
                return importlib.metadata.version("nemo-fabric-runtime")
            except importlib.metadata.PackageNotFoundError:
                return None

        async def setup(self, environment: BaseEnvironment) -> None:
            setup_dirs = ["/logs/agent", "/tmp"]
            if self.fabric_config_bundle is not None:
                setup_dirs.append(self.fabric_config_target)
            result = await environment.exec(
                "mkdir -p " + " ".join(shlex.quote(path) for path in setup_dirs),
                timeout_sec=30,
            )
            ensure_success("Fabric setup failed", result)
            if self.fabric_config_bundle is not None:
                await environment.upload_dir(
                    self.fabric_config_bundle,
                    self.fabric_config_target,
                )
            if self.fabric_package:
                result = await environment.exec(
                    fabric_install_command(
                        fabric_python=self.fabric_python,
                        package=self.fabric_package,
                        venv_path=self.fabric_venv_path,
                    ),
                    cwd=self.fabric_cwd,
                    env=self._install_env,
                    timeout_sec=self.fabric_timeout_sec,
                )
                ensure_success("Fabric package installation failed", result)
            elif self.fabric_install_command:
                result = await environment.exec(
                    self.fabric_install_command,
                    cwd=self.fabric_cwd,
                    env=self._install_env,
                    timeout_sec=self.fabric_timeout_sec,
                )
                ensure_success("Fabric install command failed", result)

        async def run(
            self,
            instruction: str,
            environment: BaseEnvironment,
            context: AgentContext,
        ) -> None:
            self._result_path = None
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
                    fabric_python=self._runner_python,
                    spec_path=remote_spec_path,
                    result_path=remote_result_path,
                    path_prefix=self._runner_path_prefix,
                ),
                cwd=self.fabric_cwd,
                env=self._runner_env,
                timeout_sec=self.fabric_timeout_sec,
            )
            ensure_success("Fabric run failed", result)

            await environment.download_file(remote_result_path, host_result_path)
            self._result_path = host_result_path

        def _build_request(self, instruction: str) -> RunRequest:
            context = {"source": "harbor"}
            session_id = getattr(self, "session_id", None)
            context_id = getattr(self, "context_id", None)
            if session_id is not None:
                context["harbor_session_id"] = session_id
            if context_id is not None:
                context["harbor_context_id"] = str(context_id)
            return RunRequest(input=instruction, context=context)

        def _build_spec(self, instruction: str) -> FabricRunPayload:
            return FabricRunPayload(
                config=self._build_config(),
                config_base_dir=self._environment_config_base_dir,
                request=self._build_request(instruction),
            )

        def _build_config(self) -> FabricConfig:
            return build_harbor_config(
                adapter_id=self.fabric_adapter_id,
                workspace=self.fabric_workspace,
                harness_settings=self.fabric_harness_settings,
                blocked_tools=self.fabric_blocked_tools,
                telemetry=self.fabric_telemetry,
                model_name=self.model_name,
                skills_dir=self.skills_dir,
                mcp_servers=tuple(
                    HarborMcpServer.model_validate(server.model_dump(mode="python"))
                    for server in self.mcp_servers
                ),
            )

        def _resolve_environment_config_base_dir(self) -> str:
            if self.fabric_config_bundle is not None:
                bundle = Path(self.fabric_config_bundle)
                if not bundle.is_dir():
                    raise ValueError(
                        f"fabric_config_bundle must be an existing directory: {bundle}"
                    )
                target = PurePosixPath(self.fabric_config_target)
                if not target.is_absolute() or ".." in target.parts:
                    raise ValueError(
                        "fabric_config_target must be an absolute task-environment path"
                    )
                if self.fabric_config_base_dir is not None:
                    raise ValueError(
                        "fabric_config_base_dir is derived from fabric_config_target when fabric_config_bundle is set"
                    )
                return str(target)

            base_dir = PurePosixPath(
                self.fabric_config_base_dir or self.fabric_workspace
            )
            if not base_dir.is_absolute() or ".." in base_dir.parts:
                raise ValueError(
                    "fabric_config_base_dir must be an absolute task-environment path"
                )
            return str(base_dir)

        def populate_context_post_run(self, context: AgentContext) -> None:
            """Populate Harbor result metadata, token counts, and cost."""

            if self._result_path is not None:
                populate_context_from_result(context, self._result_path)
            populate_context_from_trajectory(context, self.logs_dir / "trajectory.json")
            populate_context_from_telemetry_summary(
                context,
                self.logs_dir / "telemetry-validation.json",
            )

        @property
        def _runner_python(self) -> str:
            if self.fabric_package is None:
                return self.fabric_python
            return str(PurePosixPath(self.fabric_venv_path) / "bin" / "python")

        @property
        def _runner_path_prefix(self) -> str | None:
            runner = PurePosixPath(self._runner_python)
            if not runner.is_absolute():
                return None
            return str(runner.parent)

        @property
        def _install_env(self) -> dict[str, str]:
            return {
                name: value
                for name, value in self._extra_env.items()
                if name in INSTALL_ENV_NAMES
            }

        @property
        def _runner_env(self) -> dict[str, str]:
            env = dict(self._extra_env)
            env["ADAPTER_PYTHON"] = self._runner_python
            return env


def build_harbor_config(
    *,
    adapter_id: str,
    workspace: str,
    harness_settings: dict[str, Any] | None = None,
    blocked_tools: list[str] | None = None,
    telemetry: Literal["none", "relay"] = "none",
    model_name: str | None = None,
    skills_dir: str | Path | None = None,
    mcp_servers: tuple[HarborMcpServer, ...] = (),
) -> FabricConfig:
    """Construct the typed config controlled by Harbor agent inputs."""

    name = f"harbor-{adapter_id.rsplit('.', maxsplit=1)[-1]}"
    artifact_root = f"{HARBOR_ARTIFACT_ROOT}/{name}"
    settings = harbor_harness_defaults(adapter_id)
    settings.update(harness_settings or {})
    config = FabricConfig(
        metadata=MetadataConfig(
            name=name,
            description="Fabric agent configured through Harbor run inputs.",
        ),
        harness=HarnessConfig(
            adapter_id=adapter_id,
            resolution="preinstalled",
            settings=settings,
        ),
        runtime=RuntimeConfig(
            input_schema="text",
            output_schema="message",
            artifacts=artifact_root,
        ),
        environment=EnvironmentConfig(
            provider="local",
            workspace=workspace,
            artifacts=artifact_root,
        ),
        tools=(ToolsConfig(blocked=list(blocked_tools)) if blocked_tools else None),
    )
    if model_name:
        config.models["default"] = ModelConfig(
            provider=model_provider(model_name),
            model=model_name,
        )
    for server in mcp_servers:
        if server.transport == "stdio":
            config.add_mcp_server(
                server.name,
                transport="stdio",
                url=cast(str, server.command),
                exposure="harness_native",
                extra_fields={"args": list(server.args)},
            )
        else:
            config.add_mcp_server(
                server.name,
                transport=server.transport,
                url=cast(str, server.url),
                exposure="harness_native",
            )
    if skills_dir is not None:
        config.add_skill_path(skills_dir)
    if telemetry == "relay":
        relay_output = f"{artifact_root}/relay"
        config.enable_relay(
            output_dir=relay_output,
            observability=RelayObservabilityConfig(
                atif=RelayAtifConfig(
                    enabled=True,
                    output_directory=relay_output,
                    filename_template="trajectory-{session_id}.atif.json",
                    agent_name=name,
                ),
                atof=RelayAtofConfig(
                    enabled=True,
                    output_directory=relay_output,
                    filename="events.atof.jsonl",
                    mode="overwrite",
                ),
            ),
        )
    return config


def model_provider(model_name: str) -> str:
    """Derive the Fabric provider from Harbor's model identifier."""

    return model_name.split("/", maxsplit=1)[0] if "/" in model_name else "openai"


def harbor_harness_defaults(adapter_id: str) -> dict[str, Any]:
    """Return the minimal unattended settings required in a Harbor task."""

    if adapter_id == "nvidia.fabric.hermes":
        return {
            "hermes_home": "/tmp/fabric-hermes",
            "terminal_timeout": 300,
        }
    if adapter_id == "nvidia.fabric.claude":
        return {
            "permission_mode": "bypassPermissions",
            "max_turns": 75,
            "timeout_seconds": 1800,
            "env": {"IS_SANDBOX": "1"},
        }
    return {}


def fabric_runner_command(
    *,
    fabric_python: str,
    spec_path: str,
    result_path: str,
    path_prefix: str | None = None,
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
    command = " ".join(parts)
    if path_prefix is not None:
        return f"PATH={shlex.quote(path_prefix)}:$PATH {command}"
    return command


def fabric_install_command(*, fabric_python: str, package: str, venv_path: str) -> str:
    """Build a shell-safe pip installation command for one package requirement."""

    if not package.strip():
        raise ValueError("fabric_package must not be empty")
    venv = PurePosixPath(venv_path)
    if not venv.is_absolute() or ".." in venv.parts:
        raise ValueError("fabric_venv_path must be an absolute task-environment path")
    venv_python = venv / "bin" / "python"
    return (
        f"{shlex.quote(fabric_python)} -m venv {shlex.quote(str(venv))} && "
        f"{shlex.quote(str(venv_python))} -m pip install "
        f"--disable-pip-version-check {shlex.quote(package)}"
    )


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


def populate_context_from_trajectory(context: AgentContext, path: Path) -> None:
    """Validate canonical ATIF with Harbor and backfill usage fields when present."""

    if not path.is_file():
        return
    from harbor.models.trajectories.trajectory import Trajectory

    try:
        trajectory = Trajectory.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        _record_host_atif_validation(context, status="failed", error=str(error))
        return
    _record_host_atif_validation(context, status="succeeded")
    metrics = trajectory.final_metrics
    if metrics is None:
        return
    context.n_input_tokens = metrics.total_prompt_tokens
    context.n_cache_tokens = metrics.total_cached_tokens
    context.n_output_tokens = metrics.total_completion_tokens
    context.cost_usd = metrics.total_cost_usd


def _record_host_atif_validation(
    context: AgentContext, *, status: str, error: str | None = None
) -> None:
    if context.metadata is None:
        context.metadata = {}
    fabric = context.metadata.setdefault("fabric", {})
    if isinstance(fabric, dict):
        fabric["harbor_atif_validation"] = {
            "status": status,
            "error": error,
        }


def populate_context_from_telemetry_summary(context: AgentContext, path: Path) -> None:
    """Attach telemetry quality evidence to Fabric's Harbor metadata."""

    if not path.is_file():
        return
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        summary = {
            "status": "failed",
            "error": "telemetry summary could not be loaded",
        }
    if context.metadata is None:
        context.metadata = {}
    fabric = context.metadata.setdefault("fabric", {})
    if isinstance(fabric, dict):
        fabric["telemetry_validation"] = summary
