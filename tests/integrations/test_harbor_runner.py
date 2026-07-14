# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib.util
import json
import os
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
ROOT_README = ROOT / "README.md"
DEMO_ROOT = ROOT / "examples" / "harbor" / "demo"
DEMO_README = DEMO_ROOT / "README.md"
DEMO_DOCKERFILE = DEMO_ROOT / "task" / "environment" / "Dockerfile"
DEMO_HOST_GATEWAY = DEMO_ROOT / "host-gateway.compose.yaml"
DEMO_SOLUTION = DEMO_ROOT / "task" / "solution" / "solve.sh"
DEMO_CONFIGS = DEMO_ROOT / "task" / "environment" / "fabric" / "configs"
SWEBENCH_ROOT = ROOT / "examples" / "harbor" / "swebench"
SWEBENCH_CONFIGS = SWEBENCH_ROOT / "configs"
CODEX_CONFIG = DEMO_ROOT / "task" / "environment" / "fabric" / "configs" / "codex.yaml"
RELAY_CONFIG = DEMO_ROOT / "task" / "environment" / "fabric" / "configs" / "hermes-relay.yaml"
INTEGRATION_README = ROOT / "examples" / "harbor" / "README.md"
SDK_INTEGRATION_README = ROOT / "python" / "src" / "nemo_fabric" / "integrations" / "harbor" / "README.md"
HARBOR_PACKAGE_INIT = SDK_INTEGRATION_README.parent / "__init__.py"


def load_codex_adapter():
    path = ROOT / "adapters/codex-cli/src/nemo_fabric_adapters/codex_cli/adapter.py"
    spec = importlib.util.spec_from_file_location("fabric_codex_adapter", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_composes_harbor_values_on_an_independent_config(tmp_path):
    from nemo_fabric import RunRequest
    from nemo_fabric.integrations.harbor.models import HarborMcpServer
    from nemo_fabric.integrations.harbor.models import HarborRunSpec
    from nemo_fabric.integrations.harbor.runner import compose_config
    from nemo_fabric.integrations.harbor.runner import load_config

    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "metadata": {"name": "harbor-demo"},
                "harness": {"adapter_id": "demo.fabric.smoke"},
                "runtime": {},
                "models": {
                    "default": {
                        "provider": "demo",
                        "model": "demo",
                        "api_key_env": "DEMO_API_KEY",
                        "temperature": 0.25,
                    }
                },
                "mcp": {
                    "servers": {
                        "base": {
                            "transport": "streamable-http",
                            "url": "https://base.example.test",
                        }
                    }
                },
                "skills": {"paths": ["./base-skill"]},
            }
        ),
        encoding="utf-8",
    )
    spec = HarborRunSpec(
        config_path=config_path,
        request=RunRequest(input="fix it"),
        model_name="openai/gpt-5.4",
        skills_dir=tmp_path / "skills",
        mcp_servers=(
            HarborMcpServer(
                name="remote",
                transport="streamable-http",
                url="https://mcp.example.test",
            ),
            HarborMcpServer(
                name="local",
                transport="stdio",
                command="mcp-server",
                args=("--stdio",),
            ),
        ),
    )

    base = load_config(config_path)
    config = compose_config(base, spec)

    assert base.models["default"].to_mapping() == {
        "provider": "demo",
        "model": "demo",
        "api_key_env": "DEMO_API_KEY",
        "temperature": 0.25,
    }
    assert base.mcp is not None and "base" in base.mcp.servers
    assert base.skills is not None and base.skills.paths == ["./base-skill"]
    assert config.models["default"].to_mapping() == {
        "provider": "openai",
        "model": "openai/gpt-5.4",
        "temperature": 0.25,
    }
    assert config.mcp is not None
    assert set(config.mcp.servers) == {"remote", "local"}
    assert config.mcp.servers["local"].url == "mcp-server"
    assert config.mcp.servers["local"].extra_fields["args"] == ["--stdio"]
    assert config.skills is not None
    assert config.skills.paths == [str(tmp_path / "skills")]
    assert json.loads(json.dumps(config.to_mapping()))["metadata"]["name"] == ("harbor-demo")


def test_runner_preserves_config_capabilities_without_harbor_replacements(tmp_path):
    from nemo_fabric import FabricConfig
    from nemo_fabric import RunRequest
    from nemo_fabric.integrations.harbor.models import HarborRunSpec
    from nemo_fabric.integrations.harbor.runner import compose_config

    base = FabricConfig.model_validate(
        {
            "metadata": {"name": "harbor-demo"},
            "harness": {"adapter_id": "demo.fabric.smoke"},
            "mcp": {
                "servers": {
                    "base": {
                        "transport": "streamable-http",
                        "url": "https://base.example.test",
                    }
                }
            },
            "skills": {"paths": ["./base-skill"]},
        }
    )
    spec = HarborRunSpec(
        config_path=tmp_path / "agent.yaml",
        request=RunRequest(input="fix it"),
    )

    config = compose_config(base, spec)

    assert config.mcp is not None and set(config.mcp.servers) == {"base"}
    assert config.skills is not None and config.skills.paths == ["./base-skill"]


def test_runner_rejects_missing_config(tmp_path):
    from nemo_fabric.integrations.harbor.runner import load_config

    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_runner_rejects_malformed_config(tmp_path):
    from nemo_fabric.integrations.harbor.runner import load_config

    config_path = tmp_path / "agent.yaml"
    config_path.write_text("harness: [", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_config(config_path)


def test_harbor_transport_models_validate_mcp_targets():
    from nemo_fabric import RunRequest
    from nemo_fabric.integrations.harbor.models import HarborMcpServer
    from nemo_fabric.integrations.harbor.models import HarborRunSpec
    from pydantic import ValidationError

    spec = HarborRunSpec.model_validate_json(
        HarborRunSpec(
            config_path="/workspace/agent.yaml",
            request=RunRequest(input="fix it"),
            mcp_servers=(
                HarborMcpServer(
                    name="github",
                    transport="streamable-http",
                    url="https://mcp.example.test",
                ),
            ),
        ).model_dump_json()
    )

    assert spec.request.input == "fix it"
    assert spec.mcp_servers[0].name == "github"
    assert "profile_paths" not in HarborRunSpec.model_json_schema()["properties"]
    with pytest.raises(ValidationError, match="require url"):
        HarborMcpServer(name="missing", transport="sse")
    with pytest.raises(ValidationError, match="require command"):
        HarborMcpServer(name="missing", transport="stdio")
    with pytest.raises(ValidationError, match="Extra inputs"):
        HarborRunSpec.model_validate(
            {
                "config_path": "/workspace/agent.yaml",
                "request": {"input": "fix it"},
                "profile_paths": [],
            }
        )


def test_each_harbor_job_delegates_to_an_independent_fabric_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from nemo_fabric.integrations.harbor import runner

    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "metadata": {"name": "harbor-demo"},
                "harness": {"adapter_id": "demo.fabric.smoke"},
                "runtime": {},
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    class FakeResult:
        def __init__(self, runtime_id: str) -> None:
            self.runtime_id = runtime_id

        def to_mapping(self) -> dict[str, str]:
            return {"runtime_id": self.runtime_id}

    class FakeFabric:
        async def run(self, config, *, base_dir, request):
            runtime_id = f"runtime-{len(calls) + 1}"
            calls.append(
                {
                    "runtime_id": runtime_id,
                    "request": request.to_mapping(),
                }
            )
            return FakeResult(runtime_id)

    monkeypatch.setattr(runner, "Fabric", FakeFabric)
    monkeypatch.setattr(
        runner,
        "publish_telemetry_evidence",
        lambda result, path, **kwargs: {},
    )
    from nemo_fabric.integrations.harbor.models import HarborRunSpec

    specs = [
        HarborRunSpec(
            config_path=config_path,
            request={
                "input": f"job {job_id}",
                "context": {"job_id": job_id},
            },
        )
        for job_id in ("job-1", "job-2")
    ]

    async def run_specs():
        return await asyncio.gather(*(runner.run(spec) for spec in specs))

    results = asyncio.run(run_specs())

    assert [result.runtime_id for result in results] == ["runtime-1", "runtime-2"]
    requests = [call["request"] for call in calls]
    assert [request["context"]["job_id"] for request in requests] == [  # type: ignore[index]
        "job-1",
        "job-2",
    ]


def test_codex_adapter_maps_fabric_request_to_cli(tmp_path):
    adapter = load_codex_adapter()

    payload = {
        "effective_config": {
            "config_root": str(tmp_path),
            "config": {
                "harness": {
                    "settings": {
                        "sandbox": "workspace-write",
                        "skip_git_repo_check": True,
                        "config_overrides": {"model_reasoning_effort": "high"},
                    }
                },
                "models": {
                    "default": {
                        "provider": "openai",
                        "model": "openai/gpt-5.4",
                    }
                },
                "runtime": {},
            },
        },
        "runtime_context": {
            "runtime_id": "harbor-test",
            "environment": {"workspace": str(tmp_path)},
        },
        "request": {"input": "Fix the calculator."},
    }

    os.environ["CODEX_HOME"] = str(tmp_path)
    profile_name = "fabric-harbor-test"
    profile_path = tmp_path / f"{profile_name}.config.toml"
    codex_settings = adapter.write_config_files(payload)
    command = adapter.build_command(
        payload,
        codex_settings=codex_settings,
    )

    assert command == [
        "codex",
        "exec",
        "--json",
        "--sandbox",
        "workspace-write",
        "--profile",
        profile_name,
        "--model",
        "gpt-5.4",
        "--skip-git-repo-check",
        "-",
    ]
    assert tomllib.loads(profile_path.read_text(encoding="utf-8")) == {"model_reasoning_effort": "high"}
    assert adapter.resolve_cwd(payload) == tmp_path


def test_codex_demo_uses_current_adapter_contract():
    config = yaml.safe_load(CODEX_CONFIG.read_text(encoding="utf-8"))
    settings = config["harness"]["settings"]

    assert config["schema_version"] == "fabric.agent/v1alpha1"
    assert config["harness"]["adapter_id"] == "nvidia.fabric.codex.cli"
    assert settings["sandbox"] == "danger-full-access"
    assert settings["skip_git_repo_check"] is True
    assert settings["config_overrides"]["model_reasoning_effort"] == "high"
    dockerfile = DEMO_DOCKERFILE.read_text(encoding="utf-8")
    assert "nemo-fabric[codex,harbor,hermes,relay,runtime]" in dockerfile
    assert "@openai/codex@0.142.4" in dockerfile


def test_harbor_demo_uses_complete_configs_without_profiles():
    from nemo_fabric import FabricConfig

    configs = sorted(DEMO_CONFIGS.glob("*.yaml"))

    assert [path.name for path in configs] == [
        "codex.yaml",
        "hermes-relay.yaml",
        "hermes.yaml",
        "smoke.yaml",
    ]
    for path in configs:
        config = FabricConfig.model_validate(yaml.safe_load(path.read_text()))
        assert config.profiles is None
    assert not list((DEMO_CONFIGS.parent / "profiles").glob("*.yaml"))


def test_harbor_smoke_config_resolves_its_local_adapter():
    from nemo_fabric import Fabric
    from nemo_fabric import RunRequest
    from nemo_fabric.integrations.harbor.models import HarborRunSpec
    from nemo_fabric.integrations.harbor.runner import compose_config
    from nemo_fabric.integrations.harbor.runner import load_config

    config_path = DEMO_CONFIGS / "smoke.yaml"
    spec = HarborRunSpec(config_path=config_path, request=RunRequest(input="fix it"))
    config = compose_config(load_config(config_path), spec)
    plan = Fabric().plan(config, base_dir=config_path.parent)

    assert plan.adapter.adapter_id == "demo.fabric.scripted"
    assert plan["adapter_descriptor"]["source"] == "local"
    assert plan["adapter_descriptor"]["root"].endswith("configs/adapters/scripted")


def test_harbor_demo_documents_explicit_cli_commands():
    demo = DEMO_README.read_text(encoding="utf-8")
    integration = INTEGRATION_README.read_text(encoding="utf-8")

    assert "run.sh" not in demo
    assert "demo/run.sh" not in integration
    assert demo.count("uv run --extra runtime --extra harbor harbor run") == 1
    for flag in (
        "--path",
        "--agent",
        "--ak",
        "--job-name",
    ):
        assert flag in demo
    for value in (
            "swe-bench/swe-bench-verified",
            "django__django-13741",
            "--task swe-bench/django__django-13741",
        "--skill",
        "--mcp-config",
        "harbor job resume",
        "telemetry-validation.json",
        "agent/trajectory.json",
    ):
        assert value in integration


def test_harbor_demo_setup_and_solution_fail_fast():
    dockerfile = DEMO_DOCKERFILE.read_text(encoding="utf-8")
    solution = DEMO_SOLUTION.read_text(encoding="utf-8")

    assert "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |" not in dockerfile
    assert "-o /tmp/rustup-init.sh" in dockerfile
    assert "if updated == source:" in solution
    assert "raise SystemExit" in solution


def test_harbor_telemetry_demo_exports_phoenix_atof_and_atif():
    config = yaml.safe_load(RELAY_CONFIG.read_text(encoding="utf-8"))
    host_gateway = yaml.safe_load(DEMO_HOST_GATEWAY.read_text(encoding="utf-8"))
    assert "relay" in config["telemetry"]["providers"]
    observability = config["relay"]["observability"]
    integration = INTEGRATION_README.read_text(encoding="utf-8")

    assert observability["openinference"] == {
        "enabled": True,
        "transport": "http_binary",
        "endpoint": "http://host.docker.internal:6006/v1/traces",
    }
    assert observability["atof"]["enabled"] is True
    assert observability["atif"]["enabled"] is True
    assert host_gateway == {
        "services": {
            "main": {
                "extra_hosts": ["host.docker.internal=host-gateway"],
            }
        }
    }
    assert "ATOF JSONL" in integration
    assert "telemetry-validation.json" in integration
    assert "canonical ATIF" in integration


def test_harbor_sdk_package_documents_execution_boundary():
    from nemo_fabric.integrations.harbor import FabricAgent

    readme = SDK_INTEGRATION_README.read_text(encoding="utf-8")
    package_init = HARBOR_PACKAGE_INIT.read_text(encoding="utf-8")

    assert FabricAgent.name() == "fabric"
    assert FabricAgent.__module__ == "nemo_fabric.integrations.harbor.fabric_agent"
    assert "nemo_fabric.integrations.harbor:FabricAgent" in readme
    assert "runner.py" in readme
    assert "calls `Fabric.run()` directly" in readme
    assert "fabric_profile_paths" not in readme
    assert "`fabric_agent.py`" in readme
    assert "class FabricAgent" not in package_init
    assert "fabric_agent import FabricAgent" in package_init


def test_swebench_matrix_uses_complete_configs_and_one_fixed_task():
    from nemo_fabric import FabricConfig

    configs = sorted(SWEBENCH_CONFIGS.glob("*.yaml"))
    assert [path.name for path in configs] == [
        "codex.yaml",
        "hermes-relay.yaml",
        "hermes-tools.yaml",
        "hermes.yaml",
    ]
    for path in configs:
        config = FabricConfig.model_validate(yaml.safe_load(path.read_text()))
        assert config.profiles is None
        assert config.environment is not None
        assert str(config.environment.workspace) == "/testbed"

    assert (SWEBENCH_CONFIGS / "adapters/hermes-cli/fabric-adapter.json").read_text() == (
        ROOT / "adapters/hermes-cli/fabric-adapter.json"
    ).read_text()
    assert (SWEBENCH_CONFIGS / "adapters/codex-cli/fabric-adapter.json").read_text() == (
        ROOT / "adapters/codex-cli/fabric-adapter.json"
    ).read_text()

    readme = INTEGRATION_README.read_text(encoding="utf-8")
    assert readme.count("django__django-13741") >= 4
    assert "500 tasks" in readme
    assert "--n-tasks 5" in readme


def test_harbor_018_factory_loads_fabric_agent(tmp_path: Path):
    from harbor.agents.factory import AgentFactory

    agent = AgentFactory.create_agent_from_import_path(
        "nemo_fabric.integrations.harbor:FabricAgent",
        logs_dir=tmp_path,
        fabric_config_path="/opt/fabric/agent.yaml",
    )

    assert agent.name() == "fabric"
    assert agent.SUPPORTS_ATIF is True


def test_harbor_018_loads_swebench_mcp_config():
    from harbor.cli.utils import load_mcp_servers

    servers = load_mcp_servers(SWEBENCH_ROOT / "mcp.json")

    assert len(servers) == 1
    assert servers[0].name == "fabric-repo-inspector"
    assert servers[0].transport == "stdio"
    assert servers[0].command == "python3"
    assert servers[0].args == ["/tmp/nemo-fabric-config/mcp/repo_inspector.py"]


def test_root_readme_routes_to_sdk_and_harbor_guides():
    readme = ROOT_README.read_text(encoding="utf-8")

    assert "runtime execution layer" in readme
    assert "docs/sdk/python.mdx" in readme
    assert "examples/harbor/README.md" in readme
    assert "examples/harbor/demo/README.md" in readme

