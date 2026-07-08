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
DEMO_ROOT = ROOT / "integrations" / "harbor" / "demo"
DEMO_README = DEMO_ROOT / "README.md"
DEMO_DOCKERFILE = DEMO_ROOT / "task" / "environment" / "Dockerfile"
DEMO_HOST_GATEWAY = DEMO_ROOT / "host-gateway.compose.yaml"
DEMO_SOLUTION = DEMO_ROOT / "task" / "solution" / "solve.sh"
DEMO_CONFIGS = DEMO_ROOT / "task" / "environment" / "fabric" / "configs"
CODEX_CONFIG = (
    DEMO_ROOT
    / "task"
    / "environment"
    / "fabric"
    / "configs"
    / "codex.yaml"
)
RELAY_CONFIG = (
    DEMO_ROOT
    / "task"
    / "environment"
    / "fabric"
    / "configs"
    / "hermes-relay.yaml"
)
INTEGRATION_README = ROOT / "integrations" / "harbor" / "README.md"
SDK_INTEGRATION_README = (
    ROOT
    / "python"
    / "src"
    / "nemo_fabric"
    / "integrations"
    / "harbor"
    / "README.md"
)


def load_codex_adapter():
    path = ROOT / "adapters/codex-cli/src/nemo_fabric_adapters/codex_cli/adapter.py"
    spec = importlib.util.spec_from_file_location("fabric_codex_adapter", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_composes_harbor_values_on_an_independent_config(tmp_path):
    from nemo_fabric import RunRequest
    from nemo_fabric.integrations.harbor.models import HarborMcpServer, HarborRunSpec
    from nemo_fabric.integrations.harbor.runner import compose_config, load_config

    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "metadata": {"name": "harbor-demo"},
                "harness": {"adapter_id": "demo.fabric.smoke"},
                "runtime": {},
                "models": {"default": {"provider": "demo", "model": "demo"}},
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
    }
    assert base.mcp is not None and "base" in base.mcp.servers
    assert base.skills is not None and base.skills.paths == ["./base-skill"]
    assert config.models["default"].to_mapping() == {
        "provider": "openai",
        "model": "openai/gpt-5.4",
    }
    assert config.mcp is not None
    assert set(config.mcp.servers) == {"remote", "local"}
    assert config.mcp.servers["local"].url == "mcp-server"
    assert config.mcp.servers["local"].extra_fields["args"] == ["--stdio"]
    assert config.skills is not None
    assert config.skills.paths == [str(tmp_path / "skills")]
    assert json.loads(json.dumps(config.to_mapping()))["metadata"]["name"] == (
        "harbor-demo"
    )


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
    from pydantic import ValidationError

    from nemo_fabric import RunRequest
    from nemo_fabric.integrations.harbor.models import HarborMcpServer, HarborRunSpec

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
    assert tomllib.loads(profile_path.read_text(encoding="utf-8")) == {
        "model_reasoning_effort": "high"
    }
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
    assert 'nemo-fabric[codex,harbor,hermes,relay]' in dockerfile
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
    from nemo_fabric import Fabric, RunRequest
    from nemo_fabric.integrations.harbor.models import HarborRunSpec
    from nemo_fabric.integrations.harbor.runner import compose_config, load_config

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
    assert demo.count("uv run --extra harbor harbor run") == 4
    for flag in (
        "--path",
        "--agent",
        "--ak",
        "--ae",
        "--model",
        "--mounts",
        "--job-name",
    ):
        assert flag in demo
    assert "OPENAI_API_KEY" not in demo
    assert "CODEX_API_KEY" not in demo
    assert "CODEX_HOME" in demo
    assert "open http://localhost:6006" not in demo


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
    observability = config["telemetry"]["config"]["components"][0]["config"]
    demo = DEMO_README.read_text(encoding="utf-8")

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
    assert '--extra-docker-compose "$DEMO_DIR/host-gateway.compose.yaml"' in demo
    assert "Docker Desktop's" not in demo
    assert "arizephoenix/phoenix" in demo
    assert "http://localhost:6006" in demo
    assert "events.atof.jsonl" in demo
    assert "*.atif.json" in demo


def test_harbor_sdk_package_documents_execution_boundary():
    from nemo_fabric.integrations.harbor import FabricAgent

    readme = SDK_INTEGRATION_README.read_text(encoding="utf-8")

    assert FabricAgent.name() == "fabric"
    assert "nemo_fabric.integrations.harbor:FabricAgent" in readme
    assert "nemo_fabric.integrations.harbor.runner" in readme
    assert "calls `Fabric.run()` directly" in readme
    assert "fabric_profile_paths" not in readme


def test_root_readme_routes_to_sdk_and_harbor_guides():
    readme = ROOT_README.read_text(encoding="utf-8")

    assert "runtime execution layer" in readme
    assert "docs/sdk/python.mdx" in readme
    assert "integrations/harbor/README.md" in readme
    assert "integrations/harbor/demo/README.md" in readme
