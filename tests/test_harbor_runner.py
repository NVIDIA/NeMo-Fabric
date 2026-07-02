# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ROOT_README = ROOT / "README.md"
DEMO_ROOT = ROOT / "integrations" / "harbor" / "demo"
DEMO_README = DEMO_ROOT / "README.md"
DEMO_DOCKERFILE = DEMO_ROOT / "task" / "environment" / "Dockerfile"
DEMO_HOST_GATEWAY = DEMO_ROOT / "host-gateway.compose.yaml"
DEMO_SOLUTION = DEMO_ROOT / "task" / "solution" / "solve.sh"
CODEX_PROFILE = (
    DEMO_ROOT
    / "task"
    / "environment"
    / "fabric"
    / "profiles"
    / "codex.yaml"
)
TELEMETRY_PROFILE = (
    DEMO_ROOT
    / "task"
    / "environment"
    / "fabric"
    / "profiles"
    / "telemetry.yaml"
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


def test_runner_loads_typed_sources_and_applies_harbor_model(tmp_path):
    from nemo_fabric.integrations.harbor.runner import load_sources

    config_path = tmp_path / "agent.yaml"
    profile_path = tmp_path / "profiles" / "codex.yaml"
    profile_path.parent.mkdir()
    config_path.write_text(
        yaml.safe_dump(
            {
                "metadata": {"name": "harbor-demo"},
                "harness": {"adapter_id": "demo.fabric.smoke"},
                "runtime": {"mode": "oneshot"},
                "models": {"default": {"provider": "demo", "model": "demo"}},
            }
        ),
        encoding="utf-8",
    )
    profile_path.write_text(
        yaml.safe_dump(
            {
                "name": "codex",
                "harness": {"adapter_id": "nvidia.fabric.codex.cli"},
            }
        ),
        encoding="utf-8",
    )

    config, profiles = load_sources(
        {
            "config_path": str(config_path),
            "profile_paths": [str(profile_path)],
            "request": {"context": {"model_name": "openai/gpt-5.4"}},
        }
    )

    assert config.models["default"] == {
        "provider": "demo",
        "model": "demo",
    }
    assert profiles[-1].models["default"] == {
        "provider": "openai",
        "model": "openai/gpt-5.4",
    }
    assert [profile.name for profile in profiles] == ["codex", "harbor_model"]
    assert json.loads(json.dumps(config.to_mapping()))["metadata"]["name"] == "harbor-demo"


def test_runner_rejects_missing_config(tmp_path):
    from nemo_fabric.integrations.harbor.runner import load_sources

    with pytest.raises(FileNotFoundError):
        load_sources({"config_path": str(tmp_path / "missing.yaml")})


def test_runner_rejects_malformed_profile(tmp_path):
    from nemo_fabric.integrations.harbor.runner import load_sources

    config_path = tmp_path / "agent.yaml"
    profile_path = tmp_path / "profile.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "metadata": {"name": "harbor-demo"},
                "harness": {"adapter_id": "demo.fabric.smoke"},
                "runtime": {"mode": "oneshot"},
            }
        ),
        encoding="utf-8",
    )
    profile_path.write_text("harness: [", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_sources(
            {
                "config_path": str(config_path),
                "profile_paths": [str(profile_path)],
            }
        )


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
                "runtime": {"mode": "oneshot"},
            },
        },
        "runtime_context": {"environment": {"workspace": str(tmp_path)}},
        "request": {"input": "Fix the calculator."},
    }

    command = adapter.build_command(payload)

    assert command == [
        "codex",
        "exec",
        "--json",
        "--ephemeral",
        "--sandbox",
        "workspace-write",
        "--config",
        'model_reasoning_effort="high"',
        "--model",
        "gpt-5.4",
        "--skip-git-repo-check",
        "-",
    ]
    assert adapter.resolve_cwd(payload) == tmp_path


def test_codex_demo_uses_current_adapter_contract():
    profile = yaml.safe_load(CODEX_PROFILE.read_text(encoding="utf-8"))
    settings = profile["harness"]["settings"]

    assert profile["runtime"]["mode"] == "oneshot"
    assert settings["sandbox"] == "danger-full-access"
    assert settings["skip_git_repo_check"] is True
    assert settings["config_overrides"]["model_reasoning_effort"] == "high"
    dockerfile = DEMO_DOCKERFILE.read_text(encoding="utf-8")
    assert 'nemo-fabric[codex,harbor,hermes,relay]' in dockerfile
    assert "@openai/codex@0.142.4" in dockerfile


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
    profile = yaml.safe_load(TELEMETRY_PROFILE.read_text(encoding="utf-8"))
    host_gateway = yaml.safe_load(DEMO_HOST_GATEWAY.read_text(encoding="utf-8"))
    observability = profile["telemetry"]["config"]["components"][0]["config"]
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
    assert "does not invoke the Fabric CLI" in readme


def test_root_readme_documents_sdk_contract_and_harbor_example():
    readme = ROOT_README.read_text(encoding="utf-8")

    assert "runtime execution layer" in readme
    assert "docs/python-sdk-contract.md" in readme
    assert "## Harbor Integration" in readme
    assert "uv run --extra harbor harbor run" in readme
    assert "nemo_fabric.integrations.harbor:FabricAgent" in readme
    assert "integrations/harbor/demo/README.md" in readme
