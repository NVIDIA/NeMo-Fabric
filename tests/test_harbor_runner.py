# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEMO_README = ROOT / "integrations" / "harbor" / "demo" / "README.md"
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
            "request": {"context": {"model_name": "openai/gpt-5-codex"}},
        }
    )

    assert config.models["default"] == {
        "provider": "demo",
        "model": "demo",
    }
    assert profiles[-1].models["default"] == {
        "provider": "openai",
        "model": "openai/gpt-5-codex",
    }
    assert [profile.name for profile in profiles] == ["codex", "harbor_model"]
    assert json.loads(json.dumps(config.to_mapping()))["metadata"]["name"] == "harbor-demo"


def test_codex_adapter_maps_fabric_request_to_cli(monkeypatch, tmp_path):
    adapter = load_codex_adapter()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    payload = {
        "effective_config": {
            "config_root": str(tmp_path),
            "config": {
                "harness": {"settings": {"sandbox": "workspace-write"}},
                "models": {
                    "default": {
                        "provider": "openai",
                        "model": "openai/gpt-5-codex",
                    }
                },
                "environment": {"workspace": str(tmp_path)},
            },
        },
        "request": {"input": "Fix the calculator."},
    }

    command, cwd = adapter.build_command(payload)

    assert command == [
        "codex",
        "exec",
        "--sandbox",
        "workspace-write",
        "--skip-git-repo-check",
        "--ephemeral",
        "--color",
        "never",
        "--model",
        "gpt-5-codex",
        "-",
    ]
    assert cwd == tmp_path


def test_harbor_demo_documents_explicit_cli_commands():
    demo = DEMO_README.read_text(encoding="utf-8")
    integration = INTEGRATION_README.read_text(encoding="utf-8")

    assert "run.sh" not in demo
    assert "demo/run.sh" not in integration
    assert demo.count("uv run --extra harbor harbor run") == 4
    for flag in ("--path", "--agent", "--ak", "--ae", "--model", "--job-name"):
        assert flag in demo


def test_harbor_sdk_package_documents_execution_boundary():
    from nemo_fabric.integrations.harbor import FabricAgent

    readme = SDK_INTEGRATION_README.read_text(encoding="utf-8")

    assert FabricAgent.name() == "fabric"
    assert "nemo_fabric.integrations.harbor:FabricAgent" in readme
    assert "nemo_fabric.integrations.harbor.runner" in readme
    assert "does not invoke the Fabric CLI" in readme
