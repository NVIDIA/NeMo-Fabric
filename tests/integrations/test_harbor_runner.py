# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
ROOT_README = ROOT / "README.md"
CALCULATOR_ROOT = ROOT / "examples" / "harbor" / "calculator"
CALCULATOR_README = CALCULATOR_ROOT / "README.md"
CALCULATOR_DOCKERFILE = CALCULATOR_ROOT / "task" / "environment" / "Dockerfile"
CALCULATOR_SOLUTION = CALCULATOR_ROOT / "task" / "solution" / "solve.sh"
CALCULATOR_FABRIC_ROOT = CALCULATOR_ROOT / "task" / "environment" / "fabric"
SWEBENCH_ROOT = ROOT / "examples" / "harbor" / "swebench"
SWEBENCH_README = SWEBENCH_ROOT / "README.md"
SWEBENCH_MCP_CONFIG = SWEBENCH_ROOT / "mcp" / "repo-inspector.mcp.json"
INTEGRATION_README = ROOT / "examples" / "harbor" / "README.md"
SDK_INTEGRATION_README = (
    ROOT / "python" / "src" / "nemo_fabric" / "integrations" / "harbor" / "README.md"
)
HARBOR_PACKAGE_INIT = SDK_INTEGRATION_README.parent / "__init__.py"

pytestmark = pytest.mark.usefixtures("requires_harbor")


def load_codex_adapter():
    from nemo_fabric_adapters.codex import adapter

    return adapter


def test_harbor_builder_constructs_complete_config_from_harbor_inputs(tmp_path):
    from nemo_fabric.integrations.harbor.fabric_agent import build_harbor_config
    from nemo_fabric.integrations.harbor.models import HarborMcpServer

    config = build_harbor_config(
        adapter_id="demo.fabric.smoke",
        workspace="/testbed",
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
    assert (
        json.loads(json.dumps(config.to_mapping()))["metadata"]["name"]
        == "harbor-smoke"
    )


def test_harbor_builder_leaves_optional_capabilities_unset():
    from nemo_fabric.integrations.harbor.fabric_agent import build_harbor_config

    config = build_harbor_config(
        adapter_id="demo.fabric.smoke",
        workspace="/testbed",
    )

    assert config.models == {}
    assert config.mcp is None
    assert config.skills is None


def test_harbor_transport_models_validate_mcp_targets():
    from nemo_fabric import RunRequest
    from nemo_fabric.integrations.harbor.fabric_agent import build_harbor_config
    from nemo_fabric.integrations.harbor.models import FabricRunPayload
    from nemo_fabric.integrations.harbor.models import HarborMcpServer
    from pydantic import ValidationError

    server = HarborMcpServer(
        name="github",
        transport="streamable-http",
        url="https://mcp.example.test",
    )
    payload = FabricRunPayload.model_validate_json(
        FabricRunPayload(
            config=build_harbor_config(
                adapter_id="demo.fabric.smoke", workspace="/testbed"
            ),
            config_base_dir="/workspace",
            request=RunRequest(input="fix it"),
        ).model_dump_json()
    )

    assert payload.request.input == "fix it"
    assert server.name == "github"
    payload_properties = FabricRunPayload.model_json_schema()["properties"]
    assert set(payload_properties) == {
        "config",
        "config_base_dir",
        "logs_dir",
        "request",
    }
    with pytest.raises(ValidationError, match="require url"):
        HarborMcpServer(name="missing", transport="sse")
    with pytest.raises(ValidationError, match="require command"):
        HarborMcpServer(name="missing", transport="stdio")
    with pytest.raises(ValidationError, match="Extra inputs"):
        FabricRunPayload.model_validate(
            {
                "config": {},
                "config_base_dir": "/workspace",
                "request": {"input": "fix it"},
                "profile_paths": [],
            }
        )
    with pytest.raises(ValidationError, match="Field required"):
        FabricRunPayload.model_validate(
            {
                "config_base_dir": "/workspace",
                "request": {"input": "fix it"},
            }
        )


def test_each_harbor_job_delegates_to_an_independent_fabric_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from nemo_fabric.integrations.harbor import runner
    from nemo_fabric.integrations.harbor.fabric_agent import build_harbor_config
    from nemo_fabric.integrations.harbor.models import FabricRunPayload

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
    payloads = [
        FabricRunPayload(
            config=build_harbor_config(
                adapter_id="demo.fabric.smoke", workspace="/testbed"
            ),
            config_base_dir=tmp_path,
            request={
                "input": f"job {job_id}",
                "context": {"job_id": job_id},
            },
        )
        for job_id in ("job-1", "job-2")
    ]

    async def run_payloads():
        return await asyncio.gather(*(runner.run(payload) for payload in payloads))

    results = asyncio.run(run_payloads())

    assert [result.runtime_id for result in results] == ["runtime-1", "runtime-2"]
    requests = [call["request"] for call in calls]
    assert [request["context"]["job_id"] for request in requests] == [  # type: ignore[index]
        "job-1",
        "job-2",
    ]


def test_codex_adapter_maps_fabric_request_to_sdk(tmp_path):
    adapter = load_codex_adapter()

    payload = {
        "effective_config": {
            "config_root": str(tmp_path),
            "config": {
                "harness": {
                    "settings": {
                        "sandbox": "workspace-write",
                        "reasoning_effort": "high",
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
        "model_binding": {
            "role": "default",
            "provider": "openai",
            "model_id": "gpt-5.4",
            "wire_protocol": "openai-responses",
            "endpoint_ref": {"kind": "provider_default"},
            "credential_ref": {"kind": "harness_managed"},
        },
        "runtime_context": {
            "runtime_id": "harbor-test",
            "environment": {"workspace": str(tmp_path)},
        },
        "request": {"input": "Fix the calculator."},
    }

    assert adapter.selected_model(payload) == "gpt-5.4"
    assert adapter.sandbox(payload) == adapter.Sandbox.workspace_write
    assert adapter._reasoning_effort(payload) == adapter.ReasoningEffort.high
    assert adapter.thread_config(payload, relay=None) == {}
    assert adapter.resolve_cwd(payload) == tmp_path


def test_claude_calculator_run_uses_current_adapter_contract():
    from nemo_fabric.integrations.harbor.fabric_agent import build_harbor_config

    config = build_harbor_config(
        adapter_id="nvidia.fabric.claude",
        workspace="/app",
        model_name="anthropic/claude-sonnet-4-5",
        harness_settings={"max_turns": 20, "timeout_seconds": 600},
    )
    settings = config.harness.settings

    assert config.harness.adapter_id == "nvidia.fabric.claude"
    assert settings["permission_mode"] == "bypassPermissions"
    assert settings["max_turns"] == 20
    assert config.models["default"].provider == "anthropic"
    dockerfile = CALCULATOR_DOCKERFILE.read_text(encoding="utf-8")
    assert "-e /opt/nemo-fabric/adapters/claude" in dockerfile
    assert "-e /opt/nemo-fabric/adapters/hermes" in dockerfile
    assert "nemo-fabric[harbor,hermes,relay,runtime]" in dockerfile
    assert "@openai/codex" not in dockerfile


def test_harbor_calculator_uses_agent_inputs_without_config_files():
    assert not list(CALCULATOR_FABRIC_ROOT.glob("*.py"))
    assert not list(CALCULATOR_FABRIC_ROOT.rglob("*.yaml"))


def test_harbor_smoke_config_resolves_its_local_adapter():
    from nemo_fabric import Fabric
    from nemo_fabric.integrations.harbor.fabric_agent import build_harbor_config

    config = build_harbor_config(
        adapter_id="demo.fabric.scripted",
        workspace="/app",
    )
    plan = Fabric().plan(config, base_dir=CALCULATOR_FABRIC_ROOT)

    assert plan.adapter.adapter_id == "demo.fabric.scripted"
    assert plan["adapter_descriptor"]["source"] == "local"
    assert plan["adapter_descriptor"]["root"].endswith("adapters/scripted")


def test_harbor_calculator_documents_explicit_cli_commands():
    calculator = CALCULATOR_README.read_text(encoding="utf-8")
    landing = INTEGRATION_README.read_text(encoding="utf-8")
    swebench = SWEBENCH_README.read_text(encoding="utf-8")

    assert "run.sh" not in calculator
    assert calculator.count("uv run --extra runtime --extra harbor harbor run") == 4
    assert landing.count("uv run --extra runtime --extra harbor harbor run") == 0
    assert swebench.count("uv run --extra runtime --extra harbor harbor run") == 5
    assert "--agent-import-path" not in landing + calculator + swebench
    assert "fabric_config_path" not in calculator
    assert "fabric_config_path" not in landing
    assert "fabric_config_path" not in swebench
    assert "fabric_config_factory" not in calculator
    assert "fabric_config_factory" not in landing
    assert "fabric_config_factory" not in swebench
    assert "fabric_workspace=/app" in calculator
    assert "--model nvidia/nemotron-3-nano-30b-a3b" in calculator
    assert "--model anthropic/claude-sonnet-4-5" in calculator
    assert 'CALCULATOR_DIR="$PWD/examples/harbor/calculator"' in calculator
    assert "calculator/README.md" in landing
    assert "swebench/README.md" in landing
    assert "fabric_adapter_id" in landing
    assert 'export TMPDIR="$HOME/harbor-tmp"' in landing
    assert "./examples/harbor/prepare_swebench.sh" in swebench
    assert 'FABRIC_PACKAGE="$(< "$FABRIC_BUNDLE/.fabric-package")"' in swebench
    assert '--ae "PIP_FIND_LINKS=$FABRIC_FIND_LINKS"' in swebench
    assert "--dataset swe-bench/swe-bench-verified" in swebench
    for flag in (
        "--path",
        "--agent",
        "--ak",
        "--job-name",
    ):
        assert flag in calculator
    for value in (
        "swe-bench/swe-bench-verified",
        "django__django-13741",
        "--task swe-bench/django__django-13741",
        "--model",
        "--skill",
        "--mcp-config",
        "fabric_blocked_tools",
        "fabric_telemetry=relay",
        "harbor job resume",
        "telemetry-validation.json",
        "agent/trajectory.json",
    ):
        assert value in swebench


def test_harbor_calculator_setup_and_solution_fail_fast():
    dockerfile = CALCULATOR_DOCKERFILE.read_text(encoding="utf-8")
    solution = CALCULATOR_SOLUTION.read_text(encoding="utf-8")

    assert (
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |" not in dockerfile
    )
    assert "-o /tmp/rustup-init.sh" in dockerfile
    assert "if updated == source:" in solution
    assert "raise SystemExit" in solution


def test_harbor_relay_telemetry_exports_direct_atof_and_atif():
    from nemo_fabric.integrations.harbor.fabric_agent import build_harbor_config

    config = build_harbor_config(
        adapter_id="nvidia.fabric.hermes",
        workspace="/app",
        model_name="nvidia/nemotron-3-nano-30b-a3b",
        telemetry="relay",
    ).to_mapping()
    assert "relay" in config["telemetry"]["providers"]
    observability = config["relay"]["observability"]
    swebench = SWEBENCH_README.read_text(encoding="utf-8")

    assert "openinference" not in observability
    assert observability["atof"]["enabled"] is True
    assert observability["atif"]["enabled"] is True
    assert not (CALCULATOR_ROOT / "host-gateway.compose.yaml").exists()
    assert "direct Relay ATOF and ATIF" in swebench
    assert "telemetry-validation.json" in swebench
    assert "canonical ATIF" in swebench


def test_harbor_sdk_package_points_to_the_public_example():
    from nemo_fabric.integrations.harbor import FabricAgent

    readme = SDK_INTEGRATION_README.read_text(encoding="utf-8")
    package_init = HARBOR_PACKAGE_INIT.read_text(encoding="utf-8")

    assert FabricAgent.name() == "fabric"
    assert FabricAgent.__module__ == "nemo_fabric.integrations.harbor.fabric_agent"
    assert "nemo_fabric.integrations.harbor:FabricAgent" in readme
    assert "examples/harbor/README.md" in readme
    assert "FabricConfig" in readme
    assert "Internals" not in readme
    assert "class FabricAgent" not in package_init
    assert "fabric_agent import FabricAgent" in package_init


def test_swebench_matrix_translates_harbor_inputs_to_typed_config(tmp_path: Path):
    from harbor.cli.utils import load_mcp_servers
    from nemo_fabric.integrations.harbor.fabric_agent import build_harbor_config
    from nemo_fabric.integrations.harbor.models import HarborMcpServer

    base = build_harbor_config(
        adapter_id="nvidia.fabric.hermes",
        workspace="/testbed",
    )
    relay = build_harbor_config(
        adapter_id="nvidia.fabric.hermes",
        workspace="/testbed",
        telemetry="relay",
        model_name="nvidia/nemotron-3-nano-30b-a3b",
        skills_dir="/harbor/skills",
        mcp_servers=tuple(
            HarborMcpServer.model_validate(server.model_dump(mode="python"))
            for server in load_mcp_servers(SWEBENCH_MCP_CONFIG)
        ),
    )
    tools = build_harbor_config(
        adapter_id="nvidia.fabric.hermes",
        workspace="/testbed",
        blocked_tools=["browser"],
        telemetry="relay",
    )
    claude = build_harbor_config(
        adapter_id="nvidia.fabric.claude",
        workspace="/testbed",
        harness_settings={
            "nemo_relay_command": "/tmp/nemo-fabric-config/.relay/bin/nemo-relay"
        },
    )

    assert base.profiles is None
    assert base.environment is not None
    assert str(base.environment.workspace) == "/testbed"
    assert base.models == {}
    assert base.skills is None
    assert base.mcp is None
    assert base.tools is None
    assert base.telemetry is None
    assert relay.models["default"].model == "nvidia/nemotron-3-nano-30b-a3b"
    assert relay.skills is not None
    assert relay.skills.paths == ["/harbor/skills"]
    assert relay.mcp is not None
    assert set(relay.mcp.servers) == {"fabric-repo-inspector"}
    assert relay.mcp.servers["fabric-repo-inspector"].extra_fields["args"] == [
        "/tmp/nemo-fabric-config/mcp/repo_inspector.py"
    ]
    assert tools.tools is not None
    assert tools.tools.blocked == ["browser"]
    assert "enabled_toolsets" not in tools.harness.settings
    assert relay.telemetry is not None
    assert "relay" in relay.telemetry.providers
    assert relay.relay is not None
    assert relay.relay.observability.atif.enabled is True
    assert relay.relay.observability.atof.enabled is True
    assert claude.harness.settings["permission_mode"] == "bypassPermissions"
    assert claude.harness.settings["env"] == {"IS_SANDBOX": "1"}
    assert claude.harness.settings["nemo_relay_command"] == (
        "/tmp/nemo-fabric-config/.relay/bin/nemo-relay"
    )
    assert not list(SWEBENCH_ROOT.rglob("*.yaml"))
    assert not (SWEBENCH_ROOT / "harbor_swebench_config.py").exists()

    # TODO: Remove the bundled copies and these equality checks after Fabric
    # discovers adapter descriptors directly from source checkouts and wheels.
    assert (SWEBENCH_ROOT / "adapters/hermes/fabric-adapter.json").read_text() == (
        ROOT / "adapters/hermes/fabric-adapter.json"
    ).read_text()
    assert (SWEBENCH_ROOT / "adapters/claude/fabric-adapter.json").read_text() == (
        ROOT / "adapters/claude/fabric-adapter.json"
    ).read_text()
    hermes_descriptor = json.loads(
        (SWEBENCH_ROOT / "adapters/hermes/fabric-adapter.json").read_text()
    )
    assert "models" in hermes_descriptor["config"]["accepts"]

    readme = SWEBENCH_README.read_text(encoding="utf-8")
    assert readme.count("django__django-13741") >= 4
    assert "--n-tasks 5" in readme


def test_harbor_lifecycle_populates_context_after_run(tmp_path: Path):
    from harbor.models.agent.context import AgentContext
    from harbor.trial.trial import Trial
    from nemo_fabric.integrations.harbor import FabricAgent

    result = {
        "agent_name": "harbor-hermes",
        "profiles": [],
        "harness": "hermes",
        "adapter_kind": "python",
        "adapter_id": "nvidia.fabric.hermes",
        "status": "succeeded",
        "runtime_id": "runtime-1",
        "invocation_id": "invocation-1",
        "request_id": "request-1",
        "output": {"response": "done"},
        "error": None,
        "artifacts": {"root": "/logs/agent", "artifacts": []},
        "telemetry": [],
        "events": [],
        "metadata": {},
    }

    class FakeEnvironment:
        async def upload_file(self, _source, _destination):
            return None

        async def exec(self, *_args, **_kwargs):
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        async def download_file(self, _source, destination):
            destination.write_text(json.dumps(result), encoding="utf-8")

    agent = FabricAgent(
        logs_dir=tmp_path,
        fabric_adapter_id="nvidia.fabric.hermes",
    )
    context = AgentContext()

    asyncio.run(agent.run("fix it", FakeEnvironment(), context))

    assert context.is_empty()
    (tmp_path / "trajectory.json").write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "session_id": "runtime-1",
                "agent": {"name": "fabric", "version": "0.1.0"},
                "steps": [{"step_id": 1, "source": "agent", "message": "done"}],
                "final_metrics": {
                    "total_prompt_tokens": 12,
                    "total_cached_tokens": 3,
                    "total_completion_tokens": 4,
                    "total_cost_usd": 0.25,
                },
            }
        ),
        encoding="utf-8",
    )

    Trial._populate_agent_context(SimpleNamespace(agent=agent), context)

    assert context.metadata["fabric"]["status"] == "succeeded"
    assert context.n_input_tokens == 12
    assert context.n_cache_tokens == 3
    assert context.n_output_tokens == 4
    assert context.cost_usd == 0.25


def test_harbor_018_factory_loads_fabric_agent(tmp_path: Path):
    from harbor.agents.factory import AgentFactory

    agent = AgentFactory.create_agent_from_import_path(
        "nemo_fabric.integrations.harbor:FabricAgent",
        logs_dir=tmp_path,
        fabric_adapter_id="nvidia.fabric.hermes",
    )

    assert agent.name() == "fabric"
    assert agent.SUPPORTS_ATIF is True


def test_swebench_mcp_config_uses_the_bundled_repo_inspector():
    from harbor.cli.utils import load_mcp_servers

    [server] = load_mcp_servers(SWEBENCH_MCP_CONFIG)

    assert server.transport == "stdio"
    assert server.command == "python3"
    assert server.args == ["/tmp/nemo-fabric-config/mcp/repo_inspector.py"]


def test_root_readme_routes_to_sdk_and_harbor_guides():
    readme = ROOT_README.read_text(encoding="utf-8")

    assert "runtime execution layer" in readme
    assert "docs/sdk/python.mdx" in readme
    assert "examples/harbor/README.md" in readme
