# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real local SDK/process paths with explicit, controlled external boundaries.

Harbor transport uses a separate guest process and local files. The embedding
server follows Gym's SDK seam without claiming a live Gym deployment. HTTP
agent responses are supplied by the existing model-free API fixture.
"""

import asyncio
import copy
import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from _utils.provenance_capture import capture_result, version_view
from nemo_fabric import Fabric, FabricConfig, RunResult
from nemo_fabric_adapter_contract.evidence import ExecutionEvidence
from nemo_fabric_adapter_contract.metadata import observed_at

ROOT = Path(__file__).parents[2]


@pytest.fixture(name="provenance_config")
def provenance_config_fixture(hermes_shim_agent_dir, tmp_path):
    os.environ["ADAPTER_PYTHON"] = sys.executable
    return FabricConfig.from_mapping(
        {
            "metadata": {"name": "provenance-path"},
            "harness": {
                "adapter_id": "test.fabric.hermes_shim",
                "resolution": "preinstalled",
            },
            "discovery": {"local_paths": ["adapters"]},
            "runtime": {"artifacts": str(tmp_path / "artifacts")},
        }
    )


@pytest.fixture(name="provenance_wheels")
def provenance_wheels_fixture(tmp_path):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for the separate-venv installation test")
    pytest.importorskip("setuptools", minversion="77")
    directory = tmp_path / "wheels"
    directory.mkdir()
    for project in ("adapter-contract/python", "adapters/python/common"):
        built = subprocess.run(
            [
                uv,
                "build",
                "--wheel",
                "--no-build-isolation",
                "--offline",
                "--python",
                sys.executable,
                "--out-dir",
                str(directory),
                str(ROOT / project),
            ],
            capture_output=True,
            timeout=60,
        )
        assert built.returncode == 0, (built.stdout + built.stderr).decode()
    return uv, sorted(directory.glob("*.whl"))


async def test_direct_and_separate_venv_contexts_are_not_joined_by_equal_versions(
    provenance_config, hermes_shim_agent_dir, tmp_path, provenance_wheels
):
    uv, wheels = provenance_wheels
    guest = tmp_path / "adapter-venv"
    venv.EnvBuilder(with_pip=False).create(guest)
    python = guest / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    process = await asyncio.create_subprocess_exec(
        uv,
        "pip",
        "install",
        "--python",
        str(python),
        "--no-deps",
        "--no-index",
        *(str(wheel) for wheel in wheels),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == 0, (stdout + stderr).decode()
    os.environ["ADAPTER_PYTHON"] = str(python)
    result = await Fabric().run(
        provenance_config, base_dir=hermes_shim_agent_dir, input="separate installation"
    )
    assert result.status == "succeeded"
    contexts = result.execution.software_environments
    caller = next(env for env in contexts if "caller_sdk" in env.participants)
    adapter = next(env for env in contexts if "adapter" in env.participants)
    assert caller.context_ref != adapter.context_ref
    interpreter = lambda env: next(
        entry.version.value for entry in env.entries if entry.role == "interpreter"
    )
    assert interpreter(caller) == interpreter(adapter)
    contract = next(entry for entry in adapter.entries if entry.role == "contract")
    assert contract.version.status == "available"
    capture_result("separate-venv", result)


@pytest.mark.usefixtures("requires_harbor")
async def test_external_platform_payload_forwards_host_snapshot_to_real_guest_process(
    hermes_shim_agent_dir, tmp_path
):
    from nemo_fabric.integrations.harbor.fabric_agent import FabricAgent

    os.environ["ADAPTER_PYTHON"] = sys.executable
    descriptor = next(hermes_shim_agent_dir.glob("adapters/*/*.fabric-adapter.json"))
    agent = FabricAgent(
        logs_dir=tmp_path / "host-logs",
        fabric_adapter_id="test.fabric.hermes_shim",
        fabric_adapter_descriptor=descriptor,
        fabric_config_base_dir="/fixture-agent",
        fabric_discovery_paths=["adapters"],
        fabric_python=sys.executable,
    )
    payload = agent._build_spec("platform dispatch")
    # Map guest paths to the local process used by this controlled transport.
    payload.config_base_dir = hermes_shim_agent_dir
    payload.config.environment.workspace = str(tmp_path)
    payload.config.runtime.artifacts = tmp_path / "guest-artifacts"
    payload.logs_dir = str(tmp_path / "guest-logs")
    forwarded = payload.config.provenance.reports.to_mapping()
    spec, output = tmp_path / "dispatch.json", tmp_path / "result.json"
    spec.write_text(payload.model_dump_json())
    command = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "nemo_fabric.integrations.harbor.runner",
        "--spec",
        str(spec),
        "--result",
        str(output),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(command.communicate(), timeout=60)
    finally:
        if command.returncode is None:
            command.kill()
            await command.wait()
    assert command.returncode == 0, (stdout + stderr).decode()
    result = RunResult.from_mapping(json.loads(output.read_text()))
    assert result.status == "succeeded"
    actual = result.execution.to_mapping()["software_environments"]
    assert (
        actual[: len(forwarded["software_environments"])]
        == forwarded["software_environments"]
    )
    assert any(
        "guest_sdk" in env.participants
        for env in result.execution.software_environments
    )
    assert any(
        "application" in env.participants
        for env in result.execution.software_environments
    )
    assert (
        agent._build_spec("second dispatch").config.provenance.reports.to_mapping()
        == forwarded
    )
    capture_result("platform-dispatch", result)


async def test_agent_server_embedding_reports_itself_without_inventing_an_outer_evaluator(
    provenance_config, hermes_shim_agent_dir
):
    application = ExecutionEvidence.from_mapping(
        {
            "software_environments": [
                {
                    "context_ref": "env-0",
                    "participants": ["application"],
                    "status": "available",
                    "complete": True,
                    "entries": [
                        {
                            "role": "integration",
                            "kind": "component",
                            "name": "agent-server-fixture",
                            "version": {
                                "status": "available",
                                "value": "0.1.dev0+embedding",
                                "source": "application:explicit_startup_report",
                                "origin": "reported",
                                "observed_at": observed_at(),
                            },
                        }
                    ],
                }
            ]
        }
    )
    provenance_config.provenance.reports = application
    fabric, app = Fabric(), FastAPI()

    @app.post("/responses")
    async def respond():
        return (
            await fabric.run(
                provenance_config, base_dir=hermes_shim_agent_dir, input="embedded SDK"
            )
        ).to_mapping()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://agent-server.test"
    ) as client:
        response = await client.post("/responses")
    response.raise_for_status()
    result = RunResult.from_mapping(response.json())
    assert result.status == "succeeded"
    participants = {
        p for env in result.execution.software_environments for p in env.participants
    }
    assert "application" in participants
    assert not ({"evaluator", "remote_evaluator"} & participants)
    capture_result("agent-server-embedding", result)


async def test_independent_remote_agent_reports_only_the_local_client(
    api_server, tmp_path
):
    os.environ["ADAPTER_PYTHON"] = sys.executable
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "remote-client-provenance"},
            "harness": {
                "adapter_id": "nvidia.fabric.remote-agent",
                "settings": {
                    "base_url": f"{api_server}/v1",
                    "api_type": "openai-completions",
                },
            },
            "discovery": {
                "local_paths": [
                    str(
                        ROOT
                        / "adapters/python/remote-agent/remote-agent.fabric-adapter.json"
                    )
                ]
            },
            "models": {"default": {"provider": "test", "model": "fabric-echo"}},
            "runtime": {"artifacts": str(tmp_path / "artifacts")},
        }
    )
    result = await Fabric().run(config, input="Hello.", base_dir=ROOT)
    assert result.status == "succeeded"
    participants = {
        p for env in result.execution.software_environments for p in env.participants
    }
    assert participants == {"caller_sdk", "core", "adapter"}
    assert result.execution.infrastructure is None
    assert any(
        entry.name == "httpx"
        for env in result.execution.software_environments
        for entry in env.entries
    )
    capture_result("remote-agent-client", result)


def test_comparison_projection_preserves_unknowns_and_names_one_changed_component():
    execution = json.loads((ROOT / "tests/fixtures/provenance/report.json").read_text())
    first = {
        "runtime_id": "one",
        "artifacts": {"root": "/tmp/one"},
        "execution": execution,
    }
    second = copy.deepcopy(first)
    second.update(runtime_id="two", artifacts={"root": "/tmp/two"})
    second["execution"]["software_environments"][0]["context_ref"] = "unrelated-ref"
    for entry in second["execution"]["software_environments"][0]["entries"]:
        entry["version"]["observed_at"] = "2026-10-28T12:00:00Z"
    assert version_view(first) == version_view(second)
    assert version_view(first)["unknowns"]
    second["execution"]["software_environments"][0]["entries"][0]["version"][
        "value"
    ] = "changed-component-version+full"
    before, after = version_view(first), version_view(second)
    changed = set(before["versions"]) ^ set(after["versions"])
    assert len(changed) == 2
    assert {record[2] for record in changed} == {"nemo-fabric-adapters-example"}
    assert before["unknowns"] == after["unknowns"]
    capture_result(
        "comparison-fixture",
        {
            "before": before,
            "after": after,
            "claim": "Compare observed versions; unknowns remain unknown.",
        },
    )


def test_comparison_includes_existing_infrastructure_versions_and_context_topology():
    execution = json.loads((ROOT / "tests/fixtures/provenance/report.json").read_text())
    observation = execution["software_environments"][0]["entries"][0]["version"]
    execution["infrastructure"] = {
        "sandbox": {
            "context_refs": ["env-0"],
            "provider": {**observation, "value": "openshell"},
            "service_version": {**observation, "value": "1.0-rc.1+server"},
        }
    }
    first = {"execution": execution}
    second = copy.deepcopy(first)
    second["execution"]["software_environments"][0]["context_ref"] = "renumbered"
    report = second["execution"]["infrastructure"]["sandbox"]
    report["context_refs"] = ["renumbered"]
    for field in ("provider", "service_version"):
        report[field]["observed_at"] = "2026-10-28T12:00:00Z"
    assert version_view(first) == version_view(second)
    report["service_version"]["value"] = "1.0-rc.2+server"
    before, after = version_view(first), version_view(second)
    changed = set(before["versions"]) ^ set(after["versions"])
    assert len(changed) == 2
    assert {record[2] for record in changed} == {"service_version"}
    assert before["unknowns"] == after["unknowns"]
    capture_result(
        "infrastructure-comparison-fixture", {"before": before, "after": after}
    )


def test_receiving_platform_boundary_restores_host_when_an_older_intermediary_drops_evidence():
    from nemo_fabric.provenance import _restore_dispatch_evidence

    report = ExecutionEvidence.from_mapping(
        json.loads((ROOT / "tests/fixtures/provenance/report.json").read_text())
    )
    report.software_environments[0].participants = ["application", "caller_sdk"]
    original = {
        "status": "failed",
        "output": {"partial": True},
        "error": {"code": "original"},
        "usage": {"input_tokens": 7},
    }
    restored = _restore_dispatch_evidence(copy.deepcopy(original), report, enabled=True)
    assert {key: restored[key] for key in original} == original
    assert (
        restored["execution"]["software_environments"][0]
        == report.to_mapping()["software_environments"][0]
    )
    assert not restored["execution"]["software_environments"][-1]["complete"]
    capture_result("legacy-platform-receive-fixture", restored)


@pytest.mark.parametrize("case", ["disabled", "retained", "conflicting_infrastructure"])
def test_dispatch_restoration_controls_and_infrastructure(case):
    from nemo_fabric.provenance import _restore_dispatch_evidence
    from nemo_fabric_adapter_contract.metadata import combine

    owner = json.loads((ROOT / "tests/fixtures/provenance/report.json").read_text())
    owner["software_environments"][0]["participants"] = ["application", "caller_sdk"]
    observation = owner["software_environments"][0]["entries"][0]["version"]
    owner["infrastructure"] = {
        "container": {
            "context_refs": ["env-0"],
            "engine": {**observation, "value": "fixture-engine"},
            "engine_version": {**observation, "value": "1.0+owner"},
        }
    }
    forwarded = ExecutionEvidence.from_mapping(owner)
    guest = combine(
        forwarded, ExecutionEvidence.gap("guest_sdk", "fixture")
    ).to_mapping()
    if case == "conflicting_infrastructure":
        guest["infrastructure"]["container"]["engine_version"]["value"] = "2.0+guest"
    original = {
        "status": "failed",
        "output": {"partial": True},
        "error": {"code": "original"},
    }
    restored = _restore_dispatch_evidence(
        {**copy.deepcopy(original), "execution": copy.deepcopy(guest)},
        forwarded,
        enabled=case != "disabled",
    )
    assert {key: restored[key] for key in original} == original
    if case == "disabled":
        assert "execution" not in restored
    elif case == "retained":
        assert restored["execution"] == guest
        assert len(restored["execution"]["software_environments"]) == 2
    else:
        assert (
            restored["execution"]["software_environments"][0]["reason"]
            == "conflicting_infrastructure_reports"
        )


async def test_explicit_engine_owner_report_is_forwarded_without_a_probe(
    provenance_config, hermes_shim_agent_dir
):
    report = ExecutionEvidence.from_mapping(
        {
            "software_environments": [
                {
                    "context_ref": "owner-context",
                    "participants": ["application"],
                    "status": "available",
                    "complete": True,
                    "entries": [],
                }
            ],
            "infrastructure": {
                "container": {
                    "context_refs": ["owner-context"],
                    "engine": {
                        "status": "available",
                        "value": "docker",
                        "source": "environment_owner:existing_report",
                        "origin": "reported",
                        "observed_at": "2026-10-26T12:00:00Z",
                    },
                    "engine_version": {
                        "status": "available",
                        "value": "29.0.0-rc.1+server-build",
                        "source": "environment_owner:existing_report",
                        "origin": "reported",
                        "observed_at": "2026-10-26T12:00:00Z",
                    },
                }
            },
        }
    )
    provenance_config.provenance.reports = report
    result = await Fabric().run(
        provenance_config, base_dir=hermes_shim_agent_dir, input="owner report"
    )
    assert result.status == "succeeded"
    container = result.execution.infrastructure["container"]
    assert (
        container["engine_version"]
        == report.to_mapping()["infrastructure"]["container"]["engine_version"]
    )
    assert container["context_refs"] == [
        result.execution.software_environments[0].context_ref
    ]
    capture_result("explicit-infrastructure-owner", result)


@pytest.mark.usefixtures("requires_harbor")
def test_harbor_disable_control_prevents_host_collection(monkeypatch, tmp_path):
    import nemo_fabric.integrations.harbor.fabric_agent as module

    def unexpected():
        raise AssertionError("host provenance collector ran while disabled")

    monkeypatch.setattr(module, "_harbor_execution_report", unexpected)
    agent = module.FabricAgent(
        logs_dir=tmp_path,
        fabric_adapter_id="nvidia.fabric.hermes",
        fabric_provenance={"enabled": False},
    )
    spec = agent._build_spec("disabled metadata")
    assert spec.config.provenance.enabled is False
    assert spec.config.provenance.reports is None
