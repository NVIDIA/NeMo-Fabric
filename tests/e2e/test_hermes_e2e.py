# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import sys
import warnings
from importlib.metadata import version as distribution_version
from pathlib import Path
from types import ModuleType

from packaging.version import Version
import pytest
import requests
import yaml
from _utils.utils import assert_atof_model, assert_atof_skill_selection

from examples.code_review_agent import (
    hermes_config,
    with_relay,
)
from nemo_fabric import (
    Fabric,
    RelayAtofConfig,
    RelayAtofFileSinkConfig,
    RelayObservabilityConfig,
)

pytestmark = pytest.mark.usefixtures("requires_hermes_agent")


@pytest.mark.usefixtures("mock_nvidia_api_key", "nemo_relay")
async def test_hermes_persistent_host_reuses_native_session(
    code_review_agent_dir: Path,
    api_server: str,
):
    os.environ["ADAPTER_PYTHON"] = sys.executable
    config = with_relay(hermes_config())
    config.models["default"].base_url = f"{api_server}/v1"

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        async with await Fabric().start_runtime(
            config,
            base_dir=code_review_agent_dir,
            streaming=True,
        ) as runtime:
            first_stream = runtime.invoke_stream(input="first")
            first_records = [record async for record in first_stream]
            first = await first_stream.result()
            second_stream = runtime.invoke_stream(input="second")
            second_records = [record async for record in second_stream]
            second = await second_stream.result()

    results = (first.to_mapping(), second.to_mapping())
    assert first_records
    assert second_records
    assert first["status"] == second["status"] == "succeeded", results
    assert first["metadata"]["adapter_runner"] == "persistent_local_host", results
    assert first["metadata"]["host_pid"] == second["metadata"]["host_pid"], results
    assert "user_count=2" in second["output"]["response"], results
    for turn in (first, second):
        assert {artifact["kind"] for artifact in turn["output"]["relay_artifacts"]} >= {
            "atof",
            "atif",
        }, turn.to_mapping()


@pytest.mark.usefixtures("mock_nvidia_api_key", "nemo_relay")
async def test_hermes_persistent_host_with_relay(
    code_review_agent_dir: Path,
    api_server: str,
):
    os.environ["ADAPTER_PYTHON"] = sys.executable
    config = with_relay(hermes_config())
    config.models["default"].base_url = f"{api_server}/v1"

    async with await Fabric().start_runtime(
        config, base_dir=code_review_agent_dir
    ) as runtime:
        first = await runtime.invoke(input="first")
        second = await runtime.invoke(input="second")

    results = (first.to_mapping(), second.to_mapping())
    assert first["status"] == second["status"] == "succeeded", results
    assert first["metadata"]["host_pid"] == second["metadata"]["host_pid"], results
    assert "user_count=2" in second["output"]["response"], results
    for turn in (first, second):
        assert turn.telemetry[0].provider == "relay", turn.to_mapping()
        assert {artifact["kind"] for artifact in turn["output"]["relay_artifacts"]} >= {
            "atof",
            "atif",
        }, turn.to_mapping()

    atof_path = next(
        Path(artifact["path"])
        for artifact in second["output"]["relay_artifacts"]
        if artifact["kind"] == "atof"
    )
    atof_records = [
        json.loads(line) for line in atof_path.read_text(encoding="utf-8").splitlines()
    ]
    assert sum(record["name"] == "hermes.session.end" for record in atof_records) == 2


@pytest.mark.usefixtures("mock_nvidia_api_key", "nemo_relay")
@pytest.mark.parametrize("enabled", [True, False])
async def test_mcp_stdio_transport(
    code_review_agent_dir: Path,
    api_server: str,
    enabled: bool,
):
    os.environ["ADAPTER_PYTHON"] = sys.executable
    tool_name = "mcp__mcp_server_time__get_current_time"
    tool_arguments = {"timezone": "America/Los_Angeles"}
    if Version(distribution_version("hermes-agent")) < Version("0.20"):
        # The released 0.19 integration accepts the configured MCP tool directly.
        tool_call = {"name": tool_name, "arguments": tool_arguments}
    else:
        # Newer Hermes versions dispatch MCP schemas through their native bridge.
        tool_call = {
            "name": "tool_call",
            "arguments": {"name": tool_name, "arguments": tool_arguments},
        }
    scenario_response = requests.post(
        f"{api_server}/_scenario",
        json={"tool_call": tool_call},
        timeout=5,
    )
    scenario_response.raise_for_status()

    config = hermes_config()
    config.models["default"].base_url = f"{api_server}/v1"
    config.tools.enabled = None
    config.add_mcp_server(
        "mcp_server_time",
        transport="stdio",
        url=sys.executable,
        args=["-m", "mcp_server_time"],
        env={"MCP_TIME_TEST": "enabled"},
    )

    if not enabled:
        config.remove_mcp_server("mcp_server_time")

    config.enable_relay(
        output_dir="./artifacts/relay-mcp",
        observability=RelayObservabilityConfig(
            atof=RelayAtofConfig(
                enabled=True,
                sinks=[
                    RelayAtofFileSinkConfig(
                        output_directory="./artifacts/relay-mcp",
                        filename="events.atof.jsonl",
                        mode="overwrite",
                    )
                ],
            ),
        ),
    )

    result = await Fabric().run(
        config,
        base_dir=code_review_agent_dir,
        input="Use the time MCP tool to get the current time in America/Los_Angeles.",
    )

    assert result["status"] == "succeeded", result.to_mapping()
    atof_path = next(
        Path(artifact["path"])
        for artifact in result["output"]["relay_artifacts"]
        if artifact["kind"] == "atof"
    )
    atof_records = [
        json.loads(line) for line in atof_path.read_text(encoding="utf-8").splitlines()
    ]
    tool_records = [record for record in atof_records if record["name"] == tool_name]
    hermes_config_path = Path(result["output"]["hermes_config_path"])
    generated_config = yaml.safe_load(hermes_config_path.read_text(encoding="utf-8"))

    if not enabled:
        assert not tool_records
        assert "mcp_servers" not in generated_config
    else:
        assert generated_config["mcp_servers"]["mcp_server_time"] == {
            "enabled": True,
            "command": sys.executable,
            "args": ["-m", "mcp_server_time"],
            "env": {"MCP_TIME_TEST": "enabled"},
        }
        assert "mcp_server_time" in result["output"]["enabled_toolsets"]
        assert {record["scope_category"] for record in tool_records} == {"start", "end"}
        tool_end = next(
            record for record in tool_records if record["scope_category"] == "end"
        )
        assert tool_end["category"] == "tool"
        assert tool_end["metadata"].get("otel.status_code") == "OK" or (
            tool_end["metadata"].get("status") == "ok"
        )
        assert "America/Los_Angeles" in tool_end["data"]


@pytest.mark.usefixtures("mock_nvidia_api_key", "nemo_relay")
@pytest.mark.parametrize("skill", ["default", "alternate", None])
async def test_skill_selection(
    code_review_agent_dir: Path,
    api_server: str,
    skill: str | None,
    default_skill: Path,
    alternate_skill: Path,
):
    os.environ["ADAPTER_PYTHON"] = sys.executable
    config = with_relay(hermes_config())
    config.models["default"].model = "fabric-echo"
    config.models["default"].base_url = f"{api_server}/v1"
    config.tools.enabled = None
    config.add_skill_path(default_skill)

    if skill == "alternate" or skill is None:
        config.remove_skill_path(default_skill)

    if skill == "alternate":
        config.add_skill_path(alternate_skill)

    if skill is not None:
        scenario_response = requests.post(
            f"{api_server}/_scenario",
            json={
                "tool_call": {
                    "name": "skill_view",
                    "arguments": {"name": skill},
                }
            },
            timeout=5,
        )
        scenario_response.raise_for_status()

    result = await Fabric().run(
        config,
        base_dir=code_review_agent_dir,
        input=f"Use the {skill} skill." if skill else "Reply without using a skill.",
    )

    assert result["status"] == "succeeded", result.to_mapping()
    assert_atof_skill_selection(result["output"], skill)


@pytest.mark.usefixtures("mock_nvidia_api_key", "nemo_relay")
@pytest.mark.parametrize("model", ["m1", "m2"])
async def test_model_selection(
    code_review_agent_dir: Path,
    api_server: str,
    model: str,
):
    os.environ["ADAPTER_PYTHON"] = sys.executable
    config = with_relay(hermes_config())
    config.models["default"].model = model
    config.models["default"].base_url = f"{api_server}/v1"

    result = await Fabric().run(
        config,
        base_dir=code_review_agent_dir,
        input="Reply with hello.",
    )

    assert result["status"] == "succeeded", result.to_mapping()
    assert_atof_model(result["output"], model)


class TestHermesE2E:
    """End-to-end Hermes relay assertions."""

    config_builder = staticmethod(hermes_config)
    adapter_kind = "python"
    adapter_runner = "persistent_local_host"
    output_adapter = "python"
    mode = "hermes"
    artifact_dir = "hermes"
    atof_platform = "fabric"

    @pytest.fixture(autouse=True)
    async def run_hermes_with_relay(
        self,
        nemo_relay: ModuleType,
        mock_nvidia_api_key: str,
        code_review_agent_dir: Path,
        api_server: str,
    ):
        os.environ["ADAPTER_PYTHON"] = sys.executable

        self.code_review_agent_dir = code_review_agent_dir
        self.api_server = api_server
        config = self.config_builder()
        config.models["default"].base_url = f"{api_server}/v1"
        config = with_relay(config)

        self.result = await Fabric().run(
            config,
            base_dir=code_review_agent_dir,
            input="Reply with exactly: relay ok",
        )

        self.output = self.result["output"]
        self.artifacts = self.result["artifacts"]
        self.artifact_root = Path(self.artifacts["root"]).resolve()
        self.relay_artifact_root = (
            self.code_review_agent_dir / "artifacts" / "relay"
        ).resolve()
        self.relay_artifacts = self.output["relay_artifacts"]

    async def test_artifacts(self):
        assert self.result["status"] == "succeeded"
        assert self.result["adapter_kind"] == self.adapter_kind
        assert self.result["metadata"]["adapter_runner"] == self.adapter_runner
        assert len(self.result.telemetry) == 1
        assert self.result.telemetry[0].provider == "relay"
        assert self.result.telemetry[0].metadata["relay_enabled"] is True
        assert "relay_mode" not in self.result.telemetry[0].metadata

        output = self.output
        assert output["adapter"] == self.output_adapter
        assert output["harness"] == "hermes"
        assert output["mode"] == self.mode
        assert output["base_url"] == f"{self.api_server}/v1"
        assert output["error"] is None
        assert output["relay_runtime"]["enabled"] is True
        assert output["relay_runtime"]["emitter"] == "hermes-agent/nemo-relay"
        assert output["failed"] is False

        assert "echo user_count=" in output["response"]

        hermes_home = Path(output["hermes_home"]).resolve()
        hermes_config_path = Path(output["hermes_config_path"]).resolve()
        assert hermes_home.is_dir()
        assert hermes_home.is_relative_to(self.code_review_agent_dir)
        assert hermes_config_path.is_file()
        assert hermes_config_path.is_relative_to(self.code_review_agent_dir)

        hermes_config = yaml.safe_load(hermes_config_path.read_text(encoding="utf-8"))
        assert hermes_config["model"]["provider"] == "nvidia"
        assert (
            hermes_config["model"]["default"]
            == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
        )
        assert hermes_config["model"]["base_url"] == f"{self.api_server}/v1"
        assert hermes_config["plugins"]["enabled"] == ["observability/nemo_relay"]
        assert output["hermes_native_config"]["plugins"] == ["observability/nemo_relay"]

        expected_artifact_root = (
            self.code_review_agent_dir / "artifacts" / self.artifact_dir
        ).resolve()
        assert self.artifact_root == expected_artifact_root
        assert self.artifact_root.is_dir()

        artifact_by_name = {
            artifact["name"]: artifact for artifact in self.artifacts["artifacts"]
        }
        assert "relay_config" in artifact_by_name
        assert "stdout" in artifact_by_name

        relay_config_path = Path(artifact_by_name["relay_config"]["path"]).resolve()
        assert relay_config_path.is_file()
        assert relay_config_path.is_relative_to(self.artifact_root)

        relay_config = json.loads(relay_config_path.read_text(encoding="utf-8"))
        assert relay_config["schema_version"] == "fabric.relay/v1alpha1"
        assert relay_config["relay"]["enabled"] is True
        assert relay_config["fabric"]["agent_name"] == "code-review-agent"

    async def test_atof_artifacts(self):
        kinds = {artifact["kind"] for artifact in self.relay_artifacts}
        assert "atof" in kinds

        atof_paths = [
            Path(artifact["path"]).resolve()
            for artifact in self.relay_artifacts
            if artifact["kind"] == "atof"
        ]
        assert atof_paths
        assert all(path.exists() for path in atof_paths)
        assert all(path.is_relative_to(self.relay_artifact_root) for path in atof_paths)

        atof_records = [
            json.loads(line) for line in atof_paths[0].read_text().strip().splitlines()
        ]
        expected_atof_fields = {
            "atof_version",
            "attributes",
            "category",
            "data",
            "kind",
            "metadata",
            "name",
            "parent_uuid",
            "scope_category",
            "timestamp",
            "uuid",
        }
        actual_atof_fields = set().union(*(record.keys() for record in atof_records))
        assert actual_atof_fields.issuperset(expected_atof_fields)

        record_kinds = {
            (record["name"], record.get("scope_category")) for record in atof_records
        }
        assert record_kinds.issuperset({("nvidia", "start"), ("nvidia", "end")})

        session_scopes = [
            record
            for record in atof_records
            if record["name"] == "hermes.session"
            or str(record["name"]).startswith("hermes-session-")
        ]
        assert {record.get("scope_category") for record in session_scopes} >= {
            "start",
            "end",
        }

        current_turn_scopes = [
            record
            for record in atof_records
            if record["name"] == "hermes.turn"
            and record.get("scope_category") in {"start", "end"}
        ]
        upstream_turn_marks = [
            record
            for record in atof_records
            if record["name"] in {"hermes.turn.start", "hermes.turn.end"}
        ]
        turn_marks = current_turn_scopes or upstream_turn_marks
        assert turn_marks

        fabric_scopes = [*session_scopes, *turn_marks]
        assert all(
            record["metadata"].get("hermes.execution_surface") == "fabric"
            or record["metadata"].get("platform") == self.atof_platform
            for record in fabric_scopes
        )

    async def test_atif_artifacts(self):
        kinds = {artifact["kind"] for artifact in self.relay_artifacts}
        assert "atif" in kinds

        atif_paths = [
            Path(artifact["path"]).resolve()
            for artifact in self.relay_artifacts
            if artifact["kind"] == "atif"
        ]
        assert atif_paths
        assert all(path.exists() for path in atif_paths)
        assert all(path.is_relative_to(self.relay_artifact_root) for path in atif_paths)

        trajectory = json.loads(atif_paths[0].read_text())
        assert trajectory["agent"]["name"] in {"code-review-agent", "Hermes Agent"}
        steps = trajectory["steps"]
        assert len(steps) == 2

        first_step = steps[0]
        assert first_step["source"] == "user"
        assert first_step["message"] == "Reply with exactly: relay ok"
        assert (
            first_step["extra"]["llm_request"]["model"]
            == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
        )

        last_step = steps[-1]
        assert last_step["source"] == "agent"
        # The upstream exporter derives this field from the provider's final
        # wire response. The mock streaming response has no final text field,
        # while Fabric's normalized response is assembled from its deltas.
        assert last_step["message"] in {"", self.output["response"]}
        assert (
            last_step["model_name"] == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
        )
        assert last_step["extra"]["invocation"]["framework"] == "nemo_relay"
        assert last_step["extra"]["invocation"]["status"] == "completed"
