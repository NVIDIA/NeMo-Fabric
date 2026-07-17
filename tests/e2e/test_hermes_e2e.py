# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from examples.code_review_agent import (
    hermes_config,
    with_relay,
)
from nemo_fabric import Fabric

pytestmark = pytest.mark.usefixtures("requires_hermes_agent")


@pytest.mark.usefixtures("mock_nvidia_api_key")
async def test_hermes_persistent_host_reuses_native_session(
    code_review_agent_dir: Path,
    api_server: str,
):
    os.environ["ADAPTER_PYTHON"] = sys.executable
    config = hermes_config()
    config.harness.settings["base_url"] = f"{api_server}/v1"

    async with await Fabric().start_runtime(
        config, base_dir=code_review_agent_dir
    ) as runtime:
        first = await runtime.invoke(input="first")
        second = await runtime.invoke(input="second")

    results = (first.to_mapping(), second.to_mapping())
    assert first["status"] == second["status"] == "succeeded", results
    assert first["metadata"]["adapter_runner"] == "persistent_local_host", results
    assert first["metadata"]["host_pid"] == second["metadata"]["host_pid"], results
    assert "user_count=2" in second["output"]["response"], results


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
        config.harness.settings["base_url"] = f"{api_server}/v1"
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
        assert output["relay_runtime"]["emitter"] == "hermes.observability/nemo_relay"
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
        assert hermes_config["model"]["default"] == "nvidia/nemotron-3-nano-30b-a3b"
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

        assert len(atof_records) == 7

        assert all(
            record["metadata"]["model"] == "nvidia/nemotron-3-nano-30b-a3b"
            and record["metadata"]["platform"] == self.atof_platform
            for record in atof_records
        )

        assert atof_records[-2]["name"] == "hermes.session.end"
        assert atof_records[-1]["scope_category"] == "end"

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
        assert len(steps) == 5

        first_step = steps[0]
        assert first_step["message"] == "hermes.turn.start"
        assert first_step["extra"]["event_payload"]["is_first_turn"] is True

        last_step = steps[-1]
        assert last_step["message"] == "hermes.session.end"
        assert last_step["extra"]["invocation"]["framework"] == "nemo_relay"
        assert last_step["extra"]["invocation"]["status"] == "completed"
