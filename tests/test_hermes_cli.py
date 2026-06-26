# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import types
from pathlib import Path

import pytest
import yaml

from _utils.utils import update_hermes_cli_relay_base_url
from nemo_fabric import FabricClient


async def test_hermes_cli_fields(hermes_command: Path, hermes_agent_dir: Path, hermes_cli_profile: str):
    # Ensure the hermes_cli adapter returns expected fields
    async with FabricClient() as client:
        result = await client.run(hermes_agent_dir,
                                  profile=hermes_cli_profile,
                                  input_text="who are you?")

    assert result["status"] == "succeeded"
    assert result["adapter_kind"] == "process"
    assert result["metadata"]["adapter_runner"] == "process"

    output = result["output"]
    assert output["adapter"] == "cli"
    assert output["command"][0] == hermes_command.as_posix()
    assert output["harness"] == "hermes"
    assert output["mode"] == "hermes_cli_oneshot"
    assert output["model"] == "test-model"

    for dir_field in ('cwd', 'fabric_home', 'fabric_invocation', 'hermes_config_path', 'hermes_home'):
        # these should all be under the agent dir
        dir_path = Path(output[dir_field]).resolve()
        assert dir_path.exists(), f"Missing path for field {dir_field}: {dir_path}"
        assert dir_path.is_relative_to(hermes_agent_dir), f"Field {dir_field} is not under agent dir: {dir_path}"
 
    for field in ('base_url', 'enabled_toolsets', 'error', 'response'):
        # Ensure these fields are present in the output, even if they are None
        assert field in output, f"Missing field in output: {field}"


async def test_hermes_cli_multi_turn(hermes_agent_dir: Path, hermes_cli_session_profile: str, hermes_state: types.ModuleType):
    """
    Test that multi-turn sessions are tracked in the hermes session database when using the hermes_cli adapter.

    This test calls the fake-hermes.py script rather than hermes itself, thus it doesn't require an API key, however
    the hermes_cli adapter does use the hermes_state module, so we can test that the session is recorded propperly.
    """
    async with await FabricClient().start(hermes_agent_dir,
                                          profile=hermes_cli_session_profile) as session:
        runtime_id = session.runtime["runtime_id"]
        await session.invoke("prompt1")
        await session.invoke("prompt2")

    session_db_path = hermes_agent_dir / "artifacts/hermes-home/state.db"
    assert session_db_path.exists(), f"Expected session DB at {session_db_path} does not exist"

    session_db = hermes_state.SessionDB(db_path=session_db_path)
    session = session_db.get_session_by_title(runtime_id)
    assert session is not None
    assert session['id'] == runtime_id
    assert session['model'] == 'test-model'
    assert session['source'] == 'fabric'
    assert session['title'] == runtime_id


class TestHermesE2E:
    """
    E2E Hermes tests, which communicate with a mock API server not requiring an API key.
    """

    @pytest.fixture(autouse=True)
    async def run_hermes_cli_relay(
        self,
        nemo_relay: types.ModuleType,
        mock_nvidia_api_key: str,
        code_review_agent_dir: Path,
        api_server: str,
    ):
        assert nemo_relay is not None
        assert mock_nvidia_api_key == "test123"
        self.code_review_agent_dir = code_review_agent_dir
        self.api_server = api_server
        update_hermes_cli_relay_base_url(code_review_agent_dir, api_server)

        async with FabricClient() as client:
            self.result = await client.run(
                code_review_agent_dir,
                profile="hermes_cli_relay",
                input_text="Reply with exactly: relay ok",
            )

        self.output = self.result["output"]
        self.artifacts = self.result["artifacts"]
        self.artifact_root = Path(self.artifacts["root"]).resolve()
        self.relay_artifacts = self.output["relay_artifacts"]

    async def test_artifacts(self):
        assert self.result["status"] == "succeeded"
        assert self.result["adapter_kind"] == "process"
        assert self.result["metadata"]["adapter_runner"] == "process"
        assert self.result["telemetry"]["relay_enabled"] is True
        assert self.result["telemetry"]["metadata"]["relay_mode"] == "sdk"

        output = self.output
        assert output["adapter"] == "cli"
        assert output["harness"] == "hermes"
        assert output["mode"] == "hermes_cli_oneshot"
        assert output["base_url"] == f"{self.api_server}/v1"
        assert output["returncode"] == 0
        assert output["error"] is None
        assert output["relay_runtime"]["enabled"] is True
        assert output["relay_runtime"]["mode"] == "sdk"
        assert output["relay_runtime"]["emitter"] == "hermes.observability/nemo_relay"

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
            self.code_review_agent_dir / "artifacts" / "hermes-cli-relay"
        ).resolve()
        assert self.artifact_root == expected_artifact_root
        assert self.artifact_root.is_dir()

        artifact_by_name = {
            artifact["name"]: artifact
            for artifact in self.artifacts["artifacts"]
        }
        assert "relay_config" in artifact_by_name
        assert "stdout" in artifact_by_name

        relay_config_path = Path(artifact_by_name["relay_config"]["path"]).resolve()
        assert relay_config_path.is_file()
        assert relay_config_path.is_relative_to(self.artifact_root)
        relay_config = json.loads(relay_config_path.read_text(encoding="utf-8"))
        assert relay_config["schema_version"] == "fabric.relay/v1alpha1"
        assert relay_config["relay"]["enabled"] is True
        assert relay_config["fabric"]["profile"] == "hermes_cli_relay"

        fabric_invocation_path = Path(output["fabric_invocation"]).resolve()
        assert fabric_invocation_path.is_file()
        assert fabric_invocation_path.is_relative_to(self.artifact_root)
        assert fabric_invocation_path.name == "adapter-invocation.json"

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
        assert all(path.is_relative_to(self.artifact_root) for path in atof_paths)

        atof_records = [
            json.loads(line)
            for line in atof_paths[0].read_text().strip().splitlines()
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
        assert len(atof_records) == 7
        assert actual_atof_fields.issuperset(expected_atof_fields)
        assert all(
            record["metadata"]["model"] == "nvidia/nemotron-3-nano-30b-a3b"
            and record["metadata"]["platform"] == "cli"
            for record in atof_records
        )
        assert (
            atof_records[0]["name"]
            == f"hermes-session-{atof_records[0]['metadata']['session_id']}"
        )

        assert atof_records[-2]["name"] == "hermes.session.end"
        assert atof_records[-1]["scope_category"] == "end"
        assert atof_records[-1]["data"]["reason"] == "shutdown"

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
        assert all(path.is_relative_to(self.artifact_root) for path in atif_paths)

        trajectory = json.loads(atif_paths[0].read_text())
        assert trajectory["agent"]["name"] in {"code-review-agent", "Hermes Agent"}
        steps = trajectory["steps"]
        assert len(steps) == 5

        first_step = steps[0]
        assert first_step["message"] == "hermes.turn.start"
        assert first_step["extra"]["event_payload"]["is_first_turn"] is True

        last_step = steps[-1]
        assert last_step["message"] == "hermes.session.end"
        assert last_step["extra"]["event_payload"]["completed"] is True
        assert last_step["extra"]["invocation"]["framework"] == "nemo_relay"
        assert last_step["extra"]["invocation"]["status"] == "completed"
