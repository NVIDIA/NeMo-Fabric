# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types
from pathlib import Path

from nemo_fabric import Fabric


async def test_hermes_cli_fields(hermes_command: Path, hermes_agent_dir: Path, hermes_cli_profile: str):
    # Ensure the hermes_cli adapter returns expected fields
    async with Fabric() as client:
        result = await client.run(
            hermes_agent_dir,
            profiles=[hermes_cli_profile],
            input="who are you?",
        )

    assert result["status"] == "succeeded"
    assert result["adapter_kind"] == "process"
    assert result["metadata"]["adapter_runner"] == "process"

    output = result["output"]
    assert output["adapter"] == "cli"
    assert output["command"][0] == hermes_command.as_posix()
    assert output["harness"] == "hermes"
    assert output["mode"] == "hermes_cli_runtime"
    assert output["model"] == "test-model"

    for dir_field in ('cwd', 'fabric_home', 'fabric_invocation', 'hermes_config_path', 'hermes_home'):
        # these should all be under the agent dir
        dir_path = Path(output[dir_field]).resolve()
        assert dir_path.exists(), f"Missing path for field {dir_field}: {dir_path}"
        assert dir_path.is_relative_to(hermes_agent_dir), f"Field {dir_field} is not under agent dir: {dir_path}"

    for field in ('base_url', 'enabled_toolsets', 'error', 'response'):
        # Ensure these fields are present in the output, even if they are None
        assert field in output, f"Missing field in output: {field}"

    assert Path(output["hermes_home"]).parts[-2:] == ("runtimes", result["runtime_id"])


async def test_hermes_cli_rejects_native_telemetry(
    hermes_agent_dir: Path,
    hermes_cli_profile: str,
):
    profile_path = hermes_agent_dir / "profiles/native-telemetry.yaml"
    profile_path.write_text(
        """schema_version: fabric.profile/v1alpha1
name: native_telemetry
telemetry:
  enabled: true
  provider: native
  config: {}
""",
        encoding="utf-8",
    )

    async with Fabric() as client:
        result = await client.run(
            hermes_agent_dir,
            profiles=[hermes_cli_profile, "native_telemetry"],
            input="who are you?",
        )

    assert result["status"] == "failed"
    assert "only relay telemetry is supported for Hermes" in result["error"]["message"]


async def test_hermes_cli_multi_turn(
    hermes_agent_dir: Path,
    hermes_cli_runtime_profile: str,
    hermes_state: types.ModuleType,
):
    """
    Test that multi-turn runtime state is tracked in the Hermes session database.

    This test calls the fake-hermes.py script rather than hermes itself, thus it doesn't require an API key, however
    the hermes_cli adapter does use the hermes_state module, so we can test that the session is recorded propperly.
    """
    async with await Fabric().start_runtime(
        hermes_agent_dir,
        profiles=[hermes_cli_runtime_profile],
    ) as runtime:
        runtime_id = runtime.runtime_id
        await runtime.invoke(input="prompt1")
        result = await runtime.invoke(input="prompt2")

    session_db_path = Path(result["output"]["hermes_home"]) / "state.db"
    assert session_db_path.exists(), f"Expected session DB at {session_db_path} does not exist"

    session_db = hermes_state.SessionDB(db_path=session_db_path)
    session = session_db.get_session_by_title(runtime_id)
    assert session is not None
    assert session['id'] == runtime_id
    assert session['model'] == 'test-model'
    assert session['source'] == 'fabric'
    assert session['title'] == runtime_id
