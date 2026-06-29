# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types
from pathlib import Path

from nemo_fabric import FabricClient


async def test_hermes_cli_fields(hermes_command: Path, hermes_agent_dir: Path, hermes_cli_profile: str):
    # Ensure the hermes_cli adapter returns expected fields
    async with FabricClient() as client:
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


async def test_hermes_cli_multi_turn(
    hermes_agent_dir: Path,
    hermes_cli_session_profile: str,
    hermes_state: types.ModuleType,
):
    """
    Test that multi-turn sessions are tracked in the hermes session database when using the hermes_cli adapter.

    This test calls the fake-hermes.py script rather than hermes itself, thus it doesn't require an API key, however
    the hermes_cli adapter does use the hermes_state module, so we can test that the session is recorded propperly.
    """
    async with await FabricClient().start_session(
        hermes_agent_dir,
        profiles=[hermes_cli_session_profile],
    ) as session:
        runtime_id = session.runtime["runtime_id"]
        await session.invoke(input="prompt1")
        await session.invoke(input="prompt2")

    session_db_path = hermes_agent_dir / "artifacts/hermes-home/state.db"
    assert session_db_path.exists(), f"Expected session DB at {session_db_path} does not exist"

    session_db = hermes_state.SessionDB(db_path=session_db_path)
    session = session_db.get_session_by_title(runtime_id)
    assert session is not None
    assert session['id'] == runtime_id
    assert session['model'] == 'test-model'
    assert session['source'] == 'fabric'
    assert session['title'] == runtime_id
