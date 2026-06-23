from pathlib import Path

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
