import os
import shutil
from pathlib import Path

import pytest

CUR_DIR = Path(__file__).parent.resolve()

@pytest.fixture(name="restore_environ")
def restore_environ_fixture():
    """
    Fixture to restore the environment variables after a test.
    Since many of the adapters rely on environment variables, this fixture ensures that any changes made to
    environment variables during a test are reverted back to their original state after the test completes.
    """
    orig_vars = os.environ.copy()
    yield os.environ

    for key, value in orig_vars.items():
        os.environ[key] = value

    # Delete any new environment variables
    # Iterating over a copy of the keys as we will potentially be deleting keys in the loop
    for key in list(os.environ.keys()):
        if key not in orig_vars:
            del os.environ[key]

@pytest.fixture(name="hermes_cli_agent_dir_src", scope="session")
def hermes_cli_agent_dir_fixture() -> Path:
    agent_dir = CUR_DIR / "fixtures" / "hermes-cli-agent"
    assert agent_dir.exists(), f"Missing fake Hermes CLI agent directory: {agent_dir}"
    return agent_dir

@pytest.fixture(name="hermes_agent_dir")
def hermes_agent_dir_fixture(hermes_cli_agent_dir_src: Path, tmp_path: Path) -> Path:
    agent_dir = tmp_path / "hermes-cli-agent"
    shutil.copytree(hermes_cli_agent_dir_src, agent_dir)
    assert agent_dir.exists(), f"Missing fake Hermes CLI agent directory: {agent_dir}"
    return agent_dir.resolve()

@pytest.fixture(name="hermes_cli_profile", scope="session")
def hermes_cli_profile_fixture() -> str:
    return "env_local"


@pytest.fixture(name="hermes_command")
def hermes_command_fixture(hermes_agent_dir: Path) -> Path:
    hermes_command = hermes_agent_dir / "bin" / "fake-hermes.py"
    assert hermes_command.exists(
    ), f"Missing fake Hermes CLI: {hermes_command}"
    return hermes_command.resolve()
