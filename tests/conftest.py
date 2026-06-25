# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import sys
import types
from pathlib import Path

import pytest

CUR_DIR = Path(__file__).parent.resolve()

@pytest.fixture(name="restore_environ", autouse=True)
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

@pytest.fixture(name="repo_root", scope="session")
def repo_root_fixture() -> Path:
    return CUR_DIR.parent.resolve()

@pytest.fixture(name="hermes_cli_agent_dir_src", scope="session")
def hermes_cli_agent_dir_fixture() -> Path:
    agent_dir = CUR_DIR / "fixtures" / "hermes-cli-agent"
    assert agent_dir.exists(), f"Missing fake Hermes CLI agent directory: {agent_dir}"
    return agent_dir

@pytest.fixture(name="hermes_agent_dir")
def hermes_agent_dir_fixture(hermes_cli_agent_dir_src: Path, tmp_path: Path) -> Path:
    """
    Creates a temporary copy of the fake Hermes CLI agent directory for testing.
    This mirrors the behavior of the smoke tests.
    """
    agent_dir = tmp_path / "hermes-cli-agent"
    shutil.copytree(hermes_cli_agent_dir_src, agent_dir)
    assert agent_dir.exists(), f"Missing fake Hermes CLI agent directory: {agent_dir}"
    return agent_dir.resolve()

@pytest.fixture(name="hermes_cli_profile", scope="session")
def hermes_cli_profile_fixture() -> str:
    return "env_local"

@pytest.fixture(name="hermes_cli_session_profile")
def hermes_cli_session_profile_fixture(repo_root: Path, hermes_agent_dir: Path) -> str:
    src_yaml = repo_root / "examples/code-review-agent/profiles/hermes-cli-session.yaml"
    assert src_yaml.exists(), f"Missing hermes-cli-session.yaml profile: {src_yaml}"
    shutil.copy(src_yaml, hermes_agent_dir / "profiles/hermes-cli-session.yaml")
    return "hermes_cli_session"


@pytest.fixture(name="hermes_command")
def hermes_command_fixture(hermes_agent_dir: Path) -> Path:
    hermes_command = hermes_agent_dir / "bin" / "fake-hermes.py"
    assert hermes_command.exists(
    ), f"Missing fake Hermes CLI: {hermes_command}"
    return hermes_command.resolve()

@pytest.fixture(name="adapters_common_src_dir", scope="session")
def adapters_common_src_dir_fixture() -> Path:
    adapters_common_src_dir = CUR_DIR.parent / "adapters" / "common" / "src"
    assert adapters_common_src_dir.exists(), f"Missing adapters common src directory: {adapters_common_src_dir}"
    return adapters_common_src_dir.resolve()

@pytest.fixture(name="adapters_common", scope="session")
def adapters_common_fixture(adapters_common_src_dir: Path) -> str:
    adapters_common = adapters_common_src_dir.as_posix()
    if adapters_common not in sys.path:
        sys.path.append(adapters_common)

    return adapters_common

@pytest.fixture(name="hermes_common", scope="session")
def hermes_common_fixture(adapters_common: str) -> types.ModuleType:
    import nemo_fabric_adapters.common.hermes as hermes_common  # noqa: E402
    return hermes_common

@pytest.fixture(name="hermes_state", scope="session")
def require_hermes_state_fixture() -> types.ModuleType:
    """
    Fixture to ensure that the hermes_state module is available for tests that require it.
    """
    try:
        import hermes_state
        return hermes_state
    except ImportError:
        pytest.skip("Skipping test because hermes-agent is not installed.")
