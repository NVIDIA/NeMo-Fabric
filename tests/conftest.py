# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest

CUR_DIR = Path(__file__).parent.resolve()
REPO_ROOT = CUR_DIR.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    shutil.copytree(hermes_cli_agent_dir_src, agent_dir, ignore=shutil.ignore_patterns("artifacts"))
    assert agent_dir.exists(), f"Missing fake Hermes CLI agent directory: {agent_dir}"
    return agent_dir.resolve()

@pytest.fixture(name="hermes_shim_agent_dir_src", scope="session")
def hermes_shim_agent_dir_src_fixture() -> Path:
    agent_dir = CUR_DIR / "fixtures" / "hermes-shim-agent"
    assert agent_dir.exists(), f"Missing Hermes shim agent directory: {agent_dir}"
    return agent_dir

def _copy_agent_dir(src_dir: Path, tmp_path: Path, agent_name: str) -> Path:
    """
    Creates a temporary copy of the specified agent directory for testing.
    This mirrors the behavior of the smoke tests.
    """
    assert src_dir.exists(), f"Missing directory: {src_dir}"
    agent_dir = tmp_path / agent_name
    shutil.copytree(src_dir, agent_dir, ignore=shutil.ignore_patterns("artifacts"))
    assert agent_dir.exists(), f"Missing {agent_name} directory: {agent_dir}"
    return agent_dir.resolve()

@pytest.fixture(name="hermes_shim_agent_dir")
def hermes_shim_agent_dir_fixture(
    hermes_shim_agent_dir_src: Path,
    tmp_path: Path,
) -> Path:
    """Creates a temporary copy of the Hermes shim agent directory."""
    return _copy_agent_dir(hermes_shim_agent_dir_src, tmp_path, "hermes-shim-agent")

@pytest.fixture(name="code_review_agent_dir")
def code_review_agent_dir_fixture(repo_root: Path, tmp_path: Path) -> Path:
    """
    Creates a writable copy of the typed example's assets for runtime tests.
    """
    return _copy_agent_dir(
        repo_root / "examples" / "code_review_agent",
        tmp_path,
        "code-review-agent",
    )


@pytest.fixture(name="file_config_agent_dir_src", scope="session")
def file_config_agent_dir_src_fixture(repo_root: Path) -> Path:
    """Return the test-only portable config package."""

    return repo_root / "tests" / "fixtures" / "file-config-agent"


@pytest.fixture(name="file_config_agent_dir")
def file_config_agent_dir_fixture(
    file_config_agent_dir_src: Path,
    tmp_path: Path,
) -> Path:
    """Create a writable copy for CLI and file-profile tests."""

    return _copy_agent_dir(file_config_agent_dir_src, tmp_path, "file-config-agent")

@pytest.fixture(name="hermes_cli_profile", scope="session")
def hermes_cli_profile_fixture() -> str:
    return "env_local"

@pytest.fixture(name="hermes_cli_runtime_profile")
def hermes_cli_runtime_profile_fixture(hermes_agent_dir: Path) -> str:
    import yaml

    config_path = hermes_agent_dir / "agent.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["harness"]["settings"]["prepare_runtime_state"] = True
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return "env_local"


@pytest.fixture(name="hermes_command")
def hermes_command_fixture(hermes_agent_dir: Path) -> Path:
    hermes_command = hermes_agent_dir / "bin" / "fake-hermes.py"
    assert hermes_command.exists(
    ), f"Missing fake Hermes CLI: {hermes_command}"
    return hermes_command.resolve()

@pytest.fixture(name="api_server")
def api_server_fixture(unused_tcp_port: int) -> Iterator[str]:
    from _utils.mock_api_server import mock_api_server
    with mock_api_server(unused_tcp_port) as base_url:
        yield base_url

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

@pytest.fixture(name="nemo_relay")
def nemo_relay_fixture() -> types.ModuleType:
    return pytest.importorskip("nemo_relay", reason="nemo-relay extra is required")

@pytest.fixture(name="hermes_state", scope="session")
def require_hermes_state_fixture() -> types.ModuleType:
    """
    Fixture to ensure that the hermes_state module is available for tests that require it.
    """
    return pytest.importorskip("hermes_state", reason="hermes extra is required")

@pytest.fixture(name="mock_nvidia_api_key")
def mock_nvidia_api_key_fixture() -> str:
    nak = "test123"
    os.environ["NVIDIA_API_KEY"] = nak
    return nak
