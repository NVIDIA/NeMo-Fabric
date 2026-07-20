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


@pytest.fixture(name="requires_harbor", scope="session")
def requires_harbor_fixture():
    try:
        import harbor  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("Harbor is not installed")


@pytest.fixture(name="requires_hermes_agent", scope="session")
def requires_hermes_agent_fixture():
    try:
        import run_agent  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("Hermes Agent is not installed")


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
    Creates a writable copy of the example's assets for runtime tests.
    """
    return _copy_agent_dir(
        repo_root / "examples" / "code_review_agent",
        tmp_path,
        "code-review-agent",
    )


@pytest.fixture(name="api_server")
def api_server_fixture(unused_tcp_port: int) -> Iterator[str]:
    from _utils.mock_api_server import mock_api_server
    with mock_api_server(unused_tcp_port) as base_url:
        yield base_url

@pytest.fixture(name="nemo_relay")
def nemo_relay_fixture() -> types.ModuleType:
    return pytest.importorskip("nemo_relay", reason="nemo-relay extra is required")

@pytest.fixture(name="mock_nvidia_api_key")
def mock_nvidia_api_key_fixture() -> str:
    nak = "test123"
    os.environ["NVIDIA_API_KEY"] = nak
    return nak
