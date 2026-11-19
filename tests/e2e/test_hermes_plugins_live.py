# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in live checks for Hermes plugins through the Fabric Hermes adapter.

Each test starts a real Hermes runtime through the public SDK against a real
model and has the agent exercise a plugin::

    RUN_FABRIC_LIVE_PLUGIN_E2E=1 \\
    FABRIC_LIVE_MODEL_BASE_URL=http://model-host:8000/v1 \\
    FABRIC_LIVE_MODEL=nvidia/nemotron-3-ultra-550b-a55b \\
    ADAPTER_PYTHON=.venv-hermes/bin/python \\
        .venv-hermes/bin/python -m pytest tests/e2e/test_hermes_plugins_live.py

Install the pinned Hermes environment first with ``just install-hermes-agent``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("RUN_FABRIC_LIVE_PLUGIN_E2E") != "1",
        reason="set RUN_FABRIC_LIVE_PLUGIN_E2E=1 to run live Hermes plugin checks",
    ),
    pytest.mark.usefixtures("requires_hermes_agent"),
]

BASE_URL = os.environ.get("FABRIC_LIVE_MODEL_BASE_URL", "")
MODEL = os.environ.get("FABRIC_LIVE_MODEL", "")
KEY_ENV = "FABRIC_LIVE_MODEL_API_KEY"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "hermes_plugins"


def _config(
    tmp_path: Path,
    settings: dict[str, Any],
    *,
    tools: list[str] | None,
    instruction: str,
    keep_state: bool = False,
):
    """A Hermes config for the live model, with ``settings`` as harness settings."""
    if not BASE_URL or not MODEL:
        pytest.skip("set FABRIC_LIVE_MODEL_BASE_URL and FABRIC_LIVE_MODEL")
    from examples.code_review_agent import hermes_config
    from nemo_fabric import (
        EnvironmentConfig,
        InstructionConfig,
        InstructionsConfig,
        ModelConfig,
        RuntimeConfig,
        ToolsConfig,
    )

    os.environ.setdefault(KEY_ENV, "unused")
    os.environ.setdefault("ADAPTER_PYTHON", sys.executable)
    config = hermes_config()
    config.skills = None
    config.models = {
        "default": ModelConfig(
            provider="custom", model=MODEL, api_key_env=KEY_ENV, base_url=BASE_URL
        )
    }
    config.harness.settings = {**config.harness.settings, **settings}
    config.tools = ToolsConfig(enabled=tools) if tools is not None else None
    config.instructions = InstructionsConfig(
        system=InstructionConfig(content=instruction)
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    config.environment = EnvironmentConfig(
        provider="local", workspace=workspace, artifacts=tmp_path / "artifacts"
    )
    config.environment.env[KEY_ENV] = os.environ[KEY_ENV]
    config.runtime = RuntimeConfig(
        input_schema="chat",
        output_schema="message",
        artifacts=tmp_path / "artifacts",
        max_turns=10,
        keep_state=keep_state,
    )
    return config


def _tool_results(result) -> list[str]:
    """The text of every tool result message in a run's output."""
    texts = []
    for message in result["output"].get("messages", []):
        if message.get("role") == "tool":
            content = message.get("content")
            texts.append(content if isinstance(content, str) else json.dumps(content))
    return texts


# --- Bundled plugins ------------------------------------------------------


async def test_security_guidance_warns_on_an_agent_file_write(tmp_path: Path):
    from nemo_fabric import Fabric

    config = _config(
        tmp_path,
        {
            "plugins": {
                "bundled": {
                    "entries": {
                        "security-guidance": {"enabled": True},
                        "browser/firecrawl": {"enabled": False},
                    }
                },
                "load_timeout_seconds": 20,
            }
        },
        tools=["file"],
        instruction="You are a coding assistant. Use the write_file tool to write files.",
    )

    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        result = await runtime.invoke(
            input=(
                "Use write_file to create loader.py in the workspace with exactly "
                "this content:\n\nimport pickle\n\n"
                "def load(path):\n    with open(path, 'rb') as f:\n"
                "        return pickle.load(f)\n\n"
                "Then reply with the tool result text verbatim."
            )
        )

    mapping = result.to_mapping()
    assert result["status"] == "succeeded", mapping
    native = result["output"]["hermes_native_config"]
    assert native["plugins"] == ["security-guidance"], native
    assert native["plugins_disabled"] == ["browser/firecrawl"], native
    assert (tmp_path / "workspace" / "loader.py").is_file(), mapping
    assert any("Security guidance" in text for text in _tool_results(result)), mapping


# --- External plugins -----------------------------------------------------


async def test_agent_calls_a_mounted_plugin_tool(tmp_path: Path):
    import shutil

    from nemo_fabric import Fabric

    # Stands in for a mounted volume.
    mount = tmp_path / "mount" / "hermes-mounted"
    shutil.copytree(FIXTURES / "hermes-mounted", mount)
    config = _config(
        tmp_path,
        {
            "plugins": {
                "external": {
                    "paths": [str(mount)],
                    "entries": {"hermes-mounted": {"enabled": True}},
                }
            }
        },
        tools=["hermes_mounted"],
        instruction="You are a helpful assistant. Use your tools.",
    )

    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        result = await runtime.invoke(
            input=(
                "Call the hermes_mounted tool with word 'fabric' and reply with "
                "the result text verbatim."
            )
        )

    mapping = result.to_mapping()
    assert result["status"] == "succeeded", mapping
    assert any("HERMES-MOUNTED-OK:fabric" in text for text in _tool_results(result)), (
        mapping
    )
    home = Path(result["output"]["hermes_home"])
    assert not (home / "plugins" / "hermes-mounted").is_symlink(), mapping
    assert (mount / "plugin.yaml").is_file()


# --- Agent-installed plugins (not supported) ------------------------------


def _git_source(tmp_path: Path, name: str) -> str:
    """A local Git repository holding the fixture plugin ``name``, as a URL."""
    import shutil
    import subprocess

    repo = tmp_path / "sources" / name
    shutil.copytree(FIXTURES / name, repo)
    git = ["git", "-C", str(repo), "-c", "user.name=fixture", "-c", "user.email=f@x"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run(
        [*git, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "fixture"],
        check=True,
    )
    return repo.as_uri()


async def test_agent_plugin_install_fails_the_next_kept_start_clearly(tmp_path: Path):
    import shutil

    from nemo_fabric import Fabric
    from nemo_fabric.errors import FabricRuntimeError

    source = _git_source(tmp_path, "hermes-installed")
    install = _config(
        tmp_path,
        {"terminal_timeout": 300},
        tools=["terminal"],
        instruction="You are a helpful assistant. Run shell commands with the terminal tool.",
        keep_state=True,
    )
    # Put the Hermes CLI beside the adapter interpreter on the agent's PATH.
    bin_dir = str(Path(os.environ["ADAPTER_PYTHON"]).parent)
    install.environment.env["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"

    async with await Fabric().start_runtime(install, base_dir=tmp_path) as runtime:
        first = await runtime.invoke(
            input=(
                "Run exactly this shell command with the terminal tool, then reply "
                f"with its last line:\n\nhermes plugins install {source} --enable"
            )
        )

    assert first["status"] == "succeeded", first.to_mapping()
    home = Path(first["output"]["hermes_home"])
    assert (home / "installs").is_dir(), first.to_mapping()

    with pytest.raises(FabricRuntimeError) as caught:
        async with await Fabric().start_runtime(install, base_dir=tmp_path):
            pass
    assert caught.value.code == "hermes_plugin_agent_installed_unsupported"
    assert str(home / "installs") in str(caught.value)

    # Removing the plugin environment, as the error says, lets the home start.
    shutil.rmtree(home / "installs")
    shutil.rmtree(home / "plugins" / "hermes-installed")
    async with await Fabric().start_runtime(install, base_dir=tmp_path) as runtime:
        assert runtime is not None
