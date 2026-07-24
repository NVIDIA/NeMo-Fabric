# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard the published adapter dependency boundary."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_pyproject(path: str) -> dict:
    return tomllib.loads(
        (ROOT_DIR / path / "pyproject.toml").read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("adapters/common", []),
        (
            "adapters/claude",
            ["nemo-fabric-adapters-common == 0.1.0", "tomli-w~=1.2"],
        ),
        (
            "adapters/codex",
            ["nemo-fabric-adapters-common == 0.1.0", "tomli-w~=1.2"],
        ),
        (
            "adapters/deepagents",
            [
                "nemo-fabric-adapters-common == 0.1.0",
                "langchain-mcp-adapters>=0.1,<0.3.0",
                "langchain-openai>=0.3",
                "langgraph-checkpoint-sqlite>=3.0,<4.0",
            ],
        ),
        ("adapters/hermes", ["nemo-fabric-adapters-common == 0.1.0"]),
    ],
)
def test_adapter_runtime_dependencies(path: str, expected: list[str]):
    project = load_pyproject(path)["project"]
    assert sorted(project.get("dependencies", [])) == sorted(expected)


def test_adapter_test_dependencies_are_root_only():
    manifest = load_pyproject("")
    expected = [
        "claude-agent-sdk==0.2.120",
        "deepagents>=0.6.12,<0.7.0",
        "hermes-agent>=0.17.0; python_version < '3.14'",
        "langchain>=1.3,<2.0",
        "langgraph>=1.2,<2.0",
        "nemo-relay[deepagents]>=0.6.0,<0.7",
        "openai-codex==0.144.4",
    ]
    assert sorted(manifest["dependency-groups"]["adapter-tests"]) == sorted(expected)
    assert "adapter-tests" not in manifest["tool"]["uv"]["default-groups"]


def test_claude_agent_extra_keeps_the_harness_explicit():
    extras = load_pyproject("")["project"]["optional-dependencies"]
    adapter = "nemo-fabric-adapters-claude == 0.1.0"

    assert extras["claude"] == [adapter]
    assert sorted(extras["claude-agent"]) == sorted(
        [adapter, "claude-agent-sdk==0.2.120"]
    )


def test_deepagents_relay_extra_does_not_install_the_harness():
    project = load_pyproject("adapters/deepagents")["project"]
    assert project["optional-dependencies"]["relay"] == ["nemo-relay>=0.6.0,<0.7"]
