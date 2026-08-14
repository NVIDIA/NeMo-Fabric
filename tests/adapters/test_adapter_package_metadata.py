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


SDK_PACKAGE_PATH = "sdk/python/nemo-fabric"
PACKAGE_VERSION = load_pyproject(SDK_PACKAGE_PATH)["project"]["version"]
RUNTIME_DEPENDENCY = f"nemo-fabric-runtime == {PACKAGE_VERSION}"
RELAY_CLI_DEPENDENCY = "nemo-relay-cli-bin>=0.7.2,<0.8"

ADAPTER_EXTRAS = {
    "claude": {
        "path": "adapters/claude",
        "sdk": f"nemo-fabric-adapters-claude[harness] == {PACKAGE_VERSION}",
        "harness": ["claude-agent-sdk==0.2.120", RELAY_CLI_DEPENDENCY],
    },
    "codex": {
        "path": "adapters/codex",
        "sdk": f"nemo-fabric-adapters-codex[harness] == {PACKAGE_VERSION}",
        "harness": ["openai-codex==0.144.4", RELAY_CLI_DEPENDENCY],
    },
    "deepagents": {
        "path": "adapters/deepagents",
        "sdk": f"nemo-fabric-adapters-deepagents[harness] == {PACKAGE_VERSION}",
        "harness": [
            "deepagents>=0.6.12,<0.7.0",
            "langchain>=1.3,<2.0",
            "langgraph>=1.2,<2.0",
        ],
        "relay": ["nemo-relay>=0.7.2,<0.8"],
        "full_relay": ["nemo-relay[deepagents]>=0.7.2,<0.8"],
    },
    "hermes-agent": {
        "path": "adapters/hermes",
        "sdk": (
            f"nemo-fabric-adapters-hermes[harness] == {PACKAGE_VERSION}; "
            "python_version < '3.14'"
        ),
        "harness": ["hermes-agent[mcp]>=0.19.0; python_version < '3.14'"],
        "relay": ["nemo-relay>=0.7.2,<0.8"],
    },
}


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("adapter-contract/python", []),
        ("adapters/common", []),
        (
            "adapters/claude",
            [
                f"nemo-fabric-adapter-contract == {PACKAGE_VERSION}",
                f"nemo-fabric-adapters-common == {PACKAGE_VERSION}",
                "tomli-w~=1.2",
            ],
        ),
        (
            "adapters/codex",
            [
                f"nemo-fabric-adapter-contract == {PACKAGE_VERSION}",
                f"nemo-fabric-adapters-common == {PACKAGE_VERSION}",
                "tomli-w~=1.2",
            ],
        ),
        (
            "adapters/deepagents",
            [
                f"nemo-fabric-adapter-contract == {PACKAGE_VERSION}",
                f"nemo-fabric-adapters-common == {PACKAGE_VERSION}",
                "langchain-mcp-adapters>=0.1,<0.3.0",
                "langchain-openai>=0.3",
                "langgraph-checkpoint-sqlite>=3.0,<4.0",
            ],
        ),
        (
            "adapters/hermes",
            [
                f"nemo-fabric-adapter-contract == {PACKAGE_VERSION}",
                f"nemo-fabric-adapters-common == {PACKAGE_VERSION}",
            ],
        ),
    ],
)
def test_adapter_runtime_dependencies(path: str, expected: list[str]):
    project = load_pyproject(path)["project"]
    assert project["version"] == PACKAGE_VERSION
    assert sorted(project.get("dependencies", [])) == sorted(expected)


def test_adapter_contract_offers_optional_pydantic_interop():
    extras = load_pyproject("adapter-contract/python")["project"][
        "optional-dependencies"
    ]

    assert extras == {"pydantic": ["pydantic>=2.12,<3"]}


def test_adapter_test_dependency_group_matches_leaf_harnesses():
    manifest = load_pyproject("")
    expected = [
        "nemo-fabric-adapters-claude[harness]",
        "nemo-fabric-adapters-codex[harness]",
        "nemo-fabric-adapters-deepagents[harness]",
        "nemo-fabric-adapters-hermes[harness]; python_version < '3.14'",
    ]
    assert sorted(manifest["dependency-groups"]["adapter-tests"]) == sorted(expected)
    assert "adapter-tests" not in manifest["tool"]["uv"]["default-groups"]


def test_sdk_package_installs_runtime_unconditionally():
    manifest = load_pyproject(SDK_PACKAGE_PATH)
    project = manifest["project"]

    assert project["dependencies"] == [RUNTIME_DEPENDENCY]
    assert manifest["tool"]["setuptools"]["packages"] == []


def test_root_project_is_a_private_development_coordinator():
    manifest = load_pyproject("")

    assert manifest["project"]["name"] == "nemo-fabric-development"
    assert manifest["project"]["version"] == "0.0.0"
    assert manifest["project"]["dependencies"] == ["nemo-fabric", RUNTIME_DEPENDENCY]
    assert manifest["tool"]["uv"]["package"] is False
    assert manifest["project"]["optional-dependencies"] == {
        "claude": ["nemo-fabric[claude]"],
        "codex": ["nemo-fabric[codex]"],
        "deepagents": ["nemo-fabric[deepagents]"],
        "harbor": ["nemo-fabric[harbor]"],
        "hermes-agent": [
            "nemo-fabric[hermes-agent]; python_version < '3.14'"
        ],
        "relay": ["nemo-fabric[relay]"],
    }


def test_sdk_relay_extra_installs_only_relay():
    extras = load_pyproject(SDK_PACKAGE_PATH)["project"]["optional-dependencies"]
    assert extras["relay"] == ["nemo-relay>=0.7.2,<0.8"]


@pytest.mark.parametrize("name", ADAPTER_EXTRAS)
def test_sdk_adapter_extras_delegate_to_leaf_harness_extras(name: str):
    extras = load_pyproject(SDK_PACKAGE_PATH)["project"]["optional-dependencies"]
    assert extras[name] == [ADAPTER_EXTRAS[name]["sdk"]]
    assert set(extras) == set(ADAPTER_EXTRAS) | {"harbor", "relay"}


@pytest.mark.parametrize("name", ADAPTER_EXTRAS)
def test_leaf_adapter_extras_separate_harness_and_relay(name: str):
    expected = ADAPTER_EXTRAS[name]
    extras = load_pyproject(expected["path"])["project"]["optional-dependencies"]
    relay = expected.get("relay", [])
    full_relay = expected.get("full_relay", relay)

    assert extras["harness"] == expected["harness"]
    assert sorted(extras["full"]) == sorted([*expected["harness"], *full_relay])

    expected_names = {"harness", "full"}
    if relay:
        expected_names.add("relay")
        assert extras["relay"] == relay
    assert set(extras) == expected_names
