# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

CI_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "ci"
sys.path.insert(0, str(CI_SCRIPTS))

import set_python_project_versions  # noqa: E402


@pytest.mark.parametrize("version", ["0.2.0a20261007", "0.2.0rc5", "0.2.0"])
def test_set_python_project_versions_updates_internal_pins_with_extras(
    tmp_path: Path,
    version: str,
):
    (tmp_path / "adapters" / "python" / "claude").mkdir(parents=True)
    (tmp_path / "adapters" / "typescript" / "pi").mkdir(parents=True)
    (tmp_path / "adapter-contract" / "python").mkdir(parents=True)
    (tmp_path / "sdk" / "python" / "nemo-fabric-collector").mkdir(parents=True)
    (tmp_path / "sdk" / "python" / "nemo-fabric").mkdir(parents=True)
    (tmp_path / "sdk" / "python" / "nemo-fabric-runtime").mkdir()
    catalog_path = (
        tmp_path / "sdk" / "python" / "nemo-fabric-adapter-catalog" / "pyproject.toml"
    )
    catalog_path.parent.mkdir()
    catalog_path.write_text(
        '[project]\nname = "nemo-fabric-adapter-catalog"\nversion = "0.2.0"\n',
        encoding="utf-8",
    )
    coordinator_path = tmp_path / "pyproject.toml"
    coordinator_path.write_text(
        """\
[project]
name = "nemo-fabric-development"
version = "0.0.0"
dependencies = [
  "nemo-fabric",
  "nemo-fabric-runtime == 0.2.0",
]
""",
        encoding="utf-8",
    )
    sdk_path = tmp_path / "sdk" / "python" / "nemo-fabric" / "pyproject.toml"
    sdk_path.write_text(
        """\
[project]
name = "nemo-fabric"
version = "0.2.0"
dependencies = [
  "nemo-fabric-runtime == 0.2.0",
]

[project.optional-dependencies]
streaming = [
  "nemo-fabric-runtime[streaming] == 0.2.0",
]
claude = [
  "nemo-fabric-adapters-claude[harness] == 0.2.0",
]
hermes-agent = [
  "nemo-fabric-adapters-hermes[full] == 0.2.0; python_version < '3.14'",
]
""",
        encoding="utf-8",
    )
    collector_path = (
        tmp_path / "sdk" / "python" / "nemo-fabric-collector" / "pyproject.toml"
    )
    collector_path.write_text(
        """\
[project]
name = "nemo-fabric-collector"
version = "0.2.0"
""",
        encoding="utf-8",
    )
    (tmp_path / "adapters" / "python" / "claude" / "pyproject.toml").write_text(
        """\
[project]
name = "nemo-fabric-adapters-claude"
version = "0.2.0"
dependencies = [
  "nemo-fabric-adapters-common == 0.2.0",
]
""",
        encoding="utf-8",
    )
    typescript_decoy = tmp_path / "adapters" / "typescript" / "pi" / "pyproject.toml"
    typescript_decoy.write_text(
        '[project]\nname = "typescript-build-helper"\n',
        encoding="utf-8",
    )
    (tmp_path / "adapter-contract" / "python" / "pyproject.toml").write_text(
        """\
[project]
name = "nemo-fabric-adapter-contract"
version = "0.2.0"
""",
        encoding="utf-8",
    )
    runtime_path = (
        tmp_path / "sdk" / "python" / "nemo-fabric-runtime" / "pyproject.toml"
    )
    runtime_path.write_text(
        """\
[project]
name = "nemo-fabric-runtime"
dynamic = ["version"]

[project.optional-dependencies]
streaming = [
  "nemo-fabric-collector == 0.2.0",
]
""",
        encoding="utf-8",
    )

    set_python_project_versions.set_python_project_versions(tmp_path, version)

    sdk_project = tomllib.loads(sdk_path.read_text(encoding="utf-8"))["project"]
    adapter_project = tomllib.loads(
        (tmp_path / "adapters" / "python" / "claude" / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )["project"]
    contract_project = tomllib.loads(
        (tmp_path / "adapter-contract" / "python" / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )["project"]

    assert sdk_project["version"] == version
    assert sdk_project["dependencies"] == [f"nemo-fabric-runtime == {version}"]
    assert sdk_project["optional-dependencies"]["streaming"] == [
        f"nemo-fabric-runtime[streaming] == {version}"
    ]
    assert sdk_project["optional-dependencies"]["claude"] == [
        f"nemo-fabric-adapters-claude[harness] == {version}"
    ]
    assert sdk_project["optional-dependencies"]["hermes-agent"] == [
        f"nemo-fabric-adapters-hermes[full] == {version}; python_version < '3.14'"
    ]
    assert adapter_project["version"] == version
    assert adapter_project["dependencies"] == [
        f"nemo-fabric-adapters-common == {version}"
    ]
    assert (
        tomllib.loads(collector_path.read_text(encoding="utf-8"))["project"]["version"]
        == version
    )
    assert typescript_decoy.read_text(encoding="utf-8") == (
        '[project]\nname = "typescript-build-helper"\n'
    )
    assert contract_project["version"] == version
    assert tomllib.loads(catalog_path.read_text())["project"]["version"] == version
    runtime_project = tomllib.loads(runtime_path.read_text(encoding="utf-8"))["project"]
    assert runtime_project["dynamic"] == ["version"]
    assert runtime_project["optional-dependencies"]["streaming"] == [
        f"nemo-fabric-collector == {version}"
    ]
    coordinator_project = tomllib.loads(coordinator_path.read_text(encoding="utf-8"))[
        "project"
    ]
    assert coordinator_project["version"] == "0.0.0"
    assert coordinator_project["dependencies"] == [
        "nemo-fabric",
        f"nemo-fabric-runtime == {version}",
    ]
