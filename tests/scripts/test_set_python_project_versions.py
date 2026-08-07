# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


CI_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "ci"
sys.path.insert(0, str(CI_SCRIPTS))

import set_python_project_versions  # noqa: E402


def test_set_python_project_versions_updates_internal_pins_with_extras(
    tmp_path: Path,
):
    (tmp_path / "adapters" / "claude").mkdir(parents=True)
    (tmp_path / "adapter-contract").mkdir()
    (tmp_path / "python").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        """\
[project]
name = "nemo-fabric"
version = "0.2.0"
dependencies = [
  "nemo-fabric-runtime == 0.2.0",
]

[project.optional-dependencies]
claude = [
  "nemo-fabric-adapters-claude[harness] == 0.2.0",
]
hermes-agent = [
  "nemo-fabric-adapters-hermes[harness] == 0.2.0; python_version < '3.14'",
]
""",
        encoding="utf-8",
    )
    (tmp_path / "adapters" / "claude" / "pyproject.toml").write_text(
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
    (tmp_path / "adapter-contract" / "pyproject.toml").write_text(
        """\
[project]
name = "nemo-fabric-adapter-contract"
version = "0.2.0"
""",
        encoding="utf-8",
    )
    runtime_path = tmp_path / "python" / "pyproject.toml"
    runtime_path.write_text(
        """\
[project]
name = "nemo-fabric-runtime"
dynamic = ["version"]
""",
        encoding="utf-8",
    )

    set_python_project_versions.set_python_project_versions(tmp_path, "0.2.0rc5")

    root_project = tomllib.loads(
        (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    adapter_project = tomllib.loads(
        (tmp_path / "adapters" / "claude" / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )["project"]
    contract_project = tomllib.loads(
        (tmp_path / "adapter-contract" / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )["project"]

    assert root_project["version"] == "0.2.0rc5"
    assert root_project["dependencies"] == ["nemo-fabric-runtime == 0.2.0rc5"]
    assert root_project["optional-dependencies"]["claude"] == [
        "nemo-fabric-adapters-claude[harness] == 0.2.0rc5"
    ]
    assert root_project["optional-dependencies"]["hermes-agent"] == [
        "nemo-fabric-adapters-hermes[harness] == 0.2.0rc5; python_version < '3.14'"
    ]
    assert adapter_project["version"] == "0.2.0rc5"
    assert adapter_project["dependencies"] == [
        "nemo-fabric-adapters-common == 0.2.0rc5"
    ]
    assert contract_project["version"] == "0.2.0rc5"
    assert tomllib.loads(runtime_path.read_text(encoding="utf-8"))["project"][
        "dynamic"
    ] == ["version"]
