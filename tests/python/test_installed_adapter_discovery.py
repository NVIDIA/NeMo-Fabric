# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the installed adapter descriptor discovery stopgap."""

from __future__ import annotations

import json
import os
import subprocess
import sysconfig
import time
import venv
from collections.abc import Callable
from pathlib import Path

import pytest
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import FabricConfigError


def _write_descriptor(
    root: Path,
    module: str,
    adapter_id: str = "test.fabric.installed",
) -> Path:
    descriptor = root / "share/nemo-fabric/adapters/test/fabric-adapter.json"
    descriptor.parent.mkdir(parents=True)
    descriptor.write_text(
        json.dumps(
            {
                "contract_version": "fabric.adapter/v1alpha1",
                "adapter_id": adapter_id,
                "harness": "installed-test",
                "adapter_kind": "python",
                "runner": {"module": module},
            }
        )
    )
    return descriptor


def _config(adapter_id: str = "test.fabric.installed") -> FabricConfig:
    return FabricConfig.from_mapping(
        {
            "metadata": {"name": "installed-adapter-test"},
            "harness": {"adapter_id": adapter_id},
        }
    )


@pytest.fixture(name="_clear_adapter_python", autouse=True)
def _clear_adapter_python_fixture() -> None:
    os.environ.pop("ADAPTER_PYTHON", None)


@pytest.fixture(name="patch_sysconfig_data")
def patch_sysconfig_data_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Path], None]:
    original_get_path = sysconfig.get_path

    def patch(data_root: Path) -> None:
        def get_path(
            name: str,
            *args: object,
            **kwargs: object,
        ) -> str | None:
            if name == "data":
                return str(data_root)
            return original_get_path(name, *args, **kwargs)

        monkeypatch.setattr(sysconfig, "get_path", get_path)

    return patch


def _python_sysconfig_path(python: Path, name: str) -> Path:
    return Path(
        subprocess.check_output(
            [
                python,
                "-c",
                f"import sysconfig; print(sysconfig.get_path({name!r}), end='')",
            ],
            text=True,
        )
    )


def test_plan_discovers_adapter_from_python_data_directory(
    tmp_path: Path,
    patch_sysconfig_data: Callable[[Path], None],
):
    data_root = tmp_path / "python-data"
    descriptor = _write_descriptor(data_root, "installed.adapter")
    patch_sysconfig_data(data_root)

    plan = Fabric().plan(_config(), base_dir=tmp_path / "agent")

    assert Path(plan["adapter_descriptor"]["path"]).samefile(descriptor)
    assert plan["adapter_descriptor"]["source"] == "local"


def test_agent_local_descriptor_overrides_installed_descriptor(
    tmp_path: Path,
    patch_sysconfig_data: Callable[[Path], None],
):
    data_root = tmp_path / "python-data"
    _write_descriptor(data_root, "installed.adapter")
    base_dir = tmp_path / "agent"
    local_descriptor = base_dir / "adapters/test/fabric-adapter.json"
    local_descriptor.parent.mkdir(parents=True)
    local_descriptor.write_text(
        json.dumps(
            {
                "contract_version": "fabric.adapter/v1alpha1",
                "adapter_id": "test.fabric.installed",
                "harness": "installed-test",
                "adapter_kind": "python",
                "runner": {"module": "local.adapter"},
            }
        )
    )
    patch_sysconfig_data(data_root)

    plan = Fabric().plan(_config(), base_dir=base_dir)

    assert Path(plan["adapter_descriptor"]["path"]).samefile(local_descriptor)
    assert (
        plan["adapter_descriptor"]["descriptor"]["runner"]["module"] == "local.adapter"
    )


def test_adapter_python_data_directory_replaces_current_data_directory(
    tmp_path: Path,
    patch_sysconfig_data: Callable[[Path], None],
):
    current_data_root = tmp_path / "current-python-data"
    _write_descriptor(
        current_data_root,
        "current.adapter",
        adapter_id="test.fabric.current-only",
    )
    patch_sysconfig_data(current_data_root)

    adapter_env = tmp_path / "adapter-env"
    venv.EnvBuilder(with_pip=False).create(adapter_env)
    adapter_python = adapter_env / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    adapter_data_root = _python_sysconfig_path(adapter_python, "data")
    adapter_descriptor = _write_descriptor(adapter_data_root, "adapter.environment")
    os.environ["ADAPTER_PYTHON"] = str(adapter_python)

    plan = Fabric().plan(_config(), base_dir=tmp_path / "agent")

    assert Path(plan["adapter_descriptor"]["path"]).samefile(adapter_descriptor)
    assert (
        plan["adapter_descriptor"]["descriptor"]["runner"]["module"]
        == "adapter.environment"
    )
    with pytest.raises(FabricConfigError, match="unknown adapter"):
        Fabric().plan(
            _config("test.fabric.current-only"),
            base_dir=tmp_path / "agent",
        )


def test_adapter_python_data_path_query_times_out(tmp_path: Path):
    adapter_env = tmp_path / "slow-adapter-env"
    venv.EnvBuilder(with_pip=False).create(adapter_env)
    adapter_python = adapter_env / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    purelib = _python_sysconfig_path(adapter_python, "purelib")
    (purelib / "slow_startup.pth").write_text("import time; time.sleep(30)\n")
    os.environ["ADAPTER_PYTHON"] = str(adapter_python)

    started = time.monotonic()
    with pytest.raises(FabricConfigError, match="timed out after 5 seconds"):
        Fabric().plan(_config(), base_dir=tmp_path / "agent")

    assert time.monotonic() - started < 10
