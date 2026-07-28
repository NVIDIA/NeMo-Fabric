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
from typing import Any

import pytest
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import FabricConfigError


ROOT = Path(__file__).resolve().parents[2]


def _closed_settings_schema(**properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }


def _write_descriptor(
    root: Path,
    module: str,
    adapter_id: str = "test.fabric.installed",
    settings_schema: dict[str, Any] | None = None,
) -> Path:
    descriptor = root / "share/nemo-fabric/adapters/test/fabric-adapter.json"
    descriptor.parent.mkdir(parents=True)
    data: dict[str, Any] = {
        "contract_version": "fabric.adapter/v1alpha1",
        "adapter_id": adapter_id,
        "harness": "installed-test",
        "adapter_kind": "python",
        "runner": {"module": module},
    }
    if settings_schema is not None:
        data["settings_schema"] = settings_schema
    descriptor.write_text(json.dumps(data))
    return descriptor


def _config(
    adapter_id: str = "test.fabric.installed",
    settings: dict[str, Any] | None = None,
) -> FabricConfig:
    return FabricConfig.from_mapping(
        {
            "metadata": {"name": "installed-adapter-test"},
            "harness": {
                "adapter_id": adapter_id,
                "settings": settings or {},
            },
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

def _create_venv(venv_dir: Path):
    # Use `symlinks=True` on Posix systems to work-around for
    # https://github.com/astral-sh/uv/issues/8879
    venv.EnvBuilder(with_pip=False,
                    symlinks=(os.name != "nt")).create(venv_dir)


@pytest.fixture(name="installed_claude_wheel", scope="session")
def installed_claude_wheel_fixture(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("installed-claude-wheel")
    wheelhouse = root / "wheelhouse"
    subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--out-dir",
            str(wheelhouse),
            str(ROOT / "adapters" / "claude"),
        ],
        check=True,
    )
    wheel = next(wheelhouse.glob("nemo_fabric_adapters_claude-*.whl"))
    adapter_env = root / "adapter-env"
    _create_venv(adapter_env)
    python = adapter_env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)],
        check=True,
    )
    descriptor = (
        _python_sysconfig_path(python, "data")
        / "share/nemo-fabric/adapters/claude/fabric-adapter.json"
    )
    assert descriptor.is_file()
    return python, descriptor


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
    _write_descriptor(
        data_root,
        "installed.adapter",
        settings_schema=_closed_settings_schema(installed_only={"type": "boolean"}),
    )
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
                "settings_schema": _closed_settings_schema(
                    local_only={"type": "boolean"}
                ),
            }
        )
    )
    patch_sysconfig_data(data_root)

    plan = Fabric().plan(
        _config(settings={"local_only": True}),
        base_dir=base_dir,
    )

    assert Path(plan["adapter_descriptor"]["path"]).samefile(local_descriptor)
    assert (
        plan["adapter_descriptor"]["descriptor"]["runner"]["module"] == "local.adapter"
    )
    with pytest.raises(FabricConfigError) as caught:
        Fabric().plan(
            _config(settings={"installed_only": True}),
            base_dir=base_dir,
        )
    message = str(caught.value)
    assert str(local_descriptor.resolve()) in message
    assert "harness.settings.installed_only" in message


def test_adapter_python_data_directory_replaces_current_data_directory(
    tmp_path: Path,
    patch_sysconfig_data: Callable[[Path], None],
):
    current_data_root = tmp_path / "current-python-data"
    _write_descriptor(
        current_data_root,
        "current.adapter",
        settings_schema=_closed_settings_schema(current_only={"type": "boolean"}),
    )
    patch_sysconfig_data(current_data_root)

    adapter_env = tmp_path / "adapter-env"
    _create_venv(adapter_env)
    adapter_python = adapter_env / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    adapter_data_root = _python_sysconfig_path(adapter_python, "data")
    adapter_descriptor = _write_descriptor(
        adapter_data_root,
        "adapter.environment",
        settings_schema=_closed_settings_schema(adapter_only={"type": "boolean"}),
    )
    os.environ["ADAPTER_PYTHON"] = str(adapter_python)

    plan = Fabric().plan(
        _config(settings={"adapter_only": True}),
        base_dir=tmp_path / "agent",
    )

    assert Path(plan["adapter_descriptor"]["path"]).samefile(adapter_descriptor)
    assert (
        plan["adapter_descriptor"]["descriptor"]["runner"]["module"]
        == "adapter.environment"
    )
    with pytest.raises(FabricConfigError) as caught:
        Fabric().plan(
            _config(settings={"current_only": True}),
            base_dir=tmp_path / "agent",
        )
    message = str(caught.value)
    assert str(adapter_descriptor.resolve()) in message
    assert "harness.settings.current_only" in message


def test_schema_less_descriptor_rejects_non_empty_settings(
    tmp_path: Path,
    patch_sysconfig_data: Callable[[Path], None],
):
    data_root = tmp_path / "python-data"
    descriptor = _write_descriptor(data_root, "schema.less")
    patch_sysconfig_data(data_root)

    plan = Fabric().plan(_config(), base_dir=tmp_path / "empty-agent")
    assert Path(plan["adapter_descriptor"]["path"]).samefile(descriptor)

    with pytest.raises(FabricConfigError) as caught:
        Fabric().plan(
            _config(settings={"unknown": True}),
            base_dir=tmp_path / "non-empty-agent",
        )
    message = str(caught.value)
    assert "test.fabric.installed" in message
    assert str(descriptor.resolve()) in message
    assert "harness.settings.unknown" in message


def test_installed_claude_wheel_supplies_metadata_and_settings_schema(
    tmp_path: Path,
    installed_claude_wheel: tuple[Path, Path],
):
    python, descriptor = installed_claude_wheel
    os.environ["ADAPTER_PYTHON"] = str(python)
    settings = {
        "setting_sources": ["project"],
        "max_budget_usd": 2.0,
        "permission_mode": "dontAsk",
    }

    plan = Fabric().plan(
        _config("nvidia.fabric.claude", settings),
        base_dir=tmp_path / "agent",
    )

    assert Path(plan["adapter_descriptor"]["path"]).samefile(descriptor)
    assert plan["adapter_descriptor"]["source"] == "local"
    assert plan.config.harness.settings == settings
    packaged_schema = plan["adapter_descriptor"]["descriptor"]["settings_schema"]
    canonical_schema = json.loads(
        (ROOT / "adapters/claude/fabric-adapter.json").read_text(encoding="utf-8")
    )["settings_schema"]
    assert packaged_schema == canonical_schema

    with pytest.raises(FabricConfigError) as caught:
        Fabric().plan(
            _config("nvidia.fabric.claude", {"unknown": True}),
            base_dir=tmp_path / "invalid-agent",
        )
    message = str(caught.value)
    assert str(descriptor.resolve()) in message
    assert "harness.settings.unknown" in message


def test_adapter_python_data_path_query_times_out(tmp_path: Path):
    adapter_env = tmp_path / "slow-adapter-env"
    _create_venv(adapter_env)
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
