# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit plugin installation preserves unrelated Hermes state."""

import json
import subprocess
import sys
from importlib.resources import files

import pytest

from nemo_fabric_adapters.hermes.install_plugins import install_tavily


@pytest.fixture(name="hermes_root")
def hermes_root_fixture(tmp_path):
    root = tmp_path / "hermes"
    (root / "plugins" / "unrelated").mkdir(parents=True)
    (root / "plugins" / "unrelated" / "keep.txt").write_text("keep")
    return root


def test_install_reports_native_plugin_path_and_preserves_identical_install(
    hermes_root,
):
    destination = install_tavily(hermes_root)
    assert destination == hermes_root / "plugins" / "web" / "tavily"
    source = files("nemo_fabric_adapters.hermes").joinpath("plugins", "tavily")
    for name in ("__init__.py", "plugin.yaml"):
        assert (destination / name).read_bytes() == source.joinpath(name).read_bytes()
    before = (destination / "__init__.py").stat().st_mtime_ns
    assert install_tavily(hermes_root) == destination
    assert (destination / "__init__.py").stat().st_mtime_ns == before
    assert (destination.stat().st_mode & 0o777) == 0o755
    assert (hermes_root / "plugins" / "unrelated" / "keep.txt").read_text() == "keep"
    assert not (hermes_root / "config.yaml").exists()


def test_conflicting_install_is_not_overwritten(hermes_root):
    destination = install_tavily(hermes_root)
    (destination / "plugin.yaml").write_text("user configuration")
    with pytest.raises(FileExistsError, match="different contents"):
        install_tavily(hermes_root)
    assert (destination / "plugin.yaml").read_text() == "user configuration"


@pytest.mark.parametrize("component", ["plugins", "plugins/web", "plugins/web/tavily"])
def test_install_does_not_follow_plugin_directory_symlinks(tmp_path, component):
    root = tmp_path / "hermes"
    root.mkdir()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    link = root / component
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(unrelated, target_is_directory=True)
    with pytest.raises(FileExistsError, match="symlink"):
        install_tavily(root)
    assert list(unrelated.iterdir()) == []


def test_missing_root_is_not_created(tmp_path):
    root = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        install_tavily(root)
    assert not root.exists()


def test_installer_module_prints_machine_readable_paths(hermes_root):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nemo_fabric_adapters.hermes.install_plugins",
            "--hermes-root",
            str(hermes_root),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "web/tavily": str(hermes_root / "plugins" / "web" / "tavily")
    }
    assert result.stderr == ""
