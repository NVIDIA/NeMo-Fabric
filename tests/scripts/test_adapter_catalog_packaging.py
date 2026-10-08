# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise release artifacts and dispatch without publishing a package."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

import pkginfo
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/ci"))
import artifactory_upload  # noqa: E402


@pytest.fixture(name="catalog_wheel", scope="module")
def catalog_wheel_fixture(
    repo_root: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    staging = tmp_path_factory.mktemp("catalog-distribution")
    source = staging / "source"
    shutil.copytree(
        repo_root / "sdk/python/nemo-fabric-adapter-catalog",
        source,
        ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__", ".venv"),
    )
    # Default uv build produces an sdist and builds the wheel from that sdist.
    subprocess.run(
        [
            "uv",
            "build",
            "--python",
            sys.executable,
            "--out-dir",
            str(staging / "dist"),
            str(source),
        ],
        cwd=staging,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert len(list((staging / "dist").glob("*.tar.gz"))) == 1
    wheels = list((staging / "dist").glob("*-py3-none-any.whl"))
    assert len(wheels) == 1
    return wheels[0]


def test_catalog_artifact_installs_without_runtime_or_harnesses(
    catalog_wheel: Path, tmp_path: Path
):
    metadata = pkginfo.Wheel(str(catalog_wheel))
    assert metadata.name == "nemo-fabric-adapter-catalog"
    assert not metadata.requires_dist
    with ZipFile(catalog_wheel) as archive:
        names = archive.namelist()
        expected = json.loads(archive.read("nemo_fabric_adapter_catalog/catalog.json"))
        assert "nemo_fabric_adapter_catalog/py.typed" in names
        assert not any(
            name.startswith("share/") or ".data/data/share/" in name for name in names
        )
    assert expected["catalog_version"] == metadata.version
    environment = tmp_path / "consumer"
    subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(environment)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "--no-index",
            str(catalog_wheel),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    probe = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            """
import json
from importlib.metadata import distributions
from importlib.resources import files
import nemo_fabric_adapter_catalog as catalog
assert {d.metadata['Name'] for d in distributions()} == {'nemo-fabric-adapter-catalog'}
bundle = json.loads(files(catalog).joinpath('catalog.json').read_text(encoding='utf-8'))
for kind, getter in [('adapters', catalog.get_adapter_descriptor), ('targets', catalog.get_target_descriptor)]:
    assert bundle[kind]
    for record_id, record in bundle[kind].items():
        assert getter(record_id) == record['descriptor']
print(json.dumps(bundle))
""",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(probe.stdout) == expected


@pytest.mark.parametrize("tag", ["v0.5.0-alpha.20261007", "v0.5.0-rc.1", "v0.5.0"])
def test_catalog_wheel_is_dispatched_to_kitmaker(
    catalog_wheel: Path, tmp_path: Path, tag: str
):
    wheels = tmp_path / "collected/wheels"
    wheels.mkdir(parents=True)
    wheel = Path(shutil.copy2(catalog_wheel, wheels))
    os.environ.update(
        CI_PROJECT_DIR=str(tmp_path),
        CI_COMMIT_TAG=tag,
        NEMO_FABRIC_CI_ARTIFACTORY_PYPI_URL="https://artifactory.example.test/wheels",
        NEMO_FABRIC_CI_ARTIFACTORY_USER="test-user",
        NEMO_FABRIC_CI_ARTIFACTORY_KEY="test-key",
        KITMAKER_URL="https://kitmaker.example.test",
        KITMAKER_API_TOKEN="test-token",
        KITMAKER_OWNER="test-owner",
    )
    projects = MagicMock(spec=requests.Response)
    projects.json.return_value = [{"name": "nemo-fabric-adapter-catalog", "id": 42}]
    release = MagicMock(spec=requests.Response)
    release.json.return_value = {
        "status": "pending",
        "project_id": 42,
        "release_uuid": "test-release",
        "message": "accepted",
    }
    with (
        patch.object(
            artifactory_upload.requests,
            "put",
            return_value=MagicMock(spec=requests.Response),
        ) as mock_put,
        patch.object(artifactory_upload.requests, "get", return_value=projects),
        patch.object(
            artifactory_upload.requests, "post", return_value=release
        ) as mock_post,
    ):
        assert artifactory_upload.main() == 0
    url = f"https://artifactory.example.test/wheels/{wheel.name}"
    assert mock_put.call_args.args == (url,)
    mock_post.assert_called_once_with(
        "https://kitmaker.example.test/api/v0/projects/42/releases",
        headers={"Authorization": "Bearer test-token"},
        json={
            "project_name": "nemo-fabric-adapter-catalog",
            "payload": [
                {
                    "pic": "test-owner",
                    "job_type": "wheel-release-job",
                    "url": url,
                    "upload": True,
                }
            ],
        },
        timeout=(30, 600),
    )
