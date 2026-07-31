# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

CI_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "ci"
sys.path.insert(0, str(CI_SCRIPTS))

import artifactory_upload  # noqa: E402


@pytest.fixture(name="release_environment")
def release_environment_fixture():
    os.environ["KITMAKER_URL"] = "https://kitmaker.example.com"
    os.environ["KITMAKER_API_TOKEN"] = "token"
    os.environ["KITMAKER_OWNER"] = "owner"


@pytest.mark.usefixtures("release_environment")
def test_perform_release_logs_release_identifiers(capsys):
    projects_response = MagicMock(spec=requests.Response)
    projects_response.json.return_value = [{"name": "nemo-fabric", "id": 3801}]
    release_response = MagicMock(spec=requests.Response)
    release_response.json.return_value = {
        "message": "Release creation accepted",
        "project_id": 3801,
        "release_uuid": "579242a9-a143-43ca-b519-c89ffc394c44",
        "status": "pending",
    }
    wheel_metadata = MagicMock()
    wheel_metadata.name = "nemo-fabric"

    with (
        patch.object(artifactory_upload.requests, "get", return_value=projects_response),
        patch.object(artifactory_upload.requests, "post", return_value=release_response),
        patch.object(artifactory_upload.pkginfo, "Wheel", return_value=wheel_metadata),
    ):
        error_count = artifactory_upload.perform_release(
            [(Path("nemo_fabric-0.2.0-py3-none-any.whl"), "https://artifactory.example.com/wheel")]
        )

    assert error_count == 0
    output = capsys.readouterr().out
    assert "Release for nemo-fabric (3801)" in output
    assert "release_uuid=579242a9-a143-43ca-b519-c89ffc394c44" in output
    assert "status=pending" in output
    assert "message=Release creation accepted" in output


@pytest.mark.parametrize("status", [artifactory_upload.FAILED_STATUS, "unexpected"])
@pytest.mark.usefixtures("release_environment")
def test_perform_release_counts_status_errors_and_continues(status):
    projects_response = MagicMock(spec=requests.Response)
    projects_response.json.return_value = [
        {"name": "package-one", "id": 1},
        {"name": "package-two", "id": 2},
    ]
    error_response = MagicMock(spec=requests.Response)
    error_response.json.return_value = {
        "message": "release error",
        "project_id": 1,
        "release_uuid": "error-uuid",
        "status": status,
    }
    success_response = MagicMock(spec=requests.Response)
    success_response.json.return_value = {
        "message": "accepted",
        "project_id": 2,
        "release_uuid": "success-uuid",
        "status": "pending",
    }
    package_one_metadata = MagicMock()
    package_one_metadata.name = "package-one"
    package_two_metadata = MagicMock()
    package_two_metadata.name = "package-two"

    with (
        patch.object(artifactory_upload.requests, "get", return_value=projects_response),
        patch.object(
            artifactory_upload.requests,
            "post",
            side_effect=[error_response, success_response],
        ) as mock_post,
        patch.object(
            artifactory_upload.pkginfo,
            "Wheel",
            side_effect=[package_one_metadata, package_two_metadata],
        ),
    ):
        error_count = artifactory_upload.perform_release(
            [
                (Path("package_one-1.0-py3-none-any.whl"), "https://artifactory.example.com/one"),
                (Path("package_two-1.0-py3-none-any.whl"), "https://artifactory.example.com/two"),
            ]
        )

    assert error_count == 1
    assert mock_post.call_count == 2


@pytest.mark.usefixtures("release_environment")
def test_perform_release_counts_request_errors_and_continues(capsys):
    projects_response = MagicMock(spec=requests.Response)
    projects_response.json.return_value = [
        {"name": "package-one", "id": 1},
        {"name": "package-two", "id": 2},
    ]
    error_response = MagicMock(spec=requests.Response)
    error_response.raise_for_status.side_effect = requests.HTTPError("server error")
    success_response = MagicMock(spec=requests.Response)
    success_response.json.return_value = {
        "message": "accepted",
        "project_id": 2,
        "release_uuid": "success-uuid",
        "status": "pending",
    }
    package_one_metadata = MagicMock()
    package_one_metadata.name = "package-one"
    package_two_metadata = MagicMock()
    package_two_metadata.name = "package-two"

    with (
        patch.object(artifactory_upload.requests, "get", return_value=projects_response),
        patch.object(
            artifactory_upload.requests,
            "post",
            side_effect=[error_response, success_response],
        ) as mock_post,
        patch.object(
            artifactory_upload.pkginfo,
            "Wheel",
            side_effect=[package_one_metadata, package_two_metadata],
        ),
    ):
        error_count = artifactory_upload.perform_release(
            [
                (Path("package_one-1.0-py3-none-any.whl"), "https://artifactory.example.com/one"),
                (Path("package_two-1.0-py3-none-any.whl"), "https://artifactory.example.com/two"),
            ]
        )

    assert error_count == 1
    assert mock_post.call_count == 2
    assert "Failed to create release for package-one: server error" in capsys.readouterr().out


def test_main_combines_upload_and_release_errors(tmp_path):
    wheels_dir = tmp_path / "collected" / "wheels"
    wheels_dir.mkdir(parents=True)
    (wheels_dir / "one.whl").touch()
    (wheels_dir / "two.whl").touch()
    os.environ["CI_PROJECT_DIR"] = str(tmp_path)
    os.environ["NEMO_FABRIC_CI_ARTIFACTORY_PYPI_URL"] = "https://artifactory.example.com"
    os.environ["NEMO_FABRIC_CI_ARTIFACTORY_USER"] = "user"
    os.environ["NEMO_FABRIC_CI_ARTIFACTORY_KEY"] = "key"
    os.environ["CI_COMMIT_TAG"] = "1.0.0"

    with (
        patch.object(
            artifactory_upload,
            "upload_wheel",
            side_effect=["https://artifactory.example.com/one.whl", requests.HTTPError("upload error")],
        ),
        patch.object(artifactory_upload, "perform_release", return_value=2) as mock_perform_release,
    ):
        result = artifactory_upload.main()

    assert result == 3
    mock_perform_release.assert_called_once()
