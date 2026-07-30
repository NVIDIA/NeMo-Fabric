# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

CI_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "ci"
sys.path.insert(0, str(CI_SCRIPTS))

import artifactory_upload  # noqa: E402


def test_perform_release_logs_release_identifiers(capsys):
    os.environ["KITMAKER_URL"] = "https://kitmaker.example.com"
    os.environ["KITMAKER_API_TOKEN"] = "token"
    os.environ["KITMAKER_OWNER"] = "owner"

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
        artifactory_upload.perform_release(
            [(Path("nemo_fabric-0.2.0-py3-none-any.whl"), "https://artifactory.example.com/wheel")]
        )

    assert (
        "Release accepted for nemo-fabric: project_id=3801, "
        "release_uuid=579242a9-a143-43ca-b519-c89ffc394c44"
    ) in capsys.readouterr().out
