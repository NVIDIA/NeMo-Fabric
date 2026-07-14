#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path
from urllib.parse import quote

import pkginfo
import requests


def upload_wheel(
    wheel_file: Path,
    artifactory_url: str,
    username: str,
    api_key: str,
) -> str:
    wheel_name = wheel_file.name
    wheel_url = f"{artifactory_url.rstrip('/')}/{quote(wheel_name, safe='/')}"
    print(f"Uploading {wheel_file} to {wheel_url}...", flush=True)
    with wheel_file.open("rb") as wheel_data:
        response = requests.put(
            wheel_url,
            data=wheel_data,
            auth=(username, api_key),
            timeout=(30, 600),
        )
    response.raise_for_status()
    return wheel_url


def perform_release(published_wheels: list[tuple[Path, str]]) -> None:
    kitmaker_url = os.environ["KITMAKER_URL"]
    kitmaker_api_token = os.environ["KITMAKER_API_TOKEN"]
    kitmaker_owner = os.environ["KITMAKER_OWNER"]
    headers = {"Authorization": f"Bearer {kitmaker_api_token}"}

    response = requests.get(
        f"{kitmaker_url}/api/v0/projects",
        headers=headers,
        timeout=(30, 600),
    )
    response.raise_for_status()
    projects = response.json()

    if len(projects) < len(published_wheels):
        print(
            f"Warning: KitMaker returned {len(projects)} projects for {len(published_wheels)} published wheels.",
            flush=True,
        )

    project_ids = {project["name"]: project["id"] for project in projects}
    for wheel_file, wheel_url in published_wheels:
        package_name = pkginfo.Wheel(str(wheel_file)).name
        package_id = project_ids[package_name]
        payload = {
            "project_name": package_name,
            "payload": [{
                "pic": kitmaker_owner,
                "job_type": "wheel-release-job",
                "url": wheel_url,
                "upload": True,
            }],
        }

        response = requests.post(
            f"{kitmaker_url}/api/v0/projects/{package_id}/releases",
            headers=headers,
            json=payload,
            timeout=(30, 600),
        )
        response.raise_for_status()


def main() -> int:
    project_dir = Path(os.environ["CI_PROJECT_DIR"])
    wheels_dir = project_dir / "collected/wheels"
    artifactory_url = os.environ["NEMO_FABRIC_CI_ARTIFACTORY_PYPI_URL"]
    username = os.environ["NEMO_FABRIC_CI_ARTIFACTORY_USER"]
    api_key = os.environ["NEMO_FABRIC_CI_ARTIFACTORY_KEY"]

    wheels = []
    published_wheels: list[tuple[Path, str]] = []
    
    print(f"Dir : {wheels_dir}", flush=True)


    for wheel_file in wheels_dir.rglob("*.whl"):
        wheels.append(wheel_file)
        try:
            wheel_url = upload_wheel(
                wheel_file,
                artifactory_url,
                username,
                api_key,
            )
            published_wheels.append((wheel_file, wheel_url))
        except Exception as e:
            print(f"Failed to upload {wheel_file}: {e}", flush=True)

    num_unpublished = len(wheels) - len(published_wheels)
    if num_unpublished > 0:
        print(f"Warning: Only {len(published_wheels)} out of {len(wheels)} wheels were uploaded successfully.",
              flush=True)
    else:
        print("All wheels uploaded to Artifactory.")

    if os.environ.get("CI_COMMIT_TAG") is not None:
        print("Performing release of published wheels to KitMaker...", flush=True)
        perform_release(published_wheels)
    else:
        print("Skipping release to KitMaker. This is not a nightly, tagged, or main branch build.", flush=True)

    return num_unpublished


if __name__ == "__main__":
    sys.exit(main())
