# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml


def update_base_url(profile_path: Path, api_server: str):
    """
    Update the base URL in a profile.

    Since the api_server uses a random available TCP port, the base_url needs to be updated for each test.

    Args:
        profile_path (Path): The absolute path to the profile YAML file.
        api_server (str): The API server URL.
    """
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile["harness"]["settings"]["base_url"] = f"{api_server}/v1"
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
