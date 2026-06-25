# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml


def update_hermes_cli_relay_base_url(code_review_agent_dir: Path, api_server: str) -> None:
    profile_path = code_review_agent_dir / "profiles" / "hermes-cli-relay.yaml"
    profile = yaml.safe_load(profile_path.read_text())
    profile["harness"]["settings"]["base_url"] = f"{api_server}/v1"
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False))
