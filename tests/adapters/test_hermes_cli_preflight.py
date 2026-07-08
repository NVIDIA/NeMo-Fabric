# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
import yaml

from nemo_fabric import Fabric

@pytest.mark.parametrize("api_key_set", [True, False])
async def test_preflight_api_key_e2e(hermes_agent_dir: Path, hermes_cli_profile: str, api_key_set: bool):
    config_path = hermes_agent_dir / "agent.yaml"
    
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    
    config["models"]["default"]["api_key_env"] = "FAB_CI_FAKE_KEY"

    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)
    
    if api_key_set:
        os.environ["FAB_CI_FAKE_KEY"] = "fake-key"
    else:
        assert "FAB_CI_FAKE_KEY" not in os.environ, "FAB_CI_FAKE_KEY should not be set in the environment for this test"
    

    result = await Fabric().run(
        hermes_agent_dir,
        profiles=[hermes_cli_profile],
        input="who are you?",
    )
    if api_key_set:
        assert result["status"] == "succeeded"
    else:
        assert result["status"] == "failed"
        assert "api_key_env=FAB_CI_FAKE_KEY is defined in the configuration but is not set in the environment" in result["error"]["message"]
