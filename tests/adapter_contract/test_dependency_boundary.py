# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that the canonical Python adapter contract does not require Pydantic."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_agent_config_import_and_round_trip_without_pydantic():
    script = """
import sys

class BlockPydantic:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pydantic" or fullname.startswith("pydantic."):
            raise ModuleNotFoundError("Pydantic import is blocked")
        return None

sys.meta_path.insert(0, BlockPydantic())

from nemo_fabric_adapter_contract.models import AgentConfig

value = {"harness": {"settings": {"profile": "dependency-free"}}}
assert AgentConfig.from_mapping(value).to_mapping() == value
assert "pydantic" not in sys.modules
"""
    env = os.environ.copy()
    contract_source = str(ROOT / "adapter-contract" / "src")
    current_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{contract_source}{os.pathsep}{current_path}"
        if current_path
        else contract_source
    )

    subprocess.run([sys.executable, "-c", script], check=True, env=env)
