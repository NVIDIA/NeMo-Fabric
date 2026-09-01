# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import nemo_fabric_adapters.common.instructions as common_instructions
import pytest
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapters.common import lifecycle


def _config(mode: str) -> AgentConfig:
    return AgentConfig.from_mapping(
        {
            "instructions": {
                "system": {
                    "content": "Follow repository policy.",
                    "mode": mode,
                }
            }
        }
    )


@pytest.mark.parametrize("mode", ["replace", "append"])
def test_system_instruction_returns_supported_mode(mode: str):
    instruction = common_instructions.system_instruction(
        _config(mode),
        adapter="Test adapter",
        supported_modes={"replace", "append"},
    )

    assert instruction is not None
    assert instruction.mode == mode


def test_system_instruction_reports_exact_unsupported_mode():
    with pytest.raises(lifecycle.LifecycleError) as caught:
        common_instructions.system_instruction(
            _config("append"),
            adapter="Test adapter",
            supported_modes={"replace"},
        )

    assert caught.value.code == "unsupported_system_instruction_mode"
    assert caught.value.metadata == {
        "field": "instructions.system.mode",
        "mode": "append",
        "supported_modes": ["replace"],
    }
