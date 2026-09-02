# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared system-instruction mode validation for adapter hosts."""

from __future__ import annotations

from collections.abc import Collection
from typing import Literal

from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentInstructionConfig
from nemo_fabric_adapters.common import lifecycle


def system_instruction(
    config: AgentConfig,
    *,
    adapter: str,
    supported_modes: Collection[Literal["replace", "append"]],
) -> AgentInstructionConfig | None:
    """Return the configured instruction after validating adapter support."""

    instruction = config.instructions.system if config.instructions else None
    if instruction is None:
        return None

    supported = sorted(set(supported_modes))
    if instruction.mode not in supported:
        raise lifecycle.LifecycleError(
            "unsupported_system_instruction_mode",
            f"{adapter} does not support instructions.system.mode="
            f"{instruction.mode!r}; supported modes: {', '.join(supported)}",
            metadata={
                "field": "instructions.system.mode",
                "mode": instruction.mode,
                "supported_modes": supported,
            },
        )
    return instruction
