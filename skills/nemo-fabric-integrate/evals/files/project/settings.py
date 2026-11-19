# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application-owned settings supplied to the consumer integration exercises."""

from dataclasses import dataclass, field


@dataclass
class AppSettings:
    name: str = "review-service"
    adapter_id: str = "nvidia.fabric.codex"
    provider: str = "openai"
    model: str = "example-model"
    api_key_env: str = "MODEL_API_KEY"
    base_url: str | None = None
    timeout_seconds: float = 60.0
    max_turns: int = 4
    system_instruction: str | None = None
    system_instruction_mode: str = "replace"
    enabled_tools: list[str] | None = None
    blocked_tools: list[str] = field(default_factory=list)


DEFAULT_SETTINGS = AppSettings()
