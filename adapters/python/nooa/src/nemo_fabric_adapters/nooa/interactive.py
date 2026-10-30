# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Types shared by the ``InteractiveAgent`` adapter and its target factories."""

from __future__ import annotations

from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemo_fabric_adapter_contract.models import AgentConfig


@dataclass(frozen=True, slots=True)
class InteractiveAgentBuildContext:
    """Fabric-owned values supplied to one registered target factory."""

    config: AgentConfig
    models: Mapping[str, Any]
    system_instruction: str | None
    settings: Mapping[str, Any]
    skill_paths: tuple[Path, ...]
    runtime_id: str
    base_dir: Path
    workspace: Path
    artifact_root: Path | None


@dataclass(slots=True)
class InteractiveAgentTarget:
    """An interactive agent with optional target-owned lifecycle policy."""

    agent: Any
    close: Callable[[], Awaitable[None] | None] | None = None
    continue_after: Callable[[Any, str, str], Awaitable[bool] | bool] | None = None
