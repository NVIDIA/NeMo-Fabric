# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport contracts for the Harbor integration."""

from __future__ import annotations

from pathlib import Path
from pathlib import PurePosixPath
from typing import Literal
from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from nemo_fabric import FabricConfig
from nemo_fabric import RunRequest


class HarborMcpServer(BaseModel):
    """One Harbor-provided MCP server."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    transport: Literal["stdio", "sse", "streamable-http"]
    url: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_target(self) -> Self:
        if self.transport == "stdio":
            if not self.command:
                raise ValueError("stdio MCP servers require command")
        elif not self.url:
            raise ValueError(f"{self.transport} MCP servers require url")
        return self


class FabricRunPayload(BaseModel):
    """Typed Fabric inputs transported into one Harbor task environment."""

    model_config = ConfigDict(extra="forbid")

    config: FabricConfig
    config_base_dir: PurePosixPath
    logs_dir: Path = Path("/logs/agent")
    request: RunRequest
