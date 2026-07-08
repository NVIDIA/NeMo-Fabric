# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport contracts for the Harbor integration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

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


class HarborRunSpec(BaseModel):
    """Host-to-environment specification for one Harbor agent run."""

    model_config = ConfigDict(extra="forbid")

    config_path: Path
    request: RunRequest
    model_name: str | None = None
    skills_dir: Path | None = None
    mcp_servers: tuple[HarborMcpServer, ...] = ()
