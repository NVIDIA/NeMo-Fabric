# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport contracts for the Harbor integration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from nemo_fabric import RunRequest


def parse_config_factory_reference(value: str) -> tuple[str, str]:
    """Validate and split a Python ``module:callable`` reference."""

    module_name, separator, callable_name = value.partition(":")
    if (
        not separator
        or not module_name
        or not callable_name
        or not all(part.isidentifier() for part in module_name.split("."))
        or not callable_name.isidentifier()
    ):
        raise ValueError("config_factory must use module:callable syntax")
    return module_name, callable_name


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

    config_factory: str = Field(min_length=3)
    config_base_dir: Path
    logs_dir: Path = Path("/logs/agent")
    request: RunRequest
    model_name: str | None = None
    skills_dir: Path | None = None
    mcp_servers: tuple[HarborMcpServer, ...] = ()

    @field_validator("config_factory")
    @classmethod
    def validate_config_factory(cls, value: str) -> str:
        parse_config_factory_reference(value)
        return value
