# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional Pydantic interoperability for adapter contract dataclasses."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel
from pydantic import TypeAdapter

from nemo_fabric_adapter_contract.codec import JsonValue
from nemo_fabric_adapter_contract.codec import json_mapping
from nemo_fabric_adapter_contract.models import AgentContractBlock


_T = TypeVar("_T")


def type_adapter(model: type[_T]) -> TypeAdapter[_T]:
    """Return a Pydantic adapter for one canonical contract dataclass."""

    return TypeAdapter(model)


def extension_schema(model: type[BaseModel]) -> dict[str, JsonValue]:
    """Return a JSON-safe schema for one descriptor extension point."""

    return json_mapping(model.model_json_schema(mode="validation"))


def set_pydantic_extensions(
    block: _T,
    value: BaseModel,
) -> _T:
    """Set one typed Pydantic extension model on a contract block."""

    if not isinstance(block, AgentContractBlock):
        raise TypeError("block must be an AgentContractBlock")
    block.set_extensions(value.model_dump(mode="json", exclude_none=True))
    return block
