# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common validation and serialization for shared value contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Self

from nemo_fabric_evidence.codec import (
    ContractValidationError,
    decode_dataclass,
    decode_field,
    encode_dataclass,
    validate_dataclass,
)


@dataclass(slots=True, kw_only=True)
class ContractModel:
    """Base for shared contract dataclasses."""

    # Pydantic reads this metadata only when its optional TypeAdapter is used.
    # It does not require importing Pydantic in the base contract package.
    __pydantic_config__: ClassVar[dict[str, Any]] = {
        "extra": "forbid",
        "allow_inf_nan": False,
    }

    def __post_init__(self) -> None:
        validate_dataclass(self)
        self._validate()

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in self.__dataclass_fields__:
            raise AttributeError(f"{type(self).__name__} has no field {name!r}")
        try:
            previous = getattr(self, name)
        except AttributeError:
            object.__setattr__(self, name, value)
            return

        decoded = decode_field(self, name, value)
        object.__setattr__(self, name, decoded)
        try:
            self._validate()
        except ContractValidationError:
            object.__setattr__(self, name, previous)
            raise

    def _validate(self) -> None:
        """Validate constraints that are more specific than field types."""

    @classmethod
    def from_mapping(cls, value: Any) -> Self:
        """Validate and decode one closed contract mapping."""

        return decode_dataclass(cls, value)

    def to_mapping(self) -> dict[str, Any]:
        """Return a detached JSON-compatible contract mapping."""

        return encode_dataclass(self)
