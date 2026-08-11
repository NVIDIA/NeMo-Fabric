# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free encoding and validation for adapter contract dataclasses."""

from __future__ import annotations

import math
import types
from collections.abc import Mapping
from dataclasses import MISSING
from dataclasses import Field
from dataclasses import fields
from enum import Enum
from functools import cache
from pathlib import Path
from typing import Any
from typing import Literal
from typing import TypeVar
from typing import Union
from typing import get_args
from typing import get_origin
from typing import get_type_hints


# JSON's recursive type cannot be expressed as a named standard-library alias
# on every supported Python version. Contract fields marked as JSON are checked
# recursively by ``json_value`` at runtime.
JsonValue = Any

_T = TypeVar("_T")


class ContractValidationError(ValueError):
    """A southbound contract value failed dependency-free validation."""

    def __init__(self, message: str, *, path: tuple[str, ...] = ()) -> None:
        self.message = message
        self.path = path
        location = ".".join(path)
        super().__init__(f"{location}: {message}" if location else message)

    def prepend(self, path: tuple[str, ...]) -> "ContractValidationError":
        """Return the same validation failure beneath an outer field path."""

        return ContractValidationError(self.message, path=(*path, *self.path))


def json_value(value: Any, *, path: tuple[str, ...] = ()) -> JsonValue:
    """Validate, copy, and return one JSON-compatible value."""

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractValidationError("must be a finite JSON number", path=path)
        return value
    if isinstance(value, list):
        return [
            json_value(item, path=(*path, str(index)))
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractValidationError(
                    "JSON object keys must be strings", path=path
                )
            result[key] = json_value(item, path=(*path, key))
        return result
    raise ContractValidationError("must be a valid JSON value", path=path)


def json_mapping(
    value: Mapping[str, Any], *, path: tuple[str, ...] = ()
) -> dict[str, JsonValue]:
    """Validate and detach one JSON object mapping."""

    result = json_value(value, path=path)
    if not isinstance(result, dict):
        raise ContractValidationError("must be a JSON object", path=path)
    return result


def decode_dataclass(model: type[_T], value: Any, *, path: tuple[str, ...] = ()) -> _T:
    """Decode one closed mapping into a contract dataclass."""

    if isinstance(value, model):
        return value
    if not isinstance(value, Mapping):
        raise ContractValidationError("must be an object", path=path)
    if any(not isinstance(key, str) for key in value):
        raise ContractValidationError("object keys must be strings", path=path)

    model_fields = {item.name: item for item in fields(model)}
    unknown = sorted(set(value).difference(model_fields))
    if unknown:
        raise ContractValidationError(
            f"unexpected field {unknown[0]!r}",
            path=path,
        )
    missing = [
        item.name
        for item in model_fields.values()
        if item.default is MISSING
        and item.default_factory is MISSING
        and item.name not in value
    ]
    if missing:
        raise ContractValidationError(
            f"missing required field {missing[0]!r}",
            path=path,
        )

    try:
        return model(**dict(value))
    except ContractValidationError as error:
        raise error.prepend(path) from error


@cache
def _resolved_type_hints(model: type[Any]) -> dict[str, Any]:
    return get_type_hints(model)


def validate_dataclass(instance: Any) -> None:
    """Validate and normalize all declared fields on a contract dataclass."""

    annotations = _resolved_type_hints(type(instance))
    for item in fields(instance):
        value = getattr(instance, item.name)
        decoded = _decode_value(
            annotations[item.name],
            value,
            path=(item.name,),
            field=item,
        )
        object.__setattr__(instance, item.name, decoded)


def decode_field(instance: Any, name: str, value: Any) -> Any:
    """Validate and normalize one field assignment."""

    item = next(item for item in fields(instance) if item.name == name)
    annotation = _resolved_type_hints(type(instance))[name]
    return _decode_value(
        annotation,
        value,
        path=(name,),
        field=item,
    )


def encode_dataclass(instance: Any) -> dict[str, Any]:
    """Return a detached JSON-compatible mapping for a contract dataclass."""

    result: dict[str, Any] = {}
    for item in fields(instance):
        value = getattr(instance, item.name)
        if item.metadata.get("omit_none") and value is None:
            continue
        if item.metadata.get("omit_empty") and not value:
            continue
        result[item.name] = _encode_value(value, path=(item.name,))
    return result


def _decode_value(
    annotation: Any,
    value: Any,
    *,
    path: tuple[str, ...],
    field: Field[Any] | None = None,
) -> Any:
    if field is not None and field.metadata.get("json"):
        if get_origin(annotation) is dict:
            return json_mapping(value, path=path)
        return json_value(value, path=path)

    origin = get_origin(annotation)
    arguments = get_args(annotation)

    if origin in (types.UnionType, Union):
        if type(None) in arguments and value is None:
            return None
        options = tuple(option for option in arguments if option is not type(None))
        errors = []
        for option in options:
            try:
                return _decode_value(option, value, path=path)
            except ContractValidationError as error:
                errors.append(error)
        if len(errors) == 1:
            raise errors[0]
        raise ContractValidationError(
            "must match one of the declared types", path=path
        ) from errors[-1]

    if origin is Literal:
        if value not in arguments or any(
            type(value) is not type(item) for item in arguments if value == item
        ):
            allowed = ", ".join(repr(item) for item in arguments)
            raise ContractValidationError(f"must be one of: {allowed}", path=path)
        return value

    if origin is list:
        if not isinstance(value, list):
            raise ContractValidationError("must be an array", path=path)
        return [
            _decode_value(arguments[0], item, path=(*path, str(index)))
            for index, item in enumerate(value)
        ]

    if origin is dict:
        if not isinstance(value, Mapping):
            raise ContractValidationError("must be an object", path=path)
        key_type, value_type = arguments
        result = {}
        for key, item in value.items():
            decoded_key = _decode_value(key_type, key, path=(*path, str(key)))
            result[decoded_key] = _decode_value(
                value_type,
                item,
                path=(*path, str(key)),
            )
        return result

    if annotation is Any:
        return value
    if annotation is type(None):
        if value is not None:
            raise ContractValidationError("must be null", path=path)
        return None
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if isinstance(value, annotation):
            return value
        try:
            return annotation(value)
        except (TypeError, ValueError) as error:
            raise ContractValidationError(
                f"must be a valid {annotation.__name__}",
                path=path,
            ) from error
    if isinstance(annotation, type) and hasattr(annotation, "from_mapping"):
        return decode_dataclass(annotation, value, path=path)
    if annotation is Path:
        if not isinstance(value, (str, Path)):
            raise ContractValidationError("must be a path string", path=path)
        return value
    if annotation is bool:
        if type(value) is not bool:
            raise ContractValidationError("must be a boolean", path=path)
        return value
    if annotation is int:
        if type(value) is not int:
            raise ContractValidationError("must be an integer", path=path)
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractValidationError("must be a number", path=path)
        try:
            result = float(value)
        except OverflowError as error:
            raise ContractValidationError(
                "must be a finite number", path=path
            ) from error
        if not math.isfinite(result):
            raise ContractValidationError("must be a finite number", path=path)
        return result
    if annotation is str:
        if not isinstance(value, str):
            raise ContractValidationError("must be a string", path=path)
        return value
    if isinstance(annotation, type) and isinstance(value, annotation):
        return value
    raise ContractValidationError("has an unsupported declared type", path=path)


def _encode_value(value: Any, *, path: tuple[str, ...]) -> Any:
    if hasattr(value, "to_mapping"):
        return value.to_mapping()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [
            _encode_value(item, path=(*path, str(index)))
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        return {
            key: _encode_value(item, path=(*path, key)) for key, item in value.items()
        }
    return json_value(value, path=path)
