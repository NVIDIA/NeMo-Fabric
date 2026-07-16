# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import helpers for code-defined Fabric configurations."""

from __future__ import annotations

import importlib

from nemo_fabric.errors import FabricConfigError
from nemo_fabric.models import FabricConfig


def load_config_factory(spec: str) -> FabricConfig:
    """Import and invoke a ``module:callable`` returning ``FabricConfig``."""

    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise FabricConfigError("factory must use module:callable syntax")
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        raise FabricConfigError(f"could not import factory module {module_name!r}: {error}") from error
    try:
        factory = getattr(module, attribute)
    except AttributeError as error:
        raise FabricConfigError(f"factory module {module_name!r} has no attribute {attribute!r}") from error
    if not callable(factory):
        raise FabricConfigError(f"factory target {spec!r} is not callable")
    try:
        config = factory()
    except Exception as error:
        raise FabricConfigError(f"factory {spec!r} failed: {error}") from error
    if not isinstance(config, FabricConfig):
        raise FabricConfigError(f"factory {spec!r} returned {type(config).__name__}; expected FabricConfig")
    return config
