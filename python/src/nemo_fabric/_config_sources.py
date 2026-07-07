# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent source normalization for the Fabric Python SDK."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

from nemo_fabric.errors import FabricConfigError
from nemo_fabric.types import FabricConfig, _FabricProfileConfig

PathSource = str | os.PathLike[str]
AgentSource = PathSource | FabricConfig
ProfileSource = str | Mapping[str, Any]
PathProfiles = str | Sequence[str]
TypedProfiles = Sequence[Mapping[str, Any]]


def is_config_source(value: Any) -> bool:
    return isinstance(value, FabricConfig)


def path_arg(value: Any) -> str:
    if isinstance(value, (str, os.PathLike)):
        return os.fspath(value)
    if isinstance(value, Mapping):
        raise FabricConfigError(
            "agent mappings are not accepted directly; "
            "use FabricConfig.from_mapping(...) first"
        )
    raise FabricConfigError("agent must be a path-like source or FabricConfig")


def path_profiles(profiles: PathProfiles | None) -> list[str]:
    if profiles is None:
        return []
    if isinstance(profiles, str):
        values = [profiles]
    elif isinstance(profiles, bytes):
        raise FabricConfigError("profiles must be profile names, not bytes")
    elif isinstance(profiles, Mapping):
        raise FabricConfigError("profiles must be profile names, not a mapping")
    else:
        values = list(profiles)
    if not all(isinstance(profile, str) and profile for profile in values):
        raise FabricConfigError("path profiles must contain only non-empty strings")
    return values


def config_profiles(
    profiles: TypedProfiles | None,
) -> list[_FabricProfileConfig]:
    if profiles is None:
        return []
    if isinstance(profiles, (str, bytes)):
        raise FabricConfigError(
            "FabricConfig profiles must contain profile mappings"
        )
    values = list(profiles)
    normalized: list[_FabricProfileConfig] = []
    for profile in values:
        if isinstance(profile, _FabricProfileConfig):
            normalized.append(profile)
        elif isinstance(profile, Mapping):
            normalized.append(_FabricProfileConfig.from_mapping(profile))
        else:
            raise FabricConfigError(
                "FabricConfig profiles must contain profile mappings"
            )
    return normalized


def validate_base_dir(agent: AgentSource, base_dir: PathSource | None) -> str | None:
    if not isinstance(agent, FabricConfig):
        if base_dir is not None:
            raise FabricConfigError("base_dir is only valid with a FabricConfig source")
        return None
    return None if base_dir is None else os.fspath(base_dir)


def config_json(config: FabricConfig) -> str:
    if not isinstance(config, FabricConfig):
        raise FabricConfigError("config must be a FabricConfig")
    return json.dumps(config.to_mapping())


def profiles_json(profiles: Sequence[_FabricProfileConfig]) -> str | None:
    if not profiles:
        return None
    return json.dumps([profile.to_mapping() for profile in profiles])
