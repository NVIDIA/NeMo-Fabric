# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the Fabric SDK inside a Harbor task environment."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, cast

import yaml

from nemo_fabric import Fabric, FabricConfig, ModelConfig, RunResult
from nemo_fabric.integrations.harbor.models import HarborRunSpec


def load_config(path: Path) -> FabricConfig:
    """Load one complete Fabric config from the task environment."""

    return FabricConfig.model_validate(load_yaml(path))


def compose_config(base: FabricConfig, spec: HarborRunSpec) -> FabricConfig:
    """Apply Harbor-owned values to an independent config copy."""

    config = base.model_copy(deep=True)
    if spec.model_name:
        config.models["default"] = ModelConfig(
            provider=model_provider(spec.model_name),
            model=spec.model_name,
        )

    config.mcp = None
    for server in spec.mcp_servers:
        if server.transport == "stdio":
            config.add_mcp_server(
                server.name,
                transport="stdio",
                url=cast(str, server.command),
                exposure="harness_native",
                extra_fields={"args": list(server.args)},
            )
        else:
            config.add_mcp_server(
                server.name,
                transport=server.transport,
                url=cast(str, server.url),
                exposure="harness_native",
            )

    config.skills = None
    if spec.skills_dir is not None:
        config.add_skill_path(spec.skills_dir)
    return config


def model_provider(model_name: str) -> str:
    """Derive the provider prefix used by the Fabric model config."""

    return model_name.split("/", maxsplit=1)[0] if "/" in model_name else "openai"


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML object in {path}")
    return value


async def run(spec: HarborRunSpec) -> RunResult:
    base = load_config(spec.config_path)
    config = compose_config(base, spec)
    result = await Fabric().run(
        config,
        base_dir=spec.config_path.parent,
        request=spec.request,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    spec = HarborRunSpec.model_validate_json(args.spec.read_text(encoding="utf-8"))
    result = asyncio.run(run(spec))
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result.to_mapping(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
