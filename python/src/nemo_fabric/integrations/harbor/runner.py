# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the Fabric SDK inside a Harbor task environment."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from pathlib import Path
from typing import cast

from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import ModelConfig
from nemo_fabric import RunResult
from nemo_fabric.integrations.harbor.models import HarborRunSpec
from nemo_fabric.integrations.harbor.models import parse_config_factory_reference
from nemo_fabric.integrations.harbor.telemetry import publish_telemetry_evidence


def load_config_factory(reference: str, base_dir: Path) -> FabricConfig:
    """Import and invoke one typed FabricConfig factory."""

    module_name, callable_name = parse_config_factory_reference(reference)

    resolved_base_dir = base_dir.resolve()
    if not resolved_base_dir.is_dir():
        raise FileNotFoundError(f"config_base_dir does not exist: {resolved_base_dir}")

    importlib.invalidate_caches()
    search_path = str(resolved_base_dir)
    sys.path.insert(0, search_path)
    try:
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            raise RuntimeError(f"config factory failed to import: {reference}") from error

        factory = getattr(module, callable_name, None)
        if not callable(factory):
            raise TypeError(f"config factory is not callable: {reference}")
        try:
            config = factory()
        except Exception as error:
            raise RuntimeError(f"config factory failed: {reference}") from error
    finally:
        sys.path.remove(search_path)

    if not isinstance(config, FabricConfig):
        raise TypeError(
            f"config factory must return FabricConfig, got {type(config).__name__}: {reference}"
        )
    return config


def compose_config(base: FabricConfig, spec: HarborRunSpec) -> FabricConfig:
    """Apply Harbor-owned values to an independent config copy."""

    config = base.model_copy(deep=True)
    if spec.model_name:
        current = config.models.get("default")
        values = current.to_mapping() if current is not None else {}
        provider = model_provider(spec.model_name)
        if current is not None and current.provider != provider:
            values.pop("api_key_env", None)
        values.update(
            provider=provider,
            model=spec.model_name,
        )
        config.models["default"] = ModelConfig.model_validate(values)

    if spec.mcp_servers:
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

    if spec.skills_dir is not None:
        config.skills = None
        config.add_skill_path(spec.skills_dir)
    return config


def model_provider(model_name: str) -> str:
    """Derive the provider prefix used by the Fabric model config."""

    return model_name.split("/", maxsplit=1)[0] if "/" in model_name else "openai"


async def run(spec: HarborRunSpec) -> RunResult:
    base = load_config_factory(spec.config_factory, spec.config_base_dir)
    config = compose_config(base, spec)
    result = await Fabric().run(
        config,
        base_dir=spec.config_base_dir,
        request=spec.request,
    )
    publish_telemetry_evidence(
        result,
        spec.logs_dir,
        harbor_session_id=spec.request.context.get("harbor_session_id"),
        harbor_context_id=spec.request.context.get("harbor_context_id"),
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
