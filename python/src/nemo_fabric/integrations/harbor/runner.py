# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the Fabric SDK inside a Harbor task environment."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import yaml

from nemo_fabric import Fabric, FabricConfig, RunRequest


def load_sources(
    spec: dict[str, Any],
) -> tuple[FabricConfig, list[dict[str, Any]]]:
    config_path = Path(spec["config_path"])
    config = FabricConfig.from_mapping(load_yaml(config_path))
    profiles = [
        load_yaml(Path(path))
        for path in spec.get("profile_paths", [])
    ]

    model_name = (spec.get("request", {}).get("context") or {}).get("model_name")
    if isinstance(model_name, str) and model_name:
        provider = model_name.split("/", maxsplit=1)[0] if "/" in model_name else "openai"
        profiles.append(
            {
                "schema_version": "fabric.profile/v1alpha1",
                "name": "harbor_model",
                "models": {
                    "default": {
                        "provider": provider,
                        "model": model_name,
                    }
                },
            }
        )
    return config, profiles


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML object in {path}")
    return value


async def run(spec: dict[str, Any]) -> dict[str, Any]:
    config, profiles = load_sources(spec)
    request = RunRequest.model_validate(spec.get("request", {}))
    result = await Fabric().run(
        config,
        profiles=profiles,
        base_dir=Path(spec["config_path"]).parent,
        request=request,
    )
    return result.to_mapping()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    result = asyncio.run(run(spec))
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
