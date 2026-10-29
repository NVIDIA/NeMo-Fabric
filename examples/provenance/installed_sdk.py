# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the public SDK against the repository's model-free conformance adapter."""

import argparse
import asyncio
import json
from pathlib import Path

from nemo_fabric import Fabric, FabricConfig

ROOT = Path(__file__).resolve().parents[2]


async def run(output: Path):
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "installed-sdk-provenance"},
            "harness": {
                "adapter_id": "test.fabric.hermes_shim",
                "resolution": "preinstalled",
            },
            "discovery": {"local_paths": ["adapters"]},
            "runtime": {"artifacts": str(output.parent / "installed-sdk-artifacts")},
        }
    )
    result = await Fabric().run(
        config,
        base_dir=ROOT / "tests/fixtures/hermes-shim-agent",
        input="installed SDK software snapshot",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_mapping(), indent=2) + "\n")
    if result.status != "succeeded":
        raise RuntimeError(result.error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.output.resolve()))
