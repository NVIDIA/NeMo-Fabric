# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the code-review agent example."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Callable

from nemo_fabric import Fabric, FabricConfig

from examples.code_review_agent.config import (
    BASE_DIR,
    codex_cli_config,
    hermes_cli_config,
    hermes_sdk_config,
    with_relay,
)

CONFIG_BUILDERS: dict[str, Callable[[], FabricConfig]] = {
    "hermes-sdk": hermes_sdk_config,
    "hermes-cli": hermes_cli_config,
    "codex-cli": codex_cli_config,
}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=CONFIG_BUILDERS, default="hermes-sdk")
    parser.add_argument("--relay", action="store_true")
    parser.add_argument("--input", default="Review the workspace changes.")
    args = parser.parse_args()

    config = CONFIG_BUILDERS[args.variant]()
    if args.relay:
        config = with_relay(config)

    result = await Fabric().run(config, base_dir=BASE_DIR, input=args.input)
    print(json.dumps(result.to_mapping(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
