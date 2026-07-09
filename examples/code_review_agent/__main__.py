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
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print the resolved run plan without starting a runtime.",
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Print the adapter response after the normalized result.",
    )
    parser.add_argument("--input", default="Review the workspace changes.")
    args = parser.parse_args()

    config = CONFIG_BUILDERS[args.variant]()
    if args.relay:
        config = with_relay(config)

    fabric = Fabric()
    if args.plan:
        output = fabric.plan(config, base_dir=BASE_DIR)
    else:
        output = await fabric.run(config, base_dir=BASE_DIR, input=args.input)
    print(json.dumps(output.to_mapping(), indent=2))
    if args.show_output and not args.plan:
        if isinstance(output.output, dict) and "response" in output.output:
            print(f"\n{output.output['response']}")
        else:
            print(f"\n{output.error.message}")


if __name__ == "__main__":
    asyncio.run(main())
