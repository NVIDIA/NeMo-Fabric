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
    claude_config,
    codex_config,
    deepagents_config,
    hermes_config,
    nooa_config,
    pi_config,
    with_relay,
    with_skill_paths,
)

CONFIG_BUILDERS: dict[str, Callable[[], FabricConfig]] = {
    "hermes": hermes_config,
    "claude": claude_config,
    "codex": codex_config,
    "deepagents": deepagents_config,
    "nooa": nooa_config,
    "pi": pi_config,
}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=CONFIG_BUILDERS, default="hermes")
    parser.add_argument("--relay", action="store_true")
    skill_group = parser.add_mutually_exclusive_group()
    skill_group.add_argument(
        "--skill-path",
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "Replace the variant's default skills with this path; repeat to "
            "configure multiple skills. Paths resolve from the example directory."
        ),
    )
    skill_group.add_argument(
        "--no-skills",
        action="store_true",
        help="Remove the variant's default skills.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Collect Relay ATOF records and print them with the terminal result.",
    )
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
    if args.stream and not args.relay:
        parser.error("--stream requires --relay")
    if args.stream and args.plan:
        parser.error("--stream cannot be combined with --plan")

    if args.variant == "pi" and args.relay:
        parser.error("the Pi adapter does not support Relay yet")

    config = CONFIG_BUILDERS[args.variant]()
    if args.skill_path is not None:
        config = with_skill_paths(config, *args.skill_path)
    elif args.no_skills:
        config = with_skill_paths(config)
    if args.relay:
        config = with_relay(config)

    fabric = Fabric()
    result = None
    if args.plan:
        output = fabric.plan(config, base_dir=BASE_DIR)
    elif args.stream:
        async with await fabric.start_runtime(
            config,
            base_dir=BASE_DIR,
            streaming=True,
        ) as runtime:
            stream = runtime.invoke_stream(input=args.input)
            records = [record async for record in stream]
            result = await stream.result()
        output = {
            "atof_records": records,
            "result": result.to_mapping(),
        }
    else:
        result = await fabric.run(config, base_dir=BASE_DIR, input=args.input)
        output = result
    mapped_output = output.to_mapping() if hasattr(output, "to_mapping") else output
    print(json.dumps(mapped_output, indent=2))

    if args.show_output and not args.plan:
        assert result is not None
        response = getattr(result.output, "response", None)
        if response is not None:
            print(f"\n{response}")
        elif result.error is not None:
            print(f"\n{result.error.message}")
        else:
            print("\n(run succeeded but output has no 'response' field)")


if __name__ == "__main__":
    asyncio.run(main())
