# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan or run the email-phishing custom-agent example."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from nemo_fabric import Fabric

from examples.langgraph_custom_agent.consumer.config import frontier_config
from examples.langgraph_custom_agent.consumer.config import public_config
from examples.langgraph_custom_agent.consumer.config import (
    with_continuation_history_limit,
)
from examples.langgraph_custom_agent.consumer.config import with_relay
from examples.langgraph_custom_agent.consumer.config import with_system_instruction
from examples.langgraph_custom_agent.consumer.config import with_temperature
from examples.langgraph_custom_agent.consumer.config import with_url_inspector_mcp


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("public", "frontier"), default="public")
    parser.add_argument("--model")
    parser.add_argument("--system-instruction")
    parser.add_argument(
        "--system-instruction-mode",
        choices=("replace", "append"),
        default="replace",
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument(
        "--max-history-entries",
        type=int,
        help="Maximum completed assessments retained by the live runtime.",
    )
    parser.add_argument("--mcp", action="store_true")
    parser.add_argument("--relay", action="store_true")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--plan", action="store_true")
    parser.add_argument(
        "--input",
        default=(
            "Urgent: your account is locked. Verify your password immediately at "
            "https://example.invalid."
        ),
    )
    parser.add_argument(
        "--follow-up",
        help="Run a second invocation on the same live Fabric runtime.",
    )
    args = parser.parse_args()

    config_factory = frontier_config if args.variant == "frontier" else public_config
    config = config_factory(args.model) if args.model else config_factory()
    if args.system_instruction:
        config = with_system_instruction(
            config,
            args.system_instruction,
            mode=args.system_instruction_mode,
        )
    if args.temperature is not None:
        config = with_temperature(config, args.temperature)
    if args.max_history_entries is not None:
        config = with_continuation_history_limit(
            config,
            args.max_history_entries,
        )
    if args.mcp:
        config = with_url_inspector_mcp(config)
    if args.relay:
        config = with_relay(config)
    fabric = Fabric()
    if args.plan:
        output = fabric.plan(config, base_dir=args.base_dir).to_mapping()
    elif args.follow_up is None:
        output = (
            await fabric.run(config, base_dir=args.base_dir, input=args.input)
        ).to_mapping()
    else:
        async with await fabric.start_runtime(
            config,
            base_dir=args.base_dir,
        ) as runtime:
            first = await runtime.invoke(input=args.input)
            second = await runtime.invoke(input=args.follow_up)
        output = {
            "invocations": [first.to_mapping(), second.to_mapping()],
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
