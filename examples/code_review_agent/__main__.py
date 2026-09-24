# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the code-review agent example."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any

from nemo_fabric import Fabric, FabricConfig

from examples.code_review_agent.config import (
    BASE_DIR,
    claude_config,
    codex_config,
    deepagents_config,
    hermes_config,
    nooa_config,
    openclaw_config,
    openhands_config,
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
    "openclaw": openclaw_config,
    "openhands": openhands_config,
    "pi": pi_config,
}


async def _invoke_runtimes(runtimes: list[Any], input_value: object) -> list[Any]:
    """Invoke every runtime and drain all tasks before lifecycle cleanup."""

    tasks = [
        asyncio.create_task(runtime.invoke(input=input_value)) for runtime in runtimes
    ]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=CONFIG_BUILDERS, default="hermes")
    parser.add_argument("--relay", action="store_true")
    parser.add_argument(
        "--pi-relay-extension-path",
        metavar="PATH",
        help="Path to the NeMo Relay 0.9 Pi extension file or package directory.",
    )
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
    parser.add_argument(
        "--service",
        action="store_true",
        help="Prepare one OpenClaw service and connect runtimes to it.",
    )
    parser.add_argument(
        "--runtime-count",
        type=int,
        default=1,
        help="Number of runtimes to invoke when --service is enabled.",
    )
    parser.add_argument(
        "--service-duration-seconds",
        type=float,
        default=0,
        help=(
            "Keep the OpenClaw service alive for this many seconds after its "
            "Fabric runtimes stop, for chat-channel testing."
        ),
    )
    parser.add_argument(
        "--telegram-token-env",
        metavar="ENV",
        help="Enable Telegram on an OpenClaw service using this token environment variable.",
    )
    parser.add_argument(
        "--telegram-allow-from",
        action="append",
        default=None,
        metavar="USER_ID",
        help="Allow a Telegram user ID; repeat for multiple users.",
    )
    parser.add_argument(
        "--telegram-api-root",
        metavar="URL",
        help="Override the Telegram Bot API root, primarily for local testing.",
    )
    parser.add_argument("--input", default="Review the workspace changes.")
    args = parser.parse_args()
    if args.stream and not args.relay:
        parser.error("--stream requires --relay")
    if args.stream and args.plan:
        parser.error("--stream cannot be combined with --plan")
    if args.variant == "openclaw" and args.relay:
        parser.error("the OpenClaw adapter does not support Relay telemetry")
    if args.variant == "openhands" and args.relay:
        parser.error("the OpenHands adapter does not support Relay telemetry")
    if args.service and args.variant != "openclaw":
        parser.error("--service requires --variant openclaw")
    if args.runtime_count < 1:
        parser.error("--runtime-count must be at least 1")
    if args.runtime_count != 1 and not args.service:
        parser.error("--runtime-count requires --service")
    if (
        not math.isfinite(args.service_duration_seconds)
        or args.service_duration_seconds < 0
    ):
        parser.error("--service-duration-seconds must be a finite non-negative number")
    if args.service_duration_seconds and not args.service:
        parser.error("--service-duration-seconds requires --service")
    telegram_options = (
        args.telegram_token_env,
        args.telegram_allow_from,
        args.telegram_api_root,
    )
    if any(option is not None for option in telegram_options) and not args.service:
        parser.error("Telegram options require --service")
    if (
        args.telegram_allow_from is not None or args.telegram_api_root is not None
    ) and args.telegram_token_env is None:
        parser.error(
            "Telegram allow-list and API-root options require --telegram-token-env"
        )
    if args.pi_relay_extension_path is not None and args.variant != "pi":
        parser.error("--pi-relay-extension-path requires --variant pi")
    if (
        args.variant == "pi"
        and args.relay
        and not args.plan
        and args.pi_relay_extension_path is None
    ):
        parser.error("Pi Relay runs require --pi-relay-extension-path")

    config = CONFIG_BUILDERS[args.variant]()
    if args.skill_path is not None:
        config = with_skill_paths(config, *args.skill_path)
    elif args.no_skills:
        config = with_skill_paths(config)
    if args.pi_relay_extension_path is not None:
        config.harness.settings["relay_extension_path"] = args.pi_relay_extension_path
    if args.relay:
        config = with_relay(config)
    if args.telegram_token_env is not None:
        telegram_account: dict[str, object] = {
            "botToken": {
                "source": "env",
                "provider": "default",
                "id": args.telegram_token_env,
            }
        }
        if args.telegram_allow_from is not None:
            telegram_account.update(
                dmPolicy="allowlist",
                allowFrom=args.telegram_allow_from,
            )
        if args.telegram_api_root is not None:
            telegram_account["apiRoot"] = args.telegram_api_root
        agent_id = config.harness.settings.get("agent_id", "default")
        config.harness.settings["channel_config"] = {
            "channels": {"telegram": {"accounts": {"default": telegram_account}}},
            "bindings": [
                {
                    "agentId": agent_id,
                    "match": {"channel": "telegram", "accountId": "default"},
                }
            ],
        }

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
    elif args.service:
        async with await fabric.prepare_service(config, base_dir=BASE_DIR) as service:
            if args.telegram_token_env is not None:
                print(
                    "OpenClaw Telegram channel is active. You can message the bot "
                    "while Fabric runtimes are running or during the service-only interval.",
                    file=sys.stderr,
                    flush=True,
                )
            async with AsyncExitStack() as runtime_stack:
                runtimes = [
                    await runtime_stack.enter_async_context(
                        await fabric.start_runtime(
                            config,
                            base_dir=BASE_DIR,
                            service=service,
                        )
                    )
                    for _ in range(args.runtime_count)
                ]
                results = await _invoke_runtimes(runtimes, args.input)
            if args.service_duration_seconds:
                print(
                    "NeMo Fabric runtimes stopped. "
                    f"OpenClaw service {service.service_id} remains active for "
                    f"{args.service_duration_seconds:g} seconds.",
                    file=sys.stderr,
                    flush=True,
                )
                await asyncio.sleep(args.service_duration_seconds)
        result = results[0]
        output = (
            result
            if len(results) == 1
            else {
                "service_id": service.service_id,
                "results": [item.to_mapping() for item in results],
            }
        )
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
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
