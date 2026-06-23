# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quickstart: a multi-turn Fabric session (start -> invoke -> stream -> stop).

The session replays the accumulated transcript as conversation history on each
turn, so the harness remembers prior turns.

Run it with an interpreter that has the ``nemo_fabric`` native binding and Hermes
installed, with an API key available:

    set -a; . ./.env; set +a          # provides NVIDIA_API_KEY
    <hermes-venv>/bin/python examples/session_quickstart.py

For a zero-setup local check of the session mechanics (no native binding, no
Hermes, no API key), run the unit smoke instead:

    python3 python/tests/smoke_sdk_sessions.py
"""

from __future__ import annotations

import asyncio

from nemo_fabric import FabricClient


async def main() -> None:
    async with await FabricClient().start(
        "examples/code-review-agent", profile="hermes_sdk"
    ) as session:
        print(f"session {session.id} [{session.status.value}]")

        result = await session.invoke("My name is Robin. Please remember it for later.")
        print(f"\n> remember my name\n  {(result.get('output') or {}).get('response')}")

        # stream() yields events as they arrive, then the final RunResult (last item).
        print("\n> what is my name? (streamed)")
        async for item in session.stream("What is my name? Reply with just the name."):
            if "status" in item:  # terminal RunResult
                print(f"  = {(item.get('output') or {}).get('response')}")
            else:  # incremental event
                print(f"  . {item.get('kind')}: {item.get('message')}")

        print(f"\ntranscript turns accumulated: {len(session.messages)}")
    print(f"\nsession [{session.status.value}] after context exit")


if __name__ == "__main__":
    asyncio.run(main())
