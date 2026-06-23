# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in integration smoke for the SDK multi-turn Session path (real Hermes).

Drives ``FabricClient.start -> invoke -> invoke -> stop`` against the Hermes SDK
adapter and asserts the session carries conversation memory across turns
(stateless multi-turn via history replay).

Unlike ``smoke_hermes_sdk.py`` (which shells out to the CLI), the session path is
SDK-only and runs the inline adapter in-process, so this must be executed by an
interpreter that has BOTH the nemo_fabric native extension and Hermes importable:

    RUN_FABRIC_HERMES_INTEGRATION=1 NVIDIA_API_KEY=... \\
        <hermes-venv>/bin/python tests/smoke_hermes_session.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "src"))


def main() -> None:
    if os.environ.get("RUN_FABRIC_HERMES_INTEGRATION") != "1":
        print("skipping: set RUN_FABRIC_HERMES_INTEGRATION=1 to run")
        return
    if not os.environ.get("NVIDIA_API_KEY"):
        raise SystemExit("NVIDIA_API_KEY is required")
    if importlib.util.find_spec("nemo_fabric._native") is None:
        print(
            "skipping: the SDK session path needs the nemo_fabric native extension "
            "(pip install -e . into this interpreter)"
        )
        return
    if importlib.util.find_spec("run_agent") is None:
        print(
            "skipping: Hermes (run_agent) is not importable; run with the Hermes "
            "venv python (set HERMES_PYTHON or invoke it directly)"
        )
        return
    asyncio.run(_run())


async def _run() -> None:
    from nemo_fabric import FabricClient, SessionStatus

    agent = str(ROOT / "examples" / "code-review-agent")
    async with await FabricClient().start(agent, profile="hermes_session") as session:
        assert session.status is SessionStatus.ACTIVE, session.status

        r1 = await session.invoke("My name is Robin. Please remember it for later.")
        assert r1["status"] == "succeeded", r1
        after_turn1 = session.messages
        assert len(after_turn1) >= 2, after_turn1

        r2 = await session.invoke("What is my name? Reply with just the name.")
        assert r2["status"] == "succeeded", r2
        # Transcript must grow (history accumulated across turns).
        assert len(session.messages) > len(after_turn1), session.messages
        # And the model must recall the name supplied in turn 1.
        response = (r2["output"].get("response") or "").lower()
        assert "robin" in response, response

    assert session.status is SessionStatus.STOPPED, session.status
    print("smoke_hermes_session ok")


if __name__ == "__main__":
    main()
