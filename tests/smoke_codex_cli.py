# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real Codex CLI smoke for Fabric one-shot and multi-turn runtimes.

    RUN_FABRIC_CODEX_INTEGRATION=1 python3 tests/smoke_codex_cli.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "src"))


def main() -> None:
    if os.environ.get("RUN_FABRIC_CODEX_INTEGRATION") != "1":
        print("skipping: set RUN_FABRIC_CODEX_INTEGRATION=1 to run")
        return
    if shutil.which("codex") is None:
        raise SystemExit("codex CLI is required")
    if importlib.util.find_spec("nemo_fabric._native") is None:
        raise SystemExit("the nemo_fabric native extension is required (pip install -e .)")
    asyncio.run(_run())


async def _run() -> None:
    from nemo_fabric import Fabric

    agent = ROOT / "examples" / "code-review-agent"
    nonce = f"fabric-{uuid.uuid4().hex[:8]}"
    async with Fabric() as client:
        oneshot = await client.run(
            agent,
            profiles=["codex_cli"],
            input="Reply with exactly: FABRIC_CODEX_ONESHOT_OK",
        )
        assert oneshot["status"] == "succeeded", oneshot.to_mapping()
        assert "fabric_codex_oneshot_ok" in oneshot["output"]["response"].lower(), (
            oneshot.to_mapping()
        )
        assert "--ephemeral" not in oneshot["output"]["command"], oneshot.to_mapping()

        async with await client.start_runtime(
            agent,
            profiles=["codex_cli"],
        ) as runtime:
            first = await runtime.invoke(input=f"Remember this value: {nonce}")
            second = await runtime.invoke(
                input="Reply with only the value I asked you to remember."
            )

        results = (first.to_mapping(), second.to_mapping())
        assert first["status"] == second["status"] == "succeeded", results
        assert first["output"]["thread_id"] == second["output"]["thread_id"], results
        assert nonce in second["output"]["response"], second.to_mapping()
        assert second["output"]["command"][-3:-1] == [
            "resume",
            first["output"]["thread_id"],
        ], second.to_mapping()

    print("smoke_codex_cli ok")


if __name__ == "__main__":
    main()
