# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn-isolation demonstration for the streaming POC — the *early-exit* path.

This demonstrates isolation for **one** run; it is not a protocol guarantee. Turns
are delimited by a 250 ms drain heuristic (`fabric_stream._DRAIN`), so a trailing
record arriving after that window could still land in the next turn. A production
build needs a positive turn boundary (per-record turn attribution or a terminal
delivery acknowledgement), not a timer.

Exercises the path where the shared-listener leak would occur (not just fully
drained sequential turns):

  * **Turn 1 — early exit with unread records.** Read only the first record, then
    break and `aclose()`. `aclose()` runs the turn to completion and drains/discards
    the *unread* records so they cannot leak into the next turn.
  * **Concurrency guard.** While turn 1 is still active, a second `invoke_stream`
    must be refused (`FabricStateError`) — one active invocation per runtime.
  * **Turn 2 — full consume.** Must be clean: no turn-1 sentinel (`ALPHA`) leaks in,
    and turn-2 record `uuid`s are disjoint from the turn-1 records that were read.

Writes an artifact (default ../two-turn-isolation.jsonl): one
`{"turn": n, "phase": …, "record": <raw ATOF>}` line per streamed record, then a
`{"summary": {...}}` line with the checks. Exit 0 iff isolated.

Run (Claude subscription/SSO — no API key; see claude/findings.md):
    unset ANTHROPIC_API_KEY
    export ANTHROPIC_CONFIG_DIR="$HOME/.claude"
    export FABRIC_RELAY_CLI="$(command -v nemo-relay)"
    export FABRIC_MODEL="claude-sonnet-4-5"
    python streaming-poc/common/two_turn_isolation.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from nemo_fabric import Fabric
from nemo_fabric.errors import FabricStateError
from nemo_fabric.models import (
    EnvironmentConfig,
    FabricConfig,
    HarnessConfig,
    MetadataConfig,
    ModelConfig,
    RelayAtofConfig,
    RelayObservabilityConfig,
    RuntimeConfig,
)

from fabric_stream import start_streaming_runtime

WORK = Path(__file__).resolve().parent.parent / ".work" / "two-turn"
OUT = Path(
    os.environ.get(
        "POC_TWO_TURN_OUT",
        str(Path(__file__).resolve().parent.parent / "two-turn-isolation.jsonl"),
    )
)


def build_config() -> FabricConfig:
    for sub in ("artifacts", "workspace", "relay"):
        (WORK / sub).mkdir(parents=True, exist_ok=True)
    cfg = FabricConfig(
        metadata=MetadataConfig(name="two-turn-isolation"),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.claude",
            resolution="preinstalled",
            settings={
                "nemo_relay_command": os.environ["FABRIC_RELAY_CLI"],
                "permission_mode": "bypassPermissions",
                "max_turns": 3,
            },
        ),
        runtime=RuntimeConfig(
            input_schema="text", output_schema="message", artifacts=str(WORK / "artifacts")
        ),
        environment=EnvironmentConfig(
            provider="local", workspace=str(WORK / "workspace"), artifacts=str(WORK / "artifacts")
        ),
    )
    cfg.models["default"] = ModelConfig(
        provider="anthropic",
        model=os.environ.get("FABRIC_MODEL", "claude-sonnet-4-5"),
        api_key_env="ANTHROPIC_API_KEY",
    )
    cfg.enable_relay(
        observability=RelayObservabilityConfig(
            atof=RelayAtofConfig(enabled=True, output_directory=str(WORK / "relay"))
        )
    )
    return cfg


async def main() -> int:
    fabric = Fabric()
    sruntime = await start_streaming_runtime(fabric, build_config(), base_dir=str(WORK))
    artifact: list[dict] = []
    try:
        # --- Turn 1: early exit, leaving records unread -------------------------
        s1 = sruntime.invoke_stream(input="Reply with exactly one word: ALPHA. Nothing else.")
        t1_read: list[dict] = []
        concurrent_rejected = None
        async for rec in s1:
            t1_read.append(rec)
            artifact.append({"turn": 1, "phase": "read-before-break", "record": rec})
            # While turn 1 is still active, a second invocation must be refused.
            try:
                sruntime.invoke_stream(input="concurrent — must be refused")
                concurrent_rejected = False
            except FabricStateError:
                concurrent_rejected = True
            break  # early exit — the rest of turn 1's records are still unread
        await s1.aclose()  # finalize: run to completion + drain/discard the unread
        t1_uuids = {r.get("uuid") for r in t1_read if r.get("uuid")}

        # --- Turn 2: full consume, must be clean --------------------------------
        s2 = sruntime.invoke_stream(input="Reply with exactly one word: BRAVO. Nothing else.")
        t2: list[dict] = []
        async for rec in s2:
            t2.append(rec)
            artifact.append({"turn": 2, "phase": "full-consume", "record": rec})
        await s2.result()
        t2_uuids = {r.get("uuid") for r in t2 if r.get("uuid")}

        t2_blob = "\n".join(json.dumps(r, ensure_ascii=False) for r in t2)
        summary = {
            "scenario": "turn1 early-exit (read 1, aclose drains unread) + concurrency guard; turn2 full consume",
            "turn1_records_read_before_break": len(t1_read),
            "turn2_records": len(t2),
            "concurrent_invoke_rejected": concurrent_rejected,
            "ALPHA_leaked_into_turn2": "ALPHA" in t2_blob,
            "uuid_overlap_count": len(t1_uuids & t2_uuids),
            "isolated": (
                concurrent_rejected is True
                and "ALPHA" not in t2_blob
                and len(t1_uuids & t2_uuids) == 0
            ),
        }
        artifact.append({"summary": summary})
        with OUT.open("w") as fh:
            for line in artifact:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    finally:
        await sruntime.aclose()

    print(json.dumps(summary, indent=2))
    print(f"\nartifact -> {OUT}")
    return 0 if summary["isolated"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
