# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-turn isolation demonstration for the streaming POC.

Runs **two** `invoke_stream` turns on **one** persistent streaming runtime and
shows there is no cross-turn leakage: each turn's live ATOF records are disjoint
(distinct `uuid`s), turn 1's answer never appears in turn 2's stream and vice
versa. This substantiates the "one listener per runtime; turns delimited by invoke
completion" claim in ../implementation-spec.md.

Writes an artifact (default: ../two-turn-isolation.jsonl) — one
`{"turn": n, "record": <raw ATOF>}` line per streamed record, then a final
`{"summary": {...}}` line with the per-turn counts, uuid-overlap check, and the
sentinel-leakage check.

Run (Claude subscription/SSO path — no API key; see claude/findings.md):
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
# Distinct sentinels so leakage is trivially detectable in the text.
TURNS = [
    ("ALPHA", "Reply with exactly one word: ALPHA. Nothing else."),
    ("BRAVO", "Reply with exactly one word: BRAVO. Nothing else."),
]


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


def _text_blob(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False)


async def main() -> int:
    fabric = Fabric()
    sruntime = await start_streaming_runtime(fabric, build_config(), base_dir=str(WORK))
    per_turn_uuids: list[set[str]] = []
    per_turn_records: list[list[dict]] = []
    try:
        with OUT.open("w") as fh:
            for n, (_sentinel, prompt) in enumerate(TURNS, start=1):
                uuids: set[str] = set()
                records: list[dict] = []
                stream = sruntime.invoke_stream(input=prompt)  # SAME persistent runtime
                async for rec in stream:
                    records.append(rec)
                    if rec.get("uuid"):
                        uuids.add(rec["uuid"])
                    fh.write(json.dumps({"turn": n, "record": rec}, ensure_ascii=False) + "\n")
                await stream.result()  # terminal, out of band; turn fully drained
                per_turn_uuids.append(uuids)
                per_turn_records.append(records)
                print(f"turn {n}: {len(records)} records, {len(uuids)} uuids", flush=True)

            # Isolation checks
            overlap = per_turn_uuids[0] & per_turn_uuids[1]
            t1_blob = "\n".join(_text_blob(r) for r in per_turn_records[0])
            t2_blob = "\n".join(_text_blob(r) for r in per_turn_records[1])
            leak = {
                "ALPHA_in_turn2": "ALPHA" in t2_blob,
                "BRAVO_in_turn1": "BRAVO" in t1_blob,
            }
            summary = {
                "runtime": "single persistent runtime, two invoke_stream turns",
                "turn1_records": len(per_turn_records[0]),
                "turn2_records": len(per_turn_records[1]),
                "uuid_overlap_count": len(overlap),
                "sentinel_leakage": leak,
                "isolated": len(overlap) == 0 and not any(leak.values()),
            }
            fh.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
    finally:
        await sruntime.aclose()

    print(json.dumps(summary, indent=2))
    print(f"\nartifact -> {OUT}")
    return 0 if summary["isolated"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
