# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one real harness through `invoke_stream` and save its raw ATOF stream.

Usage:
    run_harness.py <adapter_id> <out.jsonl> [input]

Env:
    FABRIC_RELAY_CLI   path to the nemo-relay gateway CLI (>=0.6.0) for Claude/Codex
    FABRIC_INPUT       default prompt if [input] not given

Prereqs: a correctly built native extension (see streaming-poc/README.md) and
provider creds (NVIDIA_API_KEY in-process; OPENAI_API_KEY Codex; ANTHROPIC_API_KEY
Claude). Every ATOF record streamed live is written to <out.jsonl>.
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

ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "nvidia.fabric.hermes"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "out.atof.jsonl")
INPUT = (
    sys.argv[3]
    if len(sys.argv) > 3
    else os.environ.get("FABRIC_INPUT", "Reply with a one-sentence greeting.")
)
WORK = Path(__file__).resolve().parent.parent / ".work" / ADAPTER.split(".")[-1]


def _model() -> ModelConfig:
    override = os.environ.get("FABRIC_MODEL")  # optional model-id override
    if "codex" in ADAPTER:  # Relay gateway requires the built-in openai provider.
        # Point the gateway upstream elsewhere (e.g. NVIDIA inference) with
        # NEMO_RELAY_OPENAI_BASE_URL + a matching OPENAI_API_KEY value.
        return ModelConfig(
            provider="openai",
            model=override or "gpt-4o-mini",
            api_key_env="OPENAI_API_KEY",
        )
    if "claude" in ADAPTER:
        return ModelConfig(
            provider="anthropic",
            model="claude-3-5-haiku-latest",
            api_key_env="ANTHROPIC_API_KEY",
        )
    return ModelConfig(provider="nvidia", model="nvidia/nemotron-3-nano-30b-a3b")


def build_config() -> FabricConfig:
    for sub in ("artifacts", "workspace", "relay", "home"):
        (WORK / sub).mkdir(parents=True, exist_ok=True)
    settings: dict = {}
    if ADAPTER.endswith("hermes"):
        settings = {"hermes_home": str(WORK / "home"), "terminal_timeout": 120}
    relay_cli = os.environ.get("FABRIC_RELAY_CLI")
    if relay_cli and ("codex" in ADAPTER or "claude" in ADAPTER):
        settings["nemo_relay_command"] = relay_cli
        # The gateway subprocess env is an allowlist, so forward upstream overrides
        # (e.g. point OpenAI/Anthropic at NVIDIA inference) via harness.settings.env.
        gw_env = {
            k: os.environ[k]
            for k in ("NEMO_RELAY_OPENAI_BASE_URL", "NEMO_RELAY_ANTHROPIC_BASE_URL")
            if os.environ.get(k)
        }
        if gw_env:
            settings["env"] = gw_env
    cfg = FabricConfig(
        metadata=MetadataConfig(name="stream-poc"),
        harness=HarnessConfig(
            adapter_id=ADAPTER, resolution="preinstalled", settings=settings
        ),
        runtime=RuntimeConfig(
            input_schema="text",
            output_schema="message",
            artifacts=str(WORK / "artifacts"),
        ),
        environment=EnvironmentConfig(
            provider="local",
            workspace=str(WORK / "workspace"),
            artifacts=str(WORK / "artifacts"),
        ),
    )
    cfg.models["default"] = _model()
    cfg.enable_relay(
        observability=RelayObservabilityConfig(
            atof=RelayAtofConfig(enabled=True, output_directory=str(WORK / "relay"))
        )
    )
    return cfg


async def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fabric = Fabric()
    sruntime = await start_streaming_runtime(fabric, build_config(), base_dir=str(WORK))
    n = 0
    result = None
    try:
        stream = sruntime.invoke_stream(input=INPUT)
        with OUT.open("w") as fh:
            async for rec in stream:
                n += 1
                fh.write(json.dumps(rec) + "\n")
                print(f"  {rec.get('kind')}/{rec.get('name')}", flush=True)
        result = await stream.result()
    finally:
        await sruntime.aclose()

    status = (
        result.get("status")
        if isinstance(result, dict)
        else getattr(result, "status", None)
    )
    print(f"\n{ADAPTER}: {n} raw ATOF records -> {OUT}  (status={status})", flush=True)
    if n == 0:
        errs = list((WORK / "artifacts").rglob("stderr.txt"))
        if errs:
            print(
                "[no events] adapter stderr tail:\n  "
                + "\n  ".join(errs[0].read_text().splitlines()[-6:])
            )
    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
