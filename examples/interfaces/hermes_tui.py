# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host a managed Hermes TUI backend for manual terminal and browser interaction."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import sys
import uuid
from pathlib import Path

from nemo_fabric import Fabric, FabricConfig
from nemo_fabric import RelayAtifConfig, RelayAtofConfig, RelayAtofFileSinkConfig
from nemo_fabric import RelayObservabilityConfig


def build_config(args: argparse.Namespace, output: Path) -> FabricConfig:
    """Prepare a file-task workspace and explicit native paths with optional Relay."""
    workspace = output / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "notes.txt").write_text(
        "Check authentication, loopback binding, and Relay trajectories.\n"
    )
    interfaces = {"terminal": {"port": args.port}}
    if args.dashboard:
        interfaces["dashboard"] = {"port": args.port}
    if args.api_port:
        interfaces["api"] = {"port": args.api_port}
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "hermes-tui-example"},
            "harness": {
                "adapter_id": "nvidia.fabric.hermes",
                "settings": {"sdk": {"enabled": args.sdk}, "interfaces": interfaces},
            },
            "models": {
                "default": {
                    "provider": "openai",
                    "model": args.model,
                    "base_url": args.base_url,
                    "api_key_env": args.api_key_env,
                    "temperature": 0.0,
                    "max_tokens": 2048,
                }
            },
            "instructions": {
                "system": {
                    "mode": "replace",
                    "content": "Complete workspace tasks using file tools. Read source files before writing. Do not use shell or web tools. Summarize actual results briefly.",
                }
            },
            "tools": {"enabled": ["file"], "blocked": ["terminal"]},
            "runtime": {"max_turns": 8},
            "environment": {
                "provider": "local",
                "workspace": str(workspace),
                "artifacts": str(output / "artifacts"),
            },
        }
    )
    if not args.no_relay:
        relay = output / "relay"
        config.enable_relay(
            output_dir=str(relay),
            observability=RelayObservabilityConfig(
                atof=RelayAtofConfig(
                    enabled=True,
                    sinks=[
                        RelayAtofFileSinkConfig(
                            output_directory=str(relay / "atof"),
                            filename="events.atof.jsonl",
                            mode="append",
                        )
                    ],
                ),
                atif=RelayAtifConfig(
                    enabled=True, output_directory=str(relay / "atif")
                ),
            ),
        )
    return config


async def run(args: argparse.Namespace) -> None:
    """Own the runtime until Enter, keeping browser/terminal clients independent."""
    output = (
        args.output_dir or Path("outputs/interfaces") / f"tui-{uuid.uuid4().hex[:12]}"
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = build_config(args, output)
    print(f"Evidence directory: {output}\nStarting managed TUI backend...", flush=True)
    async with await Fabric().start_runtime(config, base_dir=Path.cwd()) as runtime:
        connection = (
            output
            / "artifacts/.fabric/hermes/runtimes"
            / runtime.runtime_id
            / "interfaces/tui.json"
        )
        (output / "runtime.json").write_text(
            json.dumps(
                {"runtime_id": runtime.runtime_id, "connection": str(connection)},
                indent=2,
            )
            + "\n"
        )
        command = shlex.join(
            [
                sys.executable,
                "-m",
                "nemo_fabric_adapters.hermes.terminal",
                "--connection",
                str(connection),
            ]
        )
        print(f"In a second terminal, run:\n{command}", flush=True)
        if args.dashboard:
            print(f"Browser: http://127.0.0.1:{args.port}", flush=True)
        print(
            f"Try: Read {output / 'workspace/notes.txt'} and write a three-item checklist to {output / 'workspace/checklist.md'} using file tools.",
            flush=True,
        )
        print(
            f"Inspect completed trajectories under {output / 'relay/atif'} while clients remain connected.",
            flush=True,
        )
        await asyncio.to_thread(
            input,
            "Press Enter here to stop the runtime (detaching a client does not stop it).\n",
        )
        await runtime.stop()
    print("Runtime stopped; attachment files removed.", flush=True)


def main() -> None:
    """Validate endpoint and port arguments before starting the interactive host."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("MODEL_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME"))
    parser.add_argument("--api-key-env", default="MODEL_API_KEY")
    parser.add_argument("--port", type=int, default=19119)
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Share the TUI backend with browser chat; requires built web assets",
    )
    parser.add_argument(
        "--sdk",
        action="store_true",
        help="Also start the embedded SDK agent; this native-only example explicitly disables it otherwise",
    )
    parser.add_argument(
        "--api-port", type=int, help="Also start a separate native API gateway"
    )
    parser.add_argument("--no-relay", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, help="New directory for workspace and evidence"
    )
    args = parser.parse_args()
    if not args.base_url or not args.model or not os.environ.get(args.api_key_env):
        parser.error(
            "provide model endpoint/name and its credential environment variable"
        )
    if not 1 <= args.port <= 65535 or (
        args.api_port is not None and not 1 <= args.api_port <= 65535
    ):
        parser.error("ports must be between 1 and 65535")
    if args.api_port == args.port:
        parser.error("API and TUI backend ports must differ")
    try:
        asyncio.run(run(args))
    except (KeyboardInterrupt, EOFError):
        pass


if __name__ == "__main__":
    main()
