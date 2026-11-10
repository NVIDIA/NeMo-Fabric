# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host one configured OpenClaw Gateway for SDK, HTTP, browser, and terminal clients."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
from nemo_fabric import Fabric, FabricConfig


def build_config(args: argparse.Namespace, output: Path) -> FabricConfig:
    """Prepare a shared file-task workspace and explicitly selected native clients."""
    workspace = output / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "notes.txt").write_text(
        "Release checklist: verify shared model configuration, file tools, and Gateway cleanup.\n"
    )
    interfaces = {}
    if args.interface in {"terminal", "both"}:
        interfaces["terminal"] = {}
    if args.interface in {"dashboard", "both"}:
        interfaces["dashboard"] = {
            "port": args.port,
            "public_origins": args.public_origin,
        }
    settings = {"agent_id": "reviewer", "interfaces": interfaces}
    if args.openclaw_command:
        settings["openclaw_command"] = args.openclaw_command
    return FabricConfig.from_mapping(
        {
            "metadata": {"name": "openclaw-interfaces-example"},
            "harness": {"adapter_id": "nvidia.fabric.openclaw", "settings": settings},
            "models": {
                "default": {
                    "provider": "fabric-model",
                    "model": args.model,
                    "base_url": args.base_url,
                    "api_key_env": args.api_key_env,
                    "temperature": 0.0,
                    "top_p": 0.9,
                    "max_tokens": 2048,
                }
            },
            "instructions": {
                "system": {
                    "mode": "replace",
                    "content": "Complete workspace tasks using the read and write tools. Read source files before writing. "
                    "Use paths relative to the workspace. Do not use shell or web tools. Summarize actual results briefly.",
                }
            },
            "tools": {"enabled": ["read", "write"], "blocked": ["exec", "process"]},
            "environment": {
                "provider": "local",
                "workspace": str(workspace),
                "artifacts": str(output / "artifacts"),
            },
        }
    )


async def run(args: argparse.Namespace) -> None:
    """Keep one Gateway alive for clients, or execute supplied SDK/HTTP tasks once."""
    output = args.output_dir.absolute()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("ADAPTER_PYTHON", sys.executable)
    config = build_config(args, output)
    fabric = Fabric()
    async with await fabric.start_runtime(config, base_dir=Path.cwd()) as runtime:
        directory = (
            output
            / "artifacts/.fabric/openclaw/runtimes"
            / runtime.runtime_id
            / "interfaces"
        )
        connection = json.loads((directory / "gateway.json").read_text())
        # These are paths and endpoints, never credential values.
        info = {
            "runtime_id": runtime.runtime_id,
            "connection": str(directory / "gateway.json"),
            "http_url": connection["http_url"],
            "agent_id": connection["agent_id"],
            "token_file": str(directory / connection["token_file"]),
            "workspace": str(output / "workspace"),
        }
        if "terminal" in config.harness.settings["interfaces"]:
            info["terminal_launcher"] = str(directory / "terminal")
        (output / "connection.json").write_text(json.dumps(info, indent=2) + "\n")
        print(json.dumps(info, indent=2), flush=True)
        if args.sdk_input:
            result = await runtime.invoke(input=args.sdk_input)
            (output / "sdk-result.json").write_text(
                json.dumps(result.to_mapping(), indent=2) + "\n"
            )
            if result.status != "succeeded":
                raise RuntimeError("SDK task failed; inspect sdk-result.json")
        if args.http_input:
            async with httpx.AsyncClient(trust_env=False, timeout=180) as client:
                response = await client.post(
                    connection["http_url"] + "/v1/chat/completions",
                    headers={
                        "Authorization": "Bearer "
                        + (directory / connection["token_file"]).read_text().strip()
                    },
                    json={
                        "model": f"openclaw/{connection['agent_id']}",
                        "user": "manual-http-client",
                        "messages": [{"role": "user", "content": args.http_input}],
                    },
                )
                response.raise_for_status()
                (output / "http-result.json").write_text(
                    json.dumps(response.json(), indent=2) + "\n"
                )
        if not args.once:
            print(
                "Gateway ready. Attach clients in other terminals; press Enter here to stop.",
                flush=True,
            )
            await asyncio.to_thread(input)
    (output / "cleanup.json").write_text(
        json.dumps(
            {
                "discovery_removed": not (directory / "gateway.json").exists(),
                "token_removed": not (directory / connection["token_file"]).exists(),
                "terminal_marker_removed": not (
                    directory / "terminal.running"
                ).exists(),
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    """Run the repository example with prepared native dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="MODEL_API_KEY")
    parser.add_argument("--openclaw-command")
    parser.add_argument(
        "--interface", choices=("terminal", "dashboard", "both"), default="terminal"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=18789,
        help="Dashboard Gateway base port; terminal-only uses automatic allocation",
    )
    parser.add_argument("--public-origin", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sdk-input")
    parser.add_argument("--http-input")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Stop after supplied tasks; otherwise wait for manual clients",
    )
    args = parser.parse_args()
    if not os.environ.get(args.api_key_env):
        parser.error(
            f"Set {args.api_key_env} to the credential required by the model endpoint"
        )
    if args.once and not (args.sdk_input or args.http_input):
        parser.error("--once requires --sdk-input or --http-input")
    if args.interface == "terminal" and (args.public_origin or args.port != 18789):
        parser.error(
            "--port and --public-origin apply to the dashboard; terminal has no port setting"
        )
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
