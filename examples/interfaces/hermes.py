# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a file task through a managed Hermes API and inspect Relay evidence."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import RelayAtifConfig
from nemo_fabric import RelayAtofConfig
from nemo_fabric import RelayAtofFileSinkConfig
from nemo_fabric import RelayObservabilityConfig

SAMPLE_NOTES = """Release changes:
- API authentication requires a bearer token.
- Dashboard binds to loopback.
- Runtime restart creates a new API token.
- File tools remain enabled; shell execution is disabled.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("MODEL_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME"))
    parser.add_argument("--api-key-env", default="MODEL_API_KEY")
    parser.add_argument("--port", type=int, default=8642)
    parser.add_argument(
        "--input-file", type=Path, help="Release notes; defaults to a bundled sample"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="New directory for workspace and evidence"
    )
    parser.add_argument(
        "--keep-running",
        action="store_true",
        help="Keep the API open for manual requests until Enter",
    )
    args = parser.parse_args()
    if not args.base_url or not args.model:
        parser.error("set --base-url and --model, or MODEL_BASE_URL and MODEL_NAME")
    if not os.environ.get(args.api_key_env):
        parser.error(
            f"set the model credential environment variable {args.api_key_env}"
        )
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def call_api(url: str, token: str, prompt: str) -> dict:
    request = Request(
        url + "/v1/chat/completions",
        data=json.dumps(
            {"model": "hermes-agent", "messages": [{"role": "user", "content": prompt}]}
        ).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urlopen(request, timeout=180) as response:
        return json.load(response)


async def run(args: argparse.Namespace) -> None:
    run_name = (
        datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    output = (args.output_dir or Path("outputs/interfaces") / run_name).resolve()
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "workspace"
    workspace.mkdir()
    source = workspace / "release-notes.md"
    source.write_text(args.input_file.read_text() if args.input_file else SAMPLE_NOTES)
    checklist = workspace / "rollout-checklist.md"
    relay = output / "relay"
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "hermes-interfaces-example"},
            "harness": {
                "adapter_id": "nvidia.fabric.hermes",
                "settings": {"interfaces": {"api": {"port": args.port}}},
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
                    "content": "Complete the requested workspace file task using file tools. Read source files before writing the answer file. Do not use shell or web tools. Summarize the actual result briefly.",
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
            atif=RelayAtifConfig(enabled=True, output_directory=str(relay / "atif")),
        ),
    )
    prompt = f"Read {source} and create {checklist} containing a concise five-item rollout validation checklist based on the release notes. Use file tools to read and write. Then report the created file path."
    api = f"http://127.0.0.1:{args.port}"
    print(f"Evidence directory: {output}", flush=True)
    async with await Fabric().start_runtime(config, base_dir=output) as runtime:
        token_path = (
            output
            / "artifacts/.fabric/hermes/runtimes"
            / runtime.runtime_id
            / "interfaces/api.token"
        )
        print(f"API: {api}\nToken file: {token_path}", flush=True)
        response = await asyncio.to_thread(
            call_api, api, token_path.read_text(), prompt
        )
        (output / "response.json").write_text(json.dumps(response, indent=2) + "\n")
        print(response["choices"][0]["message"]["content"], flush=True)
        if args.keep_running:
            await asyncio.to_thread(
                input, "API remains available. Press Enter to stop: "
            )

    atof_files = sorted(relay.rglob("*.atof.jsonl"))
    atif_files = sorted((relay / "atif").rglob("*.json"))
    events = [
        json.loads(line)
        for path in atof_files
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    tools = sorted(
        {
            event["name"]
            for event in events
            if event.get("category") == "tool"
            and event.get("scope_category") == "start"
        }
    )
    populated_atif = [
        path for path in atif_files if json.loads(path.read_text()).get("steps")
    ]
    atif_tools = {
        call["function_name"]
        for path in populated_atif
        for step in json.loads(path.read_text())["steps"]
        for call in step.get("tool_calls", [])
    }
    checks = {
        "output_file_written": checklist.is_file()
        and bool(checklist.read_text().strip()),
        "file_tools_observed": {"read_file", "write_file"}.issubset(tools),
        "atof_events_written": bool(events),
        "atif_trajectory_written": bool(populated_atif),
        "atif_tool_steps_written": {"read_file", "write_file"}.issubset(atif_tools),
        "api_token_removed": not token_path.exists(),
    }
    summary = {
        "runtime_id": runtime.runtime_id,
        "output_file": str(checklist),
        "atof_files": [str(p) for p in atof_files],
        "atif_files": [str(p) for p in populated_atif],
        "tools": tools,
        "checks": checks,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not all(checks.values()):
        raise RuntimeError(
            "Example checks failed; inspect summary.json. ATOF alone does not establish ATIF export."
        )


def main() -> int:
    args = parse_args()
    try:
        asyncio.run(run(args))
    except (RuntimeError, OSError, URLError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
