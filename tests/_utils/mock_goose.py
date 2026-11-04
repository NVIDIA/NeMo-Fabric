#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock `goose run --output-format stream-json` used by adapter tests.

Keywords in the task select an outcome. Each run adds fixed token counts to
cumulative session totals, as Goose reports them.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

args = sys.argv[1:]
root = Path(os.environ["GOOSE_PATH_ROOT"])
sessions = root / "data" / "sessions"
if args == ["session", "list", "--format", "json"]:
    print(json.dumps([{"name": path.stem} for path in sorted(sessions.glob("*.json"))]))
    raise SystemExit(0)
if args[:1] != ["run"]:
    raise SystemExit(2)
name = args[args.index("--name") + 1]
session = sessions / f"{name}.json"
if "--resume" in args and not session.exists():
    print(f"Error: No session found with name '{name}'", file=sys.stderr)
    raise SystemExit(1)
state = (
    json.loads(session.read_text())
    if "--resume" in args
    else {"turns": [], "tokens": 0}
)
task = sys.stdin.read()
with open(os.environ["FAKE_GOOSE_CAPTURE"], "a", encoding="utf-8") as stream:
    config = root / "config" / "config.yaml"
    stream.write(
        json.dumps(
            {
                "args": args,
                "cwd": os.getcwd(),
                "task": task,
                "env": {
                    key: value
                    for key, value in os.environ.items()
                    if key.startswith(("GOOSE_", "OPENAI_"))
                },
                "config": json.loads(config.read_text()),
                "history": state["turns"],
            }
        )
        + "\n"
    )
session.parent.mkdir(parents=True, exist_ok=True)
state["turns"].append(task)
state["tokens"] += 1
session.write_text(json.dumps(state))
print("goose is ready")


def emit(event):
    print(json.dumps(event), flush=True)


def message(text, *, model=True):
    metadata = {"userVisible": True, "agentVisible": True}
    if model:
        metadata["inference"] = {"provider": "openai", "requestedModel": "mock"}
    # Goose streams model text as chunks that share one message ID.
    for chunk in (text[:5], text[5:]) if model else (text,):
        emit(
            {
                "type": "message",
                "message": {
                    "id": "m",
                    "role": "assistant",
                    "content": [{"type": "text", "text": chunk}],
                    "metadata": metadata,
                },
            }
        )


if "SLEEP" in task or "BACKGROUND" in task:
    # Goose starts each shell command in its own process group and reads its output.
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        process_group=0,
    )
    Path(os.environ["FAKE_GOOSE_CHILD_PID"]).write_text(str(child.pid))
if "SLEEP" in task:
    time.sleep(60)
if "EXIT_FAILURE" in task:
    emit({"type": "error", "error": "provider rejected the request"})
    raise SystemExit(1)
if "EXTENSION_FAILURE" in task:
    print(
        "Failed to start extension 'docs' (IO error), continuing without it",
        file=sys.stderr,
    )
if "MAX_TURNS" in task:
    message(
        "I've reached the maximum number of actions I can do without user input. "
        "Would you like me to continue?",
        model=False,
    )
elif "MODEL_ERROR" in task:
    message("Ran into this error: Server error: 503.", model=False)
elif "SILENT" not in task:
    message(f"turn {len(state['turns'])}: {task}")
tokens = state["tokens"]
emit(
    {
        "type": "complete",
        "total_tokens": 12 * tokens,
        "input_tokens": 10 * tokens,
        "output_tokens": 2 * tokens,
        # Goose omits a counter that the model endpoint does not report.
        **({} if "NO_CACHE" in task else {"cache_read_input_tokens": 3 * tokens}),
        "cache_write_input_tokens": 0,
    }
)
