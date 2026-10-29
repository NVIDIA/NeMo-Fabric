# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Return a blocking Codex decision when a Relay tool check fails."""

from __future__ import annotations

import subprocess
import sys

# Relay has 30 seconds for delivery; the harness allows 45 seconds for this hook.
RELAY_TOOL_CHECK_TIMEOUT_SECONDS = 35


def _block() -> int:
    # Codex requires both exit code 2 and a nonempty reason. Keep diagnostics
    # independent of child output, which can contain model or credential data.
    print("NeMo Relay could not approve this tool call.", file=sys.stderr)
    return 2


def main() -> int:
    command = sys.argv[1:]
    if len(command) != 4 or command[1:] != [
        "hook-forward",
        "codex",
        "--fail-closed",
    ]:
        return _block()
    try:
        completed = subprocess.run(
            command,
            stdin=sys.stdin.buffer,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=RELAY_TOOL_CHECK_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return _block()
    if completed.returncode != 0:
        return _block()
    sys.stdout.buffer.write(completed.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
