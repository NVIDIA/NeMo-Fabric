# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive a `native-events.jsonl` fixture from a captured `events.atof.jsonl`.

The raw *native* harness events are surfaced two ways in the Relay ATOF capture:
  * Gateway harnesses (Claude/Codex): the native provider streaming event is
    embedded in each ``llm.chunk`` record's ``data`` (Anthropic message/content
    -block events; OpenAI Responses turn/item events) — we emit that ``data``.
  * In-process harnesses (Hermes/Deep Agents): the ATOF scope/mark IS a thin
    envelope over the native callback — we project the native-relevant fields.

Usage: derive_native_events.py <events.atof.jsonl> <native-events.jsonl> [max_bytes_per_line]
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    inp, outp = sys.argv[1], sys.argv[2]
    max_bytes = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    n = 0
    with open(inp, encoding="utf-8") as f, open(outp, "w", encoding="utf-8") as o:
        for line in f:
            line = line.strip()
            if not line or (max_bytes and len(line) > max_bytes):
                continue
            r = json.loads(line)
            if r.get("name") == "llm.chunk":
                native = r.get("data") or {}
            else:
                native = {
                    "kind": r.get("kind"),
                    "name": r.get("name"),
                    "category": r.get("category"),
                    "scope_category": r.get("scope_category"),
                    "data": r.get("data"),
                }
            o.write(json.dumps(native) + "\n")
            n += 1
    print(f"{n} native events -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
