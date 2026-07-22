# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract the payloads embedded in a captured `events.atof.jsonl`.

This is an ATOF *inspection* helper, **not** native evidence: it only surfaces
what Relay already put into ATOF (each ``llm.chunk``'s ``data`` for gateway
harnesses; the scope/mark projection for in-process harnesses). It cannot recover
native fields Relay dropped. For genuine native evidence — teed from the SDK
stream *before* Relay — use ``native_recorder.py`` and see each harness's
``native-events.jsonl``.

Usage: extract_relay_payloads.py <events.atof.jsonl> <relay-payloads.jsonl> [max_bytes_per_line]
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
    print(f"{n} relay payloads -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
