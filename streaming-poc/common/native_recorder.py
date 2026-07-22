# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""POC-only native-event recorder (tees the harness SDK stream BEFORE Relay).

This exists solely to produce *independent* native evidence for the streaming
POC — proof that the raw SDK stream carries fields ATOF does not, so the
raw-ATOF-passthrough decision is justified by comparison, not assumption. It is
**not** a Fabric public API and must not ship in the adapters.

Each call appends one line to ``$POC_NATIVE_RECORD``:

    {"sequence": 12, "source": "langgraph", "event_type": "updates",
     "native": { ...complete SDK event, secrets redacted... }}

Secrets are redacted by key; event fields (text deltas, tool arguments, IDs,
ordering, parent relationships) are preserved verbatim.

Temporary capture seams (apply, run once, revert):
  * Hermes (adapters/hermes/.../adapter.py, after ``discover_plugins``):
        patch ``hermes_cli.plugins.PluginManager.invoke_hook`` to
        ``record("hermes-cli", hook_name, dict(kwargs))`` then delegate.
  * Deep Agents (adapters/deepagents/.../adapter.py, in the ``astream`` loop):
        ``record("langgraph", str(mode), {"namespace": list(namespace), "chunk": chunk})``
        for every ``(namespace, mode, chunk)`` tuple, before the projection.
Enable with env ``POC_NATIVE_RECORD=<path>`` and ``POC_RECORDER_DIR=<this dir>``.
"""

from __future__ import annotations

import json
import os
from typing import Any

_SECRET_HINTS = ("api_key", "apikey", "token", "authorization", "secret", "password")
_seq = 0


def _redact(value: Any, key: str = "") -> Any:
    if key and any(h in key.lower() for h in _SECRET_HINTS) and isinstance(value, str):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Preserve structured SDK objects where possible; else fall back to repr.
    for attr in ("model_dump", "dict", "to_dict"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return _redact(fn())
            except Exception:
                break
    data = getattr(value, "__dict__", None)
    if isinstance(data, dict) and data:
        return {"__type__": type(value).__name__, **_redact(data)}
    return {"__type__": type(value).__name__, "__repr__": repr(value)[:2000]}


def record(source: str, event_type: str, native: Any) -> None:
    """Append one native event to ``$POC_NATIVE_RECORD`` (no-op if unset)."""
    global _seq
    path = os.environ.get("POC_NATIVE_RECORD")
    if not path:
        return
    _seq += 1
    line = json.dumps(
        {
            "sequence": _seq,
            "source": source,
            "event_type": str(event_type),
            "native": _redact(native),
        },
        ensure_ascii=False,
    )
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
