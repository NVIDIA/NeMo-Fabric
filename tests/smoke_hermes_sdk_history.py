# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression: the Hermes SDK adapter resolves conversation history per-invoke.

A per-invocation request context history (set by the SDK on each turn) takes
precedence over static harness settings history, which is what lets the SDK
drive multi-turn sessions without mutating the agent config. Dependency-free:
no Hermes runtime and no native extension required (the adapter only imports
those inside ``run_hermes_sdk``, not at module load).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "adapters" / "hermes-sdk" / "src")
)

from nemo_fabric_adapters.hermes_sdk.adapter import resolve_history


def _payload(*, request_history=None, settings_history=None) -> dict:
    payload: dict = {
        "effective_config": {"config": {"harness": {"settings": {}}}},
        "request": {},
    }
    if settings_history is not None:
        payload["effective_config"]["config"]["harness"]["settings"][
            "history"
        ] = settings_history
    if request_history is not None:
        payload["request"] = {"context": {"history": request_history}}
    return payload


def main() -> None:
    req = [{"role": "user", "content": "from request"}]
    setpt = [{"role": "user", "content": "from settings"}]

    # per-invoke request context wins over static settings
    assert resolve_history(_payload(request_history=req, settings_history=setpt)) == req

    # falls back to static settings when the request carries no history
    assert resolve_history(_payload(settings_history=setpt)) == setpt

    # nothing set -> falsy (no history for a first turn)
    assert not resolve_history(_payload())

    # empty request history falls back to settings (first-turn semantics)
    assert resolve_history(_payload(request_history=[], settings_history=setpt)) == setpt

    print("smoke_hermes_sdk_history ok")


if __name__ == "__main__":
    main()
