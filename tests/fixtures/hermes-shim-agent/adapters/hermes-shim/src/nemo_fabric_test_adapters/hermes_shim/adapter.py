#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-only Hermes-shaped adapter shim."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    payload = json.load(sys.stdin)
    output = run(payload)
    print(json.dumps(output, sort_keys=True))
    if output.get("failed"):
        raise SystemExit(2)


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """Test adapter entrypoint used by SDK smoke tests."""

    return run_selected_mode(payload)


def effective_config(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("effective_config") or {}


def fabric_config(payload: dict[str, Any]) -> dict[str, Any]:
    return effective_config(payload).get("config") or {}


def runtime_context(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("runtime_context") or {}


def request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("request") or {}


def environment_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return runtime_context(payload).get("environment") or payload.get("environment") or {}


def settings_payload(payload: dict[str, Any]) -> dict[str, Any]:
    harness = (fabric_config(payload).get("harness") or {})
    return harness.get("settings") or payload.get("settings") or {}


def capability_plan(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("capability_plan") or payload.get("capabilities") or {}


def run_selected_mode(payload: dict[str, Any]) -> dict[str, Any]:
    settings = settings_payload(payload)
    if settings.get("mode") == "swebench_shim":
        return run_swebench_shim(payload)
    return run_shim(payload)


def run_shim(payload: dict[str, Any]) -> dict[str, Any]:
    settings = settings_payload(payload)
    request = request_payload(payload)
    context = runtime_context(payload)
    environment = environment_payload(payload)
    capabilities = capability_plan(payload)

    return {
        "harness": "hermes",
        "adapter": "test-shim",
        "mode": "shim",
        "received": request.get("input"),
        "session_id": context.get("session_id") or context.get("runtime_id"),
        "workspace": environment.get("workspace") or settings.get("workspace"),
        "native_skill_paths": (capabilities.get("native") or {}).get("skill_paths", []),
        "native_mcp_servers": sorted((capabilities.get("native") or {}).get("mcp_servers", {}).keys()),
        "managed_skill_paths": (capabilities.get("managed") or {}).get("skill_paths", []),
        "managed_mcp_servers": sorted((capabilities.get("managed") or {}).get("mcp_servers", {}).keys()),
        "capability_routes": capabilities.get("routes", []),
        "telemetry": payload.get("telemetry"),
    }


def run_swebench_shim(payload: dict[str, Any]) -> dict[str, Any]:
    settings = settings_payload(payload)
    request = request_payload(payload)
    context = request.get("context", {})
    environment = environment_payload(payload)
    workspace = Path(environment.get("workspace") or settings.get("workspace") or ".")
    target_file = workspace / settings.get("target_file", "calculator.py")
    before = settings.get("expected_before")
    after = settings.get("replacement")
    new_file = settings.get("new_file")
    new_file_contents = settings.get("new_file_contents", "")

    original = target_file.read_text(encoding="utf-8")
    updated = original
    failed = False
    error = None
    if before is not None and after is not None:
        if before not in original:
            failed = True
            error = f"expected_before anchor was not found in {target_file}"
        else:
            updated = original.replace(before, after)
    elif after is not None:
        updated = after
    if not failed:
        target_file.write_text(updated, encoding="utf-8")
        if new_file:
            new_path = workspace / new_file
            new_path.parent.mkdir(parents=True, exist_ok=True)
            new_path.write_text(new_file_contents, encoding="utf-8")

    return {
        "harness": "hermes",
        "adapter": "test-shim",
        "mode": "swebench_shim",
        "task_style": "swe_bench",
        "received": request.get("input"),
        "task": context.get("task") or context.get("swebench"),
        "workspace": str(workspace),
        "target_file": str(target_file),
        "new_file": str(workspace / new_file) if new_file else None,
        "changed": original != updated,
        "failed": failed,
        "error": error,
    }


if __name__ == "__main__":
    main()
