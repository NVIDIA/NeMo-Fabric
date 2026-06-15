#!/usr/bin/env python3
"""Test-only Hermes-shaped adapter shim."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    payload = json.load(sys.stdin)
    output = run_selected_mode(payload)
    print(json.dumps(output, sort_keys=True))
    if output.get("failed"):
        raise SystemExit(2)


def run_selected_mode(payload: dict[str, Any]) -> dict[str, Any]:
    settings = payload.get("settings", {})
    if settings.get("mode") == "swebench_shim":
        return run_swebench_shim(payload)
    return run_shim(payload)


def run_shim(payload: dict[str, Any]) -> dict[str, Any]:
    settings = payload.get("settings", {})
    request = payload.get("request", {})
    environment = payload.get("environment", {})
    capabilities = payload.get("capabilities", {})

    return {
        "harness": "hermes",
        "adapter": "test-shim",
        "mode": "shim",
        "agent_profile": settings.get("agent_profile"),
        "received": request.get("input"),
        "workspace": environment.get("workspace") or settings.get("workspace"),
        "native_skill_paths": (capabilities.get("native") or {}).get("skill_paths", []),
        "native_mcp_servers": sorted((capabilities.get("native") or {}).get("mcp_servers", {}).keys()),
        "managed_skill_paths": (capabilities.get("managed") or {}).get("skill_paths", []),
        "managed_mcp_servers": sorted((capabilities.get("managed") or {}).get("mcp_servers", {}).keys()),
        "capability_routes": capabilities.get("routes", []),
        "telemetry": payload.get("telemetry"),
    }


def run_swebench_shim(payload: dict[str, Any]) -> dict[str, Any]:
    settings = payload.get("settings", {})
    request = payload.get("request", {})
    context = request.get("context", {})
    environment = payload.get("environment", {})
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
