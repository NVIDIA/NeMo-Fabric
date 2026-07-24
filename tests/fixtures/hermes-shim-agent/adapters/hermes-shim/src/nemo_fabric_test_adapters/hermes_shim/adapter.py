#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-only Hermes-shaped adapter shim."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nemo_fabric_adapters.common import lifecycle


def main() -> None:
    lifecycle.serve(ShimRuntime)


class ShimRuntime:
    def __init__(self) -> None:
        self._start_payload: dict[str, Any] | None = None

    async def start(self, payload: dict[str, Any]) -> None:
        self._start_payload = payload

    async def invoke(self, invocation: dict[str, Any]) -> dict[str, Any]:
        if self._start_payload is None:
            raise lifecycle.LifecycleError(
                "hermes_runtime_not_started",
                "shim runtime is not started",
            )
        payload = {
            **self._start_payload,
            "runtime_context": invocation.get("runtime_context"),
            "request": invocation.get("request"),
        }
        return run_selected_mode(payload)

    async def stop(self) -> None:
        self._start_payload = None


def fabric_config(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("config") or {}


def runtime_context(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("runtime_context") or {}


def request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("request") or {}


def environment_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return (
        runtime_context(payload).get("environment") or payload.get("environment") or {}
    )


def settings_payload(payload: dict[str, Any]) -> dict[str, Any]:
    harness = fabric_config(payload).get("harness") or {}
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
        "runtime_id": context.get("runtime_id"),
        "workspace": environment.get("workspace"),
        "native_skill_paths": (capabilities.get("native") or {}).get("skill_paths", []),
        "native_mcp_servers": sorted(
            (capabilities.get("native") or {}).get("mcp_servers", {}).keys()
        ),
        "managed_skill_paths": (capabilities.get("managed") or {}).get(
            "skill_paths", []
        ),
        "managed_mcp_servers": sorted(
            (capabilities.get("managed") or {}).get("mcp_servers", {}).keys()
        ),
        "capability_routes": capabilities.get("routes", []),
        "telemetry": payload.get("telemetry"),
    }


def run_swebench_shim(payload: dict[str, Any]) -> dict[str, Any]:
    settings = settings_payload(payload)
    request = request_payload(payload)
    context = request.get("context", {})
    environment = environment_payload(payload)
    workspace_value = environment.get("workspace")
    if not isinstance(workspace_value, str) or not workspace_value:
        raise ValueError("runtime_context.environment.workspace is required")
    workspace = Path(workspace_value)
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
