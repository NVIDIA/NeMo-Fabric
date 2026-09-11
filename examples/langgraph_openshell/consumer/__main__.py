# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the complete two-turn Portable Courier vertical slice."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from nemo_fabric import EnvironmentReference
from nemo_fabric import Fabric

from examples.langgraph_openshell.consumer.config import courier_config


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", required=True, help="Immutable agent runtime image reference."
    )
    parser.add_argument("--gateway", default="http://127.0.0.1:18080")
    parser.add_argument("--base-dir", type=Path, default=Path(".tmp/portable-courier"))
    parser.add_argument("--sandbox-name")
    parser.add_argument("--sandbox-id")
    parser.add_argument("--fabric-sandbox-name")
    args = parser.parse_args()
    if bool(args.sandbox_name) != bool(args.sandbox_id):
        parser.error("--sandbox-name and --sandbox-id must be provided together")
    if args.sandbox_name and args.fabric_sandbox_name:
        parser.error("caller-owned and Fabric-owned sandbox names are mutually exclusive")
    args.base_dir.mkdir(parents=True, exist_ok=True)

    fabric = Fabric()
    caller_owned = args.sandbox_name is not None
    config = courier_config(
        gateway=args.gateway,
        image=args.image,
        ownership="caller_owned" if caller_owned else "fabric_owned",
        sandbox_name=args.fabric_sandbox_name,
    )
    if caller_owned:
        reference = EnvironmentReference.from_mapping(
            {
                "provider": "openshell",
                "resource": {
                    "sandbox_name": args.sandbox_name,
                    "sandbox_id": args.sandbox_id,
                },
            }
        )
        environment = await fabric.attach_environment(
            config, reference, base_dir=args.base_dir
        )
    else:
        environment = await fabric.prepare_environment(config, base_dir=args.base_dir)
    runtime = None
    try:
        runtime = await fabric.start_runtime_in(
            config, environment, base_dir=args.base_dir
        )
        routed = await runtime.invoke(input="route")
        delivered = await runtime.invoke(input="deliver")
    finally:
        try:
            if runtime is not None:
                await runtime.stop()
        finally:
            await fabric.release_environment(environment)

    expected_attempts = [
        {"route": "https://example.com/priority-lane", "outcome": "http_403"},
        {"route": "https://example.com/", "outcome": "http_200"},
    ]
    if routed.runtime_id != delivered.runtime_id:
        raise RuntimeError("both turns must execute in one Fabric runtime session")
    if not environment.metadata.get("openshell.policy_attached"):
        raise RuntimeError("OpenShell did not report a creation-time policy")
    if routed.output.get("attempts") != expected_attempts:
        raise RuntimeError(
            f"OpenShell policy evidence did not match: {routed.output!r}"
        )
    if delivered.output.get("delivery_status") != "delivered":
        raise RuntimeError(f"LangGraph state was not retained: {delivered.output!r}")
    if delivered.output.get("selected_route") != "https://example.com/":
        raise RuntimeError(
            f"the allowed fallback route was not retained: {delivered.output!r}"
        )
    if len(delivered.artifacts.artifacts) != 1:
        raise RuntimeError("the delivery turn must return exactly one receipt artifact")

    receipt_path = Path(delivered.artifacts.artifacts[0].path)
    if not receipt_path.is_file():
        raise RuntimeError(f"Fabric did not collect the receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("runtime_id") != routed.runtime_id:
        raise RuntimeError("the receipt does not prove the same runtime session")
    summary = {
        "environment": {
            "ownership": environment.ownership,
            "provider": environment.provider,
            "environment_id": environment.environment_id,
            "sandbox_id": environment.metadata.get("openshell.sandbox_id"),
            "sandbox_resource_version": environment.metadata.get(
                "openshell.sandbox_resource_version"
            ),
            "runtime_image": environment.metadata.get("openshell.runtime_image"),
            "policy_attached": environment.metadata.get("openshell.policy_attached"),
        },
        "runtime_id": routed.runtime_id,
        "turn_1": routed.output.to_mapping(),
        "turn_2": delivered.output.to_mapping(),
        "collected_artifact": {
            "path": str(receipt_path),
            "contents": receipt,
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
