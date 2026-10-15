# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive the echo agent in OpenShell through the normal host SDK.

Build a Fabric config for a caller-owned or Fabric-owned sandbox, start the
task runtime, invoke it twice, and collect the declared artifact on the host.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from nemo_fabric import (
    DiscoveryConfig,
    EnvironmentConfig,
    EnvironmentReference,
    Fabric,
    FabricConfig,
    HarnessConfig,
    MetadataConfig,
    RunRequest,
    RuntimeConfig,
)


ADAPTER_ROOT = Path(__file__).resolve().parent / "adapter"
TASK_ADAPTER_ROOT = "/opt/fabric/agents/echo"
TASK_WORKSPACE = "/sandbox/workspace"
TASK_ARTIFACTS = "/sandbox/artifacts"
TASK_COMMAND = ["fabric-agent-ctl", "serve"]


def _json_input(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _record_fabric_owned_sandbox(args: argparse.Namespace, environment: Any) -> None:
    if args.ownership != "fabric_owned":
        return
    state_file = os.environ.get("OPENSHELL_FABRIC_SANDBOX_STATE_FILE")
    sandbox_id = environment.connection.get("sandbox_id")
    if not state_file or not isinstance(sandbox_id, str) or not sandbox_id:
        raise RuntimeError("prepared environment did not expose its sandbox identity")
    path = Path(state_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps({"name": args.sandbox_name, "id": sandbox_id}) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _config(
    args: argparse.Namespace,
    *,
    task_workspace: str,
    task_artifacts: str,
    task_discovery_path: str,
) -> FabricConfig:
    connection: dict[str, Any] = {
        "gateway": args.gateway,
        "workspace": "default",
    }
    settings: dict[str, Any] = {
        "image": args.image,
        "command": TASK_COMMAND,
        "policy_yaml": Path(args.policy_file).read_text(encoding="utf-8"),
        "task_discovery_paths": [task_discovery_path],
    }
    if args.ownership == "caller_owned":
        connection["sandbox_id"] = args.sandbox_id
    else:
        settings["sandbox_name"] = args.sandbox_name

    artifacts_dir = Path(args.results_file).resolve().parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return FabricConfig(
        metadata=MetadataConfig(
            name="openshell-echo",
            description="Stateful example agent run through the OpenShell provider.",
        ),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.example.openshell.echo",
            resolution="image_provided",
        ),
        discovery=DiscoveryConfig(local_paths=[ADAPTER_ROOT]),
        runtime=RuntimeConfig(artifacts=str(artifacts_dir)),
        environment=EnvironmentConfig(
            provider="openshell",
            ownership=args.ownership,
            control_location="in_env_control",
            workspace=task_workspace,
            artifacts=task_artifacts,
            connection=connection,
            settings=settings,
        ),
    )


async def run_demo(
    args: argparse.Namespace,
    *,
    task_workspace: str = TASK_WORKSPACE,
    task_artifacts: str = TASK_ARTIFACTS,
    task_discovery_path: str = TASK_ADAPTER_ROOT,
) -> dict[str, Any]:
    """Exercise environment, runtime, invoke, and result lifecycles via the SDK."""

    fabric = Fabric()
    config = _config(
        args,
        task_workspace=task_workspace,
        task_artifacts=task_artifacts,
        task_discovery_path=task_discovery_path,
    )
    environment = None
    runtime = None
    try:
        if args.ownership == "caller_owned":
            reference = EnvironmentReference.from_mapping(
                {
                    "provider": "openshell",
                    "resource": {
                        "sandbox_name": args.sandbox_name,
                        "sandbox_id": args.sandbox_id,
                    },
                }
            )
            environment = await fabric.attach_environment(config, reference)
        else:
            environment = await fabric.prepare_environment(config)
            _record_fabric_owned_sandbox(args, environment)

        runtime = await fabric.start_runtime_in(config, environment)
        first = await runtime.invoke(
            request=RunRequest(
                input=_json_input(args.input),
                request_id="openshell-echo-first",
                context={"turn": 1},
            )
        )
        second = await runtime.invoke(
            request=RunRequest(
                input={"follow_up": True},
                request_id="openshell-echo-second",
                context={"turn": 2},
            )
        )
        for result in (first, second):
            if result.status != "succeeded":
                raise RuntimeError(f"echo invocation {result.request_id} failed")

        receipt_paths = [
            str(artifact.path)
            for artifact in second.artifacts.artifacts
            if artifact.name == "echo"
        ]
        if len(receipt_paths) != 1:
            raise RuntimeError("second invocation did not return the echo receipt")
        summary = {
            "ownership": args.ownership,
            "runtime_id": runtime.runtime_id,
            "first": first.to_mapping(),
            "second": second.to_mapping(),
            "receipt_paths": receipt_paths,
        }
        results_file = Path(args.results_file)
        results_file.parent.mkdir(parents=True, exist_ok=True)
        results_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return summary
    finally:
        try:
            if runtime is not None:
                await runtime.stop()
        finally:
            if environment is not None:
                await fabric.release_environment(environment)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ownership", choices=("caller_owned", "fabric_owned"), required=True)
    parser.add_argument("--gateway", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--policy-file", required=True)
    parser.add_argument("--results-file", required=True)
    parser.add_argument("--input", default="hello from the SDK consumer")
    parser.add_argument("--sandbox-name", required=True)
    parser.add_argument("--sandbox-id")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.ownership == "caller_owned" and not args.sandbox_id:
        raise SystemExit("--sandbox-id is required for caller_owned mode")
    asyncio.run(run_demo(args))


if __name__ == "__main__":
    main()
