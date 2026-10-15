# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the custom LangGraph agent in OpenShell through the host Fabric SDK.

Build task-visible configuration for the sandbox, invoke the agent through
the task-side runtime, and collect its results on the host. Relay and MCP
options reuse the same agent implementation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

from nemo_fabric import EnvironmentConfig
from nemo_fabric import EnvironmentReference
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import RunRequest

from examples.langgraph_custom_agent.consumer.config import public_config
from examples.langgraph_custom_agent.consumer.config import with_relay
from examples.langgraph_custom_agent.consumer.config import with_url_inspector_mcp
from examples.langgraph_custom_agent.consumer.config import PUBLIC_DEFAULT_MODEL

ADAPTER_ROOT = Path(__file__).resolve().parents[1] / "adapter"
TASK_ROOT = "/opt/fabric/examples/langgraph_custom_agent"
TASK_ADAPTER_ROOT = f"{TASK_ROOT}/adapter"
TASK_PYTHON = "/opt/fabric/bin/python"
TASK_WORKSPACE = "/sandbox/langgraph/workspace"
TASK_ARTIFACTS = "/sandbox/langgraph/artifacts"
TASK_MCP_SERVER = f"{TASK_ROOT}/mcp/url_inspector.py"


def build_config(args: argparse.Namespace) -> FabricConfig:
    """Build the custom-agent config with task-image paths and OpenShell settings."""

    config = public_config(args.model)
    config.harness.resolution = "image_provided"
    model = config.models["default"]
    model.provider = "openai"
    model.api_key_env = "OPENAI_API_KEY"
    model.base_url = args.base_url
    connection: dict[str, Any] = {
        "gateway": args.gateway,
        "workspace": args.workspace,
    }
    if args.ownership == "caller_owned":
        connection["sandbox_id"] = args.sandbox_id
    config.environment = EnvironmentConfig(
        provider="openshell",
        ownership=args.ownership,
        control_location="in_env_control",
        workspace=TASK_WORKSPACE,
        artifacts=TASK_ARTIFACTS,
        connection=connection,
        settings={
            "image": args.image,
            "command": ["fabric-agent-ctl", "serve"],
            "policy_yaml": Path(args.policy_file).read_text(encoding="utf-8"),
            "providers": ["hosted"],
            "sandbox_name": args.sandbox_name,
            "task_discovery_paths": [TASK_ADAPTER_ROOT],
        },
    )
    # `runtime.artifacts` is collected by the host SDK. Keep it on the host;
    # `TASK_ARTIFACTS` is the separate path visible inside the sandbox.
    host_artifacts = Path(args.results_file).resolve().parent / "artifacts"
    config.runtime.artifacts = host_artifacts
    if args.relay:
        config = with_relay(config)
        config.runtime.artifacts = host_artifacts
        config.relay.output_dir = f"{TASK_ARTIFACTS}/relay"
        observability = config.relay.observability
        if observability is not None:
            if observability.atif is not None:
                observability.atif.output_directory = f"{TASK_ARTIFACTS}/relay"
            if observability.atof is not None:
                for sink in observability.atof.sinks or []:
                    if sink.type == "file":
                        sink.output_directory = f"{TASK_ARTIFACTS}/relay"
    if args.mcp:
        config = with_url_inspector_mcp(config)
        server = config.mcp.servers["url-inspector"]
        server.url = TASK_PYTHON
        server.args = [TASK_MCP_SERVER]
    return config


async def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    """Attach or prepare, run one or two requests, and clean up the SDK lifecycle."""

    fabric = Fabric()
    config = build_config(args)
    environment = None
    runtime = None
    report: dict[str, Any] = {
        "status": "failed",
        "ownership": args.ownership,
        "runtime_id": None,
        "invocations": [],
        "error": None,
    }
    try:
        if args.ownership == "caller_owned":
            environment = await fabric.attach_environment(
                config,
                EnvironmentReference.from_mapping(
                    {
                        "provider": "openshell",
                        "resource": {
                            "sandbox_name": args.sandbox_name,
                            "sandbox_id": args.sandbox_id,
                        },
                    }
                ),
            )
        else:
            environment = await fabric.prepare_environment(config)

        runtime = await fabric.start_runtime_in(config, environment)
        report["runtime_id"] = runtime.runtime_id
        relay_session_root = str(uuid4()) if args.relay else None
        inputs = [args.input]
        if args.follow_up is not None:
            inputs.append(args.follow_up)
        for index, value in enumerate(inputs, start=1):
            result = await runtime.invoke(
                request=RunRequest(
                    input=value,
                    request_id=f"langgraph-openshell-turn-{index}",
                    context={"turn": index},
                    relay_session_root=relay_session_root,
                )
            )
            report["invocations"].append(result.to_mapping())
            if result.status != "succeeded":
                report["error"] = {
                    "code": "invoke_failed",
                    "request_id": result.request_id,
                }
                break
        if report["error"] is None:
            report["status"] = "succeeded"
    except Exception as error:
        if os.environ.get("OPENSHELL_LANGGRAPH_DEBUG_ERRORS") == "true":
            print(f"LangGraph SDK failure: {error}", file=sys.stderr)
        failure = {"code": getattr(error, "code", None) or type(error).__name__}
        stage = getattr(error, "stage", None)
        if isinstance(stage, str):
            failure["stage"] = stage
        report["error"] = failure
    finally:
        try:
            if runtime is not None:
                try:
                    await runtime.stop()
                except Exception as error:
                    report["status"] = "failed"
                    report["error"] = report["error"] or {"code": type(error).__name__}
        finally:
            if environment is not None:
                try:
                    await fabric.release_environment(environment)
                except Exception as error:
                    report["status"] = "failed"
                    report["error"] = report["error"] or {"code": type(error).__name__}

    result_path = Path(args.results_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ownership", choices=("caller_owned", "fabric_owned"), required=True
    )
    parser.add_argument("--gateway", required=True)
    parser.add_argument("--workspace", default="default")
    parser.add_argument("--image", required=True)
    parser.add_argument("--policy-file", required=True)
    parser.add_argument("--results-file", required=True)
    parser.add_argument("--sandbox-name", required=True)
    parser.add_argument("--sandbox-id")
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENSHELL_LANGGRAPH_MODEL", PUBLIC_DEFAULT_MODEL),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "OPENSHELL_LANGGRAPH_BASE_URL", "https://integrate.api.nvidia.com/v1"
        ),
    )
    parser.add_argument(
        "--input",
        default="Urgent: your account is locked. Verify your password immediately at https://example.invalid.",
    )
    parser.add_argument("--follow-up", default="Team lunch is at noon.")
    parser.add_argument("--relay", action="store_true")
    parser.add_argument("--mcp", action="store_true")
    args = parser.parse_args()
    if args.ownership == "caller_owned" and not args.sandbox_id:
        parser.error("--sandbox-id is required for caller_owned")
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("NVIDIA_API_KEY")):
        parser.error(
            "OPENAI_API_KEY or NVIDIA_API_KEY is required for the model provider"
        )
    report = asyncio.run(run_demo(args))
    if report["status"] != "succeeded":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
