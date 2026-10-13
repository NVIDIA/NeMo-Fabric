# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the code-review agent through the normal SDK and OpenShell provider."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from nemo_fabric import (
    EnvironmentConfig,
    EnvironmentReference,
    Fabric,
    FabricConfig,
    RunRequest,
)

from examples.code_review_agent.config import openclaw_config


DEFAULT_PROMPTS = (
    "Review calculator.py and remember review tag cedar-4827.",
    "Recall the review tag from our previous turn, summarize the finding, and "
    "suggest the smallest correction.",
)
TASK_ROOT = "/sandbox/code-review"


def build_config(args: argparse.Namespace) -> FabricConfig:
    """Build the standard OpenClaw config with OpenShell environment settings."""

    config = openclaw_config()
    config.metadata.name = "openshell-code-review"
    model = config.models["default"]
    model.provider = "openai"
    model.model = args.model
    model.base_url = args.base_url
    model.api_key_env = "OPENAI_API_KEY"
    model.max_tokens = 1024
    config.skills.paths = [f"{TASK_ROOT}/skills/code-review"]
    config.runtime.artifacts = str(
        Path(args.results_file).resolve().parent / "artifacts"
    )
    config.environment = EnvironmentConfig(
        provider="openshell",
        ownership=args.ownership,
        control_location="in_env_control",
        workspace=f"{TASK_ROOT}/repos/my-service",
        artifacts=f"{TASK_ROOT}/artifacts/code-review",
        connection={"gateway": args.gateway, "workspace": args.workspace},
        settings={
            "image": args.image,
            "command": ["fabric-agent-ctl", "serve"],
            "policy_yaml": Path(args.policy_file).read_text(encoding="utf-8"),
            "providers": ["hosted"],
            "sandbox_name": args.sandbox_name,
        },
    )
    if args.ownership == "caller_owned":
        config.environment.connection["sandbox_id"] = args.sandbox_id
    return config


async def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    """Attach or prepare, run two SDK invocations, then stop and release."""

    fabric = Fabric()
    config = build_config(args)
    environment = None
    runtime = None
    report: dict[str, Any] = {
        "status": "failed",
        "ownership": args.ownership,
        "runtime_id": None,
        "reviews": [],
        "error": None,
    }
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

        runtime = await fabric.start_runtime_in(config, environment)
        report["runtime_id"] = runtime.runtime_id
        for index, prompt in enumerate(args.inputs, start=1):
            result = await runtime.invoke(
                request=RunRequest(
                    input=prompt,
                    request_id=f"code-review-turn-{index}",
                    context={"turn": index},
                )
            )
            report["reviews"].append(result.to_mapping())
            if result.status != "succeeded":
                report["error"] = {
                    "code": "invoke_failed",
                    "request_id": result.request_id,
                }
                break
        if report["error"] is None:
            report["status"] = "succeeded"
    except Exception as error:
        # Keep process output out of reports; SDK exceptions may include command data.
        report["error"] = {"code": type(error).__name__}
    finally:
        try:
            if runtime is not None:
                try:
                    await runtime.stop()
                except Exception as error:
                    report["status"] = "failed"
                    report["error"] = report["error"] or {
                        "code": type(error).__name__
                    }
        finally:
            if environment is not None:
                try:
                    await fabric.release_environment(environment)
                except Exception as error:
                    report["status"] = "failed"
                    report["error"] = report["error"] or {
                        "code": type(error).__name__
                    }

    result_path = Path(args.results_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ownership", choices=("caller_owned", "fabric_owned"), required=True)
    parser.add_argument("--gateway", required=True)
    parser.add_argument("--workspace", default="default")
    parser.add_argument("--image", required=True)
    parser.add_argument("--policy-file", required=True)
    parser.add_argument("--results-file", required=True)
    parser.add_argument("--sandbox-name", required=True)
    parser.add_argument("--sandbox-id")
    parser.add_argument(
        "--model", default="nvidia/nemotron-3.5-lightning-30b-a3b"
    )
    parser.add_argument(
        "--base-url", default="https://integrate.api.nvidia.com/v1"
    )
    parser.add_argument("--input", dest="inputs", action="append")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.ownership == "caller_owned" and not args.sandbox_id:
        raise SystemExit("--sandbox-id is required for caller_owned mode")
    args.inputs = args.inputs or list(DEFAULT_PROMPTS)
    if len(args.inputs) != 2:
        raise SystemExit("--input must be supplied exactly twice")
    if not os.environ.get("NVIDIA_API_KEY"):
        raise SystemExit("NVIDIA_API_KEY is required to configure the hosted provider")
    report = asyncio.run(run_demo(args))
    if report["status"] != "succeeded":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
