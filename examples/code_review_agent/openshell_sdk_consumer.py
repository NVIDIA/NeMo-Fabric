# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the OpenClaw code-review agent through the normal host SDK.

Configure an OpenShell sandbox, start the task-side Fabric runtime, invoke the
agent across review turns, and collect the report on the host.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from nemo_fabric import (
    EnvironmentConfig,
    EnvironmentReference,
    Fabric,
    FabricConfig,
    RunRequest,
    ToolsConfig,
)

from examples.code_review_agent.config import (
    CODE_REVIEW_INSTRUCTION,
    hermes_config,
    openclaw_config,
    with_relay,
)


DEFAULT_PROMPTS = (
    "Review calculator.py and remember review tag cedar-4827.",
    "Recall the review tag from our previous turn, summarize the finding, and "
    "suggest the smallest correction.",
)
TASK_ROOT = "/sandbox/code-review"
TASK_ARTIFACT_ROOT = "/sandbox/artifacts"


def build_config(args: argparse.Namespace) -> FabricConfig:
    """Build a harness config with OpenShell environment settings."""

    harness = getattr(args, "harness", "openclaw")
    relay_enabled = getattr(args, "relay", False)
    if harness == "hermes":
        config = hermes_config()
        config.instructions.system.content = CODE_REVIEW_INSTRUCTION
        config.tools = ToolsConfig(enabled=["terminal", "read_file", "search_files"])
    else:
        config = openclaw_config()
    if relay_enabled:
        if harness != "hermes":
            raise ValueError("Relay is supported for the Hermes harness only")
        config = with_relay(config)
    config.metadata.name = "openshell-code-review"
    model = config.models["default"]
    model.provider = "openai"
    model.model = args.model
    model.base_url = args.base_url
    model.api_key_env = "OPENAI_API_KEY"
    model.max_tokens = 1024
    config.skills.paths = [f"{TASK_ROOT}/skills/code-review"]
    config.runtime.artifacts = Path(args.results_file).resolve().parent / "artifacts"
    config.environment = EnvironmentConfig(
        provider="openshell",
        ownership=args.ownership,
        control_location="in_env_control",
        workspace=f"{TASK_ROOT}/repos/my-service",
        artifacts=TASK_ARTIFACT_ROOT,
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
            _record_fabric_owned_sandbox(args, environment)

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
        if os.environ.get("OPENSHELL_CODE_REVIEW_DEBUG_ERRORS") == "true":
            print(f"Code-review SDK failure: {error}", file=sys.stderr)
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
    parser.add_argument("--harness", choices=("openclaw", "hermes"), default="openclaw")
    parser.add_argument("--relay", action="store_true")
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "OPENSHELL_CODE_REVIEW_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "OPENSHELL_CODE_REVIEW_BASE_URL", "https://integrate.api.nvidia.com/v1"
        ),
    )
    parser.add_argument("--input", dest="inputs", action="append")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.ownership == "caller_owned" and not args.sandbox_id:
        raise SystemExit("--sandbox-id is required for caller_owned mode")
    if args.relay and args.harness != "hermes":
        raise SystemExit("--relay requires --harness hermes")
    args.inputs = args.inputs or list(DEFAULT_PROMPTS)
    if len(args.inputs) != 2:
        raise SystemExit("--input must be supplied exactly twice")
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("NVIDIA_API_KEY")):
        raise SystemExit("OPENAI_API_KEY or NVIDIA_API_KEY is required for the model provider")
    report = asyncio.run(run_demo(args))
    if report["status"] != "succeeded":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
