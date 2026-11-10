# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real inference qualification using the code-review example on host and OpenShell.

Set CODE_REVIEW_DEADLINE_LIVE=1, CODE_REVIEW_MODEL_URL, CODE_REVIEW_MODEL,
CODE_REVIEW_HERMES_PYTHON, CODE_REVIEW_DEEPAGENTS_PYTHON,
CODE_REVIEW_GATEWAY, CODE_REVIEW_IMAGE and CODE_REVIEW_POLICY.
CODE_REVIEW_DEADLINE_CASES selects smoke, finite, unlimited, timeout (default all).
CODE_REVIEW_EVIDENCE selects the retained JSON report path.
"""

import asyncio
import json
import os
from pathlib import Path
import time
from uuid import uuid4

import pytest
from nemo_fabric import (
    CheckRequest,
    DiscoveryConfig,
    EnvironmentConfig,
    Fabric,
    FabricRuntimeError,
    ToolsConfig,
)

from examples.code_review_agent.config import (
    BASE_DIR,
    CODE_REVIEW_INSTRUCTION,
    deepagents_config,
    hermes_config,
)

ROOT = Path(__file__).resolve().parents[2]
LIVE = os.environ.get("CODE_REVIEW_DEADLINE_LIVE") == "1"
pytestmark = pytest.mark.skipif(
    not LIVE, reason="live code-review qualification is opt-in"
)


def build_config(harness, location, mode, directory, token):
    config = {"hermes": hermes_config, "deepagents": deepagents_config}[harness]()
    sandbox = location == "openshell"
    task_root = Path("/sandbox/code-review") if sandbox else BASE_DIR
    task_evidence = Path("/sandbox/artifacts") if sandbox else directory
    python = (
        (
            "/opt/deepagents/bin/python"
            if harness == "deepagents"
            else "/opt/fabric/bin/python"
        )
        if sandbox
        else os.environ[f"CODE_REVIEW_{harness.upper()}_PYTHON"]
    )
    delay = {"smoke": 0, "finite": 165, "unlimited": 165, "timeout": 45}[mode]
    config.runtime.timeout_seconds = {
        "smoke": 180,
        "finite": 900,
        "unlimited": None,
        "timeout": 15,
    }[mode]
    config.runtime.max_turns = 20
    config.runtime.artifacts = directory / "artifacts"
    config.skills.paths = [str(task_root / "skills/code-review")]
    config.instructions.system.content = CODE_REVIEW_INSTRUCTION + (
        " For this review, obtain the source only through the read_review_source MCP tool. "
        "Call part=1 once, wait for its response, then call part=2 once. "
        "Do not use file or terminal tools. In your final answer quote the review receipt "
        "from part 2 verbatim and state the integer returned by answer()."
    )
    model = config.models["default"]
    model.provider = "openai"
    model.model = os.environ["CODE_REVIEW_MODEL"]
    model.base_url = (
        os.environ.get(
            "CODE_REVIEW_SANDBOX_MODEL_URL", os.environ["CODE_REVIEW_MODEL_URL"]
        )
        if sandbox
        else os.environ["CODE_REVIEW_MODEL_URL"]
    )
    model.api_key_env = "OPENAI_API_KEY"
    model.max_tokens = 2048
    if harness == "hermes":
        config.harness.settings["max_tokens"] = 2048
        config.tools = ToolsConfig(enabled=["mcp-review"])
    descriptor = ROOT / f"adapters/python/{harness}/{harness}.fabric-adapter.json"
    config.discovery = DiscoveryConfig(local_paths=[descriptor])
    if sandbox:
        config.harness.resolution = "image_provided"
        config.environment = EnvironmentConfig(
            provider="openshell",
            ownership="fabric_owned",
            control_location="in_env_control",
            workspace=str(task_root / "repos/my-service"),
            artifacts=str(task_evidence),
            env={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]},
            connection={
                "gateway": os.environ["CODE_REVIEW_GATEWAY"],
                "workspace": "default",
            },
            settings={
                "image": os.environ["CODE_REVIEW_IMAGE"],
                "command": [
                    "env",
                    f"ADAPTER_PYTHON={python}",
                    "fabric-agent-ctl",
                    "serve",
                ],
                "policy_yaml": Path(os.environ["CODE_REVIEW_POLICY"]).read_text(),
                "task_discovery_paths": [f"/opt/fabric/adapters/python/{harness}"],
            },
        )
        server = "/opt/fabric/qualification/deadline_review_source.py"
    else:
        config.environment.workspace = str(BASE_DIR / "repos/my-service")
        config.environment.artifacts = directory / "artifacts"
        config.environment.env = {"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]}
        server = str(ROOT / "tests/e2e/deadline_review_source.py")
    config.add_mcp_server(
        "review",
        transport="stdio",
        url=python,
        args=[
            server,
            "--source",
            str(task_root / "repos/my-service/calculator.py"),
            "--delay",
            str(delay),
            "--receipt",
            str(task_evidence / "review-source.receipt"),
            "--token",
            token,
        ],
    )
    return config


async def test_code_review_deadline_matrix(tmp_path):
    # With live mode selected, absent prerequisites fail rather than becoming skips.
    for name in (
        "CODE_REVIEW_MODEL_URL",
        "CODE_REVIEW_MODEL",
        "CODE_REVIEW_HERMES_PYTHON",
        "CODE_REVIEW_DEEPAGENTS_PYTHON",
        "CODE_REVIEW_GATEWAY",
        "CODE_REVIEW_IMAGE",
        "CODE_REVIEW_POLICY",
        "OPENAI_API_KEY",
    ):
        assert os.environ[name], f"{name} is required"
    modes = os.environ.get(
        "CODE_REVIEW_DEADLINE_CASES", "smoke,finite,unlimited,timeout"
    ).split(",")
    assert set(modes) <= {"smoke", "finite", "unlimited", "timeout"}
    locations = os.environ.get(
        "CODE_REVIEW_DEADLINE_LOCATIONS", "host,openshell"
    ).split(",")
    assert set(locations) <= {"host", "openshell"}
    evidence = Path(os.environ["CODE_REVIEW_EVIDENCE"])
    records = []
    host_start_lock = asyncio.Lock()

    async def run(harness, location, mode):
        directory = tmp_path / f"{harness}-{location}-{mode}"
        directory.mkdir()
        token = f"review-{uuid4().hex}"
        config = build_config(harness, location, mode, directory, token)
        fabric = Fabric()
        environment = runtime = None
        record = {
            "harness": harness,
            "location": location,
            "mode": mode,
            "timeout_seconds": config.runtime.timeout_seconds,
            "model": os.environ["CODE_REVIEW_MODEL"],
            "status": "failed",
            "cleanup": "pending",
            "expected_receipt": token,
        }
        try:
            if location == "openshell":
                environment = await fabric.prepare_environment(config)
                record["sandbox_id"] = environment.connection["sandbox_id"]
                # A ready sandbox can precede its command-bridge socket. Probe
                # responsiveness before dispatching a lifecycle operation.
                async with asyncio.timeout(60):
                    while True:
                        report = await fabric.check(
                            environment=environment,
                            request=CheckRequest(scope="server"),
                        )
                        if report.outcome == "passed":
                            break
                        await asyncio.sleep(0.2)
                runtime = await fabric.start_runtime_in(
                    config, environment, base_dir=BASE_DIR
                )
            else:
                async with host_start_lock:
                    previous = os.environ.get("ADAPTER_PYTHON")
                    os.environ["ADAPTER_PYTHON"] = os.environ[
                        f"CODE_REVIEW_{harness.upper()}_PYTHON"
                    ]
                    try:
                        runtime = await fabric.start_runtime(config, base_dir=BASE_DIR)
                    finally:
                        if previous is None:
                            os.environ.pop("ADAPTER_PYTHON", None)
                        else:
                            os.environ["ADAPTER_PYTHON"] = previous
            started = time.monotonic()
            try:
                result = await runtime.invoke(
                    input="Review calculator.py using read_review_source part 1 then part 2, as instructed."
                )
                record["elapsed_seconds"] = time.monotonic() - started
                record["result"] = result.to_mapping()
                if mode == "timeout":
                    assert result.status == "failed"
                    assert result.error.code == "timeout"
                else:
                    assert result.status == "succeeded", result.to_mapping()
                    assert record["result"]["output"]["completed"] is True
                    output = json.dumps(record["result"]["output"]["response"])
                    assert token in output, (
                        "The review must quote the tool's completion receipt"
                    )
                    assert "41" in output, (
                        "The review must inspect the actual example source"
                    )
                    if mode in ("finite", "unlimited"):
                        assert record["elapsed_seconds"] > 320
            except FabricRuntimeError as error:
                record["elapsed_seconds"] = time.monotonic() - started
                record["error"] = {"code": error.code, "stage": error.stage}
                assert (
                    mode == "timeout"
                    and error.code == "timeout"
                    and error.stage == "invoke"
                )
            if mode == "timeout":
                assert record["timeout_seconds"] - 1 <= record["elapsed_seconds"] < 45
            record["status"] = "passed"
        except Exception as error:
            record["failure"] = str(error)
        finally:
            cleanup_errors = []
            if runtime is not None:
                try:
                    await runtime.stop()
                except Exception as error:
                    cleanup_errors.append({"operation": "stop", "message": str(error)})
            if environment is not None:
                try:
                    await fabric.release_environment(environment)
                except Exception as error:
                    cleanup_errors.append(
                        {"operation": "release", "message": str(error)}
                    )
            if cleanup_errors:
                record["cleanup"] = "failed"
                record["status"] = "failed"
                record["cleanup_errors"] = cleanup_errors
            else:
                record["cleanup"] = "passed"
            records.append(record)
            evidence.parent.mkdir(parents=True, exist_ok=True)
            evidence.write_text(
                json.dumps({"records": records}, indent=2, default=str) + "\n"
            )
            print(
                f"{harness}/{location}/{mode}: {record['status']} cleanup={record['cleanup']} elapsed={record.get('elapsed_seconds')}",
                flush=True,
            )

    # Validate setup before long work. Long modes use independent runtimes and
    # run together so qualification spends about six minutes in controlled tools.
    phases = [
        [mode for mode in modes if mode == "smoke"],
        [mode for mode in modes if mode in ("finite", "unlimited")],
        [mode for mode in modes if mode == "timeout"],
    ]
    for phase in phases:
        await asyncio.wait_for(
            asyncio.gather(
                *(
                    run(harness, location, mode)
                    for mode in phase
                    for harness in ("hermes", "deepagents")
                    for location in locations
                )
            ),
            timeout=1200,
        )
        assert all(record["status"] == "passed" for record in records), records
