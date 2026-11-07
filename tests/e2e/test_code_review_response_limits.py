# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify normalized response budgets with two native harnesses and real inference.

Uses the endpoint, policy and interpreter variables documented by
 test_code_review_deadlines.py. Set CODE_REVIEW_RESPONSE_LIMIT_LIVE=1 and
CODE_REVIEW_RESPONSE_EVIDENCE to a local report path. Evidence is never checked in.
"""

import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
from nemo_fabric import CheckRequest, Fabric, ResponseLimits
from tests.e2e.test_code_review_deadlines import BASE_DIR, build_config

pytestmark = pytest.mark.skipif(
    os.environ.get("CODE_REVIEW_RESPONSE_LIMIT_LIVE") != "1",
    reason="live inference qualification is opt-in",
)


async def test_code_review_response_limit_matrix(tmp_path):
    records = []
    start_lock = asyncio.Lock()
    evidence = Path(os.environ["CODE_REVIEW_RESPONSE_EVIDENCE"])

    async def run(harness, location, rejected):
        directory = tmp_path / f"{harness}-{location}-{rejected}"
        directory.mkdir()
        token = f"review-{uuid4().hex}"
        config = build_config(harness, location, "smoke", directory, token)
        config.runtime.response_limits = ResponseLimits(
            max_response_bytes=1 if rejected else 8 * 1024 * 1024
        )
        fabric = Fabric()
        runtime = environment = None
        record = {
            "harness": harness,
            "location": location,
            "expected": "rejected" if rejected else "complete",
            "status": "failed",
        }
        try:
            if location == "openshell":
                environment = await fabric.prepare_environment(config)
                async with asyncio.timeout(60):
                    while (
                        await fabric.check(
                            environment=environment,
                            request=CheckRequest(scope="server"),
                        )
                    ).outcome != "passed":
                        await asyncio.sleep(0.2)
                runtime = await fabric.start_runtime_in(
                    config, environment, base_dir=BASE_DIR
                )
            else:
                async with start_lock:
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
            result = await runtime.invoke(
                input="Review calculator.py using read_review_source part 1 then part 2, as instructed."
            )
            record["result"] = result.to_mapping()
            if rejected:
                assert result.status == "failed", result.to_mapping()
                assert result.error.code == "response_limit_exceeded"
                assert result.error.metadata["limit"] == "max_response_bytes"
                assert result.error.metadata["configured"] == 1
                assert "RunResult" in result.error.message
                assert result.output is None
            else:
                assert result.status == "succeeded", result.to_mapping()
                assert result.output["completed"] is True
                response = json.dumps(result.output["response"])
                assert token in response
                assert "41" in response
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
                record["status"] = "failed"
                record["cleanup_errors"] = cleanup_errors
            records.append(record)
            evidence.parent.mkdir(parents=True, exist_ok=True)
            evidence.write_text(
                json.dumps({"records": records}, indent=2, default=str) + "\n"
            )
            print(
                f"{harness}/{location}/{record['expected']}: {record['status']}",
                flush=True,
            )

    async with asyncio.timeout(600):
        await asyncio.gather(
            *(
                run(harness, location, rejected)
                for harness in ("hermes", "deepagents")
                for location in ("host", "openshell")
                for rejected in (False, True)
            )
        )
    assert len(records) == 8
    assert all(record["status"] == "passed" for record in records), records
