# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential-free regression tests for the Harbor runner boundary."""

import asyncio
import json
import sys
from contextlib import AsyncExitStack, nullcontext
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.usefixtures("requires_harbor")


@pytest.fixture(name="run_result")
def run_result_fixture():
    from nemo_fabric import RunResult

    return RunResult.from_mapping(
        {
            "agent_name": "harbor-test",
            "harness": "test",
            "adapter_kind": "process",
            "adapter_id": "test.adapter",
            "runtime_id": "runtime-test",
            "invocation_id": "invocation-test",
            "request_id": "request-test",
            "status": "succeeded",
            "output": {"response": "done"},
            "error": None,
            "artifacts": {"artifacts": []},
            "telemetry": [],
            "events": [],
            "metadata": {},
        }
    )


@pytest.fixture(name="bridge")
def bridge_fixture(tmp_path, run_result):
    from harbor.environments.base import BaseEnvironment, ExecResult
    from nemo_fabric.integrations.harbor import FabricAgent

    document = run_result.to_mapping()
    environment = AsyncMock(spec=BaseEnvironment)
    environment.exec.return_value = ExecResult(return_code=0, stdout="", stderr="")

    async def download(source, target):
        target.write_text(json.dumps(document), encoding="utf-8")

    environment.download_file.side_effect = download
    agent = FabricAgent(logs_dir=tmp_path, fabric_adapter_id="test.adapter")
    return agent, environment, document


@pytest.mark.parametrize("status", ["failed", "cancelled"])
@pytest.mark.parametrize("return_code", [0, 1])
async def test_bridge_rejects_unsuccessful_result_and_preserves_evidence(
    bridge, status, return_code
):
    from harbor.models.agent.context import AgentContext

    agent, environment, document = bridge
    document["status"] = status
    environment.exec.return_value.return_code = return_code
    context = AgentContext()

    with pytest.raises(RuntimeError, match=status):
        await agent.run("test", environment, context)

    assert agent._result_path.is_file()
    assert context.is_empty()
    agent.populate_context_post_run(context)
    assert context.metadata["fabric"]["status"] == status
    assert context.metadata["fabric"]["runtime_id"] == "runtime-test"


@pytest.mark.parametrize("failure", ["write", "replace"])
async def test_optional_evidence_write_failure_preserves_downloaded_result(
    bridge, monkeypatch, failure
):
    from pathlib import Path
    from harbor.models.agent.context import AgentContext

    agent, environment, document = bridge
    original_write, original_replace = Path.write_text, Path.replace
    injected = []

    def write(path, content, *args, **kwargs):
        if failure == "write" and path.parent.name.startswith(".fabric-result-"):
            original_write(path, "partial write", *args, **kwargs)
            injected.append(failure)
            raise OSError("injected write failure")
        return original_write(path, content, *args, **kwargs)

    def replace(path, target):
        if failure == "replace" and path.parent.name.startswith(".fabric-result-"):
            injected.append(failure)
            raise OSError("injected replace failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "write_text", write)
    monkeypatch.setattr(Path, "replace", replace)
    context = AgentContext()
    await agent.run("test", environment, context)
    assert injected == [failure]
    assert json.loads(agent._result_path.read_text()) == document
    assert not list(agent.logs_dir.glob(".fabric-result-*"))
    agent.populate_context_post_run(context)
    assert context.metadata["fabric"]["status"] == "succeeded"


async def test_cancellation_takes_precedence_over_error_classification(bridge):
    from harbor.models.agent.context import AgentContext

    agent, environment, document = bridge
    document["status"] = "cancelled"
    document["error"] = {
        "stage": "invoke",
        "code": "timeout",
        "message": "cancelled",
        "retryable": False,
    }
    with pytest.raises(RuntimeError, match="status: cancelled") as caught:
        await agent.run("test", environment, AgentContext())
    assert type(caught.value) is RuntimeError


@pytest.mark.parametrize("external_cancellation", [False, True])
async def test_invocation_cancellation_is_not_harbor_job_cancellation(
    bridge, tmp_path, monkeypatch, external_cancellation
):
    from functools import partial
    from logging import Logger
    from uuid import uuid4

    import harbor.job as job_module
    from harbor.job import Job
    from harbor.models.agent.context import AgentContext
    from harbor.models.job.config import JobConfig, RetryConfig
    from harbor.models.task.id import LocalTaskId
    from harbor.models.trial.config import TrialConfig
    from harbor.models.trial.paths import TrialPaths
    from harbor.models.trial.result import AgentInfo, TrialResult
    from harbor.trial.queue import TrialQueue
    from harbor.trial.trial import Trial

    agent, environment, document = bridge
    document["status"] = "cancelled"
    environment.exec.return_value.return_code = 1
    if external_cancellation:
        environment.exec.side_effect = asyncio.CancelledError("job interrupted")

    trials = []
    for name in ("cancelled-invocation", "healthy"):
        config = TrialConfig(task={"path": tmp_path}, trial_name=name)
        result = TrialResult(
            task_name="test",
            trial_name=name,
            trial_uri=(tmp_path / name).as_uri(),
            task_id=LocalTaskId(path=tmp_path),
            task_checksum="test",
            config=config,
            agent_info=AgentInfo(name="fabric", version="test"),
            agent_result=AgentContext(),
        )
        mock_trial = MagicMock(spec=Trial)
        mock_trial.logger = MagicMock(spec=Logger)
        mock_trial.config = config
        mock_trial.result = result
        mock_trial.paths = MagicMock(spec=TrialPaths)
        mock_trial.paths.exception_message_path = tmp_path / f"{name}-exception.txt"
        mock_trial.agent = agent
        mock_trial.agent_environment = environment
        mock_trial.user_agent = None
        mock_trial._now.side_effect = lambda: datetime.now(timezone.utc)
        mock_trial._phase_network_policy.return_value = AsyncExitStack()
        mock_trial._log_context.return_value = nullcontext()
        mock_trial._record_exception.side_effect = partial(
            Trial._record_exception, mock_trial
        )
        mock_trial.run.side_effect = partial(Trial.run, mock_trial)
        trials.append(mock_trial)

    cancelled, healthy = trials
    cancelled._run.side_effect = partial(
        Trial._run_agent_phase,
        cancelled,
        target=cancelled.result,
        instruction="test",
        timeout_sec=60,
        user=None,
    )
    cancelled._recover_outputs.side_effect = lambda: agent.populate_context_post_run(
        cancelled.result.agent_result
    )
    monkeypatch.setattr(Trial, "create", AsyncMock(side_effect=trials))
    monkeypatch.setattr(job_module, "record_command_job_id", MagicMock())
    monkeypatch.setattr(job_module, "capture_job_finished_async", MagicMock())

    mock_job = MagicMock(spec=Job)
    mock_job.config = JobConfig(quiet=True)
    mock_job._id = uuid4()
    mock_job._existing_job_result = None
    mock_job._existing_trial_results = []
    mock_job._existing_trial_configs = []
    mock_job._trial_configs = [trial.config for trial in trials]
    mock_job._remaining_trial_configs = mock_job._trial_configs
    mock_job._n_retries = 0
    mock_job._metrics = {"adhoc": []}
    mock_job._job_config_path = tmp_path / "config.json"
    mock_job._trial_queue = TrialQueue(
        n_concurrent=1, retry_config=RetryConfig(max_retries=0)
    )
    mock_job._run_trials_with_queue.side_effect = partial(
        Job._run_trials_with_queue, mock_job
    )
    summary_path = tmp_path / "job-result.json"
    mock_job._write_job_result.side_effect = lambda **_: summary_path.write_text(
        mock_job._job_result.model_dump_json(exclude={"trial_results"})
    )

    if external_cancellation:
        with pytest.raises(asyncio.CancelledError):
            await Job.run(mock_job)
        assert json.loads(summary_path.read_text())["finished_at"] is None
        return

    result = await Job.run(mock_job)
    assert result.finished_at is not None
    assert result.stats.n_completed_trials == 2
    assert result.stats.n_errored_trials == 1
    assert result.stats.n_cancelled_trials == 0
    assert cancelled.result.exception_info.exception_type == "RuntimeError"
    assert cancelled.result.agent_result.metadata["fabric"]["status"] == "cancelled"
    assert healthy.result.exception_info is None
    healthy.run.assert_awaited_once()
    assert json.loads(summary_path.read_text())["stats"]["n_completed_trials"] == 2


async def test_bridge_does_not_mask_process_failure_when_result_is_missing(bridge):
    from harbor.models.agent.context import AgentContext

    agent, environment, _ = bridge
    environment.exec.return_value.return_code = 2
    environment.exec.return_value.stderr = "runner import failed"
    environment.download_file.side_effect = FileNotFoundError("no result")
    with pytest.raises(RuntimeError, match="runner import failed"):
        await agent.run("test", environment, AgentContext())


async def test_bridge_does_not_accept_success_result_from_failed_process(bridge):
    from harbor.models.agent.context import AgentContext

    agent, environment, _ = bridge
    environment.exec.return_value.return_code = 2
    environment.exec.return_value.stderr = "post-run failure"
    with pytest.raises(RuntimeError, match="post-run failure"):
        await agent.run("test", environment, AgentContext())
    assert agent._result_path.is_file()


@pytest.mark.parametrize("invalid_result", [{}, {"telemetry": 1}])
async def test_bridge_does_not_mask_process_failure_with_invalid_result(
    bridge, invalid_result
):
    from harbor.models.agent.context import AgentContext

    agent, environment, document = bridge
    if invalid_result:
        document.update(invalid_result)
    else:
        document.clear()
    environment.exec.return_value.return_code = 2
    environment.exec.return_value.stderr = "runner failed before completing result"
    context = AgentContext()
    with pytest.raises(RuntimeError, match="runner failed before completing result"):
        await agent.run("test", environment, context)
    assert agent._result_path is None
    assert list(agent.logs_dir.glob("fabric-result-*.json"))
    agent.populate_context_post_run(context)
    assert context.is_empty()


@pytest.fixture(name="runner_cli")
def runner_cli_fixture(tmp_path, monkeypatch):
    from nemo_fabric.integrations.harbor.fabric_agent import build_harbor_config
    from nemo_fabric.integrations.harbor.models import FabricRunPayload

    spec = tmp_path / "spec.json"
    result = tmp_path / "result.json"
    payload = FabricRunPayload(
        config=build_harbor_config(adapter_id="test.adapter", workspace="/app"),
        config_base_dir="/app",
        request={"input": "test"},
    )
    spec.write_text(payload.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["runner", "--spec", str(spec), "--result", str(result)]
    )
    return result


@pytest.mark.parametrize("status", ["succeeded", "failed", "cancelled"])
def test_runner_writes_normalized_evidence_before_exit(
    runner_cli, monkeypatch, run_result, status
):
    from nemo_fabric import RunResult
    from nemo_fabric.integrations.harbor import runner

    document = run_result.to_mapping()
    document["status"] = status
    mock_run = AsyncMock(return_value=RunResult.from_mapping(document))
    monkeypatch.setattr(runner, "run", mock_run)
    if status == "succeeded":
        runner.main()
    else:
        with pytest.raises(SystemExit) as caught:
            runner.main()
        assert caught.value.code == 1
    assert json.loads(runner_cli.read_text())["status"] == status


async def test_bridge_rejects_error_even_with_succeeded_status(bridge):
    from harbor.models.agent.context import AgentContext

    agent, environment, document = bridge
    document["error"] = {
        "stage": "stop",
        "code": "runtime_stop_failed",
        "message": "stop failed",
        "retryable": False,
    }
    with pytest.raises(RuntimeError, match="stop failed"):
        await agent.run("test", environment, AgentContext())


def test_runner_rejects_error_even_with_succeeded_status(
    runner_cli, monkeypatch, run_result
):
    from nemo_fabric import RunResult
    from nemo_fabric.integrations.harbor import runner

    document = run_result.to_mapping()
    document["error"] = {
        "stage": "stop",
        "code": "runtime_stop_failed",
        "message": "stop failed",
        "retryable": False,
    }
    monkeypatch.setattr(
        runner, "run", AsyncMock(return_value=RunResult.from_mapping(document))
    )
    with pytest.raises(SystemExit) as caught:
        runner.main()
    assert caught.value.code == 1
    assert json.loads(runner_cli.read_text())["error"]["code"] == "runtime_stop_failed"


async def test_completed_wrong_answer_is_not_an_execution_failure(bridge):
    from harbor.models.agent.context import AgentContext

    agent, environment, document = bridge
    document["output"] = {"response": "incorrect answer"}
    document["metadata"]["verifier_reward"] = 0
    await agent.run("test", environment, AgentContext())
    assert agent._result_path.is_file()


@pytest.mark.parametrize(
    ("code", "exception"),
    [("timeout", TimeoutError), ("connection_failed", RuntimeError)],
)
async def test_bridge_classifies_only_canonical_deadline(bridge, code, exception):
    from harbor.models.agent.context import AgentContext

    agent, environment, document = bridge
    document["status"] = "failed"
    document["error"] = {
        "stage": "invoke",
        "code": code,
        "message": "timeout-looking diagnostic",
        "retryable": False,
    }
    with pytest.raises(exception) as caught:
        await agent.run("test", environment, AgentContext())
    assert type(caught.value) is exception


@pytest.mark.parametrize(
    ("code", "exception"),
    [("timeout", TimeoutError), ("configuration_failed", RuntimeError)],
)
async def test_bridge_retains_runner_failure_record(bridge, code, exception):
    from harbor.models.agent.context import AgentContext

    agent, environment, document = bridge
    document.clear()
    document["runner_error"] = {
        "stage": "start",
        "code": code,
        "message": "startup failure",
        "retryable": False,
    }
    environment.exec.return_value.return_code = 1
    context = AgentContext()
    with pytest.raises(exception):
        await agent.run("test", environment, context)
    agent.populate_context_post_run(context)
    assert context.metadata["fabric"]["runner_error"] == document["runner_error"]


@pytest.mark.parametrize("runner_failure", [False, True])
async def test_inner_deadline_enters_harbor_agent_timeout_path(
    bridge, runner_failure, tmp_path
):
    from harbor.models.trial.paths import TrialPaths
    from harbor.models.trial.result import TrialResult
    from harbor.trial.errors import AgentTimeoutError
    from harbor.trial.trial import Trial

    agent, environment, document = bridge
    error = {
        "stage": "invoke",
        "code": "timeout",
        "message": "inner invocation deadline",
        "retryable": False,
    }
    if runner_failure:
        document.clear()
        document["runner_error"] = error
    else:
        document.update(status="failed", error=error)
    environment.exec.return_value.return_code = 1

    mock_trial = MagicMock(spec=Trial)
    mock_trial.agent = agent
    mock_trial.agent_environment = environment
    mock_trial.user_agent = None
    mock_trial.paths = TrialPaths(trial_dir=tmp_path)
    mock_trial._now.return_value = datetime.now(timezone.utc)
    mock_trial._phase_network_policy = MagicMock(return_value=AsyncExitStack())
    mock_trial._log_context.return_value = nullcontext()
    target = MagicMock(spec=TrialResult)

    with pytest.raises(AgentTimeoutError) as caught:
        await Trial._run_agent_phase(
            mock_trial,
            target=target,
            instruction="test",
            timeout_sec=60,
            user=None,
        )

    assert type(caught.value.__cause__) is TimeoutError
    assert "inner invocation deadline" in str(caught.value.__cause__)
    assert target.agent_execution.finished_at is not None
    agent.populate_context_post_run(target.agent_result)
    failure = target.agent_result.metadata["fabric"]
    diagnostic = failure["runner_error"] if runner_failure else failure["error"]
    assert diagnostic["code"] == "timeout"


@pytest.mark.parametrize("stage", ["start", "invoke", "stop"])
def test_runner_serializes_structured_deadline(runner_cli, monkeypatch, stage):
    from nemo_fabric import FabricRuntimeError
    from nemo_fabric.integrations.harbor import runner

    monkeypatch.setattr(
        runner,
        "run",
        AsyncMock(
            side_effect=FabricRuntimeError("deadline", stage=stage, code="timeout")
        ),
    )
    with pytest.raises(SystemExit) as caught:
        runner.main()
    assert caught.value.code == 1
    assert json.loads(runner_cli.read_text())["runner_error"] == {
        "stage": stage,
        "code": "timeout",
        "message": "deadline",
        "retryable": False,
    }


def test_runner_unexpected_failure_does_not_expose_inputs(runner_cli, monkeypatch):
    from nemo_fabric.integrations.harbor import runner

    monkeypatch.setattr(
        runner, "run", AsyncMock(side_effect=ValueError("secret-sentinel"))
    )
    with pytest.raises(SystemExit):
        runner.main()
    evidence = runner_cli.read_text()
    assert "secret-sentinel" not in evidence
    assert json.loads(evidence)["runner_error"]["code"] is None


def test_runner_invalid_spec_preserves_safe_evidence(runner_cli):
    # argv was set by the fixture; invalid payloads must not persist Pydantic inputs.
    spec = runner_cli.with_name("spec.json")
    spec.write_text('{"unexpected":"secret-sentinel"}', encoding="utf-8")
    from nemo_fabric.integrations.harbor import runner

    with pytest.raises(SystemExit):
        runner.main()
    assert "secret-sentinel" not in runner_cli.read_text()
    assert "ValidationError" in runner_cli.read_text()


@pytest.mark.parametrize("stage", ["start", "invoke"])
def test_runner_retains_obtained_failure_evidence(runner_cli, monkeypatch, stage):
    from nemo_fabric import FabricRuntimeError
    from nemo_fabric.integrations.harbor import runner

    error = FabricRuntimeError(
        "original",
        stage=stage,
        code="timeout",
        details={
            "runtime_evidence": {
                "request_id": "request-test",
                "runtime_id": "runtime-test" if stage == "invoke" else None,
            }
        },
    )
    monkeypatch.setattr(runner, "run", AsyncMock(side_effect=error))
    with pytest.raises(SystemExit):
        runner.main()
    document = json.loads(runner_cli.read_text())
    assert document["runner_error"]["message"] == "original"
    assert document["evidence"]["request_id"] == "request-test"
    assert document["evidence"]["runtime_id"] == (
        "runtime-test" if stage == "invoke" else None
    )
    assert document["evidence"]["coverage"]["usage"] == "missing"
    assert document["evidence"]["usage"] is None


async def test_native_failure_retains_runtime_identity_and_cleanup(monkeypatch):
    from nemo_fabric import FabricRuntimeError
    from nemo_fabric.runtime import _run_native_lifecycle

    native = MagicMock()
    original = FabricRuntimeError(
        "original",
        stage="invoke",
        code="timeout",
        details={"runtime_evidence": {"usage": {"output_tokens": 7}}},
    )
    native.start_runtime.return_value = json.dumps({"runtime_id": "runtime-test"})
    native.invoke_runtime.side_effect = original
    native.stop_runtime.return_value = "[]"
    with pytest.raises(FabricRuntimeError) as caught:
        await _run_native_lifecycle(native, {}, {"request_id": "request-test"})
    assert caught.value is original
    assert caught.value.details["runtime_evidence"]["runtime_id"] == "runtime-test"
    assert caught.value.details["runtime_evidence"]["cleanup"] == "completed"
    assert caught.value.details["runtime_evidence"]["usage"]["output_tokens"] == 7
    native.stop_runtime.assert_called_once()


@pytest.mark.parametrize(
    "malformed",
    [
        {"system_stats": {"elapsed_seconds": -1}},
        {"usage": {"input_tokens": -1}},
        {
            "artifacts": [
                {},
                {"name": "trace", "kind": "atof", "path": "/logs/agent/trace.jsonl"},
            ]
        },
        {"events": [{}]},
        {"runtime_id": {"secret": "sentinel"}},
        {"usage": {"cost_usd": float("nan")}},
        {"execution": "x" * 262145},
    ],
)
def test_malformed_optional_evidence_preserves_original_failure(malformed):
    from nemo_fabric.integrations.harbor.models import FabricRunnerFailure

    failure = FabricRunnerFailure.model_validate(
        {
            "runner_error": {
                "stage": "invoke",
                "code": "timeout",
                "message": "original",
            },
            "evidence": malformed,
        }
    )
    assert failure.runner_error.code == "timeout"
    assert failure.runner_error.message == "original"
    assert any(
        state in ("invalid", "truncated")
        for state in failure.evidence.coverage.values()
    )
    if "artifacts" in malformed:
        assert len(failure.evidence.artifacts) == 1


def test_failure_evidence_without_observations_is_explicit(runner_cli, monkeypatch):
    from nemo_fabric.integrations.harbor import runner

    monkeypatch.setattr(runner, "run", AsyncMock(side_effect=RuntimeError("secret")))
    with pytest.raises(SystemExit):
        runner.main()
    evidence = json.loads(runner_cli.read_text())["evidence"]
    assert evidence["partial_result"] is None
    assert evidence["usage"] is None
    assert evidence["system_stats"] is None
    assert evidence["runtime_id"] is None
    assert evidence["events"] == []
    assert evidence["coverage"]["usage"] == "missing"


@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_cancellation_retains_completed_native_cleanup(cleanup_fails):
    import threading
    from nemo_fabric.runtime import _run_native_lifecycle

    ready, release = threading.Event(), threading.Event()
    native = MagicMock()
    native.start_runtime.return_value = json.dumps({"runtime_id": "runtime-test"})

    def invoke(*args):
        ready.set()
        assert release.wait(5)
        raise RuntimeError("invoke failed")

    native.invoke_runtime.side_effect = invoke
    if cleanup_fails:
        native.stop_runtime.side_effect = RuntimeError("stop failed")
    else:
        native.stop_runtime.return_value = "[]"
    task = asyncio.create_task(
        _run_native_lifecycle(native, {}, {"request_id": "request-test"})
    )
    assert await asyncio.to_thread(ready.wait, 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    native.stop_runtime.assert_called_once()
    evidence = caught.value._fabric_runtime_evidence
    assert evidence["runtime_id"] == "runtime-test"
    assert evidence["cleanup"] == ("failed" if cleanup_fails else "completed")


def test_runner_serializes_cancellation_without_turning_it_into_success(
    runner_cli, monkeypatch
):
    from nemo_fabric.integrations.harbor import runner

    error = asyncio.CancelledError()
    error._fabric_runtime_evidence = {
        "runtime_id": "runtime-test",
        "cleanup": "completed",
    }
    monkeypatch.setattr(runner, "run", AsyncMock(side_effect=error))
    with pytest.raises(asyncio.CancelledError):
        runner.main()
    document = json.loads(runner_cli.read_text())
    assert document["evidence"]["cleanup"] == "completed"
    assert "status" not in document


@pytest.mark.parametrize("malformed", [False, True])
def test_failure_recovers_only_matching_bounded_stats(tmp_path, malformed):
    from nemo_fabric import FabricRuntimeError
    from nemo_fabric.integrations.harbor.failure import failure_evidence
    from nemo_fabric.integrations.harbor.models import FabricRunPayload
    from nemo_fabric.integrations.harbor.fabric_agent import build_harbor_config

    payload = FabricRunPayload(
        config=build_harbor_config(adapter_id="test.adapter", workspace=str(tmp_path)),
        config_base_dir=tmp_path,
        request={"request_id": "request-test", "input": "test"},
    )
    payload.config.runtime.artifacts = str(tmp_path / "artifacts")
    home = tmp_path / "artifacts" / ".fabric" / "runtime-test" / "invocation-test"
    home.mkdir(parents=True)
    (home.parent / "startup").mkdir()
    stats = {
        "elapsed_seconds": 0.25,
        "before": {"status": "unavailable", "reason": "not available"},
        "after": {"status": "unsupported", "reason": "not supported"},
    }
    summary = {
        "schema_version": "fabric.system_stats/v1alpha1",
        "runtime_id": "runtime-test",
        "invocation_id": "invocation-test",
        "request_id": "wrong" if malformed else "request-test",
        "system_stats": stats,
    }
    (home / "system-stats-summary.json").write_text(json.dumps(summary))
    (home / "adapter-invocation.json").write_text("secret sentinel")
    error = FabricRuntimeError(
        "original", details={"runtime_evidence": {"runtime_id": "runtime-test"}}
    )
    evidence = failure_evidence(error, payload)
    assert len(evidence["artifacts"]) == 2
    assert "secret sentinel" not in json.dumps(evidence)
    if malformed:
        assert evidence["system_stats"] is None
        assert evidence["coverage"]["system_stats"] == "invalid"
    else:
        assert evidence["system_stats"]["elapsed_seconds"] == 0.25
        assert evidence["coverage"]["system_stats"] == "partial"
    assert evidence["usage"] is None
    assert evidence["coverage"]["usage"] == "missing"


async def test_failure_consumer_keeps_partial_usage_out_of_totals(bridge):
    from harbor.models.agent.context import AgentContext

    agent, environment, document = bridge
    document.clear()
    document.update(
        runner_error={"stage": "start", "code": "timeout", "message": "original"},
        evidence={"usage": {"output_tokens": 7}},
    )
    context = AgentContext(
        n_output_tokens=100,
        cost_usd=1,
        metadata={
            "caller": {"source": "kept"},
            "fabric": {"telemetry_validation": {"status": "missing"}},
        },
    )
    environment.exec.return_value.return_code = 1
    with pytest.raises(TimeoutError, match="original"):
        await agent.run("test", environment, context)
    agent.populate_context_post_run(context)
    assert context.metadata["fabric"]["evidence"]["usage"]["output_tokens"] == 7
    assert context.n_output_tokens is None
    assert context.cost_usd is None
    assert context.metadata["caller"] == {"source": "kept"}
    assert context.metadata["fabric"]["telemetry_validation"] == {"status": "missing"}
    specs = list(agent.logs_dir.glob("fabric-run-*.json"))
    assert len(specs) == 1
    spec = json.loads(specs[0].read_text())
    assert spec["config"]["harness"]["adapter_id"] == "test.adapter"
    assert spec["request"]["input"] == "test"


def test_runner_retains_result_obtained_before_publication_failure(
    runner_cli, monkeypatch, run_result
):
    from nemo_fabric import Fabric
    from nemo_fabric.integrations.harbor import runner

    monkeypatch.setattr(Fabric, "run", AsyncMock(return_value=run_result))
    monkeypatch.setattr(
        runner, "publish_telemetry_evidence", MagicMock(side_effect=OSError("secret"))
    )
    with pytest.raises(SystemExit):
        runner.main()
    document = json.loads(runner_cli.read_text())
    assert document["runner_error"]["message"] == "Fabric runner failed (OSError)"
    assert document["evidence"]["partial_result"]["output"] == {"response": "done"}
    assert "status" not in document
    assert document["evidence"]["coverage"]["partial_result"] == "partial"


def test_failure_evidence_preserves_cause_types_without_raw_diagnostics():
    from nemo_fabric import FabricRuntimeError
    from nemo_fabric.integrations.harbor.failure import failure_evidence

    try:
        try:
            raise OSError("credential sentinel")
        except OSError as cause:
            raise FabricRuntimeError("original", code="timeout") from cause
    except FabricRuntimeError as error:
        evidence = failure_evidence(error, None)
    assert evidence["exception_type"] == "FabricRuntimeError"
    assert evidence["cause_types"] == ["OSError"]
    assert "credential sentinel" not in json.dumps(evidence)


def test_oversized_optional_dimension_keeps_valid_siblings():
    from nemo_fabric.integrations.harbor.models import FabricRunnerFailure

    failure = FabricRunnerFailure.model_validate(
        {
            "runner_error": {"message": "original"},
            "evidence": {
                "usage": {"output_tokens": 7},
                "artifacts": [
                    {
                        "name": "trace",
                        "kind": "atof",
                        "path": "/logs/trace",
                        "metadata": {"large": "x" * 262145},
                    }
                ],
            },
        }
    )
    assert failure.evidence.usage["output_tokens"] == 7
    assert failure.evidence.artifacts == []
    assert failure.evidence.coverage["artifacts"] == "truncated"


def test_reference_limits_survive_producer_consumer_roundtrip():
    from nemo_fabric.integrations.harbor.models import FabricRunnerFailure

    failure = FabricRunnerFailure.model_validate(
        {
            "runner_error": {"message": "original"},
            "evidence": {
                "artifacts": [
                    {"name": "trace", "kind": "atof", "path": f"/logs/{n}"}
                    for n in range(70)
                ]
            },
        }
    )
    received = FabricRunnerFailure.model_validate_json(failure.model_dump_json())
    assert len(received.evidence.artifacts) == 64
    assert received.evidence.coverage["artifacts"] == "truncated"


def test_unsupported_failure_evidence_keeps_error_without_interpreting_payload():
    from nemo_fabric.integrations.harbor.models import FabricRunnerFailure

    failure = FabricRunnerFailure.model_validate(
        {
            "runner_error": {"message": "original"},
            "evidence": {"schema_version": "future", "usage": {"output_tokens": 100}},
        }
    )
    assert failure.runner_error.message == "original"
    assert failure.evidence.usage is None
    assert failure.evidence.coverage["usage"] == "unsupported"


def test_malformed_identity_does_not_drop_valid_observations():
    from nemo_fabric.integrations.harbor.models import FabricRunnerFailure

    failure = FabricRunnerFailure.model_validate(
        {
            "runner_error": {"message": "original"},
            "evidence": {"runtime_id": {}, "usage": {"output_tokens": 7}},
        }
    )
    assert failure.evidence.runtime_id is None
    assert failure.evidence.usage["output_tokens"] == 7
    assert failure.evidence.coverage["identity"] == "invalid"
