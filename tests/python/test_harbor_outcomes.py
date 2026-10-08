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
async def test_inner_deadline_enters_harbor_agent_timeout_path(bridge, runner_failure):
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
