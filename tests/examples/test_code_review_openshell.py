# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initial scope: separate OpenShell calls retain one runtime for two reviews."""

import json
import stat
import sys
from unittest.mock import MagicMock

import pytest
from nemo_fabric import Fabric, FabricConfig

from examples.code_review_agent.config import openclaw_config


@pytest.fixture(name="mock_client")
def client_fixture():
    client = MagicMock()

    def execute(sandbox, command, **kwargs):
        operation = command[1] if command[0] == "fabric-agent-ctl" else "serve"
        if operation == "serve":
            return MagicMock(exit_code=0, stdout="", stderr="")
        result = {"runtime_id": "runtime-1"}
        if operation == "invoke":
            result = {
                "runtime_id": "runtime-1",
                "fabric_result": {
                    "status": "succeeded",
                    "runtime_id": "runtime-1",
                    "output": {"response": "calculator.py: answer() returns 41."},
                },
            }
        response = {
            "operation": operation,
            "status": "succeeded",
            "changed": None if operation == "invoke" else True,
            "result": result,
            "error": None,
        }
        return MagicMock(exit_code=0, stdout=json.dumps(response) + "\n", stderr="")

    client.exec.side_effect = execute
    return client


def test_reviews_use_separate_calls_and_one_runtime_without_changing_input_config(
    mock_client,
):
    from examples.code_review_agent.openshell import OpenShellCodeReview

    config = openclaw_config()
    before = config.to_mapping()
    reviewer = OpenShellCodeReview(mock_client, sandbox="review", workspace="test")
    report = reviewer.run(config, ["Review calculator.py", "Explain the finding"])
    commands = [call.args[1] for call in mock_client.exec.call_args_list]
    assert commands[1] == ["sh", "-c", "test -S /sandbox/fabric.sock"]
    assert [command[1] for command in commands[2:]] == [
        "start",
        "invoke",
        "invoke",
        "stop",
    ]
    assert len(commands) == 6
    assert "fabric-agent-ctl serve" in commands[0][2]
    for call in mock_client.exec.call_args_list:
        assert call.args[0] == "review"
        assert call.kwargs["workspace"] == "test"
        assert call.kwargs["no_login_shell"] is True
        assert call.kwargs["timeout_seconds"] <= 300
    for command in commands[3:]:
        assert command[command.index("--runtime-id") + 1] == "runtime-1"
    staged_config = json.loads(mock_client.exec.call_args_list[2].kwargs["stdin"])
    assert staged_config["harness"]["adapter_id"] == "nvidia.fabric.openclaw"
    assert staged_config["environment"]["workspace"] == "./code-review/repos/my-service"
    assert config.to_mapping() == before
    assert report["runtime_id"] == "runtime-1"
    assert len(report["reviews"]) == 2
    assert report["reviews"][0]["output"]["response"].startswith("calculator.py")
    assert report["stop"]["status"] == "succeeded"


def test_demo_waits_for_socket_and_starts_without_a_health_command(mock_client):
    from examples.code_review_agent.openshell import OpenShellCodeReview

    report = OpenShellCodeReview(mock_client, sandbox="review", workspace="test").run(
        openclaw_config(), ["Review", "Follow up"]
    )
    commands = [call.args[1] for call in mock_client.exec.call_args_list]
    assert all(command[:2] != ["fabric-agent-ctl", "check"] for command in commands)
    assert commands[1] == ["sh", "-c", "test -S /sandbox/fabric.sock"]
    start = next(
        command for command in commands if command[:2] == ["fabric-agent-ctl", "start"]
    )
    assert "--expected-generation" not in start
    stop = commands[-1]
    assert "--expected-generation" not in stop
    assert report["status"] == "succeeded"


@pytest.mark.parametrize(
    "failure", ["structured_error", "malformed_response", "timeout"]
)
def test_failed_invocation_is_not_retried_and_still_attempts_stop(mock_client, failure):
    from examples.code_review_agent.openshell import OpenShellCodeReview

    execute = mock_client.exec.side_effect

    def invoke_failure(sandbox, command, **kwargs):
        if command[:2] == ["fabric-agent-ctl", "invoke"]:
            if failure == "timeout":
                raise TimeoutError("PRIVATE")
            if failure == "malformed_response":
                return MagicMock(exit_code=0, stdout="PRIVATE\n{}\n", stderr="PRIVATE")
            return MagicMock(
                exit_code=1,
                stdout=json.dumps(
                    {
                        "operation": "invoke",
                        "status": "failed",
                        "changed": None,
                        "result": None,
                        "error": {"code": "invoke_failed", "effects": "unknown"},
                    }
                )
                + "\n",
                stderr="PRIVATE",
            )
        return execute(sandbox, command, **kwargs)

    mock_client.exec.side_effect = invoke_failure
    report = OpenShellCodeReview(mock_client, sandbox="review", workspace="test").run(
        openclaw_config(), ["Review", "Follow up"]
    )
    commands = [call.args[1] for call in mock_client.exec.call_args_list]
    assert (
        sum(command[:2] == ["fabric-agent-ctl", "invoke"] for command in commands) == 1
    )
    assert commands[-1][:2] == ["fabric-agent-ctl", "stop"]
    assert report["status"] == "failed"
    assert report["error"]["effects"] == "unknown"
    assert "PRIVATE" not in json.dumps(report)


def test_stop_failure_is_reported_without_losing_review_results(mock_client):
    from examples.code_review_agent.openshell import OpenShellCodeReview

    execute = mock_client.exec.side_effect

    def stop_failure(sandbox, command, **kwargs):
        if command[:2] == ["fabric-agent-ctl", "stop"]:
            return MagicMock(
                exit_code=1,
                stdout=json.dumps(
                    {
                        "operation": "stop",
                        "status": "failed",
                        "changed": None,
                        "result": None,
                        "error": {"code": "stop_failed", "effects": "unknown"},
                    }
                )
                + "\n",
                stderr="",
            )
        return execute(sandbox, command, **kwargs)

    mock_client.exec.side_effect = stop_failure
    report = OpenShellCodeReview(mock_client, sandbox="review", workspace="test").run(
        openclaw_config(), ["Review", "Follow up"]
    )
    assert report["status"] == "failed"
    assert len(report["reviews"]) == 2
    assert report["stop"]["error"]["code"] == "stop_failed"


def test_uncertain_start_does_not_guess_runtime_identity_or_retry(mock_client):
    from examples.code_review_agent.openshell import OpenShellCodeReview

    execute = mock_client.exec.side_effect

    def start_failure(sandbox, command, **kwargs):
        if command[:2] == ["fabric-agent-ctl", "start"]:
            raise TimeoutError("PRIVATE")
        return execute(sandbox, command, **kwargs)

    mock_client.exec.side_effect = start_failure
    report = OpenShellCodeReview(mock_client, sandbox="review", workspace="test").run(
        openclaw_config(), ["Review", "Follow up"]
    )
    commands = [call.args[1] for call in mock_client.exec.call_args_list]
    assert len(commands) == 3
    assert report["runtime_id"] is None
    assert report["error"]["effects"] == "unknown"
    assert "PRIVATE" not in json.dumps(report)


def test_cli_uses_the_attached_openai_provider_credentials(
    mock_client, monkeypatch, capsys, tmp_path
):
    from examples.code_review_agent import openshell

    module = MagicMock()
    module.SandboxClient.return_value.__enter__.return_value = mock_client
    monkeypatch.setitem(sys.modules, "openshell", module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review",
            "--gateway",
            "127.0.0.1:18072",
            "--sandbox",
            "review",
            "--workspace",
            "test",
            "--input",
            "Review",
            "--input",
            "Follow up",
        ],
    )
    assert openshell.main() == 0
    model = json.loads(mock_client.exec.call_args_list[2].kwargs["stdin"])["models"][
        "default"
    ]
    assert model["api_key_env"] == "OPENAI_API_KEY"
    assert model["base_url"] == "https://integrate.api.nvidia.com/v1"
    # Validate the staged model against the real public SDK/installed descriptor.
    planned = Fabric().plan(
        FabricConfig.model_validate(
            {
                "metadata": {"name": "review"},
                "harness": {"adapter_id": "nvidia.fabric.openclaw"},
                "models": {"default": model},
            }
        ),
        base_dir=tmp_path,
    )
    assert (
        planned.get("adapter_descriptor")["descriptor"]["adapter_id"]
        == "nvidia.fabric.openclaw"
    )
    module.SandboxClient.assert_called_once_with("127.0.0.1:18072")
    assert json.loads(capsys.readouterr().out)["status"] == "succeeded"


def test_socket_wait_respects_remaining_deadline(mock_client, monkeypatch):
    from examples.code_review_agent import openshell

    clock = [0.0]
    monkeypatch.setattr(openshell.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(openshell.time, "sleep", lambda seconds: None)
    execute = mock_client.exec.side_effect

    def unavailable(sandbox, command, **kwargs):
        if command == ["sh", "-c", "test -S /sandbox/fabric.sock"]:
            clock[0] = 3.0
            return MagicMock(exit_code=1, stdout="", stderr="PRIVATE")
        return execute(sandbox, command, **kwargs)

    mock_client.exec.side_effect = unavailable
    report = openshell.OpenShellCodeReview(
        mock_client, sandbox="review", workspace="test", startup_seconds=2
    ).run(openclaw_config(), ["Review", "Follow up"])
    assert mock_client.exec.call_args_list[1].kwargs["timeout_seconds"] <= 2
    assert len(mock_client.exec.call_args_list) == 2
    assert report["status"] == "failed"
    assert report["runtime_id"] is None
    assert "PRIVATE" not in json.dumps(report)


@pytest.mark.parametrize("invocation_fails", [False, True])
def test_cli_records_success_and_failure_without_raw_diagnostics(
    mock_client, monkeypatch, capsys, tmp_path, invocation_fails
):
    from examples.code_review_agent import openshell

    execute = mock_client.exec.side_effect

    def execute_review(sandbox, command, **kwargs):
        if invocation_fails and command[:2] == ["fabric-agent-ctl", "invoke"]:
            raise TimeoutError("PRIVATE_DIAGNOSTIC")
        result = execute(sandbox, command, **kwargs)
        result.stderr = "PRIVATE_DIAGNOSTIC"
        return result

    mock_client.exec.side_effect = execute_review
    module = MagicMock()
    module.SandboxClient.from_active_cluster.return_value.__enter__.return_value = (
        mock_client
    )
    monkeypatch.setitem(sys.modules, "openshell", module)
    output_dir = tmp_path / "recorded-review"
    prompts = ["Review calculator.py and remember cedar-1234", "Recall the tag"]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review",
            "--sandbox",
            "review",
            "--workspace",
            "test",
            "--output-dir",
            str(output_dir),
            "--input",
            prompts[0],
            "--input",
            prompts[1],
        ],
    )

    assert openshell.main() == (1 if invocation_fails else 0)

    recorded = json.loads((output_dir / "report.json").read_text())
    transcript = (output_dir / "transcript.md").read_text()
    assert recorded == json.loads(capsys.readouterr().out)
    assert recorded["sandbox"] == "review"
    assert recorded["workspace"] == "test"
    assert recorded["prompts"] == prompts
    assert recorded["model"]["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert recorded["started_at"].endswith("+00:00")
    assert recorded["elapsed_seconds"] >= 0
    expected = ["serve", "wait_socket", "start", "invoke"]
    if not invocation_fails:
        expected.append("invoke")
    expected.append("stop")
    assert [call["operation"] for call in recorded["commands"]] == expected
    assert all(call["elapsed_seconds"] >= 0 for call in recorded["commands"])
    assert recorded["commands"][3]["exit_code"] == (None if invocation_fails else 0)
    assert "runtime-1" in transcript
    assert prompts[0] in transcript
    assert (
        "command_outcome_unknown" if invocation_fails else "calculator.py"
    ) in transcript
    assert "PRIVATE_DIAGNOSTIC" not in json.dumps(recorded) + transcript
    # Windows does not implement Unix permission modes through chmod/stat.
    if sys.platform != "win32":
        assert stat.S_IMODE(output_dir.stat().st_mode) == 0o700
        for name in ("report.json", "transcript.md"):
            assert stat.S_IMODE((output_dir / name).stat().st_mode) == 0o600


def test_cli_refuses_to_overwrite_recorded_results_before_sandbox_work(
    mock_client, monkeypatch, tmp_path, capsys
):
    from examples.code_review_agent import openshell

    module = MagicMock()
    monkeypatch.setitem(sys.modules, "openshell", module)
    output_dir = tmp_path / "existing-review"
    output_dir.mkdir()
    existing = output_dir / "report.json"
    existing.write_text("KEEP")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review",
            "--sandbox",
            "review",
            "--workspace",
            "test",
            "--output-dir",
            str(output_dir),
            "--input",
            "Review",
            "--input",
            "Follow up",
        ],
    )

    with pytest.raises(SystemExit) as caught:
        openshell.main()

    assert caught.value.code == 2
    assert "already exists" in capsys.readouterr().err
    assert existing.read_text() == "KEEP"
    module.SandboxClient.from_active_cluster.assert_not_called()
    mock_client.exec.assert_not_called()


def test_command_record_includes_readiness_polls_without_repeating_agent_work(
    mock_client, monkeypatch
):
    from examples.code_review_agent import openshell

    execute = mock_client.exec.side_effect
    checks = 0

    def delayed_ready(sandbox, command, **kwargs):
        nonlocal checks
        if command == ["sh", "-c", "test -S /sandbox/fabric.sock"]:
            checks += 1
            if checks == 1:
                return MagicMock(exit_code=1, stdout="", stderr="PRIVATE")
        return execute(sandbox, command, **kwargs)

    monkeypatch.setattr(openshell.time, "sleep", lambda seconds: None)
    mock_client.exec.side_effect = delayed_ready
    report = openshell.OpenShellCodeReview(
        mock_client, sandbox="review", workspace="test"
    ).run(openclaw_config(), ["Review", "Follow up"])

    assert report["status"] == "succeeded"
    assert [call["operation"] for call in report["commands"]] == [
        "serve",
        "wait_socket",
        "wait_socket",
        "start",
        "invoke",
        "invoke",
        "stop",
    ]
    assert report["commands"][1]["exit_code"] == 1
    assert report["commands"][2]["exit_code"] == 0
    assert "PRIVATE" not in json.dumps(report)


def test_failed_review_preserves_normalized_error_partial_output_and_artifacts(
    mock_client, tmp_path
):
    from examples.code_review_agent import openshell

    execute = mock_client.exec.side_effect
    failed = {
        "runtime_id": "runtime-1",
        "status": "failed",
        "output": {"response": "calculator.py: partial finding"},
        "error": {
            "code": "model_rate_limited",
            "message": "Model rate limited",
            "retryable": True,
        },
        "artifacts": {"files": [{"name": "partial-review", "path": "review.txt"}]},
    }

    def failed_review(sandbox, command, **kwargs):
        if command[:2] == ["fabric-agent-ctl", "invoke"]:
            return MagicMock(
                exit_code=1,
                stdout=json.dumps(
                    {
                        "operation": "invoke",
                        "status": "failed",
                        "changed": None,
                        "result": {"runtime_id": "runtime-1", "fabric_result": failed},
                        "error": {"code": "invoke_failed", "effects": "unknown"},
                    }
                )
                + "\n",
                stderr="PRIVATE",
            )
        return execute(sandbox, command, **kwargs)

    mock_client.exec.side_effect = failed_review
    report = openshell.OpenShellCodeReview(
        mock_client, sandbox="review", workspace="test"
    ).run(openclaw_config(), ["Review", "Follow up"])
    assert report["status"] == "failed"
    assert report["reviews"] == [failed]
    assert report["stop"]["status"] == "succeeded"
    assert [call["operation"] for call in report["commands"]].count("invoke") == 1
    openshell._write_record(tmp_path, report)
    transcript = (tmp_path / "transcript.md").read_text()
    assert "partial finding" in transcript
    assert "model_rate_limited" in transcript
    assert "review.txt" in transcript
    assert "PRIVATE" not in json.dumps(report) + transcript


@pytest.mark.parametrize(
    "output", [None, "text", [], {"response": None}, {"response": ["text"]}]
)
def test_cli_records_failed_review_with_null_or_non_text_output(
    mock_client, tmp_path, monkeypatch, capsys, output
):
    from examples.code_review_agent import openshell

    execute = mock_client.exec.side_effect
    failed = {
        "runtime_id": "runtime-1",
        "status": "failed",
        "output": output,
        "error": {"code": "model_unavailable", "message": "Model unavailable"},
    }

    def failed_review(sandbox, command, **kwargs):
        if command[:2] == ["fabric-agent-ctl", "invoke"]:
            return MagicMock(
                exit_code=1,
                stdout=json.dumps(
                    {
                        "operation": "invoke",
                        "status": "failed",
                        "changed": None,
                        "result": {"runtime_id": "runtime-1", "fabric_result": failed},
                        "error": {"code": "invoke_failed", "effects": "unknown"},
                    }
                )
                + "\n",
                stderr="PRIVATE",
            )
        return execute(sandbox, command, **kwargs)

    mock_client.exec.side_effect = failed_review
    module = MagicMock()
    module.SandboxClient.from_active_cluster.return_value.__enter__.return_value = (
        mock_client
    )
    monkeypatch.setitem(sys.modules, "openshell", module)
    output_dir = tmp_path / "recorded-failure"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review",
            "--sandbox",
            "review",
            "--workspace",
            "test",
            "--output-dir",
            str(output_dir),
            "--input",
            "Review",
            "--input",
            "Follow up",
        ],
    )

    assert openshell.main() == 1
    report = json.loads((output_dir / "report.json").read_text())
    transcript = (output_dir / "transcript.md").read_text()
    assert report == json.loads(capsys.readouterr().out)
    assert report["reviews"] == [failed]
    assert report["stop"]["status"] == "succeeded"
    assert report["error"]["code"] == "invoke_failed"
    assert [call["operation"] for call in report["commands"]].count("invoke") == 1
    assert "No text response was returned." in transcript
    assert "model_unavailable" in transcript
    assert "PRIVATE" not in json.dumps(report) + transcript


def test_transcript_rendering_failure_preserves_json_report(
    mock_client, tmp_path, monkeypatch
):
    from examples.code_review_agent import openshell

    report = openshell.OpenShellCodeReview(
        mock_client, sandbox="review", workspace="test"
    ).run(openclaw_config(), ["Review", "Follow up"])
    monkeypatch.setattr(
        openshell,
        "_render_transcript",
        MagicMock(side_effect=ValueError("PRIVATE")),
        raising=False,
    )
    openshell._write_record(tmp_path, report)
    assert json.loads((tmp_path / "report.json").read_text()) == report
    transcript = (tmp_path / "transcript.md").read_text()
    assert "See report.json" in transcript
    assert "PRIVATE" not in transcript
