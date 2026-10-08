# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from nemo_fabric import RunResult
from nemo_fabric.integrations.harbor.telemetry import TelemetryValidationError
from nemo_fabric.integrations.harbor.telemetry import publish_telemetry_evidence


def make_result(*artifacts: dict[str, str]) -> RunResult:
    return RunResult.from_mapping(
        {
            "agent_name": "harbor-demo",
            "harness": "hermes",
            "adapter_kind": "python",
            "adapter_id": "nvidia.fabric.hermes",
            "status": "succeeded",
            "runtime_id": "runtime-1",
            "invocation_id": "invocation-1",
            "request_id": "request-1",
            "output": {"response": "done"},
            "error": None,
            "artifacts": {"root": None, "artifacts": list(artifacts)},
            "telemetry": [],
            "events": [],
            "metadata": {},
        }
    )


def artifact(name: str, kind: str, path: Path) -> dict[str, str]:
    return {
        "name": name,
        "kind": kind,
        "path": str(path),
        "media_type": "application/json",
    }


def task_artifact(name: str, kind: str, path: Path, logs_dir: Path) -> dict[str, str]:
    return artifact(name, kind, Path("/logs/agent") / path.relative_to(logs_dir))


@pytest.fixture(name="finalized_relay_run")
def finalized_relay_run_fixture(tmp_path):
    logs = tmp_path / "agent"
    directory = logs / "fabric-artifacts" / "relay" / "runtime-1"
    directory.mkdir(parents=True)
    path = directory / "trajectory-session.atif.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "session_id": "relay-session",
                "agent": {"name": "pi", "version": "1.0"},
                "steps": [{"step_id": 1, "source": "agent", "message": "done"}],
            }
        )
    )
    mapping = make_result().to_mapping()
    mapping["artifacts"]["root"] = "/logs/agent/fabric-artifacts"
    plugin = logs / "fabric-artifacts" / ".fabric" / "runtime-1" / "plugins.toml"
    plugin.parent.mkdir(parents=True)
    plugin.write_text("""
[[components]]
kind = "observability"
enabled = true
[components.config.atif]
enabled = true
output_directory = "/logs/agent/fabric-artifacts/relay/runtime-1"
filename_template = "trajectory-{session_id}.atif.json"
""")
    mapping["output"]["relay_runtime"] = {
        "plugin_config_path": "/logs/agent/fabric-artifacts/.fabric/runtime-1/plugins.toml"
    }
    mapping["telemetry"] = [
        {
            "provider": "relay",
            "kind": "trace",
            "metadata": {
                "relay_config": {
                    "components": [
                        {
                            "kind": "observability",
                            "enabled": True,
                            "config": {
                                "atif": {
                                    "enabled": True,
                                    "output_directory": "/logs/agent/fabric-artifacts/relay/runtime-1",
                                    "filename_template": "trajectory-{session_id}.atif.json",
                                }
                            },
                        }
                    ]
                }
            },
        }
    ]
    return logs, path, mapping


@pytest.mark.parametrize("runtime_stopped", [False, True])
def test_finalized_relay_atif_collected_only_after_shutdown(
    finalized_relay_run, runtime_stopped
):
    logs, path, mapping = finalized_relay_run
    summary = publish_telemetry_evidence(
        RunResult.from_mapping(mapping),
        logs,
        strict=True,
        runtime_stopped=runtime_stopped,
    )
    assert (logs / "trajectory.json").exists() is runtime_stopped
    assert summary["atif"]["files"] == ([str(path)] if runtime_stopped else [])


@pytest.mark.parametrize(
    "mode",
    [
        "other-runtime",
        "symlink",
        "ambiguous",
        "malformed",
        "plugin-symlink",
        "plugin-malformed",
        "nested-template",
    ],
)
def test_finalized_relay_atif_rejects_unsafe_or_invalid_artifacts(
    finalized_relay_run, mode
):
    logs, path, mapping = finalized_relay_run
    plugin = logs / "fabric-artifacts" / ".fabric" / "runtime-1" / "plugins.toml"
    atif = mapping["telemetry"][0]["metadata"]["relay_config"]["components"][0][
        "config"
    ]["atif"]
    if mode == "other-runtime":
        atif["output_directory"] = "/logs/agent/fabric-artifacts/relay/runtime-other"
        plugin = logs / "fabric-artifacts" / ".fabric" / "runtime-1" / "plugins.toml"
        plugin.write_text(
            plugin.read_text().replace("relay/runtime-1", "relay/runtime-other")
        )
    elif mode == "symlink":
        target = logs / "outside.json"
        path.rename(target)
        path.symlink_to(target)
    elif mode == "ambiguous":
        path.with_name("trajectory-second.atif.json").write_bytes(path.read_bytes())
    elif mode == "plugin-symlink":
        target = logs / "outside-plugins.toml"
        plugin.rename(target)
        plugin.symlink_to(target)
    elif mode == "plugin-malformed":
        plugin.write_text("[")
    elif mode == "nested-template":
        plugin.write_text(
            plugin.read_text().replace("trajectory-{session_id}", "nested/{session_id}")
        )
    else:
        path.write_text("[]")
    with pytest.raises(TelemetryValidationError):
        publish_telemetry_evidence(
            RunResult.from_mapping(mapping), logs, strict=True, runtime_stopped=True
        )
    assert not (logs / "trajectory.json").exists()


@pytest.mark.parametrize("relay_runtime", [None, [], "invalid", 1])
def test_finalized_relay_ignores_unrecognized_output(
    finalized_relay_run, relay_runtime
):
    logs, _, mapping = finalized_relay_run
    mapping["output"]["relay_runtime"] = relay_runtime
    summary = publish_telemetry_evidence(
        RunResult.from_mapping(mapping), logs, strict=True, runtime_stopped=True
    )
    assert summary["status"] == "not_emitted"
    assert not (logs / "trajectory.json").exists()


@pytest.mark.parametrize(
    "config",
    [
        "components = 1",
        "components = [1]",
        '[[components]]\nkind = "observability"\nconfig = 1',
        '[[components]]\nkind = "observability"\n[components.config]\natif = 1',
    ],
)
def test_finalized_relay_malformed_tables_are_telemetry_failures(
    finalized_relay_run, config
):
    logs, _, mapping = finalized_relay_run
    plugin = logs / "fabric-artifacts" / ".fabric" / "runtime-1" / "plugins.toml"
    plugin.write_text(config)
    summary = publish_telemetry_evidence(
        RunResult.from_mapping(mapping), logs, runtime_stopped=True
    )
    assert summary["status"] == "failed"
    assert not (logs / "trajectory.json").exists()


@pytest.mark.parametrize("filename", ["a" * 40 + ".json", "a" * 200 + ".txt"])
def test_finalized_relay_adjacent_template_placeholders(finalized_relay_run, filename):
    logs, path, mapping = finalized_relay_run
    path.rename(path.with_name(filename))
    plugin = logs / "fabric-artifacts" / ".fabric" / "runtime-1" / "plugins.toml"
    plugin.write_text(
        plugin.read_text().replace(
            "trajectory-{session_id}.atif.json", "{session_id}" * 40 + ".json"
        )
    )
    summary = publish_telemetry_evidence(
        RunResult.from_mapping(mapping), logs, strict=True, runtime_stopped=True
    )
    assert (summary["status"] == "succeeded") is filename.endswith(".json")


def test_publish_telemetry_validates_and_promotes_atif(tmp_path: Path):
    logs = tmp_path / "agent"
    logs.mkdir()
    atof = logs / "events.atof.jsonl"
    atof.write_text(
        json.dumps(
            {
                "atof_version": "1.0",
                "kind": "event",
                "name": "agent.end",
                "timestamp": "2026-07-13T00:00:00Z",
                "uuid": "019f616c-5eb1-7c92-928f-b4130bd4a519",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    atif = logs / "relay" / "trajectory-run.atif.json"
    atif.parent.mkdir()
    atif.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "session_id": "runtime-1",
                "agent": {"name": "fabric", "version": "0.1.0"},
                "steps": [{"step_id": 1, "source": "agent", "message": "done"}],
                "final_metrics": {
                    "total_prompt_tokens": 12,
                    "total_completion_tokens": 4,
                    "total_steps": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    result = make_result(
        task_artifact("atof", "atof", atof, logs),
        task_artifact("atif", "atif", atif, logs),
    )

    summary = publish_telemetry_evidence(result, logs, strict=True)

    assert summary["status"] == "succeeded"
    assert summary["atof"]["records"] == 1
    assert summary["atif"]["steps"] == 1
    assert summary["atif"]["validator"] == "fabric_structural"
    assert (logs / "trajectory.json").read_bytes() == atif.read_bytes()
    assert json.loads((logs / "telemetry-validation.json").read_text())["status"] == (
        "succeeded"
    )


def test_publish_telemetry_accepts_relay_owned_atif_session_id(tmp_path: Path):
    logs = tmp_path / "agent"
    logs.mkdir()
    atif = logs / "relay-trajectory.json"
    atif.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "session_id": "019f616c-5eb1-7c92-928f-b4130bd4a519",
                "agent": {"name": "fabric", "version": "0.1.0"},
                "steps": [{"step_id": 1, "source": "agent", "message": "done"}],
            }
        ),
        encoding="utf-8",
    )

    summary = publish_telemetry_evidence(
        make_result(task_artifact("atif", "atif", atif, logs)),
        logs,
        strict=True,
        harbor_session_id="harbor-session-1",
    )

    assert summary["atif"]["session_id"] == "019f616c-5eb1-7c92-928f-b4130bd4a519"
    assert summary["atif"]["promoted"].endswith("trajectory.json")


def test_publish_telemetry_resolves_collected_task_artifact_paths(tmp_path: Path):
    logs = tmp_path / "agent"
    collected = logs / "fabric-artifacts" / "trajectory.json"
    collected.parent.mkdir(parents=True)
    collected.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "session_id": "relay-session-1",
                "agent": {"name": "fabric", "version": "0.1.0"},
                "steps": [{"step_id": 1, "source": "agent", "message": "done"}],
            }
        ),
        encoding="utf-8",
    )
    result = make_result(
        artifact(
            "atif",
            "atif",
            Path("/logs/agent/fabric-artifacts/trajectory.json"),
        )
    )

    summary = publish_telemetry_evidence(result, logs, strict=True)

    assert summary["atif"]["files"] == [str(collected)]
    assert (logs / "trajectory.json").is_file()


def test_publish_telemetry_rejects_collected_path_escape(tmp_path: Path):
    result = make_result(
        artifact(
            "atif",
            "atif",
            Path("/logs/agent/../outside/trajectory.json"),
        )
    )

    with pytest.raises(TelemetryValidationError, match="escapes /logs/agent"):
        publish_telemetry_evidence(result, tmp_path / "agent", strict=True)


def test_artifact_path_escape_is_rejected_without_host_path_remapping():
    from nemo_fabric.integrations.harbor.telemetry import _resolve_artifact_path

    with pytest.raises(TelemetryValidationError, match="escapes /logs/agent"):
        _resolve_artifact_path(
            Path("/logs/agent/../outside/trajectory.json"),
            Path("/logs/agent"),
        )


def test_publish_telemetry_rejects_absolute_path_outside_task_logs(tmp_path: Path):
    result = make_result(artifact("atif", "atif", Path("/etc/passwd")))

    with pytest.raises(TelemetryValidationError, match="escapes /logs/agent"):
        publish_telemetry_evidence(result, tmp_path / "agent", strict=True)


@pytest.mark.parametrize("operation", ["mkdir", "write_text"])
def test_publish_telemetry_ignores_evidence_write_failure_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
):
    def fail_write(*args, **kwargs):
        raise OSError("evidence storage unavailable")

    monkeypatch.setattr(Path, operation, fail_write)

    summary = publish_telemetry_evidence(make_result(), tmp_path / "agent")

    assert summary["status"] == "not_emitted"


def test_publish_telemetry_propagates_evidence_write_failure_in_strict_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_write(*args, **kwargs):
        raise OSError("evidence storage unavailable")

    monkeypatch.setattr(Path, "write_text", fail_write)

    with pytest.raises(OSError, match="evidence storage unavailable"):
        publish_telemetry_evidence(make_result(), tmp_path / "agent", strict=True)


def test_publish_telemetry_records_quality_failure_without_changing_run(tmp_path: Path):
    logs = tmp_path / "agent"
    logs.mkdir()
    malformed = logs / "events.atof.jsonl"
    malformed.write_text("{}\n", encoding="utf-8")
    result = make_result(task_artifact("atof", "atof", malformed, logs))

    summary = publish_telemetry_evidence(result, logs)

    assert summary["status"] == "failed"
    assert "ATOF record missing" in summary["error"]

    with pytest.raises(TelemetryValidationError, match="ATOF record missing"):
        publish_telemetry_evidence(result, logs, strict=True)


def test_publish_telemetry_rejects_ambiguous_atif(tmp_path: Path):
    logs = tmp_path / "agent"
    logs.mkdir()
    first = logs / "first.json"
    second = logs / "second.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    result = make_result(
        task_artifact("atif-1", "atif", first, logs),
        task_artifact("atif-2", "atif", second, logs),
    )

    with pytest.raises(TelemetryValidationError, match="at most one ATIF"):
        publish_telemetry_evidence(result, logs, strict=True)


def test_publish_telemetry_rejects_structurally_invalid_atif(tmp_path: Path):
    logs = tmp_path / "agent"
    logs.mkdir()
    atif = logs / "invalid-trajectory.json"
    atif.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "session_id": "runtime-1",
                "agent": {"name": "fabric", "version": "0.1.0"},
                "steps": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        TelemetryValidationError, match="steps must be a non-empty array"
    ):
        publish_telemetry_evidence(
            make_result(task_artifact("atif", "atif", atif, logs)),
            logs,
            strict=True,
        )


def test_publish_telemetry_rejects_obvious_credentials(tmp_path: Path):
    logs = tmp_path / "agent"
    logs.mkdir()
    atof = logs / "events.atof.jsonl"
    atof.write_text(
        json.dumps(
            {
                "atof_version": "1.0",
                "kind": "event",
                "name": "agent.end",
                "timestamp": "2026-07-13T00:00:00Z",
                "uuid": "019f616c-5eb1-7c92-928f-b4130bd4a519",
                "data": {"api_key": "sk-thismustneverbeleak123456"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TelemetryValidationError, match="resembles a credential"):
        publish_telemetry_evidence(
            make_result(task_artifact("atof", "atof", atof, logs)),
            logs,
            strict=True,
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("atof_version", None),
        ("atof_version", "version-one"),
        ("kind", []),
        ("name", ""),
        ("timestamp", "not-a-timestamp"),
        ("timestamp", "2026-07-13T00:00:00"),
        ("uuid", "not-a-uuid"),
    ],
)
def test_publish_telemetry_rejects_invalid_atof_fields(
    tmp_path: Path,
    field: str,
    invalid: object,
):
    logs = tmp_path / "agent"
    logs.mkdir()
    record = {
        "atof_version": "1.0",
        "kind": "event",
        "name": "agent.end",
        "timestamp": "2026-07-13T00:00:00Z",
        "uuid": "019f616c-5eb1-7c92-928f-b4130bd4a519",
    }
    record[field] = invalid
    atof = logs / "events.atof.jsonl"
    atof.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(
        TelemetryValidationError,
        match=rf"ATOF record .*{re.escape(field)}.*{re.escape(str(atof))}:1",
    ):
        publish_telemetry_evidence(
            make_result(task_artifact("atof", "atof", atof, logs)),
            logs,
            strict=True,
        )
