# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract and process-lifecycle tests for Hermes through ``nemo-relay run``."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from nemo_fabric_adapters.common.relay_gateway import RelayCliContract
import nemo_fabric_adapters.common.utils as common_utils
from nemo_fabric_adapters.hermes import adapter
from nemo_fabric_adapters.hermes import relay_cli


@pytest.mark.parametrize(
    ("relay_enabled", "plugins_enabled", "raises"),
    [
        (False, [], False),
        (False, ["custom/plugin"], False),
        (False, ["observability/nemo_relay"], True),
        (True, ["observability/nemo_relay"], False),
    ],
)
def test_validate_hermes_relay_plugin_mode(
    relay_enabled: bool,
    plugins_enabled: list[str],
    raises: bool,
):
    payload = {
        "telemetry_plan": {
            "providers": ["relay"] if relay_enabled else [],
            "relay_enabled": relay_enabled,
        },
        "config": {
            "harness": {"settings": {"plugins_enabled": plugins_enabled}},
        },
    }

    if raises:
        with pytest.raises(
            ValueError,
            match=(
                "observability/nemo_relay cannot be enabled directly; "
                "enable Fabric Relay telemetry instead"
            ),
        ):
            adapter.validate_hermes_relay_plugin_mode(payload)
    else:
        adapter.validate_hermes_relay_plugin_mode(payload)


def test_build_hermes_args_uses_public_gateway_contract(tmp_path: Path):
    launch = _launch(tmp_path)

    command = [
        str(launch.relay_executable),
        "run",
        "--config",
        "/tmp/config.toml",
        "--agent",
        "hermes",
        "--",
        *relay_cli.build_hermes_args(launch, "repair the service"),
    ]

    assert "--plugin-config-path" not in command
    assert command[command.index("--provider") + 1] == "custom"
    assert command[command.index("--continue") + 1] == "runtime-123"
    assert command[command.index("--toolsets") + 1] == "terminal,file"
    assert "repair the service" not in relay_cli.redact_command(command)


def test_quiet_response_removes_only_pinned_hermes_startup_diagnostic():
    stdout = (
        "\x1b[2m  ⚠ tirith security scanner enabled but not available "
        "— command scanning will use pattern matching only\x1b[0m\r\n"
        "first response line\nsecond response line\n"
    )

    assert relay_cli.quiet_response(stdout) == (
        "first response line\nsecond response line"
    )


@pytest.mark.parametrize(
    ("version", "accepted"),
    [
        ("0.18.2", True),
        ("0.18.9", True),
        ("0.18.1", False),
        ("0.19.0", False),
    ],
)
def test_hermes_cli_version_contract(tmp_path: Path, version: str, accepted: bool):
    executable = tmp_path / f"hermes-{version}"
    executable.write_text(
        f"#!/bin/sh\nprintf 'Hermes Agent v{version} (2026.7.7.2)\\n'\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)

    if accepted:
        assert relay_cli.hermes_cli_version(executable) == tuple(
            int(value) for value in version.split(".")
        )
    else:
        with pytest.raises(relay_cli.HermesRelayError, match="unsupported Hermes"):
            relay_cli.hermes_cli_version(executable)


@pytest.mark.parametrize(
    ("provider", "base_url", "message"),
    [
        ("anthropic", "https://api.anthropic.com", "OpenAI-compatible provider"),
        ("custom", None, "OpenAI-compatible base URL"),
    ],
)
def test_validate_openai_upstream_rejects_incompatible_routes(
    provider: str, base_url: str | None, message: str
):
    with pytest.raises(relay_cli.HermesRelayError, match=message):
        relay_cli.validate_openai_upstream({}, {"provider": provider}, base_url)


def test_child_environment_forwards_only_explicit_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    monkeypatch.setenv("SELECTED_MODEL_KEY", "model-secret")
    monkeypatch.setenv("RELAY_HEADER_TOKEN", "relay-secret")
    monkeypatch.setenv("RELAY_S3_SECRET", "storage-secret")
    monkeypatch.setenv("UNRELATED_HOST_SECRET", "must-not-leak")

    env = relay_cli.child_environment(
        {"env": {"EXPLICIT_SETTING": "present"}},
        {"api_key_env": "SELECTED_MODEL_KEY"},
        {
            "components": [
                {
                    "config": {
                        "header_env": {"authorization": "RELAY_HEADER_TOKEN"},
                        "secret_access_key_var": "RELAY_S3_SECRET",
                    }
                }
            ]
        },
        tmp_path / "hermes-home",
    )

    assert env["SELECTED_MODEL_KEY"] == "model-secret"
    assert env["OPENAI_API_KEY"] == "model-secret"
    assert env["RELAY_HEADER_TOKEN"] == "relay-secret"
    assert env["RELAY_S3_SECRET"] == "storage-secret"
    assert env["EXPLICIT_SETTING"] == "present"
    assert "UNRELATED_HOST_SECRET" not in env
    assert env["HOME"] == str(tmp_path / "hermes-home")
    assert env["XDG_CONFIG_HOME"] == str(tmp_path / "hermes-home" / ".config")


async def test_runner_executes_colocated_relay_config_and_collects_artifacts(
    tmp_path: Path,
):
    executable = _write_fake_relay(tmp_path / "fake-relay.py")
    record_path = tmp_path / "record.json"
    launch = _launch(
        tmp_path,
        relay_executable=executable,
        env={
            **os.environ,
            "FAKE_RELAY_RECORD": str(record_path),
            "EXPECTED_CHILD_ONLY": "present",
        },
    )
    runner = relay_cli.HermesRelayRunner(launch)

    result = await runner.invoke("hello from Fabric", "invocation/1")

    assert result.returncode == 0
    assert result.stdout.strip() == "relay response: hello from Fabric"
    assert result.config_path.parent == result.plugin_config_path.parent
    assert result.config_path.name == "config.toml"
    assert result.plugin_config_path.name == "plugins.toml"
    assert "hello from Fabric" not in result.command
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["agent"] == "hermes"
    assert record["child_only"] == "present"
    assert record["hermes_command"] == str(launch.hermes_executable)
    assert record["hooks_path"] == str(launch.hermes_config_path)
    collected = common_utils.collect_relay_artifacts(result.plugin_config)
    artifacts = {artifact["kind"] for artifact in collected}
    assert artifacts == {"atof", "atif"}
    assert all(
        Path(artifact["path"]).is_relative_to(result.config_path.parent)
        for artifact in collected
    )


async def test_runner_preserves_response_tail_when_output_is_truncated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    executable = _write_fake_relay(tmp_path / "fake-relay.py")
    monkeypatch.setattr(relay_cli, "MAX_CAPTURE_BYTES", 256)
    launch = _launch(
        tmp_path,
        relay_executable=executable,
        env={
            **os.environ,
            "FAKE_RELAY_RECORD": str(tmp_path / "record.json"),
            "FAKE_RELAY_PREFIX_BYTES": "512",
        },
    )

    result = await relay_cli.HermesRelayRunner(launch).invoke(
        "tail response", "large-output"
    )

    assert result.truncated is True
    assert result.stdout.endswith("relay response: tail response\n")
    assert "...[output truncated]..." in result.stdout


async def test_normalized_output_preserves_response_after_late_failure(
    tmp_path: Path,
):
    executable = _write_fake_relay(tmp_path / "fake-relay.py")
    launch = _launch(
        tmp_path,
        relay_executable=executable,
        env={
            **os.environ,
            "FAKE_RELAY_RECORD": str(tmp_path / "record.json"),
            "FAKE_RELAY_EXIT": "7",
        },
    )
    runtime = adapter.HermesRuntime()
    runtime._relay_runner = relay_cli.HermesRelayRunner(launch)
    runtime._runtime_id = launch.runtime_id
    runtime._model_config = launch.model_config
    runtime._settings = launch.settings
    runtime._base_url = launch.base_url
    runtime._hermes_home = launch.hermes_home
    runtime._hermes_config_path = launch.hermes_config_path
    runtime._hermes_config = {"plugins": {"enabled": []}}
    runtime._hermes_cli_version = (0, 18, 2)

    output = await runtime._invoke_relay(
        {"runtime_context": {"invocation_id": "late-failure"}},
        "completed answer",
    )

    assert output["returncode"] == 7
    assert output["failed"] is True
    assert output["completed"] is True
    assert output["response"] == "relay response: completed answer"
    assert "status 7" in output["error"]
    assert output["relay_runtime"]["model_event_source"] == "gateway"
    assert output["relay_runtime"]["hook_event_policy"] == "lifecycle_context_only"


async def test_concurrent_runtimes_isolate_config_and_child_environment(
    tmp_path: Path,
):
    executable = _write_fake_relay(tmp_path / "fake-relay.py")
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = relay_cli.HermesRelayRunner(
        _launch(
            first_root,
            relay_executable=executable,
            env={
                **os.environ,
                "FAKE_RELAY_RECORD": str(first_root / "record.json"),
                "EXPECTED_CHILD_ONLY": "first",
                "FAKE_RELAY_DELAY": "0.2",
            },
        )
    )
    second = relay_cli.HermesRelayRunner(
        _launch(
            second_root,
            relay_executable=executable,
            env={
                **os.environ,
                "FAKE_RELAY_RECORD": str(second_root / "record.json"),
                "EXPECTED_CHILD_ONLY": "second",
                "FAKE_RELAY_DELAY": "0.2",
            },
        )
    )

    first_result, second_result = await asyncio.gather(
        first.invoke("first prompt", "same-id"),
        second.invoke("second prompt", "same-id"),
    )

    assert first_result.config_path != second_result.config_path
    assert (
        json.loads((first_root / "record.json").read_text(encoding="utf-8"))[
            "child_only"
        ]
        == "first"
    )
    assert (
        json.loads((second_root / "record.json").read_text(encoding="utf-8"))[
            "child_only"
        ]
        == "second"
    )
    assert "EXPECTED_CHILD_ONLY" not in os.environ


async def test_runner_cancellation_interrupts_process_group_and_restores_overlay(
    tmp_path: Path,
):
    executable = _write_cancellable_relay(tmp_path / "cancellable-relay.py")
    marker = tmp_path / "hermes-config-overlay"
    pid_path = tmp_path / "relay.pid"
    child_pid_path = tmp_path / "child.pid"
    launch = _launch(
        tmp_path,
        relay_executable=executable,
        env={
            **os.environ,
            "FAKE_OVERLAY": str(marker),
            "FAKE_PID": str(pid_path),
            "FAKE_CHILD_PID": str(child_pid_path),
        },
    )
    runner = relay_cli.HermesRelayRunner(launch)
    task = asyncio.create_task(runner.invoke("cancel me", "cancel-invocation"))
    await _wait_for_path(pid_path)
    await _wait_for_path(child_pid_path)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not marker.exists()
    assert not _process_exists(int(pid_path.read_text(encoding="utf-8")))
    assert not _process_exists(int(child_pid_path.read_text(encoding="utf-8")))


async def test_relay_enabled_runtime_selects_cli_without_mutating_host_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    relay_config = tmp_path / "relay.json"
    relay_config.write_text(
        json.dumps(
            {
                "relay": {
                    "config": {
                        "version": 2,
                        "atof": {
                            "enabled": True,
                            "sinks": [{"type": "file"}],
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FABRIC_RELAY_CONFIG_PATH", str(relay_config))
    monkeypatch.setenv("TEST_API_KEY", "test-only-secret")
    monkeypatch.setenv("HOME", "/host/home")
    monkeypatch.delenv("HERMES_HOME", raising=False)
    relay_executable = tmp_path / "nemo-relay"
    hermes_executable = tmp_path / "hermes"
    monkeypatch.setattr(
        relay_cli,
        "resolve_executable",
        lambda _root, _value, *, name: (
            relay_executable if "Relay" in name else hermes_executable
        ),
    )
    monkeypatch.setattr(
        adapter.relay_gateway,
        "relay_cli_contract",
        lambda _path: RelayCliContract((0, 6, 0), 2),
    )
    monkeypatch.setattr(relay_cli, "hermes_cli_version", lambda _path: (0, 18, 2))
    monkeypatch.setattr(adapter, "_ensure_hermes_runtime_session", lambda *args: None)
    payload = {
        "agent_name": "gateway-agent",
        "base_dir": str(tmp_path),
        "runtime_context": {
            "runtime_id": "runtime-gateway",
            "environment": {"workspace": str(tmp_path)},
            "telemetry": {"relay_enabled": True},
        },
        "telemetry_plan": {"providers": ["relay"], "relay_enabled": True},
        "config": {
            "harness": {
                "settings": {
                    "hermes_home": "hermes-home",
                    "plugins_enabled": [
                        "custom/plugin",
                        "observability/nemo_relay",
                    ],
                }
            },
            "models": {
                "default": {
                    "provider": "nvidia",
                    "model": "nvidia/test-model",
                    "api_key_env": "TEST_API_KEY",
                    "settings": {"base_url": "https://model.example/v1"},
                }
            },
        },
    }
    runtime = adapter.HermesRuntime()

    await runtime.start(payload)

    assert runtime._agent is None
    assert runtime._relay_runner is not None
    assert os.environ["HOME"] == "/host/home"
    assert "HERMES_HOME" not in os.environ
    assert runtime._hermes_config["plugins"]["enabled"] == ["custom/plugin"]
    assert runtime._relay_runner._launch.env["TEST_API_KEY"] == "test-only-secret"
    assert runtime._relay_runner._launch.env["OPENAI_API_KEY"] == "test-only-secret"
    await runtime.stop()


def _launch(
    tmp_path: Path,
    *,
    relay_executable: Path | None = None,
    env: dict[str, str] | None = None,
) -> relay_cli.HermesRelayLaunch:
    hermes_config = tmp_path / "hermes-home" / "config.yaml"
    hermes_config.parent.mkdir(parents=True, exist_ok=True)
    hermes_config.write_text("model: {}\n", encoding="utf-8")
    hermes_executable = tmp_path / "hermes"
    hermes_executable.touch()
    return relay_cli.HermesRelayLaunch(
        relay_executable=relay_executable or tmp_path / "nemo-relay",
        relay_contract=RelayCliContract((0, 6, 0), 2),
        hermes_executable=hermes_executable,
        hermes_config_path=hermes_config,
        hermes_home=hermes_config.parent,
        cwd=tmp_path,
        env=env or os.environ.copy(),
        base_url="https://model.example/v1",
        model="nvidia/test-model",
        runtime_id="runtime-123",
        settings={"enabled_toolsets": ["terminal", "file"]},
        model_config={"provider": "nvidia"},
        plugin_config={
            "version": 1,
            "components": [
                {
                    "kind": "observability",
                    "enabled": True,
                    "config": {
                        "version": 2,
                        "atof": {
                            "enabled": True,
                            "sinks": [
                                {
                                    "type": "file",
                                    "filename": "events.jsonl",
                                    "mode": "overwrite",
                                }
                            ],
                        },
                        "atif": {
                            "enabled": True,
                            "filename_template": "trajectory-{session_id}.json",
                        },
                    },
                }
            ],
        },
    )


def _write_fake_relay(path: Path) -> Path:
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
import time
import tomllib
from pathlib import Path

args = sys.argv[1:]
time.sleep(float(os.environ.get("FAKE_RELAY_DELAY", "0")))
config_path = Path(args[args.index("--config") + 1])
config = tomllib.loads(config_path.read_text())
plugins = tomllib.loads((config_path.parent / "plugins.toml").read_text())
inner = args[args.index("--") + 1:]
prompt = inner[inner.index("--query") + 1]
sys.stdout.write("x" * int(os.environ.get("FAKE_RELAY_PREFIX_BYTES", "0")))
atof = plugins["components"][0]["config"]["atof"]["sinks"][0]
atof_dir = Path(atof["output_directory"])
atof_dir.mkdir(parents=True, exist_ok=True)
(atof_dir / atof["filename"]).write_text("{}")
atif = plugins["components"][0]["config"]["atif"]
atif_dir = Path(atif["output_directory"])
atif_dir.mkdir(parents=True, exist_ok=True)
(atif_dir / "trajectory-test.json").write_text("{}")
Path(os.environ["FAKE_RELAY_RECORD"]).write_text(json.dumps({
    "agent": args[args.index("--agent") + 1],
    "child_only": os.environ.get("EXPECTED_CHILD_ONLY"),
    "hermes_command": config["agents"]["hermes"]["command"],
    "hooks_path": config["agents"]["hermes"]["hooks_path"],
}))
print(f"relay response: {prompt}")
raise SystemExit(int(os.environ.get("FAKE_RELAY_EXIT", "0")))
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _write_cancellable_relay(path: Path) -> Path:
    path.write_text(
        """#!/usr/bin/env python3
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

marker = Path(os.environ["FAKE_OVERLAY"])
marker.write_text("active")
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
Path(os.environ["FAKE_PID"]).write_text(str(os.getpid()))
Path(os.environ["FAKE_CHILD_PID"]).write_text(str(child.pid))

def stop(_signum, _frame):
    child.terminate()
    child.wait(timeout=5)
    marker.unlink(missing_ok=True)
    raise SystemExit(130)

signal.signal(signal.SIGINT, stop)
while True:
    time.sleep(0.1)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


async def _wait_for_path(path: Path) -> None:
    for _ in range(100):
        if path.exists():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path}")


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
