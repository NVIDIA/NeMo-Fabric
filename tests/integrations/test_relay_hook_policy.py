# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise NeMo Fabric's generated hooks with the released Relay CLI."""

import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess

import pytest
import tomli_w

from nemo_fabric_adapters.common.relay_hooks import render_relay_hooks
from nemo_fabric_adapters.common.relay_gateway import (
    RelayGatewayLaunch,
    find_available_tcp_port,
    relay_cli_contract,
    start_relay_gateway,
    stop_relay_gateway,
)


@pytest.fixture
def relay_cli(tmp_path, monkeypatch):
    executable = shutil.which("nemo-relay")
    if executable is None:
        pytest.skip("NeMo Relay CLI is not installed")
    # Keep hook identity, plugin discovery, and state separate from the host.
    for name, directory in {
        "HOME": "home",
        "XDG_CONFIG_HOME": "config",
        "XDG_STATE_HOME": "state",
        "XDG_CACHE_HOME": "cache",
    }.items():
        path = tmp_path / directory
        path.mkdir()
        monkeypatch.setenv(name, str(path))
    monkeypatch.setenv("NEMO_RELAY_CONFIG_SCOPE", "user")
    monkeypatch.delenv("NEMO_RELAY_TRANSPARENT_RUN", raising=False)
    return Path(executable), relay_cli_contract(Path(executable))


def run_generated_hook(executable, agent, event, gateway_url, tmp_path):
    hook = render_relay_hooks(agent, executable)["hooks"][event][0]["hooks"][0]
    return subprocess.run(
        shlex.split(hook["command"]),
        input=json.dumps(
            {
                "hook_event_name": event,
                "session_id": f"fabric-hook-policy-{agent}",
                "cwd": str(tmp_path),
                "prompt": "hook policy test",
                "tool_name": "Read",
                "tool_use_id": "fabric-hook-read",
                "tool_input": {"file_path": "README.md"},
                "tool_response": "test result",
            }
        ),
        env={**os.environ, "NEMO_RELAY_GATEWAY_URL": gateway_url},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=hook["timeout"],
        check=False,
    )


@pytest.mark.parametrize("agent", ["claude", "codex"])
@pytest.mark.parametrize(
    ("event", "expected_status"),
    [("PreToolUse", 2), ("PostToolUse", 0), ("SessionStart", 0)],
)
def test_generated_hook_delivery_failure_policy(
    tmp_path, relay_cli, agent, event, expected_status
):
    executable, contract = relay_cli
    if event == "PreToolUse" and contract.version < (0, 11, 0):
        expected_status = 1
    # A bound socket without a listener rejects connections and reserves the port.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        completed = run_generated_hook(
            executable, agent, event, f"http://127.0.0.1:{port}", tmp_path
        )

    assert completed.returncode == expected_status


@pytest.mark.parametrize("agent", ["claude", "codex"])
def test_generated_hooks_deliver_to_live_gateway(tmp_path, relay_cli, agent):
    executable, _contract = relay_cli
    output = tmp_path / "telemetry"
    config = tmp_path / "plugins.toml"
    config.write_text(
        tomli_w.dumps(
            {
                "version": 1,
                "components": [
                    {
                        "kind": "observability",
                        "enabled": True,
                        "config": {
                            "version": 4,
                            "atof": {
                                "enabled": True,
                                "sinks": [
                                    {
                                        "type": "file",
                                        "output_directory": str(output),
                                        "filename": "events.jsonl",
                                    }
                                ],
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    port = find_available_tcp_port()
    url = f"http://127.0.0.1:{port}"
    process = start_relay_gateway(
        launch=RelayGatewayLaunch(
            executable=executable,
            config_path=config,
            bind=f"127.0.0.1:{port}",
            url=url,
            log_path=tmp_path / "gateway.log",
        ),
        cwd=tmp_path,
    )
    try:
        for event in (
            "SessionStart",
            "UserPromptSubmit",
            "PreToolUse",
            "PostToolUse",
            "Stop",
        ):
            completed = run_generated_hook(executable, agent, event, url, tmp_path)
            assert completed.returncode == 0
            if event == "SessionStart":
                response = json.loads(completed.stdout)
                assert response == ({"continue": True} if agent == "claude" else {})
    finally:
        stop_relay_gateway(process)

    records = [
        json.loads(line)
        for path in output.rglob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records, "The generated hooks did not produce any ATOF records"
    assert f"fabric-hook-policy-{agent}" in json.dumps(records)
