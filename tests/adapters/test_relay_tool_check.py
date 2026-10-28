# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from unittest.mock import Mock

import pytest

from nemo_fabric_adapters.common import relay_tool_check


@pytest.mark.parametrize("status", [1, 2, -9])
def test_failed_tool_check_blocks_without_exposing_child_output(
    monkeypatch, capsys, status
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "relay_tool_check",
            "/path with spaces/relay",
            "hook-forward",
            "codex",
            "--fail-closed",
        ],
    )
    runner = Mock(
        return_value=subprocess.CompletedProcess(
            [], status, b"private model data", b"private credential data"
        )
    )
    monkeypatch.setattr(relay_tool_check.subprocess, "run", runner)

    assert relay_tool_check.main() == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.strip() == "NeMo Relay could not approve this tool call."
    assert runner.call_args.args[0] == [
        "/path with spaces/relay",
        "hook-forward",
        "codex",
        "--fail-closed",
    ]


@pytest.mark.parametrize(
    "error", [FileNotFoundError(), subprocess.TimeoutExpired("relay", 35)]
)
def test_tool_check_launch_and_timeout_failures_block(monkeypatch, capsys, error):
    monkeypatch.setattr(
        sys,
        "argv",
        ["relay_tool_check", "relay", "hook-forward", "codex", "--fail-closed"],
    )
    runner = Mock(side_effect=error)
    monkeypatch.setattr(relay_tool_check.subprocess, "run", runner)

    assert relay_tool_check.main() == 2
    assert capsys.readouterr().err.strip()
    assert runner.call_args.kwargs["timeout"] < 45


def test_successful_tool_check_preserves_the_host_decision(monkeypatch, capsys):
    decision = b'{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow"}}\n'
    monkeypatch.setattr(
        sys,
        "argv",
        ["relay_tool_check", "relay", "hook-forward", "codex", "--fail-closed"],
    )
    monkeypatch.setattr(
        relay_tool_check.subprocess,
        "run",
        Mock(return_value=subprocess.CompletedProcess([], 0, decision, b"")),
    )

    assert relay_tool_check.main() == 0
    output = capsys.readouterr()
    assert output.out.encode() == decision
    assert output.err == ""


def test_tool_check_ignores_workspace_import_shadows(tmp_path, monkeypatch):
    import os
    import shlex
    from pathlib import Path

    from nemo_fabric_adapters.common.relay_hooks import render_relay_hooks

    package = tmp_path / "nemo_fabric_adapters" / "common"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "relay_tool_check.py").write_text("raise SystemExit(0)\n")
    (tmp_path / "subprocess.py").write_text("raise SystemExit(0)\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    hook = render_relay_hooks("codex", Path("missing-relay"))["hooks"]["PreToolUse"][0][
        "hooks"
    ][0]

    completed = subprocess.run(
        shlex.split(hook["command"]),
        input=b"{}",
        capture_output=True,
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout=5,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == b""
    assert completed.stderr.strip() == b"NeMo Relay could not approve this tool call."
