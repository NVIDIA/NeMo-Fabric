# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml


def assert_relay_disabled_native_observability(result: dict):
    """Assert telemetry-off runs still surface native harness evidence."""

    artifact_by_name = {
        artifact["name"]: artifact
        for artifact in result["artifacts"]["artifacts"]
    }
    assert "stdout" in artifact_by_name
    assert "relay_config" not in artifact_by_name
    assert not any(name.startswith("relay_") for name in artifact_by_name)

    stdout_path = Path(artifact_by_name["stdout"]["path"])
    assert stdout_path.is_file()
    assert stdout_path.read_text(encoding="utf-8").strip()

    event_kinds = {event["kind"] for event in result["events"]}
    assert {"runtime_start", "invocation_start", "invocation_end"} <= event_kinds

    telemetry = result["telemetry"]
    assert telemetry is not None
    assert telemetry["relay_enabled"] is False


def assert_process_adapter_native_observability(result: dict):
    """Assert process adapters preserve native evidence and clean process output."""

    assert_relay_disabled_native_observability(result)
    assert result["output"]["returncode"] == 0
    assert result["output"]["stderr"] == ""


def update_hermes_cli_relay_base_url(code_review_agent_dir: Path, api_server: str):
    """
    Update the base URL in the Hermes CLI relay profile.

    Since the api_server uses a random available TCP port, the base_url needs to be updated for each test.

    Args:
        code_review_agent_dir (Path): The path to the code review agent directory.
        api_server (str): The API server URL.
    """
    profile_path = code_review_agent_dir / "profiles" / "hermes-cli-relay.yaml"
    profile = yaml.safe_load(profile_path.read_text())
    profile["harness"]["settings"]["base_url"] = f"{api_server}/v1"
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False))
