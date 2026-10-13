# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run two code-review turns through separate OpenShell SDK exec calls."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from nemo_fabric import FabricConfig

from examples.code_review_agent.config import openclaw_config

if TYPE_CHECKING:
    from openshell import SandboxClient


def _failure(operation: str, code: str) -> dict[str, Any]:
    return {
        "operation": operation,
        "status": "failed",
        "changed": None,
        "result": None,
        "error": {"code": code, "effects": "unknown"},
    }


class OpenShellCodeReview:
    """The consumer owns sandbox selection; the runtime server owns the runtime."""

    def __init__(
        self,
        client: SandboxClient,
        *,
        sandbox: str,
        workspace: str,
        timeout_seconds: int = 300,
        startup_seconds: float = 30,
    ):
        self.client = client
        self.sandbox = sandbox
        self.workspace = workspace
        self.timeout_seconds = timeout_seconds
        self.startup_seconds = startup_seconds
        self.commands: list[dict[str, Any]] = []

    def _exec(
        self,
        command: list[str],
        *,
        stdin: bytes | None = None,
        timeout: int | None = None,
        operation: str | None = None,
    ):
        started = time.monotonic()
        executed = None
        try:
            executed = self.client.exec(
                self.sandbox,
                command,
                workspace=self.workspace,
                stdin=stdin,
                timeout_seconds=timeout
                if timeout is not None
                else self.timeout_seconds,
                no_login_shell=True,
            )
            return executed
        finally:
            self.commands.append(
                {
                    "operation": operation
                    or (command[1] if command[0] == "fabric-agent-ctl" else "serve"),
                    "exit_code": executed.exit_code if executed is not None else None,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            )

    def _command(
        self,
        operation: str,
        *flags: str,
        value: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        try:
            stdin = (
                json.dumps(value, allow_nan=False).encode()
                if value is not None
                else None
            )
            executed = self._exec(
                ["fabric-agent-ctl", operation, *flags], stdin=stdin, timeout=timeout
            )
            if (
                len(executed.stdout.encode()) > 4 * 1024 * 1024
                or len(executed.stdout.splitlines()) != 1
            ):
                return _failure(operation, "invalid_response")
            response = json.loads(executed.stdout)
            if (
                not isinstance(response, dict)
                or response.get("operation") != operation
                or response.get("status") not in ("succeeded", "failed", "unsupported")
                or (executed.exit_code == 0) != (response["status"] == "succeeded")
                or (
                    response["status"] == "succeeded"
                    and not isinstance(response.get("result"), dict)
                )
                or (
                    response["status"] != "succeeded"
                    and not isinstance(response.get("error"), dict)
                )
            ):
                return _failure(operation, "invalid_response")
            return response
        except Exception:
            # Missing responses cannot prove a dispatched command stopped.
            return _failure(operation, "command_outcome_unknown")

    def run(self, config: FabricConfig, prompts: Sequence[str]) -> dict[str, Any]:
        """Return the reviews and a command trace without raw process diagnostics."""
        self.commands = []
        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        report = self._run(config, prompts)
        model = config.models["default"]
        return {
            **report,
            "sandbox": self.sandbox,
            "workspace": self.workspace,
            "started_at": started_at,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "model": {
                "provider": model.provider,
                "model": model.model,
                "base_url": model.base_url,
                "api_key_env": model.api_key_env,
            },
            "prompts": list(prompts),
            "commands": self.commands,
        }

    def _run(self, config: FabricConfig, prompts: Sequence[str]) -> dict[str, Any]:
        """Start once, invoke twice, and attempt stop without retrying agent work."""
        if len(prompts) != 2:
            raise ValueError("Provide exactly two code-review prompts")
        config = config.model_copy(deep=True)
        config.environment.workspace = "./code-review/repos/my-service"
        config.environment.artifacts = "./artifacts/code-review"
        config.runtime.artifacts = "./artifacts/code-review"
        config.skills.paths = ["./code-review/skills/code-review"]
        agent = config.metadata.name
        report: dict[str, Any] = {
            "status": "failed",
            "runtime_id": None,
            "reviews": [],
            "stop": None,
            "error": None,
        }
        try:
            boot = self._exec(
                [
                    "sh",
                    "-c",
                    "umask 077; nohup fabric-agent-ctl serve </dev/null "
                    ">/sandbox/fabric-server.log 2>&1 &",
                ]
            )
            if boot.exit_code != 0:
                report["error"] = _failure("serve", "host_start_failed")["error"]
                return report
        except Exception:
            report["error"] = _failure("serve", "command_outcome_unknown")["error"]
            return report
        deadline = time.monotonic() + self.startup_seconds
        ready = False
        while time.monotonic() < deadline:
            timeout = max(1, min(10, int(deadline - time.monotonic())))
            try:
                ready = (
                    self._exec(
                        ["sh", "-c", "test -S /sandbox/fabric.sock"],
                        timeout=timeout,
                        operation="wait_socket",
                    ).exit_code
                    == 0
                )
            except Exception:
                ready = False
            if ready:
                break
            time.sleep(0.2)
        if not ready:
            report["error"] = _failure("serve", "host_start_failed")["error"]
            return report
        started = self._command(
            "start",
            "--agent",
            agent,
            "--config",
            "-",
            value=config.to_mapping(),
        )
        if started["status"] != "succeeded":
            report["error"] = started["error"]
            return report
        runtime_id = started["result"].get("runtime_id")
        if not isinstance(runtime_id, str) or not runtime_id:
            report["error"] = _failure("start", "invalid_response")["error"]
            return report
        report["runtime_id"] = runtime_id
        try:
            for prompt in prompts:
                response = self._command(
                    "invoke",
                    "--agent",
                    agent,
                    "--runtime-id",
                    runtime_id,
                    "--input",
                    "-",
                    value={"message": prompt},
                )
                result = (response.get("result") or {}).get("fabric_result")
                if response["status"] != "succeeded":
                    if (
                        isinstance(result, dict)
                        and result.get("runtime_id") == runtime_id
                    ):
                        report["reviews"].append(result)
                    report["error"] = response["error"]
                    break
                if (
                    not isinstance(result, dict)
                    or result.get("runtime_id") != runtime_id
                    or result.get("status") != "succeeded"
                ):
                    report["error"] = _failure("invoke", "invalid_response")["error"]
                    break
                report["reviews"].append(result)
        finally:
            report["stop"] = self._command(
                "stop",
                "--agent",
                agent,
                "--runtime-id",
                runtime_id,
            )
            if report["stop"]["status"] != "succeeded" and report["error"] is None:
                report["error"] = report["stop"]["error"]
        if report["error"] is None:
            report["status"] = "succeeded"
        return report


def _render_transcript(report: dict[str, Any]) -> str:
    transcript = [
        "# OpenShell Code Review",
        f"Sandbox: `{report['sandbox']}` in workspace `{report['workspace']}`",
        f"Started: {report['started_at']}",
        f"Model: `{report['model']['model']}`",
        f"Runtime: `{report['runtime_id']}`",
        f"Status: {report['status']}",
    ]
    for index, prompt in enumerate(report["prompts"]):
        transcript.extend([f"## Turn {index + 1}", "### Prompt", prompt, "### Review"])
        if index < len(report["reviews"]):
            review = report["reviews"][index]
            output = review.get("output")
            response = output.get("response") if isinstance(output, dict) else None
            transcript.append(
                response
                if isinstance(response, str)
                else "No text response was returned."
            )
            if review.get("error") is not None:
                transcript.extend(
                    ["### Fabric error", json.dumps(review["error"], indent=2)]
                )
            if review.get("artifacts"):
                transcript.extend(
                    ["### Artifacts", json.dumps(review["artifacts"], indent=2)]
                )
        else:
            transcript.append("No successful review result was returned.")
    stop = report["stop"]
    transcript.extend(["## Stop", stop["status"] if stop else "Not attempted"])
    if report["error"] is not None:
        transcript.extend(["## Error", json.dumps(report["error"], indent=2)])
    return "\n\n".join(transcript) + "\n"


def _write_record(output_dir: Path, report: dict[str, Any]) -> None:
    # Save the authoritative result before rendering the human-readable copy.
    path = output_dir / "report.json"
    with path.open("x", encoding="utf-8") as file:
        path.chmod(0o600)
        file.write(json.dumps(report, indent=2) + "\n")
    try:
        transcript = _render_transcript(report)
    except (AttributeError, KeyError, TypeError, ValueError):
        transcript = "# OpenShell Code Review\n\nTranscript rendering failed. See report.json for the complete results.\n"
    path = output_dir / "transcript.md"
    with path.open("x", encoding="utf-8") as file:
        path.chmod(0o600)
        file.write(transcript)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandbox", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument(
        "--gateway",
        help="Local plaintext OpenShell gateway host:port; otherwise use the registered gateway",
    )
    parser.add_argument("--model", default="nvidia/nemotron-3.5-lightning-30b-a3b")
    parser.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Create a new directory containing report.json and transcript.md",
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Repeat twice: initial review and follow-up",
    )
    args = parser.parse_args()
    if len(args.input) != 2:
        parser.error("--input must be supplied exactly twice")
    if args.output_dir is not None:
        try:
            args.output_dir.mkdir(mode=0o700, parents=True)
        except FileExistsError:
            parser.error("--output-dir already exists; choose a new results directory")
        except OSError:
            parser.error("--output-dir could not be created")
    # Only the consumer example depends on OpenShell; the Fabric SDK does not.
    from openshell import SandboxClient

    config = openclaw_config()
    model = config.models["default"]
    model.provider = "openai"
    model.model = args.model
    model.base_url = args.base_url
    model.api_key_env = args.api_key_env
    model.max_tokens = 1024
    with (
        SandboxClient(args.gateway)
        if args.gateway
        else SandboxClient.from_active_cluster()
    ) as client:
        report = OpenShellCodeReview(
            client, sandbox=args.sandbox, workspace=args.workspace
        ).run(config, args.input)
    if args.output_dir is not None:
        _write_record(args.output_dir, report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
