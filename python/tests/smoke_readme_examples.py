# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: the README "Use Fabric" examples stay accurate and runnable.

WS4 guardrail. The documented Python SDK examples (``plan`` / ``doctor`` /
``plan_config`` and the source-tree CLI-command form) are mirrored here and run
against the real example agent, so an API change breaks this test instead of
silently rotting the README. A drift guard additionally asserts the README still
contains each documented invocation verbatim, keeping the prose and the
executable mirror in sync. The CLI snippets are exercised by ``tests/smoke_cli.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from nemo_fabric import FabricClient

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
EXAMPLE_AGENT = ROOT / "examples" / "code-review-agent"

# Exact invocations the README documents and this smoke mirrors. If the README
# changes any of these, update the executable mirror below (and vice versa).
DOCUMENTED_SNIPPETS = [
    "fabric plan examples/code-review-agent --profile hermes_sdk",
    "fabric plan examples/code-review-agent --profile env_local --profile mcp_github",
    "fabric doctor examples/code-review-agent --profile hermes_sdk",
    'plan = client.plan(agent, profile="hermes_sdk")',
    'report = await client.doctor(agent, profile="hermes_sdk")',
    "plan = client.plan_config(",
    '"harness": {"adapter_id": "nvidia.fabric.hermes.sdk"},',
    'base_dir="examples/code-review-agent",',
    'client = FabricClient(command=("cargo", "run", "-q", "-p", "fabric-cli", "--"))',
]

# The exact typed-config dict shown in the README "plan_config" example.
README_PLAN_CONFIG = {
    "schema_version": "fabric.agent/v1alpha1",
    "metadata": {"name": "code-review-agent"},
    "harness": {"adapter_id": "nvidia.fabric.hermes.sdk"},
    "models": {
        "default": {
            "provider": "nvidia",
            "model": "nvidia/nemotron-3-nano-30b-a3b",
        }
    },
    "runtime": {
        "mode": "session",
        "transport": "library",
        "input_schema": "chat",
        "output_schema": "message",
    },
}


def readme_documents_each_example() -> None:
    """The README still contains every invocation this smoke mirrors."""

    text = README.read_text(encoding="utf-8")
    missing = [snippet for snippet in DOCUMENTED_SNIPPETS if snippet not in text]
    assert not missing, f"README no longer documents these examples verbatim: {missing}"


async def readme_python_examples_run() -> None:
    """The documented Python SDK examples execute and return documented shapes."""

    agent = EXAMPLE_AGENT
    async with FabricClient() as client:
        plan = client.plan(agent, profile="hermes_sdk")
        report = await client.doctor(agent, profile="hermes_sdk")
        typed_plan = client.plan_config(README_PLAN_CONFIG, base_dir=agent)

    # README prints plan["agent_name"] and report["checks"].
    assert plan["agent_name"] == "code-review-agent", plan["agent_name"]
    assert report["checks"], "doctor returned no checks"
    assert typed_plan["agent_name"] == "code-review-agent"
    assert (
        typed_plan["adapter_descriptor"]["descriptor"]["adapter_id"]
        == "nvidia.fabric.hermes.sdk"
    )

    # The documented source-tree form selects the CLI command path.
    cli_client = FabricClient(command=("cargo", "run", "-q", "-p", "fabric-cli", "--"))
    assert cli_client.command == ("cargo", "run", "-q", "-p", "fabric-cli", "--")


def main() -> None:
    readme_documents_each_example()
    asyncio.run(readme_python_examples_run())
    print("smoke_readme_examples ok")


if __name__ == "__main__":
    main()
