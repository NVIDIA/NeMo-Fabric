# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: the README "Use Fabric" examples stay accurate and runnable."""

from __future__ import annotations

import asyncio
from pathlib import Path

from nemo_fabric import Fabric, FabricConfig

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
EXAMPLE_AGENT = ROOT / "examples" / "code-review-agent"

# Exact invocations the README documents and this smoke mirrors. If the README
# changes any of these, update the executable mirror below (and vice versa).
DOCUMENTED_SNIPPETS = [
    "fabric plan examples/code-review-agent --profile hermes_sdk",
    "fabric plan examples/code-review-agent --profile env_local --profile mcp_github",
    "fabric doctor examples/code-review-agent --profile hermes_sdk",
    'plan = client.plan(agent, profiles=["hermes_sdk"])',
    'report = await client.doctor(agent, profiles=["hermes_sdk"])',
    "config = FabricConfig.from_mapping(",
    "plan = client.plan(",
    "result = await client.run(",
    'session_id="job-123",',
    '"harness": {"adapter_id": "nvidia.fabric.hermes.sdk"},',
    'base_dir="examples/code-review-agent",',
    "### Multi-Turn SDK Sessions",
    "### Interactive CLI Chat",
    "Fabric().start_session(",
    'profiles=["hermes_session"],',
    'session_id="review-session-123",',
    "fabric chat examples/code-review-agent \\",
    "--profile hermes_cli_session",
    "--session-id review-session-123",
    "--verbose",
    "adapter supports sessions",
    "The CLI is a separate interface over the same Rust",
]

# The exact typed-config dict shown in the README example.
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
    async with Fabric() as client:
        plan = client.plan(agent, profiles=["hermes_sdk"])
        report = await client.doctor(agent, profiles=["hermes_sdk"])
        typed_plan = client.plan(
            FabricConfig.from_mapping(README_PLAN_CONFIG),
            base_dir=agent,
        )

    # README prints plan["agent_name"] and report["checks"].
    assert plan["agent_name"] == "code-review-agent", plan["agent_name"]
    assert report["checks"], "doctor returned no checks"
    assert typed_plan["agent_name"] == "code-review-agent"
    assert typed_plan.adapter.adapter_id == "nvidia.fabric.hermes.sdk"


def main() -> None:
    readme_documents_each_example()
    asyncio.run(readme_python_examples_run())
    print("smoke_readme_examples ok")


if __name__ == "__main__":
    main()
