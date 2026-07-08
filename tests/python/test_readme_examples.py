# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: the README quick start stays accurate and runnable."""

from __future__ import annotations

from pathlib import Path

from nemo_fabric import Fabric

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
EXAMPLE_AGENT = ROOT / "examples" / "code-review-agent"

# The README stays a quick start and routes detailed SDK usage to canonical docs.
DOCUMENTED_SNIPPETS = [
    "fabric doctor examples/code-review-agent --profile hermes_sdk",
    "fabric run examples/code-review-agent \\",
    "result = await Fabric().run(",
    '"examples/code-review-agent",',
    'profiles=["hermes_sdk"],',
    "[Python SDK guide](docs/sdk/python.mdx)",
    "[generated Python API reference](docs/reference/api/python-library-reference/index.md)",
]

DETAILED_SDK_SNIPPETS = (
    "config = FabricConfigModel(",
    "request = RunRequestModel(",
    "### Multi-Turn SDK Runtimes",
)


def readme_documents_each_example() -> None:
    """The README still contains every invocation this smoke mirrors."""

    text = README.read_text(encoding="utf-8")
    missing = [snippet for snippet in DOCUMENTED_SNIPPETS if snippet not in text]
    assert not missing, f"README no longer documents these examples verbatim: {missing}"
    duplicates = [snippet for snippet in DETAILED_SDK_SNIPPETS if snippet in text]
    assert not duplicates, f"README duplicates detailed SDK guide examples: {duplicates}"


async def readme_python_examples_run() -> None:
    """The README quick-start package remains resolvable and diagnosable."""

    agent = EXAMPLE_AGENT
    client = Fabric()
    plan = client.plan(agent, profiles=["hermes_sdk"])
    report = await client.doctor(agent, profiles=["hermes_sdk"])

    assert plan["agent_name"] == "code-review-agent", plan["agent_name"]
    assert report["checks"], "doctor returned no checks"


async def test_readme_examples():
    readme_documents_each_example()
    await readme_python_examples_run()
