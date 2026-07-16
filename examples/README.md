<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Examples

This directory holds runnable Fabric examples.

## Code review agent

[`code_review_agent`](code_review_agent/README.md) demonstrates the
application-facing Python SDK contract:

- constructing complete `FabricConfig` values with Pydantic models;
- creating harness, environment, capability, and telemetry variants from deep
  copies;
- resolving relative workspace and skill paths with `base_dir`;
- running maintained Hermes, Codex, and Deep Agents adapters through the Python SDK.

Start with:

```bash
just build-all
.venv/bin/python -m examples.code_review_agent \
  --input "Reply with exactly: fabric works"
```

## Harbor

[`harbor`](harbor/README.md) demonstrates how to evaluate Fabric agents with
Harbor while preserving Fabric's typed configuration workflow. Harbor manages
the task environment, retries, concurrency, verification, rewards, and result
layout. `FabricAgent` translates Harbor inputs—including the harness, model,
skills, MCP servers, tool policy, and telemetry—into the final `FabricConfig`.

The walkthroughs include:

- a calculator walkthrough with a deterministic, credential-free integration
  smoke test and optional LLM-backed Hermes and Claude runs; and
- a SWE-Bench workflow for running Hermes and Claude, comparing capability
  variations, inspecting Relay telemetry, and verifying real coding tasks.

Start with the shared setup and execution model in the
[Harbor guide](harbor/README.md).
