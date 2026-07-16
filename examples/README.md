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

[`harbor`](harbor/README.md) shows how to run Fabric agents in Harbor. Start
with one SWE-Bench task, switch between Hermes and Claude, vary skills, MCP
servers, tools, and telemetry through Harbor options, then verify the reward
and Fabric artifacts. The [calculator demo](harbor/demo/README.md) provides a
smaller credential-free starting point plus the same harness variations.
