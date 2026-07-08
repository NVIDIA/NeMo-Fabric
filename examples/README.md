<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Examples

This directory holds runnable Fabric SDK examples.

The code-review example demonstrates the application-facing contract:

- constructing complete `FabricConfig` values with Pydantic models;
- creating harness, environment, capability, and telemetry variants from deep
  copies;
- resolving relative workspace and skill paths with `base_dir`;
- running maintained Hermes and Codex adapters through the Python SDK.

Start with:

```bash
just build-all
.venv/bin/python -m examples.code_review_agent \
  --input "Reply with exactly: fabric works"
```

Portable manifest and profile behavior is covered by
`tests/fixtures/file-config-agent`. The dependency-free Hermes shim used by
runtime tests lives under `tests/fixtures/hermes-shim-agent`; neither fixture is
a public SDK example.
