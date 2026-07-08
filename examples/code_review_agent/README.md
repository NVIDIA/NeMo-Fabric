<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Code Review Agent

This example constructs complete `FabricConfig` values with the public Pydantic
models. Each harness, environment, capability, or telemetry variant starts from
an independent deep copy; the example does not use file-backed profiles.

Run the default Hermes SDK variant from the repository root:

```bash
.venv/bin/python -m examples.code_review_agent \
  --input "Reply with exactly: fabric works"
```

See `config.py` for the base configuration and clone-based variant functions.
