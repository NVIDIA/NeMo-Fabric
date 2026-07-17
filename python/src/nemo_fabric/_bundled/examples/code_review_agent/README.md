<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Editable Code Review Example

This directory is ordinary Python source copied by `nemo-fabric example init`.
Edit `config.py` to choose a real harness, model, tools, or telemetry. The
credential-free default uses the bundled scripted adapter so the scaffold can
be validated immediately.

From the directory containing this package, run:

```bash
nemo-fabric run \
  --factory my_agent.config:build_config \
  --base-dir my_agent \
  --input "Review the workspace"
```
