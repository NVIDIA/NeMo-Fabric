<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric OpenClaw Adapter

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Fabric)](https://github.com/NVIDIA/NeMo-Fabric/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Fabric/)

`nemo-fabric-adapters-openclaw` provides the OpenClaw adapter for NVIDIA NeMo
Fabric. It starts an isolated OpenClaw Gateway and invokes its OpenAI-compatible
Chat Completions endpoint.

Install OpenClaw separately with npm, then install the adapter:

```bash
# Requires Node.js >=24.16.0 <25 or >=26.1.0.
npm install --global openclaw@latest --allow-scripts=openclaw
pip install "nemo-fabric[openclaw]"
```

The adapter does not install OpenClaw. Relay telemetry is not supported.
