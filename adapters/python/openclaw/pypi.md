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
# Requires Node.js >=24.22.1 <25 or >=26.12.1, and npm 11.16+.
npm install --global openclaw@2026.9.34 --allow-scripts=openclaw
pip install "nemo-fabric[openclaw]"
```

The adapter supports only OpenClaw's default `openclaw` agent runtime. It does
not install OpenClaw. Relay telemetry is not supported.

When upgrading a caller-owned Gateway from OpenClaw 2026.9.4, stop the Gateway and back up its state directory. Run `openclaw doctor --fix` with the 2026.9.34 executable and the same `OPENCLAW_STATE_DIR` and `OPENCLAW_CONFIG_PATH`, then restart the Gateway. The older session database requires this migration. Keep the backup for rollback. Managed NeMo Fabric runtimes start with fresh temporary state.
