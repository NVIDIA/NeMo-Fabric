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

When upgrading a caller-owned Gateway from OpenClaw 2026.9.4, stop the Gateway and run `openclaw backup create --verify` with the Gateway's existing `OPENCLAW_STATE_DIR` and `OPENCLAW_CONFIG_PATH` before upgrading. The backup must include the active config, its include files, and all configured agent directories, including paths outside the state directory. Protect the archive because it can contain credentials.

Before upgrading, check the backup result and manifest for the active config, its include files, and every configured agent directory. For paths marked `covered`, confirm that the archive contains the data under a parent directory. Fix missing required paths and rerun the backup. Continue with the upgrade after confirming that all required data is backed up.

Run `openclaw doctor --fix` with the 2026.9.34 executable and the same path settings, then restart the Gateway. The older session database requires this migration. Keep the verified backup for rollback. Managed NeMo Fabric runtimes start with fresh temporary state.
