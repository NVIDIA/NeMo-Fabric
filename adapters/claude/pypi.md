<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Claude Adapter

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Fabric)](https://github.com/NVIDIA/NeMo-Fabric/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Fabric/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Fabric?color=green)](https://github.com/NVIDIA/NeMo-Fabric/releases)

Provides a NeMo Fabric adapter for use with [Claude Code](https://claude.com/).

## Install

Installation can be performed using the `nemo-fabric` meta package with the `claude` extra, or by installing the adapter and harness packages separately. The following table shows which components each installation provides:

| Installation | Runtime | Adapter | Harness | NeMo Relay CLI |
| --- | --- | --- | --- | --- |
| `pip install "nemo-fabric[claude]"` | Yes | Yes | Yes | No |
| `pip install "nemo-fabric-adapters-claude[harness]"` | No | Yes | Yes | No |
| `pip install nemo-fabric-adapters-claude` | No | Yes | No | No |

The `full` extra is equivalent to `harness`. Relay is optional for ordinary
runs. NeMo Relay telemetry and streaming require the `nemo-relay` CLI tool to be installed, refer to the [NeMo Relay installation guide](https://nvidia-nemo-fabric.docs.buildwithfern.com/nemo/fabric/getting-started/install#nemo-relay-cli).

Refer to the [installation guide](https://nvidia-nemo-fabric.docs.buildwithfern.com/nemo/fabric/getting-started/install) for more details.

