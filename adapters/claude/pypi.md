<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Claude Adapter

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Fabric)](https://github.com/NVIDIA/NeMo-Fabric/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Fabric/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Fabric?color=green)](https://github.com/NVIDIA/NeMo-Fabric/releases)

![Diagram showing NeMo Fabric connecting applications, evaluations, and reinforcement learning rollouts to Hermes, Codex, Claude, and Deep Agents, with results, artifacts, and telemetry as outputs.](https://raw.githubusercontent.com/NVIDIA/NeMo-Fabric/refs/heads/main/assets/fabric-hero-option2.png)

`nemo-fabric-adapters-claude` provides a NeMo Fabric adapter for [Claude Code](https://claude.com/product/claude-code).

## Install

Installation can be performed using the `nemo-fabric` meta package with the `claude` extra, or by installing the adapter and harness packages separately. The following table shows which components each installation provides:

| Installation | Runtime | Adapter | Harness | NeMo Relay CLI |
| --- | --- | --- | --- | --- |
| `pip install "nemo-fabric[claude]"` | Yes | Yes | Yes | No |
| `pip install "nemo-fabric-adapters-claude[harness]"` | No | Yes | Yes | No |
| `pip install nemo-fabric-adapters-claude` | No | Yes | No | No |

The `full` extra is equivalent to `harness`. NeMo Relay is optional for ordinary
runs. NeMo Relay telemetry and streaming require the `nemo-relay` CLI tool.
Refer to the [NeMo Relay installation guide](https://docs.nvidia.com/nemo/fabric/getting-started/install#nemo-relay-cli) for instructions.

Refer to the [installation guide](https://docs.nvidia.com/nemo/fabric/getting-started/install) for more details.
