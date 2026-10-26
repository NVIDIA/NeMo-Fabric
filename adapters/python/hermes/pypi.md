<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Hermes Agent Adapter

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Fabric)](https://github.com/NVIDIA/NeMo-Fabric/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Fabric/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Fabric?color=green)](https://github.com/NVIDIA/NeMo-Fabric/releases)

![Diagram showing NeMo Fabric connecting applications, evaluation systems, and reinforcement learning rollouts to harnesses and custom agents, with results, artifacts, and telemetry as outputs.](https://raw.githubusercontent.com/NVIDIA/NeMo-Fabric/refs/heads/main/assets/fabric-hero.png)

`nemo-fabric-adapters-hermes` provides a NeMo Fabric adapter for use with [Hermes Agent](https://hermes-agent.nousresearch.com/).

## Install

The adapter package supports Python 3.11 through 3.14. Use [Hermes Agent v0.22.5](https://github.com/NousResearch/hermes-agent/releases/tag/v0.22.5) with Python 3.14 and Relay 0.10 on supported platforms. This release includes dependency security fixes absent from v2026.9.24. Hermes Agent 0.20 and later is not installable from PyPI. Follow the [Hermes Agent installation guide](https://hermes-agent.nousresearch.com/docs/installation), select v0.22.5, then install the NeMo Fabric packages into the same Python environment.

The following table shows which NeMo Fabric components each package expression
provides. None of these expressions installs Hermes Agent:

| Installation | Runtime | Adapter | Harness | NeMo Relay Python Package |
| --- | --- | --- | --- | --- |
| `pip install nemo-fabric nemo-fabric-adapters-hermes` | Yes | Yes | No | No |
| `pip install "nemo-fabric-adapters-hermes[full]"` | No | Yes | No | Yes |
| `pip install "nemo-fabric-adapters-hermes[relay]"` | No | Yes | No | Yes |
| `pip install nemo-fabric-adapters-hermes` | No | Yes | No | No |

NeMo Fabric tests Hermes Agent v0.22.5, including Relay telemetry and streaming, on Python 3.14. Hermes Agent and the adapter's `relay` and `full` extras require `nemo-relay>=0.10,<0.11` on supported platforms. Keep Hermes in a separate Python environment when the NeMo Fabric application uses Relay 0.11 for other adapters. The adapter checks the Hermes integration and Relay version at startup when telemetry is enabled. Hermes Agent declares its Python package version as `0.0.0`; identify v0.22.5 by its Git commit `bf0203a794a07aac3ff764972ab9087baa4e4f4e` when reviewing scanner results.

Refer to the [installation guide](https://docs.nvidia.com/nemo/fabric/getting-started/install) for more details.
