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

The adapter supports Python 3.11 through 3.14. The merged upstream Hermes Relay
integration requires Python 3.14. Hermes Agent 0.20 and later is not installable from PyPI. Install Hermes Agent
by following the
[Hermes Agent installation guide](https://hermes-agent.nousresearch.com/docs/installation),
then install the NeMo Fabric packages into the Python environment that runs
Hermes Agent.

The following table shows which NeMo Fabric components each package expression
provides. None of these expressions installs Hermes Agent:

| Installation | Runtime | Adapter | Harness | NeMo Relay Python Package |
| --- | --- | --- | --- | --- |
| `pip install nemo-fabric nemo-fabric-adapters-hermes` | Yes | Yes | No | No |
| `pip install "nemo-fabric-adapters-hermes[full]"` | No | Yes | No | Yes |
| `pip install "nemo-fabric-adapters-hermes[relay]"` | No | Yes | No | Yes |
| `pip install nemo-fabric-adapters-hermes` | No | Yes | No | No |

Released Hermes Agent v2026.9.24 requires Relay 0.8.x. Ordinary runs remain
supported in a separate environment. Relay telemetry and streaming require a
Hermes release with Relay 0.9 support; NeMo Fabric tests the merged upstream
revision until such a release is available. Use the `relay` and `full` extras
only with a compatible Hermes checkout. The adapter fails at startup if Relay
telemetry is enabled with an incompatible Hermes or Relay installation.

Refer to the [installation guide](https://docs.nvidia.com/nemo/fabric/getting-started/install) for more details.
