<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Hermes Agent Adapter

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Fabric)](https://github.com/NVIDIA/NeMo-Fabric/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Fabric/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Fabric?color=green)](https://github.com/NVIDIA/NeMo-Fabric/releases)

![Diagram showing NeMo Fabric connecting applications, evaluations, and reinforcement learning rollouts to Hermes, Codex, Claude, and Deep Agents, with results, artifacts, and telemetry as outputs.](https://raw.githubusercontent.com/NVIDIA/NeMo-Fabric/refs/heads/main/assets/fabric-hero-option2.png)

`nemo-fabric-adapters-hermes` provides a NeMo Fabric adapter for use with [Hermes Agent](https://hermes-agent.nousresearch.com/).

## Install

Hermes Agent and this adapter require Python versions 3.11 through 3.13.
The following table shows which components each installation provides:

| Installation | Runtime | Adapter | Harness | NeMo Relay Python Package |
| --- | --- | --- | --- | --- |
| `pip install "nemo-fabric[hermes-agent]"` | Yes | Yes | No | No |
| `pip install "nemo-fabric[hermes-agent,relay]"` | Yes | Yes | No | Yes |
| `pip install "nemo-fabric-adapters-hermes[full]"` | No | Yes | No | Yes |
| `pip install "nemo-fabric-adapters-hermes[relay]"` | No | Yes | No | Yes |
| `pip install nemo-fabric-adapters-hermes` | No | Yes | No | No |

As of v0.20 Hermes Agent is no longer installable from PyPI. Refer to the [Hermes Agent installation guide](https://hermes-agent.nousresearch.com/docs/installation) for details.

NeMo Relay is optional for ordinary runs. NeMo Relay telemetry and streaming
require one of the installations in the table that includes the NeMo Relay
Python package.

Refer to the [installation guide](https://docs.nvidia.com/nemo/fabric/getting-started/install) for more details.
